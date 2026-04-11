"""v0.7 FastAPI server mode — expose pagewiki as HTTP endpoints.

Design goals
------------

* **Warm state**: scan the vault once at startup, keep the tree +
  wiki-link index in memory so repeat requests skip the expensive
  scan + summarize pipeline.
* **Session-based chat**: each chat client gets a session id with
  its own conversation history. Sessions are stored in-process (no
  Redis dependency) and expire after ``SESSION_TTL_SECONDS``.
* **Optional**: FastAPI + uvicorn are imported lazily so users who
  don't need the server don't pay the install cost.

Endpoints
---------

* ``GET  /health``         — liveness probe
* ``POST /scan``           — refresh the in-memory tree
* ``POST /ask``            — single-shot query
* ``POST /chat``           — session-based multi-turn query
* ``DELETE /chat/{sid}``   — clear a chat session

Run with::

    pagewiki serve --vault <path> --folder Research --port 8000

Then::

    curl -X POST http://localhost:8000/ask \\
        -H 'Content-Type: application/json' \\
        -d '{"query": "What is X?"}'

Note on annotations
-------------------

This module intentionally does NOT use ``from __future__ import annotations``.
The Pydantic request/response models defined below are consumed by
FastAPI's type adapter, which needs resolved (non-ForwardRef) class
references to build its request validators. Enabling PEP 563 lazy
annotations would turn ``AskRequest`` into a string that FastAPI
can't resolve out of this module's namespace.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import SummaryCache, TreeCache
from .retrieval import (
    RetrievalResult,
    TraceStep,
    run_decomposed_retrieval,
    run_retrieval,
)
from .tree import NoteTier, TreeNode
from .usage import UsageTracker
from .usage_store import UsageStore
from .vault import (
    build_long_subtrees,
    filter_tree,
    scan_folder,
    scan_multi_vault,
    summarize_atomic_notes,
)
from .webui import build_ui_html
from .wiki_links import LinkIndex, build_link_index

ChatFn = Callable[[str], str]

# Chat sessions expire after 1 hour of inactivity.
SESSION_TTL_SECONDS = 3600


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChatSession:
    """Server-side conversation state for one chat client."""

    sid: str
    history: list[tuple[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


@dataclass
class ServerState:
    """In-memory state shared across all HTTP handlers."""

    vaults: list[Path]
    folder: str | None
    model: str
    num_ctx: int
    max_workers: int
    chat_fn: ChatFn
    root: TreeNode
    link_index: LinkIndex
    summary_cache: SummaryCache
    tree_cache: TreeCache
    sessions: dict[str, ChatSession] = field(default_factory=dict)
    # v0.9 cumulative usage tracker — shared by every /ask and /chat
    # call so operators can monitor token budgets across the whole
    # server lifetime via GET /usage.
    tracker: UsageTracker = field(default_factory=UsageTracker)
    # v0.10 optional SQLite-backed usage persistence. When set, the
    # /usage endpoint reports historical totals in addition to the
    # in-memory tracker. Enabled via `pagewiki serve --usage-db PATH`.
    usage_store: UsageStore | None = None
    # v0.16 optional prompt-cache chat_fn. When set, per-request
    # endpoints (/ask/ws, future /ask/stream) can opt into the
    # prompt-cache path on a per-query basis via the request frame.
    # Signature: (system, user) -> str. Wired up by cli.serve.
    system_chat_fn: object | None = None

    def rescan(self) -> dict[str, int]:
        """Re-scan the vault(s) and rebuild caches in place.

        Returns a small dict summarizing the refreshed tree
        (note_count, long_count, summarized).
        """
        if len(self.vaults) > 1:
            new_root = scan_multi_vault([(v, self.folder) for v in self.vaults])
        else:
            new_root = scan_folder(self.vaults[0], self.folder)

        summarized = summarize_atomic_notes(
            new_root,
            self.chat_fn,
            summary_cache=self.summary_cache,
            model_id=self.model,
            max_workers=self.max_workers,
        )

        long_count = sum(
            1
            for n in new_root.walk()
            if n.kind == "note" and n.tier == NoteTier.LONG
        )
        if long_count > 0:
            build_long_subtrees(
                new_root,
                vault_root=self.vaults[0],
                model_id=self.model,
                chat_fn=self.chat_fn,
                cache=self.tree_cache,
            )

        self.root = new_root
        self.link_index = build_link_index(new_root)

        note_count = sum(1 for n in new_root.walk() if n.kind == "note")
        return {
            "note_count": note_count,
            "long_count": long_count,
            "summarized": summarized,
        }


def _prune_expired_sessions(state: ServerState) -> None:
    """Drop sessions that haven't been touched in SESSION_TTL_SECONDS."""
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [sid for sid, s in state.sessions.items() if s.last_active < cutoff]
    for sid in expired:
        del state.sessions[sid]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ─────────────────────────────────────────────────────────────────────────────


def create_app(state: ServerState):  # -> fastapi.FastAPI
    """Build a FastAPI application bound to the given server state.

    FastAPI + Pydantic are imported inside this function so they
    stay optional dependencies — ``pagewiki serve`` will surface a
    clear install hint if either is missing.
    """
    try:
        from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise RuntimeError(
            "FastAPI is required for `pagewiki serve`. Install with:\n"
            "    pip install 'pagewiki[server]'\n"
            f"Original error: {e}"
        ) from e

    app = FastAPI(title="pagewiki API", version="0.16.0")

    # ── Request/response models ────────────────────────────────────────────

    class AskRequest(BaseModel):
        query: str = Field(..., min_length=1)
        decompose: bool = False
        tags: list[str] | None = None
        after: str | None = None
        before: str | None = None

    class ChatRequest(BaseModel):
        query: str = Field(..., min_length=1)
        session_id: str | None = None
        decompose: bool = False

    class TraceStepOut(BaseModel):
        phase: str
        node_id: str | None
        detail: str

    class AskResponse(BaseModel):
        answer: str
        cited_nodes: list[str]
        iterations_used: int
        trace: list[TraceStepOut]

    class ChatResponse(BaseModel):
        answer: str
        cited_nodes: list[str]
        session_id: str
        turn: int

    class ScanResponse(BaseModel):
        note_count: int
        long_count: int
        summarized: int

    class PhaseUsage(BaseModel):
        calls: int
        prompt: int
        completion: int
        elapsed: float

    class UsageResponse(BaseModel):
        total_calls: int
        total_prompt_tokens: int
        total_completion_tokens: int
        total_tokens: int
        total_elapsed: float
        by_phase: dict[str, PhaseUsage]
        # v0.10: optional persistent totals from SQLite store.
        # ``None`` when the server was started without --usage-db.
        persistent_total_calls: int | None = None
        persistent_total_prompt: int | None = None
        persistent_total_completion: int | None = None

    class UsageEventOut(BaseModel):
        timestamp: float
        phase: str
        prompt: int
        completion: int
        elapsed: float

    class UsageDailyOut(BaseModel):
        date: str
        total_calls: int
        total_prompt: int
        total_completion: int
        total_elapsed: float
        by_phase: dict[str, dict]

    class UsageHistoryResponse(BaseModel):
        summary: UsageResponse
        events: list[UsageEventOut]
        daily: list[UsageDailyOut]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _trace_to_out(trace: list[TraceStep]) -> list[TraceStepOut]:
        return [
            TraceStepOut(phase=t.phase, node_id=t.node_id, detail=t.detail)
            for t in trace
        ]

    def _filtered_root(
        tags: list[str] | None,
        after: str | None,
        before: str | None,
    ) -> TreeNode:
        if not tags and not after and not before:
            return state.root
        return filter_tree(state.root, tags=tags, after=after, before=before)

    def _run(
        query: str,
        *,
        decompose: bool,
        history: list[tuple[str, str]] | None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> RetrievalResult:
        root = _filtered_root(tags, after, before)
        if decompose:
            return run_decomposed_retrieval(
                query, root, state.chat_fn, link_index=state.link_index,
            )
        return run_retrieval(
            query, root, state.chat_fn,
            link_index=state.link_index,
            history=history,
        )

    # ── Endpoints ──────────────────────────────────────────────────────────

    # ── v0.14 Web UI ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def web_ui() -> "HTMLResponse":
        """Return the embedded single-page Web UI (v0.14).

        The HTML is self-contained (no external CSS/JS) so opening
        ``http://localhost:8000/`` in a browser gives an immediate
        chat-like interface over ``/ask/stream``. Set
        ``PAGEWIKI_UI_HTML`` to point at a custom HTML file.
        """
        return HTMLResponse(build_ui_html())

    @app.get("/health")
    def health() -> dict[str, Any]:
        note_count = sum(1 for n in state.root.walk() if n.kind == "note")
        return {
            "status": "ok",
            "version": "0.16.0",
            "model": state.model,
            "vault_count": len(state.vaults),
            "note_count": note_count,
            "active_sessions": len(state.sessions),
        }

    @app.post("/scan", response_model=ScanResponse)
    def scan() -> ScanResponse:
        summary = state.rescan()
        return ScanResponse(**summary)

    # FastAPI idiomatic Body(...) default is used below. Ruff's B008
    # rule is silenced per handler because FastAPI's dependency
    # injection requires the call to live in the default slot.

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest = Body(...)) -> AskResponse:  # noqa: B008
        result = _run(
            req.query,
            decompose=req.decompose,
            history=None,
            tags=req.tags,
            after=req.after,
            before=req.before,
        )
        return AskResponse(
            answer=result.answer,
            cited_nodes=result.cited_nodes,
            iterations_used=result.iterations_used,
            trace=_trace_to_out(result.trace),
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest = Body(...)) -> ChatResponse:  # noqa: B008
        _prune_expired_sessions(state)

        sid = req.session_id or uuid.uuid4().hex
        session = state.sessions.get(sid)
        if session is None:
            session = ChatSession(sid=sid)
            state.sessions[sid] = session

        session.last_active = time.time()

        result = _run(
            req.query,
            decompose=req.decompose,
            history=session.history,
        )

        session.history.append((req.query, result.answer[:500]))

        return ChatResponse(
            answer=result.answer,
            cited_nodes=result.cited_nodes,
            session_id=sid,
            turn=len(session.history),
        )

    @app.delete("/chat/{sid}")
    def clear_chat(sid: str) -> dict[str, str]:
        if sid not in state.sessions:
            raise HTTPException(status_code=404, detail=f"Session {sid} not found")
        del state.sessions[sid]
        return {"status": "cleared", "session_id": sid}

    # ── v0.9 usage endpoints ──────────────────────────────────────────────

    @app.get("/usage", response_model=UsageResponse)
    def get_usage() -> UsageResponse:
        """Return cumulative token usage since server startup (v0.9).

        When the server was started with ``--usage-db PATH``,
        ``persistent_total_*`` fields are also populated from the
        SQLite store covering the entire store lifetime (v0.10).
        """
        t = state.tracker

        persistent_calls = None
        persistent_prompt = None
        persistent_completion = None
        if state.usage_store is not None:
            summary = state.usage_store.query_summary()
            persistent_calls = summary.total_calls
            persistent_prompt = summary.total_prompt
            persistent_completion = summary.total_completion

        return UsageResponse(
            total_calls=t.total_calls,
            total_prompt_tokens=t.total_prompt_tokens,
            total_completion_tokens=t.total_completion_tokens,
            total_tokens=t.total_tokens,
            total_elapsed=t.total_elapsed,
            by_phase={
                phase: PhaseUsage(
                    calls=int(b["calls"]),
                    prompt=int(b["prompt"]),
                    completion=int(b["completion"]),
                    elapsed=float(b["elapsed"]),
                )
                for phase, b in t.by_phase().items()
            },
            persistent_total_calls=persistent_calls,
            persistent_total_prompt=persistent_prompt,
            persistent_total_completion=persistent_completion,
        )

    @app.post("/usage/reset")
    def reset_usage() -> dict[str, str]:
        """Clear the cumulative usage tracker (v0.9)."""
        state.tracker.events.clear()
        return {"status": "reset"}

    @app.get("/usage/history", response_model=UsageHistoryResponse)
    def usage_history(
        since: str | None = None,
        until: str | None = None,
        phase: str | None = None,
        limit: int = 100,
        include_daily: bool = True,
    ) -> UsageHistoryResponse:
        """Query historical usage from the SQLite store (v0.14).

        Returns a combined payload of:
          * ``summary``  — cumulative totals over the window, with
                           phase-broken down. Includes current-session
                           in-memory tracker state.
          * ``events``   — the ``limit`` most recent raw events in the
                           window (or fewer if the store is smaller).
          * ``daily``    — per-day rollup rows in the window (empty
                           when ``include_daily=False``).

        Query parameters:
          * ``since``    — ISO date or timestamp (events >= this).
          * ``until``    — ISO date or timestamp (events < this).
          * ``phase``    — restrict to a single phase bucket.
          * ``limit``    — max recent events (default 100).
          * ``include_daily`` — include the daily rollup list.

        Returns ``503`` if the server wasn't started with
        ``--usage-db`` (no persistent store).
        """
        if state.usage_store is None:
            raise HTTPException(
                status_code=503,
                detail="Usage persistence is not enabled. "
                "Start `pagewiki serve --usage-db PATH`.",
            )

        def _parse(s: str | None) -> float | None:
            if s is None:
                return None
            try:
                from datetime import datetime as _dt
                return _dt.fromisoformat(s).timestamp()
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid ISO timestamp: {s}",
                ) from e

        since_ts = _parse(since)
        until_ts = _parse(until)

        store_summary = state.usage_store.query_summary(
            since=since_ts, until=until_ts,
        )

        # Build the summary response using SAME shape as /usage so
        # clients can reuse the parser. Mix in-memory tracker (current
        # session) with persistent totals.
        t = state.tracker
        summary_resp = UsageResponse(
            total_calls=t.total_calls,
            total_prompt_tokens=t.total_prompt_tokens,
            total_completion_tokens=t.total_completion_tokens,
            total_tokens=t.total_tokens,
            total_elapsed=t.total_elapsed,
            by_phase={
                p: PhaseUsage(
                    calls=int(b["calls"]),
                    prompt=int(b["prompt"]),
                    completion=int(b["completion"]),
                    elapsed=float(b["elapsed"]),
                )
                for p, b in store_summary.by_phase.items()
            },
            persistent_total_calls=store_summary.total_calls,
            persistent_total_prompt=store_summary.total_prompt,
            persistent_total_completion=store_summary.total_completion,
        )

        events = state.usage_store.query_events(
            since=since_ts,
            until=until_ts,
            phase=phase,
            limit=limit,
        )
        event_out = [
            UsageEventOut(
                timestamp=e.timestamp,
                phase=e.phase,
                prompt=e.prompt,
                completion=e.completion,
                elapsed=e.elapsed,
            )
            for e in events
        ]

        daily_out: list[UsageDailyOut] = []
        if include_daily:
            # Rollup any missing days before querying so the response
            # reflects the latest state.
            state.usage_store.rollup_range(since=since, until=until)
            for row in state.usage_store.query_daily(since=since, until=until):
                daily_out.append(
                    UsageDailyOut(
                        date=row["date"],
                        total_calls=row["total_calls"],
                        total_prompt=row["total_prompt"],
                        total_completion=row["total_completion"],
                        total_elapsed=row["total_elapsed"],
                        by_phase=row["by_phase"],
                    )
                )

        return UsageHistoryResponse(
            summary=summary_resp,
            events=event_out,
            daily=daily_out,
        )

    @app.get("/usage/history/stream")
    def usage_history_stream(
        poll_interval: float = 2.0,
        initial_limit: int = 20,
        max_events: int = 0,
        max_duration: float = 900.0,
    ):
        """Stream usage events as SSE frames (v0.15).

        On connect, emits an ``initial`` frame with the N most recent
        events, then polls the ``usage_events`` table every
        ``poll_interval`` seconds and emits an ``event`` frame for
        each newly-appended row. Plus a ``heartbeat`` every poll
        cycle so the client sees life signals even on an idle store.

        Query parameters:
          * ``poll_interval``  — seconds between polls (default 2.0)
          * ``initial_limit``  — how many recent events to backfill (default 20)
          * ``max_events``     — stop after N new events (0 = unlimited)
          * ``max_duration``   — hard cap on total stream seconds (default 900)

        Returns ``503`` when ``--usage-db`` isn't set.
        """
        if state.usage_store is None:
            raise HTTPException(
                status_code=503,
                detail="Usage persistence is not enabled.",
            )

        import json as _json

        store = state.usage_store

        # Capture the current tip so we only stream NEW events.
        initial_events = store.query_events(limit=initial_limit)

        def _event_payload(e) -> dict:
            return {
                "timestamp": e.timestamp,
                "phase": e.phase,
                "prompt": e.prompt,
                "completion": e.completion,
                "elapsed": e.elapsed,
            }

        last_seen_ts = (
            initial_events[0].timestamp if initial_events else 0.0
        )

        def event_stream():
            nonlocal last_seen_ts
            # 1. Initial snapshot.
            snapshot = [
                _event_payload(e) for e in reversed(initial_events)
            ]
            yield "event: initial\n"
            yield (
                f"data: {_json.dumps({'events': snapshot}, ensure_ascii=False)}\n\n"
            )

            import time as _time

            emitted = 0
            # Cap the loop at ``max_duration`` seconds so idle
            # clients don't hold the worker forever. Clients who
            # want a longer subscription should reconnect.
            start = _time.time()
            while _time.time() - start < max_duration:
                _time.sleep(max(0.1, poll_interval))
                fresh = store.query_events(limit=500)
                # Keep only events strictly newer than what we've
                # already shipped. Ordering returned by
                # query_events is "most-recent first", so reverse
                # for a stable stream order.
                new_rows = [e for e in fresh if e.timestamp > last_seen_ts]
                if new_rows:
                    for e in reversed(new_rows):
                        payload = _event_payload(e)
                        yield "event: event\n"
                        yield (
                            f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"
                        )
                        emitted += 1
                        last_seen_ts = max(last_seen_ts, e.timestamp)
                        if max_events and emitted >= max_events:
                            yield "event: done\n"
                            yield "data: {}\n\n"
                            return
                else:
                    yield "event: heartbeat\n"
                    yield (
                        f"data: {_json.dumps({'last_seen': last_seen_ts})}\n\n"
                    )
            yield "event: done\n"
            yield "data: {}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── v0.10/v0.11 SSE streaming endpoints ───────────────────────────────

    def _stream_retrieval(
        query: str,
        *,
        decompose: bool,
        tags: list[str] | None,
        after: str | None,
        before: str | None,
        history: list[tuple[str, str]] | None,
        session: "ChatSession | None",
        include_session: bool,
    ):
        """Shared SSE streaming worker used by /ask/stream and /chat/stream.

        Spawns a background thread that runs retrieval, bridges
        ``TraceStep`` events to an SSE-friendly generator, and
        emits live ``usage`` deltas between iterations (v0.11).

        When ``session`` is provided, the turn is recorded in the
        session's history after retrieval completes.
        """
        import json as _json
        import queue
        import threading as _threading

        event_queue: queue.Queue[tuple[str, dict] | None] = queue.Queue()

        # Per-request tracker so we can emit live deltas without
        # interfering with the server-wide cumulative tracker.
        # Writes to both — the shared tracker stays authoritative
        # for /usage, while this local one drives per-request SSE.
        local_tracker = UsageTracker()

        def on_event(step: TraceStep) -> None:
            event_queue.put(
                (
                    "trace",
                    {
                        "phase": step.phase,
                        "node_id": step.node_id,
                        "detail": step.detail,
                    },
                )
            )
            # v0.11: emit a live usage snapshot after each trace so
            # the client can show a running token meter.
            event_queue.put(
                (
                    "usage",
                    {
                        "total_calls": local_tracker.total_calls,
                        "prompt_tokens": local_tracker.total_prompt_tokens,
                        "completion_tokens": local_tracker.total_completion_tokens,
                        "total_tokens": local_tracker.total_tokens,
                        "elapsed": local_tracker.total_elapsed,
                    },
                )
            )

        # Wrap the shared chat_fn so every call updates BOTH trackers.
        # v0.13: snapshot the shared tracker before/after each call so
        # we can extract the REAL token counts that LiteLLM just
        # reported (rather than a char/3 estimate). If the shared
        # tracker wasn't advanced by this call (e.g. the underlying
        # chat_fn is tracker-less), we fall back to the char/3
        # estimate so the SSE client still gets a usage frame.
        import time as _time

        shared_tracker = state.tracker

        def tracked_chat_fn(prompt: str) -> str:
            pre_prompt = shared_tracker.total_prompt_tokens
            pre_completion = shared_tracker.total_completion_tokens

            t0 = _time.time()
            response_text = state.chat_fn(prompt)
            elapsed = _time.time() - t0

            delta_prompt = shared_tracker.total_prompt_tokens - pre_prompt
            delta_completion = (
                shared_tracker.total_completion_tokens - pre_completion
            )

            if delta_prompt == 0 and delta_completion == 0:
                # Shared tracker didn't advance — the underlying
                # chat_fn doesn't report real tokens. Fall back to
                # the char/3 heuristic so the client still sees a
                # meaningful (if approximate) usage frame.
                delta_prompt = max(1, len(prompt) // 3)
                delta_completion = max(1, len(response_text) // 3)

            local_tracker.record(
                "other",
                delta_prompt,
                delta_completion,
                elapsed,
            )
            return response_text

        def worker() -> None:
            try:
                root = _filtered_root(tags, after, before)
                if decompose:
                    result = run_decomposed_retrieval(
                        query, root, tracked_chat_fn,
                        link_index=state.link_index,
                        on_event=on_event,
                    )
                else:
                    result = run_retrieval(
                        query, root, tracked_chat_fn,
                        link_index=state.link_index,
                        on_event=on_event,
                        history=history,
                    )

                answer_payload: dict = {
                    "answer": result.answer,
                    "cited_nodes": result.cited_nodes,
                    "iterations_used": result.iterations_used,
                    "usage": {
                        "prompt_tokens": local_tracker.total_prompt_tokens,
                        "completion_tokens": local_tracker.total_completion_tokens,
                        "total_tokens": local_tracker.total_tokens,
                    },
                }

                if include_session and session is not None:
                    session.history.append((query, result.answer[:500]))
                    answer_payload["session_id"] = session.sid
                    answer_payload["turn"] = len(session.history)

                event_queue.put(("answer", answer_payload))
            except Exception as exc:  # pragma: no cover — defensive
                event_queue.put(("error", {"message": str(exc)}))
            finally:
                event_queue.put(None)  # sentinel

        _threading.Thread(target=worker, daemon=True).start()

        def event_stream():
            while True:
                item = event_queue.get()
                if item is None:
                    break
                event_name, data = item
                yield f"event: {event_name}\n"
                yield f"data: {_json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest = Body(...)):  # noqa: B008
        """Stream retrieval events as Server-Sent Events (v0.10).

        Events:

          * ``trace``  — one per ``TraceStep`` emitted by the loop
          * ``usage``  — cumulative token deltas after each step (v0.11)
          * ``answer`` — the final answer + cited nodes + usage summary
          * ``error``  — unexpected exceptions during retrieval
        """
        return _stream_retrieval(
            req.query,
            decompose=req.decompose,
            tags=req.tags,
            after=req.after,
            before=req.before,
            history=None,
            session=None,
            include_session=False,
        )

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest = Body(...)):  # noqa: B008
        """Stream chat retrieval with session history (v0.11).

        Mirrors ``/ask/stream`` but threads the server-side
        ``ChatSession`` through so follow-up questions see prior
        turns. The final ``answer`` event includes the session_id
        (new or reused) and turn number.
        """
        _prune_expired_sessions(state)
        sid = req.session_id or uuid.uuid4().hex
        session = state.sessions.get(sid)
        if session is None:
            session = ChatSession(sid=sid)
            state.sessions[sid] = session
        session.last_active = time.time()

        return _stream_retrieval(
            req.query,
            decompose=req.decompose,
            tags=None,
            after=None,
            before=None,
            history=list(session.history),
            session=session,
            include_session=True,
        )

    # ── v0.12 WebSocket endpoint with interrupt support ───────────────────

    @app.websocket("/ask/ws")
    async def ask_ws(websocket: WebSocket) -> None:
        """Bidirectional WebSocket channel for retrieval with cancellation.

        Protocol:

          Client → Server messages::

            {"type": "ask", "query": "...", "decompose": false}
            {"type": "cancel"}
            {"type": "ping"}

          Server → Client messages::

            {"type": "trace",  "phase": "...", "node_id": "...", "detail": "..."}
            {"type": "usage",  "total_calls": N, "prompt_tokens": ..., ...}
            {"type": "answer", "answer": "...", "cited_nodes": [...], "iterations_used": N}
            {"type": "cancelled"}
            {"type": "error",  "message": "..."}
            {"type": "pong"}

        One retrieval runs per connection at a time. Sending
        ``{"type": "ask", ...}`` while a query is in flight queues
        the new request after the current one finishes.
        """
        import asyncio as _asyncio
        import json as _json
        import queue as _queue
        import threading as _threading

        await websocket.accept()

        stop_event = _threading.Event()
        event_queue: _queue.Queue[tuple[str, dict] | None] = _queue.Queue()

        local_tracker = UsageTracker()
        import time as _time

        shared_tracker_ws = state.tracker

        def tracked_chat_fn(prompt: str) -> str:
            pre_prompt = shared_tracker_ws.total_prompt_tokens
            pre_completion = shared_tracker_ws.total_completion_tokens

            t0 = _time.time()
            text = state.chat_fn(prompt)
            elapsed = _time.time() - t0

            # v0.13: real delta from shared tracker with char/3 fallback.
            delta_prompt = shared_tracker_ws.total_prompt_tokens - pre_prompt
            delta_completion = (
                shared_tracker_ws.total_completion_tokens - pre_completion
            )
            if delta_prompt == 0 and delta_completion == 0:
                delta_prompt = max(1, len(prompt) // 3)
                delta_completion = max(1, len(text) // 3)

            local_tracker.record(
                "other",
                delta_prompt,
                delta_completion,
                elapsed,
            )
            return text

        def on_event(step: TraceStep) -> None:
            event_queue.put(
                (
                    "trace",
                    {
                        "phase": step.phase,
                        "node_id": step.node_id,
                        "detail": step.detail,
                    },
                )
            )
            event_queue.put(
                (
                    "usage",
                    {
                        "total_calls": local_tracker.total_calls,
                        "prompt_tokens": local_tracker.total_prompt_tokens,
                        "completion_tokens": local_tracker.total_completion_tokens,
                        "total_tokens": local_tracker.total_tokens,
                        "elapsed": local_tracker.total_elapsed,
                    },
                )
            )

        def run_query(
            query: str,
            *,
            decompose: bool,
            max_tokens: int | None,
            json_mode: bool,
            reuse_context: bool,
            prompt_cache: bool,
        ) -> None:
            # v0.16: per-request prompt cache opt-in. When the client
            # asks for it AND the server was started with --prompt-cache
            # (state.system_chat_fn attached), thread it into retrieval.
            active_system_chat_fn = (
                state.system_chat_fn if (prompt_cache and state.system_chat_fn) else None
            )
            try:
                if decompose:
                    # Decompose doesn't accept system_chat_fn yet; use
                    # regular path (json_mode and decompose override
                    # prompt cache benefit anyway).
                    result = run_decomposed_retrieval(
                        query, state.root, tracked_chat_fn,
                        link_index=state.link_index,
                        on_event=on_event,
                        max_tokens=max_tokens,
                        tracker=local_tracker,
                        json_mode=json_mode,
                        reuse_context=reuse_context,
                    )
                else:
                    result = run_retrieval(
                        query, state.root, tracked_chat_fn,
                        link_index=state.link_index,
                        on_event=on_event,
                        should_stop=stop_event.is_set,
                        max_tokens=max_tokens,
                        tracker=local_tracker,
                        json_mode=json_mode,
                        reuse_context=reuse_context,
                        system_chat_fn=active_system_chat_fn,
                    )
                if stop_event.is_set():
                    event_queue.put(("cancelled", {}))
                else:
                    event_queue.put(
                        (
                            "answer",
                            {
                                "answer": result.answer,
                                "cited_nodes": result.cited_nodes,
                                "iterations_used": result.iterations_used,
                            },
                        )
                    )
            except Exception as exc:  # pragma: no cover
                event_queue.put(("error", {"message": str(exc)}))
            finally:
                event_queue.put(None)  # sentinel

        current_worker: _threading.Thread | None = None

        async def drain_queue() -> None:
            """Pull events from the sync queue and push them to the socket."""
            while True:
                item = await _asyncio.to_thread(event_queue.get)
                if item is None:
                    return
                event_name, data = item
                await websocket.send_text(
                    _json.dumps({"type": event_name, **data}, ensure_ascii=False)
                )

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = _json.loads(raw)
                except _json.JSONDecodeError:
                    await websocket.send_text(
                        _json.dumps({"type": "error", "message": "invalid JSON"})
                    )
                    continue

                msg_type = msg.get("type")

                if msg_type == "ping":
                    await websocket.send_text(_json.dumps({"type": "pong"}))
                    continue

                if msg_type == "cancel":
                    stop_event.set()
                    continue

                if msg_type == "ask":
                    if current_worker is not None and current_worker.is_alive():
                        await websocket.send_text(
                            _json.dumps(
                                {
                                    "type": "error",
                                    "message": "a query is already in flight",
                                }
                            )
                        )
                        continue

                    query = msg.get("query", "")
                    if not query:
                        await websocket.send_text(
                            _json.dumps(
                                {"type": "error", "message": "empty query"}
                            )
                        )
                        continue

                    # Reset per-query state.
                    stop_event.clear()
                    while not event_queue.empty():
                        try:
                            event_queue.get_nowait()
                        except _queue.Empty:
                            break

                    # v0.15: accept token_split / max_tokens /
                    # json_mode / reuse_context from the ask frame.
                    parsed_max_tokens = msg.get("max_tokens")
                    token_split_spec = msg.get("token_split")
                    if token_split_spec and parsed_max_tokens:
                        # Client-side split is: A:B:C ratios. Apply
                        # the same parse logic as the CLI. Only the
                        # retrieve+synth portion matters for the
                        # run_retrieval call.
                        parts = str(token_split_spec).split(":")
                        try:
                            ratios = [float(p) for p in parts]
                            if len(ratios) == 3 and sum(ratios) > 0:
                                s = sum(ratios)
                                retrieve_cap = int(
                                    parsed_max_tokens * (ratios[1] + ratios[2]) / s
                                )
                                parsed_max_tokens = max(1, retrieve_cap)
                        except ValueError:
                            pass  # fall through with original max_tokens

                    current_worker = _threading.Thread(
                        target=run_query,
                        args=(query,),
                        kwargs={
                            "decompose": bool(msg.get("decompose", False)),
                            "max_tokens": parsed_max_tokens,
                            "json_mode": bool(msg.get("json_mode", False)),
                            "reuse_context": bool(msg.get("reuse_context", False)),
                            "prompt_cache": bool(msg.get("prompt_cache", False)),
                        },
                        daemon=True,
                    )
                    current_worker.start()
                    # Drain until this query finishes (sentinel).
                    await drain_queue()
                    continue

                await websocket.send_text(
                    _json.dumps(
                        {"type": "error", "message": f"unknown type: {msg_type}"}
                    )
                )
        except WebSocketDisconnect:
            # Client vanished — best-effort signal to the worker.
            stop_event.set()
            return

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Server bootstrap (called from cli.py)
# ─────────────────────────────────────────────────────────────────────────────


def build_initial_state(
    vaults: list[Path],
    *,
    folder: str | None,
    model: str,
    num_ctx: int,
    max_workers: int,
    chat_fn: ChatFn,
) -> ServerState:
    """Scan the vault(s) and build the initial warm state.

    This happens once at server startup. All HTTP handlers share the
    resulting state.
    """
    if len(vaults) > 1:
        root = scan_multi_vault([(v, folder) for v in vaults])
    else:
        root = scan_folder(vaults[0], folder)

    primary_vault = vaults[0]
    summary_cache = SummaryCache(primary_vault)
    tree_cache = TreeCache(primary_vault)

    summarize_atomic_notes(
        root, chat_fn,
        summary_cache=summary_cache,
        model_id=model,
        max_workers=max_workers,
    )

    long_count = sum(
        1 for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
    )
    if long_count > 0:
        build_long_subtrees(
            root,
            vault_root=primary_vault,
            model_id=model,
            chat_fn=chat_fn,
            cache=tree_cache,
        )

    link_index = build_link_index(root)

    return ServerState(
        vaults=vaults,
        folder=folder,
        model=model,
        num_ctx=num_ctx,
        max_workers=max_workers,
        chat_fn=chat_fn,
        root=root,
        link_index=link_index,
        summary_cache=summary_cache,
        tree_cache=tree_cache,
    )
