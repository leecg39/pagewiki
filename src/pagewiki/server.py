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

Then:

    curl -X POST http://localhost:8000/ask \
        -H 'Content-Type: application/json' \
        -d '{"query": "What is X?"}'
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cache import SummaryCache, TreeCache
from .retrieval import (
    RetrievalResult,
    TraceStep,
    run_decomposed_retrieval,
    run_retrieval,
)
from .tree import NoteTier, TreeNode
from .vault import (
    build_long_subtrees,
    filter_tree,
    scan_folder,
    scan_multi_vault,
    summarize_atomic_notes,
)
from .wiki_links import LinkIndex, build_link_index

if TYPE_CHECKING:
    from fastapi import FastAPI

ChatFn = Callable[[str], str]

# Chat sessions expire after 1 hour of inactivity.
SESSION_TTL_SECONDS = 3600


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────


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


@dataclass
class ChatSession:
    """Server-side conversation state for one chat client."""

    sid: str
    history: list[tuple[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


def _prune_expired_sessions(state: ServerState) -> None:
    """Drop sessions that haven't been touched in SESSION_TTL_SECONDS."""
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [sid for sid, s in state.sessions.items() if s.last_active < cutoff]
    for sid in expired:
        del state.sessions[sid]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ─────────────────────────────────────────────────────────────────────────────


def create_app(state: ServerState) -> FastAPI:
    """Build a FastAPI application bound to the given server state.

    FastAPI + Pydantic are imported inside this function so they
    stay optional dependencies — ``pagewiki serve`` will surface a
    clear install hint if either is missing.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise RuntimeError(
            "FastAPI is required for `pagewiki serve`. Install with:\n"
            "    pip install 'pagewiki[server]'\n"
            f"Original error: {e}"
        ) from e

    app = FastAPI(title="pagewiki API", version="0.7.0")

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

    @app.get("/health")
    def health() -> dict[str, Any]:
        note_count = sum(1 for n in state.root.walk() if n.kind == "note")
        return {
            "status": "ok",
            "version": "0.7.0",
            "model": state.model,
            "vault_count": len(state.vaults),
            "note_count": note_count,
            "active_sessions": len(state.sessions),
        }

    @app.post("/scan", response_model=ScanResponse)
    def scan() -> ScanResponse:
        summary = state.rescan()
        return ScanResponse(**summary)

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
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
    def chat(req: ChatRequest) -> ChatResponse:
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
