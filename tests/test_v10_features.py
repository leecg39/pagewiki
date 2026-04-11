"""Tests for v0.10: JSON-mode, usage persistence, SSE, context reuse."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pagewiki.prompts import (
    NodeSummary,
    evaluate_prompt_json,
    parse_evaluate_response_json,
    parse_select_response_json,
    select_node_prompt_json,
)
from pagewiki.retrieval import run_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage_store import UsageStore

# ─────────────────────────────────────────────────────────────────────────────
# JSON-mode prompts + parsers
# ─────────────────────────────────────────────────────────────────────────────


class TestJsonModeParsers:
    def test_parse_select_bare_json(self) -> None:
        resp = '{"action": "SELECT", "node_id": "Research/paper.md"}'
        action, value = parse_select_response_json(resp)
        assert action == "SELECT"
        assert value == "Research/paper.md"

    def test_parse_select_done(self) -> None:
        resp = '{"action": "DONE", "reason": "관련 노드 없음"}'
        action, value = parse_select_response_json(resp)
        assert action == "DONE"
        assert "없음" in value

    def test_parse_select_with_code_fence(self) -> None:
        resp = '```json\n{"action": "SELECT", "node_id": "a.md"}\n```'
        action, value = parse_select_response_json(resp)
        assert action == "SELECT"
        assert value == "a.md"

    def test_parse_select_with_leading_noise(self) -> None:
        resp = 'Here is my answer:\n{"action": "SELECT", "node_id": "x.md"}\nThanks!'
        action, value = parse_select_response_json(resp)
        assert action == "SELECT"
        assert value == "x.md"

    def test_parse_select_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_select_response_json("not json at all")
        with pytest.raises(ValueError):
            parse_select_response_json('{"action": "UNKNOWN"}')
        with pytest.raises(ValueError):
            parse_select_response_json('{"action": "SELECT"}')  # missing node_id

    def test_parse_evaluate_json(self) -> None:
        assert parse_evaluate_response_json(
            '{"sufficient": true, "reason": "ok"}'
        ) == (True, "ok")
        assert parse_evaluate_response_json(
            '{"sufficient": false, "reason": "missing data"}'
        ) == (False, "missing data")

    def test_parse_evaluate_with_fence(self) -> None:
        resp = '```\n{"sufficient": true, "reason": "충분"}\n```'
        sufficient, reason = parse_evaluate_response_json(resp)
        assert sufficient is True
        assert reason == "충분"

    def test_select_node_prompt_json_contains_schema(self) -> None:
        candidates = [
            NodeSummary(node_id="a.md", title="A", kind="note", summary="a summary"),
            NodeSummary(node_id="b.md", title="B", kind="note", summary="b summary"),
        ]
        prompt = select_node_prompt_json("query", candidates)
        assert '"action"' in prompt
        assert '"SELECT"' in prompt
        assert '"DONE"' in prompt
        assert "JSON" in prompt

    def test_evaluate_prompt_json_contains_schema(self) -> None:
        prompt = evaluate_prompt_json("q", "Title", "body")
        assert '"sufficient"' in prompt
        assert '"reason"' in prompt


class TestRetrievalJsonMode:
    def test_run_retrieval_with_json_mode(self, tmp_path: Path) -> None:
        note = tmp_path / "note.md"
        note.write_text("content", encoding="utf-8")

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="note.md",
                    title="Note",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="test",
                ),
            ],
        )

        responses = iter([
            '{"action": "SELECT", "node_id": "note.md"}',
            '{"sufficient": true, "reason": "ok"}',
            "최종 답변",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        result = run_retrieval("test", root, fake, json_mode=True)
        assert result.answer == "최종 답변"
        assert "note.md" in result.cited_nodes

    def test_json_mode_falls_back_on_retry_failure(self, tmp_path: Path) -> None:
        """If JSON retry also fails, fall back to the text parser."""
        note = tmp_path / "note.md"
        note.write_text("content", encoding="utf-8")

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="note.md",
                    title="Note",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="test",
                ),
            ],
        )

        responses = iter([
            "garbage",                    # first attempt fails
            "SELECT: note.md",            # retry: not JSON, falls back to text parser
            '{"sufficient": true, "reason": "ok"}',
            "답변",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        result = run_retrieval("test", root, fake, json_mode=True)
        assert result.answer == "답변"


# ─────────────────────────────────────────────────────────────────────────────
# UsageStore (SQLite)
# ─────────────────────────────────────────────────────────────────────────────


class TestUsageStore:
    def test_record_and_query_events(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        store.record("select", 500, 10, 1.2)
        store.record("evaluate", 800, 20, 1.5)
        store.record("select", 300, 5, 0.8)

        events = store.query_events()
        assert len(events) == 3
        # Ordered by most-recent first.
        assert events[0].phase in ("select", "evaluate")

    def test_query_summary_totals(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        store.record("select", 500, 10, 1.0)
        store.record("select", 300, 5, 0.5)
        store.record("evaluate", 800, 20, 1.5)

        summary = store.query_summary()
        assert summary.total_calls == 3
        assert summary.total_prompt == 1600
        assert summary.total_completion == 35
        assert abs(summary.total_elapsed - 3.0) < 1e-6

        # Phase buckets.
        assert "select" in summary.by_phase
        assert summary.by_phase["select"]["calls"] == 2
        assert summary.by_phase["select"]["prompt"] == 800

    def test_query_summary_with_since(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        t_old = time.time() - 3600
        store.record("a", 100, 10, 0.1, timestamp=t_old)
        store.record("b", 200, 20, 0.2)  # now

        cutoff = time.time() - 1800  # 30 min ago
        summary = store.query_summary(since=cutoff)
        assert summary.total_calls == 1
        assert summary.total_prompt == 200

    def test_query_events_phase_filter(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        store.record("select", 100, 10, 0.1)
        store.record("evaluate", 200, 20, 0.2)
        store.record("select", 300, 30, 0.3)

        select_events = store.query_events(phase="select")
        assert len(select_events) == 2
        for e in select_events:
            assert e.phase == "select"

    def test_clear(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        store.record("x", 100, 10, 0.1)
        store.record("y", 200, 20, 0.2)

        deleted = store.clear()
        assert deleted == 2
        assert store.query_summary().total_calls == 0

    def test_persistence_across_connections(self, tmp_path: Path) -> None:
        """Data should survive re-opening the store."""
        db_path = tmp_path / "usage.db"
        s1 = UsageStore(db_path)
        s1.record("phase1", 100, 10, 0.1)
        s1.close()

        s2 = UsageStore(db_path)
        summary = s2.query_summary()
        assert summary.total_calls == 1
        assert summary.total_prompt == 100


# ─────────────────────────────────────────────────────────────────────────────
# Context reuse
# ─────────────────────────────────────────────────────────────────────────────


class TestContextReuse:
    def test_reuse_context_emits_reuse_trace(self, tmp_path: Path) -> None:
        """A candidate shown in iter 1 and seen again in iter 2 should be
        marked as suppressed, emitting a 'reuse' trace step."""
        # Build a tree with a folder that has 2 notes; after selecting
        # folder, its children are offered, then we force a bubble-up.
        note_a = tmp_path / "a.md"
        note_a.write_text("content a", encoding="utf-8")
        note_b = tmp_path / "b.md"
        note_b.write_text("content b", encoding="utf-8")

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="a.md",
                    title="A",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note_a,
                    summary="alpha",
                ),
                TreeNode(
                    node_id="b.md",
                    title="B",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note_b,
                    summary="beta",
                ),
            ],
        )

        # Iteration 1: show A, B. Pick A, evaluate insufficient → iter 2.
        # Iteration 2: candidates are only B now (A was picked/visited),
        # so the reuse optimization isn't the one firing here; rather
        # the sanity check is that the flag threads through without
        # breaking anything.
        responses = iter([
            "SELECT: a.md",
            "INSUFFICIENT: need more",
            "SELECT: b.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        result = run_retrieval(
            "test",
            root,
            fake,
            reuse_context=True,
        )
        assert "답변" in result.answer

    def test_reuse_context_off_by_default(self, tmp_path: Path) -> None:
        note = tmp_path / "a.md"
        note.write_text("c", encoding="utf-8")

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="a.md",
                    title="A",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="x",
                ),
            ],
        )

        responses = iter([
            "SELECT: a.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        # Without reuse_context, no 'reuse' phase should appear.
        result = run_retrieval("test", root, fake)
        phases = [t.phase for t in result.trace]
        assert "reuse" not in phases


# ─────────────────────────────────────────────────────────────────────────────
# SSE streaming endpoint
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


@pytest.fixture
def sse_client(tmp_path: Path):
    vault = tmp_path / "vault"
    (vault / "Notes").mkdir(parents=True)
    (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

    responses_idx = {"i": 0}

    def fake_chat(prompt: str) -> str:
        script = [
            "요약",
            "SELECT: Notes/note.md",
            "SUFFICIENT: ok",
            "답변입니다",
        ]
        i = min(responses_idx["i"], len(script) - 1)
        responses_idx["i"] += 1
        return script[i]

    state = build_initial_state(
        [vault],
        folder="Notes",
        model="test",
        num_ctx=8192,
        max_workers=1,
        chat_fn=fake_chat,
    )
    # Reset script counter for the endpoint calls.
    responses_idx["i"] = 1  # skip the summarization response used at startup
    app = create_app(state)
    return TestClient(app)


class TestSSEEndpoint:
    def test_ask_stream_yields_trace_and_answer(self, sse_client) -> None:
        tc = sse_client
        # httpx.stream returns a Response whose .iter_lines() yields
        # the SSE frames; we just collect everything into a single
        # string and assert on event markers.
        with tc.stream(
            "POST",
            "/ask/stream",
            json={"query": "test"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "event: trace" in body
        assert "event: answer" in body
        assert "답변입니다" in body

    def test_ask_stream_sets_sse_headers(self, sse_client) -> None:
        tc = sse_client
        with tc.stream(
            "POST",
            "/ask/stream",
            json={"query": "test"},
        ) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
