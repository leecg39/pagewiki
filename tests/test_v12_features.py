"""Tests for v0.12: daily rollup, WebSocket, cross-vault, plugin server mode."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from pagewiki.retrieval import run_cross_vault_retrieval, run_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage_store import UsageStore

# ─────────────────────────────────────────────────────────────────────────────
# Daily rollup (UsageStore)
# ─────────────────────────────────────────────────────────────────────────────


class TestDailyRollup:
    def test_rollup_day_aggregates_events(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")

        # Record three events on the same day.
        today = date.today().isoformat()
        today_ts = datetime.fromisoformat(today).timestamp()
        store.record("select", 500, 10, 1.0, timestamp=today_ts + 100)
        store.record("select", 300, 5, 0.5, timestamp=today_ts + 200)
        store.record("evaluate", 800, 20, 1.5, timestamp=today_ts + 300)

        written = store.rollup_day(today)
        assert written == 1

        rows = store.query_daily()
        assert len(rows) == 1
        row = rows[0]
        assert row["date"] == today
        assert row["total_calls"] == 3
        assert row["total_prompt"] == 1600
        assert row["total_completion"] == 35
        assert "select" in row["by_phase"]
        assert row["by_phase"]["select"]["calls"] == 2

    def test_rollup_day_invalid_date_raises(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        with pytest.raises(ValueError):
            store.rollup_day("not-a-date")

    def test_rollup_day_empty_is_idempotent(self, tmp_path: Path) -> None:
        """An empty day writes a zero row once, then is a no-op."""
        store = UsageStore(tmp_path / "usage.db")

        # First call: no events → writes a zero row.
        first = store.rollup_day("2024-01-01")
        assert first == 1

        # Second call: row already exists → returns 0 (idempotent).
        second = store.rollup_day("2024-01-01")
        assert second == 0

        rows = store.query_daily()
        assert len(rows) == 1
        assert rows[0]["total_calls"] == 0

    def test_rollup_range_defaults_to_earliest_and_yesterday(
        self, tmp_path: Path
    ) -> None:
        store = UsageStore(tmp_path / "usage.db")

        # Two events on two different past days.
        three_days_ago = (date.today() - timedelta(days=3)).isoformat()
        two_days_ago = (date.today() - timedelta(days=2)).isoformat()
        ts3 = datetime.fromisoformat(three_days_ago).timestamp() + 100
        ts2 = datetime.fromisoformat(two_days_ago).timestamp() + 100
        store.record("a", 100, 10, 0.1, timestamp=ts3)
        store.record("b", 200, 20, 0.2, timestamp=ts2)

        written = store.rollup_range()
        # At least 2 days rolled up (may be more depending on range).
        assert written >= 2

        rows = store.query_daily()
        dates = {r["date"] for r in rows}
        assert three_days_ago in dates
        assert two_days_ago in dates

    def test_rollup_range_with_explicit_bounds(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")

        ts = datetime.fromisoformat("2024-06-15").timestamp() + 100
        store.record("x", 100, 10, 0.1, timestamp=ts)

        written = store.rollup_range(since="2024-06-15", until="2024-06-15")
        assert written == 1

        rows = store.query_daily(since="2024-06-15", until="2024-06-15")
        assert len(rows) == 1
        assert rows[0]["total_prompt"] == 100

    def test_clear_removes_both_tables(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        store.record("x", 100, 10, 0.1, timestamp=time.time())
        store.rollup_day(date.today().isoformat())

        store.clear()
        assert store.query_summary().total_calls == 0
        assert store.query_daily() == []


# ─────────────────────────────────────────────────────────────────────────────
# Cross-vault retrieval
# ─────────────────────────────────────────────────────────────────────────────


def _one_note_tree(tmp_path: Path, name: str, content: str) -> TreeNode:
    note = tmp_path / f"{name}.md"
    note.write_text(content, encoding="utf-8")
    return TreeNode(
        node_id="",
        title=name,
        kind="folder",
        children=[
            TreeNode(
                node_id=f"{name}.md",
                title=name.capitalize(),
                kind="note",
                tier=NoteTier.ATOMIC,
                file_path=note,
                summary=f"{name} summary",
            ),
        ],
    )


class TestCrossVaultRetrieval:
    def test_runs_retrieval_per_vault_and_synthesizes(
        self, tmp_path: Path
    ) -> None:
        v1_dir = tmp_path / "v1"
        v1_dir.mkdir()
        v2_dir = tmp_path / "v2"
        v2_dir.mkdir()

        root_v1 = _one_note_tree(v1_dir, "alpha", "alpha content")
        root_v2 = _one_note_tree(v2_dir, "beta", "beta content")

        responses = iter([
            # Vault 1
            "SELECT: alpha.md",
            "SUFFICIENT: ok",
            "v1 답변",
            # Vault 2
            "SELECT: beta.md",
            "SUFFICIENT: ok",
            "v2 답변",
            # Synthesis
            "종합 답변",
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        result = run_cross_vault_retrieval(
            "query",
            [root_v1, root_v2],
            fake_chat,
            vault_labels=["v1", "v2"],
        )
        assert result.answer == "종합 답변"
        # Citations prefixed with vault label.
        assert any(c.startswith("v1::") for c in result.cited_nodes)
        assert any(c.startswith("v2::") for c in result.cited_nodes)

    def test_cross_vault_emits_trace_events(self, tmp_path: Path) -> None:
        v1_dir = tmp_path / "v1"
        v1_dir.mkdir()
        root = _one_note_tree(v1_dir, "note", "content")

        responses = iter([
            "SELECT: note.md",
            "SUFFICIENT: ok",
            "답변",
            "합성",
        ])

        events = []

        def fake_chat(prompt: str) -> str:
            return next(responses)

        run_cross_vault_retrieval(
            "query",
            [root],
            fake_chat,
            vault_labels=["only"],
            on_event=lambda s: events.append(s),
        )
        phases = [e.phase for e in events]
        assert "cross-vault" in phases

    def test_empty_vault_list_raises(self) -> None:
        with pytest.raises(ValueError):
            run_cross_vault_retrieval("q", [], lambda p: "")

    def test_mismatched_labels_raises(self, tmp_path: Path) -> None:
        root = _one_note_tree(tmp_path, "x", "content")
        with pytest.raises(ValueError):
            run_cross_vault_retrieval(
                "q",
                [root],
                lambda p: "",
                vault_labels=["a", "b"],  # mismatched length
            )


# ─────────────────────────────────────────────────────────────────────────────
# should_stop interrupt
# ─────────────────────────────────────────────────────────────────────────────


class TestShouldStop:
    def test_should_stop_aborts_loop(self, tmp_path: Path) -> None:
        note = tmp_path / "n.md"
        note.write_text("content", encoding="utf-8")
        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="n.md",
                    title="N",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="x",
                ),
            ],
        )

        # Always return True so the first iteration check aborts immediately.
        result = run_retrieval(
            "test",
            root,
            lambda p: "should not be called",
            should_stop=lambda: True,
        )
        assert "근거 부족" in result.answer or "중단" in str(result.trace)

    def test_should_stop_none_is_default(self, tmp_path: Path) -> None:
        note = tmp_path / "n.md"
        note.write_text("content", encoding="utf-8")
        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="n.md",
                    title="N",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="x",
                ),
            ],
        )

        responses = iter([
            "SELECT: n.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake(p: str) -> str:
            return next(responses)

        result = run_retrieval("test", root, fake)
        assert result.answer == "답변"


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


@pytest.fixture
def ws_client(tmp_path: Path):
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
    app = create_app(state)
    return TestClient(app)


class TestWebSocketAsk:
    def test_ping_pong(self, ws_client) -> None:
        tc = ws_client
        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            response = json.loads(ws.receive_text())
            assert response["type"] == "pong"

    def test_ask_returns_trace_and_answer(self, ws_client) -> None:
        tc = ws_client
        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(json.dumps({"type": "ask", "query": "test"}))

            got_trace = False
            got_usage = False
            got_answer = False
            for _ in range(20):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "trace":
                    got_trace = True
                elif msg["type"] == "usage":
                    got_usage = True
                elif msg["type"] == "answer":
                    got_answer = True
                    break
                elif msg["type"] == "error":
                    pytest.fail(f"server error: {msg}")

            assert got_trace
            assert got_usage
            assert got_answer

    def test_invalid_json_rejected(self, ws_client) -> None:
        tc = ws_client
        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text("not json")
            response = json.loads(ws.receive_text())
            assert response["type"] == "error"
            assert "JSON" in response["message"]

    def test_empty_query_rejected(self, ws_client) -> None:
        tc = ws_client
        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(json.dumps({"type": "ask", "query": ""}))
            response = json.loads(ws.receive_text())
            assert response["type"] == "error"
