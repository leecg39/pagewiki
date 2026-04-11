"""Tests for v0.14: DB retention, /usage/history, Web UI, --token-split, prompt caching."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest

from pagewiki.cli import _parse_token_split
from pagewiki.prompts import (
    EVALUATE_SYSTEM,
    FINAL_ANSWER_SYSTEM,
    SELECT_NODE_SYSTEM,
    NodeSummary,
    evaluate_user_prompt,
    final_answer_user_prompt,
    select_node_user_prompt,
)
from pagewiki.retrieval import run_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage import UsageTracker
from pagewiki.usage_store import UsageStore
from pagewiki.webui import build_ui_html

# ─────────────────────────────────────────────────────────────────────────────
# UsageStore retention
# ─────────────────────────────────────────────────────────────────────────────


class TestUsageStoreRetention:
    def test_prune_events_before_removes_old(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")

        # Record events at various timestamps.
        now = time.time()
        store.record("a", 100, 10, 0.1, timestamp=now - 86400 * 30)  # 30 days ago
        store.record("b", 200, 20, 0.2, timestamp=now - 86400 * 10)  # 10 days ago
        store.record("c", 300, 30, 0.3, timestamp=now - 86400 * 1)   # 1 day ago

        # Prune events older than 5 days.
        cutoff = now - 86400 * 5
        deleted = store.prune_events_before(cutoff)
        assert deleted == 2  # 30-day + 10-day rows gone

        # Only the 1-day-old event should remain.
        remaining = store.query_summary()
        assert remaining.total_calls == 1
        assert remaining.total_prompt == 300

    def test_prune_preserves_daily_rollups(self, tmp_path: Path) -> None:
        """Rolling retention must aggregate affected days before deleting."""
        store = UsageStore(tmp_path / "usage.db")

        now = time.time()
        old_ts = now - 86400 * 15
        store.record("old", 500, 50, 0.5, timestamp=old_ts)

        store.prune_older_than_days(7)

        # Daily rollups should still have the old day's totals.
        rows = store.query_daily()
        assert rows
        old_date = datetime.fromtimestamp(old_ts).date().isoformat()
        matching = [r for r in rows if r["date"] == old_date]
        assert matching
        assert matching[0]["total_prompt"] == 500

    def test_prune_older_than_days_convenience(self, tmp_path: Path) -> None:
        store = UsageStore(tmp_path / "usage.db")
        now = time.time()
        store.record("x", 100, 10, 0.1, timestamp=now - 86400 * 20)
        store.record("y", 200, 20, 0.2, timestamp=now - 3600)  # 1 hour ago

        deleted = store.prune_older_than_days(7)
        assert deleted == 1

        summary = store.query_summary()
        assert summary.total_calls == 1
        assert summary.total_prompt == 200


# ─────────────────────────────────────────────────────────────────────────────
# --token-split parser
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenSplit:
    def test_none_returns_nones(self) -> None:
        assert _parse_token_split(None, None) == (None, None, None)
        assert _parse_token_split("", 1000) == (None, None, None)

    def test_basic_split(self) -> None:
        s, r, f = _parse_token_split("20:60:20", 10000)
        assert s == 2000
        assert r == 6000
        assert f == 2000

    def test_non_percentage_ratios(self) -> None:
        # 1:3:1 should give 20%/60%/20%.
        s, r, f = _parse_token_split("1:3:1", 10000)
        assert s == 2000
        assert r == 6000
        assert f == 2000

    def test_requires_total(self) -> None:
        import click
        with pytest.raises(click.UsageError):
            _parse_token_split("20:60:20", None)

    def test_rejects_invalid_shape(self) -> None:
        import click
        with pytest.raises(click.UsageError):
            _parse_token_split("20:60", 1000)
        with pytest.raises(click.UsageError):
            _parse_token_split("a:b:c", 1000)

    def test_rejects_negative(self) -> None:
        import click
        with pytest.raises(click.UsageError):
            _parse_token_split("-1:60:20", 1000)

    def test_rejects_all_zero(self) -> None:
        import click
        with pytest.raises(click.UsageError):
            _parse_token_split("0:0:0", 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt caching — user-only variants + system constants
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptCaching:
    def test_system_constants_exported(self) -> None:
        """Stable system strings should be importable as public constants."""
        assert SELECT_NODE_SYSTEM
        assert EVALUATE_SYSTEM
        assert FINAL_ANSWER_SYSTEM
        # Each should be a plain str (so Ollama can cache it).
        assert isinstance(SELECT_NODE_SYSTEM, str)
        assert isinstance(EVALUATE_SYSTEM, str)
        assert isinstance(FINAL_ANSWER_SYSTEM, str)

    def test_select_user_prompt_excludes_system(self) -> None:
        candidates = [
            NodeSummary(node_id="a.md", title="A", kind="note", summary="a"),
        ]
        user = select_node_user_prompt("query", candidates)
        assert SELECT_NODE_SYSTEM not in user
        # But must still contain the query + candidate info.
        assert "query" in user
        assert "a.md" in user

    def test_evaluate_user_prompt_excludes_system(self) -> None:
        user = evaluate_user_prompt("q", "Title", "body content")
        assert EVALUATE_SYSTEM not in user
        assert "Title" in user
        assert "body content" in user
        assert "SUFFICIENT" in user  # format reminder still there

    def test_final_user_prompt_excludes_system(self) -> None:
        user = final_answer_user_prompt("q", [("Note", "content")])
        assert FINAL_ANSWER_SYSTEM not in user
        assert "Note" in user
        assert "content" in user

    def test_run_retrieval_uses_system_chat_fn(self, tmp_path: Path) -> None:
        """When system_chat_fn is provided, retrieval must split system/user."""
        note = tmp_path / "n.md"
        note.write_text("body", encoding="utf-8")

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

        system_calls: list[tuple[str, str]] = []
        plain_calls: list[str] = []
        responses = iter([
            "SELECT: n.md",
            "SUFFICIENT: ok",
            "최종 답변",
        ])

        def system_chat(system: str, user: str) -> str:
            system_calls.append((system, user))
            return next(responses)

        def plain_chat(prompt: str) -> str:
            plain_calls.append(prompt)
            return next(responses)

        result = run_retrieval(
            "query",
            root,
            plain_chat,
            system_chat_fn=system_chat,
        )

        assert result.answer == "최종 답변"
        # All three phases should have gone through system_chat_fn.
        assert len(system_calls) == 3
        # plain_chat must NOT have been called.
        assert plain_calls == []
        # Systems should match the three stable constants.
        systems_used = {s for s, _ in system_calls}
        assert SELECT_NODE_SYSTEM in systems_used
        assert EVALUATE_SYSTEM in systems_used
        assert FINAL_ANSWER_SYSTEM in systems_used


# ─────────────────────────────────────────────────────────────────────────────
# Web UI HTML
# ─────────────────────────────────────────────────────────────────────────────


class TestWebUI:
    def test_html_is_self_contained(self) -> None:
        html = build_ui_html()
        assert html.startswith("<!doctype html>")
        # No external script/style references.
        assert "<script src=" not in html
        assert "<link rel=\"stylesheet\"" not in html
        # Embedded inline style + script tags present.
        assert "<style>" in html
        assert "<script>" in html

    def test_html_calls_ask_stream(self) -> None:
        html = build_ui_html()
        assert "/ask/stream" in html
        # Cancel + decompose wiring.
        assert "AbortController" in html
        assert "decompose" in html

    def test_title_substitution(self) -> None:
        html = build_ui_html(title="CustomApp")
        assert "CustomApp" in html
        assert "__TITLE__" not in html


# ─────────────────────────────────────────────────────────────────────────────
# Server: /usage/history + / (web UI) + budget split integration
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


@pytest.fixture
def server_with_db(tmp_path: Path):
    vault = tmp_path / "vault"
    (vault / "Notes").mkdir(parents=True)
    (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

    tracker = UsageTracker()
    db = tmp_path / "usage.db"
    store = UsageStore(db)

    responses_idx = {"i": 0}

    def chat_fn(prompt: str) -> str:
        script = ["요약", "SELECT: Notes/note.md", "SUFFICIENT: ok", "답변"]
        i = min(responses_idx["i"], len(script) - 1)
        responses_idx["i"] += 1
        tracker.record("other", 100, 10, 0.1)
        store.record("other", 100, 10, 0.1)
        return script[i]

    state = build_initial_state(
        [vault],
        folder="Notes",
        model="test",
        num_ctx=8192,
        max_workers=1,
        chat_fn=chat_fn,
    )
    state.tracker = tracker
    state.usage_store = store
    app = create_app(state)
    return TestClient(app), state


class TestUsageHistoryEndpoint:
    def test_history_returns_events(self, server_with_db) -> None:
        tc, state = server_with_db

        # Populate the store with some historical events.
        state.usage_store.record("select", 500, 10, 1.0)
        state.usage_store.record("evaluate", 800, 20, 1.5)

        response = tc.get("/usage/history?limit=50")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "events" in data
        assert "daily" in data
        assert len(data["events"]) >= 2

    def test_history_with_phase_filter(self, server_with_db) -> None:
        tc, state = server_with_db

        state.usage_store.record("select", 500, 10, 1.0)
        state.usage_store.record("evaluate", 800, 20, 1.5)

        response = tc.get("/usage/history?phase=select")
        assert response.status_code == 200
        data = response.json()
        # Events should all be of the select phase.
        for e in data["events"]:
            assert e["phase"] == "select"

    def test_history_503_when_no_store(self, tmp_path: Path) -> None:
        """/usage/history should return 503 when no usage_store is attached."""
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=lambda p: "요약",
        )
        # Explicitly no usage_store.
        state.usage_store = None
        app = create_app(state)
        tc = TestClient(app)

        response = tc.get("/usage/history")
        assert response.status_code == 503

    def test_history_invalid_timestamp_400(self, server_with_db) -> None:
        tc, _ = server_with_db
        response = tc.get("/usage/history?since=not-a-date")
        assert response.status_code == 400


class TestWebUIEndpoint:
    def test_root_serves_html(self, server_with_db) -> None:
        tc, _ = server_with_db
        response = tc.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        body = response.text
        assert "<!doctype html>" in body
        assert "pagewiki" in body
        assert "/ask/stream" in body
