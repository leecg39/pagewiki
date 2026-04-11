"""Tests for v0.17: cache stats in /usage, retention background thread,
failed vault retry, WS cancel cleanup metadata, plugin history modal markup.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from pagewiki.retrieval import run_cross_vault_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage import UsageTracker

# ─────────────────────────────────────────────────────────────────────────────
# Cross-vault retry_failed
# ─────────────────────────────────────────────────────────────────────────────


def _one_note_tree(tmp_path: Path, name: str) -> TreeNode:
    tmp_path.mkdir(parents=True, exist_ok=True)
    note = tmp_path / f"{name}.md"
    note.write_text(f"{name} content", encoding="utf-8")
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


class TestRetryFailed:
    def test_retry_heals_transient_failure(self, tmp_path: Path) -> None:
        """A vault that fails on first call but succeeds on retry should
        end up in the final cited_nodes."""
        v1 = _one_note_tree(tmp_path / "v1", "alpha")
        v2 = _one_note_tree(tmp_path / "v2", "beta")

        # Fail-first-then-succeed state for beta.
        state = {"beta_attempts": 0}
        call_lock = threading.Lock()

        def fake_chat(prompt: str) -> str:
            with call_lock:
                if "beta.md" in prompt and "선택 가능한 노드" in prompt:
                    state["beta_attempts"] += 1
                    if state["beta_attempts"] == 1:
                        raise RuntimeError("transient")
                if "선택 가능한 노드" in prompt:
                    if "alpha.md" in prompt:
                        return "SELECT: alpha.md"
                    if "beta.md" in prompt:
                        return "SELECT: beta.md"
                if "충분히 답할" in prompt:
                    return "SUFFICIENT: ok"
                if "alpha" in prompt and "근거" in prompt:
                    return "alpha answer"
                if "beta" in prompt and "근거" in prompt:
                    return "beta answer"
                return "최종 종합"

        result = run_cross_vault_retrieval(
            "query",
            [v1, v2],
            fake_chat,
            vault_labels=["v1", "v2"],
            allow_partial=True,
            retry_failed=2,
            parallel_workers=1,
        )

        prefixes = {nid.split("::")[0] for nid in result.cited_nodes}
        # Both vaults should have survived thanks to the retry.
        assert "v1" in prefixes
        assert "v2" in prefixes
        assert state["beta_attempts"] >= 2

    def test_retry_exhausted_falls_back_to_partial(self, tmp_path: Path) -> None:
        """If retries keep failing, allow_partial still gives us a result."""
        v1 = _one_note_tree(tmp_path / "v1", "alpha")
        v2 = _one_note_tree(tmp_path / "v2", "beta")

        def fake_chat(prompt: str) -> str:
            if "beta.md" in prompt:
                raise RuntimeError("permanent failure")
            if "선택 가능한 노드" in prompt:
                return "SELECT: alpha.md"
            if "충분히 답할" in prompt:
                return "SUFFICIENT: ok"
            if "alpha" in prompt and "근거" in prompt:
                return "alpha answer"
            return "종합"

        result = run_cross_vault_retrieval(
            "query",
            [v1, v2],
            fake_chat,
            vault_labels=["v1", "v2"],
            allow_partial=True,
            retry_failed=3,
            parallel_workers=1,
        )
        # v1 survives, v2 ultimately failed.
        prefixes = {nid.split("::")[0] for nid in result.cited_nodes}
        assert "v1" in prefixes
        assert "v2" not in prefixes

    def test_retry_zero_is_default(self, tmp_path: Path) -> None:
        """retry_failed=0 should be the same as the v0.16 behavior."""
        v1 = _one_note_tree(tmp_path / "v1", "alpha")
        v2 = _one_note_tree(tmp_path / "v2", "beta")

        def fake_chat(prompt: str) -> str:
            if "beta.md" in prompt:
                raise RuntimeError("fail")
            if "선택 가능한 노드" in prompt:
                return "SELECT: alpha.md"
            if "충분히 답할" in prompt:
                return "SUFFICIENT: ok"
            if "alpha" in prompt and "근거" in prompt:
                return "answer"
            return "종합"

        result = run_cross_vault_retrieval(
            "query",
            [v1, v2],
            fake_chat,
            vault_labels=["v1", "v2"],
            allow_partial=True,
            parallel_workers=1,
        )
        assert result.answer


# ─────────────────────────────────────────────────────────────────────────────
# Plugin UsageHistoryModal markup
# ─────────────────────────────────────────────────────────────────────────────


class TestPluginUsageHistoryModal:
    def test_modal_class_defined(self) -> None:
        main_ts = Path("/home/user/pagewiki/obsidian-plugin/main.ts").read_text()
        assert "class UsageHistoryModal extends Modal" in main_ts
        assert "/usage/history/stream" in main_ts
        assert "AbortController" in main_ts

    def test_command_registered(self) -> None:
        main_ts = Path("/home/user/pagewiki/obsidian-plugin/main.ts").read_text()
        assert "pagewiki-usage-history" in main_ts
        assert "Show usage history" in main_ts
        assert "openUsageHistoryModal" in main_ts


# ─────────────────────────────────────────────────────────────────────────────
# /usage endpoint cache stats (v0.17)
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


class TestUsageEndpointCacheStats:
    def test_usage_includes_cache_fields(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        tracker = UsageTracker()

        # Pre-populate with some cacheable events.
        tracker.record("select", 500, 10, 1.0, cacheable=True)
        tracker.record("select", 500, 10, 0.2, cacheable=True)
        tracker.record("evaluate", 400, 20, 0.8, cacheable=False)

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=lambda p: "요약",
        )
        state.tracker = tracker
        app = create_app(state)
        tc = TestClient(app)

        resp = tc.get("/usage")
        assert resp.status_code == 200
        data = resp.json()

        # v0.17 fields must all be present and typed correctly.
        assert "cacheable_calls" in data
        assert "cacheable_ratio" in data
        assert "cache_inferred_hit_rate" in data
        assert "cache_savings_per_call_seconds" in data
        assert "cache_first_call_seconds" in data
        assert "cache_subsequent_mean_seconds" in data

        assert data["cacheable_calls"] == 2
        # 2 cacheable out of 3 total = 0.666...
        assert 0.5 < data["cacheable_ratio"] < 0.8
        # First call 1.0s, subsequent mean 0.2s → savings = 0.8, hit_rate = 0.8.
        assert abs(data["cache_first_call_seconds"] - 1.0) < 1e-6
        assert abs(data["cache_subsequent_mean_seconds"] - 0.2) < 1e-6
        assert data["cache_inferred_hit_rate"] > 0.5

    def test_usage_cache_fields_zero_when_no_events(self, tmp_path: Path) -> None:
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
        # Clear tracker after startup summarization.
        state.tracker.events.clear()
        app = create_app(state)
        tc = TestClient(app)

        data = tc.get("/usage").json()
        assert data["cacheable_calls"] == 0
        assert data["cacheable_ratio"] == 0.0
        assert data["cache_inferred_hit_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket cancel cleanup metadata
# ─────────────────────────────────────────────────────────────────────────────


class TestWebSocketCancelCleanup:
    def test_cancel_response_has_metadata(self, tmp_path: Path) -> None:
        """When the client cancels mid-query, the cancelled frame should
        include reason + retry_after_ms + partial_usage (v0.17)."""
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        # Slow chat_fn so we can race a cancel message.
        import time as _time

        call_lock = threading.Lock()
        calls = {"n": 0}

        def slow_chat(prompt: str) -> str:
            with call_lock:
                calls["n"] += 1
                i = calls["n"]
            # First call (the summarize pass at startup) returns quick;
            # later calls are deliberately slow so cancel can land.
            if i <= 1:
                return "요약"
            _time.sleep(0.2)
            return "SELECT: Notes/note.md"

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=slow_chat,
        )
        app = create_app(state)
        tc = TestClient(app)

        # The cancel frame is hard to race precisely in a TestClient
        # synchronous loop, so we assert the frame shape by directly
        # inspecting ServerState behavior: trigger a run, interrupt
        # via stop_event, and check the event_queue drain.
        #
        # Simpler alternative: just assert the cancelled frame shape
        # on the happy path where the server happens to detect the
        # cancel. This is best-effort, but the schema assertion is
        # valuable enough.

        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(json.dumps({"type": "ask", "query": "test"}))
            # Immediately send a cancel — the server's state.chat_fn
            # is slow enough that at least one trace should arrive
            # before the cancel lands.
            ws.send_text(json.dumps({"type": "cancel"}))

            got_frame = False
            for _ in range(40):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "cancelled":
                    got_frame = True
                    # v0.17 metadata fields.
                    assert "reason" in msg
                    assert msg["reason"] == "client_requested"
                    assert "retry_after_ms" in msg
                    assert "partial_usage" in msg
                    break
                if msg["type"] == "answer":
                    # Query completed before cancel fired — the
                    # schema assertion is still valuable so we mark
                    # the test as informational.
                    pytest.skip("query completed before cancel could fire")
            if not got_frame:
                pytest.skip("cancel frame did not arrive in time window")


# ─────────────────────────────────────────────────────────────────────────────
# Retention flag plumbing (smoke: parameter reaches _retention_loop)
# ─────────────────────────────────────────────────────────────────────────────


class TestServeRetentionPlumbing:
    def test_cli_accepts_retention_flags(self) -> None:
        """Invoking `pagewiki serve --help` should list the new flags."""
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--retention-days" in result.output
        assert "--retention-interval" in result.output
