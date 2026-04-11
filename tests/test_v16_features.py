"""Tests for v0.16: parallel-wave budget check, inferred cache latency,
Web UI history view, plugin WS prompt-cache, cross-vault allow_partial.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from pagewiki.retrieval import run_cross_vault_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage import UsageTracker
from pagewiki.vault import scan_folder, summarize_atomic_notes
from pagewiki.webui import build_ui_html

# ─────────────────────────────────────────────────────────────────────────────
# Parallel summarize: per-wave budget check
# ─────────────────────────────────────────────────────────────────────────────


class TestParallelSummarizeBudget:
    def test_budget_stops_mid_wave(self, tmp_path: Path) -> None:
        """With a tight budget, summarize_atomic_notes should stop before
        processing every note even in the parallel path."""
        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)
        for i in range(10):
            (notes / f"note{i}.md").write_text("a " * 1500, encoding="utf-8")

        root = scan_folder(vault, "Notes")
        tracker = UsageTracker()
        call_count = {"n": 0}

        def fake_chat(prompt: str) -> str:
            call_count["n"] += 1
            # Each call burns 1000 tokens.
            tracker.record("summarize", 1000, 100, 0.01)
            return f"summary {call_count['n']}"

        # Budget of 4500 tokens = ~4-5 calls before we stop.
        count = summarize_atomic_notes(
            root,
            fake_chat,
            max_workers=2,  # parallel path
            max_tokens=4500,
            tracker=tracker,
        )

        # We should stop well before all 10 notes.
        assert count <= 8
        assert call_count["n"] <= 8
        # And we should have processed more than 0.
        assert count >= 1

    def test_budget_parallel_serial_fallback(self, tmp_path: Path) -> None:
        """max_workers=1 + budget still works (serial path)."""
        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)
        for i in range(5):
            (notes / f"n{i}.md").write_text("a " * 1500, encoding="utf-8")

        root = scan_folder(vault, "Notes")
        tracker = UsageTracker()

        def fake_chat(prompt: str) -> str:
            tracker.record("summarize", 1000, 100, 0.01)
            return "summary"

        count = summarize_atomic_notes(
            root,
            fake_chat,
            max_workers=1,
            max_tokens=2500,
            tracker=tracker,
        )
        assert count <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Inferred cache latency savings
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheLatencySavings:
    def test_fewer_than_2_samples_returns_zeros(self) -> None:
        t = UsageTracker()
        t.record("a", 100, 10, 0.5, cacheable=True)
        info = t.cacheable_latency_savings()
        assert info["samples"] == 1
        assert info["first_call_seconds"] == 0.0
        assert info["subsequent_mean_seconds"] == 0.0

    def test_speedup_positive_inferred_hit_rate(self) -> None:
        t = UsageTracker()
        # First call slow (cold), rest fast (cache hits).
        t.record("a", 100, 10, 1.0, cacheable=True)
        t.record("a", 100, 10, 0.2, cacheable=True)
        t.record("a", 100, 10, 0.1, cacheable=True)

        info = t.cacheable_latency_savings()
        assert info["samples"] == 3
        assert info["first_call_seconds"] == 1.0
        assert abs(info["subsequent_mean_seconds"] - 0.15) < 1e-6
        assert info["savings_per_call_seconds"] > 0
        # inferred_hit_rate = 0.85/1.0 = 0.85.
        assert 0.8 < info["inferred_hit_rate"] <= 1.0

    def test_no_speedup_zero_hit_rate(self) -> None:
        t = UsageTracker()
        t.record("a", 100, 10, 1.0, cacheable=True)
        t.record("a", 100, 10, 1.1, cacheable=True)  # slower!
        info = t.cacheable_latency_savings()
        # Savings negative → clamped to 0 hit rate.
        assert info["inferred_hit_rate"] == 0.0
        assert info["savings_per_call_seconds"] < 0

    def test_non_cacheable_events_ignored(self) -> None:
        t = UsageTracker()
        t.record("a", 100, 10, 5.0, cacheable=False)  # ignored
        t.record("a", 100, 10, 1.0, cacheable=True)
        t.record("a", 100, 10, 0.5, cacheable=True)
        info = t.cacheable_latency_savings()
        assert info["samples"] == 2
        assert info["first_call_seconds"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Web UI historical view markup
# ─────────────────────────────────────────────────────────────────────────────


class TestWebUIHistoryView:
    def test_history_card_present(self) -> None:
        html = build_ui_html()
        assert "history-card" in html
        assert "/usage/history/stream" in html
        assert "Historical usage events" in html
        # Controls must be wired.
        assert 'id="history-start"' in html
        assert 'id="history-stop"' in html

    def test_history_handlers_defined(self) -> None:
        html = build_ui_html()
        assert "startHistoryStream" in html
        assert "stopHistoryStream" in html
        assert "handleHistoryFrame" in html
        assert "appendHistoryRow" in html

    def test_history_table_has_header(self) -> None:
        html = build_ui_html()
        assert "history-table" in html
        # Header columns.
        assert "timestamp" in html
        assert "phase" in html
        assert "prompt" in html
        assert "completion" in html


# ─────────────────────────────────────────────────────────────────────────────
# Cross-vault allow_partial
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


class TestCrossVaultAllowPartial:
    def test_partial_mode_synthesizes_from_survivors(self, tmp_path: Path) -> None:
        """allow_partial=True should produce a final answer even when one
        vault's chat_fn raises."""
        v1 = _one_note_tree(tmp_path / "v1", "alpha")
        v2 = _one_note_tree(tmp_path / "v2", "beta")
        v3 = _one_note_tree(tmp_path / "v3", "gamma")

        # v2 will intentionally blow up. Use a content-aware chat_fn
        # so the scripted responses don't race.
        call_lock = threading.Lock()

        def fake_chat(prompt: str) -> str:
            with call_lock:
                if "beta.md" in prompt:
                    raise RuntimeError("simulated v2 failure")
                # Select phase.
                if "선택 가능한 노드" in prompt:
                    if "alpha.md" in prompt:
                        return "SELECT: alpha.md"
                    if "gamma.md" in prompt:
                        return "SELECT: gamma.md"
                    return "DONE: no match"
                # Evaluate phase.
                if "충분히 답할" in prompt:
                    return "SUFFICIENT: ok"
                # Per-vault final + cross-vault synth.
                if "alpha" in prompt and "근거" in prompt:
                    return "alpha answer"
                if "gamma" in prompt and "근거" in prompt:
                    return "gamma answer"
                return "최종 종합 답변"

        result = run_cross_vault_retrieval(
            "query",
            [v1, v2, v3],
            fake_chat,
            vault_labels=["v1", "v2", "v3"],
            allow_partial=True,
            parallel_workers=1,  # deterministic order for assertions
        )

        # We should get a final synthesized answer even with v2 failing.
        assert result.answer
        # v1 and v3 citations should be present, v2 should be absent.
        prefixes = {nid.split("::")[0] for nid in result.cited_nodes}
        assert "v1" in prefixes
        assert "v3" in prefixes
        assert "v2" not in prefixes
        # The trace should record v2's failure.
        trace_detail = " ".join(
            t.detail for t in result.trace if t.phase == "cross-vault"
        )
        assert "v2" in trace_detail
        assert "실패" in trace_detail or "fail" in trace_detail.lower()

    def test_strict_mode_raises(self, tmp_path: Path) -> None:
        """allow_partial=False (default) should re-raise."""
        v1 = _one_note_tree(tmp_path / "v1", "alpha")

        def fake_chat(prompt: str) -> str:
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            run_cross_vault_retrieval(
                "query",
                [v1],
                fake_chat,
                vault_labels=["v1"],
                allow_partial=False,
                parallel_workers=1,
            )

    def test_all_failed_returns_error_answer(self, tmp_path: Path) -> None:
        """If every vault fails under allow_partial, we get an error answer,
        not an unhandled exception."""
        v1 = _one_note_tree(tmp_path / "v1", "alpha")
        v2 = _one_note_tree(tmp_path / "v2", "beta")

        def fake_chat(prompt: str) -> str:
            raise RuntimeError("all broken")

        result = run_cross_vault_retrieval(
            "query",
            [v1, v2],
            fake_chat,
            vault_labels=["v1", "v2"],
            allow_partial=True,
            parallel_workers=1,
        )
        assert "전체 실패" in result.answer
        assert "v1" in result.answer
        assert "v2" in result.answer
        assert result.cited_nodes == []


# ─────────────────────────────────────────────────────────────────────────────
# Plugin WebSocket prompt_cache toggle (server side)
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


class TestServerPromptCacheOptIn:
    def test_prompt_cache_field_accepted_by_ws(self, tmp_path: Path) -> None:
        """The /ask/ws ``ask`` frame should accept a ``prompt_cache`` field
        and route through state.system_chat_fn when provided."""
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        plain_calls: list[str] = []
        system_calls: list[tuple[str, str]] = []

        def plain_chat(prompt: str) -> str:
            plain_calls.append(prompt)
            # Summarize / cross-vault synth
            if "선택 가능한 노드" in prompt:
                return "SELECT: Notes/note.md"
            if "충분히 답할" in prompt:
                return "SUFFICIENT: ok"
            return "plain 답변"

        def system_chat(system: str, user: str) -> str:
            system_calls.append((system, user))
            if "선택 가능한 노드" in user or "node_id" in user:
                return "SELECT: Notes/note.md"
            if "충분히 답할" in user or "sufficient" in user.lower():
                return "SUFFICIENT: ok"
            return "system 답변"

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=plain_chat,
        )
        state.system_chat_fn = system_chat
        app = create_app(state)
        tc = TestClient(app)

        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "ask",
                        "query": "test",
                        "prompt_cache": True,
                    }
                )
            )
            got_answer = False
            for _ in range(30):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "answer":
                    got_answer = True
                    break
                if msg["type"] == "error":
                    pytest.fail(f"unexpected error: {msg}")
            assert got_answer

        # system_chat_fn should have been called (at least for select + eval).
        assert len(system_calls) >= 2

    def test_prompt_cache_ignored_when_server_disabled(
        self, tmp_path: Path,
    ) -> None:
        """When state.system_chat_fn is None (server started without
        --prompt-cache), a ``prompt_cache: true`` request is silently
        ignored and the regular chat_fn handles the call."""
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        calls: list[str] = []

        def chat(prompt: str) -> str:
            calls.append(prompt)
            if "선택 가능한 노드" in prompt:
                return "SELECT: Notes/note.md"
            if "충분히 답할" in prompt:
                return "SUFFICIENT: ok"
            return "답변"

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=chat,
        )
        # Explicitly no system_chat_fn.
        state.system_chat_fn = None
        app = create_app(state)
        tc = TestClient(app)

        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "ask",
                        "query": "test",
                        "prompt_cache": True,
                    }
                )
            )
            for _ in range(30):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "answer":
                    break
                if msg["type"] == "error":
                    pytest.fail(f"unexpected error: {msg}")

        # Regular chat_fn should have fielded every call.
        assert len(calls) >= 3
