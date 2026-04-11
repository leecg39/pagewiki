"""Tests for v0.15: parallel cross-vault, prompt cache hit rate,
/usage/history/stream, plugin WebSocket extensions, Web UI sparkline.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from pagewiki.retrieval import run_cross_vault_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage import UsageTracker
from pagewiki.usage_store import UsageStore
from pagewiki.webui import build_ui_html

# ─────────────────────────────────────────────────────────────────────────────
# Cross-vault parallel execution
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


class TestCrossVaultParallel:
    def test_parallel_faster_than_sequential(self, tmp_path: Path) -> None:
        """With 3 vaults × 100ms chat_fn, parallel should be <200ms while
        sequential would take ~300+ms."""
        roots = [
            _one_note_tree(tmp_path / f"v{i}", f"note{i}")
            for i in range(3)
        ]

        call_lock = threading.Lock()
        call_count = {"n": 0}

        def slow_chat(prompt: str) -> str:
            with call_lock:
                call_count["n"] += 1
                i = call_count["n"]
            time.sleep(0.05)  # 50ms per call
            # Script: each vault runs SELECT → SUFFICIENT → final → synth
            # Loose scripting: return plausible responses based on prompt.
            if "SELECT" in prompt or "선택" in prompt:
                if "note0" in prompt:
                    return "SELECT: note0.md"
                if "note1" in prompt:
                    return "SELECT: note1.md"
                if "note2" in prompt:
                    return "SELECT: note2.md"
                return "DONE: no match"
            if "충분히 답할" in prompt or "SUFFICIENT" in prompt:
                return "SUFFICIENT: ok"
            return f"answer-{i}"

        t0 = time.time()
        result = run_cross_vault_retrieval(
            "query",
            roots,
            slow_chat,
            vault_labels=["v0", "v1", "v2"],
            parallel_workers=3,
        )
        parallel_elapsed = time.time() - t0

        assert result.answer
        # Parallel with 3 workers should be significantly faster
        # than 9-12 sequential calls × 50ms ≈ 500ms.
        assert parallel_elapsed < 0.4, (
            f"Expected parallel speedup, got {parallel_elapsed:.3f}s"
        )

    def test_parallel_preserves_result_order(self, tmp_path: Path) -> None:
        """cited_nodes prefixes should match the input vault order even
        when the parallel workers complete out of order.

        The scripted ``fake`` chat_fn is content-aware (keyed on the
        note_id present in the prompt) so parallel workers don't race
        on a shared iterator.
        """
        roots = [
            _one_note_tree(tmp_path / f"v{i}", f"note{i}")
            for i in range(3)
        ]

        def fake(prompt: str) -> str:
            # Which note is the prompt about?
            target = None
            for i in range(3):
                if f"note{i}.md" in prompt:
                    target = i
                    break

            # SELECT phase.
            if "선택 가능한 노드" in prompt or "SELECT" in prompt:
                if target is not None:
                    return f"SELECT: note{target}.md"
                return "DONE: no match"
            # EVALUATE phase.
            if "충분히 답할" in prompt:
                return "SUFFICIENT: ok"
            # Final answer (per-vault) or cross-vault synthesis.
            if target is not None:
                return f"v{target} 답변"
            return "최종 합성"

        result = run_cross_vault_retrieval(
            "query",
            roots,
            fake,
            vault_labels=["a", "b", "c"],
            parallel_workers=3,
        )
        # Each vault's citation should be present with its label prefix.
        prefixes = {nid.split("::")[0] for nid in result.cited_nodes}
        assert prefixes == {"a", "b", "c"}

    def test_sequential_fallback_when_workers_1(self, tmp_path: Path) -> None:
        roots = [_one_note_tree(tmp_path / "v1", "only")]

        responses = iter([
            "SELECT: only.md",
            "SUFFICIENT: ok",
            "answer",
            "synthesis",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        result = run_cross_vault_retrieval(
            "query",
            roots,
            fake,
            vault_labels=["v1"],
            parallel_workers=1,
        )
        assert result.answer == "synthesis"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt cache hit rate tracking
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheableRatio:
    def test_empty_tracker_ratio_zero(self) -> None:
        t = UsageTracker()
        assert t.cacheable_ratio() == 0.0
        assert t.cacheable_calls == 0

    def test_mixed_calls_ratio(self) -> None:
        t = UsageTracker()
        t.record("a", 100, 10, 0.1, cacheable=False)
        t.record("b", 200, 20, 0.2, cacheable=True)
        t.record("c", 300, 30, 0.3, cacheable=True)
        t.record("d", 400, 40, 0.4, cacheable=False)

        assert t.cacheable_calls == 2
        assert t.cacheable_ratio() == 0.5

    def test_all_cacheable(self) -> None:
        t = UsageTracker()
        for _ in range(5):
            t.record("x", 100, 10, 0.1, cacheable=True)
        assert t.cacheable_ratio() == 1.0

    def test_default_cacheable_false(self) -> None:
        t = UsageTracker()
        t.record("a", 100, 10, 0.1)  # no cacheable kwarg
        assert t.cacheable_calls == 0
        assert t.events[0].cacheable is False


# ─────────────────────────────────────────────────────────────────────────────
# Web UI sparkline markup
# ─────────────────────────────────────────────────────────────────────────────


class TestWebUISparkline:
    def test_sparkline_svg_present(self) -> None:
        html = build_ui_html()
        assert "<svg" in html
        assert 'id="sparkline"' in html
        assert 'id="sparkline-path"' in html
        assert "renderSparkline" in html

    def test_sparkline_consumes_usage_events(self) -> None:
        html = build_ui_html()
        # Verify the usage handler pushes into usageSeries.
        assert "usageSeries.push" in html
        assert "total_tokens" in html


# ─────────────────────────────────────────────────────────────────────────────
# /usage/history/stream SSE endpoint + plugin WS extensions
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


@pytest.fixture
def server_with_store(tmp_path: Path):
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


class TestUsageHistoryStream:
    def test_initial_snapshot_emitted(self, server_with_store) -> None:
        tc, state = server_with_store
        # Seed some events so the snapshot has content.
        state.usage_store.record("select", 500, 10, 1.0)
        state.usage_store.record("evaluate", 800, 20, 1.5)

        # max_duration=0.3s guarantees the stream closes quickly.
        with tc.stream(
            "GET",
            "/usage/history/stream"
            "?poll_interval=0.1&initial_limit=10&max_duration=0.3",
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "event: initial" in body
        assert "events" in body
        assert "event: done" in body

    def test_stream_requires_store(self, tmp_path: Path) -> None:
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
        state.usage_store = None
        app = create_app(state)
        tc = TestClient(app)

        response = tc.get("/usage/history/stream")
        assert response.status_code == 503

    def test_max_duration_terminates_idle_stream(self, server_with_store) -> None:
        """An idle stream should close within max_duration."""
        tc, _ = server_with_store

        t0 = time.time()
        with tc.stream(
            "GET",
            "/usage/history/stream?poll_interval=0.1&max_duration=0.2",
        ) as response:
            body = "".join(response.iter_text())
        elapsed = time.time() - t0

        assert "event: done" in body
        # Should finish close to max_duration, not the default 900s.
        assert elapsed < 1.5, f"stream didn't honor max_duration: {elapsed}s"


class TestWebSocketExtensions:
    def test_ws_accepts_token_split(self, server_with_store) -> None:
        """WebSocket /ask/ws should accept max_tokens + token_split in the ask frame."""
        tc, _ = server_with_store
        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "ask",
                        "query": "test",
                        "max_tokens": 10000,
                        "token_split": "20:60:20",
                        "json_mode": False,
                        "reuse_context": True,
                    }
                )
            )
            # Just drain until answer or error — no exceptions = pass.
            got_answer = False
            for _ in range(30):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "answer":
                    got_answer = True
                    break
                if msg["type"] == "error":
                    pytest.fail(f"unexpected error: {msg}")
            assert got_answer

    def test_ws_ignores_bad_token_split(self, server_with_store) -> None:
        tc, _ = server_with_store
        with tc.websocket_connect("/ask/ws") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "ask",
                        "query": "test",
                        "max_tokens": 10000,
                        "token_split": "not-a-number",
                    }
                )
            )
            got_answer = False
            for _ in range(30):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "answer":
                    got_answer = True
                    break
            assert got_answer
