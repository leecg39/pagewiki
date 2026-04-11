"""Tests for v0.8 features: usage tracking, parse retry, BM25 pre-ranking."""

from __future__ import annotations

import pytest

from pagewiki.prompts import build_retry_prompt, parse_select_response
from pagewiki.retrieval import run_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage import (
    UsageTracker,
    make_tracking_chat_fn,
    make_tracking_str_chat_fn,
)

# ─────────────────────────────────────────────────────────────────────────────
# Usage tracking
# ─────────────────────────────────────────────────────────────────────────────


class TestUsageTracker:
    def test_empty_tracker(self) -> None:
        t = UsageTracker()
        assert t.total_calls == 0
        assert t.total_tokens == 0
        assert t.by_phase() == {}

    def test_record_and_aggregate(self) -> None:
        t = UsageTracker()
        t.record("summarize", 100, 20, 0.5)
        t.record("summarize", 200, 30, 0.8)
        t.record("select", 500, 10, 1.2)

        assert t.total_calls == 3
        assert t.total_prompt_tokens == 800
        assert t.total_completion_tokens == 60
        assert t.total_tokens == 860
        assert abs(t.total_elapsed - 2.5) < 1e-6

        by_phase = t.by_phase()
        assert by_phase["summarize"]["calls"] == 2
        assert by_phase["summarize"]["prompt"] == 300
        assert by_phase["select"]["calls"] == 1
        assert by_phase["select"]["prompt"] == 500

    def test_make_tracking_chat_fn_with_llm_response_like(self) -> None:
        t = UsageTracker()

        class FakeResponse:
            text = "answer"
            prompt_tokens = 150
            completion_tokens = 40

        def fake_inner(prompt: str):
            return FakeResponse()

        wrapped = make_tracking_chat_fn(fake_inner, t, phase="select")
        result = wrapped("test prompt")
        assert result == "answer"
        assert t.total_calls == 1
        assert t.total_prompt_tokens == 150
        assert t.total_completion_tokens == 40

    def test_make_tracking_str_chat_fn_estimates_tokens(self) -> None:
        t = UsageTracker()

        def fake_inner(prompt: str) -> str:
            return "a" * 90

        wrapped = make_tracking_str_chat_fn(fake_inner, t, phase="other")
        wrapped("x" * 30)
        assert t.total_calls == 1
        assert t.total_prompt_tokens == 30 // 3  # char/3 estimate
        assert t.total_completion_tokens == 90 // 3

    def test_thread_safety(self) -> None:
        """UsageTracker lock should serialize concurrent record() calls."""
        import threading

        t = UsageTracker()
        n_threads = 10
        n_per_thread = 50

        def worker():
            for _ in range(n_per_thread):
                t.record("parallel", 10, 5, 0.01)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert t.total_calls == n_threads * n_per_thread
        assert t.total_prompt_tokens == n_threads * n_per_thread * 10


# ─────────────────────────────────────────────────────────────────────────────
# Parse retry
# ─────────────────────────────────────────────────────────────────────────────


class TestParseRetry:
    def test_build_retry_prompt_appends_reminder(self) -> None:
        retry = build_retry_prompt("original prompt", "bad format")
        assert "original prompt" in retry
        assert "재시도" in retry
        assert "bad format" in retry

    def test_retrieval_recovers_from_parse_failure(self, tmp_path) -> None:
        """A malformed first SELECT response should trigger retry, not abort."""
        note_path = tmp_path / "note.md"
        note_path.write_text("content", encoding="utf-8")

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
                    file_path=note_path,
                    summary="test",
                ),
            ],
        )

        responses = iter([
            "쓰레기 응답 — SELECT나 DONE 없음",  # malformed
            "SELECT: note.md",                    # retry succeeds
            "SUFFICIENT: ok",
            "최종 답변",
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        result = run_retrieval("query", root, fake_chat)
        assert result.answer == "최종 답변"
        # Retry trace step should be present.
        phases = [t.detail for t in result.trace]
        assert any("parse error (retrying)" in d for d in phases)
        assert any("retry succeeded" in d for d in phases)

    def test_retrieval_aborts_if_both_parse_attempts_fail(self, tmp_path) -> None:
        """If retry also fails, the loop should cleanly abort with a reason."""
        note_path = tmp_path / "note.md"
        note_path.write_text("content", encoding="utf-8")

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
                    file_path=note_path,
                    summary="test",
                ),
            ],
        )

        responses = iter([
            "쓰레기 응답 1",  # malformed
            "쓰레기 응답 2",  # retry also malformed
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        result = run_retrieval("query", root, fake_chat)
        # Loop aborts — no answer from notes, falls through to "근거 부족".
        assert "근거 부족" in result.answer
        phases = [t.detail for t in result.trace]
        assert any("retry failed" in d for d in phases)

    def test_parse_select_response_unchanged(self) -> None:
        """Sanity check: existing parser still works."""
        assert parse_select_response("SELECT: foo.md") == ("SELECT", "foo.md")
        assert parse_select_response("DONE: no match") == ("DONE", "no match")
        with pytest.raises(ValueError):
            parse_select_response("garbage")


# ─────────────────────────────────────────────────────────────────────────────
# BM25-style candidate ranking
# ─────────────────────────────────────────────────────────────────────────────


class TestRanking:
    def test_tokenize_basic(self) -> None:
        from pagewiki.ranking import tokenize
        assert tokenize("Hello, World!") == ["hello", "world"]
        assert tokenize("a b c") == []  # all shorter than min_len=2
        assert tokenize("") == []

    def test_tokenize_korean(self) -> None:
        from pagewiki.ranking import tokenize
        tokens = tokenize("어텐션 메커니즘과 transformer")
        assert "어텐션" in tokens
        assert "메커니즘과" in tokens
        assert "transformer" in tokens

    def test_score_rewards_term_overlap(self) -> None:
        from pagewiki.ranking import score_candidate, tokenize
        q = tokenize("attention mechanism")
        s1 = score_candidate(q, "This note is about attention mechanisms")
        s2 = score_candidate(q, "Something completely unrelated")
        assert s1 > s2
        assert s2 == 0.0

    def test_rank_candidates_orders_by_relevance(self) -> None:
        from pagewiki.ranking import rank_candidates
        cands = [
            ("Boring Note", "A note about gardening tips"),
            ("Transformer Paper", "Transformer attention mechanism explained"),
            ("ML Intro", "Intro to machine learning"),
        ]
        ranked = rank_candidates("attention transformer", cands)
        # First entry should be the Transformer paper.
        top_idx = ranked[0][0]
        assert cands[top_idx][0] == "Transformer Paper"

    def test_rank_stable_on_zero_scores(self) -> None:
        """When nothing matches, original order should be preserved."""
        from pagewiki.ranking import rank_candidates
        cands = [
            ("A", "alpha content"),
            ("B", "beta content"),
            ("C", "gamma content"),
        ]
        ranked = rank_candidates("xyz unrelated", cands)
        # All scores are 0; stable sort keeps original indices.
        indices = [i for i, _ in ranked]
        assert indices == [0, 1, 2]

    def test_retrieval_uses_ranking(self, tmp_path) -> None:
        """Retrieval should reorder candidates so the best match appears first."""
        note_a = tmp_path / "alpha.md"
        note_a.write_text("# Alpha\n\nUnrelated content.", encoding="utf-8")
        note_b = tmp_path / "beta.md"
        note_b.write_text(
            "# Beta\n\nTransformer attention is key.", encoding="utf-8",
        )

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="alpha.md",
                    title="Gardening",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note_a,
                    summary="Tips for gardening",
                ),
                TreeNode(
                    node_id="beta.md",
                    title="Transformer",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note_b,
                    summary="Attention mechanism explained",
                ),
            ],
        )

        observed_prompts: list[str] = []
        responses = iter([
            "SELECT: beta.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake_chat(prompt: str) -> str:
            observed_prompts.append(prompt)
            return next(responses)

        run_retrieval("transformer attention", root, fake_chat)

        # The SELECT prompt should list Transformer BEFORE Gardening.
        select_prompt = observed_prompts[0]
        transformer_idx = select_prompt.find("Transformer")
        gardening_idx = select_prompt.find("Gardening")
        assert transformer_idx != -1
        assert gardening_idx != -1
        assert transformer_idx < gardening_idx
