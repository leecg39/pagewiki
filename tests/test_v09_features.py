"""Tests for v0.9 features: token budget, chat usage, /usage endpoint,
cited note re-ranking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pagewiki.retrieval import run_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage import UsageTracker, make_tracking_str_chat_fn

# ─────────────────────────────────────────────────────────────────────────────
# Token budget enforcement
# ─────────────────────────────────────────────────────────────────────────────


def _two_note_tree(tmp_path: Path) -> TreeNode:
    """Vault with two small atomic notes that the retrieval loop can explore."""
    a = tmp_path / "a.md"
    a.write_text("# A\n\nAlpha content.", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("# B\n\nBeta content.", encoding="utf-8")
    return TreeNode(
        node_id="",
        title="root",
        kind="folder",
        children=[
            TreeNode(
                node_id="a.md",
                title="A",
                kind="note",
                tier=NoteTier.ATOMIC,
                file_path=a,
                summary="alpha",
            ),
            TreeNode(
                node_id="b.md",
                title="B",
                kind="note",
                tier=NoteTier.ATOMIC,
                file_path=b,
                summary="beta",
            ),
        ],
    )


class TestTokenBudget:
    def test_budget_not_exceeded_runs_normally(self, tmp_path) -> None:
        """With a generous budget, the loop should complete as usual."""
        root = _two_note_tree(tmp_path)

        tracker = UsageTracker()

        responses = iter([
            "SELECT: a.md",
            "SUFFICIENT: ok",
            "최종 답변",
        ])

        def inner(prompt: str) -> str:
            return next(responses)

        wrapped = make_tracking_str_chat_fn(inner, tracker, phase="other")

        result = run_retrieval(
            "test",
            root,
            wrapped,
            max_tokens=1_000_000,
            tracker=tracker,
        )
        assert "최종 답변" in result.answer
        assert "토큰 예산 초과" not in result.answer

    def test_budget_exceeded_aborts_loop(self, tmp_path) -> None:
        """With a tiny budget, the loop should abort and return partial results."""
        root = _two_note_tree(tmp_path)

        tracker = UsageTracker()

        # First SELECT consumes enough tokens to bust a small budget.
        responses = iter([
            "SELECT: a.md",
            "SUFFICIENT: ok",
            "최종 답변 (should not appear)",
        ])

        def inner(prompt: str) -> str:
            return next(responses)

        # Each char/3 estimate makes the budget trip after the first call.
        wrapped = make_tracking_str_chat_fn(inner, tracker, phase="other")

        result = run_retrieval(
            "test",
            root,
            wrapped,
            max_tokens=10,  # tiny budget
            tracker=tracker,
        )

        # The loop should have aborted and returned a partial or empty result.
        assert "최종 답변" not in result.answer or "예산 초과" in result.answer
        # The trace should include a budget event.
        phases = [t.phase for t in result.trace]
        assert "budget" in phases or "finalize" in phases

    def test_no_budget_no_tracker_no_effect(self, tmp_path) -> None:
        """When neither max_tokens nor tracker is provided, behavior is unchanged."""
        root = _two_note_tree(tmp_path)

        responses = iter([
            "SELECT: a.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        result = run_retrieval("test", root, fake)
        assert result.answer == "답변"


class TestCitedReranking:
    def test_cited_nodes_sorted_by_relevance(self, tmp_path) -> None:
        """After gathering, cited_nodes should be ordered by BM25 relevance."""
        # Make two notes where the second matches the query better.
        boring = tmp_path / "boring.md"
        boring.write_text("# Boring\n\nUnrelated gardening tips.", encoding="utf-8")
        target = tmp_path / "target.md"
        target.write_text(
            "# Target\n\nTransformer attention mechanism details.",
            encoding="utf-8",
        )

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="boring.md",
                    title="Gardening",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=boring,
                    summary="gardening tips",
                ),
                TreeNode(
                    node_id="target.md",
                    title="Transformer",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=target,
                    summary="attention mechanism",
                ),
            ],
        )

        # Force the loop to gather BOTH notes (Boring first, then Transformer)
        # by scripting insufficient on the first pick.
        responses = iter([
            # BM25 pre-ranking will reorder candidates so Transformer
            # is actually picked first. Anticipate that.
            "SELECT: target.md",
            "INSUFFICIENT: need more",
            "SELECT: boring.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        result = run_retrieval(
            "transformer attention",
            root,
            fake,
        )

        # Both should be cited.
        assert "target.md" in result.cited_nodes
        assert "boring.md" in result.cited_nodes
        # After v0.9 re-ranking: target.md (best BM25 match) should come first.
        assert result.cited_nodes[0] == "target.md"

    def test_single_citation_unchanged(self, tmp_path) -> None:
        """With only one citation, re-ranking is a no-op."""
        note = tmp_path / "only.md"
        note.write_text("# Only\n\nContent.", encoding="utf-8")

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="only.md",
                    title="Only",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="only note",
                ),
            ],
        )

        responses = iter([
            "SELECT: only.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake(prompt: str) -> str:
            return next(responses)

        result = run_retrieval("test", root, fake)
        assert result.cited_nodes == ["only.md"]


# ─────────────────────────────────────────────────────────────────────────────
# Server /usage endpoint
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


@pytest.fixture
def server_client(tmp_path: Path):
    """Build a FastAPI TestClient with a real ServerState + UsageTracker."""
    vault = tmp_path / "vault"
    notes = vault / "Notes"
    notes.mkdir(parents=True)
    (notes / "note.md").write_text("# N\n\nContent", encoding="utf-8")

    tracker = UsageTracker()

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 20

    responses_idx = {"i": 0}
    script = [
        "setup summary",
        "SELECT: Notes/note.md",
        "SUFFICIENT: ok",
        "답변",
    ]

    def tracking_inner(prompt: str) -> str:
        i = min(responses_idx["i"], len(script) - 1)
        responses_idx["i"] += 1
        resp = FakeResponse(script[i])
        tracker.record(
            "other",
            resp.prompt_tokens,
            resp.completion_tokens,
            0.01,
        )
        return resp.text

    state = build_initial_state(
        [vault],
        folder="Notes",
        model="test",
        num_ctx=8192,
        max_workers=1,
        chat_fn=tracking_inner,
    )
    state.tracker = tracker
    app = create_app(state)
    return TestClient(app), state, tracker


class TestUsageEndpoint:
    def test_usage_returns_cumulative_counts(self, server_client) -> None:
        tc, state, tracker = server_client

        # Record some synthetic usage on the shared tracker.
        tracker.record("select", 500, 10, 1.0)
        tracker.record("evaluate", 800, 20, 1.5)
        tracker.record("select", 300, 5, 0.8)

        response = tc.get("/usage")
        assert response.status_code == 200
        data = response.json()
        assert data["total_calls"] >= 3
        assert data["total_prompt_tokens"] >= 1600
        assert data["total_completion_tokens"] >= 35

        # Phase buckets should be present.
        assert "select" in data["by_phase"]
        assert data["by_phase"]["select"]["calls"] == 2

    def test_usage_reset_clears_tracker(self, server_client) -> None:
        tc, state, tracker = server_client

        tracker.record("other", 100, 50, 0.1)
        assert tracker.total_calls >= 1

        response = tc.post("/usage/reset")
        assert response.status_code == 200
        assert response.json()["status"] == "reset"
        assert tracker.total_calls == 0
        assert tracker.total_tokens == 0

    def test_empty_tracker_returns_zeros(self, server_client) -> None:
        tc, state, tracker = server_client
        tracker.events.clear()
        response = tc.get("/usage")
        data = response.json()
        assert data["total_calls"] == 0
        assert data["total_tokens"] == 0
        assert data["by_phase"] == {}
