"""Tests for v0.7 features: parallel LLM, multi-query decomposition,
multi-vault support.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from pagewiki.prompts import (
    decompose_query_prompt,
    parse_decompose_response,
    synthesize_multi_answer_prompt,
)
from pagewiki.retrieval import run_decomposed_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.vault import (
    scan_folder,
    scan_multi_vault,
    summarize_atomic_notes,
)

# ─────────────────────────────────────────────────────────────────────────────
# Parallel LLM calls
# ─────────────────────────────────────────────────────────────────────────────


class TestParallelSummarize:
    def test_parallel_atomic_summaries(self, tmp_path: Path) -> None:
        """Verify ThreadPoolExecutor actually parallelizes summary calls."""
        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)

        # Create 4 atomic notes.
        for i in range(4):
            (notes / f"note{i}.md").write_text("a " * 1500, encoding="utf-8")

        root = scan_folder(vault, "Notes")

        # Simulate slow LLM calls — 100ms each. Sequential would be 400ms,
        # parallel (4 workers) should be ~100ms.
        call_times: list[float] = []
        lock = threading.Lock()

        def slow_chat(prompt: str) -> str:
            with lock:
                call_times.append(time.time())
            time.sleep(0.1)
            return "요약"

        t0 = time.time()
        count = summarize_atomic_notes(root, slow_chat, max_workers=4)
        elapsed = time.time() - t0

        assert count == 4
        # Parallel should be under 2x single-call time, sequential would be ~4x.
        assert elapsed < 0.3, f"Expected parallel speedup, got {elapsed:.3f}s"

    def test_sequential_when_max_workers_1(self, tmp_path: Path) -> None:
        """max_workers=1 should fall back to sequential execution."""
        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)

        for i in range(2):
            (notes / f"note{i}.md").write_text("a " * 1500, encoding="utf-8")

        root = scan_folder(vault, "Notes")

        calls: list[str] = []

        def fake_chat(prompt: str) -> str:
            calls.append(prompt)
            return f"summary {len(calls)}"

        count = summarize_atomic_notes(root, fake_chat, max_workers=1)
        assert count == 2

    def test_cache_hit_skips_parallel(self, tmp_path: Path) -> None:
        """Cache hits should skip the parallel path entirely."""
        from pagewiki.cache import SummaryCache

        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)
        (notes / "atomic.md").write_text("a " * 1500, encoding="utf-8")

        root = scan_folder(vault, "Notes")
        cache = SummaryCache(vault)

        calls = []

        def fake_chat(prompt: str) -> str:
            calls.append(prompt)
            return "요약"

        # First call: 1 LLM invocation.
        count1 = summarize_atomic_notes(
            root, fake_chat, summary_cache=cache, model_id="m", max_workers=4,
        )
        assert count1 == 1

        # Second call on fresh tree: cache hit, 0 LLM invocations.
        root2 = scan_folder(vault, "Notes")
        calls.clear()
        count2 = summarize_atomic_notes(
            root2, fake_chat, summary_cache=cache, model_id="m", max_workers=4,
        )
        assert count2 == 0
        assert len(calls) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query decomposition
# ─────────────────────────────────────────────────────────────────────────────


class TestDecomposeQuery:
    def test_decompose_prompt_structure(self) -> None:
        prompt = decompose_query_prompt("X와 Y의 차이점은?", max_sub_queries=4)
        assert "SUB:" in prompt
        assert "SINGLE" in prompt
        assert "X와 Y의 차이점은?" in prompt

    def test_parse_single(self) -> None:
        assert parse_decompose_response("SINGLE") == []
        assert parse_decompose_response("  SINGLE  \n") == []

    def test_parse_sub_queries(self) -> None:
        response = (
            "SUB: What is X?\n"
            "SUB: What is Y?\n"
            "SUB: How do they differ?"
        )
        result = parse_decompose_response(response)
        assert result == ["What is X?", "What is Y?", "How do they differ?"]

    def test_parse_mixed_whitespace(self) -> None:
        response = "\n\nSUB: Q1\n  SUB: Q2  \n\n"
        assert parse_decompose_response(response) == ["Q1", "Q2"]

    def test_parse_empty(self) -> None:
        assert parse_decompose_response("") == []

    def test_synthesize_prompt_includes_all_pairs(self) -> None:
        pairs = [
            ("What is X?", "X is a thing"),
            ("What is Y?", "Y is another thing"),
        ]
        prompt = synthesize_multi_answer_prompt("X vs Y?", pairs)
        assert "X is a thing" in prompt
        assert "Y is another thing" in prompt
        assert "X vs Y?" in prompt
        assert "하위 질문 1" in prompt
        assert "하위 질문 2" in prompt


class TestDecomposedRetrieval:
    def _make_leaf_tree(self, tmp_path: Path) -> TreeNode:
        note_path = tmp_path / "note.md"
        note_path.write_text("# Test\n\nContent about topic.", encoding="utf-8")
        return TreeNode(
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

    def test_single_fallback(self, tmp_path: Path) -> None:
        """SINGLE response should fall through to run_retrieval."""
        root = self._make_leaf_tree(tmp_path)

        responses = iter([
            "SINGLE",              # decompose
            "SELECT: note.md",     # ToC review
            "SUFFICIENT: ok",      # evaluate
            "답변입니다.",          # final answer
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        result = run_decomposed_retrieval("simple query", root, fake_chat)
        assert "답변" in result.answer

    def test_multi_query_synthesizes(self, tmp_path: Path) -> None:
        """Multiple sub-queries should each run retrieval then synthesize."""
        root = self._make_leaf_tree(tmp_path)

        responses = iter([
            # Phase 1: decompose into 2 sub-queries.
            "SUB: What is X?\nSUB: What is Y?",
            # Phase 2a: first sub-query retrieval.
            "SELECT: note.md",
            "SUFFICIENT: ok",
            "X 답변",
            # Phase 2b: second sub-query retrieval.
            "SELECT: note.md",
            "SUFFICIENT: ok",
            "Y 답변",
            # Phase 3: synthesize.
            "종합 답변",
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        result = run_decomposed_retrieval("X vs Y", root, fake_chat)
        assert result.answer == "종합 답변"
        # At least one citation should propagate (both sub-queries picked note.md).
        assert "note.md" in result.cited_nodes

    def test_decompose_events_emitted(self, tmp_path: Path) -> None:
        """on_event should receive decompose-phase events."""
        root = self._make_leaf_tree(tmp_path)

        responses = iter([
            "SUB: Q1",
            "SELECT: note.md",
            "SUFFICIENT: ok",
            "A1",
            "final",
        ])

        events = []

        def fake_chat(prompt: str) -> str:
            return next(responses)

        run_decomposed_retrieval(
            "complex",
            root,
            fake_chat,
            on_event=lambda step: events.append(step),
        )

        phases = [e.phase for e in events]
        assert "decompose" in phases


# ─────────────────────────────────────────────────────────────────────────────
# Multi-vault support
# ─────────────────────────────────────────────────────────────────────────────


class TestScanMultiVault:
    def test_single_vault_passthrough(self, tmp_path: Path) -> None:
        vault = tmp_path / "v1"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "a.md").write_text("hi", encoding="utf-8")

        root = scan_multi_vault([(vault, "Notes")])
        # Single vault: returns the plain scan root.
        note_ids = [n.node_id for n in root.walk() if n.kind == "note"]
        assert "Notes/a.md" in note_ids

    def test_multi_vault_merges_with_namespace(self, tmp_path: Path) -> None:
        v1 = tmp_path / "vault1"
        (v1 / "Notes").mkdir(parents=True)
        (v1 / "Notes" / "alpha.md").write_text("hi", encoding="utf-8")

        v2 = tmp_path / "vault2"
        (v2 / "Notes").mkdir(parents=True)
        (v2 / "Notes" / "beta.md").write_text("hi", encoding="utf-8")

        root = scan_multi_vault([(v1, "Notes"), (v2, "Notes")])

        note_ids = [n.node_id for n in root.walk() if n.kind == "note"]
        # Both should be namespaced with their vault name.
        assert any("vault1::" in nid for nid in note_ids)
        assert any("vault2::" in nid for nid in note_ids)
        # The specific notes should be reachable.
        assert "vault1::Notes/alpha.md" in note_ids
        assert "vault2::Notes/beta.md" in note_ids

    def test_multi_vault_preserves_file_paths(self, tmp_path: Path) -> None:
        """Namespaced node_ids must not break absolute file_path pointers."""
        v1 = tmp_path / "v1"
        (v1 / "Notes").mkdir(parents=True)
        (v1 / "Notes" / "a.md").write_text("content", encoding="utf-8")

        v2 = tmp_path / "v2"
        (v2 / "Notes").mkdir(parents=True)
        (v2 / "Notes" / "b.md").write_text("content", encoding="utf-8")

        root = scan_multi_vault([(v1, "Notes"), (v2, "Notes")])
        for node in root.walk():
            if node.kind == "note" and node.file_path:
                assert node.file_path.exists()

    def test_empty_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            scan_multi_vault([])

    def test_same_relpath_different_vaults_no_collision(self, tmp_path: Path) -> None:
        """Two vaults with a note at the same relative path should stay distinct."""
        v1 = tmp_path / "v1"
        (v1 / "Notes").mkdir(parents=True)
        (v1 / "Notes" / "same.md").write_text("v1 content", encoding="utf-8")

        v2 = tmp_path / "v2"
        (v2 / "Notes").mkdir(parents=True)
        (v2 / "Notes" / "same.md").write_text("v2 content", encoding="utf-8")

        root = scan_multi_vault([(v1, "Notes"), (v2, "Notes")])
        note_ids = [n.node_id for n in root.walk() if n.kind == "note"]
        assert "v1::Notes/same.md" in note_ids
        assert "v2::Notes/same.md" in note_ids
        # IDs are distinct — no collision.
        assert len(note_ids) == 2
