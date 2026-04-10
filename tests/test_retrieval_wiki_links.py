"""Tests for v0.2 wiki-link cross-reference traversal in the retrieval loop.

When a note evaluated as INSUFFICIENT contains ``[[OtherNote]]`` links,
the retrieval loop should enqueue those targets as cross-reference
candidates so the LLM can follow the vault's knowledge graph.

All LLM calls are mocked via a scripted ChatFn. The ``LinkIndex`` is
built automatically from the vault fixtures — no mocking of the link
layer needed because the test vaults contain real wiki-links.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pagewiki.retrieval import RetrievalResult, run_retrieval
from pagewiki.vault import scan_folder
from pagewiki.wiki_links import build_link_index


def _scripted_chat(responses: list[str]) -> Callable[[str], str]:
    """Replay ``responses`` in order; extra calls return a safe DONE."""
    idx = {"i": 0}

    def _call(prompt: str) -> str:
        if idx["i"] >= len(responses):
            return "DONE: script exhausted"
        reply = responses[idx["i"]]
        idx["i"] += 1
        return reply

    return _call


@pytest.fixture
def linked_vault(tmp_path: Path) -> Path:
    """Vault with three notes where wiki-links form a chain:

        intro.md → [[revenue]] → [[report]]

    Filenames match wiki-link targets (Obsidian convention: ``[[foo]]``
    resolves to ``foo.md`` via case-insensitive title matching).
    """
    vault = tmp_path / "vault"
    research = vault / "Research"
    research.mkdir(parents=True)

    # ~600 tokens → ATOMIC, links to revenue
    (research / "intro.md").write_text(
        "# Intro\n\n이 문서는 매출 분석의 개요입니다. "
        "자세한 내용은 [[revenue]]를 참고하세요. " * 40,
        encoding="utf-8",
    )
    # ~800 tokens → ATOMIC, links to report
    (research / "revenue.md").write_text(
        "# Revenue\n\n2024년 3분기 매출은 1조 2천억원입니다. "
        "연간 추이는 [[report]]에서 확인 가능합니다. " * 50,
        encoding="utf-8",
    )
    # ~600 tokens → ATOMIC, no outgoing links
    (research / "report.md").write_text(
        "# Report\n\n2024년 연간 매출은 4조 5천억원입니다. " * 40,
        encoding="utf-8",
    )
    return vault


class TestWikiLinkTraversal:
    def test_cross_ref_candidate_offered_after_insufficient(
        self, linked_vault: Path
    ) -> None:
        """After evaluating intro.md as INSUFFICIENT, its outgoing
        ``[[Q3 Revenue]]`` link should appear as a cross-ref candidate
        that the LLM can pick in the next iteration."""
        root = scan_folder(linked_vault, "Research")

        responses = [
            "SELECT: Research/intro.md",           # iter 1: pick intro
            "INSUFFICIENT: 매출 수치 없음",          # iter 1: evaluate
            "SELECT: Research/revenue.md",      # iter 2: pick cross-ref
            "SUFFICIENT: 매출 수치 확인",            # iter 2: evaluate
            "2024년 Q3 매출은 1조 2천억원. [[Q3 Revenue]]",
        ]

        result = run_retrieval("Q3 매출 얼마?", root, _scripted_chat(responses))

        assert "Research/revenue.md" in result.cited_nodes
        assert "Research/intro.md" in result.cited_nodes
        # Verify cross-ref trace entry exists
        cross_ref_steps = [s for s in result.trace if s.phase == "cross-ref"]
        assert len(cross_ref_steps) >= 1
        assert any("revenue" in s.node_id for s in cross_ref_steps)

    def test_transitive_cross_ref_chain(self, linked_vault: Path) -> None:
        """intro → [[Q3 Revenue]] → [[Annual Report]] should be
        transitively followable across multiple iterations."""
        root = scan_folder(linked_vault, "Research")

        responses = [
            "SELECT: Research/intro.md",           # iter 1
            "INSUFFICIENT: 개요만 있음",
            "SELECT: Research/revenue.md",      # iter 2 (cross-ref from intro)
            "INSUFFICIENT: 연간 추이 필요",
            "SELECT: Research/report.md",   # iter 3 (cross-ref from q3)
            "SUFFICIENT: 연간 매출 확인",
            "종합 답변. [[Annual Report]]",
        ]

        result = run_retrieval(
            "연간 매출?", root, _scripted_chat(responses), max_iterations=5
        )

        assert len(result.cited_nodes) == 3
        assert "Research/report.md" in result.cited_nodes

    def test_already_visited_links_not_re_offered(
        self, linked_vault: Path
    ) -> None:
        """If a wiki-link target was already visited via tree traversal,
        it should not re-appear as a cross-ref candidate."""
        root = scan_folder(linked_vault, "Research")

        responses = [
            "SELECT: Research/revenue.md",      # iter 1: visit q3 first
            "INSUFFICIENT: 개요 필요",
            "SELECT: Research/intro.md",           # iter 2: visit intro
            "INSUFFICIENT: 매출 수치 없음",
            # intro links to [[Q3 Revenue]], but it's already visited.
            # Only annual_report should remain as a candidate.
            "SELECT: Research/report.md",   # iter 3
            "SUFFICIENT: ok",
            "최종 답변",
        ]

        result = run_retrieval(
            "전체 요약", root, _scripted_chat(responses), max_iterations=5
        )

        # q3_revenue should appear only once in cited_nodes
        assert result.cited_nodes.count("Research/revenue.md") == 1

    def test_cross_ref_trace_records_source(self, linked_vault: Path) -> None:
        """The trace should record which note's link caused the
        cross-ref enqueue, for audit trail purposes."""
        root = scan_folder(linked_vault, "Research")

        responses = [
            "SELECT: Research/intro.md",
            "INSUFFICIENT: 부족",
            "DONE: 더 이상 필요 없음",
        ]

        result = run_retrieval("질문", root, _scripted_chat(responses))

        cross_ref_steps = [s for s in result.trace if s.phase == "cross-ref"]
        assert len(cross_ref_steps) >= 1
        step = cross_ref_steps[0]
        assert "intro" in step.detail
        assert "revenue" in step.detail

    def test_no_cross_refs_when_no_links(self, tmp_path: Path) -> None:
        """Notes without wiki-links should not generate cross-ref
        candidates — the loop should behave identically to v0.1."""
        vault = tmp_path / "vault"
        research = vault / "Research"
        research.mkdir(parents=True)

        (research / "standalone.md").write_text(
            "# Standalone\n\n독립적인 노트입니다. " * 50,
            encoding="utf-8",
        )

        root = scan_folder(vault, "Research")
        responses = [
            "SELECT: Research/standalone.md",
            "SUFFICIENT: ok",
            "답변입니다.",
        ]

        result = run_retrieval("질문", root, _scripted_chat(responses))

        cross_ref_steps = [s for s in result.trace if s.phase == "cross-ref"]
        assert len(cross_ref_steps) == 0

    def test_cross_ref_with_pre_built_link_index(
        self, linked_vault: Path
    ) -> None:
        """Passing a pre-built ``link_index`` should work identically
        to letting ``run_retrieval`` build one internally."""
        root = scan_folder(linked_vault, "Research")
        index = build_link_index(root)

        responses = [
            "SELECT: Research/intro.md",
            "INSUFFICIENT: 부족",
            "SELECT: Research/revenue.md",
            "SUFFICIENT: ok",
            "답변.",
        ]

        result = run_retrieval(
            "Q3?", root, _scripted_chat(responses), link_index=index
        )

        assert "Research/revenue.md" in result.cited_nodes


class TestCrossRefPromptAnnotation:
    """Verify that cross-ref candidates show the ``linked_from``
    annotation in the select prompt."""

    def test_linked_from_appears_in_prompt(self, tmp_path: Path) -> None:
        """When cursor is inside a subfolder, a cross-ref target from
        another subfolder should appear with a 교차참조 annotation."""
        vault = tmp_path / "vault"
        overview = vault / "Research" / "Overview"
        finance = vault / "Research" / "Finance"
        overview.mkdir(parents=True)
        finance.mkdir(parents=True)

        # intro links to [[revenue]] which is in a different subfolder
        (overview / "intro.md").write_text(
            "# Intro\n\n내용. [[revenue]] 참고. " * 40,
            encoding="utf-8",
        )
        (finance / "revenue.md").write_text(
            "# Revenue\n\n매출 데이터. " * 50,
            encoding="utf-8",
        )

        root = scan_folder(vault, "Research")

        prompts_seen: list[str] = []
        response_idx = {"i": 0}
        responses = [
            # Descend into Overview
            "SELECT: Research/Overview",
            # Pick intro inside Overview
            "SELECT: Research/Overview/intro.md",
            "INSUFFICIENT: 부족",
            # Now at cursor=root, revenue is a cross-ref from intro.
            # But Finance/ is a tree candidate. revenue itself should
            # appear as a cross-ref since it's not a direct child of root.
            "DONE: 끝",
        ]

        def _capturing_chat(prompt: str) -> str:
            prompts_seen.append(prompt)
            if response_idx["i"] >= len(responses):
                return "DONE: exhausted"
            reply = responses[response_idx["i"]]
            response_idx["i"] += 1
            return reply

        run_retrieval("질문", root, _capturing_chat)

        # The prompt after INSUFFICIENT should contain a cross-ref
        # annotation for revenue.md
        assert len(prompts_seen) >= 4
        last_select = prompts_seen[3]  # select, select, evaluate, select
        assert "교차참조" in last_select
