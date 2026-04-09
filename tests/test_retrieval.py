"""Multi-hop retrieval loop tests.

All LLM calls are mocked via a scripted ChatFn so these tests run fast and
deterministically without Ollama.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pagewiki.retrieval import RetrievalResult, run_retrieval
from pagewiki.vault import scan_folder


def _make_scripted_chat(responses: list[str]) -> Callable[[str], str]:
    """Return a chat_fn that replays `responses` in order.

    Extra calls beyond the scripted list return a fallback DONE response so
    the loop terminates instead of hanging.
    """
    idx = {"i": 0}

    def _call(prompt: str) -> str:
        if idx["i"] >= len(responses):
            return "DONE: script exhausted"
        reply = responses[idx["i"]]
        idx["i"] += 1
        return reply

    return _call


@pytest.fixture
def research_vault(tmp_path: Path) -> Path:
    """Small vault with a Research folder containing 3 notes of different tiers."""
    vault = tmp_path / "vault"
    research = vault / "Research"
    research.mkdir(parents=True)

    # ~300 tokens → MICRO
    (research / "intro.md").write_text(
        "# Intro\n\n간단한 소개 노트입니다. " * 5,
        encoding="utf-8",
    )
    # ~1500 tokens → ATOMIC
    (research / "q3_revenue.md").write_text(
        "# Q3 Revenue\n\n2024년 3분기 매출은 1조 2천억원입니다. " * 100,
        encoding="utf-8",
    )
    # ~5000 tokens → LONG
    (research / "annual_report.md").write_text(
        "# Annual Report 2024\n\n연간 보고서 전체 내용. " * 400,
        encoding="utf-8",
    )
    return vault


def test_retrieval_happy_path_single_hop(research_vault: Path) -> None:
    """Query finds the right note on first try, evaluator says SUFFICIENT."""
    root = scan_folder(research_vault, "Research")
    # Fixed child order after sorted(): annual_report, intro, q3_revenue

    chat = _make_scripted_chat(
        [
            "SELECT: Research/q3_revenue.md",  # Phase 2
            "SUFFICIENT: 매출 수치가 명시됨",  # Phase 3
            "2024년 Q3 매출은 1조 2천억원입니다. [[Q3 Revenue]]",  # Phase 4
        ]
    )

    result = run_retrieval("2024년 3분기 매출 얼마였어?", root, chat)

    assert isinstance(result, RetrievalResult)
    assert "1조 2천억원" in result.answer
    assert "Research/q3_revenue.md" in result.cited_nodes
    assert result.iterations_used == 1


def test_retrieval_multi_hop_insufficient_then_gather(research_vault: Path) -> None:
    """First note is insufficient; loop picks a second note before finalizing."""
    root = scan_folder(research_vault, "Research")

    chat = _make_scripted_chat(
        [
            "SELECT: Research/intro.md",       # iter 1 — pick intro
            "INSUFFICIENT: 매출 데이터 없음",    # iter 1 — evaluate
            "SELECT: Research/q3_revenue.md",  # iter 2 — pick revenue note
            "SUFFICIENT: 매출 포함",            # iter 2 — evaluate
            "종합하면 2024년 Q3 매출은 1조 2천억원. [[Q3 Revenue]]",  # Phase 4
        ]
    )

    result = run_retrieval("Q3 매출 요약", root, chat)

    assert result.iterations_used == 2
    assert len(result.cited_nodes) == 2
    assert "Research/intro.md" in result.cited_nodes
    assert "Research/q3_revenue.md" in result.cited_nodes


def test_retrieval_done_action_stops_loop(research_vault: Path) -> None:
    """When LLM returns DONE, loop terminates without gathering notes."""
    root = scan_folder(research_vault, "Research")

    chat = _make_scripted_chat(["DONE: 이 볼트엔 답이 없음"])

    result = run_retrieval("완전히 무관한 질문", root, chat)

    assert result.cited_nodes == []
    assert "근거 부족" in result.answer
    assert result.iterations_used == 1


def test_retrieval_max_iterations_safeguard(research_vault: Path) -> None:
    """Loop respects max_iterations cap when evaluator keeps saying INSUFFICIENT."""
    root = scan_folder(research_vault, "Research")

    # Each iteration needs 2 responses (select + evaluate)
    responses = [
        "SELECT: Research/intro.md",
        "INSUFFICIENT: 부족",
        "SELECT: Research/q3_revenue.md",
        "INSUFFICIENT: 부족",
        "SELECT: Research/annual_report.md",
        "INSUFFICIENT: 부족",
        # Fallback DONE after that (from _make_scripted_chat)
        "최종 답변: 근거를 종합한 결과...",
    ]
    chat = _make_scripted_chat(responses)

    result = run_retrieval("매출 질문", root, chat, max_iterations=5)

    # Should have explored up to max_iterations notes (or run out of candidates)
    assert result.iterations_used <= 5
    # All 3 notes should be cited since each was marked insufficient but gathered
    assert len(result.cited_nodes) == 3


def test_retrieval_invalid_node_id_retries(research_vault: Path) -> None:
    """If LLM picks a non-existent node_id, loop marks it visited and retries."""
    root = scan_folder(research_vault, "Research")

    chat = _make_scripted_chat(
        [
            "SELECT: nonexistent_node",         # iter 1 — invalid
            "SELECT: Research/q3_revenue.md",   # iter 2 — valid
            "SUFFICIENT: ok",
            "최종 답변입니다.",
        ]
    )

    result = run_retrieval("질문", root, chat, max_iterations=5)

    assert "Research/q3_revenue.md" in result.cited_nodes
    # The invalid id should appear in the trace
    trace_node_ids = [s.node_id for s in result.trace]
    assert "nonexistent_node" in trace_node_ids


def test_retrieval_empty_vault_returns_no_evidence(tmp_path: Path) -> None:
    """Empty folder → loop exits gracefully with a 'no evidence' answer."""
    vault = tmp_path / "vault"
    (vault / "Empty").mkdir(parents=True)

    root = scan_folder(vault, "Empty")
    chat = _make_scripted_chat([])  # no calls expected

    result = run_retrieval("아무 질문", root, chat)

    assert result.cited_nodes == []
    assert "근거 부족" in result.answer
