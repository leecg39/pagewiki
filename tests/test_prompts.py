"""Prompt template and parser tests — pure functions, no LLM required."""

from __future__ import annotations

import pytest

from pagewiki.prompts import (
    NodeSummary,
    evaluate_prompt,
    final_answer_prompt,
    parse_evaluate_response,
    parse_select_response,
    select_node_prompt,
)


def test_select_prompt_contains_query_and_candidates() -> None:
    prompt = select_node_prompt(
        "2024년 3분기 매출 분석",
        [
            NodeSummary(node_id="Research/q3.md", title="Q3 Revenue", kind="note"),
            NodeSummary(node_id="Research/risks.md", title="Risk Factors", kind="note"),
        ],
    )
    assert "2024년 3분기 매출 분석" in prompt
    assert "Research/q3.md" in prompt
    assert "Research/risks.md" in prompt
    assert "SELECT:" in prompt
    assert "DONE:" in prompt


def test_select_prompt_includes_breadcrumb() -> None:
    prompt = select_node_prompt(
        "x",
        [NodeSummary(node_id="a", title="A", kind="folder")],
        path_so_far=["Root", "Subfolder"],
    )
    assert "Root > Subfolder" in prompt


def test_parse_select_response_select() -> None:
    action, value = parse_select_response("SELECT: Research/q3.md")
    assert action == "SELECT"
    assert value == "Research/q3.md"


def test_parse_select_response_done() -> None:
    action, value = parse_select_response("DONE: 모든 후보가 관련 없음")
    assert action == "DONE"
    assert "관련 없음" in value


def test_parse_select_response_tolerates_preamble() -> None:
    # Some models prepend reasoning before the final line
    response = "먼저 후보를 분석해봅니다...\nSELECT: node_42"
    action, value = parse_select_response(response)
    assert action == "SELECT"
    assert value == "node_42"


def test_parse_select_response_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        parse_select_response("I cannot decide.")


def test_evaluate_prompt_truncates_long_content() -> None:
    long_content = "a" * 50_000
    prompt = evaluate_prompt("q", "big note", long_content)
    # Should be capped at 12000 chars of content
    assert len(prompt) < 20_000


def test_parse_evaluate_sufficient() -> None:
    ok, reason = parse_evaluate_response("SUFFICIENT: 모든 정보 포함")
    assert ok is True
    assert "정보 포함" in reason


def test_parse_evaluate_insufficient() -> None:
    ok, reason = parse_evaluate_response("INSUFFICIENT: Q4 데이터 없음")
    assert ok is False
    assert "Q4" in reason


def test_final_answer_prompt_includes_all_notes() -> None:
    prompt = final_answer_prompt(
        "요약 부탁",
        [
            ("Note A", "content A"),
            ("Note B", "content B"),
        ],
    )
    assert "Note A" in prompt
    assert "Note B" in prompt
    assert "content A" in prompt
    assert "content B" in prompt
    assert "[[제목]]" in prompt  # citation format hint
