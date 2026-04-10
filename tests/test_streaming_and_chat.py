"""Tests for v0.6 streaming output and chat-mode prompt helpers."""

from __future__ import annotations

from pagewiki.prompts import (
    final_answer_with_history_prompt,
    rewrite_query_with_context,
)
from pagewiki.retrieval import TraceStep, run_retrieval
from pagewiki.tree import NoteTier, TreeNode


def _simple_tree() -> TreeNode:
    """A root with one atomic note."""
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
                summary="테스트 노트",
            ),
        ],
    )


class TestStreamingCallback:
    def test_on_event_receives_trace_steps(self) -> None:
        root = _simple_tree()
        events: list[TraceStep] = []

        responses = iter([
            "DONE: 관련 노드 없음",
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        run_retrieval(
            "test query",
            root,
            fake_chat,
            on_event=lambda step: events.append(step),
        )

        # At minimum: select DONE + finalize
        assert len(events) >= 2
        phases = [e.phase for e in events]
        assert "select" in phases
        assert "finalize" in phases

    def test_on_event_fires_for_evaluate(self, tmp_path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text("# Test\n\nContent about topic.", encoding="utf-8")

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
                    summary="테스트",
                ),
            ],
        )

        events: list[TraceStep] = []
        responses = iter([
            "SELECT: note.md",
            "SUFFICIENT: 충분한 정보",
            "최종 답변입니다.",
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        run_retrieval(
            "test",
            root,
            fake_chat,
            on_event=lambda step: events.append(step),
        )

        phases = [e.phase for e in events]
        assert "evaluate" in phases

    def test_none_callback_is_safe(self) -> None:
        """on_event=None should not break the loop."""
        root = _simple_tree()

        result = run_retrieval(
            "test",
            root,
            lambda p: "DONE: no match",
            on_event=None,
        )
        assert result.answer  # didn't crash


class TestChatPrompts:
    def test_rewrite_query_includes_history(self) -> None:
        history = [
            ("Transformer란?", "Transformer는 어텐션 기반 모델입니다."),
        ]
        prompt = rewrite_query_with_context("장점은?", history)
        assert "Transformer" in prompt
        assert "장점" in prompt
        assert "Q1:" in prompt

    def test_rewrite_query_limits_to_3_turns(self) -> None:
        history = [(f"Question_{i}", f"Answer_{i}") for i in range(10)]
        prompt = rewrite_query_with_context("follow-up", history)
        # Only the last 3 turns should appear (indices 7, 8, 9).
        assert "Question_7" in prompt
        assert "Question_8" in prompt
        assert "Question_9" in prompt
        # Earlier turns must be excluded.
        assert "Question_0" not in prompt
        assert "Question_6" not in prompt

    def test_final_answer_with_history(self) -> None:
        history = [
            ("Transformer란?", "어텐션 기반 모델"),
        ]
        gathered = [("Paper", "Transformer uses self-attention...")]
        prompt = final_answer_with_history_prompt("장점은?", gathered, history)
        assert "[이전 대화]" in prompt
        assert "Transformer" in prompt
        assert "장점" in prompt

    def test_final_answer_without_history(self) -> None:
        gathered = [("Paper", "content")]
        prompt = final_answer_with_history_prompt("query", gathered, [])
        assert "[이전 대화]" not in prompt


class TestRetrievalWithHistory:
    def test_history_passed_to_final_answer(self, tmp_path) -> None:
        note_path = tmp_path / "note.md"
        note_path.write_text("# Test\n\nContent.", encoding="utf-8")

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

        prompts_seen: list[str] = []
        responses = iter([
            "SELECT: note.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake_chat(prompt: str) -> str:
            prompts_seen.append(prompt)
            return next(responses)

        history = [("이전 질문", "이전 답변")]

        run_retrieval(
            "후속 질문",
            root,
            fake_chat,
            history=history,
        )

        # The final prompt (last one) should contain history context.
        final_prompt = prompts_seen[-1]
        assert "이전 대화" in final_prompt
        assert "이전 질문" in final_prompt
