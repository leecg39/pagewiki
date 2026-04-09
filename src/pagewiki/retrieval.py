"""Multi-hop reasoning retrieval loop.

This module is the heart of pagewiki's v0.1.1. It takes a query plus a Layer 1
tree and walks it via LLM reasoning instead of vector similarity:

    1. ToC Review:    present top-level candidates to the LLM
    2. Select Node:   LLM picks one by node_id
    3. Descend:       folder → recurse; note → load content
    4. Extract/Eval:  is this note sufficient?
    5. Conditional:   loop back or finalize

Design: the loop is pure logic. All LLM calls go through the injected
`chat_fn: Callable[[str], str]` so tests can script responses without
requiring Ollama to be running.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import (
    NodeSummary,
    evaluate_prompt,
    final_answer_prompt,
    parse_evaluate_response,
    parse_select_response,
    select_node_prompt,
)
from .tree import NoteTier, TreeNode

ChatFn = Callable[[str], str]

# Safety caps — prevent runaway LLM costs on pathological trees
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_GATHERED_NOTES = 4
DEFAULT_MAX_NOTE_CHARS = 20_000


@dataclass
class TraceStep:
    """Single step in the reasoning trace — useful for logging and debugging."""

    phase: str  # "select" | "evaluate" | "finalize"
    node_id: str | None
    detail: str


@dataclass
class RetrievalResult:
    """Outcome of a full retrieval loop."""

    answer: str
    cited_nodes: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    iterations_used: int = 0


def _children_as_summaries(node: TreeNode) -> list[NodeSummary]:
    """Project TreeNode children into the prompt-friendly NodeSummary shape."""
    return [
        NodeSummary(
            node_id=child.node_id,
            title=child.title,
            kind=child.kind,
            summary=child.summary,
            token_count=child.token_count,
        )
        for child in node.children
    ]


def _load_note_content(node: TreeNode) -> str:
    """Read a note's file content, truncated to the safety cap."""
    if node.file_path is None:
        return ""
    path = Path(node.file_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) > DEFAULT_MAX_NOTE_CHARS:
        text = text[:DEFAULT_MAX_NOTE_CHARS] + "\n\n[... truncated ...]"
    return text


def run_retrieval(
    query: str,
    root: TreeNode,
    chat_fn: ChatFn,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_gathered: int = DEFAULT_MAX_GATHERED_NOTES,
) -> RetrievalResult:
    """Run the multi-hop reasoning loop.

    Args:
        query: User question in natural language.
        root: Layer 1 tree root (from vault.scan_folder).
        chat_fn: Callable that takes a prompt string and returns the LLM's raw
            text response. Production uses `ollama_client.chat(...).text`;
            tests inject a fake that returns scripted responses.
        max_iterations: Safety cap on total select→evaluate cycles.
        max_gathered: Stop exploring once this many notes have been collected.

    Returns:
        RetrievalResult with answer, cited node ids, and reasoning trace.
    """
    trace: list[TraceStep] = []
    gathered: list[tuple[str, str]] = []  # (title, content)
    cited: list[str] = []
    visited_ids: set[str] = set()
    path_so_far: list[str] = []

    # Cursor into the tree. Starts at root; moves down as folders are selected.
    cursor: TreeNode = root

    iteration = 0
    done_reason: str | None = None

    while iteration < max_iterations and len(gathered) < max_gathered:
        iteration += 1

        candidates = [
            child for child in cursor.children if child.node_id not in visited_ids
        ]
        if not candidates:
            # Nothing left to look at under this cursor — bubble up
            if cursor is root:
                done_reason = "모든 후보 노드를 탐색 완료"
                break
            # Walk back up by searching for the parent. The simple approach:
            # just reset to root and rely on visited_ids to skip seen branches.
            cursor = root
            path_so_far = []
            continue

        # Phase 1+2: present candidates and ask the LLM to pick one
        cursor_view = TreeNode(
            node_id=cursor.node_id,
            title=cursor.title,
            kind=cursor.kind,
            children=candidates,
        )
        prompt = select_node_prompt(
            query,
            _children_as_summaries(cursor_view),
            path_so_far=path_so_far or None,
        )
        response = chat_fn(prompt)

        try:
            action, value = parse_select_response(response)
        except ValueError as e:
            trace.append(TraceStep("select", None, f"parse error: {e}"))
            done_reason = f"응답 파싱 실패: {e}"
            break

        if action == "DONE":
            trace.append(TraceStep("select", None, f"DONE: {value}"))
            done_reason = value
            break

        # action == "SELECT"
        picked = next((c for c in candidates if c.node_id == value), None)
        if picked is None:
            trace.append(
                TraceStep("select", value, f"invalid node_id (not in candidates)")
            )
            # Mark visited to avoid re-offering and retry the loop
            visited_ids.add(value)
            continue

        visited_ids.add(picked.node_id)
        trace.append(TraceStep("select", picked.node_id, f"picked {picked.title}"))

        if picked.kind == "folder":
            # Descend into subfolder — next iteration runs ToC Review on it
            cursor = picked
            path_so_far.append(picked.title)
            continue

        # picked.kind == "note" — Phase 3: Extract & Evaluate
        content = _load_note_content(picked)
        if not content:
            trace.append(TraceStep("evaluate", picked.node_id, "empty content"))
            continue

        eval_prompt = evaluate_prompt(query, picked.title, content)
        eval_response = chat_fn(eval_prompt)
        try:
            sufficient, reason = parse_evaluate_response(eval_response)
        except ValueError as e:
            trace.append(TraceStep("evaluate", picked.node_id, f"parse error: {e}"))
            # Treat parse failure as "gather this and keep going"
            gathered.append((picked.title, content))
            cited.append(picked.node_id)
            continue

        gathered.append((picked.title, content))
        cited.append(picked.node_id)
        trace.append(
            TraceStep(
                "evaluate",
                picked.node_id,
                f"{'SUFFICIENT' if sufficient else 'INSUFFICIENT'}: {reason}",
            )
        )

        if sufficient:
            done_reason = reason
            break

    # Phase 4: Final Answer
    if not gathered:
        answer = (
            f"[근거 부족] 탐색 결과 관련 노트를 찾지 못했습니다. "
            f"이유: {done_reason or '후보 소진'}"
        )
    else:
        final_prompt = final_answer_prompt(query, gathered)
        answer = chat_fn(final_prompt).strip()

    trace.append(
        TraceStep(
            "finalize",
            None,
            f"gathered={len(gathered)}, iterations={iteration}, reason={done_reason or 'max_iter'}",
        )
    )

    return RetrievalResult(
        answer=answer,
        cited_nodes=cited,
        trace=trace,
        iterations_used=iteration,
    )
