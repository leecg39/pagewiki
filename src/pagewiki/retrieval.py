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
from typing import TYPE_CHECKING

from .prompts import (
    NodeSummary,
    build_retry_prompt,
    decompose_query_prompt,
    evaluate_prompt,
    final_answer_prompt,
    final_answer_with_history_prompt,
    parse_decompose_response,
    parse_evaluate_response,
    parse_select_response,
    select_node_prompt,
    synthesize_multi_answer_prompt,
)
from .tree import TreeNode
from .wiki_links import LinkIndex, build_link_index

if TYPE_CHECKING:
    from .usage import UsageTracker

ChatFn = Callable[[str], str]
EventCallback = Callable[["TraceStep"], None] | None

# Safety caps — prevent runaway LLM costs on pathological trees
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_GATHERED_NOTES = 4
DEFAULT_MAX_NOTE_CHARS = 20_000

@dataclass
class TraceStep:
    """Single step in the reasoning trace — useful for logging and debugging."""

    phase: str  # "select" | "evaluate" | "finalize" | "cross-ref"
    node_id: str | None
    detail: str


@dataclass
class RetrievalResult:
    """Outcome of a full retrieval loop."""

    answer: str
    cited_nodes: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    iterations_used: int = 0


def _node_as_summary(
    node: TreeNode,
    *,
    linked_from: str | None = None,
) -> NodeSummary:
    """Project a single TreeNode into the prompt-friendly NodeSummary shape."""
    return NodeSummary(
        node_id=node.node_id,
        title=node.title,
        kind=node.kind,
        summary=node.summary,
        token_count=node.token_count,
        linked_from=linked_from,
        tags=node.tags or None,
        date=node.date,
    )


def _children_as_summaries(node: TreeNode) -> list[NodeSummary]:
    """Project TreeNode children into the prompt-friendly NodeSummary shape."""
    return [_node_as_summary(child) for child in node.children]


def _promote_to_note(node: TreeNode, root: TreeNode) -> TreeNode:
    """If ``node`` is a section, walk up to its enclosing note.

    Section node_ids follow the ``<rel_path>#<id>`` convention set by
    ``pageindex_adapter``, so splitting on ``#`` gives the note's
    node_id. Returns ``node`` itself if it's already a note (or if the
    enclosing note can't be found).
    """
    if node.kind != "section":
        return node
    if "#" in node.node_id:
        note_id = node.node_id.rsplit("#", 1)[0]
        found = root.find(note_id)
        if found is not None:
            return found
    return node


def _load_note_content(node: TreeNode) -> str:
    """Read a note's file content, truncated to the safety cap.

    For ``kind == "section"`` nodes this uses ``node.line_range`` to
    read only the slice of the underlying markdown file that belongs to
    the section — crucial for LONG notes where loading the whole file
    would blow the context window.
    """
    if node.file_path is None:
        return ""
    path = Path(node.file_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")

    if node.kind == "section" and node.line_range is not None:
        start, end = node.line_range
        # line_range uses 1-indexed inclusive start, exclusive end to
        # match extract_nodes_from_markdown semantics.
        lines = text.split("\n")
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), max(start_idx, end - 1))
        text = "\n".join(lines[start_idx:end_idx])

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
    link_index: LinkIndex | None = None,
    on_event: EventCallback = None,
    history: list[tuple[str, str]] | None = None,
    max_tokens: int | None = None,
    tracker: UsageTracker | None = None,
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
        link_index: Pre-built wiki-link index. If ``None``, one is built
            automatically from ``root``. Pass a pre-built index to avoid
            redundant I/O when the caller (e.g. ``cli.ask``) has already
            built one.
        on_event: Optional callback invoked for each ``TraceStep`` as it
            is recorded. Used by the CLI to stream progress in real-time.
        history: Optional list of prior ``(question, answer)`` turns from
            chat mode. When provided, the final-answer prompt includes
            the conversation context.
        max_tokens: Optional hard cap on total tokens used during this
            retrieval (v0.9). Requires ``tracker`` to be provided.
            When the tracker's total exceeds this value, the loop
            aborts cleanly with a "budget exceeded" reason.
        tracker: Optional ``UsageTracker`` instance used for budget
            enforcement. The caller is responsible for wiring
            ``chat_fn`` through a tracking wrapper so usage is
            actually recorded.

    Returns:
        RetrievalResult with answer, cited node ids, and reasoning trace.
    """
    trace: list[TraceStep] = []

    def _record(step: TraceStep) -> None:
        trace.append(step)
        if on_event is not None:
            on_event(step)

    gathered: list[tuple[str, str]] = []  # (title, content)
    cited: list[str] = []
    visited_ids: set[str] = set()
    path_so_far: list[str] = []

    # Build the wiki-link index for cross-reference traversal (v0.2).
    if link_index is None:
        link_index = build_link_index(root)

    # Cross-reference pool: wiki-link targets discovered during
    # the loop that are outside the current cursor's subtree.
    # Maps node_id → (TreeNode, linked_from_label).
    cross_ref_pool: dict[str, tuple[TreeNode, str]] = {}

    # Cursor into the tree. Starts at root; moves down as folders are selected.
    cursor: TreeNode = root

    iteration = 0
    done_reason: str | None = None

    def _budget_exceeded() -> bool:
        if max_tokens is None or tracker is None:
            return False
        return tracker.total_tokens >= max_tokens

    while iteration < max_iterations and len(gathered) < max_gathered:
        iteration += 1

        # v0.9 token budget check — abort the loop if the caller-
        # specified cap has been hit. We check before every LLM call
        # rather than after so a single oversized prompt can't blow
        # past the budget silently.
        if _budget_exceeded():
            done_reason = (
                f"토큰 예산 초과: {tracker.total_tokens:,} >= {max_tokens:,}"
            )
            _record(TraceStep("budget", None, done_reason))
            break

        tree_candidates = [
            child for child in cursor.children if child.node_id not in visited_ids
        ]

        # Merge wiki-link cross-reference candidates that are not
        # already present as tree children under the current cursor.
        tree_ids = {c.node_id for c in tree_candidates}
        xref_candidates: list[tuple[TreeNode, str]] = [
            (node, label)
            for nid, (node, label) in cross_ref_pool.items()
            if nid not in visited_ids and nid not in tree_ids
        ]

        candidates = tree_candidates + [node for node, _ in xref_candidates]

        if not candidates:
            # Nothing left to look at under this cursor — bubble up
            if cursor is root and not xref_candidates:
                done_reason = "모든 후보 노드를 탐색 완료"
                break
            # Walk back up by searching for the parent. The simple approach:
            # just reset to root and rely on visited_ids to skip seen branches.
            cursor = root
            path_so_far = []
            continue

        # Phase 1+2: present candidates and ask the LLM to pick one.
        # Build NodeSummary list with cross-ref annotations.
        summaries: list[NodeSummary] = [
            _node_as_summary(child) for child in tree_candidates
        ]
        for node, label in xref_candidates:
            summaries.append(_node_as_summary(node, linked_from=label))

        # v0.8: pre-rank candidates by BM25-style term overlap so the
        # most likely matches appear first in the prompt. This is a
        # zero-LLM hint that helps the model land on the right node
        # in fewer iterations on large vaults.
        if len(summaries) > 1:
            from .ranking import rank_candidates

            searchable = [
                (s.title, f"{s.title} {s.summary or ''}")
                for s in summaries
            ]
            order = rank_candidates(query, searchable)
            reordered_summaries = [summaries[i] for i, _ in order]
            reordered_candidates = [candidates[i] for i, _ in order]
            summaries = reordered_summaries
            candidates = reordered_candidates

        prompt = select_node_prompt(
            query,
            summaries,
            path_so_far=path_so_far or None,
        )
        response = chat_fn(prompt)

        try:
            action, value = parse_select_response(response)
        except ValueError as e:
            # v0.8 parse-retry: give the LLM one more chance with a
            # stricter format reminder before aborting the loop.
            _record(
                TraceStep(
                    "select",
                    None,
                    f"parse error (retrying): {e}",
                )
            )
            retry_response = chat_fn(build_retry_prompt(prompt, str(e)))
            try:
                action, value = parse_select_response(retry_response)
                _record(TraceStep("select", None, "retry succeeded"))
            except ValueError as e2:
                _record(TraceStep("select", None, f"retry failed: {e2}"))
                done_reason = f"응답 파싱 실패: {e2}"
                break

        if action == "DONE":
            _record(TraceStep("select", None, f"DONE: {value}"))
            done_reason = value
            break

        # action == "SELECT"
        picked = next((c for c in candidates if c.node_id == value), None)
        if picked is None:
            _record(
                TraceStep("select", value, "invalid node_id (not in candidates)")
            )
            # Mark visited to avoid re-offering and retry the loop
            visited_ids.add(value)
            continue

        visited_ids.add(picked.node_id)
        is_xref = value in cross_ref_pool
        _record(
            TraceStep(
                "select",
                picked.node_id,
                f"picked {picked.title}"
                + (" (cross-ref)" if is_xref else ""),
            )
        )

        # If following a cross-ref that lives outside the current
        # cursor's subtree, reset the cursor so descend logic works.
        if is_xref:
            cursor = root
            path_so_far = []

        if picked.kind == "folder":
            # Descend into subfolder — next iteration runs ToC Review on it
            cursor = picked
            path_so_far.append(picked.title)
            continue

        # LONG notes with a Layer 2 sub-tree behave like folders: instead
        # of loading the whole note body, descend into the PageIndex
        # section tree and let the next iteration pick a specific section.
        if picked.kind == "note" and picked.children:
            cursor = picked
            path_so_far.append(picked.title)
            continue

        # picked.kind == "note" (leaf) or "section" — Phase 3: Extract & Evaluate
        content = _load_note_content(picked)
        if not content:
            _record(TraceStep("evaluate", picked.node_id, "empty content"))
            continue

        eval_prompt = evaluate_prompt(query, picked.title, content)
        eval_response = chat_fn(eval_prompt)
        try:
            sufficient, reason = parse_evaluate_response(eval_response)
        except ValueError as e:
            _record(TraceStep("evaluate", picked.node_id, f"parse error: {e}"))
            # Treat parse failure as "gather this and keep going"
            gathered.append((picked.title, content))
            cited.append(picked.node_id)
            continue

        gathered.append((picked.title, content))
        cited.append(picked.node_id)
        _record(
            TraceStep(
                "evaluate",
                picked.node_id,
                f"{'SUFFICIENT' if sufficient else 'INSUFFICIENT'}: {reason}",
            )
        )

        # v0.2: after evaluating a note, enqueue its outgoing wiki-link
        # targets as cross-reference candidates for future iterations.
        # Section targets are promoted to their enclosing note so the
        # normal descend logic applies.
        source_note_id = (
            picked.node_id.rsplit("#", 1)[0]
            if "#" in picked.node_id
            else picked.node_id
        )
        for link in link_index.outgoing(source_note_id):
            target = _promote_to_note(link.target, root)
            if target.node_id not in visited_ids and target.node_id not in cross_ref_pool:
                label = f"{Path(source_note_id).stem} → [[{link.raw_target}]]"
                cross_ref_pool[target.node_id] = (target, label)
                _record(
                    TraceStep(
                        "cross-ref",
                        target.node_id,
                        f"enqueued from {source_note_id} via [[{link.raw_target}]]",
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
    elif _budget_exceeded():
        # v0.9 — budget already exhausted; return the raw evidence
        # concatenated as a partial answer rather than paying for
        # one more synthesis call.
        parts = ["[토큰 예산 초과 — 부분 결과만 반환]"]
        for title, content in gathered:
            parts.append(f"\n## {title}\n{content[:1000]}")
        answer = "\n".join(parts)
    elif history:
        final_prompt = final_answer_with_history_prompt(query, gathered, history)
        answer = chat_fn(final_prompt).strip()
    else:
        final_prompt = final_answer_prompt(query, gathered)
        answer = chat_fn(final_prompt).strip()

    # v0.9 cited-note re-ranking: sort citations by BM25 relevance to
    # the original query so the most important source appears first
    # in the cited_nodes list. Discovery order is an artifact of
    # traversal, not relevance — this is a zero-LLM fix. When gathering
    # 0 or 1 notes there is nothing to reorder.
    if len(gathered) > 1:
        from .ranking import rank_candidates

        searchable = [
            (title, f"{title} {content[:2000]}")
            for title, content in gathered
        ]
        order = rank_candidates(query, searchable)
        reordered_cited = [cited[i] for i, _ in order]
        cited = reordered_cited

    _record(
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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query decomposition (v0.7)
# ─────────────────────────────────────────────────────────────────────────────


def run_decomposed_retrieval(
    query: str,
    root: TreeNode,
    chat_fn: ChatFn,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_gathered: int = DEFAULT_MAX_GATHERED_NOTES,
    link_index: LinkIndex | None = None,
    on_event: EventCallback = None,
    max_sub_queries: int = 4,
    max_tokens: int | None = None,
    tracker: UsageTracker | None = None,
) -> RetrievalResult:
    """Decompose a complex query into sub-queries, retrieve in parallel, synthesize.

    The workflow is:

      1. Ask the LLM to split the query into 1-N sub-questions. If it
         returns ``SINGLE``, fall through to a normal ``run_retrieval``.
      2. Run ``run_retrieval`` once per sub-query. They share the
         same tree and link_index but have independent reasoning traces.
      3. Synthesize a single final answer from the per-sub-query
         answers using ``synthesize_multi_answer_prompt``.

    All sub-query cited_nodes are merged (deduplicated) into the result.

    The ``max_tokens`` / ``tracker`` pair is forwarded to every
    per-sub-query ``run_retrieval`` call so the budget is enforced
    across the entire decomposed flow, not just one sub-query.
    """
    # Phase 1: decompose.
    decomp_prompt = decompose_query_prompt(query, max_sub_queries=max_sub_queries)
    decomp_response = chat_fn(decomp_prompt)
    sub_queries = parse_decompose_response(decomp_response)

    if not sub_queries:
        # Simple query — use the standard single-pass loop.
        if on_event is not None:
            on_event(TraceStep("decompose", None, "SINGLE: 단일 쿼리로 처리"))
        return run_retrieval(
            query, root, chat_fn,
            max_iterations=max_iterations,
            max_gathered=max_gathered,
            link_index=link_index,
            on_event=on_event,
            max_tokens=max_tokens,
            tracker=tracker,
        )

    if on_event is not None:
        on_event(
            TraceStep(
                "decompose",
                None,
                f"{len(sub_queries)}개 하위 질문으로 분해: "
                + " | ".join(sub_queries),
            )
        )

    # Phase 2: run retrieval per sub-query (sequential; each hits the
    # LLM ~5-10 times so parallelizing them would thrash Ollama).
    sub_results: list[tuple[str, RetrievalResult]] = []
    merged_trace: list[TraceStep] = []
    merged_cited: list[str] = []

    for i, sub in enumerate(sub_queries, start=1):
        if on_event is not None:
            on_event(
                TraceStep(
                    "decompose",
                    None,
                    f"[{i}/{len(sub_queries)}] 실행: {sub}",
                )
            )
        sub_result = run_retrieval(
            sub, root, chat_fn,
            max_iterations=max_iterations,
            max_gathered=max_gathered,
            link_index=link_index,
            on_event=on_event,
            max_tokens=max_tokens,
            tracker=tracker,
        )
        sub_results.append((sub, sub_result))
        merged_trace.extend(sub_result.trace)
        for nid in sub_result.cited_nodes:
            if nid not in merged_cited:
                merged_cited.append(nid)

    # Phase 3: synthesize.
    sub_qa_pairs = [(sub_q, sub_r.answer) for sub_q, sub_r in sub_results]
    synth_prompt = synthesize_multi_answer_prompt(query, sub_qa_pairs)
    final_answer = chat_fn(synth_prompt).strip()

    merged_trace.append(
        TraceStep(
            "decompose",
            None,
            f"synthesized {len(sub_results)} sub-answers into final response",
        )
    )
    if on_event is not None:
        on_event(merged_trace[-1])

    total_iterations = sum(r.iterations_used for _, r in sub_results)
    return RetrievalResult(
        answer=final_answer,
        cited_nodes=merged_cited,
        trace=merged_trace,
        iterations_used=total_iterations,
    )
