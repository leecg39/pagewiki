"""Multi-query decomposition orchestrator (v0.7, split out in v0.13).

Sits on top of ``core.run_retrieval``. When the decompose prompt
returns a list of sub-queries, we run ``run_retrieval`` once per
sub-query and synthesize the results with ``synthesize_multi_answer_prompt``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..prompts import (
    decompose_query_prompt,
    parse_decompose_response,
    synthesize_multi_answer_prompt,
)
from ..tree import TreeNode
from ..wiki_links import LinkIndex
from .core import run_retrieval
from .types import (
    DEFAULT_MAX_GATHERED_NOTES,
    DEFAULT_MAX_ITERATIONS,
    ChatFn,
    EventCallback,
    RetrievalResult,
    TraceStep,
)

if TYPE_CHECKING:
    from ..usage import UsageTracker


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
    json_mode: bool = False,
    reuse_context: bool = False,
) -> RetrievalResult:
    """Decompose a complex query into sub-queries, retrieve, synthesize."""
    decomp_prompt = decompose_query_prompt(query, max_sub_queries=max_sub_queries)
    decomp_response = chat_fn(decomp_prompt)
    sub_queries = parse_decompose_response(decomp_response)

    if not sub_queries:
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
            json_mode=json_mode,
            reuse_context=reuse_context,
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
            json_mode=json_mode,
            reuse_context=reuse_context,
        )
        sub_results.append((sub, sub_result))
        merged_trace.extend(sub_result.trace)
        for nid in sub_result.cited_nodes:
            if nid not in merged_cited:
                merged_cited.append(nid)

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
