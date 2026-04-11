"""Cross-vault retrieval orchestrator (v0.12, split out in v0.13, parallel v0.15).

Sits on top of ``core.run_retrieval`` and ``decompose.run_decomposed_retrieval``.
Runs a full retrieval per vault and synthesizes the per-vault answers
into one via ``synthesize_multi_answer_prompt``.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..prompts import synthesize_multi_answer_prompt
from ..tree import TreeNode
from ..wiki_links import LinkIndex
from .core import run_retrieval
from .decompose import run_decomposed_retrieval
from .types import (
    DEFAULT_MAX_GATHERED_NOTES,
    DEFAULT_MAX_ITERATIONS,
    ChatFn,
    EventCallback,
    RetrievalResult,
    SystemChatFn,
    TraceStep,
)

if TYPE_CHECKING:
    from ..usage import UsageTracker


def run_cross_vault_retrieval(
    query: str,
    vault_roots: list[TreeNode],
    chat_fn: ChatFn,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_gathered: int = DEFAULT_MAX_GATHERED_NOTES,
    link_indexes: list[LinkIndex | None] | None = None,
    vault_labels: list[str] | None = None,
    on_event: EventCallback = None,
    max_tokens: int | None = None,
    tracker: UsageTracker | None = None,
    json_mode: bool = False,
    reuse_context: bool = False,
    should_stop: Callable[[], bool] | None = None,
    decompose: bool = False,
    system_chat_fn: SystemChatFn | None = None,
    parallel_workers: int = 1,
) -> RetrievalResult:
    """Run a retrieval loop independently per vault, then synthesize (v0.12).

    v0.15: ``parallel_workers > 1`` runs the per-vault retrievals on a
    ``ThreadPoolExecutor`` since each vault's loop is pure Python + I/O
    bound on the shared Ollama backend (which happily serves concurrent
    requests). Ordering of results is preserved so ``cited_nodes``
    prefixes stay deterministic.

    See ``retrieval/__init__.py`` for detailed parameter docs.
    """
    if not vault_roots:
        raise ValueError("run_cross_vault_retrieval requires at least one vault")

    n = len(vault_roots)
    labels = vault_labels or [f"vault-{i + 1}" for i in range(n)]
    indexes = link_indexes or [None] * n
    if len(labels) != n or len(indexes) != n:
        raise ValueError(
            "vault_labels / link_indexes must match vault_roots length"
        )

    if on_event is not None:
        parallel_note = (
            f" (parallel x{parallel_workers})" if parallel_workers > 1 else ""
        )
        on_event(
            TraceStep(
                "cross-vault",
                None,
                f"{n}개 vault에 대해 개별 retrieval 실행{parallel_note}: "
                f"{', '.join(labels)}",
            )
        )

    def _run_one(
        i: int, vault_root: TreeNode, label: str, link_idx: LinkIndex | None,
    ) -> tuple[int, str, RetrievalResult]:
        if on_event is not None:
            on_event(
                TraceStep(
                    "cross-vault",
                    None,
                    f"[{i + 1}/{n}] {label} 탐색 시작",
                )
            )
        if decompose:
            sub_result = run_decomposed_retrieval(
                query,
                vault_root,
                chat_fn,
                max_iterations=max_iterations,
                max_gathered=max_gathered,
                link_index=link_idx,
                on_event=on_event,
                max_tokens=max_tokens,
                tracker=tracker,
                json_mode=json_mode,
                reuse_context=reuse_context,
            )
        else:
            sub_result = run_retrieval(
                query,
                vault_root,
                chat_fn,
                max_iterations=max_iterations,
                max_gathered=max_gathered,
                link_index=link_idx,
                on_event=on_event,
                max_tokens=max_tokens,
                tracker=tracker,
                json_mode=json_mode,
                reuse_context=reuse_context,
                should_stop=should_stop,
                system_chat_fn=system_chat_fn,
            )
        return i, label, sub_result

    indexed_inputs = list(
        enumerate(zip(vault_roots, labels, indexes, strict=True))
    )

    if parallel_workers > 1 and n > 1:
        # v0.15 parallel path. Note: ``on_event`` is shared across
        # threads. TraceStep.append is a plain list op which is
        # CPython-atomic, so the interleaved events are safe to
        # collect even if visually out of order.
        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = [
                pool.submit(_run_one, i, vr, lbl, lidx)
                for i, (vr, lbl, lidx) in indexed_inputs
            ]
            raw = [f.result() for f in futures]
    else:
        raw = [
            _run_one(i, vr, lbl, lidx)
            for i, (vr, lbl, lidx) in indexed_inputs
        ]

    # Re-sort by the original index so per-vault outputs stay ordered.
    raw.sort(key=lambda t: t[0])

    per_vault_results: list[tuple[str, RetrievalResult]] = []
    merged_cited: list[str] = []
    merged_trace: list[TraceStep] = []
    for _i, label, sub_result in raw:
        per_vault_results.append((label, sub_result))
        merged_trace.extend(sub_result.trace)
        for nid in sub_result.cited_nodes:
            prefixed = f"{label}::{nid}"
            if prefixed not in merged_cited:
                merged_cited.append(prefixed)

    pairs = [
        (f"{label}에서 찾은 내용", result.answer)
        for label, result in per_vault_results
    ]
    synth_prompt = synthesize_multi_answer_prompt(query, pairs)
    final_answer = chat_fn(synth_prompt).strip()

    merged_trace.append(
        TraceStep(
            "cross-vault",
            None,
            f"synthesized {n} vault results into final answer",
        )
    )
    if on_event is not None:
        on_event(merged_trace[-1])

    total_iterations = sum(r.iterations_used for _, r in per_vault_results)
    return RetrievalResult(
        answer=final_answer,
        cited_nodes=merged_cited,
        trace=merged_trace,
        iterations_used=total_iterations,
    )
