"""Core multi-hop reasoning loop (v0.13 refactor split).

Moved out of ``retrieval.py`` into its own module to make the 300+
line main loop easier to audit and test. The public API
(``run_retrieval``) is re-exported by ``retrieval/__init__.py`` so
no callers need to change.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..prompts import (
    EVALUATE_SYSTEM,
    FINAL_ANSWER_SYSTEM,
    SELECT_NODE_SYSTEM,
    NodeSummary,
    build_retry_prompt,
    evaluate_prompt,
    evaluate_prompt_json,
    evaluate_user_prompt,
    final_answer_prompt,
    final_answer_user_prompt,
    final_answer_with_history_prompt,
    parse_evaluate_response,
    parse_evaluate_response_json,
    parse_select_response,
    parse_select_response_json,
    select_node_prompt,
    select_node_prompt_json,
    select_node_user_prompt,
)
from ..prompts_en import (
    EN_EVALUATE_SYSTEM,
    EN_FINAL_ANSWER_SYSTEM,
    EN_SELECT_NODE_SYSTEM,
    evaluate_prompt_en,
    final_answer_prompt_en,
    select_node_prompt_en,
)
from ..tree import TreeNode
from ..wiki_links import LinkIndex, build_link_index
from .helpers import _load_note_content, _node_as_summary, _promote_to_note
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
    json_mode: bool = False,
    reuse_context: bool = False,
    should_stop: Callable[[], bool] | None = None,
    system_chat_fn: SystemChatFn | None = None,
    lang: str = "ko",
) -> RetrievalResult:
    """Run the multi-hop reasoning loop.

    See ``retrieval/__init__.py`` for parameter documentation — kept
    brief here because this module is internal.

    v0.14 ``system_chat_fn``: when provided, the select/evaluate/final
    phases use user-only prompt variants (``select_node_user_prompt``
    etc.) and invoke ``system_chat_fn(system, user)`` instead of the
    regular ``chat_fn(prompt)``. The ``system`` argument is the
    appropriate stable constant from ``prompts`` (``SELECT_NODE_SYSTEM``,
    ``EVALUATE_SYSTEM``, ``FINAL_ANSWER_SYSTEM``). Keeping the system
    prefix identical across calls lets Ollama reuse the KV cache for
    the prefix tokens, cutting repeat-call latency substantially.

    Note that JSON-mode and history-aware final answers still use
    the full (system+user) prompt path — the caching opt-in only
    applies to the plain-text code path.
    """
    use_prompt_cache = system_chat_fn is not None
    use_en = (lang or "ko").lower() == "en"
    # Pick language-appropriate system constants for the prompt-cache path.
    active_select_system = EN_SELECT_NODE_SYSTEM if use_en else SELECT_NODE_SYSTEM
    active_evaluate_system = EN_EVALUATE_SYSTEM if use_en else EVALUATE_SYSTEM
    active_final_system = EN_FINAL_ANSWER_SYSTEM if use_en else FINAL_ANSWER_SYSTEM
    trace: list[TraceStep] = []

    def _record(step: TraceStep) -> None:
        trace.append(step)
        if on_event is not None:
            on_event(step)

    gathered: list[tuple[str, str]] = []  # (title, content)
    cited: list[str] = []
    visited_ids: set[str] = set()
    shown_ids: set[str] = set()
    path_so_far: list[str] = []

    if link_index is None:
        link_index = build_link_index(root)

    cross_ref_pool: dict[str, tuple[TreeNode, str]] = {}
    cursor: TreeNode = root

    iteration = 0
    done_reason: str | None = None

    def _budget_exceeded() -> bool:
        if max_tokens is None or tracker is None:
            return False
        return tracker.total_tokens >= max_tokens

    def _should_stop() -> bool:
        return should_stop is not None and should_stop()

    while iteration < max_iterations and len(gathered) < max_gathered:
        iteration += 1

        if _should_stop():
            done_reason = "사용자 요청으로 중단"
            _record(TraceStep("cancel", None, done_reason))
            break

        if _budget_exceeded():
            done_reason = (
                f"토큰 예산 초과: {tracker.total_tokens:,} >= {max_tokens:,}"
            )
            _record(TraceStep("budget", None, done_reason))
            break

        tree_candidates = [
            child for child in cursor.children if child.node_id not in visited_ids
        ]
        tree_ids = {c.node_id for c in tree_candidates}
        xref_candidates: list[tuple[TreeNode, str]] = [
            (node, label)
            for nid, (node, label) in cross_ref_pool.items()
            if nid not in visited_ids and nid not in tree_ids
        ]

        candidates = tree_candidates + [node for node, _ in xref_candidates]

        if not candidates:
            if cursor is root and not xref_candidates:
                done_reason = "모든 후보 노드를 탐색 완료"
                break
            cursor = root
            path_so_far = []
            continue

        summaries: list[NodeSummary] = [
            _node_as_summary(child) for child in tree_candidates
        ]
        for node, label in xref_candidates:
            summaries.append(_node_as_summary(node, linked_from=label))

        # v0.8 BM25 pre-ranking.
        if len(summaries) > 1:
            from ..ranking import rank_candidates

            searchable = [
                (s.title, f"{s.title} {s.summary or ''}")
                for s in summaries
            ]
            order = rank_candidates(query, searchable)
            summaries = [summaries[i] for i, _ in order]
            candidates = [candidates[i] for i, _ in order]

        # v0.10 context reuse.
        effective_path = path_so_far
        if reuse_context and path_so_far and len(path_so_far) > 3:
            effective_path = [
                "...(생략)",
                *path_so_far[-3:],
            ]
        if reuse_context and shown_ids:
            pre_len = len(summaries)
            kept = [
                (s, c) for s, c in zip(summaries, candidates, strict=True)
                if s.node_id not in shown_ids
            ]
            if kept:
                summaries = [s for s, _ in kept]
                candidates = [c for _, c in kept]
                if len(summaries) < pre_len:
                    _record(
                        TraceStep(
                            "reuse",
                            None,
                            f"suppressed {pre_len - len(summaries)} already-shown candidates",
                        )
                    )

        if json_mode:
            prompt = select_node_prompt_json(
                query, summaries, path_so_far=effective_path or None,
            )
        elif use_prompt_cache:
            # v0.14 prompt caching: send the stable SELECT system
            # preamble separately so Ollama can reuse its KV cache.
            # Prompt cache + lang=en reuses the user-only Korean
            # prompt structure (the header is already stripped).
            prompt = select_node_user_prompt(
                query, summaries, path_so_far=effective_path or None,
            )
        elif use_en:
            prompt = select_node_prompt_en(
                query, summaries, path_so_far=effective_path or None,
            )
        else:
            prompt = select_node_prompt(
                query, summaries, path_so_far=effective_path or None,
            )

        for s in summaries:
            shown_ids.add(s.node_id)

        if use_prompt_cache and not json_mode:
            response = system_chat_fn(active_select_system, prompt)
        else:
            response = chat_fn(prompt)

        _select_parser = (
            parse_select_response_json if json_mode else parse_select_response
        )

        try:
            action, value = _select_parser(response)
        except ValueError as e:
            _record(
                TraceStep("select", None, f"parse error (retrying): {e}")
            )
            retry_prompt = build_retry_prompt(prompt, str(e))
            if use_prompt_cache and not json_mode:
                retry_response = system_chat_fn(active_select_system, retry_prompt)
            else:
                retry_response = chat_fn(retry_prompt)
            try:
                action, value = _select_parser(retry_response)
                _record(TraceStep("select", None, "retry succeeded"))
            except ValueError as e2:
                if json_mode:
                    try:
                        action, value = parse_select_response(retry_response)
                        _record(
                            TraceStep(
                                "select",
                                None,
                                "JSON retry failed, fell back to text parser",
                            )
                        )
                    except ValueError:
                        _record(
                            TraceStep("select", None, f"retry failed: {e2}"),
                        )
                        done_reason = f"응답 파싱 실패: {e2}"
                        break
                else:
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

        if is_xref:
            cursor = root
            path_so_far = []

        if picked.kind == "folder":
            cursor = picked
            path_so_far.append(picked.title)
            continue

        if picked.kind == "note" and picked.children:
            cursor = picked
            path_so_far.append(picked.title)
            continue

        # Phase 3: Extract & Evaluate
        content = _load_note_content(picked)
        if not content:
            _record(TraceStep("evaluate", picked.node_id, "empty content"))
            continue

        if json_mode:
            eval_prompt = evaluate_prompt_json(query, picked.title, content)
            eval_response = chat_fn(eval_prompt)
        elif use_prompt_cache:
            eval_prompt = evaluate_user_prompt(query, picked.title, content)
            eval_response = system_chat_fn(active_evaluate_system, eval_prompt)
        elif use_en:
            eval_prompt = evaluate_prompt_en(query, picked.title, content)
            eval_response = chat_fn(eval_prompt)
        else:
            eval_prompt = evaluate_prompt(query, picked.title, content)
            eval_response = chat_fn(eval_prompt)

        _eval_parser = (
            parse_evaluate_response_json if json_mode else parse_evaluate_response
        )

        try:
            sufficient, reason = _eval_parser(eval_response)
        except ValueError as e:
            _record(TraceStep("evaluate", picked.node_id, f"parse error: {e}"))
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

        # v0.2 wiki-link cross-reference enqueue.
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
        parts = ["[토큰 예산 초과 — 부분 결과만 반환]"]
        for title, content in gathered:
            parts.append(f"\n## {title}\n{content[:1000]}")
        answer = "\n".join(parts)
    elif history:
        # History-aware final answer still uses the full (system+user)
        # path — caching doesn't help since the history content varies.
        final_prompt = final_answer_with_history_prompt(query, gathered, history)
        answer = chat_fn(final_prompt).strip()
    elif use_prompt_cache:
        final_user = final_answer_user_prompt(query, gathered)
        answer = system_chat_fn(active_final_system, final_user).strip()
    elif use_en:
        final_prompt = final_answer_prompt_en(query, gathered)
        answer = chat_fn(final_prompt).strip()
    else:
        final_prompt = final_answer_prompt(query, gathered)
        answer = chat_fn(final_prompt).strip()

    # v0.9 cited-note re-ranking.
    if len(gathered) > 1:
        from ..ranking import rank_candidates

        searchable = [
            (title, f"{title} {content[:2000]}")
            for title, content in gathered
        ]
        order = rank_candidates(query, searchable)
        cited = [cited[i] for i, _ in order]

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
