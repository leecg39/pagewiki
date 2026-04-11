"""Prompt templates for the multi-hop reasoning loop.

Each template is a pure function of its inputs — no side effects, no LLM
calls. This makes them trivially unit-testable and easy to iterate on.

Phase names mirror docs/ARCHITECTURE.md §4:
  1. ToC Review
  2. Select Node
  3. Extract & Evaluate
  4. Final Answer
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSummary:
    """Lightweight view of a tree node used when presenting options to the LLM.

    We avoid passing full TreeNode objects into prompts so this module has
    zero coupling to pydantic / tree.py.
    """

    node_id: str
    title: str
    kind: str  # folder | note | section
    summary: str = ""
    token_count: int | None = None
    linked_from: str | None = None  # e.g. "intro.md → [[Q3 Revenue]]"
    tags: list[str] | None = None  # frontmatter tags (v0.6)
    date: str | None = None  # frontmatter date (v0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: ToC Review + Phase 2: Select Node
# ─────────────────────────────────────────────────────────────────────────────

SELECT_NODE_SYSTEM = (
    "당신은 지식 베이스 탐색 전문가입니다. "
    "사용자의 질문에 답하기 위해 가장 관련 있는 노드를 정확히 하나 선택합니다. "
    "벡터 유사도가 아닌 논리적 추론에 기반해 판단하세요."
)


def select_node_prompt(
    query: str,
    candidates: list[NodeSummary],
    *,
    path_so_far: list[str] | None = None,
) -> str:
    """Build the prompt that asks the LLM to pick the next node to explore.

    The LLM must respond with a single line in the format:
        SELECT: <node_id>
    or
        DONE: <reason>   # if none of the candidates are relevant

    path_so_far shows the breadcrumb of already-visited nodes so the model
    doesn't loop back.
    """
    lines: list[str] = []
    lines.append(SELECT_NODE_SYSTEM)
    lines.append("")
    lines.append(f"[사용자 질문]\n{query}")
    lines.append("")

    if path_so_far:
        lines.append(f"[지금까지 탐색한 경로]\n{' > '.join(path_so_far)}")
        lines.append("")

    lines.append("[선택 가능한 노드]")
    for idx, cand in enumerate(candidates, start=1):
        meta = f"[{cand.kind}]"
        if cand.token_count is not None:
            meta += f" ~{cand.token_count}토큰"
        if cand.tags:
            meta += f" #{', #'.join(cand.tags)}"
        if cand.date:
            meta += f" ({cand.date})"
        if cand.linked_from:
            meta += f" [교차참조: {cand.linked_from}]"
        summary_part = f" — {cand.summary}" if cand.summary else ""
        lines.append(f"{idx}. {meta} {cand.title}{summary_part}")
        lines.append(f"   node_id: {cand.node_id}")
    lines.append("")

    lines.append(
        "위 후보 중 질문에 가장 관련 있는 노드 하나를 선택하세요.\n"
        "응답 형식은 반드시 아래 중 하나여야 합니다:\n"
        "  SELECT: <node_id>\n"
        "  DONE: <이유>  (관련 노드가 전혀 없을 때)\n"
        "다른 설명은 추가하지 마세요."
    )
    return "\n".join(lines)


def select_node_user_prompt(
    query: str,
    candidates: list[NodeSummary],
    *,
    path_so_far: list[str] | None = None,
) -> str:
    """Same as ``select_node_prompt`` but WITHOUT the system preamble (v0.14).

    Used by prompt-caching callers that pass ``SELECT_NODE_SYSTEM``
    as the ``system`` role to the LLM separately. Keeping the
    system portion stable across calls lets Ollama's KV cache reuse
    the prefix, which significantly speeds up repeat retrievals on
    large vaults.
    """
    lines: list[str] = []
    lines.append(f"[사용자 질문]\n{query}")
    lines.append("")
    if path_so_far:
        lines.append(f"[지금까지 탐색한 경로]\n{' > '.join(path_so_far)}")
        lines.append("")
    lines.append("[선택 가능한 노드]")
    for idx, cand in enumerate(candidates, start=1):
        meta = f"[{cand.kind}]"
        if cand.token_count is not None:
            meta += f" ~{cand.token_count}토큰"
        if cand.tags:
            meta += f" #{', #'.join(cand.tags)}"
        if cand.date:
            meta += f" ({cand.date})"
        if cand.linked_from:
            meta += f" [교차참조: {cand.linked_from}]"
        summary_part = f" — {cand.summary}" if cand.summary else ""
        lines.append(f"{idx}. {meta} {cand.title}{summary_part}")
        lines.append(f"   node_id: {cand.node_id}")
    lines.append("")
    lines.append(
        "위 후보 중 질문에 가장 관련 있는 노드 하나를 선택하세요.\n"
        "응답 형식은 반드시 아래 중 하나여야 합니다:\n"
        "  SELECT: <node_id>\n"
        "  DONE: <이유>  (관련 노드가 전혀 없을 때)\n"
        "다른 설명은 추가하지 마세요."
    )
    return "\n".join(lines)


def evaluate_user_prompt(query: str, note_title: str, note_content: str) -> str:
    """User-only variant of ``evaluate_prompt`` for prompt caching (v0.14)."""
    return (
        f"[사용자 질문]\n{query}\n\n"
        f"[로드된 노트: {note_title}]\n{note_content[:12000]}\n\n"
        "이 내용만으로 질문에 충분히 답할 수 있습니까?\n"
        "응답 형식:\n"
        "  SUFFICIENT: <한 줄 이유>\n"
        "  INSUFFICIENT: <부족한 정보>\n"
        "다른 설명은 추가하지 마세요."
    )


def final_answer_user_prompt(
    query: str,
    gathered_notes: list[tuple[str, str]],
) -> str:
    """User-only variant of ``final_answer_prompt`` for prompt caching (v0.14)."""
    lines = [f"[사용자 질문]\n{query}", "", "[근거 노트]"]
    for idx, (title, content) in enumerate(gathered_notes, start=1):
        lines.append(f"\n--- 노트 {idx}: {title} ---\n{content[:8000]}")
    lines.append("")
    lines.append(
        "위 근거를 바탕으로 질문에 한국어로 답변하세요. "
        "답변 끝에 참조한 노트 제목을 [[제목]] 형식으로 나열하세요."
    )
    return "\n".join(lines)


def parse_select_response(response: str) -> tuple[str, str]:
    """Parse the LLM's SELECT/DONE line.

    Returns:
        (action, value) where action ∈ {"SELECT", "DONE"}.

    Raises:
        ValueError: if the response doesn't match either format.
    """
    for line in response.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("SELECT:"):
            return "SELECT", stripped[len("SELECT:") :].strip()
        if stripped.startswith("DONE:"):
            return "DONE", stripped[len("DONE:") :].strip()
    raise ValueError(f"Could not parse SELECT/DONE from response: {response!r}")


def build_retry_prompt(original_prompt: str, error_reason: str) -> str:
    """Append a retry reminder to ``original_prompt`` for v0.8 parse retry.

    When the LLM produces unparseable output, we re-send the prompt
    with a terse note reiterating the format requirement. This is
    cheaper than a full rewrite and empirically recovers most
    Gemma 4 format slips.
    """
    return (
        original_prompt
        + "\n\n[재시도 요청] 이전 응답을 파싱할 수 없습니다 "
        + f"(오류: {error_reason}).\n"
        + "지정된 형식 외의 텍스트는 절대 포함하지 마세요."
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON-mode prompts (v0.10)
# ─────────────────────────────────────────────────────────────────────────────


def select_node_prompt_json(
    query: str,
    candidates: list[NodeSummary],
    *,
    path_so_far: list[str] | None = None,
) -> str:
    """Like ``select_node_prompt`` but asks the LLM to respond in JSON.

    Expected response (parsed by ``parse_select_response_json``)::

        {"action": "SELECT", "node_id": "Research/paper.md"}
        {"action": "DONE", "reason": "관련 노드 없음"}

    The JSON may be wrapped in ``json ... `` markdown code fences;
    the parser tolerates that. Non-JSON text outside the object is
    treated as noise and stripped before parsing.
    """
    lines = [
        SELECT_NODE_SYSTEM,
        "",
        "[JSON 모드]",
        "응답은 반드시 아래 두 형식 중 하나의 JSON 객체여야 합니다:",
        '  {"action": "SELECT", "node_id": "<node_id>"}',
        '  {"action": "DONE", "reason": "<이유>"}',
        "JSON 외의 텍스트, 설명, 마크다운 헤더는 출력하지 마세요.",
        "",
        f"[사용자 질문]\n{query}",
        "",
    ]
    if path_so_far:
        lines.append(f"[지금까지 탐색한 경로]\n{' > '.join(path_so_far)}")
        lines.append("")

    lines.append("[선택 가능한 노드]")
    for idx, cand in enumerate(candidates, start=1):
        meta = f"[{cand.kind}]"
        if cand.token_count is not None:
            meta += f" ~{cand.token_count}토큰"
        if cand.tags:
            meta += f" #{', #'.join(cand.tags)}"
        if cand.date:
            meta += f" ({cand.date})"
        if cand.linked_from:
            meta += f" [교차참조: {cand.linked_from}]"
        summary_part = f" — {cand.summary}" if cand.summary else ""
        lines.append(f"{idx}. {meta} {cand.title}{summary_part}")
        lines.append(f"   node_id: {cand.node_id}")
    lines.append("")
    lines.append("JSON 응답:")
    return "\n".join(lines)


def parse_select_response_json(response: str) -> tuple[str, str]:
    """Parse a JSON-mode SELECT/DONE response.

    Tolerates:
      * markdown code fences (```json ... ``` or ``` ... ```)
      * leading/trailing whitespace or commentary
      * either ``node_id`` or ``reason`` present but not both

    Raises:
        ValueError: if no valid JSON object matching the expected
            schema can be extracted.
    """
    import json
    import re

    # Strip code fences if present.
    text = response.strip()
    fence_re = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
    m = fence_re.search(text)
    if m:
        text = m.group(1).strip()

    # Try to locate the first JSON object in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {response!r}")

    snippet = text[start : end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e} in {snippet!r}") from e

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object, got {type(payload).__name__}")

    action = payload.get("action", "").upper()
    if action == "SELECT":
        node_id = payload.get("node_id") or payload.get("node") or ""
        if not node_id:
            raise ValueError("SELECT response missing node_id")
        return "SELECT", str(node_id).strip()

    if action == "DONE":
        reason = payload.get("reason") or payload.get("message") or ""
        return "DONE", str(reason).strip()

    raise ValueError(f"Unknown action: {action!r}")


def evaluate_prompt_json(query: str, note_title: str, note_content: str) -> str:
    """JSON-mode variant of ``evaluate_prompt``.

    Expected response::

        {"sufficient": true,  "reason": "한 줄 이유"}
        {"sufficient": false, "reason": "부족한 정보"}
    """
    return (
        f"{EVALUATE_SYSTEM}\n\n"
        f"[JSON 모드]\n"
        f"응답은 반드시 아래 형식의 JSON 객체여야 합니다:\n"
        f'  {{"sufficient": true, "reason": "<한 줄 이유>"}}\n'
        f'  {{"sufficient": false, "reason": "<부족한 정보>"}}\n'
        f"JSON 외의 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{query}\n\n"
        f"[로드된 노트: {note_title}]\n{note_content[:12000]}\n\n"
        f"JSON 응답:"
    )


def parse_evaluate_response_json(response: str) -> tuple[bool, str]:
    """Parse a JSON-mode SUFFICIENT/INSUFFICIENT response."""
    import json
    import re

    text = response.strip()
    fence_re = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
    m = fence_re.search(text)
    if m:
        text = m.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {response!r}")

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")

    if "sufficient" not in payload:
        raise ValueError("JSON missing 'sufficient' field")

    sufficient = bool(payload["sufficient"])
    reason = str(payload.get("reason", "")).strip()
    return sufficient, reason


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Extract & Evaluate
# ─────────────────────────────────────────────────────────────────────────────

EVALUATE_SYSTEM = (
    "당신은 지식 평가 전문가입니다. "
    "주어진 노트 내용만으로 사용자 질문에 충분히 답할 수 있는지 판단하세요."
)


def evaluate_prompt(query: str, note_title: str, note_content: str) -> str:
    """Ask the LLM whether the loaded note is sufficient to answer the query.

    The LLM must respond with a single line:
        SUFFICIENT: <one-line reason>
    or
        INSUFFICIENT: <what's missing>
    """
    return (
        f"{EVALUATE_SYSTEM}\n\n"
        f"[사용자 질문]\n{query}\n\n"
        f"[로드된 노트: {note_title}]\n{note_content[:12000]}\n\n"
        "이 내용만으로 질문에 충분히 답할 수 있습니까?\n"
        "응답 형식:\n"
        "  SUFFICIENT: <한 줄 이유>\n"
        "  INSUFFICIENT: <부족한 정보>\n"
        "다른 설명은 추가하지 마세요."
    )


def parse_evaluate_response(response: str) -> tuple[bool, str]:
    """Parse evaluation response into (is_sufficient, reason)."""
    for line in response.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("SUFFICIENT:"):
            return True, stripped[len("SUFFICIENT:") :].strip()
        if stripped.startswith("INSUFFICIENT:"):
            return False, stripped[len("INSUFFICIENT:") :].strip()
    raise ValueError(f"Could not parse SUFFICIENT/INSUFFICIENT: {response!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Final Answer
# ─────────────────────────────────────────────────────────────────────────────

FINAL_ANSWER_SYSTEM = (
    "당신은 연구 어시스턴트입니다. "
    "주어진 근거 노트들만 사용해 사용자 질문에 답변하세요. "
    "근거에 없는 내용은 추측하지 말고 '근거 부족'이라고 명시하세요."
)


def final_answer_prompt(
    query: str,
    gathered_notes: list[tuple[str, str]],
) -> str:
    """Build the final answer prompt.

    Args:
        query: Original user question.
        gathered_notes: List of (title, content) tuples collected during the loop.
    """
    lines = [FINAL_ANSWER_SYSTEM, "", f"[사용자 질문]\n{query}", ""]
    lines.append("[근거 노트]")
    for idx, (title, content) in enumerate(gathered_notes, start=1):
        lines.append(f"\n--- 노트 {idx}: {title} ---\n{content[:8000]}")
    lines.append("")
    lines.append(
        "위 근거를 바탕으로 질문에 한국어로 답변하세요. "
        "답변 끝에 참조한 노트 제목을 [[제목]] 형식으로 나열하세요."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic note summarization (used by vault layer)
# ─────────────────────────────────────────────────────────────────────────────


def atomic_summary_prompt(title: str, content: str) -> str:
    """One-line Korean summary prompt for an atomic note."""
    return (
        "다음 노트를 한국어 한 문장(100자 이내)으로 요약하세요. "
        "문장 외 다른 텍스트는 출력하지 마세요.\n\n"
        f"[제목] {title}\n\n"
        f"[본문]\n{content[:4000]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Long-note section summarization (PageIndex sub-tree nodes)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Multi-query decomposition (v0.7)
# ─────────────────────────────────────────────────────────────────────────────


def decompose_query_prompt(query: str, max_sub_queries: int = 4) -> str:
    """Build a prompt that decomposes a complex question into sub-queries.

    The LLM returns up to ``max_sub_queries`` independent questions
    that together cover the original. Simple questions return a
    single-line response.

    Response format (one sub-query per line):
        SUB: <sub-question>
        SUB: <sub-question>
        ...

    Or for simple questions:
        SINGLE
    """
    return (
        "당신은 질문 분해 전문가입니다.\n"
        f"복합 질문을 최대 {max_sub_queries}개의 독립적인 하위 질문으로 분해하세요.\n"
        "각 하위 질문은 서로 다른 정보를 찾아야 하며, 합쳐서 원래 질문에 답할 수 있어야 합니다.\n"
        "이미 단순한 질문이면 'SINGLE'만 출력하세요.\n\n"
        "응답 형식 (다른 텍스트 금지):\n"
        "  SUB: <하위 질문 1>\n"
        "  SUB: <하위 질문 2>\n"
        "  ...\n"
        "또는\n"
        "  SINGLE\n\n"
        f"[원래 질문]\n{query}"
    )


def parse_decompose_response(response: str) -> list[str]:
    """Parse the LLM's decomposition response.

    Returns an empty list if the response indicates the query is
    already simple (``SINGLE``), otherwise returns the list of
    extracted sub-questions.
    """
    sub_queries: list[str] = []
    for line in response.strip().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SINGLE"):
            return []
        if stripped.upper().startswith("SUB:"):
            sub = stripped[4:].strip()
            if sub:
                sub_queries.append(sub)
    return sub_queries


def synthesize_multi_answer_prompt(
    original_query: str,
    sub_qa_pairs: list[tuple[str, str]],
) -> str:
    """Build a prompt that synthesizes sub-query answers into a final response.

    Each ``(sub_query, sub_answer)`` pair is presented as context,
    and the LLM produces a single coherent answer to the original question.
    """
    lines = [
        "당신은 연구 어시스턴트입니다.",
        "여러 하위 질문의 답변을 종합해 원래 질문에 대한 최종 답변을 작성하세요.",
        "하위 답변에 있는 사실만 사용하고, 없는 내용은 추측하지 마세요.",
        "",
        f"[원래 질문]\n{original_query}",
        "",
        "[하위 질문별 답변]",
    ]
    for i, (sub_q, sub_a) in enumerate(sub_qa_pairs, start=1):
        lines.append(f"\n--- 하위 질문 {i}: {sub_q} ---")
        lines.append(sub_a[:3000])
    lines.append("")
    lines.append(
        "위 정보를 종합해 원래 질문에 한국어로 답변하세요. "
        "답변은 명확하고 구조적이어야 하며, 각 하위 답변에서 참조한 "
        "노트 제목([[제목]])을 그대로 인용하세요."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation context (v0.6 chat mode)
# ─────────────────────────────────────────────────────────────────────────────


def rewrite_query_with_context(
    new_query: str,
    history: list[tuple[str, str]],
) -> str:
    """Build a prompt that rewrites a follow-up question into a standalone query.

    The LLM sees the recent conversation history and the new message,
    then produces a self-contained rephrasing. If the new query is
    already standalone, the LLM returns it unchanged.

    Args:
        new_query: The latest user message.
        history: List of (question, answer_summary) from previous turns.
    """
    lines = [
        "당신은 질문 재작성 전문가입니다.",
        "대화 맥락을 참고해 후속 질문을 독립적인 질문으로 재작성하세요.",
        "이미 독립적인 질문이면 그대로 출력하세요.",
        "재작성된 질문만 출력하고 다른 설명은 추가하지 마세요.",
        "",
        "[대화 이력]",
    ]
    for i, (q, a) in enumerate(history[-3:], start=1):  # last 3 turns max
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a[:200]}")
    lines.append("")
    lines.append(f"[새 질문]\n{new_query}")
    lines.append("")
    lines.append("재작성된 독립 질문:")
    return "\n".join(lines)


def final_answer_with_history_prompt(
    query: str,
    gathered_notes: list[tuple[str, str]],
    history: list[tuple[str, str]],
) -> str:
    """Final answer prompt that includes conversation history for coherence.

    Same as ``final_answer_prompt`` but prepends recent history so the
    LLM can reference prior answers and avoid repetition.
    """
    lines = [FINAL_ANSWER_SYSTEM, ""]

    if history:
        lines.append("[이전 대화]")
        for i, (q, a) in enumerate(history[-3:], start=1):
            lines.append(f"Q{i}: {q}")
            lines.append(f"A{i}: {a[:300]}")
        lines.append("")

    lines.append(f"[사용자 질문]\n{query}")
    lines.append("")
    lines.append("[근거 노트]")
    for idx, (title, content) in enumerate(gathered_notes, start=1):
        lines.append(f"\n--- 노트 {idx}: {title} ---\n{content[:8000]}")
    lines.append("")
    lines.append(
        "위 근거와 이전 대화 맥락을 바탕으로 질문에 한국어로 답변하세요. "
        "답변 끝에 참조한 노트 제목을 [[제목]] 형식으로 나열하세요."
    )
    return "\n".join(lines)


def section_summary_prompt(note_title: str, section_title: str, section_text: str) -> str:
    """One-line Korean summary prompt for a single section of a long note.

    Used by the Layer 2 adapter when generating summaries for PageIndex
    sub-tree nodes so the retrieval loop can do ToC review at the section
    level.
    """
    return (
        "다음은 한 연구 노트의 한 섹션입니다. "
        "이 섹션의 핵심 주제를 한국어 한 문장(80자 이내)으로 요약하세요. "
        "문장 외 다른 텍스트는 출력하지 마세요.\n\n"
        f"[노트 제목] {note_title}\n"
        f"[섹션 제목] {section_title}\n\n"
        f"[섹션 본문]\n{section_text[:4000]}"
    )
