"""v1.1 English prompt variants.

Mirrors the Korean prompts in ``prompts.py`` but substitutes every
instruction and format marker with an English equivalent. Used when
the user passes ``--lang en`` to ``ask`` or ``chat``.

Design notes
------------

* The **format markers** (``SELECT:``, ``DONE:``, ``SUFFICIENT:``,
  ``INSUFFICIENT:``, ``SUB:``, ``SINGLE``) are kept identical to the
  Korean version so the existing ``parse_*`` helpers still work.
  Only the surrounding instructional text is translated.
* Each English builder uses ``EN_``-prefixed system constants so a
  bilingual ``system_chat_fn`` caller can switch between them
  without touching ``prompts.py`` internals.
* JSON-mode variants reuse the same JSON schema, so only the
  human instruction lines are rewritten.
"""

from __future__ import annotations

from .prompts import NodeSummary

# ─────────────────────────────────────────────────────────────────────────────
# English system constants (parallel to SELECT_NODE_SYSTEM etc.)
# ─────────────────────────────────────────────────────────────────────────────

EN_SELECT_NODE_SYSTEM = (
    "You are a knowledge-base navigation expert. "
    "Select exactly one node that is most likely to answer the user's question. "
    "Rely on logical reasoning over the tree structure, not vector similarity."
)

EN_EVALUATE_SYSTEM = (
    "You are a knowledge-sufficiency evaluator. "
    "Decide whether the given note alone is enough to answer the question."
)

EN_FINAL_ANSWER_SYSTEM = (
    "You are a research assistant. "
    "Answer the user's question using ONLY the provided evidence notes. "
    "If a claim isn't in the evidence, say so instead of inventing one."
)


# ─────────────────────────────────────────────────────────────────────────────
# English prompt builders
# ─────────────────────────────────────────────────────────────────────────────


def select_node_prompt_en(
    query: str,
    candidates: list[NodeSummary],
    *,
    path_so_far: list[str] | None = None,
) -> str:
    """English-language select-node prompt."""
    lines: list[str] = []
    lines.append(EN_SELECT_NODE_SYSTEM)
    lines.append("")
    lines.append(f"[User question]\n{query}")
    lines.append("")
    if path_so_far:
        lines.append(f"[Path so far]\n{' > '.join(path_so_far)}")
        lines.append("")
    lines.append("[Candidate nodes]")
    for idx, cand in enumerate(candidates, start=1):
        meta = f"[{cand.kind}]"
        if cand.token_count is not None:
            meta += f" ~{cand.token_count} tokens"
        if cand.tags:
            meta += f" #{', #'.join(cand.tags)}"
        if cand.date:
            meta += f" ({cand.date})"
        if cand.linked_from:
            meta += f" [x-ref: {cand.linked_from}]"
        summary_part = f" — {cand.summary}" if cand.summary else ""
        lines.append(f"{idx}. {meta} {cand.title}{summary_part}")
        lines.append(f"   node_id: {cand.node_id}")
    lines.append("")
    lines.append(
        "Pick the single most relevant candidate.\n"
        "Respond with EXACTLY one of these lines (no other text):\n"
        "  SELECT: <node_id>\n"
        "  DONE: <reason>   (when none of the candidates are relevant)"
    )
    return "\n".join(lines)


def evaluate_prompt_en(query: str, note_title: str, note_content: str) -> str:
    """English-language evaluate prompt."""
    return (
        f"{EN_EVALUATE_SYSTEM}\n\n"
        f"[User question]\n{query}\n\n"
        f"[Loaded note: {note_title}]\n{note_content[:12000]}\n\n"
        "Is this note sufficient to answer the question?\n"
        "Respond with EXACTLY one line:\n"
        "  SUFFICIENT: <one-line reason>\n"
        "  INSUFFICIENT: <what's missing>"
    )


def final_answer_prompt_en(
    query: str,
    gathered_notes: list[tuple[str, str]],
) -> str:
    """English-language final-answer prompt."""
    lines = [EN_FINAL_ANSWER_SYSTEM, "", f"[User question]\n{query}", ""]
    lines.append("[Evidence notes]")
    for idx, (title, content) in enumerate(gathered_notes, start=1):
        lines.append(f"\n--- Note {idx}: {title} ---\n{content[:8000]}")
    lines.append("")
    lines.append(
        "Answer the question in English using only the evidence above. "
        "End your answer with a list of cited note titles in [[Title]] format."
    )
    return "\n".join(lines)


def atomic_summary_prompt_en(title: str, content: str) -> str:
    """English one-line summary prompt for atomic notes."""
    return (
        "Summarize the following note in ONE English sentence (under 120 chars). "
        "Output only the sentence, no preamble.\n\n"
        f"[Title] {title}\n\n"
        f"[Body]\n{content[:4000]}"
    )


def decompose_query_prompt_en(query: str, max_sub_queries: int = 4) -> str:
    """English multi-query decomposition prompt."""
    return (
        "You are a query decomposition expert.\n"
        f"Break a complex question into at most {max_sub_queries} independent "
        "sub-questions. Each sub-question should seek different information, "
        "and together they should cover the original.\n"
        "If the question is already simple, output 'SINGLE' alone.\n\n"
        "Response format (no other text):\n"
        "  SUB: <sub-question 1>\n"
        "  SUB: <sub-question 2>\n"
        "  ...\n"
        "or\n"
        "  SINGLE\n\n"
        f"[Original question]\n{query}"
    )


def synthesize_multi_answer_prompt_en(
    original_query: str,
    sub_qa_pairs: list[tuple[str, str]],
) -> str:
    """English synthesis prompt for multi-query results."""
    lines = [
        "You are a research assistant.",
        "Combine the sub-question answers below into one cohesive final answer.",
        "Use only facts present in the sub-answers; do not invent anything.",
        "",
        f"[Original question]\n{original_query}",
        "",
        "[Sub-question answers]",
    ]
    for i, (sub_q, sub_a) in enumerate(sub_qa_pairs, start=1):
        lines.append(f"\n--- Sub-question {i}: {sub_q} ---")
        lines.append(sub_a[:3000])
    lines.append("")
    lines.append(
        "Synthesize a single English answer. Preserve any [[Title]] "
        "citations from the sub-answers exactly as they appear."
    )
    return "\n".join(lines)
