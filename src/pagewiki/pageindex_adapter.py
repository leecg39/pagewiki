"""Layer 2: PageIndex sub-tree builder for LONG-tier notes.

For every Layer 1 leaf with ``tier == LONG``, this module calls into the
vendored PageIndex markdown tree builder (``_vendor.pageindex``) and
grafts the resulting section hierarchy onto the leaf's ``children``.

Design choices (docs/ARCHITECTURE.md §0):

* **Local-only**: no cloud API. All summarization happens through the
  injected ``chat_fn``, which production wires to Ollama+Gemma via
  LiteLLM and tests wire to a scripted fake.
* **Vendored algorithm**: we do not reinvent PageIndex's tree building.
  We use the exact upstream logic for ``extract_nodes_from_markdown``,
  ``build_tree_from_nodes``, and ``tree_thinning_for_index``. See
  ``src/pagewiki/_vendor/pageindex/`` for license and attribution.
* **chat_fn DI**: summary generation is a separate pass after pure tree
  building so unit tests can call ``build_long_note_subtree`` with a
  fake ``chat_fn`` — or skip summaries entirely by omitting it.

Output schema translation
-------------------------

Upstream PageIndex nodes look like::

    {"title": ..., "node_id": "0003", "text": ...,
     "line_num": 42, "nodes": [...]}

pagewiki nodes use ``children`` instead of ``nodes`` and carry a
``kind="section"`` tag plus a ``line_range`` for partial-read retrieval.
The per-section ``text`` field is dropped (it can be re-read from the
file on demand) — we only keep the title, summary, and line range.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ._vendor.pageindex import (
    build_tree_from_nodes,
    extract_node_text_content,
    extract_nodes_from_markdown,
    tree_thinning_for_index,
    update_node_list_with_text_token_count,
)
from .prompts import section_summary_prompt
from .tree import TreeNode

ChatFn = Callable[[str], str]

# When a section has fewer than this many tokens, skip the LLM summary
# and use a truncated copy of the section text as its own summary. This
# mirrors the upstream ``summary_token_threshold`` default and keeps
# micro-sections cheap.
DEFAULT_SUMMARY_TOKEN_THRESHOLD = 200

# When thinning is enabled, sections under this token count get merged
# into their parent. 500 matches our MICRO tier boundary so very small
# sub-sections do not inflate the tree depth.
DEFAULT_THINNING_MIN_TOKENS = 500


def build_long_note_subtree(
    note_path: Path,
    note_title: str,
    *,
    chat_fn: ChatFn | None = None,
    if_thinning: bool = True,
    thinning_min_tokens: int = DEFAULT_THINNING_MIN_TOKENS,
    summary_token_threshold: int = DEFAULT_SUMMARY_TOKEN_THRESHOLD,
) -> list[TreeNode]:
    """Build a PageIndex-style section tree for one LONG markdown note.

    Args:
        note_path: Absolute path to the ``.md`` file.
        note_title: The Layer 1 note title (used to contextualize section
            summaries — "section X of note Y").
        chat_fn: Optional LLM callable. When provided, every section
            whose text exceeds ``summary_token_threshold`` gets a
            one-line summary. When ``None`` (e.g. in unit tests or when
            ``--skip-summaries`` is set), sections fall back to their
            own truncated text as a placeholder summary.
        if_thinning: Merge sub-sections smaller than
            ``thinning_min_tokens`` into their parent to keep the tree
            shallow. Enabled by default because Obsidian notes often
            have many tiny headings.
        thinning_min_tokens: Token threshold for thinning.
        summary_token_threshold: Sections below this skip the LLM call.

    Returns:
        A list of ``TreeNode`` objects in pagewiki's schema, ready to
        assign to the parent LONG note's ``children`` field.
    """
    markdown_content = note_path.read_text(encoding="utf-8")

    # Phase A: pure-python tree extraction (no LLM).
    raw_nodes, lines = extract_nodes_from_markdown(markdown_content)
    if not raw_nodes:
        # No headings in the note — nothing to split on. Return empty so
        # the retrieval loop falls back to loading the whole note.
        return []

    nodes_with_text = extract_node_text_content(raw_nodes, lines)

    if if_thinning:
        nodes_with_text = update_node_list_with_text_token_count(nodes_with_text)
        nodes_with_text = tree_thinning_for_index(
            nodes_with_text, min_node_token=thinning_min_tokens
        )

    raw_tree = build_tree_from_nodes(nodes_with_text)
    total_lines = len(lines)

    # Phase B: translate upstream dict nodes into pagewiki TreeNode,
    # optionally calling chat_fn to generate per-section summaries.
    return [
        _to_tree_node(
            raw,
            note_title=note_title,
            note_path=note_path,
            total_lines=total_lines,
            next_line_num=None,
            chat_fn=chat_fn,
            summary_token_threshold=summary_token_threshold,
        )
        for raw in _with_end_line(raw_tree, total_lines)
    ]


def _with_end_line(
    nodes: list[dict],
    total_lines: int,
) -> list[tuple[dict, int]]:
    """Pair each raw node with the line number where the next sibling
    starts (or ``total_lines + 1`` for the last sibling). Used so that
    translated ``TreeNode`` instances carry an exclusive ``line_range``
    end for partial-read retrieval.
    """
    result: list[tuple[dict, int]] = []
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            next_start = nodes[i + 1]["line_num"]
        else:
            next_start = total_lines + 1
        result.append((node, next_start))
    return result


def _to_tree_node(
    raw: dict | tuple[dict, int],
    *,
    note_title: str,
    note_path: Path,
    total_lines: int,
    next_line_num: int | None,
    chat_fn: ChatFn | None,
    summary_token_threshold: int,
) -> TreeNode:
    """Convert an upstream PageIndex dict into a pagewiki ``TreeNode``.

    Accepts either a raw dict (then ``next_line_num`` supplies the end
    of the line range) or a ``(dict, next_line_num)`` tuple produced by
    ``_with_end_line``. The tuple form is used at the top level where we
    enumerate siblings in a loop.
    """
    if isinstance(raw, tuple):
        raw, next_line_num = raw

    start_line = int(raw["line_num"])
    end_line = next_line_num if next_line_num is not None else total_lines + 1

    section_text = raw.get("text", "")
    summary = _generate_section_summary(
        note_title=note_title,
        section_title=raw["title"],
        section_text=section_text,
        chat_fn=chat_fn,
        summary_token_threshold=summary_token_threshold,
    )

    # Stable node id: "<note_rel_path>#<upstream_node_id>" would be ideal
    # but we don't have the vault root here. The upstream zfill id is
    # unique within the note, which is enough for visited_ids tracking
    # in the retrieval loop (scoped per query).
    node_id = f"{note_path.name}#{raw['node_id']}"

    child_raw = raw.get("nodes") or []
    paired_children = _with_end_line(child_raw, end_line - 1)
    children = [
        _to_tree_node(
            pair,
            note_title=note_title,
            note_path=note_path,
            total_lines=total_lines,
            next_line_num=None,
            chat_fn=chat_fn,
            summary_token_threshold=summary_token_threshold,
        )
        for pair in paired_children
    ]

    return TreeNode(
        node_id=node_id,
        title=raw["title"],
        summary=summary,
        kind="section",
        file_path=note_path,
        token_count=_approx_tokens(section_text),
        line_range=(start_line, end_line),
        children=children,
    )


def _generate_section_summary(
    *,
    note_title: str,
    section_title: str,
    section_text: str,
    chat_fn: ChatFn | None,
    summary_token_threshold: int,
) -> str:
    """Produce a one-line summary for a section.

    Short sections (below ``summary_token_threshold``) or calls made
    without a ``chat_fn`` use a truncated copy of the section text. The
    LLM is only invoked when both a callable is provided and the
    section is large enough to warrant abstraction.
    """
    if not section_text:
        return ""

    tokens = _approx_tokens(section_text)
    if chat_fn is None or tokens < summary_token_threshold:
        return _truncate_for_summary(section_text)

    prompt = section_summary_prompt(note_title, section_title, section_text)
    raw = chat_fn(prompt).strip().strip("\"'").strip()
    # Defensive: an empty LLM response should still produce _something_
    # so the ToC review phase has a handle on the section.
    return raw or _truncate_for_summary(section_text)


def _truncate_for_summary(text: str, max_chars: int = 160) -> str:
    """Take the first non-empty *body* line as a cheap inline summary.

    Used when the LLM is not called. Lines that are markdown headings
    (``#``, ``##``, …) are skipped because the section title already
    carries that information — surfacing it again in the summary would
    be redundant noise in ToC review prompts.
    """
    first_line = next(
        (
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        "",
    )
    if len(first_line) > max_chars:
        return first_line[:max_chars].rstrip() + "…"
    return first_line


def _approx_tokens(text: str) -> int:
    """Matches ``pagewiki.vault._CHARS_PER_TOKEN`` (char/3 heuristic)."""
    if not text:
        return 0
    return max(1, len(text) // 3)
