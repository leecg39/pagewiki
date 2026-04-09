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

# v0.1.3: synthetic "(intro)" section created when the h1-flatten
# optimization drops a wrapping h1 whose body carries non-empty text.
# Preserving the text prevents abstracts / TL;DR paragraphs directly
# under the title line from silently disappearing.
INTRO_SECTION_TITLE = "(intro)"
INTRO_SECTION_NODE_ID_SUFFIX = "intro"


def build_long_note_subtree(
    note_path: Path,
    note_title: str,
    *,
    chat_fn: ChatFn | None = None,
    if_thinning: bool = True,
    thinning_min_tokens: int = DEFAULT_THINNING_MIN_TOKENS,
    summary_token_threshold: int = DEFAULT_SUMMARY_TOKEN_THRESHOLD,
    node_id_prefix: str | None = None,
    flatten_matching_h1: bool = True,
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
        node_id_prefix: String used to namespace section ``node_id`` s
            so they are globally unique within a vault. Production
            callers (``vault.build_long_subtrees``) pass the vault-
            relative path, e.g. ``"Research/paper.md"``, so two notes
            with the same filename in different folders produce
            distinct ids like ``Research/paper.md#0002`` vs
            ``Archive/paper.md#0002``. When ``None`` (the default used
            by direct unit-test callers), falls back to the bare
            filename — safe for tests but **unsafe for real vaults**
            because the retrieval ``visited_ids`` set would silently
            dedupe sections from same-named files and drop evidence.
            (v0.1.3 — fixes PR #1 review comment.)
        flatten_matching_h1: When ``True`` (default), notes that are
            wrapped in a single h1 whose title matches ``note_title``
            have that h1 promoted out — retrieval sees the h2 children
            directly without a redundant one-choice ToC step. Any body
            text that was directly under the h1 (abstract, TL;DR) is
            preserved as a synthetic ``(intro)`` section so no content
            is dropped. (v0.1.3 addition.)

    Returns:
        A list of ``TreeNode`` objects in pagewiki's schema, ready to
        assign to the parent LONG note's ``children`` field.
    """
    markdown_content = note_path.read_text(encoding="utf-8")

    # Fall back to the bare filename when no prefix is provided — this
    # preserves v0.1.2 behavior for unit tests that call the adapter
    # directly. Production callers (vault.build_long_subtrees) MUST
    # pass the vault-relative path.
    effective_prefix = node_id_prefix if node_id_prefix is not None else note_path.name

    # Phase A: pure-python tree extraction (no LLM).
    raw_nodes, lines = extract_nodes_from_markdown(markdown_content)
    if not raw_nodes:
        # No headings in the note — nothing to split on. Return empty so
        # the retrieval loop falls back to loading the whole note.
        return []

    nodes_with_text = extract_node_text_content(raw_nodes, lines)

    # Detect the flatten opportunity against the *pre-thinning* enriched
    # list so any intro body text can be captured before thinning
    # potentially folds it into a parent. Application of the flatten
    # waits until after the tree is built so all preconditions
    # (single root, title match, non-empty children) can be re-checked
    # on the final TreeNode structure.
    flatten_info: tuple[str, int, int] | None = None
    if flatten_matching_h1:
        flatten_info = _extract_intro_from_root(nodes_with_text, note_title)

    if if_thinning:
        nodes_with_text = update_node_list_with_text_token_count(nodes_with_text)
        nodes_with_text = tree_thinning_for_index(
            nodes_with_text, min_node_token=thinning_min_tokens
        )

    raw_tree = build_tree_from_nodes(nodes_with_text)
    total_lines = len(lines)

    # Phase B: translate upstream dict nodes into pagewiki TreeNode,
    # optionally calling chat_fn to generate per-section summaries.
    children = [
        _to_tree_node(
            raw,
            note_title=note_title,
            note_path=note_path,
            total_lines=total_lines,
            next_line_num=None,
            chat_fn=chat_fn,
            summary_token_threshold=summary_token_threshold,
            node_id_prefix=effective_prefix,
        )
        for raw in _with_end_line(raw_tree, total_lines)
    ]

    # Phase C (v0.1.3): if the tree ended up as a single h1 wrapper
    # whose title matches the note, promote its children to top level.
    if flatten_matching_h1:
        children = _apply_h1_flatten(
            children,
            note_path=note_path,
            note_title=note_title,
            flatten_info=flatten_info,
            chat_fn=chat_fn,
            node_id_prefix=effective_prefix,
            summary_token_threshold=summary_token_threshold,
        )

    return children


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
    node_id_prefix: str,
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

    # Namespace section ids by the caller-supplied prefix (typically
    # the vault-relative path) so two same-named notes in different
    # folders don't collide in the retrieval ``visited_ids`` set.
    # Fixes PR #1 review comment (v0.1.3).
    node_id = f"{node_id_prefix}#{raw['node_id']}"

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
            node_id_prefix=node_id_prefix,
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


# ─────────────────────────────────────────────────────────────────────────────
# h1-title flatten optimization (v0.1.3)
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_title(title: str) -> str:
    """Normalize a title for fuzzy equality: strip + lowercase + collapse ws."""
    return " ".join(title.strip().lower().split())


def _titles_match(a: str, b: str) -> bool:
    """Case- and whitespace-insensitive title comparison."""
    return _normalize_title(a) == _normalize_title(b)


def _extract_intro_from_root(
    enriched: list[dict],
    note_title: str,
) -> tuple[str, int, int] | None:
    """Return ``(intro_text, h1_line, first_child_line)`` if flatten should apply.

    Flatten fires when:
      * the enriched list is non-empty,
      * the first node is an h1,
      * its title matches ``note_title`` (normalized),
      * it is the ONLY top-level h1,
      * it has at least one deeper descendant (otherwise there's
        nothing to promote to the top level).

    Returns ``None`` in any failing case, meaning "do not flatten".
    """
    if not enriched:
        return None

    root = enriched[0]
    if root.get("level") != 1:
        return None
    if not _titles_match(root.get("title", ""), note_title):
        return None

    # ``root`` must be the only top-level h1 — two coequal h1s would be
    # ambiguous and dropping one of them would lose sibling content.
    for other in enriched[1:]:
        if other.get("level") == 1:
            return None

    # The first descendant (level > 1) marks where the intro body ends.
    first_child_line: int | None = None
    for node in enriched[1:]:
        if node.get("level", 0) > 1:
            first_child_line = node["line_num"]
            break
    if first_child_line is None:
        # h1 with no nested headings — nothing to flatten into.
        return None

    # Strip the leading "# Title" header out of the root's body.
    raw_text = root.get("text", "")
    body_lines = raw_text.split("\n")
    if body_lines and body_lines[0].lstrip().startswith("#"):
        body_lines = body_lines[1:]
    intro_text = "\n".join(body_lines).strip()

    return (intro_text, root["line_num"], first_child_line)


def _apply_h1_flatten(
    children: list[TreeNode],
    *,
    note_path: Path,
    note_title: str,
    flatten_info: tuple[str, int, int] | None,
    chat_fn: ChatFn | None,
    node_id_prefix: str,
    summary_token_threshold: int,
) -> list[TreeNode]:
    """Promote a matching single-h1 root's children to the top level.

    All preconditions are re-checked against the final ``TreeNode``
    structure (thinning may have collapsed the tree further than the
    pre-check assumed). If any fails, ``children`` is returned
    unchanged.
    """
    if flatten_info is None:
        return children
    if len(children) != 1:
        return children

    root = children[0]
    if root.kind != "section":
        return children
    if not _titles_match(root.title, note_title):
        return children
    if not root.children:
        # Thinning folded everything into the root — flattening would
        # yield an empty tree. Keep the root as-is.
        return children

    promoted: list[TreeNode] = list(root.children)

    intro_text, h1_line, first_child_line = flatten_info
    if intro_text:
        intro_node = _make_intro_section(
            note_path=note_path,
            note_title=note_title,
            intro_text=intro_text,
            start_line=h1_line,
            end_line=first_child_line,
            chat_fn=chat_fn,
            node_id_prefix=node_id_prefix,
            summary_token_threshold=summary_token_threshold,
        )
        promoted = [intro_node, *promoted]

    return promoted


def _make_intro_section(
    *,
    note_path: Path,
    note_title: str,
    intro_text: str,
    start_line: int,
    end_line: int,
    chat_fn: ChatFn | None,
    node_id_prefix: str,
    summary_token_threshold: int,
) -> TreeNode:
    """Build the synthetic ``(intro)`` section preserving h1 body text.

    Its ``line_range`` spans from the h1 header line (inclusive) to
    the first descendant header line (exclusive) so the retrieval
    loader reads exactly the intro slice — no more, no less.
    """
    summary = _generate_section_summary(
        note_title=note_title,
        section_title=INTRO_SECTION_TITLE,
        section_text=intro_text,
        chat_fn=chat_fn,
        summary_token_threshold=summary_token_threshold,
    )

    return TreeNode(
        node_id=f"{node_id_prefix}#{INTRO_SECTION_NODE_ID_SUFFIX}",
        title=INTRO_SECTION_TITLE,
        summary=summary,
        kind="section",
        file_path=note_path,
        token_count=_approx_tokens(intro_text),
        line_range=(start_line, end_line),
        children=[],
    )
