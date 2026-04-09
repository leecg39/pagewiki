"""Layer 1.5: `[[wiki-link]]` cross-reference index for Obsidian vaults.

This module is the v0.1.4 scaffolding for the v0.2 retrieval-loop
integration that will let ``run_retrieval`` follow outgoing links
when the current note references another note the user's question
is indirectly about.

Design decisions (see ``docs/v0.2-design.md`` §6 for full context):

  * Q1 — title ambiguity: ``resolve(target)`` returns *every* note
    whose normalized title matches the link target. Callers
    (specifically the v0.2 retrieval loop) decide what to do with
    multiple matches.
  * Q2 — section anchors: ``[[Alpha#Methods]]`` uses best-match
    fallback — try to walk ``Alpha.md``'s Layer 2 section tree and
    find a section whose normalized title equals ``"Methods"``; on
    miss, fall back to the note root.
  * Q3 — release shape: v0.1.4 ships this module and
    ``scan --show-graph`` only. The retrieval loop is intentionally
    untouched until v0.2.

What this module does
---------------------

``build_link_index(root)`` walks a Layer 1 tree (from
``vault.scan_folder``), re-reads each note's markdown body, extracts
every ``[[target]]`` / ``[[target|alias]]`` / ``[[target#anchor]]``
reference with its anchor preserved, and builds four lookup tables:

  * ``_notes_by_normalized_title`` — for ``resolve``
  * ``_outgoing``                  — per-source link list
  * ``_backlinks``                 — per-target link list (reverse)
  * ``_dangling``                  — unresolved link list

Why re-read the files
---------------------

``vault.extract_wiki_links`` (v0.1) deliberately strips anchors and
aliases, so ``TreeNode.wiki_links`` is a flat list of target strings.
The Q2 decision requires anchor information, so the index has to
re-parse the markdown with a richer regex. We accept the extra I/O
because link-index construction runs once per scan and is already
bounded by the Layer 2 sub-tree build.

Test injection
--------------

``build_link_index(root, reader=...)`` accepts a ``reader`` callable
for unit tests that don't want to touch the filesystem. Production
defaults to ``Path.read_text(encoding="utf-8")``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ._text import normalize_title
from .tree import TreeNode

# Richer than ``vault._WIKI_LINK_RE`` — this one captures the anchor
# and alias as separate groups so the index can honor Q2's best-match
# section-anchor fallback.
#
# Pattern breakdown:
#   \[\[              opening [[
#   ([^\]|#]+)        group 1: target ("Alpha") — no ], |, or #
#   (?:#([^\]|]+))?   group 2: optional anchor after #
#   (?:\|[^\]]*)?     optional alias after |, not captured
#   \]\]              closing ]]
_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|[^\]]*)?\]\]")


@dataclass
class ResolvedLink:
    """One resolved ``[[target]]`` reference.

    ``target`` is the note (or section, when the anchor hit) that the
    link points to after Q2's best-match fallback has been applied.
    ``raw_target`` and ``section_anchor`` preserve the original parse
    so retrieval / logging can show the user what was written.
    """

    source: TreeNode
    target: TreeNode
    raw_target: str
    section_anchor: str | None = None


@dataclass
class LinkIndexStats:
    """Summary statistics printed by ``scan --show-graph``."""

    total_links: int = 0
    dangling_count: int = 0
    ambiguous_links: int = 0
    top_linked_to: list[tuple[str, int]] = field(default_factory=list)
    top_outgoing: list[tuple[str, int]] = field(default_factory=list)


class LinkIndex:
    """Built once per scan; read many times during retrieval / reporting.

    The internal dictionaries are intentionally immutable after
    ``build_link_index`` returns — this class has no mutator methods.
    Tests can stub file I/O by passing a custom ``reader`` to
    ``build_link_index``.
    """

    def __init__(
        self,
        *,
        notes_by_normalized_title: dict[str, list[TreeNode]],
        outgoing: dict[str, list[ResolvedLink]],
        backlinks: dict[str, list[ResolvedLink]],
        dangling: list[tuple[str, str]],
        stats: LinkIndexStats,
    ) -> None:
        self._notes_by_normalized_title = notes_by_normalized_title
        self._outgoing = outgoing
        self._backlinks = backlinks
        self._dangling = dangling
        self._stats = stats

    def resolve(self, target: str) -> list[TreeNode]:
        """Return every note whose normalized title matches ``target``.

        Per Q1 decision, this returns **all** matches — the caller
        (the v0.2 retrieval loop) decides what to do with multiple
        hits. Empty list on unresolved target.
        """
        normalized = normalize_title(target)
        return list(self._notes_by_normalized_title.get(normalized, []))

    def outgoing(self, node_id: str) -> list[ResolvedLink]:
        """Return every link whose source is the note with ``node_id``."""
        return list(self._outgoing.get(node_id, []))

    def backlinks(self, node_id: str) -> list[ResolvedLink]:
        """Return every link whose target is the note with ``node_id``.

        For links that resolved to a *section* under ``node_id`` via
        Q2's anchor fallback, the backlink is indexed under the
        section's node_id, not the parent note's. Callers that want
        "all backlinks to this note, including section-anchor hits"
        should walk the note's sub-tree and union the results.
        """
        return list(self._backlinks.get(node_id, []))

    def dangling(self) -> list[tuple[str, str]]:
        """Return every ``(source_node_id, raw_target)`` pair where the
        target could not be resolved to any note in the vault."""
        return list(self._dangling)

    def stats(self) -> LinkIndexStats:
        """Return a snapshot of link-graph summary statistics."""
        return self._stats

    def __len__(self) -> int:
        """Total number of resolved links in the index."""
        return sum(len(v) for v in self._outgoing.values())


def _extract_links_with_anchors(text: str) -> list[tuple[str, str | None]]:
    """Run the richer regex over a body and return ``(target, anchor)`` tuples.

    Separated from the index builder so unit tests can exercise the
    regex in isolation.
    """
    results: list[tuple[str, str | None]] = []
    for match in _LINK_RE.finditer(text):
        target = match.group(1).strip()
        anchor_raw = match.group(2)
        anchor = anchor_raw.strip() if anchor_raw else None
        if target:
            results.append((target, anchor or None))
    return results


def _apply_anchor_fallback(note: TreeNode, anchor: str | None) -> TreeNode:
    """Q2: if ``anchor`` matches a section title in ``note`` 's Layer 2
    tree, return that section node; otherwise fall back to ``note``.

    The search walks every descendant, not just direct children, so
    ``[[Paper#Training]]`` resolves to the ``Training`` h3 even when
    it is nested under a ``Methods`` h2.
    """
    if anchor is None or not note.children:
        return note

    normalized_anchor = normalize_title(anchor)
    for descendant in note.walk():
        if descendant is note:
            # walk() yields self first; skip it so we match children only.
            continue
        if descendant.kind != "section":
            continue
        if normalize_title(descendant.title) == normalized_anchor:
            return descendant
    return note  # best-match fallback on miss


def _collect_notes(root: TreeNode) -> list[TreeNode]:
    """Depth-first collect every ``kind == "note"`` under ``root``."""
    return [n for n in root.walk() if n.kind == "note"]


def _build_title_index(
    notes: list[TreeNode],
) -> dict[str, list[TreeNode]]:
    """Build the normalized-title → [notes...] lookup table."""
    index: dict[str, list[TreeNode]] = defaultdict(list)
    for note in notes:
        key = normalize_title(note.title)
        index[key].append(note)
    return dict(index)


def _compute_stats(
    notes: list[TreeNode],
    outgoing: dict[str, list[ResolvedLink]],
    backlinks: dict[str, list[ResolvedLink]],
    dangling: list[tuple[str, str]],
    title_index: dict[str, list[TreeNode]],
    *,
    top_n: int = 10,
) -> LinkIndexStats:
    """Aggregate per-note counts for the ``scan --show-graph`` table."""
    total_links = sum(len(v) for v in outgoing.values())

    # "Ambiguous links" counts each source→raw_target pair where the
    # title_index had more than one candidate note. A link that
    # resolved against a 2-way tie contributes 2 entries to outgoing
    # (one per candidate), so divide by 1 (we want source×raw_target
    # uniqueness). Count via set-tracking on the source side.
    ambiguous_pairs: set[tuple[str, str]] = set()
    for source_id, links in outgoing.items():
        for link in links:
            normalized = normalize_title(link.raw_target)
            candidates = title_index.get(normalized, [])
            if len(candidates) > 1:
                ambiguous_pairs.add((source_id, normalized))

    # Top linked-to: count backlinks per target *note title*, not per
    # section node_id, so section-anchor hits and note-root hits
    # aggregate under the same parent. Walk backlinks and attribute
    # each one to its target note's title.
    incoming_per_title: dict[str, int] = defaultdict(int)
    for target_id, links in backlinks.items():
        if not links:
            continue
        # Take the title from the first link's target; all entries
        # under this key share the same target node by construction.
        title = links[0].target.title
        incoming_per_title[title] += len(links)
    top_linked_to = sorted(
        incoming_per_title.items(), key=lambda kv: (-kv[1], kv[0])
    )[:top_n]

    outgoing_per_title: dict[str, int] = defaultdict(int)
    for source_id, links in outgoing.items():
        if not links:
            continue
        title = links[0].source.title
        outgoing_per_title[title] += len(links)
    top_outgoing = sorted(
        outgoing_per_title.items(), key=lambda kv: (-kv[1], kv[0])
    )[:top_n]

    return LinkIndexStats(
        total_links=total_links,
        dangling_count=len(dangling),
        ambiguous_links=len(ambiguous_pairs),
        top_linked_to=top_linked_to,
        top_outgoing=top_outgoing,
    )


def build_link_index(
    root: TreeNode,
    *,
    reader: Callable[[Path], str] | None = None,
) -> LinkIndex:
    """Walk ``root``, parse every note's body, and build a ``LinkIndex``.

    Args:
        root: Layer 1 tree root returned by ``vault.scan_folder``.
            Both MICRO / ATOMIC / LONG notes are scanned — link
            resolution is tier-agnostic.
        reader: Optional ``Path -> str`` callable for test injection.
            Defaults to ``Path.read_text(encoding="utf-8")``. Notes
            whose ``file_path`` is ``None`` or whose read raises
            ``OSError`` are skipped silently; their content simply
            contributes nothing to the index.

    Returns:
        A ``LinkIndex`` whose lookup methods are safe to call from
        the retrieval loop. Construction does not mutate ``root``.
    """
    if reader is None:

        def _default_reader(path: Path) -> str:
            return path.read_text(encoding="utf-8")

        reader = _default_reader

    notes = _collect_notes(root)
    title_index = _build_title_index(notes)

    outgoing: dict[str, list[ResolvedLink]] = defaultdict(list)
    backlinks: dict[str, list[ResolvedLink]] = defaultdict(list)
    dangling: list[tuple[str, str]] = []

    for note in notes:
        if note.file_path is None:
            continue
        try:
            text = reader(note.file_path)
        except OSError:
            continue

        for raw_target, anchor in _extract_links_with_anchors(text):
            normalized = normalize_title(raw_target)
            matches = title_index.get(normalized, [])

            if not matches:
                dangling.append((note.node_id, raw_target))
                continue

            for target_note in matches:
                final_target = _apply_anchor_fallback(target_note, anchor)
                link = ResolvedLink(
                    source=note,
                    target=final_target,
                    raw_target=raw_target,
                    section_anchor=anchor,
                )
                outgoing[note.node_id].append(link)
                backlinks[final_target.node_id].append(link)

    stats = _compute_stats(
        notes,
        dict(outgoing),
        dict(backlinks),
        dangling,
        title_index,
    )

    return LinkIndex(
        notes_by_normalized_title=title_index,
        outgoing=dict(outgoing),
        backlinks=dict(backlinks),
        dangling=dangling,
        stats=stats,
    )
