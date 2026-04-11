"""Pure helpers used by the retrieval loop (v0.13 refactor).

Kept tiny and side-effect-free so the main ``core.run_retrieval``
stays focused on loop control flow. All I/O is contained here in
``_load_note_content``; the rest is tree walking.
"""

from __future__ import annotations

from pathlib import Path

from ..prompts import NodeSummary
from ..tree import TreeNode
from .types import DEFAULT_MAX_NOTE_CHARS


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
