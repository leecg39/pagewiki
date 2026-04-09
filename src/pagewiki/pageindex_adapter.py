"""Layer 2: Thin adapter around the official PageIndex SDK.

For every Layer 1 leaf with tier=LONG, this module calls PageIndex's
`submit_document` + `get_tree` and grafts the returned hierarchical
JSON tree onto the Layer 1 node's `children`.

Intentionally minimal: we do NOT wrap PageIndex's `chat()` — that happens
in the retrieval loop (see Phase 2 of ARCHITECTURE.md).

The SDK import is deferred so that the rest of pagewiki can be imported
in environments without `pageindex` installed (e.g. unit tests on Layer 1
classification alone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tree import TreeNode


def _get_client() -> Any:
    """Lazy import + construct. Raises if PAGEINDEX_API_KEY is not set
    and no local mode is configured.
    """
    try:
        from pageindex import PageIndexClient  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pageindex package is required for long-note processing. "
            "Install with `pip install pageindex`."
        ) from e

    return PageIndexClient()


def build_long_note_subtree(
    note_path: Path,
    *,
    max_tokens_per_node: int = 20_000,
    max_pages_per_node: int = 10,
) -> list[TreeNode]:
    """Submit a long markdown note to PageIndex and return its sub-tree children.

    The returned list becomes the `children` of the Layer 1 leaf for this note.

    Notes on defaults:
      * max_tokens_per_node=20,000 matches PageIndex's documented default.
        Gemma 4's 128K context easily fits this; consider raising to 50,000
        for very dense financial-appendix-style notes.
      * Markdown header auto-detect is enabled by default in PageIndex, so
        Obsidian notes with #/##/### structure are consumed directly.
    """
    client = _get_client()

    document = client.submit_document(
        str(note_path),
        max_tokens_per_node=max_tokens_per_node,
        max_pages_per_node=max_pages_per_node,
    )
    raw_tree = client.get_tree(document.id)

    return [_to_tree_node(child) for child in raw_tree.get("children", [])]


def _to_tree_node(raw: dict[str, Any]) -> TreeNode:
    """Convert a raw PageIndex dict node into our unified TreeNode schema."""
    page_range = raw.get("page_range")
    return TreeNode(
        node_id=raw["id"] if "id" in raw else raw.get("node_id", ""),
        title=raw.get("title", ""),
        summary=raw.get("summary", ""),
        kind="section",
        page_range=tuple(page_range) if page_range else None,
        children=[_to_tree_node(c) for c in raw.get("children", [])],
    )
