"""JSON tree node schema for pagewiki.

Mirrors the PageIndex node format but extends it with Obsidian-specific fields
(file path, wiki-link outgoing edges, 3-tier classification).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class NoteTier(str, Enum):
    """3-tier classification from docs/ARCHITECTURE.md §3."""

    MICRO = "micro"  # < 500 tokens — title-only leaf
    ATOMIC = "atomic"  # 500~3000 tokens — 1-line summary leaf
    LONG = "long"  # > 3000 tokens — delegates to PageIndex sub-tree


class TreeNode(BaseModel):
    """Unified tree node used by both Layer 1 (vault) and Layer 2 (note).

    Layer 1 nodes represent folders and notes.
    Layer 2 nodes come directly from PageIndex SDK (`get_tree()`) for long notes.
    """

    node_id: str
    title: str
    summary: str = ""
    kind: Literal["folder", "note", "section"] = "note"
    tier: NoteTier | None = None

    # Obsidian-specific (Layer 1 only)
    file_path: Path | None = None
    token_count: int | None = None
    wiki_links: list[str] = Field(default_factory=list)

    # PageIndex-compatible (Layer 2)
    # For PDF sources: (start_page, end_page). Unused in markdown mode.
    page_range: tuple[int, int] | None = None
    # For markdown sources: (start_line_1indexed, end_line_exclusive).
    # Populated by pageindex_adapter when splitting LONG notes into sections.
    line_range: tuple[int, int] | None = None

    # Recursive children
    children: list[TreeNode] = Field(default_factory=list)

    def walk(self):
        """Depth-first traversal yielding every node."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, node_id: str) -> TreeNode | None:
        for node in self.walk():
            if node.node_id == node_id:
                return node
        return None


TreeNode.model_rebuild()
