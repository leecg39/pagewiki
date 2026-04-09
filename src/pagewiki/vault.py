"""Layer 1: Obsidian vault scanner and 3-tier classifier.

Responsibilities:
  1. Walk a vault folder and read every .md note
  2. Count tokens (approximate) and classify into micro/atomic/long tiers
  3. Extract outgoing [[wiki-links]]
  4. Build the Layer 1 TreeNode hierarchy (folders → notes)

Long notes are marked with tier=LONG; the actual PageIndex sub-tree
generation happens in pageindex_adapter.py so this module stays
free of LLM calls.
"""

from __future__ import annotations

import re
from pathlib import Path

from .tree import NoteTier, TreeNode

# Heuristic: 1 token ≈ 4 chars for English, ≈ 2 chars for Korean.
# Use the conservative English ratio; real tokenization happens in PageIndex.
_CHARS_PER_TOKEN = 3

# 3-tier thresholds (docs/ARCHITECTURE.md §3)
MICRO_MAX_TOKENS = 500
ATOMIC_MAX_TOKENS = 3000

# [[wiki-link]] or [[target|alias]]
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate without loading a tokenizer."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def classify(token_count: int) -> NoteTier:
    if token_count < MICRO_MAX_TOKENS:
        return NoteTier.MICRO
    if token_count < ATOMIC_MAX_TOKENS:
        return NoteTier.ATOMIC
    return NoteTier.LONG


def extract_wiki_links(text: str) -> list[str]:
    """Return a list of raw link targets (no alias, no section anchor)."""
    return [match.group(1).strip() for match in _WIKI_LINK_RE.finditer(text)]


def _note_to_node(path: Path, vault_root: Path) -> TreeNode:
    """Read a single .md note and build its Layer 1 leaf node.

    tier=LONG notes keep `children=[]` here; pageindex_adapter fills them
    in a second pass so this function stays I/O-bound only.
    """
    text = path.read_text(encoding="utf-8")
    tokens = estimate_tokens(text)
    tier = classify(tokens)
    rel = path.relative_to(vault_root)

    return TreeNode(
        node_id=str(rel),
        title=path.stem,
        kind="note",
        tier=tier,
        file_path=path,
        token_count=tokens,
        wiki_links=extract_wiki_links(text),
    )


def scan_folder(vault_root: Path, folder: str | None = None) -> TreeNode:
    """Scan a folder (or entire vault) and return the Layer 1 tree root.

    Args:
        vault_root: Path to the Obsidian vault root (the folder containing .obsidian/).
        folder: Optional subfolder (relative to vault_root) to restrict the scan.

    Returns:
        Root TreeNode whose children mirror the directory structure.
    """
    vault_root = vault_root.resolve()
    scan_root = vault_root / folder if folder else vault_root
    if not scan_root.is_dir():
        raise FileNotFoundError(f"Vault folder not found: {scan_root}")

    return _build_folder_node(scan_root, vault_root)


def _build_folder_node(folder: Path, vault_root: Path) -> TreeNode:
    """Recursively build a folder TreeNode."""
    rel = folder.relative_to(vault_root)
    node = TreeNode(
        node_id=str(rel) if str(rel) != "." else "",
        title=folder.name or vault_root.name,
        kind="folder",
    )

    for entry in sorted(folder.iterdir()):
        # Skip Obsidian internals and pagewiki log folder
        if entry.name.startswith(".") or entry.name == ".pagewiki-log":
            continue

        if entry.is_dir():
            node.children.append(_build_folder_node(entry, vault_root))
        elif entry.suffix == ".md":
            node.children.append(_note_to_node(entry, vault_root))

    return node
