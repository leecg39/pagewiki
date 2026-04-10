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
from collections.abc import Callable
from pathlib import Path

from .cache import SummaryCache, TreeCache
from .frontmatter import parse_frontmatter
from .pageindex_adapter import build_long_note_subtree
from .prompts import atomic_summary_prompt
from .tree import NoteTier, TreeNode

ChatFn = Callable[[str], str]

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

    fm = parse_frontmatter(text)

    return TreeNode(
        node_id=str(rel),
        title=path.stem,
        kind="note",
        tier=tier,
        file_path=path,
        token_count=tokens,
        wiki_links=extract_wiki_links(text),
        tags=fm.tags,
        date=fm.date,
        aliases=fm.aliases,
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


def summarize_atomic_notes(
    root: TreeNode,
    chat_fn: ChatFn,
    *,
    skip_existing: bool = True,
    summary_cache: SummaryCache | None = None,
    model_id: str | None = None,
) -> int:
    """Fill in one-line summaries for every tier=ATOMIC note under `root`.

    Walks the Layer 1 tree and calls `chat_fn` once per ATOMIC note. MICRO
    notes are skipped (title-only is fine) and LONG notes are handled by
    the PageIndex adapter, so they're skipped here too.

    Args:
        root: Layer 1 tree root.
        chat_fn: LLM callable (production: ollama_client.chat wrapper).
        skip_existing: If True, don't re-summarize notes that already have a
            non-empty `summary` field. Enables cheap incremental re-runs.
        summary_cache: Optional on-disk cache. When provided, summaries are
            loaded from cache if the source note is unchanged, avoiding
            redundant LLM calls on repeated ``ask`` invocations.
        model_id: LLM model identifier, required when ``summary_cache``
            is provided (part of the cache key).

    Returns:
        Number of notes actually summarized (i.e. LLM calls made).
    """
    calls = 0
    for node in root.walk():
        if node.kind != "note" or node.tier != NoteTier.ATOMIC:
            continue
        if skip_existing and node.summary:
            continue
        if node.file_path is None:
            continue

        # Try the on-disk cache first (v0.6).
        if summary_cache is not None and model_id is not None:
            cached = summary_cache.load(node.file_path, model_id)
            if cached is not None:
                node.summary = cached
                continue

        content = node.file_path.read_text(encoding="utf-8")
        prompt = atomic_summary_prompt(node.title, content)
        summary = chat_fn(prompt).strip()
        # Strip quotes if the model wrapped its answer
        summary = summary.strip("\"'").strip()
        node.summary = summary
        calls += 1

        # Persist to cache for next run.
        if summary_cache is not None and model_id is not None:
            summary_cache.save(node.file_path, model_id, summary)

    return calls


def build_long_subtrees(
    root: TreeNode,
    *,
    vault_root: Path,
    model_id: str,
    chat_fn: ChatFn | None = None,
    cache: TreeCache | None = None,
) -> tuple[int, int]:
    """Populate ``children`` for every LONG note under ``root``.

    Walks the Layer 1 tree, finds notes tagged ``tier == LONG``, and
    delegates to ``pageindex_adapter.build_long_note_subtree`` to build
    the PageIndex-style section hierarchy. Results are cached on disk
    via ``TreeCache`` so repeat scans are near-instant.

    Args:
        root: Layer 1 tree root from ``scan_folder``.
        vault_root: Absolute path to the Obsidian vault. Used as the
            base for the ``.pagewiki-cache/`` directory.
        model_id: LLM model identifier (e.g. ``ollama/gemma4:26b``).
            Part of the cache key so cached trees built by a different
            model are automatically rejected.
        chat_fn: Optional LLM callable for generating per-section
            summaries. When ``None``, sections fall back to truncated
            body text (faster, worse ToC review quality).
        cache: Optional pre-constructed cache. If omitted, a new one
            rooted at ``vault_root`` is created.

    Returns:
        ``(built, from_cache)`` — the count of LONG notes processed
        fresh vs. served from the cache.
    """
    if cache is None:
        cache = TreeCache(vault_root)

    built = 0
    from_cache = 0

    for node in root.walk():
        if node.kind != "note" or node.tier != NoteTier.LONG:
            continue
        if node.file_path is None:
            continue

        # ``node.node_id`` is already the vault-relative path (assigned
        # by ``_note_to_node``), so reuse it directly as the section
        # prefix. This guarantees ``Research/paper.md#0002`` and
        # ``Archive/paper.md#0002`` stay distinct in the retrieval
        # ``visited_ids`` set. Fixes PR #1 review comment (v0.1.3).
        node_path = node.file_path
        node_title = node.title
        node_prefix = node.node_id

        def _build(
            _path=node_path, _title=node_title, _prefix=node_prefix
        ) -> list[TreeNode]:
            return build_long_note_subtree(
                _path,
                _title,
                chat_fn=chat_fn,
                node_id_prefix=_prefix,
            )

        children, hit = cache.load_or_build(node.file_path, model_id, _build)
        node.children = children
        if hit:
            from_cache += 1
        else:
            built += 1

    return built, from_cache


def filter_tree(
    root: TreeNode,
    *,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
) -> TreeNode:
    """Return a pruned copy of ``root`` keeping only notes that match filters.

    Folder nodes are kept if they still have at least one matching
    descendant after filtering. Section children inherit their parent
    note's filter result.

    Args:
        tags: Keep notes whose ``tags`` field intersects with this set
            (case-insensitive). ``None`` disables tag filtering.
        after: Keep notes with ``date >= after`` (ISO prefix, e.g.
            ``"2024-01"``). Notes without a date are *kept*.
        before: Keep notes with ``date <= before``. Notes without a date
            are *kept*.
    """
    if tags is None and after is None and before is None:
        return root

    tag_set = {t.lower() for t in tags} if tags else None

    def _matches(node: TreeNode) -> bool:
        if node.kind != "note":
            return True  # folders handled by child presence

        if tag_set is not None:
            note_tags = {t.lower() for t in node.tags}
            if not note_tags & tag_set:
                return False

        if node.date:
            if after is not None and node.date < after:
                return False
            if before is not None and node.date > before:
                return False

        return True

    def _prune(node: TreeNode) -> TreeNode | None:
        if node.kind == "note":
            return node if _matches(node) else None

        # Folder: recursively prune children.
        new_children = []
        for child in node.children:
            pruned = _prune(child)
            if pruned is not None:
                new_children.append(pruned)

        if not new_children and node is not root:
            return None

        copy = node.model_copy(update={"children": new_children})
        return copy

    result = _prune(root)
    return result if result is not None else root.model_copy(update={"children": []})
