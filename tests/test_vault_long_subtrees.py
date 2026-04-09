"""Tests for ``vault.build_long_subtrees``.

Covers the happy path (LONG note gets its children populated), the
cache short-circuit (second call reuses the first), and the no-op
cases (no LONG notes, or note with no headings).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pagewiki.cache import TreeCache
from pagewiki.tree import NoteTier
from pagewiki.vault import build_long_subtrees, scan_folder


def _make_long_note(content_lines: int = 500) -> str:
    """Generate a markdown body >3000 tokens to guarantee LONG tier."""
    body = "\n".join(f"Paragraph line {i}." * 3 for i in range(content_lines))
    return (
        "# Research Paper\n\n"
        "Opening.\n\n"
        "## Background\n\n"
        f"{body}\n\n"
        "## Methods\n\n"
        f"{body}\n\n"
        "## Results\n\n"
        f"{body}\n"
    )


@pytest.fixture
def vault_with_long_note(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    research = vault / "Research"
    research.mkdir(parents=True)
    (research / "paper.md").write_text(_make_long_note(), encoding="utf-8")
    (research / "short.md").write_text(
        "# Short\n\ntiny body\n", encoding="utf-8"
    )
    return vault


class TestBuildLongSubtrees:
    def test_populates_children_for_long_notes_only(
        self, vault_with_long_note: Path
    ) -> None:
        root = scan_folder(vault_with_long_note, "Research")
        built, from_cache = build_long_subtrees(
            root,
            vault_root=vault_with_long_note,
            model_id="test-model",
            chat_fn=None,
        )
        assert built == 1
        assert from_cache == 0

        long_nodes = [
            n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
        ]
        assert len(long_nodes) == 1
        assert len(long_nodes[0].children) > 0

        # The short note must NOT have had a sub-tree built.
        short_nodes = [
            n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.MICRO
        ]
        assert all(len(n.children) == 0 for n in short_nodes)

    def test_second_call_hits_cache(self, vault_with_long_note: Path) -> None:
        cache = TreeCache(vault_with_long_note)

        root1 = scan_folder(vault_with_long_note, "Research")
        built1, _ = build_long_subtrees(
            root1,
            vault_root=vault_with_long_note,
            model_id="test-model",
            chat_fn=None,
            cache=cache,
        )
        assert built1 == 1

        root2 = scan_folder(vault_with_long_note, "Research")
        built2, from_cache2 = build_long_subtrees(
            root2,
            vault_root=vault_with_long_note,
            model_id="test-model",
            chat_fn=None,
            cache=cache,
        )
        assert built2 == 0
        assert from_cache2 == 1

    def test_no_long_notes_is_noop(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        f = vault / "Research"
        f.mkdir(parents=True)
        (f / "tiny.md").write_text("# tiny\n\nbody\n", encoding="utf-8")

        root = scan_folder(vault, "Research")
        built, from_cache = build_long_subtrees(
            root, vault_root=vault, model_id="m", chat_fn=None
        )
        assert built == 0
        assert from_cache == 0

    def test_same_filename_in_different_folders_stays_distinct(
        self, tmp_path: Path
    ) -> None:
        """Regression for PR #1 review: two notes both named
        ``paper.md`` in different folders must produce sections with
        *distinct* node_ids, otherwise ``retrieval.visited_ids``
        deduplicates them and silently drops evidence from one of the
        branches. Fixed in v0.1.3 by threading the vault-relative
        path through ``node_id_prefix``."""
        vault = tmp_path / "vault"
        research = vault / "Research"
        archive = vault / "Archive"
        research.mkdir(parents=True)
        archive.mkdir(parents=True)

        long_body = _make_long_note()
        (research / "paper.md").write_text(long_body, encoding="utf-8")
        (archive / "paper.md").write_text(long_body, encoding="utf-8")

        root = scan_folder(vault)  # whole vault, both folders
        built, _ = build_long_subtrees(
            root, vault_root=vault, model_id="m", chat_fn=None
        )
        assert built == 2

        research_note = next(
            n for n in root.walk() if n.node_id == "Research/paper.md"
        )
        archive_note = next(
            n for n in root.walk() if n.node_id == "Archive/paper.md"
        )

        def section_ids(note) -> list[str]:
            return [s.node_id for s in note.walk() if s.kind == "section"]

        research_ids = section_ids(research_note)
        archive_ids = section_ids(archive_note)
        assert research_ids, "research note should have sections"
        assert archive_ids, "archive note should have sections"

        overlap = set(research_ids) & set(archive_ids)
        assert not overlap, (
            f"section ids collide across folders: {sorted(overlap)!r}"
        )

        # Prefixes must be derived from the vault-relative path, not
        # the bare filename.
        assert all(nid.startswith("Research/paper.md#") for nid in research_ids)
        assert all(nid.startswith("Archive/paper.md#") for nid in archive_ids)
