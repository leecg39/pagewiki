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
