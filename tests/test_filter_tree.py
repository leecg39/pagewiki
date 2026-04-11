"""Tests for ``pagewiki.vault.filter_tree`` and frontmatter integration."""

from __future__ import annotations

from pathlib import Path

from pagewiki.tree import NoteTier, TreeNode
from pagewiki.vault import filter_tree, scan_folder


def _make_tree() -> TreeNode:
    """Build a minimal tree with tags and dates for filter testing."""
    return TreeNode(
        node_id="",
        title="root",
        kind="folder",
        children=[
            TreeNode(
                node_id="a.md",
                title="A",
                kind="note",
                tier=NoteTier.ATOMIC,
                tags=["research", "ml"],
                date="2024-06-01",
            ),
            TreeNode(
                node_id="b.md",
                title="B",
                kind="note",
                tier=NoteTier.ATOMIC,
                tags=["personal"],
                date="2025-01-15",
            ),
            TreeNode(
                node_id="c.md",
                title="C",
                kind="note",
                tier=NoteTier.MICRO,
                tags=[],
                date=None,
            ),
            TreeNode(
                node_id="sub",
                title="sub",
                kind="folder",
                children=[
                    TreeNode(
                        node_id="sub/d.md",
                        title="D",
                        kind="note",
                        tier=NoteTier.ATOMIC,
                        tags=["research"],
                        date="2024-11-01",
                    ),
                ],
            ),
        ],
    )


class TestFilterTree:
    def test_no_filters_returns_original(self) -> None:
        root = _make_tree()
        assert filter_tree(root) is root

    def test_filter_by_tag(self) -> None:
        root = _make_tree()
        filtered = filter_tree(root, tags=["research"])
        note_ids = [n.node_id for n in filtered.walk() if n.kind == "note"]
        assert "a.md" in note_ids
        assert "sub/d.md" in note_ids
        assert "b.md" not in note_ids
        # C has no tags → filtered out when tag filter is active
        assert "c.md" not in note_ids

    def test_filter_by_tag_case_insensitive(self) -> None:
        root = _make_tree()
        filtered = filter_tree(root, tags=["ML"])
        note_ids = [n.node_id for n in filtered.walk() if n.kind == "note"]
        assert "a.md" in note_ids

    def test_filter_by_date_after(self) -> None:
        root = _make_tree()
        filtered = filter_tree(root, after="2025-01")
        note_ids = [n.node_id for n in filtered.walk() if n.kind == "note"]
        assert "b.md" in note_ids
        # C has no date → kept (not excluded)
        assert "c.md" in note_ids
        assert "a.md" not in note_ids

    def test_filter_by_date_before(self) -> None:
        root = _make_tree()
        filtered = filter_tree(root, before="2024-08")
        note_ids = [n.node_id for n in filtered.walk() if n.kind == "note"]
        assert "a.md" in note_ids
        assert "b.md" not in note_ids

    def test_filter_combined(self) -> None:
        root = _make_tree()
        filtered = filter_tree(root, tags=["research"], after="2024-10")
        note_ids = [n.node_id for n in filtered.walk() if n.kind == "note"]
        assert note_ids == ["sub/d.md"]

    def test_empty_folder_pruned(self) -> None:
        root = _make_tree()
        filtered = filter_tree(root, tags=["personal"])
        # sub/ folder should be pruned since D doesn't match.
        folder_ids = [n.node_id for n in filtered.walk() if n.kind == "folder"]
        assert "sub" not in folder_ids


class TestFrontmatterIntegration:
    def test_scan_parses_frontmatter(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)

        (notes / "tagged.md").write_text(
            "---\ntags: [research, ml]\ndate: 2024-06-01\n---\n"
            + "a " * 1500,
            encoding="utf-8",
        )
        (notes / "plain.md").write_text("a " * 1500, encoding="utf-8")

        root = scan_folder(vault, "Notes")
        by_title = {n.title: n for n in root.walk() if n.kind == "note"}

        assert by_title["tagged"].tags == ["research", "ml"]
        assert by_title["tagged"].date == "2024-06-01"
        assert by_title["plain"].tags == []
        assert by_title["plain"].date is None
