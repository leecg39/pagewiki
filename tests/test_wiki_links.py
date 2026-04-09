"""Tests for Phase 1 of v0.2 — the `[[wiki-link]]` resolution index.

Covers the two design decisions recorded in ``docs/v0.2-design.md``:

  * Q1 — title ambiguity: ``resolve`` returns *every* matching note
    (list semantics), not a single "best" one.
  * Q2 — section anchor: ``[[Alpha#Methods]]`` tries to match a
    section descendant inside Alpha's Layer 2 tree; falls back to
    the note root on miss.

All tests run without Ollama, without the filesystem-heavy full
adapter pipeline, and without any LLM calls. Where a test needs a
Layer 2 sub-tree to exercise the anchor-matching path, it injects
``children`` directly onto the ``TreeNode`` rather than running the
real ``build_long_note_subtree`` — faster and keeps the concerns
isolated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pagewiki._text import normalize_title
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.vault import scan_folder
from pagewiki.wiki_links import (
    LinkIndex,
    LinkIndexStats,
    ResolvedLink,
    _apply_anchor_fallback,
    _extract_links_with_anchors,
    build_link_index,
)


# ─────────────────────────────────────────────────────────────────────────────
# Regex-level unit tests for the richer link parser
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractLinksWithAnchors:
    def test_plain_target(self) -> None:
        assert _extract_links_with_anchors("see [[Alpha]] here") == [
            ("Alpha", None)
        ]

    def test_target_with_alias_strips_alias(self) -> None:
        # [[Alpha|별명]] → target="Alpha", no anchor, alias discarded.
        assert _extract_links_with_anchors("[[Alpha|별명]]") == [("Alpha", None)]

    def test_target_with_section_anchor(self) -> None:
        assert _extract_links_with_anchors("[[Alpha#Methods]]") == [
            ("Alpha", "Methods")
        ]

    def test_target_with_anchor_and_alias(self) -> None:
        assert _extract_links_with_anchors("[[Alpha#Methods|자세히]]") == [
            ("Alpha", "Methods")
        ]

    def test_multiple_links_in_one_paragraph(self) -> None:
        text = "Compare [[Beta]] vs [[Gamma#Training]] per [[Delta|델타]]."
        assert _extract_links_with_anchors(text) == [
            ("Beta", None),
            ("Gamma", "Training"),
            ("Delta", None),
        ]

    def test_no_links_returns_empty(self) -> None:
        assert _extract_links_with_anchors("plain text with no links") == []


# ─────────────────────────────────────────────────────────────────────────────
# LinkIndex: resolution + dangling + stats (via full build_link_index path)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_vault(tmp_path: Path) -> Path:
    """Three notes under ``Notes/`` with a small link graph.

    Layout:

      - alpha.md   →  [[Beta]], [[Missing]]
      - beta.md    →  [[Alpha]]   (forms a 2-cycle with alpha)
      - gamma.md   →  [[Alpha]], [[Beta]]
    """
    vault = tmp_path / "vault"
    notes_dir = vault / "Notes"
    notes_dir.mkdir(parents=True)

    (notes_dir / "alpha.md").write_text(
        "# Alpha\nBody text that references [[Beta]] and [[Missing]].\n",
        encoding="utf-8",
    )
    (notes_dir / "beta.md").write_text(
        "# Beta\nBody text with [[Alpha]] reference.\n",
        encoding="utf-8",
    )
    (notes_dir / "gamma.md").write_text(
        "# Gamma\nLinks to [[Alpha]] and [[Beta]].\n",
        encoding="utf-8",
    )
    return vault


class TestLinkIndexResolution:
    def test_resolve_exact_title_match(self, simple_vault: Path) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)
        results = index.resolve("Alpha")
        assert len(results) == 1
        assert results[0].title == "alpha"

    def test_resolve_is_case_insensitive(self, simple_vault: Path) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)
        # "BETA" normalizes to the same key as "beta"
        assert index.resolve("BETA") == index.resolve("beta")
        assert len(index.resolve("BETA")) == 1

    def test_resolve_nonexistent_returns_empty_list(
        self, simple_vault: Path
    ) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)
        assert index.resolve("Nonexistent") == []


class TestLinkIndexAmbiguity:
    """Q1 decision: ``resolve`` returns ALL matching notes, not one."""

    def test_resolve_ambiguous_returns_all(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        research = vault / "Research"
        drafts = vault / "Drafts"
        research.mkdir(parents=True)
        drafts.mkdir(parents=True)

        (research / "alpha.md").write_text(
            "# Alpha\nResearch version.", encoding="utf-8"
        )
        (drafts / "alpha.md").write_text(
            "# Alpha\nDraft version.", encoding="utf-8"
        )
        # A source note that references [[Alpha]] — the ambiguity is
        # independent of which note the source is.
        (research / "caller.md").write_text(
            "# Caller\nSee [[Alpha]] for details.", encoding="utf-8"
        )

        root = scan_folder(vault)
        index = build_link_index(root)

        matches = index.resolve("Alpha")
        assert len(matches) == 2, "both alpha.md notes should resolve"
        match_ids = {m.node_id for m in matches}
        assert "Research/alpha.md" in match_ids
        assert "Drafts/alpha.md" in match_ids

        # The ambiguity also surfaces as ambiguous_links in stats —
        # one source → two candidates = 1 ambiguous pair.
        stats = index.stats()
        assert stats.ambiguous_links == 1


class TestLinkIndexOutgoingAndBacklinks:
    def test_outgoing_lists_only_resolved_links(
        self, simple_vault: Path
    ) -> None:
        """``outgoing`` returns successfully-resolved links only. Dangling
        targets live in ``dangling()`` and are deliberately excluded
        from ``outgoing`` so retrieval callers never receive a
        half-constructed ``ResolvedLink`` with a null target."""
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)

        links = index.outgoing("Notes/alpha.md")
        raw_targets = sorted(link.raw_target for link in links)
        # Only Beta — Missing is in dangling(), not outgoing().
        assert raw_targets == ["Beta"]
        # Every returned link has a non-null target.
        assert all(link.target is not None for link in links)

    def test_outgoing_excludes_dangling_links(
        self, simple_vault: Path
    ) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)
        links = index.outgoing("Notes/alpha.md")
        # "Missing" is dangling and should not be in outgoing()
        assert all(link.raw_target != "Missing" for link in links)
        # But it should be in the dangling list
        dangling_targets = [raw for _, raw in index.dangling()]
        assert "Missing" in dangling_targets

    def test_backlinks_are_reverse_of_outgoing(
        self, simple_vault: Path
    ) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)

        # alpha.md is linked from beta.md and gamma.md
        incoming_to_alpha = index.backlinks("Notes/alpha.md")
        source_ids = sorted(link.source.node_id for link in incoming_to_alpha)
        assert source_ids == ["Notes/beta.md", "Notes/gamma.md"]


class TestCycle:
    """Cycle A→B→A is representable in the index and does not crash
    construction; retrieval-side cycle prevention is v0.2 territory."""

    def test_two_cycle_is_representable(self, simple_vault: Path) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)

        # alpha → beta
        alpha_outgoing = index.outgoing("Notes/alpha.md")
        assert any(link.target.node_id == "Notes/beta.md" for link in alpha_outgoing)

        # beta → alpha (closes the cycle)
        beta_outgoing = index.outgoing("Notes/beta.md")
        assert any(link.target.node_id == "Notes/alpha.md" for link in beta_outgoing)


# ─────────────────────────────────────────────────────────────────────────────
# Q2: section-anchor best-match fallback
# ─────────────────────────────────────────────────────────────────────────────


def _make_note_with_sections(
    *,
    note_id: str,
    note_title: str,
    file_path: Path,
    section_titles: list[str],
) -> TreeNode:
    """Build a TreeNode with ``kind="note"`` + synthetic section children.

    Used by the Q2 tests so they don't need to run the full Layer 2
    adapter pipeline to get a tree that has sections to match against.
    """
    children = [
        TreeNode(
            node_id=f"{note_id}#{i:04d}",
            title=title,
            kind="section",
            file_path=file_path,
            line_range=(1, 10),
        )
        for i, title in enumerate(section_titles, start=1)
    ]
    return TreeNode(
        node_id=note_id,
        title=note_title,
        kind="note",
        tier=NoteTier.LONG,
        file_path=file_path,
        children=children,
    )


class TestAnchorFallback:
    def test_anchor_match_returns_section(self, tmp_path: Path) -> None:
        note = _make_note_with_sections(
            note_id="Notes/paper.md",
            note_title="Paper",
            file_path=tmp_path / "paper.md",
            section_titles=["Methods", "Results", "Discussion"],
        )
        section = _apply_anchor_fallback(note, "Methods")
        assert section.title == "Methods"
        assert section.kind == "section"

    def test_anchor_match_is_case_insensitive(self, tmp_path: Path) -> None:
        note = _make_note_with_sections(
            note_id="Notes/paper.md",
            note_title="Paper",
            file_path=tmp_path / "paper.md",
            section_titles=["Methods", "Results"],
        )
        section = _apply_anchor_fallback(note, "methods")
        assert section.title == "Methods"

    def test_anchor_miss_falls_back_to_note_root(
        self, tmp_path: Path
    ) -> None:
        note = _make_note_with_sections(
            note_id="Notes/paper.md",
            note_title="Paper",
            file_path=tmp_path / "paper.md",
            section_titles=["Methods", "Results"],
        )
        result = _apply_anchor_fallback(note, "Nonexistent Section")
        # Miss → fall back to the note itself, not some arbitrary section.
        assert result is note

    def test_no_anchor_returns_note_root(self, tmp_path: Path) -> None:
        note = _make_note_with_sections(
            note_id="Notes/paper.md",
            note_title="Paper",
            file_path=tmp_path / "paper.md",
            section_titles=["Methods"],
        )
        assert _apply_anchor_fallback(note, None) is note

    def test_note_without_sections_returns_note_root(
        self, tmp_path: Path
    ) -> None:
        note = TreeNode(
            node_id="Notes/atomic.md",
            title="Atomic",
            kind="note",
            tier=NoteTier.ATOMIC,
            file_path=tmp_path / "atomic.md",
            children=[],  # no Layer 2 tree
        )
        # Even with an anchor, a childless note falls back to itself.
        assert _apply_anchor_fallback(note, "Methods") is note

    def test_anchor_finds_nested_section(self, tmp_path: Path) -> None:
        """``[[Paper#Training]]`` should find Training even when it is
        nested under Methods, not a direct child of the note."""
        methods = TreeNode(
            node_id="Notes/paper.md#0001",
            title="Methods",
            kind="section",
            file_path=tmp_path / "paper.md",
            line_range=(1, 20),
            children=[
                TreeNode(
                    node_id="Notes/paper.md#0002",
                    title="Training",
                    kind="section",
                    file_path=tmp_path / "paper.md",
                    line_range=(10, 15),
                )
            ],
        )
        note = TreeNode(
            node_id="Notes/paper.md",
            title="Paper",
            kind="note",
            tier=NoteTier.LONG,
            file_path=tmp_path / "paper.md",
            children=[methods],
        )
        result = _apply_anchor_fallback(note, "Training")
        assert result.title == "Training"
        assert result.node_id == "Notes/paper.md#0002"


class TestBuildLinkIndexWithAnchorHits:
    """End-to-end: a link like ``[[Paper#Methods]]`` should produce a
    ``ResolvedLink`` whose ``target`` is the Methods section, not the
    Paper note root, when Paper has that section available."""

    def test_end_to_end_anchor_resolution(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        notes_dir = vault / "Notes"
        notes_dir.mkdir(parents=True)

        # paper.md is just body — scan_folder will make a note node
        # without sections. We then manually attach a section child
        # before calling build_link_index.
        (notes_dir / "paper.md").write_text(
            "# Paper\nabstract body text.\n", encoding="utf-8"
        )
        (notes_dir / "reviewer.md").write_text(
            "# Reviewer\nSee [[Paper#Methods]] for details.\n",
            encoding="utf-8",
        )

        root = scan_folder(vault, "Notes")

        # Inject a synthetic Methods section onto paper.md so the
        # anchor-matching path fires.
        paper_note = next(
            n for n in root.walk() if n.kind == "note" and n.title == "paper"
        )
        paper_note.children = [
            TreeNode(
                node_id="Notes/paper.md#0001",
                title="Methods",
                kind="section",
                file_path=paper_note.file_path,
                line_range=(2, 10),
            )
        ]

        index = build_link_index(root)
        outgoing = index.outgoing("Notes/reviewer.md")
        assert len(outgoing) == 1
        link = outgoing[0]
        assert link.raw_target == "Paper"
        assert link.section_anchor == "Methods"
        # Target should be the Methods section, not the note root
        assert link.target.title == "Methods"
        assert link.target.node_id == "Notes/paper.md#0001"


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────


class TestLinkIndexStats:
    def test_stats_totals(self, simple_vault: Path) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)
        stats = index.stats()

        # Links: alpha→Beta, beta→Alpha, gamma→Alpha, gamma→Beta = 4 resolved
        # Dangling: alpha→Missing = 1
        assert stats.total_links == 4
        assert stats.dangling_count == 1
        assert stats.ambiguous_links == 0  # no title collisions in this fixture

    def test_stats_top_linked_and_outgoing(
        self, simple_vault: Path
    ) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)
        stats = index.stats()

        # alpha and beta each get 2 incoming links
        top_titles = {title for title, _ in stats.top_linked_to}
        assert "alpha" in top_titles
        assert "beta" in top_titles

        # gamma has 2 outgoing, alpha and beta each have 1
        top_out_dict = dict(stats.top_outgoing)
        assert top_out_dict.get("gamma") == 2


# ─────────────────────────────────────────────────────────────────────────────
# Miscellaneous
# ─────────────────────────────────────────────────────────────────────────────


class TestReaderInjection:
    def test_reader_callback_is_used_instead_of_disk(
        self, tmp_path: Path
    ) -> None:
        """build_link_index should honor the injected reader — tests
        that need to avoid disk I/O can short-circuit it."""
        vault = tmp_path / "vault"
        notes_dir = vault / "Notes"
        notes_dir.mkdir(parents=True)
        # Write *boring* content so the default reader would see
        # zero links; then inject a reader that returns content with
        # a link, and verify the index is built from the injected data.
        (notes_dir / "alpha.md").write_text("# Alpha\nboring.\n", encoding="utf-8")
        (notes_dir / "beta.md").write_text("# Beta\nboring.\n", encoding="utf-8")

        root = scan_folder(vault, "Notes")

        def fake_reader(path: Path) -> str:
            if path.name == "alpha.md":
                return "# Alpha\nbody with [[Beta]] link."
            return "# Beta\nbody."

        index = build_link_index(root, reader=fake_reader)
        outgoing = index.outgoing("Notes/alpha.md")
        assert len(outgoing) == 1
        assert outgoing[0].raw_target == "Beta"

    def test_reader_oserror_is_swallowed(self, tmp_path: Path) -> None:
        """If a note's file can't be read, the build skips it without
        crashing the rest of the scan."""
        vault = tmp_path / "vault"
        notes_dir = vault / "Notes"
        notes_dir.mkdir(parents=True)
        (notes_dir / "alpha.md").write_text("# Alpha\n[[Beta]]\n", encoding="utf-8")
        (notes_dir / "beta.md").write_text("# Beta\n[[Alpha]]\n", encoding="utf-8")

        root = scan_folder(vault, "Notes")

        def flaky_reader(path: Path) -> str:
            if path.name == "alpha.md":
                raise OSError("disk exploded")
            return path.read_text(encoding="utf-8")

        index = build_link_index(root, reader=flaky_reader)
        # alpha was unreadable → contributes nothing
        assert index.outgoing("Notes/alpha.md") == []
        # beta was fine → its outgoing to alpha is preserved
        beta_outgoing = index.outgoing("Notes/beta.md")
        assert len(beta_outgoing) == 1
        assert beta_outgoing[0].target.title == "alpha"


class TestLinkIndexLen:
    def test_len_counts_total_resolved_links(
        self, simple_vault: Path
    ) -> None:
        root = scan_folder(simple_vault, "Notes")
        index = build_link_index(root)
        # Same count as stats.total_links
        assert len(index) == index.stats().total_links == 4
