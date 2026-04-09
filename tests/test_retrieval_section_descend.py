"""Tests for the section-descend path added in v0.1.2.

When a LONG note has a Layer 2 sub-tree, the retrieval loop must:
  1. Select the note from the folder's ToC,
  2. Instead of loading the whole note, present its section ToC,
  3. Let the LLM pick a specific section,
  4. Load ONLY that section's line range for evaluation.

These tests wire a scripted ``chat_fn`` so the selection path is
deterministic and independent of any real model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pagewiki.pageindex_adapter import build_long_note_subtree
from pagewiki.retrieval import run_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.vault import scan_folder


def _scripted_chat(responses: list[str]) -> Callable[[str], str]:
    """Replay ``responses`` in order; extra calls return a safe DONE."""
    idx = {"i": 0}

    def _call(prompt: str) -> str:
        if idx["i"] >= len(responses):
            return "DONE: script exhausted"
        reply = responses[idx["i"]]
        idx["i"] += 1
        return reply

    return _call


@pytest.fixture
def vault_with_sectioned_note(tmp_path: Path) -> tuple[Path, TreeNode]:
    """Build a vault whose single LONG note has a populated sub-tree.

    The filename ``notes.md`` deliberately does NOT match the h1
    ``# Paper``, so the v0.1.3 h1-flatten optimization stays dormant
    and these tests keep exercising the full multi-level descend path
    (folder → note → root section → leaf section). Flattened-tree
    behavior has its own test class.
    """
    vault = tmp_path / "vault"
    research = vault / "Research"
    research.mkdir(parents=True)

    body_filler = "Lorem ipsum dolor sit amet. " * 200
    note = research / "notes.md"
    note.write_text(
        "# Paper\n\n"
        f"Opening paragraph. {body_filler}\n\n"
        "## Methods\n\n"
        f"Methods body. {body_filler}\n\n"
        "## Results\n\n"
        f"Results body. {body_filler}\n",
        encoding="utf-8",
    )

    root = scan_folder(vault, "Research")
    # Inject the sub-tree directly (skipping the cache for test isolation).
    # We pass note.stem ("notes") as note_title so _extract_intro_from_root
    # sees the title mismatch and leaves the root section intact.
    long_nodes = [
        n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
    ]
    assert len(long_nodes) == 1
    long_nodes[0].children = build_long_note_subtree(note, "notes", chat_fn=None)

    return vault, root


class TestRetrievalSectionDescend:
    def test_long_note_with_children_behaves_like_folder(
        self, vault_with_sectioned_note: tuple[Path, TreeNode]
    ) -> None:
        """LONG note with sub-tree should trigger a folder-style descend
        rather than an immediate whole-file load.

        The retrieval loop walks folder → note → root-section → leaf-section,
        so the scripted chat must issue one SELECT per descent level.
        """
        _, root = vault_with_sectioned_note
        long_nodes = [
            n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
        ]
        long_note = long_nodes[0]
        assert len(long_note.children) > 0

        # Descend path: long_note.children[0] is the root section ("Paper"),
        # whose own children are the leaf sections ("Methods", "Results", …).
        root_section = long_note.children[0]
        leaf_sections = [c for c in root_section.children if c.kind == "section"]
        assert leaf_sections, "expected at least one leaf section to target"
        target_leaf = leaf_sections[0]

        responses = [
            f"SELECT: {long_note.node_id}",    # folder → LONG note
            f"SELECT: {root_section.node_id}", # note → root section
            f"SELECT: {target_leaf.node_id}",  # root section → leaf section
            "SUFFICIENT: found the section",   # evaluate
            "최종 답변입니다. [[Paper]]",        # final answer
        ]
        result = run_retrieval(
            "what are the results?", root, _scripted_chat(responses)
        )
        # Citations must include at least one section node id (they are
        # formatted as "<file>#<zfill-id>" by the adapter).
        assert any("#" in cid for cid in result.cited_nodes), (
            f"expected section node_id in citations, got {result.cited_nodes!r}"
        )
        assert result.answer  # non-empty

    def test_section_load_respects_line_range(
        self, vault_with_sectioned_note: tuple[Path, TreeNode]
    ) -> None:
        """``_load_note_content`` on a section returns only the lines
        inside that section's ``line_range`` — not the whole file.

        The outermost root section covers lines 1..N and therefore
        matches the whole file; only leaf/child sections are expected
        to be strict subsets, and at least one such section must exist.
        """
        from pagewiki.retrieval import _load_note_content

        _, root = vault_with_sectioned_note
        long_note = next(
            n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
        )
        root_section = long_note.children[0]
        child_sections = [c for c in root_section.children if c.kind == "section"]
        assert child_sections, "test fixture must expose at least one child section"

        whole = long_note.file_path.read_text(encoding="utf-8")
        strict_subset_count = 0
        for section in child_sections:
            assert section.line_range is not None
            partial = _load_note_content(section)
            assert partial, f"section {section.title!r} produced empty content"
            # The partial content must be a substring of the original file,
            # guaranteeing we did not fabricate text.
            assert partial.strip() in whole
            if len(partial) < len(whole):
                strict_subset_count += 1
        assert strict_subset_count >= 1, (
            "at least one child section must load strictly less than the whole file"
        )


class TestRetrievalWithFlattenedTree:
    """With v0.1.3 h1-flatten, a matching-title note loses one level
    of descent — retrieval goes folder → note → leaf section directly,
    skipping the redundant root section that wrapped the whole file."""

    @pytest.fixture
    def flattened_vault(self, tmp_path: Path) -> tuple[Path, TreeNode]:
        vault = tmp_path / "vault"
        research = vault / "Research"
        research.mkdir(parents=True)
        # Filename matches the h1 title → flatten kicks in.
        body_filler = "Lorem ipsum dolor sit amet. " * 200
        (research / "paper.md").write_text(
            "# Paper\n\n"
            f"Abstract content we want preserved as (intro). {body_filler}\n\n"
            "## Methods\n\n"
            f"Methods body. {body_filler}\n\n"
            "## Results\n\n"
            f"Results body. {body_filler}\n",
            encoding="utf-8",
        )

        root = scan_folder(vault, "Research")
        from pagewiki.vault import build_long_subtrees

        build_long_subtrees(
            root,
            vault_root=vault,
            model_id="ollama/gemma4:26b",
            chat_fn=None,
        )
        return vault, root

    def test_flattened_note_children_have_no_paper_wrapper(
        self, flattened_vault: tuple[Path, TreeNode]
    ) -> None:
        _, root = flattened_vault
        long_note = next(
            n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
        )
        child_titles = [c.title for c in long_note.children]
        # The "Paper" h1 wrapper should be gone; (intro) + h2 sections
        # should be promoted to the top level directly.
        assert "Paper" not in child_titles
        assert "(intro)" in child_titles
        assert "Methods" in child_titles
        assert "Results" in child_titles

    def test_flattened_note_needs_one_fewer_select(
        self, flattened_vault: tuple[Path, TreeNode]
    ) -> None:
        _, root = flattened_vault
        long_note = next(
            n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
        )
        methods = next(c for c in long_note.children if c.title == "Methods")

        # Descend path is now: folder → note → leaf section. Three
        # SELECTs instead of four.
        responses = [
            f"SELECT: {long_note.node_id}",   # Research folder → note
            f"SELECT: {methods.node_id}",     # note → Methods directly
            "SUFFICIENT: found methods",
            "최종 답변입니다.",
        ]
        result = run_retrieval(
            "what methods did they use?", root, _scripted_chat(responses)
        )
        assert methods.node_id in result.cited_nodes

    def test_flattened_intro_section_is_reachable(
        self, flattened_vault: tuple[Path, TreeNode]
    ) -> None:
        """The synthetic (intro) section must be a first-class
        retrieval target — selectable, loadable, and its loaded content
        must contain the abstract without any subsequent section body."""
        from pagewiki.retrieval import _load_note_content

        _, root = flattened_vault
        long_note = next(
            n for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
        )
        intro = next(c for c in long_note.children if c.title == "(intro)")

        responses = [
            f"SELECT: {long_note.node_id}",
            f"SELECT: {intro.node_id}",
            "SUFFICIENT: abstract found",
            "요약된 답변.",
        ]
        result = run_retrieval(
            "what does the abstract say?", root, _scripted_chat(responses)
        )
        assert intro.node_id in result.cited_nodes

        intro_content = _load_note_content(intro)
        assert "Abstract content" in intro_content
        assert "Methods body." not in intro_content
        assert "Results body." not in intro_content
