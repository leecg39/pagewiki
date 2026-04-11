"""Tests for Layer 2 PageIndex sub-tree adapter.

These exercise the pure-Python tree-building path (no Ollama, no
LiteLLM). LLM summarization is covered by injecting a scripted
``chat_fn`` and asserting it was called the expected number of times
with the expected prompts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pagewiki.pageindex_adapter import (
    DEFAULT_SUMMARY_TOKEN_THRESHOLD,
    build_long_note_subtree,
)

# Reusable sample markdown with a three-level heading structure. The
# body of section "1. Intro" is kept short so thinning will merge it;
# sections 2 and 3 are large enough to survive.
_SAMPLE_MD = """# Root Title

Opening paragraph.

## 1. Intro

Short intro body.

## 2. Methods

""" + ("Methods body line. " * 200) + """

### 2.1 Dataset

""" + ("Dataset description. " * 200) + """

## 3. Results

""" + ("Results body line. " * 200) + """
"""


@pytest.fixture
def long_note(tmp_path: Path) -> Path:
    """Write the sample markdown to a temp file and return its path."""
    path = tmp_path / "long_note.md"
    path.write_text(_SAMPLE_MD, encoding="utf-8")
    return path


class TestBuildLongNoteSubtree:
    """``build_long_note_subtree`` end-to-end behavior."""

    def test_returns_section_nodes(self, long_note: Path) -> None:
        """A note with headings yields ``kind="section"`` children."""
        tree = build_long_note_subtree(long_note, "Long Note", chat_fn=None)
        assert len(tree) == 1, "one root heading expected"
        root = tree[0]
        assert root.kind == "section"
        assert root.title == "Root Title"

    def test_line_range_is_1_indexed_with_exclusive_end(
        self, long_note: Path
    ) -> None:
        """Each section carries a line_range that matches upstream
        semantics: 1-indexed inclusive start, exclusive end."""
        tree = build_long_note_subtree(long_note, "Long Note", chat_fn=None)
        root = tree[0]
        assert root.line_range is not None
        start, end = root.line_range
        assert start == 1
        assert end > start

        # Every child's start line should fall inside the parent range.
        for child in root.children:
            assert child.line_range is not None
            cstart, cend = child.line_range
            assert start <= cstart < end
            assert cstart < cend

    def test_no_headings_returns_empty(self, tmp_path: Path) -> None:
        """A body-only note with zero headings produces no sub-tree."""
        note = tmp_path / "plain.md"
        note.write_text("Just a body with no headings at all.", encoding="utf-8")
        assert build_long_note_subtree(note, "Plain", chat_fn=None) == []

    def test_chat_fn_called_only_for_large_sections(
        self, long_note: Path
    ) -> None:
        """``chat_fn`` should only be invoked when the section exceeds
        ``summary_token_threshold``; small sections use the fallback."""
        calls: list[str] = []

        def fake_chat(prompt: str) -> str:
            calls.append(prompt)
            return "요약된 섹션입니다."

        tree = build_long_note_subtree(
            long_note,
            "Long Note",
            chat_fn=fake_chat,
            summary_token_threshold=DEFAULT_SUMMARY_TOKEN_THRESHOLD,
        )
        # The methods/results sections are well over 200 tokens,
        # so at least one summary call must have happened.
        assert len(calls) >= 1
        # Every invocation should have received a section_summary prompt.
        assert all("섹션" in c for c in calls)
        # The returned tree must have replaced the fallback with the
        # scripted summary for at least one node.
        summaries = [
            n.summary for n in tree[0].walk() if n.kind == "section"
        ]
        assert "요약된 섹션입니다." in summaries

    def test_summary_fallback_skips_heading_lines(
        self, long_note: Path
    ) -> None:
        """Without a ``chat_fn``, the fallback summary should be the
        first non-heading body line, not the heading itself."""
        tree = build_long_note_subtree(long_note, "Long Note", chat_fn=None)
        for node in tree[0].walk():
            if node.kind == "section" and node.summary:
                assert not node.summary.lstrip().startswith("#"), (
                    f"summary should not start with a markdown heading: {node.summary!r}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# v0.1.3: h1-title flatten optimization
# ─────────────────────────────────────────────────────────────────────────────


class TestH1TitleFlatten:
    """The flatten optimization promotes h2 children of a matching h1 root
    to top-level sections, optionally preserving an intro body as a
    synthetic ``(intro)`` section."""

    def _write_note(self, tmp_path: Path, name: str, body: str) -> Path:
        note = tmp_path / name
        note.write_text(body, encoding="utf-8")
        return note

    def test_flatten_when_filename_matches_h1(self, tmp_path: Path) -> None:
        body = (
            "# Paper\n"
            "An abstract paragraph with real content lives here under "
            "the h1 so we can verify it is preserved.\n"
            "\n## Methods\n"
            + ("Methods body line. " * 200)
            + "\n\n## Results\n"
            + ("Results body line. " * 200)
            + "\n\n## Discussion\n"
            + ("Discussion body line. " * 200)
        )
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(note, "paper", chat_fn=None)
        titles = [c.title for c in children]
        # Intro preserved, then h2 children promoted
        assert titles == ["(intro)", "Methods", "Results", "Discussion"]
        for child in children:
            assert child.kind == "section"

    def test_intro_section_line_range_covers_h1_body_only(
        self, tmp_path: Path
    ) -> None:
        body = (
            "# Paper\nAbstract: important content here.\n\n"
            "## Methods\n" + ("methods body. " * 200)
            + "\n\n## Results\n" + ("results body. " * 200)
        )
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(note, "paper", chat_fn=None)

        intro = next(c for c in children if c.title == "(intro)")
        assert intro.line_range is not None
        start, end = intro.line_range
        assert start == 1
        assert end > start

        # Loading the intro slice must include "Abstract:" but not any
        # subsequent "## " header.
        from pagewiki.retrieval import _load_note_content

        sliced = _load_note_content(intro)
        assert "Abstract:" in sliced
        assert "## " not in sliced

    def test_flatten_without_intro_body_omits_synthetic_section(
        self, tmp_path: Path
    ) -> None:
        body = (
            "# Paper\n## Methods\n"
            + ("methods body. " * 200)
            + "\n\n## Results\n"
            + ("results body. " * 200)
        )
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(note, "paper", chat_fn=None)
        titles = [c.title for c in children]
        assert "(intro)" not in titles
        assert titles == ["Methods", "Results"]

    def test_no_flatten_when_filename_differs(self, tmp_path: Path) -> None:
        body = "# Paper\nbody\n\n## Methods\n" + ("m " * 300)
        note = self._write_note(tmp_path, "notes.md", body)
        children = build_long_note_subtree(
            note, "notes", chat_fn=None, if_thinning=False
        )
        # Root "Paper" is kept intact
        assert len(children) == 1
        assert children[0].title == "Paper"

    def test_flatten_is_case_and_whitespace_insensitive(
        self, tmp_path: Path
    ) -> None:
        body = "#    My  Paper   \n\n## Methods\n" + ("m " * 300)
        note = self._write_note(tmp_path, "my_paper.md", body)
        # Different case + extra whitespace; normalized comparison
        # should still treat them as equal.
        children = build_long_note_subtree(
            note, "MY PAPER", chat_fn=None, if_thinning=False
        )
        assert [c.title for c in children] == ["Methods"]

    def test_no_flatten_when_multiple_top_level_h1s(
        self, tmp_path: Path
    ) -> None:
        body = (
            "# Paper\nbody\n\n## Sub\ncontent\n\n"
            "# Another Topic\nmore body\n\n## Sub2\n" + ("s " * 300)
        )
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(
            note, "paper", chat_fn=None, if_thinning=False
        )
        titles = [c.title for c in children]
        # Two h1s = ambiguous, no flatten
        assert "Paper" in titles
        assert "Another Topic" in titles

    def test_no_flatten_when_h1_has_no_descendants(
        self, tmp_path: Path
    ) -> None:
        body = "# Paper\nJust body text, no sub-sections."
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(
            note, "paper", chat_fn=None, if_thinning=False
        )
        # Single h1 with no descendants — nothing to flatten into.
        assert len(children) == 1
        assert children[0].title == "Paper"

    def test_flatten_opt_out(self, tmp_path: Path) -> None:
        body = "# Paper\nintro\n\n## Methods\n" + ("m " * 300)
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(
            note,
            "paper",
            chat_fn=None,
            flatten_matching_h1=False,
            if_thinning=False,
        )
        assert len(children) == 1
        assert children[0].title == "Paper"
        assert [c.title for c in children[0].children] == ["Methods"]

    def test_intro_node_id_is_stable(self, tmp_path: Path) -> None:
        """The synthetic intro id must be deterministic so the
        retrieval visited_ids set and cache behave predictably."""
        body = "# Paper\nabstract here\n\n## Methods\n" + ("m " * 300)
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(
            note, "paper", chat_fn=None, if_thinning=False
        )
        intro = next(c for c in children if c.title == "(intro)")
        assert intro.node_id == "paper.md#intro"

    def test_node_id_prefix_namespaces_sections(self, tmp_path: Path) -> None:
        """Passing ``node_id_prefix`` should replace the bare-filename
        default used by direct unit-test callers."""
        body = "# Paper\nintro\n\n## Methods\n" + ("m " * 300)
        note = self._write_note(tmp_path, "paper.md", body)
        children = build_long_note_subtree(
            note,
            "paper",
            chat_fn=None,
            node_id_prefix="Research/paper.md",
            if_thinning=False,
        )
        for child in children:
            assert child.node_id.startswith("Research/paper.md#"), (
                f"expected Research/paper.md# prefix, got {child.node_id!r}"
            )
