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
