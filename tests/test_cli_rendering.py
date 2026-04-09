"""CLI rendering helpers — regression tests for markup-safe output.

These tests exist because Rich's ``Console.print`` parses square
brackets as style markup. An early ``scan --show-graph`` implementation
printed dangling links via ``f"[[{raw_target}]]"``, which Rich parsed
as a double-bracket escape wrapping an unknown style tag and rendered
as an empty ``[]`` — losing the raw_target entirely when the target
string happened to look like a valid tag name (any ASCII word, e.g.
``missing-note``). Non-ASCII targets such as ``[[링크]]`` survived only
because they did not match the tag grammar.

The fix extracts a ``_format_dangling_line`` helper that escapes both
user-supplied strings via ``rich.markup.escape`` and uses literal
``\\[`` escapes for the outer brackets. These tests pin that behavior
by rendering the helper's output through a real ``Console`` and
asserting on the captured plain text.
"""

from __future__ import annotations

import io

from rich.console import Console

from pagewiki.cli import _format_dangling_line


def _render(markup: str) -> str:
    """Render a Rich markup string through an isolated Console and
    return its plain-text output (no ANSI, no trailing newline)."""
    buffer = io.StringIO()
    console = Console(
        file=buffer, force_terminal=False, color_system=None, width=200
    )
    console.print(markup)
    return buffer.getvalue().rstrip("\n")


class TestFormatDanglingLine:
    def test_ascii_target_is_preserved(self) -> None:
        """Before the fix, ``missing-note`` vanished because Rich read
        ``[missing-note]`` as an unknown style tag and swallowed it."""
        rendered = _render(_format_dangling_line("index.md", "missing-note"))
        assert "missing-note" in rendered
        assert "index.md" in rendered
        assert "[[missing-note]]" in rendered

    def test_non_ascii_target_is_preserved(self) -> None:
        """Non-ASCII targets worked by accident before the fix; pin it."""
        rendered = _render(_format_dangling_line("note.md", "링크"))
        assert "[[링크]]" in rendered

    def test_path_prefixed_target_is_preserved(self) -> None:
        """Path-prefixed targets contain ``/`` but still have to round-trip
        through the renderer intact."""
        rendered = _render(
            _format_dangling_line("caller.md", "wrong-folder/rag")
        )
        assert "[[wrong-folder/rag]]" in rendered

    def test_target_with_square_bracket_does_not_crash(self) -> None:
        """A raw target that itself contains brackets (defensive —
        unlikely in real vaults) should not break rendering."""
        rendered = _render(_format_dangling_line("caller.md", "odd[name]"))
        assert "odd[name]" in rendered

    def test_source_with_brackets_in_path_is_preserved(self) -> None:
        """Real Obsidian paths occasionally contain brackets in filenames.
        Escape must cover the source_id too, not just the raw_target."""
        rendered = _render(
            _format_dangling_line("notes/[draft] paper.md", "missing")
        )
        assert "[draft] paper.md" in rendered
        assert "[[missing]]" in rendered
