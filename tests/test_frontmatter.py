"""Tests for ``pagewiki.frontmatter``."""

from __future__ import annotations

from pagewiki.frontmatter import Frontmatter, parse_frontmatter


class TestParseFrontmatter:
    def test_no_frontmatter(self) -> None:
        assert parse_frontmatter("# Hello\nworld") == Frontmatter()

    def test_basic_tags_inline(self) -> None:
        text = "---\ntags: [research, ml]\ndate: 2024-11-15\n---\n# Body"
        fm = parse_frontmatter(text)
        assert fm.tags == ["research", "ml"]
        assert fm.date == "2024-11-15"

    def test_dash_list_tags(self) -> None:
        text = "---\ntags:\n  - alpha\n  - beta\n---\nBody"
        fm = parse_frontmatter(text)
        assert fm.tags == ["alpha", "beta"]

    def test_aliases(self) -> None:
        text = '---\naliases: ["Transformer Paper", "Attention"]\n---\n'
        fm = parse_frontmatter(text)
        assert fm.aliases == ["Transformer Paper", "Attention"]

    def test_singular_tag(self) -> None:
        text = "---\ntag: research\n---\n"
        fm = parse_frontmatter(text)
        assert fm.tags == ["research"]

    def test_raw_dict_preserved(self) -> None:
        text = "---\ntags: [a]\ncustom_key: hello\n---\n"
        fm = parse_frontmatter(text)
        assert fm.raw["custom_key"] == "hello"

    def test_empty_frontmatter(self) -> None:
        text = "---\n---\nBody"
        fm = parse_frontmatter(text)
        assert fm == Frontmatter()

    def test_no_closing_fence(self) -> None:
        text = "---\ntags: [a]\n# Body"
        fm = parse_frontmatter(text)
        assert fm == Frontmatter()

    def test_quoted_values_stripped(self) -> None:
        text = '---\ntags: ["ml", "ai"]\n---\n'
        fm = parse_frontmatter(text)
        assert fm.tags == ["ml", "ai"]

    def test_date_only(self) -> None:
        text = "---\ndate: 2025-01\n---\n"
        fm = parse_frontmatter(text)
        assert fm.date == "2025-01"
        assert fm.tags == []
