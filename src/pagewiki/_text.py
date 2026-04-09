"""Shared text helpers used across pagewiki modules.

Having these as a separate module lets both ``pageindex_adapter``
(for h1-flatten title matching) and ``wiki_links`` (for
``[[target]]`` resolution) reuse the exact same normalization rule
without introducing a cross-module import cycle.

The normalization rule is intentionally simple:

  1. Strip leading/trailing whitespace.
  2. Lowercase.
  3. Collapse any internal whitespace runs to single spaces.

That covers the common cases — Obsidian users write titles as
``Research Paper``, ``research paper``, or ``  Research  Paper  ``
interchangeably — without introducing heavier transforms (Unicode
folding, punctuation stripping) that would risk false positives.
"""

from __future__ import annotations


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy equality.

    The normalized form is whitespace- and case-insensitive so
    ``"Research Paper"``, ``"research paper"``, and
    ``"  Research   Paper  "`` all hash to the same key.
    """
    return " ".join(title.strip().lower().split())


def titles_match(a: str, b: str) -> bool:
    """Return True if ``a`` and ``b`` normalize to the same key."""
    return normalize_title(a) == normalize_title(b)
