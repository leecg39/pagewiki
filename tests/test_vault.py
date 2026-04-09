"""Layer 1 classifier and wiki-link extraction tests.

These tests are pure I/O + pydantic and don't require Ollama or PageIndex,
so they can run in any CI environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pagewiki.tree import NoteTier
from pagewiki.vault import (
    ATOMIC_MAX_TOKENS,
    MICRO_MAX_TOKENS,
    classify,
    estimate_tokens,
    extract_wiki_links,
    scan_folder,
)


def test_classify_boundaries() -> None:
    assert classify(0) == NoteTier.MICRO
    assert classify(MICRO_MAX_TOKENS - 1) == NoteTier.MICRO
    assert classify(MICRO_MAX_TOKENS) == NoteTier.ATOMIC
    assert classify(ATOMIC_MAX_TOKENS - 1) == NoteTier.ATOMIC
    assert classify(ATOMIC_MAX_TOKENS) == NoteTier.LONG
    assert classify(10_000) == NoteTier.LONG


def test_estimate_tokens_monotonic() -> None:
    short = estimate_tokens("hello")
    longer = estimate_tokens("hello " * 100)
    assert longer > short
    assert estimate_tokens("") == 1  # floor


def test_extract_wiki_links() -> None:
    text = "See [[Alpha]] and [[Beta|alias]] and [[Gamma#section]]. No link here."
    links = extract_wiki_links(text)
    assert links == ["Alpha", "Beta", "Gamma"]


def test_scan_folder_classifies_three_tiers(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    research = vault / "Research"
    research.mkdir(parents=True)

    # micro: ~20 chars = ~7 tokens
    (research / "micro.md").write_text("Short note about X.", encoding="utf-8")
    # atomic: 500~3000 tokens → ~1500~9000 chars
    (research / "atomic.md").write_text("a " * 1500, encoding="utf-8")
    # long: > 3000 tokens → >9000 chars
    (research / "long.md").write_text("x " * 5000, encoding="utf-8")

    root = scan_folder(vault, "Research")
    notes = {n.title: n for n in root.walk() if n.kind == "note"}

    assert notes["micro"].tier == NoteTier.MICRO
    assert notes["atomic"].tier == NoteTier.ATOMIC
    assert notes["long"].tier == NoteTier.LONG


def test_scan_folder_ignores_dotfolders(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    folder = vault / "Notes"
    folder.mkdir(parents=True)
    (folder / "real.md").write_text("content", encoding="utf-8")

    # Should be skipped
    pagewiki_log = folder / ".pagewiki-log"
    pagewiki_log.mkdir()
    (pagewiki_log / "ignored.md").write_text("log", encoding="utf-8")

    root = scan_folder(vault, "Notes")
    titles = [n.title for n in root.walk() if n.kind == "note"]
    assert "real" in titles
    assert "ignored" not in titles


def test_scan_folder_raises_on_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_folder(tmp_path, "DoesNotExist")
