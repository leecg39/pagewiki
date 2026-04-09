"""Test the atomic note summarization pass."""

from __future__ import annotations

from pathlib import Path

from pagewiki.tree import NoteTier
from pagewiki.vault import scan_folder, summarize_atomic_notes


def test_summarize_atomic_notes_calls_llm_only_for_atomic(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    notes = vault / "Notes"
    notes.mkdir(parents=True)

    (notes / "micro.md").write_text("tiny", encoding="utf-8")
    (notes / "atomic.md").write_text("a " * 1500, encoding="utf-8")
    (notes / "long.md").write_text("x " * 5000, encoding="utf-8")

    root = scan_folder(vault, "Notes")

    calls: list[str] = []

    def fake_chat(prompt: str) -> str:
        calls.append(prompt)
        return f"요약 {len(calls)}"

    count = summarize_atomic_notes(root, fake_chat)

    # Exactly one atomic note → exactly one LLM call
    assert count == 1
    assert len(calls) == 1

    # Verify the summary landed on the ATOMIC node
    for node in root.walk():
        if node.kind == "note" and node.tier == NoteTier.ATOMIC:
            assert node.summary == "요약 1"
        if node.kind == "note" and node.tier == NoteTier.MICRO:
            assert node.summary == ""
        if node.kind == "note" and node.tier == NoteTier.LONG:
            assert node.summary == ""


def test_summarize_strips_quotes(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    notes = vault / "N"
    notes.mkdir(parents=True)
    (notes / "atomic.md").write_text("a " * 1500, encoding="utf-8")

    root = scan_folder(vault, "N")

    def fake_chat(prompt: str) -> str:
        return '"요약된 문장입니다"'

    summarize_atomic_notes(root, fake_chat)

    for node in root.walk():
        if node.tier == NoteTier.ATOMIC:
            assert node.summary == "요약된 문장입니다"


def test_summarize_skips_existing(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    notes = vault / "N"
    notes.mkdir(parents=True)
    (notes / "atomic.md").write_text("a " * 1500, encoding="utf-8")

    root = scan_folder(vault, "N")

    # Pre-fill summary
    for node in root.walk():
        if node.tier == NoteTier.ATOMIC:
            node.summary = "existing"

    calls: list[str] = []
    summarize_atomic_notes(root, lambda p: (calls.append(p), "new summary")[1])

    assert len(calls) == 0  # skipped
