"""Tests for ``pagewiki.cache.SummaryCache``."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pagewiki.cache import SummaryCache


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def note(vault: Path) -> Path:
    path = vault / "atomic.md"
    path.write_text("a " * 1500, encoding="utf-8")
    return path


class TestSummaryCacheHitMiss:
    def test_miss_on_first_access(self, vault: Path, note: Path) -> None:
        cache = SummaryCache(vault)
        assert cache.load(note, model_id="ollama/gemma4:26b") is None

    def test_hit_after_save(self, vault: Path, note: Path) -> None:
        cache = SummaryCache(vault)
        cache.save(note, "ollama/gemma4:26b", "요약 문장입니다")
        loaded = cache.load(note, "ollama/gemma4:26b")
        assert loaded == "요약 문장입니다"

    def test_miss_after_mtime_bump(self, vault: Path, note: Path) -> None:
        cache = SummaryCache(vault)
        cache.save(note, "ollama/gemma4:26b", "old summary")
        time.sleep(0.01)
        note.write_text(note.read_text() + "\nnew line\n", encoding="utf-8")
        assert cache.load(note, "ollama/gemma4:26b") is None

    def test_miss_on_model_change(self, vault: Path, note: Path) -> None:
        cache = SummaryCache(vault)
        cache.save(note, "ollama/gemma4:26b", "summary")
        assert cache.load(note, "ollama/gemma4:e4b") is None


class TestSummaryCacheWithVault:
    def test_summarize_uses_cache(self, tmp_path: Path) -> None:
        """Integration: summarize_atomic_notes skips LLM when cache hits."""
        from pagewiki.vault import scan_folder, summarize_atomic_notes

        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)
        (notes / "atomic.md").write_text("a " * 1500, encoding="utf-8")

        root = scan_folder(vault, "Notes")
        cache = SummaryCache(vault)

        calls: list[str] = []

        def fake_chat(prompt: str) -> str:
            calls.append(prompt)
            return "요약 결과"

        # First call: LLM invoked, cached.
        count1 = summarize_atomic_notes(
            root, fake_chat, summary_cache=cache, model_id="m",
        )
        assert count1 == 1
        assert len(calls) == 1

        # Second call on a fresh tree (simulates next `ask` invocation).
        root2 = scan_folder(vault, "Notes")
        calls.clear()
        count2 = summarize_atomic_notes(
            root2, fake_chat, summary_cache=cache, model_id="m",
        )
        assert count2 == 0  # no LLM calls — served from cache
        assert len(calls) == 0

        # Verify the summary was loaded from cache.
        for node in root2.walk():
            if node.kind == "note":
                assert node.summary == "요약 결과"
