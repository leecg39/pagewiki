"""Tests for ``pagewiki.cache.TreeCache``.

The cache must:
  * miss cleanly the first time a note is seen
  * hit on an unchanged note with the same model id
  * miss after the note's mtime changes
  * miss when the model id changes
  * miss when the stored adapter_version is stale
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pagewiki.cache import CACHE_DIR_NAME, TREES_SUBDIR, TreeCache
from pagewiki.tree import TreeNode


def _sample_children() -> list[TreeNode]:
    """A minimal two-level tree fragment used as cache payload."""
    return [
        TreeNode(
            node_id="sample.md#0001",
            title="Root",
            summary="root summary",
            kind="section",
            line_range=(1, 10),
            children=[
                TreeNode(
                    node_id="sample.md#0002",
                    title="Child",
                    summary="child summary",
                    kind="section",
                    line_range=(2, 9),
                )
            ],
        )
    ]


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def note(vault: Path) -> Path:
    """A throwaway markdown file we can mutate to bump mtime."""
    path = vault / "sample.md"
    path.write_text("# Root\n\nbody line one.\n", encoding="utf-8")
    return path


class TestTreeCacheHitMiss:
    def test_miss_on_first_access(self, vault: Path, note: Path) -> None:
        cache = TreeCache(vault)
        assert cache.load(note, model_id="ollama/gemma4:26b") is None

    def test_hit_after_save(self, vault: Path, note: Path) -> None:
        cache = TreeCache(vault)
        cache.save(note, "ollama/gemma4:26b", _sample_children())

        loaded = cache.load(note, "ollama/gemma4:26b")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].title == "Root"
        assert loaded[0].children[0].title == "Child"

    def test_miss_after_mtime_bump(self, vault: Path, note: Path) -> None:
        cache = TreeCache(vault)
        cache.save(note, "ollama/gemma4:26b", _sample_children())

        # Force a new mtime. Sleep 10 ms to defeat filesystem caching on
        # platforms where mtime has millisecond resolution.
        time.sleep(0.01)
        note.write_text(note.read_text() + "\nnew line\n", encoding="utf-8")
        assert cache.load(note, "ollama/gemma4:26b") is None

    def test_miss_on_model_change(self, vault: Path, note: Path) -> None:
        cache = TreeCache(vault)
        cache.save(note, "ollama/gemma4:26b", _sample_children())
        assert cache.load(note, "ollama/gemma4:e4b") is None

    def test_miss_on_adapter_version_change(
        self, vault: Path, note: Path
    ) -> None:
        cache = TreeCache(vault)
        cache.save(note, "ollama/gemma4:26b", _sample_children())

        # Tamper with the stored key to simulate an older pagewiki version.
        cache_file = next((vault / CACHE_DIR_NAME / TREES_SUBDIR).glob("*.json"))
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        payload["key"]["adapter_version"] = "v0.0.1"
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

        assert cache.load(note, "ollama/gemma4:26b") is None


class TestLoadOrBuild:
    def test_first_call_builds_then_caches(
        self, vault: Path, note: Path
    ) -> None:
        cache = TreeCache(vault)
        build_count = {"n": 0}

        def builder() -> list[TreeNode]:
            build_count["n"] += 1
            return _sample_children()

        children1, hit1 = cache.load_or_build(note, "m", builder)
        assert hit1 is False
        assert build_count["n"] == 1
        assert len(children1) == 1

        # Second call must not invoke the builder again.
        children2, hit2 = cache.load_or_build(
            note,
            "m",
            lambda: (_ for _ in ()).throw(
                RuntimeError("builder must not run on cache hit")
            ),
        )
        assert hit2 is True
        assert children2[0].title == "Root"

    def test_cache_directory_structure(
        self, vault: Path, note: Path
    ) -> None:
        cache = TreeCache(vault)
        cache.save(note, "m", _sample_children())
        assert (vault / CACHE_DIR_NAME / TREES_SUBDIR).is_dir()
        files = list((vault / CACHE_DIR_NAME / TREES_SUBDIR).glob("*.json"))
        assert len(files) == 1
        # Filename should be a hex sha1 digest (40 chars).
        stem = files[0].stem
        assert len(stem) == 40
        int(stem, 16)  # raises if non-hex
