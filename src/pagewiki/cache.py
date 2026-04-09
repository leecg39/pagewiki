"""On-disk cache for Layer 2 PageIndex sub-trees.

Rebuilding a LONG note's section tree for every ``pagewiki ask`` call
is expensive (one LLM call per sizeable section). This module persists
the built sub-trees in ``{vault_root}/.pagewiki-cache/trees/`` so that
subsequent scans can reuse them whenever the source note is unchanged.

Invalidation key
----------------

The cache key is a tuple of:

* ``abs_path``         — the absolute path of the note
* ``mtime_ns``         — nanosecond-precision modification time
* ``model_id``         — the LLM model id used to generate summaries
* ``adapter_version``  — bump this constant when the adapter output
                         schema changes in a way that makes old caches
                         semantically wrong

Any mismatch forces a rebuild. The file path itself is derived from a
SHA-1 of the note's absolute path so that vault re-orgs do not produce
filename collisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .tree import TreeNode

# Bump when the TreeNode serialization changes in an incompatible way.
ADAPTER_VERSION = "v0.1.2"

CACHE_DIR_NAME = ".pagewiki-cache"
TREES_SUBDIR = "trees"


@dataclass(frozen=True)
class CacheKey:
    """Composite key identifying one cached sub-tree build.

    ``model_id`` is part of the key because different models can
    produce different summaries; reusing a ``gemma4:e4b`` cache under
    a later ``gemma4:26b`` run would silently degrade quality.
    """

    abs_path: str
    mtime_ns: int
    model_id: str
    adapter_version: str = ADAPTER_VERSION

    def to_dict(self) -> dict[str, str | int]:
        return {
            "abs_path": self.abs_path,
            "mtime_ns": self.mtime_ns,
            "model_id": self.model_id,
            "adapter_version": self.adapter_version,
        }


class TreeCache:
    """Manages on-disk sub-tree caches under ``{vault_root}/.pagewiki-cache/``.

    The cache is deliberately simple: one JSON file per note, no LRU,
    no compaction. Users can safely ``rm -rf .pagewiki-cache`` to
    force a full rebuild.
    """

    def __init__(self, vault_root: Path) -> None:
        self.root = Path(vault_root).resolve() / CACHE_DIR_NAME / TREES_SUBDIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _file_for(self, note_path: Path) -> Path:
        """Return the JSON cache file path for one note.

        We hash the absolute path rather than mirroring the vault
        directory structure so that nested folders with Korean or
        filesystem-sensitive characters never break the cache write.
        """
        abs_str = str(Path(note_path).resolve())
        digest = hashlib.sha1(abs_str.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def load(self, note_path: Path, model_id: str) -> list[TreeNode] | None:
        """Return the cached sub-tree children for ``note_path`` if still valid.

        Returns ``None`` on any form of cache miss:
          * file does not exist
          * JSON is malformed
          * the stored key does not match the current
            ``(abs_path, mtime_ns, model_id, adapter_version)``
        """
        cache_file = self._file_for(note_path)
        if not cache_file.exists():
            return None

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        stored_key = payload.get("key", {})
        try:
            current_mtime = Path(note_path).stat().st_mtime_ns
        except OSError:
            return None

        if (
            stored_key.get("abs_path") != str(Path(note_path).resolve())
            or stored_key.get("mtime_ns") != current_mtime
            or stored_key.get("model_id") != model_id
            or stored_key.get("adapter_version") != ADAPTER_VERSION
        ):
            return None

        children = payload.get("children", [])
        try:
            return [TreeNode.model_validate(child) for child in children]
        except Exception:
            # Schema drift from an older pagewiki version.
            return None

    def save(
        self,
        note_path: Path,
        model_id: str,
        children: list[TreeNode],
    ) -> None:
        """Persist ``children`` as the cached sub-tree for ``note_path``."""
        try:
            mtime = Path(note_path).stat().st_mtime_ns
        except OSError:
            # If the file vanished between build and save, just skip —
            # better to rebuild next time than to persist stale state.
            return

        key = CacheKey(
            abs_path=str(Path(note_path).resolve()),
            mtime_ns=mtime,
            model_id=model_id,
        )
        payload = {
            "key": key.to_dict(),
            "children": [child.model_dump(mode="json") for child in children],
        }
        cache_file = self._file_for(note_path)
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_or_build(
        self,
        note_path: Path,
        model_id: str,
        builder: Callable[[], list[TreeNode]],
    ) -> tuple[list[TreeNode], bool]:
        """Return cached children if valid, otherwise call ``builder`` and save.

        Returns:
            A tuple ``(children, from_cache)`` so callers can report
            hit/miss counts in CLI progress output.
        """
        cached = self.load(note_path, model_id)
        if cached is not None:
            return cached, True

        built = builder()
        self.save(note_path, model_id, built)
        return built, False
