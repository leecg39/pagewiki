"""v0.4 incremental re-indexing + mtime-based change detection.

Persists the mtime snapshot of every note after a scan so subsequent
runs can quickly detect which notes were added, modified, or deleted
without re-reading file content.

The watch loop polls the vault directory at a configurable interval
and reports changes in real time — useful for keeping the PageIndex
cache warm while editing notes in Obsidian.

State file: ``{vault}/.pagewiki-cache/scan-state.json``

Design notes
------------

* Poll-based rather than inotify/fsevents — zero external
  dependencies and works identically on macOS/Linux/Windows.
* The scan state is a simple ``{relative_path: mtime_ns}`` dict.
  Deletes are detected by comparing keys, not by filesystem events.
* The state file is written atomically (write-then-rename) so a
  crash mid-scan never corrupts the previous snapshot.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

CACHE_DIR_NAME = ".pagewiki-cache"
STATE_FILENAME = "scan-state.json"

DEFAULT_POLL_INTERVAL = 10  # seconds


@dataclass(frozen=True)
class FileChange:
    """One detected change in the vault."""

    path: str  # vault-relative path
    kind: str  # "added" | "modified" | "deleted"


@dataclass
class ChangeSet:
    """Result of comparing current filesystem state against a previous snapshot."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def as_file_changes(self) -> list[FileChange]:
        """Flat list of all changes for iteration."""
        changes: list[FileChange] = []
        for p in self.added:
            changes.append(FileChange(p, "added"))
        for p in self.modified:
            changes.append(FileChange(p, "modified"))
        for p in self.deleted:
            changes.append(FileChange(p, "deleted"))
        return changes


def _state_path(vault_root: Path) -> Path:
    """Return the path to the scan state JSON file."""
    return vault_root / CACHE_DIR_NAME / STATE_FILENAME


def _snapshot_mtimes(
    vault_root: Path,
    subfolder: str | None = None,
) -> dict[str, int]:
    """Walk the vault and collect ``{relative_path: mtime_ns}`` for every ``.md`` file."""
    scan_root = vault_root / subfolder if subfolder else vault_root
    if not scan_root.exists():
        return {}

    mtimes: dict[str, int] = {}
    for md_file in sorted(scan_root.rglob("*.md")):
        # Skip hidden directories (like .pagewiki-cache, .obsidian)
        parts = md_file.relative_to(vault_root).parts
        if any(p.startswith(".") for p in parts):
            continue
        rel = str(md_file.relative_to(vault_root))
        try:
            mtimes[rel] = md_file.stat().st_mtime_ns
        except OSError:
            continue
    return mtimes


def save_state(
    vault_root: Path,
    subfolder: str | None = None,
) -> dict[str, int]:
    """Snapshot the current mtimes and persist to the state file.

    Returns the snapshot dict.
    """
    mtimes = _snapshot_mtimes(vault_root, subfolder)

    state_file = _state_path(vault_root)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "subfolder": subfolder,
        "mtimes": mtimes,
    }

    # Atomic write via temp file
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.rename(state_file)

    return mtimes


def load_state(vault_root: Path) -> dict[str, int] | None:
    """Load the previously saved mtime snapshot.

    Returns ``None`` if no state file exists or it's corrupt.
    """
    state_file = _state_path(vault_root)
    if not state_file.exists():
        return None

    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    mtimes = payload.get("mtimes")
    if not isinstance(mtimes, dict):
        return None

    return mtimes


def detect_changes(
    vault_root: Path,
    subfolder: str | None = None,
) -> ChangeSet:
    """Compare current filesystem state against the saved snapshot.

    If no previous snapshot exists, all current files are reported as
    "added" (equivalent to a first-run full scan).
    """
    previous = load_state(vault_root)
    current = _snapshot_mtimes(vault_root, subfolder)

    if previous is None:
        return ChangeSet(added=sorted(current.keys()))

    changes = ChangeSet()

    # Added or modified
    for path, mtime in sorted(current.items()):
        if path not in previous:
            changes.added.append(path)
        elif previous[path] != mtime:
            changes.modified.append(path)

    # Deleted
    for path in sorted(previous.keys()):
        if path not in current:
            changes.deleted.append(path)

    return changes


def watch_loop(
    vault_root: Path,
    subfolder: str | None = None,
    *,
    interval: int = DEFAULT_POLL_INTERVAL,
    callback: Callable[[ChangeSet], None] | None = None,
    max_cycles: int | None = None,
) -> None:
    """Poll-based file watcher loop.

    Args:
        vault_root: Obsidian vault root.
        subfolder: Optional subfolder scope.
        interval: Seconds between polls.
        callback: Called with each non-empty ``ChangeSet``. If ``None``,
            changes are printed to stdout.
        max_cycles: Stop after this many poll cycles (for testing).
            ``None`` means run forever.
    """
    # Use existing state if available; only take a fresh snapshot
    # when there is no prior state (first run).
    if load_state(vault_root) is None:
        save_state(vault_root, subfolder)

    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        time.sleep(interval)
        cycles += 1

        changes = detect_changes(vault_root, subfolder)
        if changes.has_changes:
            # Update the snapshot
            save_state(vault_root, subfolder)
            if callback is not None:
                callback(changes)
