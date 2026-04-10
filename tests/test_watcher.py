"""Tests for v0.4 incremental re-indexing + mtime-based watcher.

All tests use tmp_path fixtures — no real vault or filesystem
watcher needed. The poll loop is tested with max_cycles=1 to
avoid blocking.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pagewiki.watcher import (
    ChangeSet,
    detect_changes,
    load_state,
    save_state,
    watch_loop,
)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Simple vault with 2 notes."""
    v = tmp_path / "vault"
    research = v / "Research"
    research.mkdir(parents=True)

    (research / "alpha.md").write_text("# Alpha\ncontent", encoding="utf-8")
    (research / "beta.md").write_text("# Beta\ncontent", encoding="utf-8")
    return v


# ─────────────────────────────────────────────────────────────────────────────
# save_state / load_state
# ─────────────────────────────────────────────────────────────────────────────


class TestScanState:
    def test_save_and_load_roundtrip(self, vault: Path) -> None:
        save_state(vault, "Research")
        loaded = load_state(vault)

        assert loaded is not None
        assert "Research/alpha.md" in loaded
        assert "Research/beta.md" in loaded
        assert len(loaded) == 2

    def test_load_missing_state_returns_none(self, vault: Path) -> None:
        assert load_state(vault) is None

    def test_load_corrupt_state_returns_none(self, vault: Path) -> None:
        state_file = vault / ".pagewiki-cache" / "scan-state.json"
        state_file.parent.mkdir(parents=True)
        state_file.write_text("not json }{", encoding="utf-8")

        assert load_state(vault) is None

    def test_skips_hidden_directories(self, vault: Path) -> None:
        hidden = vault / ".obsidian"
        hidden.mkdir()
        (hidden / "config.md").write_text("hidden", encoding="utf-8")

        mtimes = save_state(vault, None)

        # .obsidian/config.md should not appear
        assert not any(".obsidian" in k for k in mtimes)

    def test_subfolder_scope(self, vault: Path) -> None:
        other = vault / "Other"
        other.mkdir()
        (other / "gamma.md").write_text("# Gamma", encoding="utf-8")

        mtimes = save_state(vault, "Research")

        # Only Research/ notes, not Other/
        assert "Research/alpha.md" in mtimes
        assert "Other/gamma.md" not in mtimes


# ─────────────────────────────────────────────────────────────────────────────
# detect_changes
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectChanges:
    def test_first_run_all_added(self, vault: Path) -> None:
        changes = detect_changes(vault, "Research")

        assert len(changes.added) == 2
        assert len(changes.modified) == 0
        assert len(changes.deleted) == 0

    def test_no_changes_after_save(self, vault: Path) -> None:
        save_state(vault, "Research")
        changes = detect_changes(vault, "Research")

        assert not changes.has_changes

    def test_detects_new_file(self, vault: Path) -> None:
        save_state(vault, "Research")

        # Add a new note
        (vault / "Research" / "gamma.md").write_text("# Gamma", encoding="utf-8")

        changes = detect_changes(vault, "Research")
        assert "Research/gamma.md" in changes.added

    def test_detects_modified_file(self, vault: Path) -> None:
        save_state(vault, "Research")

        # Modify alpha — need to ensure mtime actually changes
        time.sleep(0.01)
        alpha = vault / "Research" / "alpha.md"
        alpha.write_text("# Alpha\nmodified content", encoding="utf-8")

        changes = detect_changes(vault, "Research")
        assert "Research/alpha.md" in changes.modified

    def test_detects_deleted_file(self, vault: Path) -> None:
        save_state(vault, "Research")

        # Delete beta
        (vault / "Research" / "beta.md").unlink()

        changes = detect_changes(vault, "Research")
        assert "Research/beta.md" in changes.deleted

    def test_mixed_changes(self, vault: Path) -> None:
        save_state(vault, "Research")

        time.sleep(0.01)
        (vault / "Research" / "alpha.md").write_text("modified", encoding="utf-8")
        (vault / "Research" / "beta.md").unlink()
        (vault / "Research" / "gamma.md").write_text("new", encoding="utf-8")

        changes = detect_changes(vault, "Research")
        assert changes.added == ["Research/gamma.md"]
        assert changes.modified == ["Research/alpha.md"]
        assert changes.deleted == ["Research/beta.md"]
        assert changes.total == 3


# ─────────────────────────────────────────────────────────────────────────────
# ChangeSet
# ─────────────────────────────────────────────────────────────────────────────


class TestChangeSet:
    def test_empty_changeset(self) -> None:
        cs = ChangeSet()
        assert not cs.has_changes
        assert cs.total == 0
        assert cs.as_file_changes() == []

    def test_as_file_changes(self) -> None:
        cs = ChangeSet(added=["a.md"], modified=["b.md"], deleted=["c.md"])
        changes = cs.as_file_changes()
        assert len(changes) == 3
        assert changes[0].kind == "added"
        assert changes[1].kind == "modified"
        assert changes[2].kind == "deleted"


# ─────────────────────────────────────────────────────────────────────────────
# watch_loop
# ─────────────────────────────────────────────────────────────────────────────


class TestWatchLoop:
    def test_callback_on_change(self, vault: Path) -> None:
        """watch_loop should call the callback when changes are detected."""
        received: list[ChangeSet] = []

        # Save initial state, then add a file
        save_state(vault, "Research")
        (vault / "Research" / "gamma.md").write_text("new", encoding="utf-8")

        watch_loop(
            vault,
            "Research",
            interval=0,  # no actual sleep in test
            callback=lambda cs: received.append(cs),
            max_cycles=1,
        )

        assert len(received) == 1
        assert received[0].has_changes
        assert "Research/gamma.md" in received[0].added

    def test_no_callback_when_no_changes(self, vault: Path) -> None:
        received: list[ChangeSet] = []

        watch_loop(
            vault,
            "Research",
            interval=0,
            callback=lambda cs: received.append(cs),
            max_cycles=1,
        )

        # No callback because nothing changed after initial snapshot
        assert len(received) == 0
