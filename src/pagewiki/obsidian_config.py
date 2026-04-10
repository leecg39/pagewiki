"""Obsidian vault auto-discovery via notesmd-cli or direct config read.

pagewiki usually asks the user to pass ``--vault /path/to/vault`` on
every command. That's friction — especially when the vault path has
spaces in it (the common ``~/Documents/Obsidian Vault`` layout), or
when the user maintains several vaults and just wants "the one
Obsidian currently has open". This module removes that friction by
auto-discovering the vault path from two sources, in order:

  1. **notesmd-cli** — Yakitrak/notesmd-cli is a Go CLI for Obsidian
     vault automation. It already knows how to enumerate vaults via
     its ``list-vaults`` and ``print-default`` commands (both of which
     accept ``--path-only`` for scripting). If notesmd-cli is on
     ``PATH``, we shell out to it — it handles OS quirks, excluded-
     files filtering, and default-vault tracking for us.

  2. **Direct ``obsidian.json`` read** — if notesmd-cli is not
     installed, we fall back to parsing Obsidian's own config file.
     We probe a list of OS-specific candidate locations:

        * macOS:   ``~/Library/Application Support/obsidian/obsidian.json``
                   **and** ``~/.config/obsidian/obsidian.json`` (notesmd-cli's
                   convention; some users create the symlink this way).
        * Linux:   ``~/.config/obsidian/obsidian.json``
        * Windows: ``%APPDATA%/obsidian/obsidian.json``

     The JSON shape that Obsidian writes is
     ``{"vaults": {"<id>": {"path": "...", "ts": ..., "open": bool}}}``.
     We parse every entry, mark any vault with ``open: true`` as the
     default, and return the list.

  3. **None** — neither notesmd-cli nor a parseable obsidian.json.
     The caller is expected to show a helpful error explaining the
     two ways to fix it.

Design notes
------------

* All subprocess calls have a 5-second timeout. Discovery should
  never block the CLI noticeably.
* OSError from missing binaries, ``FileNotFoundError`` from missing
  config files, and ``json.JSONDecodeError`` from corrupted configs
  are all swallowed — they just cause the strategy to be skipped.
* Tests inject a fake ``runner`` and a fake ``config_paths`` so the
  whole discovery pipeline can be exercised without touching real
  filesystems or subprocesses.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

NOTESMD_CLI = "notesmd-cli"
_SUBPROCESS_TIMEOUT_SEC = 5


@dataclass(frozen=True)
class VaultInfo:
    """A single Obsidian vault entry discovered on the system."""

    name: str
    path: Path
    is_default: bool = False


# Runner / config_paths are injected by tests; production uses defaults.
Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]
ConfigPathsProvider = Callable[[], list[Path]]


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Production subprocess runner. Short timeout, captured text output."""
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SEC,
        check=False,
    )


def _default_config_paths() -> list[Path]:
    """OS-aware list of ``obsidian.json`` locations to probe.

    Ordering matters: the first existing file wins, so we put the
    OS-native location first and the notesmd-cli-compatible
    (``~/.config/obsidian``) location second.
    """
    home = Path.home()
    candidates: list[Path] = []
    if sys.platform == "darwin":
        candidates.append(
            home / "Library" / "Application Support" / "obsidian" / "obsidian.json"
        )
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "obsidian" / "obsidian.json")
    # The ``~/.config/obsidian/`` path is valid on Linux natively and
    # on macOS when the user has followed notesmd-cli's setup docs.
    candidates.append(home / ".config" / "obsidian" / "obsidian.json")
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: notesmd-cli
# ─────────────────────────────────────────────────────────────────────────────


def _notesmd_cli_on_path() -> bool:
    return shutil.which(NOTESMD_CLI) is not None


def _notesmd_cli_print_default(runner: Runner) -> Path | None:
    """Shell out to ``notesmd-cli print-default --path-only``.

    Returns ``None`` if notesmd-cli is not on PATH, the command
    failed, the output was empty, or the printed path does not
    actually exist on disk.
    """
    if not _notesmd_cli_on_path():
        return None
    try:
        result = runner([NOTESMD_CLI, "print-default", "--path-only"])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        return None
    return path


def _notesmd_cli_list_vaults(runner: Runner) -> list[VaultInfo]:
    """Shell out to ``notesmd-cli list-vaults --json``.

    The tool's ``--json`` flag is documented in its README. We accept
    either a list or a dict at the top level (notesmd-cli versions
    have differed on this) and coerce both to a list of
    ``VaultInfo``.
    """
    if not _notesmd_cli_on_path():
        return []
    try:
        result = runner([NOTESMD_CLI, "list-vaults", "--json"])
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return []

    entries: list[dict] = []
    if isinstance(data, list):
        entries = [e for e in data if isinstance(e, dict)]
    elif isinstance(data, dict):
        # Handle both {"vaults": [...]} and bare {name: {path, ...}} shapes.
        if "vaults" in data and isinstance(data["vaults"], list):
            entries = [e for e in data["vaults"] if isinstance(e, dict)]
        else:
            for name, info in data.items():
                if isinstance(info, dict):
                    e = dict(info)
                    e.setdefault("name", name)
                    entries.append(e)

    vaults: list[VaultInfo] = []
    for entry in entries:
        path_raw = entry.get("path") or entry.get("Path")
        if not path_raw:
            continue
        path = Path(str(path_raw)).expanduser()
        name = entry.get("name") or entry.get("Name") or path.name
        is_default = bool(
            entry.get("default")
            or entry.get("is_default")
            or entry.get("isDefault")
        )
        vaults.append(VaultInfo(name=str(name), path=path, is_default=is_default))
    return vaults


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: direct obsidian.json read
# ─────────────────────────────────────────────────────────────────────────────


def _read_obsidian_config(
    config_paths_provider: ConfigPathsProvider,
) -> list[VaultInfo]:
    """Parse the first existing ``obsidian.json`` from the candidate list.

    Obsidian writes its vault registry as
    ``{"vaults": {"<hash>": {"path": "...", "ts": ..., "open": bool}}}``.
    We convert every entry to a ``VaultInfo`` and mark any vault whose
    ``open`` flag is true as the default.
    """
    for path in config_paths_provider():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # Obsidian writes a top-level dict, but a corrupted /
        # hand-edited config can be a list, null, or scalar. Guard
        # before calling ``.get`` so we fall through to the next
        # candidate instead of crashing with AttributeError.
        if not isinstance(data, dict):
            continue

        vaults_data = data.get("vaults")
        if not isinstance(vaults_data, dict):
            continue

        vaults: list[VaultInfo] = []
        for _vault_id, info in vaults_data.items():
            if not isinstance(info, dict):
                continue
            path_raw = info.get("path")
            if not path_raw:
                continue
            vault_path = Path(str(path_raw)).expanduser()
            vaults.append(
                VaultInfo(
                    name=vault_path.name,
                    path=vault_path,
                    is_default=bool(info.get("open", False)),
                )
            )
        return vaults
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Public surface
# ─────────────────────────────────────────────────────────────────────────────


def discover_default_vault(
    *,
    runner: Runner | None = None,
    config_paths_provider: ConfigPathsProvider | None = None,
) -> Path | None:
    """Try to auto-discover the user's default Obsidian vault path.

    Strategy (first success wins):

      1. ``notesmd-cli print-default --path-only``
      2. First ``open: true`` vault in ``obsidian.json``
      3. First existing vault in ``obsidian.json`` (even if not open)
      4. ``None`` — no vault could be discovered

    The caller is expected to treat ``None`` as "show an error and
    exit"; this function never raises.
    """
    if runner is None:
        runner = _default_runner
    if config_paths_provider is None:
        config_paths_provider = _default_config_paths

    path = _notesmd_cli_print_default(runner)
    if path is not None:
        return path

    vaults = _read_obsidian_config(config_paths_provider)
    for vault in vaults:
        if vault.is_default and vault.path.exists():
            return vault.path
    for vault in vaults:
        if vault.path.exists():
            return vault.path

    return None


def list_known_vaults(
    *,
    runner: Runner | None = None,
    config_paths_provider: ConfigPathsProvider | None = None,
) -> list[VaultInfo]:
    """Return every vault the user has opened in Obsidian.

    Prefers ``notesmd-cli list-vaults --json`` if the binary is
    available; falls back to parsing ``obsidian.json`` directly.
    The two strategies are NOT merged — we use whichever one
    succeeds first. Duplicate entries across vaults are possible if
    the user has manually edited their config, but we do not
    deduplicate here (the caller can do that on ``path`` equality
    if needed).
    """
    if runner is None:
        runner = _default_runner
    if config_paths_provider is None:
        config_paths_provider = _default_config_paths

    vaults = _notesmd_cli_list_vaults(runner)
    if vaults:
        return vaults

    return _read_obsidian_config(config_paths_provider)


def build_open_command(
    note_title: str,
    *,
    section_anchor: str | None = None,
    vault_name: str | None = None,
) -> list[str]:
    """Build an argv list for ``notesmd-cli open <note> [--section <anchor>] [--vault <name>]``.

    This is a pure-function helper — it does NOT invoke the command.
    Callers that want to actually spawn notesmd-cli can feed the
    result straight into ``subprocess.run``. Callers that just want
    a copy-paste hint can feed it into ``shlex.join`` for a
    terminal-ready string.
    """
    argv: list[str] = [NOTESMD_CLI, "open", note_title]
    if section_anchor:
        argv.extend(["--section", section_anchor])
    if vault_name:
        argv.extend(["--vault", vault_name])
    return argv
