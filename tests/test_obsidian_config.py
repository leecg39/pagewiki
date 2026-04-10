"""Tests for ``pagewiki.obsidian_config`` — vault auto-discovery.

All subprocess calls are mocked via the ``runner`` injection seam,
and all filesystem reads go through the ``config_paths_provider``
seam. No real ``notesmd-cli`` or real ``obsidian.json`` is touched.

The ``_notesmd_cli_on_path`` helper (which calls ``shutil.which``)
is patched via ``monkeypatch`` so tests can simulate both
"installed" and "missing" worlds deterministically.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pagewiki.obsidian_config import (
    VaultInfo,
    build_open_command,
    discover_default_vault,
    list_known_vaults,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fake_runner(
    *,
    stdout: str = "",
    returncode: int = 0,
) -> callable:
    """Return a subprocess runner that always produces the given result."""

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv,
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    return _run


def _runner_raising(exc: Exception):
    """Return a runner that always raises the given exception."""

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise exc

    return _run


def _no_config_paths():
    return []


def _config_paths_from(*paths: Path):
    """Build a ``config_paths_provider`` that returns the given list."""

    def _provider() -> list[Path]:
        return list(paths)

    return _provider


def _write_obsidian_config(
    path: Path,
    vaults_payload: dict,
) -> None:
    """Write a minimal ``obsidian.json`` at ``path`` with the given vaults dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vaults": vaults_payload}), encoding="utf-8")


@pytest.fixture
def notesmd_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate notesmd-cli being on PATH."""
    monkeypatch.setattr(
        "pagewiki.obsidian_config.shutil.which",
        lambda name: "/fake/bin/notesmd-cli" if name == "notesmd-cli" else None,
    )


@pytest.fixture
def notesmd_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate notesmd-cli NOT being on PATH."""
    monkeypatch.setattr(
        "pagewiki.obsidian_config.shutil.which",
        lambda name: None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# discover_default_vault — notesmd-cli strategy
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscoverViaNotesmdCli:
    def test_print_default_returns_valid_path(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        vault_dir = tmp_path / "MyVault"
        vault_dir.mkdir()

        runner = _fake_runner(stdout=str(vault_dir) + "\n")
        result = discover_default_vault(
            runner=runner, config_paths_provider=_no_config_paths
        )
        assert result == vault_dir

    def test_print_default_ignores_path_that_does_not_exist(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        runner = _fake_runner(stdout="/nonexistent/vault\n")
        result = discover_default_vault(
            runner=runner, config_paths_provider=_no_config_paths
        )
        # Should fall through — notesmd said X but X doesn't exist.
        assert result is None

    def test_print_default_returncode_nonzero_is_skipped(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        runner = _fake_runner(stdout="", returncode=1)
        assert (
            discover_default_vault(
                runner=runner, config_paths_provider=_no_config_paths
            )
            is None
        )

    def test_print_default_empty_stdout_is_skipped(
        self, notesmd_installed
    ) -> None:
        runner = _fake_runner(stdout="   \n")
        assert (
            discover_default_vault(
                runner=runner, config_paths_provider=_no_config_paths
            )
            is None
        )

    def test_subprocess_oserror_is_swallowed(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        runner = _runner_raising(OSError("binary exploded"))
        # Should fall through to obsidian.json (empty here), then None.
        assert (
            discover_default_vault(
                runner=runner, config_paths_provider=_no_config_paths
            )
            is None
        )

    def test_subprocess_timeout_is_swallowed(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        runner = _runner_raising(
            subprocess.TimeoutExpired(cmd="notesmd-cli", timeout=5)
        )
        assert (
            discover_default_vault(
                runner=runner, config_paths_provider=_no_config_paths
            )
            is None
        )

    def test_notesmd_not_on_path_skips_strategy_1(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        """With notesmd-cli missing, the runner should not even be
        called — strategy 1 is skipped entirely."""
        called = {"n": 0}

        def runner(argv: list[str]):
            called["n"] += 1
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="/whatever\n", stderr=""
            )

        discover_default_vault(
            runner=runner, config_paths_provider=_no_config_paths
        )
        assert called["n"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# discover_default_vault — obsidian.json fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscoverViaObsidianConfig:
    def test_picks_vault_marked_open_true(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        vault_a = tmp_path / "VaultA"
        vault_b = tmp_path / "VaultB"
        vault_a.mkdir()
        vault_b.mkdir()

        config_path = tmp_path / "obsidian.json"
        _write_obsidian_config(
            config_path,
            {
                "id-a": {"path": str(vault_a), "open": False},
                "id-b": {"path": str(vault_b), "open": True},
            },
        )

        result = discover_default_vault(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(config_path),
        )
        assert result == vault_b

    def test_falls_back_to_first_existing_vault_when_none_open(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        vault_a = tmp_path / "VaultA"
        vault_a.mkdir()

        config_path = tmp_path / "obsidian.json"
        _write_obsidian_config(
            config_path,
            {
                "id-a": {"path": str(vault_a), "open": False},
            },
        )

        result = discover_default_vault(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(config_path),
        )
        assert result == vault_a

    def test_skips_vaults_whose_paths_no_longer_exist(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        real_vault = tmp_path / "RealVault"
        real_vault.mkdir()

        config_path = tmp_path / "obsidian.json"
        _write_obsidian_config(
            config_path,
            {
                "id-gone": {"path": "/nonexistent/GoneVault", "open": True},
                "id-real": {"path": str(real_vault), "open": False},
            },
        )

        result = discover_default_vault(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(config_path),
        )
        # The "open" vault is gone; should fall back to the real one.
        assert result == real_vault

    def test_corrupted_config_is_skipped(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "obsidian.json"
        config_path.write_text("not valid json at all }{", encoding="utf-8")

        result = discover_default_vault(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(config_path),
        )
        assert result is None

    def test_top_level_non_dict_json_is_skipped(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        """Regression for chatgpt-codex-connector P1 review on PR #3:

        ``_read_obsidian_config`` used to call ``.get('vaults')`` on
        whatever ``json.loads`` returned. A corrupt / hand-edited
        ``obsidian.json`` whose top-level value is a JSON list
        (``[]``), null, or scalar would crash discovery with
        ``AttributeError: 'list' object has no attribute 'get'``
        instead of gracefully falling through to the next candidate.

        The fix guards ``isinstance(data, dict)`` before touching
        ``.get``.
        """
        config_path = tmp_path / "obsidian.json"
        config_path.write_text("[]", encoding="utf-8")

        result = discover_default_vault(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(config_path),
        )
        assert result is None

    def test_missing_config_file_is_skipped(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        fake = tmp_path / "obsidian.json"  # never created
        result = discover_default_vault(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(fake),
        )
        assert result is None

    def test_first_existing_path_in_provider_wins(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        """If multiple candidate config paths are provided, the first
        one that exists determines the result — later files are not
        consulted."""
        vault_a = tmp_path / "VaultA"
        vault_b = tmp_path / "VaultB"
        vault_a.mkdir()
        vault_b.mkdir()

        first_config = tmp_path / "first" / "obsidian.json"
        second_config = tmp_path / "second" / "obsidian.json"
        _write_obsidian_config(
            first_config, {"id-a": {"path": str(vault_a), "open": True}}
        )
        _write_obsidian_config(
            second_config, {"id-b": {"path": str(vault_b), "open": True}}
        )

        result = discover_default_vault(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(first_config, second_config),
        )
        assert result == vault_a


# ─────────────────────────────────────────────────────────────────────────────
# list_known_vaults
# ─────────────────────────────────────────────────────────────────────────────


class TestListKnownVaults:
    def test_notesmd_json_list_parsed(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        va = tmp_path / "VaultA"
        vb = tmp_path / "VaultB"
        va.mkdir()
        vb.mkdir()

        runner = _fake_runner(
            stdout=json.dumps(
                [
                    {"name": "research", "path": str(va), "default": True},
                    {"name": "drafts", "path": str(vb)},
                ]
            )
        )
        vaults = list_known_vaults(
            runner=runner, config_paths_provider=_no_config_paths
        )
        assert len(vaults) == 2
        names = {v.name for v in vaults}
        assert names == {"research", "drafts"}
        defaults = [v for v in vaults if v.is_default]
        assert len(defaults) == 1 and defaults[0].name == "research"

    def test_notesmd_dict_list_parsed(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        """Some notesmd-cli versions may emit a dict-shaped JSON
        instead of a list. The parser should handle both."""
        va = tmp_path / "VaultA"
        va.mkdir()

        runner = _fake_runner(
            stdout=json.dumps(
                {"alpha": {"path": str(va), "isDefault": True}}
            )
        )
        vaults = list_known_vaults(
            runner=runner, config_paths_provider=_no_config_paths
        )
        assert len(vaults) == 1
        assert vaults[0].name == "alpha"
        assert vaults[0].is_default is True

    def test_falls_back_to_obsidian_config(
        self, notesmd_missing, tmp_path: Path
    ) -> None:
        va = tmp_path / "VaultA"
        va.mkdir()
        config_path = tmp_path / "obsidian.json"
        _write_obsidian_config(
            config_path, {"id-a": {"path": str(va), "open": True}}
        )

        vaults = list_known_vaults(
            runner=_runner_raising(OSError("not installed")),
            config_paths_provider=_config_paths_from(config_path),
        )
        assert len(vaults) == 1
        assert vaults[0].path == va
        assert vaults[0].is_default is True

    def test_both_strategies_empty_returns_empty(
        self, notesmd_missing
    ) -> None:
        vaults = list_known_vaults(
            runner=_runner_raising(OSError()),
            config_paths_provider=_no_config_paths,
        )
        assert vaults == []

    def test_notesmd_malformed_json_is_skipped(
        self, notesmd_installed
    ) -> None:
        runner = _fake_runner(stdout="not json")
        vaults = list_known_vaults(
            runner=runner, config_paths_provider=_no_config_paths
        )
        assert vaults == []


# ─────────────────────────────────────────────────────────────────────────────
# build_open_command
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildOpenCommand:
    def test_basic(self) -> None:
        assert build_open_command("paper") == ["notesmd-cli", "open", "paper"]

    def test_with_section(self) -> None:
        assert build_open_command("paper", section_anchor="Methods") == [
            "notesmd-cli",
            "open",
            "paper",
            "--section",
            "Methods",
        ]

    def test_with_vault_name(self) -> None:
        assert build_open_command(
            "paper", section_anchor="Methods", vault_name="obsidian"
        ) == [
            "notesmd-cli",
            "open",
            "paper",
            "--section",
            "Methods",
            "--vault",
            "obsidian",
        ]

    def test_empty_section_anchor_is_omitted(self) -> None:
        # An empty string for section_anchor should NOT add --section
        # to the argv — empty means "no anchor", not "anchor == ''".
        assert build_open_command("paper", section_anchor="") == [
            "notesmd-cli",
            "open",
            "paper",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy order — notesmd-cli takes precedence
# ─────────────────────────────────────────────────────────────────────────────


class TestStrategyOrder:
    def test_notesmd_success_skips_config_read(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        """If strategy 1 (notesmd-cli) succeeds, strategy 2
        (obsidian.json) should NOT be consulted."""
        vault_from_notesmd = tmp_path / "FromNotesmd"
        vault_from_config = tmp_path / "FromConfig"
        vault_from_notesmd.mkdir()
        vault_from_config.mkdir()

        runner = _fake_runner(stdout=str(vault_from_notesmd) + "\n")

        provider_called = {"n": 0}

        def tracking_provider() -> list[Path]:
            provider_called["n"] += 1
            cfg = tmp_path / "obsidian.json"
            _write_obsidian_config(
                cfg, {"id-x": {"path": str(vault_from_config), "open": True}}
            )
            return [cfg]

        result = discover_default_vault(
            runner=runner, config_paths_provider=tracking_provider
        )
        assert result == vault_from_notesmd
        assert provider_called["n"] == 0, (
            "strategy 2 should be skipped when strategy 1 succeeds"
        )

    def test_notesmd_failure_triggers_config_read(
        self, notesmd_installed, tmp_path: Path
    ) -> None:
        vault_from_config = tmp_path / "FromConfig"
        vault_from_config.mkdir()

        # notesmd-cli returns a nonexistent path — treated as failure.
        runner = _fake_runner(stdout="/nowhere\n")

        cfg = tmp_path / "obsidian.json"
        _write_obsidian_config(
            cfg, {"id-x": {"path": str(vault_from_config), "open": True}}
        )

        result = discover_default_vault(
            runner=runner, config_paths_provider=_config_paths_from(cfg),
        )
        assert result == vault_from_config


# ─────────────────────────────────────────────────────────────────────────────
# VaultInfo dataclass basics
# ─────────────────────────────────────────────────────────────────────────────


class TestVaultInfo:
    def test_is_frozen(self) -> None:
        info = VaultInfo(name="x", path=Path("/tmp/x"))
        with pytest.raises(AttributeError):
            info.name = "y"  # type: ignore[misc]

    def test_default_is_default_false(self) -> None:
        info = VaultInfo(name="x", path=Path("/tmp/x"))
        assert info.is_default is False
