"""Tests for v0.18 polish: CHANGELOG, error messages, Web UI keyboard shortcuts."""

from __future__ import annotations

from pathlib import Path

from pagewiki.webui import build_ui_html

# ─────────────────────────────────────────────────────────────────────────────
# CHANGELOG.md
# ─────────────────────────────────────────────────────────────────────────────


class TestChangelog:
    def test_changelog_file_exists(self) -> None:
        path = Path("/home/user/pagewiki/CHANGELOG.md")
        assert path.exists()

    def test_changelog_has_current_version(self) -> None:
        text = Path("/home/user/pagewiki/CHANGELOG.md").read_text()
        assert "[0.18.0]" in text
        assert "[0.17.0]" in text
        assert "[0.6.0]" in text

    def test_changelog_follows_keepachangelog(self) -> None:
        """Every version section should declare what was Added/Changed/etc."""
        text = Path("/home/user/pagewiki/CHANGELOG.md").read_text()
        # At least one Added heading per version.
        assert text.count("### Added") >= 5


# ─────────────────────────────────────────────────────────────────────────────
# Web UI keyboard shortcuts
# ─────────────────────────────────────────────────────────────────────────────


class TestWebUIKeyboardShortcuts:
    def test_esc_cancel_wired(self) -> None:
        html = build_ui_html()
        assert 'e.key === "Escape"' in html
        assert "abortController.abort" in html

    def test_placeholder_mentions_shortcuts(self) -> None:
        html = build_ui_html()
        assert "Cmd/Ctrl+Enter" in html
        assert "Esc" in html or "Escape" in html

    def test_buttons_have_title_tooltips(self) -> None:
        html = build_ui_html()
        assert 'title="Cmd/Ctrl+Enter"' in html
        assert 'title="Esc"' in html


# ─────────────────────────────────────────────────────────────────────────────
# CLI error message quality
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorMessages:
    def test_invalid_iso_timestamp_has_examples(self, tmp_path: Path) -> None:
        """usage-report --since with a bad date should print example formats."""
        from click.testing import CliRunner

        from pagewiki.cli import main
        from pagewiki.usage_store import UsageStore

        db = tmp_path / "usage.db"
        UsageStore(db).close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(db), "--since", "not-a-date"],
        )
        assert result.exit_code != 0
        # Error message should include actionable examples.
        assert "Examples" in result.output
        assert "2024" in result.output

    def test_vault_auto_discovery_error_lists_options(self) -> None:
        """When --vault is None and auto-discovery fails, the error should
        list all three remediation options."""
        # The actual error path requires mocking notesmd-cli + obsidian.json;
        # we smoke-test the error string presence in the source.
        src = Path("/home/user/pagewiki/src/pagewiki/cli.py").read_text()
        assert "Pass --vault explicitly" in src
        assert "notesmd-cli" in src
        assert "obsidian.json" in src
