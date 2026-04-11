"""Stability assertions for the v1.0.0 release.

These tests lock in the public surface so future 1.x changes can't
silently remove a shipped symbol or break a documented behavior.
If any of these fail, we're looking at a 2.0 candidate — not a
1.x minor.
"""

from __future__ import annotations

from pathlib import Path


class TestVersionAndMetadata:
    def test_version_is_1_x(self) -> None:
        import pagewiki
        # 1.x stable line — anything 1.y.z is acceptable.
        assert pagewiki.__version__.startswith("1.")

    def test_pyproject_marks_production_stable(self) -> None:
        text = Path("/home/user/pagewiki/pyproject.toml").read_text()
        assert 'version = "1.' in text
        assert "Development Status :: 5 - Production/Stable" in text

    def test_plugin_manifest_matches(self) -> None:
        import json

        import pagewiki
        data = json.loads(
            Path("/home/user/pagewiki/obsidian-plugin/manifest.json").read_text()
        )
        # Plugin manifest stays in lockstep with the Python package.
        assert data["version"] == pagewiki.__version__

    def test_changelog_has_1_0_entry(self) -> None:
        text = Path("/home/user/pagewiki/CHANGELOG.md").read_text()
        assert "[1.0.0]" in text

    def test_release_notes_exist(self) -> None:
        notes = Path("/home/user/pagewiki/docs/RELEASE_NOTES_v1.0.md")
        assert notes.exists()
        assert len(notes.read_text()) > 1000


class TestPublicAPI:
    """Lock in the public importable surface so 1.x can't remove it."""

    def test_top_level_symbols(self) -> None:
        import pagewiki
        # __version__ is the only guaranteed top-level symbol.
        assert hasattr(pagewiki, "__version__")

    def test_retrieval_package_exports(self) -> None:
        from pagewiki.retrieval import (
            ChatFn,
            EventCallback,
            RetrievalResult,
            SystemChatFn,
            TraceStep,
            run_cross_vault_retrieval,
            run_decomposed_retrieval,
            run_retrieval,
        )
        assert callable(run_retrieval)
        assert callable(run_decomposed_retrieval)
        assert callable(run_cross_vault_retrieval)
        # Type aliases must still import cleanly.
        assert ChatFn is not None
        assert SystemChatFn is not None
        assert EventCallback is not None
        # Dataclasses.
        assert RetrievalResult.__name__ == "RetrievalResult"
        assert TraceStep.__name__ == "TraceStep"

    def test_usage_exports(self) -> None:
        from pagewiki.usage import (
            UsageEvent,
            UsageTracker,
            make_tracking_chat_fn,
            make_tracking_str_chat_fn,
        )
        assert UsageEvent.__name__ == "UsageEvent"
        assert UsageTracker.__name__ == "UsageTracker"
        assert callable(make_tracking_chat_fn)
        assert callable(make_tracking_str_chat_fn)

    def test_usage_store_exports(self) -> None:
        from pagewiki.usage_store import (
            PersistedEvent,
            UsageStore,
            UsageSummary,
        )
        assert UsageStore.__name__ == "UsageStore"
        assert PersistedEvent.__name__ == "PersistedEvent"
        assert UsageSummary.__name__ == "UsageSummary"

    def test_tree_and_vault_exports(self) -> None:
        from pagewiki.tree import NoteTier, TreeNode
        from pagewiki.vault import (
            build_long_subtrees,
            filter_tree,
            scan_folder,
            scan_multi_vault,
            summarize_atomic_notes,
            vault_for_note,
        )
        assert NoteTier.MICRO is not None
        assert TreeNode.__name__ == "TreeNode"
        assert callable(scan_folder)
        assert callable(scan_multi_vault)
        assert callable(filter_tree)
        assert callable(vault_for_note)
        assert callable(build_long_subtrees)
        assert callable(summarize_atomic_notes)

    def test_prompts_exports(self) -> None:
        from pagewiki.prompts import (
            EVALUATE_SYSTEM,
            FINAL_ANSWER_SYSTEM,
            SELECT_NODE_SYSTEM,
            NodeSummary,
            build_retry_prompt,
            decompose_query_prompt,
            evaluate_prompt,
            final_answer_prompt,
            select_node_prompt,
        )
        # System constants must be strings (stable Ollama KV cache).
        assert isinstance(SELECT_NODE_SYSTEM, str)
        assert isinstance(EVALUATE_SYSTEM, str)
        assert isinstance(FINAL_ANSWER_SYSTEM, str)
        # Prompt builders callable.
        assert callable(select_node_prompt)
        assert callable(evaluate_prompt)
        assert callable(final_answer_prompt)
        assert callable(decompose_query_prompt)
        assert callable(build_retry_prompt)
        assert NodeSummary.__name__ == "NodeSummary"

    def test_cache_exports(self) -> None:
        from pagewiki.cache import (
            ADAPTER_VERSION,
            CACHE_DIR_NAME,
            CacheKey,
            SummaryCache,
            TreeCache,
        )
        assert CACHE_DIR_NAME == ".pagewiki-cache"
        assert isinstance(ADAPTER_VERSION, str)
        assert TreeCache.__name__ == "TreeCache"
        assert SummaryCache.__name__ == "SummaryCache"
        assert CacheKey.__name__ == "CacheKey"


class TestCLIStability:
    """Every shipped CLI command must remain reachable."""

    def test_all_commands_registered(self) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        for cmd in ("scan", "ask", "chat", "compile", "watch", "vaults", "serve", "usage-report"):
            assert cmd in result.output, f"missing command: {cmd}"

    def test_ask_has_v1_flags(self) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["ask", "--help"])
        assert result.exit_code == 0
        # Sample a representative set from each version era.
        for flag in (
            "--vault",  # v0.1
            "--folder",  # v0.1
            "--decompose",  # v0.7
            "--max-workers",  # v0.7
            "--usage",  # v0.8
            "--max-tokens",  # v0.9
            "--json-mode",  # v0.10
            "--reuse-context",  # v0.10
            "--extra-vault",  # v0.7
            "--tag",  # v0.6
            "--after",  # v0.6
            "--before",  # v0.6
            "--token-split",  # v0.14
            "--prompt-cache",  # v0.14
            "--per-vault",  # v0.12
            "--allow-partial",  # v0.16
            "--retry-failed",  # v0.17
        ):
            assert flag in result.output, f"missing flag on ask: {flag}"

    def test_version_flag_reports_1_x(self) -> None:
        from click.testing import CliRunner

        import pagewiki
        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert pagewiki.__version__ in result.output
