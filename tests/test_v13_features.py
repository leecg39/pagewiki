"""Tests for v0.13: chat flags, real token counts, cross-vault+decompose,
usage-report CSV/JSON, retrieval package split.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from pagewiki.retrieval import run_cross_vault_retrieval
from pagewiki.tree import NoteTier, TreeNode
from pagewiki.usage import UsageTracker
from pagewiki.usage_store import UsageStore

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval package re-export sanity (v0.13 refactor)
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrievalPackage:
    def test_public_symbols_importable(self) -> None:
        """All public symbols remain importable from pagewiki.retrieval."""
        from pagewiki.retrieval import (
            ChatFn,
            EventCallback,
            RetrievalResult,
            TraceStep,
            run_cross_vault_retrieval,
            run_decomposed_retrieval,
            run_retrieval,
        )
        assert callable(run_retrieval)
        assert callable(run_decomposed_retrieval)
        assert callable(run_cross_vault_retrieval)
        assert RetrievalResult.__name__ == "RetrievalResult"
        assert TraceStep.__name__ == "TraceStep"
        # ChatFn / EventCallback are type aliases — just make sure
        # the import succeeds without raising.
        assert ChatFn is not None
        assert EventCallback is not None

    def test_private_helpers_still_exported(self) -> None:
        """Tests that probe internals keep working after the split."""
        from pagewiki.retrieval import (
            _children_as_summaries,
            _load_note_content,
            _node_as_summary,
            _promote_to_note,
        )
        assert callable(_load_note_content)
        assert callable(_node_as_summary)
        assert callable(_children_as_summaries)
        assert callable(_promote_to_note)

    def test_submodule_imports_work(self) -> None:
        """Direct submodule imports should also work for power users."""
        from pagewiki.retrieval import core as core_mod
        from pagewiki.retrieval import helpers as helper_mod
        from pagewiki.retrieval import types as types_mod

        assert callable(core_mod.run_retrieval)
        assert callable(helper_mod._load_note_content)
        assert types_mod.TraceStep.__name__ == "TraceStep"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-vault + decompose composition
# ─────────────────────────────────────────────────────────────────────────────


def _one_note_tree(tmp_path: Path, name: str) -> TreeNode:
    note = tmp_path / f"{name}.md"
    note.write_text(f"{name} content", encoding="utf-8")
    return TreeNode(
        node_id="",
        title=name,
        kind="folder",
        children=[
            TreeNode(
                node_id=f"{name}.md",
                title=name.capitalize(),
                kind="note",
                tier=NoteTier.ATOMIC,
                file_path=note,
                summary=f"{name} summary",
            ),
        ],
    )


class TestCrossVaultDecomposeCombination:
    def test_decompose_flag_forwarded_to_per_vault(self, tmp_path: Path) -> None:
        v1 = tmp_path / "v1"
        v1.mkdir()
        v2 = tmp_path / "v2"
        v2.mkdir()
        root1 = _one_note_tree(v1, "alpha")
        root2 = _one_note_tree(v2, "beta")

        # Script enough responses for:
        # - vault1 decompose (SINGLE fallback) → select → eval → answer
        # - vault2 decompose (SINGLE fallback) → select → eval → answer
        # - final cross-vault synthesis
        responses = iter([
            "SINGLE",
            "SELECT: alpha.md",
            "SUFFICIENT: ok",
            "v1 답변",
            "SINGLE",
            "SELECT: beta.md",
            "SUFFICIENT: ok",
            "v2 답변",
            "최종 합성 답변",
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        result = run_cross_vault_retrieval(
            "query",
            [root1, root2],
            fake_chat,
            vault_labels=["v1", "v2"],
            decompose=True,
        )
        assert result.answer == "최종 합성 답변"
        # Both vault citations prefixed.
        assert any("v1::" in c for c in result.cited_nodes)
        assert any("v2::" in c for c in result.cited_nodes)

    def test_decompose_false_uses_plain_retrieval(self, tmp_path: Path) -> None:
        v1 = tmp_path / "v1"
        v1.mkdir()
        root1 = _one_note_tree(v1, "alpha")

        responses = iter([
            "SELECT: alpha.md",
            "SUFFICIENT: ok",
            "v1 답변",
            "최종",
        ])

        def fake_chat(prompt: str) -> str:
            return next(responses)

        result = run_cross_vault_retrieval(
            "query", [root1], fake_chat, vault_labels=["v1"], decompose=False,
        )
        assert result.answer == "최종"


# ─────────────────────────────────────────────────────────────────────────────
# usage-report CSV/JSON export
# ─────────────────────────────────────────────────────────────────────────────


class TestUsageReportFormats:
    @pytest.fixture
    def populated_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "usage.db"
        store = UsageStore(db)
        store.record("select", 500, 10, 1.0)
        store.record("evaluate", 800, 20, 1.5)
        store.record("select", 300, 5, 0.8)
        store.close()
        return db

    def test_json_output_is_valid_json(self, populated_db: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(populated_db), "--format", "json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total"]["calls"] == 3
        assert payload["total"]["prompt"] == 1600
        assert "select" in payload["by_phase"]
        assert payload["by_phase"]["select"]["calls"] == 2

    def test_json_output_has_no_rich_markup(self, populated_db: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(populated_db), "--format", "json"],
        )
        assert result.exit_code == 0
        # No Rich color codes or bracket markup.
        assert "\x1b[" not in result.output
        assert "[bold" not in result.output

    def test_csv_output_is_parseable(self, populated_db: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(populated_db), "--format", "csv"],
        )
        assert result.exit_code == 0
        reader = csv.reader(io.StringIO(result.output))
        rows = list(reader)
        header = rows[0]
        assert header == ["section", "key", "calls", "prompt", "completion", "elapsed"]

        # Should contain a total row + phase rows.
        sections = {row[0] for row in rows[1:]}
        assert "total" in sections
        assert "phase" in sections

    def test_csv_output_includes_recent_events(self, populated_db: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "usage-report",
                "--db",
                str(populated_db),
                "--format",
                "csv",
                "--recent",
                "10",
            ],
        )
        assert result.exit_code == 0
        reader = csv.reader(io.StringIO(result.output))
        rows = list(reader)
        sections = {row[0] for row in rows[1:]}
        assert "event" in sections

    def test_table_format_still_works(self, populated_db: Path) -> None:
        """Default table output should be unchanged."""
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(populated_db)],
        )
        assert result.exit_code == 0
        assert "total_calls=3" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Server real token counts (v0.13)
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


class TestServerRealTokens:
    def test_sse_usage_uses_shared_tracker_delta(self, tmp_path: Path) -> None:
        """When the underlying chat_fn records real tokens into the shared
        tracker, the SSE stream's per-request tracker should pick up those
        exact values instead of char/3 estimates."""
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        shared_tracker = UsageTracker()

        # Custom chat_fn that pretends LiteLLM returned real token counts
        # (777 prompt / 88 completion) and records them into the tracker.
        script_idx = {"i": 0}

        def tracked_chat(prompt: str) -> str:
            script = [
                "요약",
                "SELECT: Notes/note.md",
                "SUFFICIENT: ok",
                "답변입니다",
            ]
            i = min(script_idx["i"], len(script) - 1)
            script_idx["i"] += 1
            shared_tracker.record("other", 777, 88, 0.1)
            return script[i]

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=tracked_chat,
        )
        state.tracker = shared_tracker

        app = create_app(state)
        tc = TestClient(app)

        with tc.stream("POST", "/ask/stream", json={"query": "test"}) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        # Parse usage frames from the SSE body.
        usage_payloads = []
        for frame in body.split("\n\n"):
            if "event: usage" in frame:
                for line in frame.split("\n"):
                    if line.startswith("data: "):
                        usage_payloads.append(json.loads(line[6:]))

        assert usage_payloads
        last = usage_payloads[-1]
        # The per-request tracker should reflect the REAL 777/88 tokens
        # (multiplied by however many calls retrieval made), NOT the
        # char/3 estimate of the input prompt.
        assert last["prompt_tokens"] >= 777
        assert last["completion_tokens"] >= 88
        # 777 is not a char/3-plausible value for our tiny scripted prompts,
        # so its presence proves the real delta path is live.
        assert last["prompt_tokens"] % 777 == 0
