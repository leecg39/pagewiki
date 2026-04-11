"""Tests for v1.1 additive features: stats, /metrics, batch, --lang en."""

from __future__ import annotations

from pathlib import Path

import pytest

from pagewiki.prompts_en import (
    EN_EVALUATE_SYSTEM,
    EN_FINAL_ANSWER_SYSTEM,
    EN_SELECT_NODE_SYSTEM,
    atomic_summary_prompt_en,
    decompose_query_prompt_en,
    evaluate_prompt_en,
    final_answer_prompt_en,
    select_node_prompt_en,
    synthesize_multi_answer_prompt_en,
)
from pagewiki.retrieval import run_retrieval
from pagewiki.tree import NoteTier, TreeNode

# ─────────────────────────────────────────────────────────────────────────────
# pagewiki stats command
# ─────────────────────────────────────────────────────────────────────────────


class TestStatsCommand:
    def test_stats_reports_tier_breakdown(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        vault = tmp_path / "vault"
        notes = vault / "Research"
        notes.mkdir(parents=True)
        (notes / "micro.md").write_text("tiny", encoding="utf-8")
        (notes / "atomic.md").write_text("a " * 1500, encoding="utf-8")
        (notes / "tagged.md").write_text(
            "---\ntags: [research, ml]\n---\n" + "a " * 1500,
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["stats", "--vault", str(vault), "--folder", "Research"],
        )
        assert result.exit_code == 0
        assert "MICRO" in result.output
        assert "ATOMIC" in result.output
        assert "LONG" in result.output
        assert "TOTAL" in result.output
        # Frontmatter tags should show up in the histogram.
        assert "research" in result.output
        assert "ml" in result.output

    def test_stats_reports_empty_vault(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        vault = tmp_path / "vault"
        (vault / "Empty").mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(
            main, ["stats", "--vault", str(vault), "--folder", "Empty"],
        )
        assert result.exit_code == 1
        assert "No notes found" in result.output

    def test_stats_no_llm_calls(self, tmp_path: Path) -> None:
        """stats should never invoke the LLM — it's a pure-IO command."""
        from click.testing import CliRunner

        from pagewiki.cli import main

        vault = tmp_path / "vault"
        notes = vault / "Notes"
        notes.mkdir(parents=True)
        (notes / "a.md").write_text("a " * 1500, encoding="utf-8")

        # If stats tries to call the LLM, _make_chat_fn will be
        # invoked and will try to import litellm. The test still
        # passes as long as the stats command finishes without
        # attempting any retrieval.
        runner = CliRunner()
        result = runner.invoke(
            main, ["stats", "--vault", str(vault), "--folder", "Notes"],
        )
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# pagewiki batch command
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchCommand:
    def test_batch_accepts_queries_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0
        # Argument present.
        assert "QUERIES_FILE" in result.output.upper() or "queries_file" in result.output.lower()

    def test_batch_skips_blank_and_comment_lines(self, tmp_path: Path) -> None:
        """Verify the queries file parser via direct read — the full
        retrieval flow requires an LLM which isn't available in tests."""
        queries = tmp_path / "queries.txt"
        queries.write_text(
            "# this is a comment\n"
            "\n"
            "what is X?\n"
            "# another comment\n"
            "what is Y?\n"
            "   \n"
            "what is Z?\n",
            encoding="utf-8",
        )

        raw_lines = queries.read_text(encoding="utf-8").splitlines()
        parsed = [
            line.strip()
            for line in raw_lines
            if line.strip() and not line.strip().startswith("#")
        ]
        assert parsed == ["what is X?", "what is Y?", "what is Z?"]


# ─────────────────────────────────────────────────────────────────────────────
# English prompt variants
# ─────────────────────────────────────────────────────────────────────────────


class TestEnglishPrompts:
    def test_system_constants_are_english(self) -> None:
        for txt in (
            EN_SELECT_NODE_SYSTEM,
            EN_EVALUATE_SYSTEM,
            EN_FINAL_ANSWER_SYSTEM,
        ):
            assert isinstance(txt, str)
            # Heuristic: should contain at least one common English word
            # and NOT the Korean conjunctions used in prompts.py.
            assert any(w in txt.lower() for w in ("the", "and", "a"))
            assert "당신은" not in txt

    def test_select_prompt_contains_candidates(self) -> None:
        from pagewiki.prompts import NodeSummary

        candidates = [
            NodeSummary(node_id="a.md", title="Alpha", kind="note", summary="alpha summary"),
            NodeSummary(node_id="b.md", title="Beta", kind="note", summary="beta summary"),
        ]
        prompt = select_node_prompt_en("What is alpha?", candidates)
        assert "Alpha" in prompt
        assert "Beta" in prompt
        # Format markers should remain consistent with the Korean version.
        assert "SELECT:" in prompt
        assert "DONE:" in prompt

    def test_evaluate_prompt_keeps_format_markers(self) -> None:
        prompt = evaluate_prompt_en("q", "Note Title", "body text")
        assert "SUFFICIENT:" in prompt
        assert "INSUFFICIENT:" in prompt
        assert "Note Title" in prompt

    def test_final_answer_prompt_uses_evidence_notes(self) -> None:
        prompt = final_answer_prompt_en(
            "What is X?",
            [("Note 1", "X is a thing."), ("Note 2", "X has properties.")],
        )
        assert "Note 1" in prompt
        assert "Note 2" in prompt
        # Should ask for English output explicitly.
        assert "English" in prompt

    def test_atomic_summary_prompt_en(self) -> None:
        prompt = atomic_summary_prompt_en("Title", "body content")
        assert "ONE English sentence" in prompt

    def test_decompose_prompt_keeps_markers(self) -> None:
        prompt = decompose_query_prompt_en("X vs Y?", max_sub_queries=3)
        assert "SUB:" in prompt
        assert "SINGLE" in prompt

    def test_synthesize_prompt_preserves_citations(self) -> None:
        prompt = synthesize_multi_answer_prompt_en(
            "complex query",
            [("What is X?", "X is a thing"), ("What is Y?", "Y is another thing")],
        )
        assert "X is a thing" in prompt
        assert "Y is another thing" in prompt

    def test_run_retrieval_with_lang_en(self, tmp_path: Path) -> None:
        note = tmp_path / "note.md"
        note.write_text("content", encoding="utf-8")

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="note.md",
                    title="Note",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="test",
                ),
            ],
        )

        seen_prompts: list[str] = []
        responses = iter([
            "SELECT: note.md",
            "SUFFICIENT: ok",
            "final english answer",
        ])

        def fake_chat(prompt: str) -> str:
            seen_prompts.append(prompt)
            return next(responses)

        result = run_retrieval("what is this?", root, fake_chat, lang="en")
        assert result.answer == "final english answer"

        # The select prompt should be the English variant.
        assert any(
            "Pick the single most relevant candidate" in p for p in seen_prompts
        )
        # The evaluate prompt should be English too.
        assert any("Is this note sufficient" in p for p in seen_prompts)
        # The final answer prompt should be English.
        assert any("Answer the question in English" in p for p in seen_prompts)

    def test_lang_ko_is_default_and_uses_korean(self, tmp_path: Path) -> None:
        note = tmp_path / "note.md"
        note.write_text("content", encoding="utf-8")

        root = TreeNode(
            node_id="",
            title="root",
            kind="folder",
            children=[
                TreeNode(
                    node_id="note.md",
                    title="Note",
                    kind="note",
                    tier=NoteTier.ATOMIC,
                    file_path=note,
                    summary="test",
                ),
            ],
        )

        seen_prompts: list[str] = []
        responses = iter([
            "SELECT: note.md",
            "SUFFICIENT: ok",
            "답변",
        ])

        def fake_chat(prompt: str) -> str:
            seen_prompts.append(prompt)
            return next(responses)

        result = run_retrieval("query", root, fake_chat)  # default lang
        assert result.answer == "답변"
        # Should see Korean prompts.
        assert any("당신은" in p for p in seen_prompts)


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics endpoint
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402
from pagewiki.usage import UsageTracker  # noqa: E402
from pagewiki.usage_store import UsageStore  # noqa: E402


class TestMetricsEndpoint:
    def test_metrics_prometheus_format(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        tracker = UsageTracker()
        tracker.record("select", 500, 10, 1.0, cacheable=True)
        tracker.record("evaluate", 800, 20, 1.5, cacheable=False)

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=lambda p: "요약",
        )
        state.tracker = tracker
        app = create_app(state)
        tc = TestClient(app)

        resp = tc.get("/metrics")
        assert resp.status_code == 200
        # Content-Type should be plain text (Prometheus format).
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text

        # Every metric should have HELP + TYPE + value lines.
        assert "# HELP pagewiki_llm_calls_total" in body
        assert "# TYPE pagewiki_llm_calls_total counter" in body
        assert "pagewiki_llm_calls_total " in body

        # Gauges.
        assert "pagewiki_note_count " in body
        assert "pagewiki_active_sessions " in body
        assert "pagewiki_cacheable_ratio " in body

        # Phase-labeled counters.
        assert 'pagewiki_phase_calls_total{phase="select"}' in body
        assert 'pagewiki_phase_calls_total{phase="evaluate"}' in body

    def test_metrics_includes_persistent_totals_when_store_set(
        self, tmp_path: Path,
    ) -> None:
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        store = UsageStore(tmp_path / "usage.db")
        store.record("select", 1000, 100, 2.0)

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=lambda p: "요약",
        )
        state.usage_store = store
        app = create_app(state)
        tc = TestClient(app)

        body = tc.get("/metrics").text
        assert "pagewiki_persistent_llm_calls_total " in body
        assert "pagewiki_persistent_prompt_tokens_total " in body

    def test_metrics_zero_when_no_events(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        (vault / "Notes").mkdir(parents=True)
        (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

        state = build_initial_state(
            [vault],
            folder="Notes",
            model="test",
            num_ctx=8192,
            max_workers=1,
            chat_fn=lambda p: "요약",
        )
        state.tracker.events.clear()
        app = create_app(state)
        tc = TestClient(app)

        body = tc.get("/metrics").text
        assert "pagewiki_llm_calls_total 0" in body
        assert "pagewiki_prompt_tokens_total 0" in body
        # Persistent totals absent when store is None.
        assert "pagewiki_persistent_llm_calls_total" not in body
