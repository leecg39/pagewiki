"""Tests for v0.11: /chat/stream, SSE usage events, per-vault cache,
usage-report CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pagewiki.usage_store import UsageStore
from pagewiki.vault import (
    build_long_subtrees_multi,
    scan_multi_vault,
    summarize_atomic_notes_multi,
    vault_for_note,
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-vault cache routing
# ─────────────────────────────────────────────────────────────────────────────


class TestVaultForNote:
    def test_finds_owning_vault(self, tmp_path: Path) -> None:
        v1 = tmp_path / "vault1"
        (v1 / "Notes").mkdir(parents=True)
        note1 = v1 / "Notes" / "a.md"
        note1.write_text("x", encoding="utf-8")

        v2 = tmp_path / "vault2"
        (v2 / "Notes").mkdir(parents=True)
        note2 = v2 / "Notes" / "b.md"
        note2.write_text("y", encoding="utf-8")

        assert vault_for_note(note1, [v1, v2]) == v1
        assert vault_for_note(note2, [v1, v2]) == v2

    def test_returns_none_when_outside_any_vault(self, tmp_path: Path) -> None:
        v1 = tmp_path / "v1"
        v1.mkdir()
        outsider = tmp_path / "stray.md"
        outsider.write_text("x", encoding="utf-8")
        assert vault_for_note(outsider, [v1]) is None

    def test_prefers_longest_match(self, tmp_path: Path) -> None:
        """Nested vault paths should route to the most specific one."""
        parent = tmp_path / "parent"
        child = parent / "child"
        (child / "Notes").mkdir(parents=True)
        note = child / "Notes" / "x.md"
        note.write_text("x", encoding="utf-8")

        # Both are valid prefixes of the note path; child is more specific.
        assert vault_for_note(note, [parent, child]) == child


class TestSummarizeMulti:
    def test_routes_each_note_to_its_vault_cache(self, tmp_path: Path) -> None:
        v1 = tmp_path / "v1"
        (v1 / "Notes").mkdir(parents=True)
        (v1 / "Notes" / "a.md").write_text("a " * 1500, encoding="utf-8")

        v2 = tmp_path / "v2"
        (v2 / "Notes").mkdir(parents=True)
        (v2 / "Notes" / "b.md").write_text("b " * 1500, encoding="utf-8")

        root = scan_multi_vault([(v1, "Notes"), (v2, "Notes")])

        calls: list[str] = []

        def fake_chat(prompt: str) -> str:
            calls.append(prompt)
            return f"summary {len(calls)}"

        count = summarize_atomic_notes_multi(
            root,
            fake_chat,
            vault_roots=[v1, v2],
            model_id="m",
            max_workers=1,
        )
        assert count == 2

        # Each vault should have its own cache directory populated.
        assert (v1 / ".pagewiki-cache" / "summaries").is_dir()
        assert (v2 / ".pagewiki-cache" / "summaries").is_dir()

        # Each should contain exactly one cache file.
        v1_files = list((v1 / ".pagewiki-cache" / "summaries").glob("*.json"))
        v2_files = list((v2 / ".pagewiki-cache" / "summaries").glob("*.json"))
        assert len(v1_files) == 1
        assert len(v2_files) == 1

    def test_cache_hits_skip_llm_across_vaults(self, tmp_path: Path) -> None:
        v1 = tmp_path / "v1"
        (v1 / "Notes").mkdir(parents=True)
        (v1 / "Notes" / "a.md").write_text("a " * 1500, encoding="utf-8")

        v2 = tmp_path / "v2"
        (v2 / "Notes").mkdir(parents=True)
        (v2 / "Notes" / "b.md").write_text("b " * 1500, encoding="utf-8")

        def fake_chat(prompt: str) -> str:
            return "요약"

        # First pass: both notes hit the LLM.
        root1 = scan_multi_vault([(v1, "Notes"), (v2, "Notes")])
        count1 = summarize_atomic_notes_multi(
            root1, fake_chat, vault_roots=[v1, v2], model_id="m", max_workers=1,
        )
        assert count1 == 2

        # Second pass: both served from their respective caches.
        calls: list[str] = []

        def counted_chat(prompt: str) -> str:
            calls.append(prompt)
            return "요약"

        root2 = scan_multi_vault([(v1, "Notes"), (v2, "Notes")])
        count2 = summarize_atomic_notes_multi(
            root2, counted_chat, vault_roots=[v1, v2], model_id="m", max_workers=1,
        )
        assert count2 == 0
        assert len(calls) == 0


class TestBuildLongMulti:
    def test_long_subtree_cache_per_vault(self, tmp_path: Path) -> None:
        v1 = tmp_path / "v1"
        (v1 / "Notes").mkdir(parents=True)
        # LONG note threshold is >3000 tokens ≈ >9000 chars.
        long_content = "# Title\n\n" + ("## Section\n\nparagraph " * 1500)
        (v1 / "Notes" / "long.md").write_text(long_content, encoding="utf-8")

        v2 = tmp_path / "v2"
        (v2 / "Notes").mkdir(parents=True)
        (v2 / "Notes" / "long.md").write_text(long_content, encoding="utf-8")

        root = scan_multi_vault([(v1, "Notes"), (v2, "Notes")])

        built, from_cache = build_long_subtrees_multi(
            root,
            vault_roots=[v1, v2],
            model_id="m",
            chat_fn=None,
        )
        assert built == 2
        assert from_cache == 0

        # Each vault has its own trees/ directory.
        assert (v1 / ".pagewiki-cache" / "trees").is_dir()
        assert (v2 / ".pagewiki-cache" / "trees").is_dir()
        assert list((v1 / ".pagewiki-cache" / "trees").glob("*.json"))
        assert list((v2 / ".pagewiki-cache" / "trees").glob("*.json"))


# ─────────────────────────────────────────────────────────────────────────────
# /chat/stream and /ask/stream usage events
# ─────────────────────────────────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


@pytest.fixture
def stream_client(tmp_path: Path):
    vault = tmp_path / "vault"
    (vault / "Notes").mkdir(parents=True)
    (vault / "Notes" / "note.md").write_text("content", encoding="utf-8")

    responses_idx = {"i": 0}

    def fake_chat(prompt: str) -> str:
        script = [
            "요약",                       # summarize at startup
            "SELECT: Notes/note.md",     # first turn
            "SUFFICIENT: ok",
            "첫 답변",
            "SELECT: Notes/note.md",     # second turn (if any)
            "SUFFICIENT: ok",
            "두번째 답변",
        ]
        i = min(responses_idx["i"], len(script) - 1)
        responses_idx["i"] += 1
        return script[i]

    state = build_initial_state(
        [vault],
        folder="Notes",
        model="test",
        num_ctx=8192,
        max_workers=1,
        chat_fn=fake_chat,
    )
    # Reset the script counter after startup summarization consumed one response.
    app = create_app(state)
    return TestClient(app), state


class TestAskStreamUsage:
    def test_ask_stream_emits_usage_events(self, stream_client) -> None:
        tc, _ = stream_client
        with tc.stream("POST", "/ask/stream", json={"query": "test"}) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "event: trace" in body
        assert "event: usage" in body
        assert "event: answer" in body
        # The answer event should include a usage summary.
        assert "total_tokens" in body


class TestChatStream:
    def test_chat_stream_creates_session(self, stream_client) -> None:
        tc, state = stream_client
        with tc.stream("POST", "/chat/stream", json={"query": "q1"}) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "event: trace" in body
        assert "event: usage" in body
        assert "event: answer" in body
        # Session should now exist in state.
        assert len(state.sessions) == 1

    def test_chat_stream_accumulates_history(self, stream_client) -> None:
        tc, state = stream_client

        with tc.stream("POST", "/chat/stream", json={"query": "q1"}) as r1:
            body1 = "".join(r1.iter_text())

        # Parse session_id out of the answer event line.
        import json

        answer_lines = [
            line for line in body1.split("\n")
            if line.startswith("data: ") and "session_id" in line
        ]
        assert answer_lines
        sid_payload = json.loads(answer_lines[-1].removeprefix("data: "))
        sid = sid_payload["session_id"]

        # Reuse the session for a second turn.
        with tc.stream(
            "POST",
            "/chat/stream",
            json={"query": "q2", "session_id": sid},
        ) as r2:
            body2 = "".join(r2.iter_text())

        assert "event: answer" in body2
        assert state.sessions[sid].history
        assert len(state.sessions[sid].history) == 2

    def test_chat_stream_headers_are_sse(self, stream_client) -> None:
        tc, _ = stream_client
        with tc.stream("POST", "/chat/stream", json={"query": "q"}) as response:
            assert response.headers["content-type"].startswith("text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# usage-report CLI
# ─────────────────────────────────────────────────────────────────────────────


class TestUsageReportCLI:
    def test_usage_report_reads_existing_db(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        db_path = tmp_path / "usage.db"
        store = UsageStore(db_path)
        store.record("select", 500, 10, 1.0)
        store.record("evaluate", 800, 20, 1.5)
        store.record("select", 300, 5, 0.8)
        store.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(db_path)],
        )
        assert result.exit_code == 0
        assert "total_calls=3" in result.output
        assert "select" in result.output
        assert "evaluate" in result.output

    def test_usage_report_empty_db(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        db_path = tmp_path / "empty.db"
        UsageStore(db_path).close()  # create but empty

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(db_path)],
        )
        assert result.exit_code == 0
        assert "No events" in result.output

    def test_usage_report_phase_filter(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        db_path = tmp_path / "usage.db"
        store = UsageStore(db_path)
        store.record("select", 500, 10, 1.0)
        store.record("evaluate", 800, 20, 1.5)
        store.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(db_path), "--phase", "select"],
        )
        assert result.exit_code == 0
        assert "select" in result.output

    def test_usage_report_recent_flag(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        db_path = tmp_path / "usage.db"
        store = UsageStore(db_path)
        store.record("select", 500, 10, 1.0)
        store.record("evaluate", 800, 20, 1.5)
        store.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(db_path), "--recent", "5"],
        )
        assert result.exit_code == 0
        assert "Most recent" in result.output

    def test_usage_report_invalid_iso_date(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from pagewiki.cli import main

        db_path = tmp_path / "usage.db"
        UsageStore(db_path).close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["usage-report", "--db", str(db_path), "--since", "not-a-date"],
        )
        assert result.exit_code != 0
        assert "Invalid ISO" in result.output
