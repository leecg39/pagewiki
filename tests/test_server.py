"""Tests for ``pagewiki.server`` HTTP API (v0.7).

Uses FastAPI's ``TestClient`` to exercise the endpoints without
spinning up a real uvicorn process. Tests are skipped cleanly if
FastAPI is not installed, matching the [server] optional extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pagewiki.server import build_initial_state, create_app  # noqa: E402


def _scripted_chat(responses: list[str]):
    """Build a scripted chat_fn that cycles through a fixed response list.

    Falls back to the last response once the script is exhausted so the
    retrieval loop can finish even if the test under-provisions responses.
    """
    iterator = iter(responses)
    last = {"r": responses[-1] if responses else ""}

    def _call(prompt: str) -> str:
        try:
            val = next(iterator)
            last["r"] = val
            return val
        except StopIteration:
            return last["r"]

    return _call


@pytest.fixture
def vault_with_notes(tmp_path: Path) -> Path:
    """A minimal vault with two atomic notes for server bootstrapping."""
    vault = tmp_path / "vault"
    notes = vault / "Notes"
    notes.mkdir(parents=True)
    (notes / "alpha.md").write_text(
        "# Alpha\n\nThis is about the Transformer architecture.",
        encoding="utf-8",
    )
    (notes / "beta.md").write_text(
        "# Beta\n\nThis is about attention mechanisms.",
        encoding="utf-8",
    )
    return vault


@pytest.fixture
def client(vault_with_notes: Path):
    """Build a TestClient backed by a real ServerState."""
    chat_fn = _scripted_chat([
        "알파 요약",           # summarize alpha
        "베타 요약",           # summarize beta
        "SELECT: Notes/alpha.md",  # ToC review
        "SUFFICIENT: ok",      # evaluate
        "답변입니다.",          # final answer
    ])
    state = build_initial_state(
        [vault_with_notes],
        folder="Notes",
        model="test-model",
        num_ctx=8192,
        max_workers=2,
        chat_fn=chat_fn,
    )
    # Swap in a fresh scripted chat_fn for endpoint calls so the
    # summarization responses don't leak into retrieval.
    state.chat_fn = _scripted_chat([
        "SELECT: Notes/alpha.md",
        "SUFFICIENT: ok",
        "Transformer는 어텐션 기반 모델입니다.",
    ])
    app = create_app(state)
    return TestClient(app), state


class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        tc, _ = client
        response = tc.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.11.0"
        assert data["note_count"] == 2
        assert data["active_sessions"] == 0


class TestScanEndpoint:
    def test_scan_refreshes_state(self, client) -> None:
        tc, state = client
        # Rebuild chat_fn for the rescan path (summarize both + retrieval).
        state.chat_fn = _scripted_chat([
            "알파 요약 새",
            "베타 요약 새",
        ])
        response = tc.post("/scan")
        assert response.status_code == 200
        data = response.json()
        assert data["note_count"] == 2
        assert data["long_count"] == 0


class TestAskEndpoint:
    def test_ask_returns_answer_and_trace(self, client) -> None:
        tc, state = client
        state.chat_fn = _scripted_chat([
            "SELECT: Notes/alpha.md",
            "SUFFICIENT: ok",
            "최종 답변",
        ])
        response = tc.post("/ask", json={"query": "transformer란?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "최종 답변"
        assert "Notes/alpha.md" in data["cited_nodes"]
        assert isinstance(data["trace"], list)
        assert data["iterations_used"] >= 1

    def test_ask_with_empty_query_rejected(self, client) -> None:
        tc, _ = client
        response = tc.post("/ask", json={"query": ""})
        assert response.status_code == 422  # pydantic validation

    def test_ask_with_tag_filter(self, client) -> None:
        tc, state = client
        state.chat_fn = _scripted_chat([
            "DONE: 해당 태그의 노트가 없음",
        ])
        response = tc.post(
            "/ask",
            json={"query": "test", "tags": ["nonexistent"]},
        )
        assert response.status_code == 200
        data = response.json()
        # With no matching notes, the answer indicates insufficient evidence.
        assert "근거 부족" in data["answer"] or data["cited_nodes"] == []

    def test_ask_with_decompose(self, client) -> None:
        tc, state = client
        state.chat_fn = _scripted_chat([
            "SINGLE",  # decompose falls through
            "SELECT: Notes/alpha.md",
            "SUFFICIENT: ok",
            "단일 답변",
        ])
        response = tc.post("/ask", json={"query": "test", "decompose": True})
        assert response.status_code == 200
        data = response.json()
        assert "단일 답변" in data["answer"]


class TestChatEndpoint:
    def test_chat_creates_session(self, client) -> None:
        tc, state = client
        state.chat_fn = _scripted_chat([
            "SELECT: Notes/alpha.md",
            "SUFFICIENT: ok",
            "첫 답변",
        ])
        response = tc.post("/chat", json={"query": "hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"]
        assert data["turn"] == 1
        assert data["answer"] == "첫 답변"
        # Session should now exist.
        assert data["session_id"] in state.sessions

    def test_chat_accumulates_turns(self, client) -> None:
        tc, state = client
        # Turn 1.
        state.chat_fn = _scripted_chat([
            "SELECT: Notes/alpha.md",
            "SUFFICIENT: ok",
            "답변 1",
        ])
        resp1 = tc.post("/chat", json={"query": "q1"})
        sid = resp1.json()["session_id"]

        # Turn 2 — reuse the session_id.
        state.chat_fn = _scripted_chat([
            "SELECT: Notes/beta.md",
            "SUFFICIENT: ok",
            "답변 2",
        ])
        resp2 = tc.post("/chat", json={"query": "q2", "session_id": sid})
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["session_id"] == sid
        assert data["turn"] == 2
        assert data["answer"] == "답변 2"

        # History must have 2 entries.
        assert len(state.sessions[sid].history) == 2

    def test_delete_chat_session(self, client) -> None:
        tc, state = client
        state.chat_fn = _scripted_chat([
            "SELECT: Notes/alpha.md",
            "SUFFICIENT: ok",
            "답변",
        ])
        create_resp = tc.post("/chat", json={"query": "hello"})
        sid = create_resp.json()["session_id"]
        assert sid in state.sessions

        delete_resp = tc.delete(f"/chat/{sid}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["status"] == "cleared"
        assert sid not in state.sessions

    def test_delete_nonexistent_session_404(self, client) -> None:
        tc, _ = client
        response = tc.delete("/chat/nonexistent-id")
        assert response.status_code == 404


class TestSessionPruning:
    def test_expired_sessions_dropped(self, client) -> None:
        """Sessions older than SESSION_TTL_SECONDS should be purged."""
        import time as _time

        from pagewiki.server import ChatSession, _prune_expired_sessions

        _, state = client

        # Inject a stale session by hand.
        stale = ChatSession(sid="stale", last_active=_time.time() - 99999)
        state.sessions["stale"] = stale

        # Inject a fresh session for contrast.
        fresh = ChatSession(sid="fresh")
        state.sessions["fresh"] = fresh

        _prune_expired_sessions(state)

        assert "stale" not in state.sessions
        assert "fresh" in state.sessions
