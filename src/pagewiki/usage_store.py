"""v0.10 SQLite-backed usage persistence.

``UsageTracker`` stores events in-memory only; they disappear on
restart. For long-running servers and for historical analysis
(daily/weekly token budgets, rolling averages, cost attribution),
we persist events to a SQLite file.

Schema
------

A single table ``usage_events`` with one row per LLM call::

    timestamp   REAL        unix epoch when the call completed
    phase       TEXT        phase bucket (select/evaluate/final/...)
    prompt      INTEGER     prompt tokens
    completion  INTEGER     completion tokens
    elapsed     REAL        seconds

Indexes on (``timestamp``) and (``phase``) keep range queries cheap
even at millions of rows. The file is opened in WAL mode so
concurrent reads don't block writes — important for a FastAPI
process that may serve ``/usage`` requests while a background
retrieval loop is still writing.

Design notes
------------

* ``UsageStore`` is a thin wrapper around a ``sqlite3.Connection``.
  All writes go through ``record()`` which also emits to an
  in-memory ``UsageTracker`` so consumers that want both the
  live delta and the persistent history get them for free.
* Every method is thread-safe: sqlite3 connections require
  check_same_thread=False for multi-threaded FastAPI use, so we
  serialize access with a single ``threading.Lock``.
* Schema migrations are intentionally absent — there's only one
  table. If the schema ever changes, bump ``SCHEMA_VERSION`` and
  drop/recreate.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    phase       TEXT    NOT NULL,
    prompt      INTEGER NOT NULL,
    completion  INTEGER NOT NULL,
    elapsed     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_events_timestamp
    ON usage_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_events_phase
    ON usage_events (phase);
"""


@dataclass(frozen=True)
class PersistedEvent:
    """One usage event loaded from the store."""

    timestamp: float
    phase: str
    prompt: int
    completion: int
    elapsed: float


@dataclass(frozen=True)
class UsageSummary:
    """Aggregated totals over a time range."""

    total_calls: int
    total_prompt: int
    total_completion: int
    total_elapsed: float
    by_phase: dict[str, dict[str, int | float]]


class UsageStore:
    """Thread-safe SQLite-backed usage persistence.

    Example::

        store = UsageStore(Path("~/pagewiki-usage.db").expanduser())
        store.record("select", 500, 10, 1.2)
        summary = store.query_summary(since=time.time() - 86400)
        print(f"last 24h: {summary.total_calls} calls, "
              f"{summary.total_prompt:,} prompt tokens")
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit so writes land immediately
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # ── Write path ─────────────────────────────────────────────────────────

    def record(
        self,
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_seconds: float,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Persist one LLM-call event."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_events "
                "(timestamp, phase, prompt, completion, elapsed) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, phase, prompt_tokens, completion_tokens, elapsed_seconds),
            )

    # ── Read path ──────────────────────────────────────────────────────────

    def query_events(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        phase: str | None = None,
        limit: int | None = None,
    ) -> list[PersistedEvent]:
        """Return matching events ordered by most-recent first."""
        sql = (
            "SELECT timestamp, phase, prompt, completion, elapsed "
            "FROM usage_events WHERE 1=1"
        )
        params: list[object] = []
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since)
        if until is not None:
            sql += " AND timestamp < ?"
            params.append(until)
        if phase is not None:
            sql += " AND phase = ?"
            params.append(phase)
        sql += " ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            PersistedEvent(
                timestamp=r[0],
                phase=r[1],
                prompt=r[2],
                completion=r[3],
                elapsed=r[4],
            )
            for r in rows
        ]

    def query_summary(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
    ) -> UsageSummary:
        """Return aggregated totals + per-phase breakdown.

        Uses SQL aggregation so the summary stays cheap even at
        millions of events.
        """
        where = "WHERE 1=1"
        params: list[object] = []
        if since is not None:
            where += " AND timestamp >= ?"
            params.append(since)
        if until is not None:
            where += " AND timestamp < ?"
            params.append(until)

        with self._lock:
            totals = self._conn.execute(
                f"SELECT COUNT(*), "
                f"       COALESCE(SUM(prompt), 0), "
                f"       COALESCE(SUM(completion), 0), "
                f"       COALESCE(SUM(elapsed), 0.0) "
                f"FROM usage_events {where}",
                params,
            ).fetchone()

            by_phase_rows = self._conn.execute(
                f"SELECT phase, "
                f"       COUNT(*), "
                f"       COALESCE(SUM(prompt), 0), "
                f"       COALESCE(SUM(completion), 0), "
                f"       COALESCE(SUM(elapsed), 0.0) "
                f"FROM usage_events {where} "
                f"GROUP BY phase",
                params,
            ).fetchall()

        by_phase: dict[str, dict[str, int | float]] = {}
        for row in by_phase_rows:
            by_phase[row[0]] = {
                "calls": row[1],
                "prompt": row[2],
                "completion": row[3],
                "elapsed": row[4],
            }

        return UsageSummary(
            total_calls=totals[0],
            total_prompt=totals[1],
            total_completion=totals[2],
            total_elapsed=totals[3],
            by_phase=by_phase,
        )

    def clear(self) -> int:
        """Drop every row. Returns the count deleted."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM usage_events")
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Expose the raw connection for batch writes (advanced)."""
        with self._lock:
            yield self._conn
