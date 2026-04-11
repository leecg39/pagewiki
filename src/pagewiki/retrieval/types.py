"""Shared types used across the retrieval subpackage (v0.13 refactor).

Separated from the main loop so both ``core.py`` and the higher-level
orchestrators (``decompose.py``, ``cross_vault.py``) can import these
without pulling in the full reasoning pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

ChatFn = Callable[[str], str]

# v0.14 prompt-caching contract: ``(system, user) -> str``. Passed
# alongside the regular ChatFn when the caller wants Ollama KV-cache
# reuse for stable system prefixes.
SystemChatFn = Callable[[str, str], str]

# Safety caps — prevent runaway LLM costs on pathological trees
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_GATHERED_NOTES = 4
DEFAULT_MAX_NOTE_CHARS = 20_000


@dataclass
class TraceStep:
    """Single step in the reasoning trace — useful for logging and debugging.

    ``phase`` is a free-form string; known values across the codebase:
    ``select``, ``evaluate``, ``finalize``, ``cross-ref``, ``decompose``,
    ``budget``, ``reuse``, ``cancel``, ``cross-vault``.
    """

    phase: str
    node_id: str | None
    detail: str


@dataclass
class RetrievalResult:
    """Outcome of a full retrieval loop."""

    answer: str
    cited_nodes: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    iterations_used: int = 0


EventCallback = Callable[[TraceStep], None] | None
