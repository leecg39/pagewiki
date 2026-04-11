"""Multi-hop reasoning retrieval package.

Split into focused modules in v0.13 (was a single 846-line file):

* ``types``       — ``TraceStep``, ``RetrievalResult``, ``ChatFn``,
                    ``EventCallback``, default safety caps
* ``helpers``     — pure tree/summary helpers + ``_load_note_content``
* ``core``        — the main ``run_retrieval`` reasoning loop
* ``decompose``   — ``run_decomposed_retrieval`` (v0.7 multi-query)
* ``cross_vault`` — ``run_cross_vault_retrieval`` (v0.12 per-vault fan-out)

The public API is re-exported at the package root so existing
imports like ``from pagewiki.retrieval import run_retrieval`` keep
working unchanged.

Shape of a typical call::

    result = run_retrieval(
        query="What is X?",
        root=scanned_tree,
        chat_fn=ollama_chat_wrapper,
        on_event=lambda step: print(step),
        tracker=usage_tracker,
        max_tokens=50_000,
    )

Each retrieval phase emits a ``TraceStep`` via ``on_event`` so CLIs
and the FastAPI server can stream progress in real time.
"""

from __future__ import annotations

from .core import run_retrieval
from .cross_vault import run_cross_vault_retrieval
from .decompose import run_decomposed_retrieval
from .helpers import (
    _children_as_summaries,
    _load_note_content,
    _node_as_summary,
    _promote_to_note,
)
from .types import (
    DEFAULT_MAX_GATHERED_NOTES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_NOTE_CHARS,
    ChatFn,
    EventCallback,
    RetrievalResult,
    TraceStep,
)

__all__ = [
    # Types
    "ChatFn",
    "EventCallback",
    "RetrievalResult",
    "TraceStep",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_GATHERED_NOTES",
    "DEFAULT_MAX_NOTE_CHARS",
    # Public entry points
    "run_retrieval",
    "run_decomposed_retrieval",
    "run_cross_vault_retrieval",
    # Private helpers exported for tests that probe internals
    "_load_note_content",
    "_node_as_summary",
    "_children_as_summaries",
    "_promote_to_note",
]
