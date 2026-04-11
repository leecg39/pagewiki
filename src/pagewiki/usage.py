"""v0.8 token usage tracking.

The retrieval/chat/compile pipelines make many LLM calls per request,
and the existing ``char/3`` heuristic in vault.py is wildly inaccurate.
This module accumulates actual usage reported by LiteLLM so users
can see where their context-window budget goes.

Design
------

``UsageTracker`` is a thread-safe counter that a wrapped ``chat_fn``
feeds. CLI commands construct one per invocation, inject it via
``make_tracking_chat_fn``, and print a summary at the end.

``UsageEvent`` records one LLM call: phase label (so callers can
bucket by operation), token counts, and elapsed time.

Phases used across the codebase:

* ``summarize``   — per-atomic-note summary
* ``section``     — per-section summary in pageindex_adapter
* ``select``      — retrieval loop Select prompt
* ``evaluate``    — retrieval loop Evaluate prompt
* ``final``       — retrieval loop final answer synthesis
* ``decompose``   — multi-query decompose + synthesize
* ``extract``     — compile entity extraction
* ``generate``    — compile wiki-page generation
* ``rewrite``     — chat follow-up query rewrite
* ``other``       — fallback bucket
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

ChatFn = Callable[[str], str]


@dataclass(frozen=True)
class UsageEvent:
    """One observed LLM call."""

    phase: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    # v0.15: True when the call was dispatched through the
    # prompt-cacheable path (stable system prefix + user-only
    # prompt). Used to compute cacheable-call ratios in
    # ``UsageTracker.cacheable_ratio()``.
    cacheable: bool = False


@dataclass
class UsageTracker:
    """Thread-safe accumulator for LLM usage across a session."""

    events: list[UsageEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_seconds: float,
        *,
        cacheable: bool = False,
    ) -> None:
        with self._lock:
            self.events.append(
                UsageEvent(
                    phase=phase,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed_seconds,
                    cacheable=cacheable,
                )
            )

    @property
    def total_prompt_tokens(self) -> int:
        with self._lock:
            return sum(e.prompt_tokens for e in self.events)

    @property
    def total_completion_tokens(self) -> int:
        with self._lock:
            return sum(e.completion_tokens for e in self.events)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self.events)

    @property
    def total_elapsed(self) -> float:
        with self._lock:
            return sum(e.elapsed_seconds for e in self.events)

    @property
    def cacheable_calls(self) -> int:
        """v0.15: count of calls that went through the prompt-cache path."""
        with self._lock:
            return sum(1 for e in self.events if e.cacheable)

    def cacheable_ratio(self) -> float:
        """Return the cacheable-call ratio (0.0 to 1.0).

        A call is "cacheable" when it was dispatched with a stable
        system prefix + user-only prompt, which is the shape Ollama's
        KV cache can reuse. This is a proxy for actual cache hit rate
        — the kernel-level hit rate is only visible inside Ollama
        itself — but it tells you what fraction of your workload is
        eligible for that optimization.
        """
        with self._lock:
            total = len(self.events)
            if total == 0:
                return 0.0
            hit = sum(1 for e in self.events if e.cacheable)
            return hit / total

    def cacheable_latency_savings(self) -> dict[str, float]:
        """Measure inferred prompt-cache speedup (v0.16).

        Ollama doesn't expose actual KV-cache hit/miss via LiteLLM, so
        we estimate the benefit from the outside: for each phase that
        has at least 2 cacheable calls, compare the **first** cacheable
        call's latency (a cache-miss proxy — the cold path) against
        the **mean** of the rest (cache-hit proxies). A negative number
        means the rest were slower than the first; a positive number
        means caching is likely helping.

        Returns::

            {
                "first_call_seconds": <float>,
                "subsequent_mean_seconds": <float>,
                "savings_per_call_seconds": <float>,
                "inferred_hit_rate": <float 0.0-1.0>,
                "samples": <int>,
            }

        ``inferred_hit_rate`` is ``savings / first_call`` clamped to
        [0, 1] — a very rough proxy that's ``1.0`` when subsequent
        calls are effectively instant and ``0.0`` when they're
        equally slow (no cache benefit).
        """
        with self._lock:
            cacheable_events = [e for e in self.events if e.cacheable]
        if len(cacheable_events) < 2:
            return {
                "first_call_seconds": 0.0,
                "subsequent_mean_seconds": 0.0,
                "savings_per_call_seconds": 0.0,
                "inferred_hit_rate": 0.0,
                "samples": len(cacheable_events),
            }

        first = cacheable_events[0]
        rest = cacheable_events[1:]
        rest_mean = sum(e.elapsed_seconds for e in rest) / len(rest)
        savings = first.elapsed_seconds - rest_mean
        if first.elapsed_seconds > 0:
            hit_rate = max(0.0, min(1.0, savings / first.elapsed_seconds))
        else:
            hit_rate = 0.0
        return {
            "first_call_seconds": first.elapsed_seconds,
            "subsequent_mean_seconds": rest_mean,
            "savings_per_call_seconds": savings,
            "inferred_hit_rate": hit_rate,
            "samples": len(cacheable_events),
        }

    def by_phase(self) -> dict[str, dict[str, int | float]]:
        """Return a per-phase breakdown suitable for tabular display."""
        buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "prompt": 0, "completion": 0, "elapsed": 0.0}
        )
        with self._lock:
            for e in self.events:
                b = buckets[e.phase]
                b["calls"] += 1
                b["prompt"] += e.prompt_tokens
                b["completion"] += e.completion_tokens
                b["elapsed"] += e.elapsed_seconds
        return {k: dict(v) for k, v in buckets.items()}


def make_tracking_chat_fn(
    inner: Callable[[str], object],
    tracker: UsageTracker,
    phase: str = "other",
) -> ChatFn:
    """Wrap a LiteLLM-style chat function so every call is recorded.

    The wrapped function must return an object with ``text``,
    ``prompt_tokens``, ``completion_tokens`` attributes (i.e.
    ``ollama_client.LLMResponse``). For callers that already return
    a bare string (tests, pageindex_adapter hooks), use
    ``make_tracking_str_chat_fn`` instead.

    The returned callable is a ``str -> str`` function to preserve
    the existing ``ChatFn`` contract throughout the codebase.
    """

    def _call(prompt: str) -> str:
        t0 = time.time()
        response = inner(prompt)
        elapsed = time.time() - t0
        text = getattr(response, "text", str(response))
        pt = getattr(response, "prompt_tokens", 0) or 0
        ct = getattr(response, "completion_tokens", 0) or 0
        tracker.record(phase, pt, ct, elapsed)
        return text

    return _call


def make_tracking_str_chat_fn(
    inner: ChatFn,
    tracker: UsageTracker,
    phase: str = "other",
    *,
    estimate_tokens: bool = True,
) -> ChatFn:
    """Wrap a plain ``str -> str`` chat function for usage tracking.

    Since the inner function doesn't report real token counts, this
    wrapper optionally falls back to a ``char/3`` estimate (matching
    the existing ``vault.estimate_tokens`` heuristic). Useful for
    tests and for retrieval phases that operate on strings directly.
    """

    def _call(prompt: str) -> str:
        t0 = time.time()
        text = inner(prompt)
        elapsed = time.time() - t0
        if estimate_tokens:
            pt = max(1, len(prompt) // 3)
            ct = max(1, len(text) // 3)
        else:
            pt = 0
            ct = 0
        tracker.record(phase, pt, ct, elapsed)
        return text

    return _call
