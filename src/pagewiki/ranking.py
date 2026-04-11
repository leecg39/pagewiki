"""v0.8 lightweight BM25-style candidate pre-ranking.

Rationale
---------

The retrieval loop presents 5-20 candidate nodes per ToC review.
Currently the LLM picks one without any hint about which is likely
most relevant. Pre-ranking candidates by term overlap with the
query lets us:

  1. Present the top-K first (truncating the list that the LLM sees,
     reducing prompt tokens).
  2. Break ties deterministically when multiple candidates have
     similar summaries.

This is a zero-LLM-cost heuristic that's intentionally simple —
no stemming, no stop-words, no tf-idf corpus. Just Jaccard-ish
character-n-gram overlap plus a small BM25-style term-frequency
boost. Good enough to surface obvious matches first while staying
language-agnostic (Korean/English mixed vaults work fine).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Split on whitespace and common punctuation. We avoid full tokenization
# libraries to keep the module dependency-free.
_SPLIT_RE = re.compile(r"[\s,\.\;\:\!\?\-\(\)\[\]\{\}\"'`/\\]+", re.UNICODE)


def tokenize(text: str, *, min_len: int = 2) -> list[str]:
    """Cheap, language-agnostic tokenizer.

    Splits on whitespace and punctuation, lowercases, drops tokens
    shorter than ``min_len``. Works reasonably for Korean (splits on
    spaces/punct only, preserving hangul syllable blocks) and English.
    """
    parts = _SPLIT_RE.split(text.lower())
    return [p for p in parts if len(p) >= min_len]


def score_candidate(
    query_tokens: Iterable[str],
    candidate_text: str,
    *,
    k1: float = 1.2,
) -> float:
    """Return a BM25-style relevance score for one candidate.

    ``query_tokens`` should come from ``tokenize(query)``. The
    candidate is re-tokenized inside to keep the API symmetric
    (the caller can build a fresh candidate list without pre-
    tokenizing).

    We skip the IDF component — it would require a full corpus
    pass, and for 20 candidates it barely helps. The ``k1`` param
    is the standard BM25 saturation constant (1.2 is the canonical
    default).
    """
    cand_tokens = tokenize(candidate_text)
    if not cand_tokens:
        return 0.0

    cand_len = len(cand_tokens)
    cand_counts: dict[str, int] = {}
    for tok in cand_tokens:
        cand_counts[tok] = cand_counts.get(tok, 0) + 1

    score = 0.0
    for qt in query_tokens:
        tf = cand_counts.get(qt, 0)
        if tf == 0:
            continue
        # BM25 term weight without IDF and without length normalization
        # (we're scoring tiny summaries, not full documents).
        score += (tf * (k1 + 1)) / (tf + k1)

    # Small bonus for short, highly-relevant candidates so a title
    # that exactly matches the query beats a huge folder summary
    # that happens to contain the same word.
    if score > 0:
        score += max(0.0, 1.0 - cand_len / 50.0)

    return score


def rank_candidates(
    query: str,
    candidates: list[tuple[str, str]],
) -> list[tuple[int, float]]:
    """Return ``(index, score)`` pairs sorted by descending score.

    ``candidates`` is a list of ``(display_text, searchable_text)``
    pairs. Most callers pass ``(title, title + " " + summary)``.
    The returned list preserves the original index so callers can
    reorder their own parallel data structures without a second
    lookup.

    Candidates with zero overlap keep their original relative order
    (stable sort) so the LLM still sees them if the top-K cutoff
    doesn't kick in.
    """
    q_tokens = tokenize(query)
    scored = [
        (i, score_candidate(q_tokens, searchable))
        for i, (_, searchable) in enumerate(candidates)
    ]
    # Stable sort by score descending.
    scored.sort(key=lambda x: -x[1])
    return scored
