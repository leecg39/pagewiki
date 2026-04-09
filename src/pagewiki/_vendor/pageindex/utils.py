"""Minimal utils shim for the vendored PageIndex markdown tree builder.

Replaces the upstream ``pageindex/utils.py`` which imports litellm,
PyPDF2, pymupdf and python-dotenv. The only symbol consumed by the
retained functions in ``page_index_md.py`` is ``count_tokens``; all
other upstream helpers are LLM- or PDF-specific and are not needed.

The heuristic here (``len(text) // 3``) matches
``pagewiki.vault._CHARS_PER_TOKEN`` so that tier classification and
tree-thinning decisions use a consistent token scale across layers.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 3


def count_tokens(text: str | None, model: str | None = None) -> int:
    """Approximate token count using a char/3 heuristic.

    The upstream implementation delegates to ``litellm.token_counter``
    which requires a real model client. For markdown tree building we
    only need a stable ordering between node sizes, which the heuristic
    preserves. ``model`` is accepted and ignored for call-site
    compatibility.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)
