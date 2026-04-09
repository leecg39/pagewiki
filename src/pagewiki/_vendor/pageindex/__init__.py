"""Minimal vendored subset of VectifyAI/PageIndex (MIT).

Upstream:   https://github.com/VectifyAI/PageIndex
Commit:     f2dcffc0b79a8ccaddcaa9f51e6a54e3b7e7020b
Vendored:   2026-04-09
License:    MIT (see ./LICENSE)

## Scope of vendoring

Only the markdown tree-building pure-Python functions from
``pageindex/page_index_md.py`` are vendored. Neither PDF processing
(``pymupdf``, ``PyPDF2``) nor LLM calls (``litellm``) are pulled in —
pagewiki supplies its own summarization via the injected ``chat_fn``.

## Modifications

1. ``page_index_md.py`` — LLM-dependent helpers (``get_node_summary``,
   ``generate_summaries_for_structure_md``, ``md_to_tree``) and the
   ``__main__`` block have been removed. The pure tree-building
   functions are kept verbatim.
2. ``utils.py`` — replaced with a tiny shim providing only
   ``count_tokens`` (char/3 heuristic, matching pagewiki/vault.py),
   so the vendored module can be imported without LiteLLM or tiktoken.

The public surface exported by this package is intentionally narrow.
"""

from .page_index_md import (
    build_tree_from_nodes,
    clean_tree_for_output,
    extract_node_text_content,
    extract_nodes_from_markdown,
    tree_thinning_for_index,
    update_node_list_with_text_token_count,
)

__all__ = [
    "build_tree_from_nodes",
    "clean_tree_for_output",
    "extract_node_text_content",
    "extract_nodes_from_markdown",
    "tree_thinning_for_index",
    "update_node_list_with_text_token_count",
]
