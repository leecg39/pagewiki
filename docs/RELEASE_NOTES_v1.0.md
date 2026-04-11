# pagewiki v1.0.0 Release Notes

**pagewiki** — Vectorless reasoning-based RAG for Obsidian vaults,
powered by local Gemma 4 via Ollama.

This is the first **production-stable** release. The project started
in v0.1 as a scaffolding experiment in January 2024 and has shipped
18 minor versions with 370+ tests, landing a complete feature set
for local-first, vectorless knowledge retrieval over Obsidian vaults.

## Why v1.0?

The public API (CLI flags, HTTP endpoints, Python module exports,
Obsidian plugin surface) is now stable. Anything breaking is
deferred to a hypothetical v2.0. Future 1.x releases will be
additive-only — new flags, new endpoints, new optimizations.

## What's in the box

### Core retrieval loop

- **Multi-hop reasoning** — LLM walks the Layer 1 vault tree +
  Layer 2 PageIndex section trees, selects the best candidate,
  evaluates sufficiency, and iterates until the answer is ready.
- **Vectorless** — no embeddings, no vector DB, no chunking.
  Just logical reasoning over tree-structured markdown.
- **Wiki-link traversal** — `[[cross-references]]` are resolved
  and added to the candidate pool during retrieval.
- **BM25 pre-ranking** — zero-LLM-cost candidate sorting before
  every SELECT prompt so the most likely match lands at the top.
- **JSON-mode prompts** — optional strict JSON schema for
  SELECT/EVALUATE with automatic fallback to the text parser.
- **Cited-note re-ranking** — final citations sorted by relevance
  rather than discovery order.

### Performance

- **Parallel LLM calls** — `--max-workers N` parallelizes
  summarization, compile entity extraction, and cross-vault
  retrieval.
- **Context reuse** — `--reuse-context` suppresses already-shown
  candidates on deep loops.
- **Token budget** — `--max-tokens N` caps total spend per query.
- **Budget split** — `--token-split 20:60:20` divides the budget
  across summarize/retrieve/synthesis phases.
- **Prompt caching** — `--prompt-cache` uses stable system
  prefixes so Ollama can reuse KV cache across calls.
- **Disk caches** — summary cache + PageIndex sub-tree cache keyed
  on `(abs_path, mtime_ns, model_id)` so repeat queries are near-instant.
- **Per-vault caches** — each vault's notes land in that vault's
  `.pagewiki-cache/` directory in multi-vault mode.

### Multi-vault

- `--extra-vault` (repeatable) adds more vaults to a single query.
- `--per-vault` runs an independent retrieval per vault and
  synthesizes the results with `synthesize_multi_answer_prompt`.
- `parallel_workers=N` runs per-vault loops concurrently.
- `--allow-partial` + `--retry-failed` keep going when individual
  vaults fail.

### Observability

- `UsageTracker` records every LLM call with real LiteLLM token
  counts, phase buckets, and a cacheable flag.
- `--usage` flag prints a Rich table breakdown + prompt-cache
  eligibility + inferred hit rate.
- `UsageStore` SQLite backend persists events across restarts
  with daily rollups, range queries, and rolling retention.
- `pagewiki usage-report` CLI — table/CSV/JSON output with
  `--since`, `--until`, `--phase`, `--recent`, `--daily`,
  `--prune-older-than` filters.

### Interaction surfaces

- **CLI**: 8 commands (`scan`, `ask`, `chat`, `compile`, `watch`,
  `vaults`, `serve`, `usage-report`) with 30+ flags.
- **HTTP API** (`pagewiki serve`, optional `[server]` extra):
  `GET /health`, `POST /scan`, `POST /ask`, `POST /ask/stream`,
  `POST /chat`, `POST /chat/stream`, `DELETE /chat/{sid}`,
  `GET /usage`, `POST /usage/reset`, `GET /usage/history`,
  `GET /usage/history/stream`, `WS /ask/ws`, `GET /` (Web UI).
- **Web UI**: Self-contained single-page HTML served at `/`.
  Streams `/ask/stream`, plots cumulative tokens on an inline
  sparkline, exposes a collapsible historical view backed by
  `/usage/history/stream`, supports Cmd/Ctrl+Enter + Esc.
- **Obsidian plugin**: Scan, Ask, Chat, Compile, Watch, Vaults,
  UsageHistory commands in the Command Palette. SSE and WebSocket
  server modes. Cancel button mid-query. Full settings tab
  exposing every flag from v0.6 through v0.17.

## Installation

```bash
# Core install
pip install pagewiki

# With HTTP server + Web UI
pip install 'pagewiki[server]'

# With dev tooling
pip install 'pagewiki[dev]'
```

Runtime requirements:
- Python 3.11+
- Ollama (`brew install ollama`) with Gemma 4 pulled locally
  (`ollama pull gemma4:26b` for 24GB+ VRAM, `gemma4:e4b` for 16GB)

## Quick start

```bash
# Scan a vault (auto-discovers via notesmd-cli or obsidian.json)
pagewiki scan --folder Research

# Ask a question
pagewiki ask "2024년 3분기 매출 관련 리서치 요약"

# Multi-turn chat with token budget
pagewiki chat --usage --max-tokens 50000

# Run as HTTP server with Web UI
pip install 'pagewiki[server]'
pagewiki serve --usage-db ~/.pagewiki/usage.db --prompt-cache
# Open http://localhost:8000/ in a browser
```

## Backwards compatibility promise

Starting with v1.0.0:

- **CLI flags**: existing flags will not be removed or have their
  semantics changed. New flags may be added.
- **HTTP endpoints**: existing paths, request shapes, and response
  shapes will not break. New fields may be added.
- **Python API**: public symbols (everything in `pagewiki/__init__.py`
  and `pagewiki/retrieval/__init__.py` `__all__`) will not be
  removed. Private symbols (leading underscore) remain internal.
- **Obsidian plugin settings**: existing settings keys will not be
  renamed. New keys may be added with safe defaults.

## What's next (v1.1+)

The project has reached feature completeness for its original
charter. Future work will likely focus on:

- Optional cloud LLM backends (while preserving the local-first
  default).
- A Rust-based hot path for the BM25 ranker and parallel
  summarizer if Python overhead becomes a bottleneck at very
  large vault sizes.
- Improved PageIndex integration as the upstream project evolves.

But v1.0 ships complete, covered, and stable.

## Acknowledgements

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — the
  original reasoning-based vectorless RAG engine and tree-building
  algorithm that pagewiki wraps.
- [Andrej Karpathy's LLM-Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
  — the filesystem-as-knowledge-base philosophy behind
  `pagewiki compile`.
- [Google Gemma 4](https://blog.google/) — the local reasoning
  engine that makes the whole thing possible on a single machine.
- [Yakitrak/notesmd-cli](https://github.com/Yakitrak/notesmd-cli)
  — vault auto-discovery + citation-open integration.

---

**SHA**: see `git log main` for the exact commit.
**License**: MIT.
