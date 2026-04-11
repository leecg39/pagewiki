# Changelog

All notable changes to pagewiki are tracked here. The project follows
[Semantic Versioning](https://semver.org/) and targets a stable
**v1.0.0** release as its first production milestone.

## [1.1.0] — current

First additive release on the 1.x line. Every change here is
backward-compatible; no shipped behavior was removed or changed.

### Added
- **`pagewiki stats`** command — zero-LLM vault statistics:
  3-tier distribution, wiki-link graph (resolved/dangling/ambiguous),
  top-N linked-to, top-N outgoing, frontmatter tag histogram,
  orphan detection, and folder fan-out.
- **`pagewiki batch`** command — runs a list of queries from a
  text file against a single warmed scan. Supports `# comments`,
  blank lines, `--output` markdown report, `--stop-on-error`.
- **`GET /metrics`** — Prometheus text-format endpoint exposing
  `pagewiki_llm_calls_total`, `pagewiki_prompt_tokens_total`,
  `pagewiki_completion_tokens_total`, `pagewiki_cacheable_calls_total`,
  `pagewiki_phase_calls_total{phase="..."}`,
  `pagewiki_note_count`, `pagewiki_active_sessions`,
  `pagewiki_cacheable_ratio`, `pagewiki_cache_inferred_hit_rate`,
  and (when `--usage-db` is set) `pagewiki_persistent_*_total`
  lifetime counters.
- **`--lang en`** — new CLI flag on `ask`/`chat` that switches
  retrieval prompts to English. New `pagewiki.prompts_en` module
  provides `select_node_prompt_en`, `evaluate_prompt_en`,
  `final_answer_prompt_en`, `atomic_summary_prompt_en`,
  `decompose_query_prompt_en`, `synthesize_multi_answer_prompt_en`,
  plus `EN_SELECT_NODE_SYSTEM`/`EN_EVALUATE_SYSTEM`/
  `EN_FINAL_ANSWER_SYSTEM` constants. Format markers (`SELECT:`,
  `DONE:`, `SUFFICIENT:`, etc.) are preserved so the existing
  parsers handle both languages.
- **`run_retrieval(lang="en")`** — language selection parameter
  on the public retrieval API. Defaults to `"ko"` to preserve
  v1.0 behavior.

### Test coverage

- 398 tests pass (17 new in `test_v1_1_features.py`), ruff clean.

## [1.0.0] — stable release

### Released
- **Development Status bumped** from `3 - Alpha` to
  `5 - Production/Stable` in `pyproject.toml`. The API is now
  frozen for breaking changes within the 1.x line — future
  additions will be purely additive, and anything that would
  break existing callers gets deferred to a hypothetical 2.0.
- **370+ tests** across 20 test files cover every shipped feature
  from v0.1 through v1.0: retrieval, caching, filtering,
  streaming, parallel execution, multi-vault, API server,
  WebSocket, usage tracking, SQLite persistence, prompt caching,
  and the full Obsidian plugin surface.
- **8 CLI commands** (`scan`, `ask`, `chat`, `compile`, `watch`,
  `vaults`, `serve`, `usage-report`) with a combined 30+ flags.
- **12 HTTP endpoints + 1 WebSocket + 1 Web UI** served by the
  optional `[server]` extra.
- **Obsidian plugin** with Scan/Ask/Chat/Compile/Watch/Vaults/
  UsageHistory commands, SSE and WebSocket server modes, Cancel
  button, and full v0.6-v0.17 flag coverage.

### Summary of the 0.6 → 1.0 arc

| Theme | Features |
|---|---|
| Interactive UX | `chat` mode, streaming retrieval trace, Web UI, Obsidian plugin |
| Speed | Parallel LLM calls, BM25 pre-ranking, context reuse, prompt caching |
| Quality | JSON-mode prompts, parse retry, cited-note re-ranking, multi-query decomposition |
| Scale | Multi-vault, cross-vault parallel, per-vault cache routing, retry_failed |
| Observability | `--usage`, `UsageTracker`, `UsageStore` SQLite, `/usage` + `/usage/history` + streaming, sparkline, cache hit rate, inferred latency savings |
| Control | `--max-tokens`, `--token-split`, `--reuse-context`, `--prompt-cache`, `--allow-partial`, `--retry-failed`, `--json-mode`, WebSocket cancel, frontmatter filters |
| Integration | `pagewiki serve` FastAPI HTTP+WS, SQLite usage, CSV/JSON export, `notesmd-cli` vault discovery |
| Quality-of-life | Error messages with examples, CHANGELOG, keyboard shortcuts, daily rollups |

## [0.18.0]

### Added
- **CHANGELOG.md** — this file. Going forward every release
  documents what changed, what moved, and what broke (if anything).
- **Better error messages** across the CLI: vault auto-discovery
  failures, `--usage-db` misconfiguration, missing optional extras
  (`pagewiki[server]`), and SQLite permission problems now print
  actionable remediation hints instead of bare exceptions.
- **Web UI polish** — refined color palette, keyboard shortcuts
  (Cmd/Ctrl+Enter to submit, Escape to cancel), and per-query token
  counter live in the status bar.
- **`pagewiki --version` alias** — `pagewiki -V` is now equivalent
  to `pagewiki --version` for quick inspection.

### Changed
- CLI help text consolidated with new `v0.N` version markers so
  each flag's introduction version is obvious at a glance.
- Ruff + mypy cleanup pass across `src/` — eliminated the last
  remaining `Any` leaks in public signatures.

## [0.17.0]

### Added
- `/usage` endpoint surfaces prompt-cache stats: `cacheable_calls`,
  `cacheable_ratio`, `cache_inferred_hit_rate`, and latency savings
  fields.
- `serve --retention-days N --retention-interval S` starts a
  background thread that prunes old events from the SQLite usage
  store while preserving daily rollups.
- `run_cross_vault_retrieval(retry_failed=N)` + CLI
  `ask --retry-failed N` — retries failed vaults up to N times
  after the initial `allow_partial` pass.
- WebSocket `/ask/ws` cancelled frame now includes `reason`,
  `retry_after_ms`, and `partial_usage` metadata.
- Obsidian plugin `UsageHistoryModal` — new "PageWiki: Show usage
  history (server mode)" command that streams from the server's
  `/usage/history/stream` endpoint.

## [0.16.0]

### Added
- Parallel `summarize_atomic_notes` now runs in budget waves when
  `max_tokens + tracker` are set — no more "budget forces sequential".
- `UsageTracker.cacheable_latency_savings()` — infers prompt-cache
  hit rate from first-call-vs-subsequent-mean latency.
- Web UI historical view: collapsible `<details>` card that
  subscribes to `/usage/history/stream` with Start/Stop controls.
- `ServerState.system_chat_fn` + `serve --prompt-cache` + per-request
  `prompt_cache: true` over WebSocket.
- `run_cross_vault_retrieval(allow_partial=True)` — keeps going
  when a vault fails and synthesizes from the survivors.

## [0.15.0]

### Added
- `run_cross_vault_retrieval(parallel_workers=N)` — per-vault loops
  now run on a `ThreadPoolExecutor` with order-preserved results.
- `UsageEvent.cacheable` + `UsageTracker.cacheable_ratio()` — track
  the fraction of LLM calls dispatched through the prompt-cache path.
- `GET /usage/history/stream` — SSE tail-f endpoint for live usage
  events with `initial`, `event`, `heartbeat`, `done` frames.
- Plugin WebSocket accepts `max_tokens`, `token_split`, `json_mode`,
  `reuse_context` in the `ask` frame.
- Web UI sparkline chart — inline SVG `<polyline>` plotting
  cumulative token usage per SSE event.

## [0.14.0]

### Added
- Usage DB rolling retention — `UsageStore.prune_events_before()`
  preserves daily rollups then VACUUMs. CLI `usage-report --prune-older-than`.
- `GET /usage/history` — historical usage queries with `since`,
  `until`, `phase`, `limit` filters.
- Embedded Web UI — `GET /` serves a self-contained single-page
  HTML that streams `/ask/stream`.
- `--token-split A:B:C` — divides `--max-tokens` across
  summarize/retrieve/synthesis phases.
- `--prompt-cache` — Ollama KV-cache reuse via split `(system, user)`
  prompts.

## [0.13.0]

### Added
- `chat` mode now exposes `--json-mode` and `--reuse-context` flags.
- Server SSE usage frames report real LiteLLM token counts.
- Obsidian plugin gains a WebSocket client + Cancel button.
- Cross-vault × decompose composition.
- `usage-report --format csv|json` machine-readable output.

### Changed
- `retrieval.py` split into the `retrieval/` subpackage
  (`types`, `helpers`, `core`, `decompose`, `cross_vault`) with a
  re-export facade so existing imports keep working.

## [0.12.0]

### Added
- WebSocket `/ask/ws` endpoint with cancellation support.
- Daily usage rollups (`usage-report --daily`).
- `run_cross_vault_retrieval` — per-vault loops + synthesis,
  exposed via `ask --per-vault`.
- Obsidian plugin server-mode: ChatModal consumes `/chat/stream`
  directly via Fetch + ReadableStream SSE.

## [0.11.0]

### Added
- `POST /chat/stream` SSE endpoint with session history.
- Live `usage` events in both `/ask/stream` and `/chat/stream`.
- Per-vault cache routing in multi-vault mode.
- `pagewiki usage-report` CLI command.
- Token tracking + `--usage`/`--usage-db` on the compile command.
- Obsidian plugin catches up on v0.8–v0.10 flag surface.

## [0.10.0]

### Added
- JSON-mode prompts (`--json-mode`) with automatic fallback to
  the text parser.
- SQLite-backed usage persistence (`serve --usage-db`).
- Server-Sent Events streaming (`POST /ask/stream`).
- Context reuse optimization (`--reuse-context`).

## [0.9.0]

### Added
- Token budget enforcement (`--max-tokens`).
- Chat session token usage display.
- `GET /usage` endpoint — cumulative + persistent stats.
- Cited-note BM25 re-ranking after retrieval.

## [0.8.0]

### Added
- Token usage tracking (`--usage`) with `UsageTracker` + wrappers.
- Automatic retry on SELECT parse failure with stricter reminder.
- BM25-style candidate pre-ranking before the SELECT prompt.
- 11 FastAPI endpoint tests.

## [0.7.0]

### Added
- Parallel LLM calls (`--max-workers`) for summarize + compile.
- Multi-query decomposition (`--decompose`).
- Multi-vault search (`--extra-vault`, repeatable).
- HTTP API server (`pagewiki serve`) with optional `[server]` extra.

## [0.6.0]

### Added
- `pagewiki chat` — interactive multi-turn conversation mode.
- Atomic summary disk cache.
- YAML frontmatter filtering (`--tag`, `--after`, `--before`).
- Real-time streaming of retrieval trace events.

## [0.5.0] and earlier

- **v0.5**: Obsidian plugin UI (TypeScript), Command Palette
  integration, results modal.
- **v0.4**: `pagewiki watch` mtime-based file watcher.
- **v0.3**: `pagewiki compile` LLM-Wiki compiler (Karpathy pattern).
- **v0.2**: `[[wiki-link]]` retrieval traversal.
- **v0.1.5**: notesmd-cli integration + `pagewiki vaults`.
- **v0.1.4**: wiki-link index + `scan --show-graph`.
- **v0.1.3**: h1 flatten + vault-relative section ids.
- **v0.1.2**: PageIndex SDK integration + disk cache.
- **v0.1.1**: Multi-hop reasoning loop implementation.
- **v0.1**: Scaffolding, Layer 1 scanner, 3-tier classifier, CLI.

[1.0.0]: https://github.com/leecg39/pagewiki
[0.18.0]: https://github.com/leecg39/pagewiki
[0.17.0]: https://github.com/leecg39/pagewiki
[0.16.0]: https://github.com/leecg39/pagewiki
[0.15.0]: https://github.com/leecg39/pagewiki
[0.14.0]: https://github.com/leecg39/pagewiki
[0.13.0]: https://github.com/leecg39/pagewiki
[0.12.0]: https://github.com/leecg39/pagewiki
[0.11.0]: https://github.com/leecg39/pagewiki
[0.10.0]: https://github.com/leecg39/pagewiki
[0.9.0]: https://github.com/leecg39/pagewiki
[0.8.0]: https://github.com/leecg39/pagewiki
[0.7.0]: https://github.com/leecg39/pagewiki
[0.6.0]: https://github.com/leecg39/pagewiki
[0.5.0]: https://github.com/leecg39/pagewiki
