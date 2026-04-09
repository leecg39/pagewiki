#!/usr/bin/env python3
"""End-to-end benchmark of pagewiki against a real Obsidian vault.

Like ``ollama_smoke.py``, this is meant to be run **on the user's own
machine**. It does not live in the pytest suite because it requires:

  * an actual Obsidian vault with LONG notes,
  * a running Ollama + Gemma 4 (or any LiteLLM-supported) backend, and
  * several minutes of wall-clock per query.

What it does
------------
1. Scans the target folder via ``vault.scan_folder`` and reports:
   - MICRO / ATOMIC / LONG counts
   - total estimated tokens
   - largest note by token count
   - scan wall-clock

2. Runs the Layer 2 sub-tree build for every LONG note (cached under
   ``{vault}/.pagewiki-cache/``), reporting:
   - fresh-build vs cache-hit counts
   - total sections produced
   - largest sub-tree by section count
   - build wall-clock

3. Runs a list of queries (from ``--queries-file`` or the default set)
   through ``retrieval.run_retrieval`` and for each one records:
   - iterations used
   - number of cited nodes
   - answer length (chars)
   - wall-clock

4. Prints a summary table and, if ``--json-out`` is set, also writes
   a machine-readable ``benchmark-<timestamp>.json`` file so the user
   can track regression across versions.

Usage
-----
::

    python scripts/benchmark_vault.py \\
        --vault ~/Documents/Obsidian \\
        --folder Research

    python scripts/benchmark_vault.py \\
        --vault ~/Documents/Obsidian \\
        --folder Research \\
        --queries-file bench-queries.txt \\
        --model ollama/gemma4:e4b \\
        --json-out

Exit code
---------
``0`` on clean completion. ``1`` if any query produced an empty answer
or an LLM call errored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pagewiki.cache import TreeCache  # noqa: E402
from pagewiki.retrieval import run_retrieval  # noqa: E402
from pagewiki.tree import NoteTier  # noqa: E402
from pagewiki.vault import (  # noqa: E402
    build_long_subtrees,
    scan_folder,
    summarize_atomic_notes,
)

DEFAULT_QUERIES = [
    "이 볼트의 핵심 주제 3가지를 요약해줘",
    "가장 최근에 작성된 리서치 노트의 주요 결론은?",
    "방법론(methodology) 관련 논의가 있는 노트들을 비교해줘",
]


@dataclass
class ScanStats:
    total_notes: int
    micro: int
    atomic: int
    long: int
    total_tokens: int
    largest_note_title: str
    largest_note_tokens: int
    scan_seconds: float


@dataclass
class BuildStats:
    long_notes_processed: int
    built_fresh: int
    from_cache: int
    total_sections: int
    largest_subtree_note: str
    largest_subtree_sections: int
    build_seconds: float


@dataclass
class QueryResult:
    query: str
    elapsed_seconds: float
    iterations_used: int
    cited_count: int
    answer_chars: int
    cited_nodes: list[str] = field(default_factory=list)
    answer_preview: str = ""
    error: str | None = None


@dataclass
class BenchmarkReport:
    vault: str
    folder: str | None
    model: str
    timestamp: str
    scan: ScanStats
    build: BuildStats
    queries: list[QueryResult]


def _make_chat_fn(model: str, num_ctx: int):
    from pagewiki.ollama_client import chat

    def _call(prompt: str) -> str:
        return chat(prompt, model=model, num_ctx=num_ctx).text

    return _call


def _count_sections(node) -> int:
    return sum(1 for child in node.walk() if child.kind == "section")


def run_benchmark(
    vault: Path,
    folder: str | None,
    queries: list[str],
    model: str,
    num_ctx: int,
    skip_summaries: bool,
) -> BenchmarkReport:
    # ─── Step 1: scan ────────────────────────────────────────────────
    scan_t0 = time.time()
    root = scan_folder(vault, folder)
    scan_elapsed = time.time() - scan_t0

    counts = {NoteTier.MICRO: 0, NoteTier.ATOMIC: 0, NoteTier.LONG: 0}
    total_tokens = 0
    largest_title = ""
    largest_tokens = 0
    total_notes = 0

    for node in root.walk():
        if node.kind != "note" or node.tier is None:
            continue
        counts[node.tier] += 1
        total_notes += 1
        tokens = node.token_count or 0
        total_tokens += tokens
        if tokens > largest_tokens:
            largest_tokens = tokens
            largest_title = node.title

    scan_stats = ScanStats(
        total_notes=total_notes,
        micro=counts[NoteTier.MICRO],
        atomic=counts[NoteTier.ATOMIC],
        long=counts[NoteTier.LONG],
        total_tokens=total_tokens,
        largest_note_title=largest_title,
        largest_note_tokens=largest_tokens,
        scan_seconds=scan_elapsed,
    )

    chat_fn = _make_chat_fn(model, num_ctx)

    if not skip_summaries:
        summarize_atomic_notes(root, chat_fn)

    # ─── Step 2: build Layer 2 sub-trees ─────────────────────────────
    build_t0 = time.time()
    built_fresh, from_cache = build_long_subtrees(
        root,
        vault_root=vault,
        model_id=model,
        chat_fn=None if skip_summaries else chat_fn,
        cache=TreeCache(vault),
    )
    build_elapsed = time.time() - build_t0

    total_sections = 0
    largest_subtree_note = ""
    largest_subtree_sections = 0
    for node in root.walk():
        if node.kind == "note" and node.tier == NoteTier.LONG and node.children:
            sections = _count_sections(node)
            total_sections += sections
            if sections > largest_subtree_sections:
                largest_subtree_sections = sections
                largest_subtree_note = node.title

    build_stats = BuildStats(
        long_notes_processed=built_fresh + from_cache,
        built_fresh=built_fresh,
        from_cache=from_cache,
        total_sections=total_sections,
        largest_subtree_note=largest_subtree_note,
        largest_subtree_sections=largest_subtree_sections,
        build_seconds=build_elapsed,
    )

    # ─── Step 3: run each query ──────────────────────────────────────
    query_results: list[QueryResult] = []
    for query in queries:
        q_t0 = time.time()
        try:
            result = run_retrieval(query, root, chat_fn)
            q_elapsed = time.time() - q_t0
            query_results.append(
                QueryResult(
                    query=query,
                    elapsed_seconds=q_elapsed,
                    iterations_used=result.iterations_used,
                    cited_count=len(result.cited_nodes),
                    answer_chars=len(result.answer),
                    cited_nodes=result.cited_nodes,
                    answer_preview=result.answer[:200],
                )
            )
        except Exception as e:  # noqa: BLE001
            query_results.append(
                QueryResult(
                    query=query,
                    elapsed_seconds=time.time() - q_t0,
                    iterations_used=0,
                    cited_count=0,
                    answer_chars=0,
                    error=str(e),
                )
            )

    return BenchmarkReport(
        vault=str(vault),
        folder=folder,
        model=model,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        scan=scan_stats,
        build=build_stats,
        queries=query_results,
    )


def print_report(report: BenchmarkReport) -> None:
    print("=" * 70)
    print(f"pagewiki benchmark  {report.timestamp}")
    print(f"  vault:   {report.vault}")
    if report.folder:
        print(f"  folder:  {report.folder}")
    print(f"  model:   {report.model}")
    print()

    s = report.scan
    print(f"[Scan] {s.total_notes} notes in {s.scan_seconds:.2f}s")
    print(f"       MICRO={s.micro}  ATOMIC={s.atomic}  LONG={s.long}")
    print(f"       total tokens ≈ {s.total_tokens:,}")
    if s.largest_note_title:
        print(
            f"       largest: {s.largest_note_title!r}"
            f" ({s.largest_note_tokens:,} tokens)"
        )
    print()

    b = report.build
    print(
        f"[Build] {b.long_notes_processed} LONG notes in {b.build_seconds:.2f}s"
        f"  ({b.built_fresh} built, {b.from_cache} cached)"
    )
    print(f"        total sections: {b.total_sections}")
    if b.largest_subtree_note:
        print(
            f"        largest sub-tree: {b.largest_subtree_note!r}"
            f" ({b.largest_subtree_sections} sections)"
        )
    print()

    print("[Queries]")
    for i, q in enumerate(report.queries, start=1):
        print(f"  {i}. {q.query}")
        if q.error:
            print(f"     ERROR: {q.error}")
            continue
        print(
            f"     elapsed={q.elapsed_seconds:.1f}s  "
            f"iter={q.iterations_used}  cited={q.cited_count}  "
            f"answer={q.answer_chars} chars"
        )
        if q.cited_nodes:
            more = " …" if len(q.cited_nodes) > 3 else ""
            print(f"     cites: {', '.join(q.cited_nodes[:3])}{more}")
        print(f"     preview: {q.answer_preview!r}")
    print()

    ok_queries = sum(
        1 for q in report.queries if q.error is None and q.answer_chars > 0
    )
    print("─" * 70)
    print(f"Summary: {ok_queries}/{len(report.queries)} queries produced answers")
    total_time = (
        s.scan_seconds + b.build_seconds + sum(q.elapsed_seconds for q in report.queries)
    )
    print(f"Total wall-clock: {total_time:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vault", type=Path, required=True, help="Path to the Obsidian vault root."
    )
    parser.add_argument(
        "--folder", default=None, help="Sub-folder to benchmark (e.g. Research)."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PAGEWIKI_MODEL", "ollama/gemma4:26b"),
        help="LiteLLM model id. Default: ollama/gemma4:26b",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=131072,
        help="Ollama context window (default: 131072).",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Optional path to a text file with one query per line. "
        "If omitted, a built-in default set is used.",
    )
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip the atomic-note summarization and the per-section "
        "summarization passes (much faster, worse ToC quality).",
    )
    parser.add_argument(
        "--json-out",
        action="store_true",
        help="Also write benchmark-<timestamp>.json for regression tracking.",
    )
    args = parser.parse_args()

    if not args.vault.exists():
        print(f"FAIL: vault not found at {args.vault}", file=sys.stderr)
        return 1

    if args.queries_file:
        queries = [
            line.strip()
            for line in args.queries_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not queries:
            print(f"FAIL: no queries found in {args.queries_file}", file=sys.stderr)
            return 1
    else:
        queries = DEFAULT_QUERIES

    report = run_benchmark(
        vault=args.vault,
        folder=args.folder,
        queries=queries,
        model=args.model,
        num_ctx=args.num_ctx,
        skip_summaries=args.skip_summaries,
    )
    print_report(report)

    if args.json_out:
        out_name = (
            f"benchmark-{report.timestamp.replace(':', '').replace('-', '')}.json"
        )
        Path(out_name).write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nMachine-readable report: {out_name}")

    ok = all(q.error is None and q.answer_chars > 0 for q in report.queries)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
