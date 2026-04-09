"""pagewiki command-line interface.

Commands:
  pagewiki scan — walk a vault folder, classify notes, print tier counts
  pagewiki ask  — run a full multi-hop reasoning query against a vault folder
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .cache import TreeCache
from .logger import QueryRecord, write_log
from .retrieval import run_retrieval
from .tree import NoteTier
from .vault import build_long_subtrees, scan_folder, summarize_atomic_notes

console = Console()


def _make_chat_fn(model: str, num_ctx: int):
    """Build a chat_fn closure bound to a specific Ollama model.

    Imported lazily so `pagewiki scan` works even when LiteLLM / Ollama are
    not installed locally.
    """
    from .ollama_client import chat

    def _call(prompt: str) -> str:
        return chat(prompt, model=model, num_ctx=num_ctx).text

    return _call


@click.group()
@click.version_option(__version__, prog_name="pagewiki")
def main() -> None:
    """pagewiki — Vectorless reasoning-based RAG for Obsidian vaults."""


@main.command()
@click.option(
    "--vault",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--folder", default=None, help="Subfolder inside the vault (e.g. Research)."
)
@click.option(
    "--build-long",
    is_flag=True,
    help="Eagerly build PageIndex sub-trees for every LONG note (no LLM; uses cache).",
)
@click.option(
    "--show-graph",
    is_flag=True,
    help="Also print the [[wiki-link]] graph summary (totals, dangling, top nodes).",
)
@click.option(
    "--model",
    default="ollama/gemma4:26b",
    help="Model id used as part of the cache key for --build-long.",
)
def scan(
    vault: Path,
    folder: str | None,
    build_long: bool,
    show_graph: bool,
    model: str,
) -> None:
    """Scan a vault folder and report 3-tier classification counts."""
    console.print(f"[bold cyan]Scanning[/] {vault}{('/' + folder) if folder else ''}")
    root = scan_folder(vault, folder)

    counts = {NoteTier.MICRO: 0, NoteTier.ATOMIC: 0, NoteTier.LONG: 0}
    total_notes = 0
    total_tokens = 0
    for node in root.walk():
        if node.kind == "note" and node.tier is not None:
            counts[node.tier] += 1
            total_notes += 1
            total_tokens += node.token_count or 0

    table = Table(title="Vault Layer 1 Summary")
    table.add_column("Tier", style="bold")
    table.add_column("Threshold")
    table.add_column("Count", justify="right")

    table.add_row("MICRO", "< 500 tokens", str(counts[NoteTier.MICRO]))
    table.add_row("ATOMIC", "500~3000 tokens", str(counts[NoteTier.ATOMIC]))
    table.add_row("LONG", "> 3000 tokens", str(counts[NoteTier.LONG]))
    table.add_row("[bold]TOTAL", "", f"[bold]{total_notes}")

    console.print(table)
    console.print(f"Estimated total tokens: [yellow]{total_tokens:,}[/]")

    if build_long and counts[NoteTier.LONG] > 0:
        console.print(
            f"\n[dim]Building PageIndex sub-trees for {counts[NoteTier.LONG]} LONG notes...[/]"
        )
        built, from_cache = build_long_subtrees(
            root,
            vault_root=vault,
            model_id=model,
            chat_fn=None,  # scan command never calls the LLM
        )
        console.print(
            f"[dim]    → {built} built, {from_cache} from cache[/]"
        )

    if show_graph:
        from .wiki_links import build_link_index

        console.print()
        index = build_link_index(root)
        stats = index.stats()

        graph_table = Table(title="Wiki-Link Graph")
        graph_table.add_column("Metric", style="bold")
        graph_table.add_column("Value", justify="right")
        graph_table.add_row("Total resolved links", str(stats.total_links))
        graph_table.add_row("Dangling links", str(stats.dangling_count))
        graph_table.add_row("Ambiguous (>1 candidate)", str(stats.ambiguous_links))
        console.print(graph_table)

        if stats.top_linked_to:
            top_in = Table(title="Top linked-to notes")
            top_in.add_column("Note", style="bold")
            top_in.add_column("Incoming", justify="right")
            for title, count in stats.top_linked_to:
                top_in.add_row(title, str(count))
            console.print(top_in)

        if stats.top_outgoing:
            top_out = Table(title="Top outgoing notes")
            top_out.add_column("Note", style="bold")
            top_out.add_column("Outgoing", justify="right")
            for title, count in stats.top_outgoing:
                top_out.add_row(title, str(count))
            console.print(top_out)

        if stats.dangling_count > 0:
            console.print(
                f"\n[bold yellow]Dangling links ({stats.dangling_count}):[/]"
            )
            for source_id, raw_target in index.dangling()[:10]:
                console.print(f"  [yellow]{source_id}[/] → [[{raw_target}]]")
            if stats.dangling_count > 10:
                console.print(
                    f"  [dim]… and {stats.dangling_count - 10} more[/]"
                )


@main.command()
@click.argument("query")
@click.option(
    "--vault",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--folder", default="Research", help="Subfolder inside the vault. Default: Research"
)
@click.option(
    "--model", default="ollama/gemma4:26b", help="LiteLLM model id."
)
@click.option(
    "--num-ctx", default=131072, type=int, help="Ollama context window."
)
@click.option(
    "--skip-summaries",
    is_flag=True,
    help="Skip the atomic-note summarization pass (faster, worse ToC).",
)
def ask(
    query: str,
    vault: Path,
    folder: str,
    model: str,
    num_ctx: int,
    skip_summaries: bool,
) -> None:
    """Run a multi-hop reasoning query against a vault folder."""
    console.print(f"[bold cyan]Q:[/] {query}")
    console.print(f"  vault={vault}  folder={folder}  model={model}\n")

    start = time.time()
    chat_fn = _make_chat_fn(model, num_ctx)

    # Step 1: scan
    console.print("[dim]1/4 Scanning vault...[/]")
    root = scan_folder(vault, folder)
    note_count = sum(1 for n in root.walk() if n.kind == "note")
    long_count = sum(
        1 for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
    )

    if note_count == 0:
        console.print(f"[red]No notes found in {vault / folder}[/]")
        sys.exit(1)

    # Step 2: summarize atomic notes (optional)
    if not skip_summaries:
        console.print("[dim]2/4 Summarizing atomic notes...[/]")
        summarized = summarize_atomic_notes(root, chat_fn)
        console.print(f"[dim]    → {summarized} notes summarized[/]")
    else:
        console.print("[dim]2/4 Skipping summarization (--skip-summaries)[/]")

    # Step 3: build Layer 2 sub-trees for LONG notes (cached)
    if long_count > 0:
        console.print(
            f"[dim]3/4 Building PageIndex sub-trees for {long_count} LONG notes...[/]"
        )
        section_chat_fn = None if skip_summaries else chat_fn
        built, from_cache = build_long_subtrees(
            root,
            vault_root=vault,
            model_id=model,
            chat_fn=section_chat_fn,
            cache=TreeCache(vault),
        )
        console.print(f"[dim]    → {built} built, {from_cache} from cache[/]")
    else:
        console.print("[dim]3/4 No LONG notes — skipping sub-tree build[/]")

    # Step 4: retrieval loop
    console.print("[dim]4/4 Running multi-hop retrieval loop...[/]\n")
    result = run_retrieval(query, root, chat_fn)

    elapsed = time.time() - start

    console.print(f"[bold green]A:[/] {result.answer}\n")

    if result.cited_nodes:
        console.print("[bold]Cited nodes:[/]")
        for cited in result.cited_nodes:
            console.print(f"  • {cited}")

    console.print(
        f"\n[dim]iterations={result.iterations_used}  elapsed={elapsed:.1f}s[/]"
    )

    record = QueryRecord(
        query=query,
        answer=result.answer,
        cited_nodes=result.cited_nodes,
        model=model,
        elapsed_seconds=elapsed,
    )
    log_path = write_log(vault / folder, record)
    console.print(f"[dim]Logged to {log_path}[/]")


if __name__ == "__main__":
    sys.exit(main())
