"""pagewiki command-line interface.

MVP v0.1 commands:
  pagewiki scan   — walk a vault folder, classify notes, print tier counts
  pagewiki ask    — run a multi-hop reasoning query (stub; implemented in v0.1.1)

`ask` currently returns a scaffolded response so the end-to-end plumbing
is verifiable without requiring Ollama/PageIndex to be running.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .logger import QueryRecord, write_log
from .tree import NoteTier
from .vault import scan_folder

console = Console()


@click.group()
@click.version_option(__version__, prog_name="pagewiki")
def main() -> None:
    """pagewiki — Vectorless reasoning-based RAG for Obsidian vaults."""


@main.command()
@click.option("--vault", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--folder", default=None, help="Subfolder inside the vault (e.g. Research).")
def scan(vault: Path, folder: str | None) -> None:
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


@main.command()
@click.argument("query")
@click.option("--vault", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--folder", default="Research", help="Subfolder inside the vault. Default: Research")
@click.option("--model", default="ollama/gemma4:26b", help="LiteLLM model id.")
def ask(query: str, vault: Path, folder: str, model: str) -> None:
    """Run a multi-hop reasoning query against a vault folder.

    NOTE: v0.1 scaffold. The actual retrieval loop lands in v0.1.1.
    """
    console.print(f"[bold cyan]Q:[/] {query}")
    console.print(f"  vault={vault}  folder={folder}  model={model}\n")

    start = time.time()
    root = scan_folder(vault, folder)
    note_count = sum(1 for n in root.walk() if n.kind == "note")

    # v0.1 stub answer — real reasoning loop arrives in v0.1.1
    stub_answer = (
        f"[v0.1 scaffold] Scanned {note_count} notes in '{folder}'. "
        "Multi-hop reasoning loop not yet implemented — see docs/ARCHITECTURE.md Phase 2."
    )
    elapsed = time.time() - start

    console.print(f"[bold green]A:[/] {stub_answer}")

    record = QueryRecord(
        query=query,
        answer=stub_answer,
        cited_nodes=[],
        model=model,
        elapsed_seconds=elapsed,
    )
    log_path = write_log(vault / folder, record)
    console.print(f"\n[dim]Logged to {log_path}[/]")


if __name__ == "__main__":
    sys.exit(main())
