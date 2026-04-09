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


def _print_notesmd_open_hints(cited_node_ids: list[str], root) -> None:
    """Print copy-paste-ready ``notesmd-cli open`` commands for each citation.

    Only emitted when ``notesmd-cli`` is actually on PATH — no point
    showing the hint block otherwise. Resolves each pagewiki
    ``<rel_path>#<zfill_id>`` citation back to its human-readable
    ``(note_title, section_title)`` pair by walking the tree, then
    formats an argv via ``obsidian_config.build_open_command`` and
    shell-quotes it for the terminal.
    """
    import shlex

    from .obsidian_config import _notesmd_cli_on_path, build_open_command

    if not _notesmd_cli_on_path():
        return

    # Build a node_id → TreeNode lookup so we can translate the
    # citation strings back into titles.
    by_id = {n.node_id: n for n in root.walk()}

    console.print("\n[dim]Open in editor via notesmd-cli:[/]")
    for cited in cited_node_ids:
        node = by_id.get(cited)
        if node is None:
            continue
        # For section citations, the note title is the parent note's
        # title and the anchor is the section title. For note-level
        # citations, there's no anchor.
        if node.kind == "section":
            # Find the enclosing note by walking back up the rel_path:
            # "Research/paper.md#0003" → note_id "Research/paper.md".
            if "#" in cited:
                note_id = cited.split("#", 1)[0]
                note = by_id.get(note_id)
            else:
                note = None
            note_title = note.title if note else Path(cited).stem
            section_title = node.title
            argv = build_open_command(note_title, section_anchor=section_title)
        else:
            argv = build_open_command(node.title)
        console.print(f"  {shlex.join(argv)}")


def _resolve_vault(vault: Path | None) -> Path:
    """Return an explicit ``--vault`` argument or auto-discover one.

    Auto-discovery tries ``notesmd-cli print-default --path-only``
    first, then falls back to reading ``obsidian.json`` directly.
    On total miss, prints a helpful error listing both the Obsidian
    config path and the notesmd-cli install instructions, then
    exits with code 1.
    """
    if vault is not None:
        return vault

    from .obsidian_config import discover_default_vault

    discovered = discover_default_vault()
    if discovered is None:
        console.print(
            "[red]No vault specified and auto-discovery failed.[/]\n\n"
            "Fix by either:\n"
            "  1. [bold]Pass --vault explicitly[/]:\n"
            '       pagewiki scan --vault "~/Documents/Obsidian Vault" ...\n\n'
            "  2. [bold]Install notesmd-cli[/] (Yakitrak/notesmd-cli) and set a default vault:\n"
            "       brew install yakitrak/notesmd-cli/notesmd-cli   # macOS\n"
            "       notesmd-cli set-default <vault-name>\n\n"
            "  3. [bold]Open your vault in Obsidian at least once[/] so it writes\n"
            "     its path to the Obsidian config file:\n"
            "       macOS:   ~/Library/Application Support/obsidian/obsidian.json\n"
            "       Linux:   ~/.config/obsidian/obsidian.json"
        )
        sys.exit(1)

    console.print(f"[dim]Auto-discovered vault: {discovered}[/]")
    return discovered


@click.group()
@click.version_option(__version__, prog_name="pagewiki")
def main() -> None:
    """pagewiki — Vectorless reasoning-based RAG for Obsidian vaults."""


@main.command()
@click.option(
    "--vault",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Obsidian vault root. If omitted, auto-discovered via "
    "notesmd-cli or obsidian.json.",
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
    "--model",
    default="ollama/gemma4:26b",
    help="Model id used as part of the cache key for --build-long.",
)
def scan(
    vault: Path | None, folder: str | None, build_long: bool, model: str
) -> None:
    """Scan a vault folder and report 3-tier classification counts."""
    vault = _resolve_vault(vault)
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


@main.command()
def vaults() -> None:
    """List every Obsidian vault discoverable via notesmd-cli or obsidian.json.

    Uses the same discovery pipeline as ``scan`` and ``ask`` — but
    just prints the results instead of picking one.
    """
    from .obsidian_config import list_known_vaults

    known = list_known_vaults()
    if not known:
        console.print(
            "[yellow]No vaults discovered.[/]\n\n"
            "Either install [bold]notesmd-cli[/] (Yakitrak/notesmd-cli),\n"
            "or open a vault in Obsidian at least once so its path is\n"
            "written to the Obsidian config."
        )
        sys.exit(1)

    table = Table(title="Obsidian Vaults")
    table.add_column("Name", style="bold")
    table.add_column("Path")
    table.add_column("Default", justify="center")
    table.add_column("Exists", justify="center")
    for vault in known:
        table.add_row(
            vault.name,
            str(vault.path),
            "✓" if vault.is_default else "",
            "✓" if vault.path.exists() else "[red]missing[/]",
        )
    console.print(table)


@main.command()
@click.argument("query")
@click.option(
    "--vault",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Obsidian vault root. If omitted, auto-discovered via "
    "notesmd-cli or obsidian.json.",
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
    vault: Path | None,
    folder: str,
    model: str,
    num_ctx: int,
    skip_summaries: bool,
) -> None:
    """Run a multi-hop reasoning query against a vault folder."""
    vault = _resolve_vault(vault)
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

        _print_notesmd_open_hints(result.cited_nodes, root)

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
