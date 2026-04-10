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
from rich.markup import escape
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


def _format_dangling_line(source_id: str, raw_target: str) -> str:
    """Format one dangling-link line for Rich console rendering.

    Both ``source_id`` and ``raw_target`` come from user vault data
    (file paths and wiki-link targets), so they may contain characters
    that Rich parses as markup — most notably square brackets, which
    turn ``[[raw_target]]`` into a vanishing style tag. We escape both
    fields before splicing them into the final markup string so the
    original text is preserved verbatim.
    """
    safe_source = escape(source_id)
    safe_target = escape(raw_target)
    return f"  [yellow]{safe_source}[/] → \\[\\[{safe_target}]]"


def _resolve_vault_name_for_hints(vault: Path) -> str | None:
    """Map a resolved vault ``Path`` to its notesmd-cli vault *name*.

    Walks ``list_known_vaults()`` and returns the name of the first
    entry whose path resolves to the same absolute path as ``vault``.
    Falls back to the vault directory's basename when nothing matches
    (e.g. notesmd-cli is not installed AND the vault is not in
    obsidian.json yet). Returns ``None`` only if ``vault.name`` is
    itself empty, which should not happen in practice.

    Used by ``_print_notesmd_open_hints`` to address the
    chatgpt-codex-connector P2 review on PR #3: hints without
    ``--vault`` can open a same-titled note in the wrong vault when
    the user has multiple vaults registered.
    """
    from .obsidian_config import list_known_vaults

    try:
        vault_resolved = vault.expanduser().resolve()
    except OSError:
        vault_resolved = vault

    for known in list_known_vaults():
        try:
            if known.path.expanduser().resolve() == vault_resolved:
                return known.name
        except OSError:
            continue
    return vault.name or None


def _print_notesmd_open_hints(
    cited_node_ids: list[str], root, *, vault: Path
) -> None:
    """Print copy-paste-ready ``notesmd-cli open`` commands for each citation.

    Only emitted when ``notesmd-cli`` is actually on PATH — no point
    showing the hint block otherwise. Resolves each pagewiki
    ``<rel_path>#<zfill_id>`` citation back to its human-readable
    ``(note_title, section_title)`` pair by walking the tree, then
    formats an argv via ``obsidian_config.build_open_command`` and
    shell-quotes it for the terminal.

    Passes ``vault_name=`` to ``build_open_command`` so the generated
    commands include ``--vault <name>`` (P2 fix on PR #3 Codex
    review) — without it, a same-titled note in a different vault
    could be opened instead of the one pagewiki actually read.
    """
    import shlex

    from .obsidian_config import _notesmd_cli_on_path, build_open_command

    if not _notesmd_cli_on_path():
        return

    vault_name = _resolve_vault_name_for_hints(vault)

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
            argv = build_open_command(
                note_title,
                section_anchor=node.title,
                vault_name=vault_name,
            )
        else:
            argv = build_open_command(node.title, vault_name=vault_name)
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
    vault: Path | None,
    folder: str | None,
    build_long: bool,
    show_graph: bool,
    model: str,
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
                console.print(_format_dangling_line(source_id, raw_target))
            if stats.dangling_count > 10:
                console.print(
                    f"  [dim]… and {stats.dangling_count - 10} more[/]"
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
    for v in known:
        table.add_row(
            v.name,
            str(v.path),
            "✓" if v.is_default else "",
            "✓" if v.path.exists() else "[red]missing[/]",
        )
    console.print(table)


@main.command()
@click.option(
    "--vault",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Obsidian vault root. If omitted, auto-discovered.",
)
@click.option(
    "--folder", default=None, help="Subfolder inside the vault (e.g. Research)."
)
@click.option(
    "--model", default="ollama/gemma4:26b", help="LiteLLM model id."
)
@click.option(
    "--num-ctx", default=131072, type=int, help="Ollama context window."
)
def compile(
    vault: Path | None,
    folder: str | None,
    model: str,
    num_ctx: int,
) -> None:
    """Compile vault notes into an LLM-Wiki (entity pages + index).

    Extracts entities from every note, generates cross-referenced wiki
    pages, and writes them to ``{vault}/LLM-Wiki/``. Follows Karpathy's
    LLM-Wiki pattern: raw sources → entity extraction → wiki compilation.
    """
    from .compile import compile_wiki

    vault = _resolve_vault(vault)
    console.print(
        f"[bold cyan]Compiling LLM-Wiki[/] from "
        f"{vault}{('/' + folder) if folder else ''}"
    )

    root = scan_folder(vault, folder)
    note_count = sum(1 for n in root.walk() if n.kind == "note")
    if note_count == 0:
        console.print("[red]No notes found to compile.[/]")
        sys.exit(1)

    console.print(f"[dim]Found {note_count} notes to process...[/]")

    chat_fn = _make_chat_fn(model, num_ctx)
    wiki_dir = compile_wiki(root, vault, chat_fn, subfolder=folder)

    # Count generated files
    generated = list(wiki_dir.glob("*.md"))
    console.print(
        f"\n[bold green]LLM-Wiki compiled![/] "
        f"{len(generated)} pages written to {wiki_dir}"
    )
    console.print(f"[dim]Open in Obsidian: index.md is the entry point.[/]")


@main.command()
@click.option(
    "--vault",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Obsidian vault root. If omitted, auto-discovered.",
)
@click.option(
    "--folder", default=None, help="Subfolder inside the vault (e.g. Research)."
)
@click.option(
    "--interval",
    default=10,
    type=int,
    help="Poll interval in seconds. Default: 10.",
)
def watch(vault: Path | None, folder: str | None, interval: int) -> None:
    """Watch the vault for file changes and report them in real time.

    Polls the vault directory at the given interval and prints a summary
    whenever notes are added, modified, or deleted. Useful for keeping
    the PageIndex cache warm while editing in Obsidian.
    """
    from .watcher import ChangeSet, detect_changes, save_state

    vault = _resolve_vault(vault)
    scope = f"{vault}{('/' + folder) if folder else ''}"
    console.print(f"[bold cyan]Watching[/] {scope} (poll every {interval}s)")
    console.print("[dim]Press Ctrl+C to stop.[/]\n")

    # Initial snapshot
    save_state(vault, folder)
    console.print(f"[dim]Initial snapshot saved.[/]")

    try:
        while True:
            import time as _time

            _time.sleep(interval)
            changes = detect_changes(vault, folder)
            if changes.has_changes:
                save_state(vault, folder)
                ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
                console.print(f"\n[bold yellow][{ts}] Changes detected:[/]")
                for path in changes.added:
                    console.print(f"  [green]+[/] {path}")
                for path in changes.modified:
                    console.print(f"  [yellow]~[/] {path}")
                for path in changes.deleted:
                    console.print(f"  [red]-[/] {path}")
                console.print(
                    f"[dim]  Total: {changes.total} change(s)[/]"
                )
    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/]")


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

    # Step 3.5: build wiki-link index for cross-reference traversal (v0.2)
    from .wiki_links import build_link_index

    link_index = build_link_index(root)

    # Step 4: retrieval loop
    console.print("[dim]4/4 Running multi-hop retrieval loop...[/]\n")
    result = run_retrieval(query, root, chat_fn, link_index=link_index)

    elapsed = time.time() - start

    console.print(f"[bold green]A:[/] {result.answer}\n")

    if result.cited_nodes:
        console.print("[bold]Cited nodes:[/]")
        for cited in result.cited_nodes:
            console.print(f"  • {cited}")

        _print_notesmd_open_hints(result.cited_nodes, root, vault=vault)

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
