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
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .cache import SummaryCache, TreeCache
from .logger import QueryRecord, write_log
from .retrieval import run_retrieval
from .tree import NoteTier
from .vault import (
    build_long_subtrees,
    build_long_subtrees_multi,
    filter_tree,
    scan_folder,
    scan_multi_vault,
    summarize_atomic_notes,
    summarize_atomic_notes_multi,
)

console = Console()


def _make_chat_fn(model: str, num_ctx: int, tracker=None, usage_store=None):
    """Build a chat_fn closure bound to a specific Ollama model.

    Imported lazily so `pagewiki scan` works even when LiteLLM / Ollama are
    not installed locally.

    When ``tracker`` is provided, every call is recorded with real
    prompt/completion token counts from LiteLLM (v0.8). When
    ``usage_store`` is additionally provided, events are also
    persisted to a SQLite file (v0.10).
    """
    from .ollama_client import chat

    if tracker is None and usage_store is None:
        def _call(prompt: str) -> str:
            return chat(prompt, model=model, num_ctx=num_ctx).text
        return _call

    import time as _time

    def _call(prompt: str) -> str:
        t0 = _time.time()
        response = chat(prompt, model=model, num_ctx=num_ctx)
        elapsed = _time.time() - t0
        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        if tracker is not None:
            tracker.record("other", prompt_tokens, completion_tokens, elapsed)
        if usage_store is not None:
            usage_store.record("other", prompt_tokens, completion_tokens, elapsed)
        return response.text

    return _call


def _make_system_chat_fn(model: str, num_ctx: int, tracker=None, usage_store=None):
    """v0.14 prompt-caching chat_fn that takes separate (system, user).

    Calls ``ollama_client.chat`` with the system prompt in the
    ``system`` parameter so the LiteLLM + Ollama layer can keep the
    KV cache for the stable prefix and only incrementally process
    the user message. Tracker/store integration mirrors
    ``_make_chat_fn``.

    v0.15: tracker records ``cacheable=True`` for these calls so
    ``UsageTracker.cacheable_ratio()`` can report the fraction of
    the workload eligible for Ollama's KV-cache reuse.
    """
    import time as _time

    from .ollama_client import chat

    def _call(system: str, user: str) -> str:
        t0 = _time.time()
        response = chat(user, model=model, num_ctx=num_ctx, system=system)
        elapsed = _time.time() - t0
        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        if tracker is not None:
            tracker.record(
                "other", prompt_tokens, completion_tokens, elapsed,
                cacheable=True,
            )
        if usage_store is not None:
            usage_store.record("other", prompt_tokens, completion_tokens, elapsed)
        return response.text

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


def _open_cited_notes(
    cited_node_ids: list[str], root, *, vault: Path
) -> None:
    """Actually spawn ``notesmd-cli open`` for each cited note.

    Unlike ``_print_notesmd_open_hints`` which only prints copy-paste
    commands, this function executes them via subprocess so the user's
    editor opens each cited section directly.
    """
    import shlex
    import subprocess

    from .obsidian_config import _notesmd_cli_on_path, build_open_command

    if not _notesmd_cli_on_path():
        console.print(
            "[yellow]notesmd-cli is not on PATH — cannot open notes. "
            "Install it or use hints without --open.[/]"
        )
        return

    vault_name = _resolve_vault_name_for_hints(vault)
    by_id = {n.node_id: n for n in root.walk()}

    console.print("\n[dim]Opening cited notes via notesmd-cli...[/]")
    for cited in cited_node_ids:
        node = by_id.get(cited)
        if node is None:
            continue
        if node.kind == "section":
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

        console.print(f"  [dim]{shlex.join(argv)}[/]")
        try:
            subprocess.run(argv, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            console.print(f"  [red]Failed: {e}[/]")


def _parse_token_split(
    spec: str | None, total: int | None,
) -> tuple[int | None, int | None, int | None]:
    """Parse a ``summarize:retrieve:synth`` ratio into absolute caps (v0.14).

    The three numbers are ratios (not percentages) and get
    normalized by their sum before being multiplied against
    ``total``. Returns ``(summarize_cap, retrieve_cap, synth_cap)``
    or ``(None, None, None)`` when spec is empty.

    ``total`` must be set when spec is set; otherwise raises
    ``click.UsageError`` (caller should have validated).

    The synth cap is currently informational — retrieval.run_retrieval
    uses a single ``max_tokens`` cap that covers both traversal and
    the final synthesis call. We pass ``retrieve_cap + synth_cap``
    to it so the synthesis step has breathing room after the
    traversal hits its own soft edge.
    """
    if not spec:
        return None, None, None
    if total is None:
        raise click.UsageError(
            "--token-split requires --max-tokens to be set",
        )
    parts = spec.split(":")
    if len(parts) != 3:
        raise click.UsageError(
            "--token-split must be three numbers: SUMMARIZE:RETRIEVE:SYNTH",
        )
    try:
        ratios = [float(p) for p in parts]
    except ValueError as e:
        raise click.UsageError(f"Invalid --token-split: {e}") from e
    if any(r < 0 for r in ratios):
        raise click.UsageError("--token-split values must be non-negative")
    s = sum(ratios)
    if s <= 0:
        raise click.UsageError("--token-split values must sum to > 0")
    summarize = max(1, int(total * ratios[0] / s))
    retrieve = max(1, int(total * ratios[1] / s))
    synth = max(1, int(total * ratios[2] / s))
    return summarize, retrieve, synth


def _resolve_multi_vault(
    vaults: tuple[Path, ...],
    fallback: Path | None,
) -> list[Path]:
    """Return a list of resolved vault paths.

    If ``vaults`` is empty, falls back to auto-discovery (same as
    ``_resolve_vault``). Otherwise returns the explicit list, keeping
    order and deduplicating by resolved absolute path.
    """
    if not vaults:
        return [_resolve_vault(fallback)]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for v in vaults:
        try:
            abs_path = v.expanduser().resolve()
        except OSError:
            abs_path = v
        if abs_path in seen:
            continue
        seen.add(abs_path)
        resolved.append(v)
    return resolved


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
@click.option(
    "--max-workers", default=4, type=int,
    help="Parallel LLM workers for entity extraction + page generation.",
)
@click.option(
    "--usage",
    is_flag=True,
    help="Print token usage breakdown at the end of compilation (v0.11).",
)
@click.option(
    "--usage-db",
    "usage_db",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Persist usage events to a SQLite database (v0.11).",
)
def compile(
    vault: Path | None,
    folder: str | None,
    model: str,
    num_ctx: int,
    max_workers: int,
    usage: bool,
    usage_db: Path | None,
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

    console.print(
        f"[dim]Found {note_count} notes to process (parallel: {max_workers})...[/]"
    )

    # v0.11: optional usage tracking + persistence for compile.
    from .usage import UsageTracker

    tracker = UsageTracker() if (usage or usage_db is not None) else None
    store = None
    if usage_db is not None:
        from .usage_store import UsageStore

        store = UsageStore(usage_db)
        console.print(f"[dim]Usage persistence: {usage_db}[/]")

    chat_fn = _make_chat_fn(model, num_ctx, tracker=tracker, usage_store=store)
    wiki_dir = compile_wiki(
        root, vault, chat_fn, subfolder=folder, max_workers=max_workers,
    )

    # Count generated files
    generated = list(wiki_dir.glob("*.md"))
    console.print(
        f"\n[bold green]LLM-Wiki compiled![/] "
        f"{len(generated)} pages written to {wiki_dir}"
    )
    console.print("[dim]Open in Obsidian: index.md is the entry point.[/]")

    if tracker is not None and tracker.total_calls > 0:
        usage_table = Table(title="Compile Token Usage (v0.11)")
        usage_table.add_column("Phase", style="bold")
        usage_table.add_column("Calls", justify="right")
        usage_table.add_column("Prompt", justify="right")
        usage_table.add_column("Completion", justify="right")
        usage_table.add_column("Elapsed (s)", justify="right")
        for phase, bucket in sorted(tracker.by_phase().items()):
            usage_table.add_row(
                phase,
                str(int(bucket["calls"])),
                f"{int(bucket['prompt']):,}",
                f"{int(bucket['completion']):,}",
                f"{bucket['elapsed']:.1f}",
            )
        usage_table.add_row(
            "[bold]TOTAL",
            str(tracker.total_calls),
            f"[bold]{tracker.total_prompt_tokens:,}",
            f"[bold]{tracker.total_completion_tokens:,}",
            f"[bold]{tracker.total_elapsed:.1f}",
        )
        console.print()
        console.print(usage_table)


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
@click.option(
    "--auto-rebuild",
    is_flag=True,
    help="Automatically rescan and rebuild Layer 2 sub-trees when changes are detected.",
)
@click.option(
    "--model",
    default="ollama/gemma4:26b",
    help="Model id used for --auto-rebuild cache key.",
)
def watch(
    vault: Path | None,
    folder: str | None,
    interval: int,
    auto_rebuild: bool,
    model: str,
) -> None:
    """Watch the vault for file changes and report them in real time.

    Polls the vault directory at the given interval and prints a summary
    whenever notes are added, modified, or deleted.

    With ``--auto-rebuild``, each detected change triggers an automatic
    rescan + Layer 2 sub-tree rebuild for LONG notes, keeping the
    PageIndex cache warm while editing in Obsidian.
    """
    from datetime import datetime as _dt

    from .watcher import detect_changes, save_state

    vault = _resolve_vault(vault)
    scope = f"{vault}{('/' + folder) if folder else ''}"
    console.print(f"[bold cyan]Watching[/] {scope} (poll every {interval}s)")
    if auto_rebuild:
        console.print("[dim]Auto-rebuild enabled: changes trigger rescan + cache rebuild.[/]")
    console.print("[dim]Press Ctrl+C to stop.[/]\n")

    # Initial snapshot
    save_state(vault, folder)
    console.print("[dim]Initial snapshot saved.[/]")

    try:
        while True:
            time.sleep(interval)
            changes = detect_changes(vault, folder)
            if changes.has_changes:
                save_state(vault, folder)
                ts = _dt.now().strftime("%H:%M:%S")
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

                if auto_rebuild:
                    console.print("[dim]  Rescanning...[/]")
                    root = scan_folder(vault, folder)
                    long_count = sum(
                        1
                        for n in root.walk()
                        if n.kind == "note" and n.tier == NoteTier.LONG
                    )
                    if long_count > 0:
                        built, from_cache = build_long_subtrees(
                            root,
                            vault_root=vault,
                            model_id=model,
                            chat_fn=None,
                        )
                        console.print(
                            f"[dim]  Rebuilt {built} sub-tree(s), "
                            f"{from_cache} from cache.[/]"
                        )
                    else:
                        console.print("[dim]  No LONG notes to rebuild.[/]")
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
    "--extra-vault",
    "extra_vaults",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Additional vault paths for multi-vault search (v0.7, repeatable).",
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
@click.option(
    "--open",
    "open_cited",
    is_flag=True,
    help="Open cited notes in the editor via notesmd-cli (instead of just printing hints).",
)
@click.option(
    "--tag",
    "filter_tags",
    multiple=True,
    help="Only consider notes with this tag (repeatable). E.g. --tag research --tag ml",
)
@click.option(
    "--after",
    "filter_after",
    default=None,
    help="Only consider notes with date >= this value. E.g. --after 2024-01",
)
@click.option(
    "--before",
    "filter_before",
    default=None,
    help="Only consider notes with date <= this value. E.g. --before 2025-06",
)
@click.option(
    "--max-workers",
    default=4,
    type=int,
    help="Parallel LLM workers for summarization. Default: 4. Set to 1 for sequential.",
)
@click.option(
    "--decompose",
    is_flag=True,
    help="Decompose complex queries into sub-questions and synthesize answers (v0.7).",
)
@click.option(
    "--usage",
    is_flag=True,
    help="Print token usage breakdown at the end of the query (v0.8).",
)
@click.option(
    "--max-tokens",
    "max_tokens",
    default=None,
    type=int,
    help="Hard cap on total tokens for this query. Loop aborts when exceeded (v0.9).",
)
@click.option(
    "--json-mode",
    "json_mode",
    is_flag=True,
    help="Use JSON-schema prompts + parser for SELECT/EVALUATE (v0.10).",
)
@click.option(
    "--reuse-context",
    "reuse_context",
    is_flag=True,
    help="Compact path_so_far + suppress already-shown candidates (v0.10).",
)
@click.option(
    "--per-vault",
    "per_vault",
    is_flag=True,
    help="Run retrieval independently per vault then synthesize (v0.12).",
)
@click.option(
    "--token-split",
    "token_split",
    default=None,
    help="Per-phase budget split, e.g. '20:60:20' = summarize:retrieve:synth (v0.14).",
)
@click.option(
    "--prompt-cache",
    "prompt_cache",
    is_flag=True,
    help="Send stable system prefixes separately for Ollama KV-cache reuse (v0.14).",
)
@click.option(
    "--allow-partial",
    "allow_partial",
    is_flag=True,
    help="With --per-vault, keep going when a vault fails and synthesize from the rest (v0.16).",
)
@click.option(
    "--retry-failed",
    "retry_failed",
    default=0,
    type=int,
    help="With --per-vault --allow-partial, retry failed vaults N times (v0.17).",
)
def ask(
    query: str,
    vault: Path | None,
    extra_vaults: tuple[Path, ...],
    folder: str,
    model: str,
    num_ctx: int,
    skip_summaries: bool,
    open_cited: bool,
    filter_tags: tuple[str, ...],
    filter_after: str | None,
    filter_before: str | None,
    max_workers: int,
    decompose: bool,
    usage: bool,
    max_tokens: int | None,
    json_mode: bool,
    reuse_context: bool,
    per_vault: bool,
    token_split: str | None,
    prompt_cache: bool,
    allow_partial: bool,
    retry_failed: int,
) -> None:
    """Run a multi-hop reasoning query against one or more vault folders."""
    console.print(f"[bold cyan]Q:[/] {query}")

    # Resolve the primary vault and collect any --extra-vault paths.
    primary = _resolve_vault(vault)
    all_vaults: list[Path] = [primary]
    for extra in extra_vaults:
        try:
            abs_extra = extra.expanduser().resolve()
        except OSError:
            abs_extra = extra
        if abs_extra not in [v.resolve() for v in all_vaults]:
            all_vaults.append(extra)

    multi_vault = len(all_vaults) > 1
    if multi_vault:
        vault_list = ", ".join(str(v) for v in all_vaults)
        console.print(f"  vaults=[{vault_list}]  folder={folder}  model={model}\n")
    else:
        console.print(f"  vault={primary}  folder={folder}  model={model}\n")

    # Keep a single canonical vault for cache / log paths.
    vault = primary

    start = time.time()

    # v0.8: optional token usage tracking (v0.9: also auto-enabled
    # when --max-tokens is set). v0.14: when --token-split is set
    # we also always need a tracker so budget enforcement can work.
    from .usage import UsageTracker
    tracker = UsageTracker() if (usage or max_tokens is not None or token_split) else None
    chat_fn = _make_chat_fn(model, num_ctx, tracker=tracker)

    # v0.14 optional prompt-caching chat_fn (split system/user).
    system_chat_fn = None
    if prompt_cache:
        system_chat_fn = _make_system_chat_fn(model, num_ctx, tracker=tracker)

    # v0.14 budget split: parse --token-split into per-phase caps.
    summarize_cap, retrieve_cap, _synth_cap = _parse_token_split(
        token_split, max_tokens,
    )
    # retrieve_cap covers BOTH traversal and the final synthesis call,
    # so we combine it with the synth allowance. When split is None,
    # the top-level max_tokens cap applies as usual.
    if retrieve_cap is not None and _synth_cap is not None:
        effective_retrieve_cap: int | None = retrieve_cap + _synth_cap
    else:
        effective_retrieve_cap = max_tokens
    if token_split:
        console.print(
            f"[dim]token-split: summarize={summarize_cap:,}, "
            f"retrieve+synth={effective_retrieve_cap:,}[/]"
        )

    # Step 1: scan (multi-vault aware)
    if multi_vault:
        console.print(f"[dim]1/4 Scanning {len(all_vaults)} vaults...[/]")
        root = scan_multi_vault([(v, folder) for v in all_vaults])
    else:
        console.print("[dim]1/4 Scanning vault...[/]")
        root = scan_folder(vault, folder)

    # Apply frontmatter filters (v0.6).
    active_filters = list(filter_tags) or None
    if active_filters or filter_after or filter_before:
        root = filter_tree(
            root, tags=active_filters, after=filter_after, before=filter_before,
        )
        filter_desc = []
        if active_filters:
            filter_desc.append(f"tags={active_filters}")
        if filter_after:
            filter_desc.append(f"after={filter_after}")
        if filter_before:
            filter_desc.append(f"before={filter_before}")
        console.print(f"[dim]    filter: {', '.join(filter_desc)}[/]")

    note_count = sum(1 for n in root.walk() if n.kind == "note")
    long_count = sum(
        1 for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
    )

    if note_count == 0:
        console.print(f"[red]No notes found in {vault / folder}[/]")
        sys.exit(1)

    # Step 2: summarize atomic notes (optional, cached, parallel in v0.7,
    # per-vault cache routing in v0.11, budget-aware in v0.14).
    if not skip_summaries:
        console.print(
            f"[dim]2/4 Summarizing atomic notes (parallel: {max_workers} workers)...[/]"
        )
        if multi_vault:
            summarized = summarize_atomic_notes_multi(
                root, chat_fn,
                vault_roots=all_vaults,
                model_id=model,
                max_workers=max_workers,
            )
        else:
            scache = SummaryCache(vault)
            summarized = summarize_atomic_notes(
                root, chat_fn, summary_cache=scache, model_id=model,
                max_workers=max_workers,
                max_tokens=summarize_cap,
                tracker=tracker,
            )
        console.print(f"[dim]    → {summarized} notes summarized (rest from cache)[/]")
    else:
        console.print("[dim]2/4 Skipping summarization (--skip-summaries)[/]")

    # Step 3: build Layer 2 sub-trees for LONG notes (cached)
    if long_count > 0:
        console.print(
            f"[dim]3/4 Building PageIndex sub-trees for {long_count} LONG notes...[/]"
        )
        section_chat_fn = None if skip_summaries else chat_fn
        if multi_vault:
            built, from_cache = build_long_subtrees_multi(
                root,
                vault_roots=all_vaults,
                model_id=model,
                chat_fn=section_chat_fn,
            )
        else:
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

    # Step 4: retrieval loop with live streaming (v0.6 streaming + v0.7 decomposition)
    if decompose:
        console.print("[dim]4/4 Running decomposed multi-hop retrieval...[/]\n")
    else:
        console.print("[dim]4/4 Running multi-hop retrieval loop...[/]\n")

    from .retrieval import TraceStep, run_cross_vault_retrieval, run_decomposed_retrieval

    _phase_icons = {
        "select": "[cyan]SELECT[/]",
        "evaluate": "[yellow]EVAL[/]",
        "cross-ref": "[magenta]XREF[/]",
        "finalize": "[green]DONE[/]",
        "decompose": "[blue]DECOMP[/]",
        "reuse": "[dim]REUSE[/]",
        "budget": "[red]BUDGET[/]",
        "cross-vault": "[bold blue]VAULT[/]",
        "cancel": "[red]CANCEL[/]",
    }

    def _on_event(step: TraceStep) -> None:
        icon = _phase_icons.get(step.phase, step.phase)
        node_part = f" [{escape(step.node_id)}]" if step.node_id else ""
        console.print(f"  {icon}{node_part} {escape(step.detail)}")

    if per_vault and multi_vault:
        # v0.12: rescan each vault into an isolated root so run_cross_vault_retrieval
        # can attribute results per vault. Re-uses the caches already warmed above.
        console.print(
            f"[dim]4/4 Running per-vault retrieval "
            f"across {len(all_vaults)} vaults...[/]\n"
        )
        per_vault_roots = []
        per_vault_labels = []
        per_vault_indexes = []
        for v in all_vaults:
            v_root = scan_folder(v, folder)
            if active_filters or filter_after or filter_before:
                v_root = filter_tree(
                    v_root, tags=active_filters,
                    after=filter_after, before=filter_before,
                )
            # Reuse summary cache since we already warmed it.
            if not skip_summaries:
                summarize_atomic_notes(
                    v_root, chat_fn,
                    summary_cache=SummaryCache(v),
                    model_id=model,
                    max_workers=max_workers,
                )
            v_long = sum(
                1 for n in v_root.walk()
                if n.kind == "note" and n.tier == NoteTier.LONG
            )
            if v_long > 0:
                build_long_subtrees(
                    v_root, vault_root=v, model_id=model,
                    chat_fn=(None if skip_summaries else chat_fn),
                    cache=TreeCache(v),
                )
            per_vault_roots.append(v_root)
            per_vault_labels.append(v.name)
            per_vault_indexes.append(build_link_index(v_root))

        result = run_cross_vault_retrieval(
            query, per_vault_roots, chat_fn,
            link_indexes=per_vault_indexes,
            vault_labels=per_vault_labels,
            on_event=_on_event,
            max_tokens=effective_retrieve_cap, tracker=tracker,
            json_mode=json_mode, reuse_context=reuse_context,
            decompose=decompose,  # v0.13 cross-vault × decompose
            system_chat_fn=system_chat_fn,
            parallel_workers=max_workers,  # v0.15 parallel fan-out
            allow_partial=allow_partial,  # v0.16 partial failure tolerance
            retry_failed=retry_failed,  # v0.17 retry count for failed vaults
        )
    elif decompose:
        result = run_decomposed_retrieval(
            query, root, chat_fn,
            link_index=link_index, on_event=_on_event,
            max_tokens=effective_retrieve_cap, tracker=tracker,
            json_mode=json_mode, reuse_context=reuse_context,
        )
    else:
        result = run_retrieval(
            query, root, chat_fn,
            link_index=link_index, on_event=_on_event,
            max_tokens=effective_retrieve_cap, tracker=tracker,
            json_mode=json_mode, reuse_context=reuse_context,
            system_chat_fn=system_chat_fn,
        )

    elapsed = time.time() - start

    console.print(f"\n[bold green]A:[/] {result.answer}\n")

    if result.cited_nodes:
        console.print("[bold]Cited nodes:[/]")
        for cited in result.cited_nodes:
            console.print(f"  • {cited}")

        if open_cited:
            _open_cited_notes(result.cited_nodes, root, vault=vault)
        else:
            _print_notesmd_open_hints(result.cited_nodes, root, vault=vault)

    console.print(
        f"\n[dim]iterations={result.iterations_used}  elapsed={elapsed:.1f}s[/]"
    )

    # v0.8: print usage summary if tracking is enabled.
    if tracker is not None and tracker.total_calls > 0:
        usage_table = Table(title="Token Usage (v0.8)")
        usage_table.add_column("Phase", style="bold")
        usage_table.add_column("Calls", justify="right")
        usage_table.add_column("Prompt", justify="right")
        usage_table.add_column("Completion", justify="right")
        usage_table.add_column("Elapsed (s)", justify="right")

        for phase, bucket in sorted(tracker.by_phase().items()):
            usage_table.add_row(
                phase,
                str(int(bucket["calls"])),
                f"{int(bucket['prompt']):,}",
                f"{int(bucket['completion']):,}",
                f"{bucket['elapsed']:.1f}",
            )
        usage_table.add_row(
            "[bold]TOTAL",
            str(tracker.total_calls),
            f"[bold]{tracker.total_prompt_tokens:,}",
            f"[bold]{tracker.total_completion_tokens:,}",
            f"[bold]{tracker.total_elapsed:.1f}",
        )
        console.print()
        console.print(usage_table)

        # v0.15: prompt-cache hit rate proxy. Only meaningful when
        # --prompt-cache is active; we still print it at 0% so
        # users can see the flag had no effect when misconfigured.
        cacheable_ratio = tracker.cacheable_ratio()
        cacheable_calls = tracker.cacheable_calls
        if tracker.total_calls > 0:
            console.print(
                f"[dim]Prompt-cache eligible: "
                f"{cacheable_calls}/{tracker.total_calls} calls "
                f"({cacheable_ratio:.1%})[/]"
            )

        # v0.16: inferred latency savings from cache reuse. Only
        # print when there are at least 2 cacheable calls so the
        # first-vs-rest comparison is meaningful.
        savings_info = tracker.cacheable_latency_savings()
        if savings_info["samples"] >= 2:
            console.print(
                f"[dim]  first={savings_info['first_call_seconds']:.2f}s, "
                f"subsequent_mean={savings_info['subsequent_mean_seconds']:.2f}s, "
                f"savings={savings_info['savings_per_call_seconds']:+.2f}s/call, "
                f"inferred_hit_rate={savings_info['inferred_hit_rate']:.1%}[/]"
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


@main.command()
@click.option(
    "--vault",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Obsidian vault root.",
)
@click.option(
    "--extra-vault",
    "extra_vaults",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Additional vault paths for multi-vault search (v0.7, repeatable).",
)
@click.option(
    "--folder", default="Research", help="Subfolder inside the vault. Default: Research",
)
@click.option("--model", default="ollama/gemma4:26b", help="LiteLLM model id.")
@click.option("--num-ctx", default=131072, type=int, help="Ollama context window.")
@click.option(
    "--skip-summaries", is_flag=True, help="Skip atomic-note summarization.",
)
@click.option(
    "--tag",
    "filter_tags",
    multiple=True,
    help="Only consider notes with this tag (repeatable).",
)
@click.option("--after", "filter_after", default=None, help="Notes with date >= value.")
@click.option("--before", "filter_before", default=None, help="Notes with date <= value.")
@click.option(
    "--max-workers", default=4, type=int,
    help="Parallel LLM workers. Default: 4.",
)
@click.option(
    "--decompose", is_flag=True,
    help="Decompose complex queries into sub-questions (v0.7).",
)
@click.option(
    "--usage", is_flag=True,
    help="Print cumulative token usage after each turn (v0.9).",
)
@click.option(
    "--max-tokens",
    "max_tokens",
    default=None,
    type=int,
    help="Per-turn token budget. Each chat turn aborts if exceeded (v0.9).",
)
@click.option(
    "--json-mode",
    "json_mode",
    is_flag=True,
    help="Use JSON-schema prompts + parser for SELECT/EVALUATE (v0.10).",
)
@click.option(
    "--reuse-context",
    "reuse_context",
    is_flag=True,
    help="Compact path_so_far + suppress already-shown candidates (v0.10).",
)
def chat(
    vault: Path | None,
    extra_vaults: tuple[Path, ...],
    folder: str,
    model: str,
    num_ctx: int,
    skip_summaries: bool,
    filter_tags: tuple[str, ...],
    filter_after: str | None,
    filter_before: str | None,
    max_workers: int,
    decompose: bool,
    usage: bool,
    max_tokens: int | None,
    json_mode: bool,
    reuse_context: bool,
) -> None:
    """Interactive multi-turn conversation against a vault folder.

    Maintains conversation history so follow-up questions build on
    previous answers. Type 'quit' or 'exit' to stop, '/clear' to
    reset the conversation.
    """
    from .prompts import rewrite_query_with_context
    from .retrieval import TraceStep as _TraceStep
    from .retrieval import run_decomposed_retrieval
    from .usage import UsageTracker
    from .wiki_links import build_link_index

    # Resolve primary + extra vaults for multi-vault support (v0.7).
    primary = _resolve_vault(vault)
    all_vaults: list[Path] = [primary]
    for extra in extra_vaults:
        try:
            abs_extra = extra.expanduser().resolve()
        except OSError:
            abs_extra = extra
        if abs_extra not in [v.resolve() for v in all_vaults]:
            all_vaults.append(extra)

    multi_vault = len(all_vaults) > 1
    vault = primary
    vault_str = (
        f"{len(all_vaults)} vaults: " + ", ".join(v.name for v in all_vaults)
        if multi_vault
        else str(vault)
    )

    console.print(
        Panel(
            f"[bold cyan]pagewiki chat[/] — interactive mode\n"
            f"{'vaults' if multi_vault else 'vault'}={vault_str}  "
            f"folder={folder}  model={model}\n"
            f"Type [bold]quit[/] to exit, [bold]/clear[/] to reset history.",
            title="pagewiki v0.7",
        )
    )

    # v0.9: tracker is cumulative across the whole chat session.
    # Auto-enabled when --max-tokens is set even without --usage.
    tracker = UsageTracker() if (usage or max_tokens is not None) else None
    chat_fn = _make_chat_fn(model, num_ctx, tracker=tracker)

    # Pre-scan once so repeated queries reuse the same tree.
    console.print("[dim]Scanning vault...[/]")
    if multi_vault:
        root = scan_multi_vault([(v, folder) for v in all_vaults])
    else:
        root = scan_folder(vault, folder)

    # Apply filters.
    active_filters = list(filter_tags) or None
    if active_filters or filter_after or filter_before:
        root = filter_tree(
            root, tags=active_filters, after=filter_after, before=filter_before,
        )

    note_count = sum(1 for n in root.walk() if n.kind == "note")
    if note_count == 0:
        console.print(f"[red]No notes found in {vault / folder}[/]")
        sys.exit(1)

    # Summarize atomic notes (cached, parallel in v0.7).
    if not skip_summaries:
        console.print(
            f"[dim]Summarizing atomic notes (parallel: {max_workers})...[/]"
        )
        scache = SummaryCache(vault)
        summarized = summarize_atomic_notes(
            root, chat_fn, summary_cache=scache, model_id=model,
            max_workers=max_workers,
        )
        console.print(f"[dim]  → {summarized} summarized (rest from cache)[/]")

    # Build Layer 2 sub-trees.
    long_count = sum(
        1 for n in root.walk() if n.kind == "note" and n.tier == NoteTier.LONG
    )
    if long_count > 0:
        console.print(f"[dim]Building sub-trees for {long_count} LONG notes...[/]")
        section_chat_fn = None if skip_summaries else chat_fn
        built, from_cache = build_long_subtrees(
            root, vault_root=vault, model_id=model,
            chat_fn=section_chat_fn, cache=TreeCache(vault),
        )
        console.print(f"[dim]  → {built} built, {from_cache} from cache[/]")

    link_index = build_link_index(root)
    console.print(f"[dim]Ready! {note_count} notes loaded.[/]\n")

    # Conversation state.
    history: list[tuple[str, str]] = []

    _phase_icons = {
        "select": "[cyan]SELECT[/]",
        "evaluate": "[yellow]EVAL[/]",
        "cross-ref": "[magenta]XREF[/]",
        "finalize": "[green]DONE[/]",
        "decompose": "[blue]DECOMP[/]",
        "reuse": "[dim]REUSE[/]",
        "budget": "[red]BUDGET[/]",
    }

    def _on_event(step: _TraceStep) -> None:
        icon = _phase_icons.get(step.phase, step.phase)
        node_part = f" [{escape(step.node_id)}]" if step.node_id else ""
        console.print(f"  {icon}{node_part} {escape(step.detail)}")

    turn = 0
    while True:
        try:
            raw = console.input("[bold cyan]Q:[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/]")
            break

        query = raw.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            console.print("[dim]Bye![/]")
            break
        if query == "/clear":
            history.clear()
            console.print("[dim]Conversation history cleared.[/]\n")
            continue

        turn += 1
        start = time.time()

        # v0.9: snapshot pre-turn usage so we can compute this turn's delta.
        pre_turn_tokens = tracker.total_tokens if tracker is not None else 0
        pre_turn_calls = tracker.total_calls if tracker is not None else 0

        # Rewrite follow-up queries into standalone form.
        effective_query = query
        if history:
            rewritten = chat_fn(rewrite_query_with_context(query, history)).strip()
            if rewritten and rewritten != query:
                console.print(f"[dim]  → rewritten: {rewritten}[/]")
                effective_query = rewritten

        # v0.9: per-turn budget tracked against pre_turn_tokens snapshot.
        # When --max-tokens is set, each turn gets its own allowance.
        turn_max_tokens = (
            (pre_turn_tokens + max_tokens) if max_tokens is not None else None
        )

        if decompose:
            result = run_decomposed_retrieval(
                effective_query, root, chat_fn,
                link_index=link_index, on_event=_on_event,
                max_tokens=turn_max_tokens, tracker=tracker,
                json_mode=json_mode, reuse_context=reuse_context,
            )
        else:
            result = run_retrieval(
                effective_query, root, chat_fn,
                link_index=link_index, on_event=_on_event, history=history,
                max_tokens=turn_max_tokens, tracker=tracker,
                json_mode=json_mode, reuse_context=reuse_context,
            )

        elapsed = time.time() - start

        console.print(f"\n[bold green]A:[/] {result.answer}\n")

        if result.cited_nodes:
            console.print("[bold]Cited:[/]")
            for cited in result.cited_nodes:
                console.print(f"  • {cited}")

        # v0.9: per-turn usage report.
        if tracker is not None:
            delta_tokens = tracker.total_tokens - pre_turn_tokens
            delta_calls = tracker.total_calls - pre_turn_calls
            console.print(
                f"[dim]turn={turn}  iterations={result.iterations_used}  "
                f"elapsed={elapsed:.1f}s  "
                f"tokens={delta_tokens:,} ({delta_calls} calls)  "
                f"cumulative={tracker.total_tokens:,}[/]\n"
            )
        else:
            console.print(
                f"[dim]turn={turn}  iterations={result.iterations_used}  "
                f"elapsed={elapsed:.1f}s[/]\n"
            )

        # Append to conversation history.
        history.append((query, result.answer[:500]))

        # Log each turn.
        record = QueryRecord(
            query=query,
            answer=result.answer,
            cited_nodes=result.cited_nodes,
            model=model,
            elapsed_seconds=elapsed,
        )
        write_log(vault / folder, record)

    # v0.9: print cumulative usage summary on exit.
    if tracker is not None and tracker.total_calls > 0:
        usage_table = Table(title="Chat Session Token Usage (v0.9)")
        usage_table.add_column("Phase", style="bold")
        usage_table.add_column("Calls", justify="right")
        usage_table.add_column("Prompt", justify="right")
        usage_table.add_column("Completion", justify="right")
        usage_table.add_column("Elapsed (s)", justify="right")
        for phase, bucket in sorted(tracker.by_phase().items()):
            usage_table.add_row(
                phase,
                str(int(bucket["calls"])),
                f"{int(bucket['prompt']):,}",
                f"{int(bucket['completion']):,}",
                f"{bucket['elapsed']:.1f}",
            )
        usage_table.add_row(
            "[bold]TOTAL",
            str(tracker.total_calls),
            f"[bold]{tracker.total_prompt_tokens:,}",
            f"[bold]{tracker.total_completion_tokens:,}",
            f"[bold]{tracker.total_elapsed:.1f}",
        )
        console.print(usage_table)


@main.command()
@click.option(
    "--vault",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Obsidian vault root.",
)
@click.option(
    "--extra-vault",
    "extra_vaults",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Additional vault paths for multi-vault search.",
)
@click.option(
    "--folder", default=None, help="Subfolder inside the vault.",
)
@click.option("--model", default="ollama/gemma4:26b", help="LiteLLM model id.")
@click.option("--num-ctx", default=131072, type=int, help="Ollama context window.")
@click.option(
    "--max-workers", default=4, type=int, help="Parallel LLM workers.",
)
@click.option("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1")
@click.option("--port", default=8000, type=int, help="Bind port. Default: 8000")
@click.option(
    "--usage-db",
    "usage_db",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional SQLite path for persistent usage tracking (v0.10).",
)
@click.option(
    "--prompt-cache",
    "enable_prompt_cache",
    is_flag=True,
    help="Attach a prompt-cache chat_fn so WS clients can opt in per-request (v0.16).",
)
@click.option(
    "--retention-days",
    "retention_days",
    default=None,
    type=int,
    help="Periodically prune usage events older than N days (v0.17). Requires --usage-db.",
)
@click.option(
    "--retention-interval",
    "retention_interval",
    default=3600,
    type=int,
    help="Seconds between retention passes. Default: 3600 (1 hour).",
)
def serve(
    vault: Path | None,
    extra_vaults: tuple[Path, ...],
    folder: str | None,
    model: str,
    num_ctx: int,
    max_workers: int,
    host: str,
    port: int,
    usage_db: Path | None,
    enable_prompt_cache: bool,
    retention_days: int | None,
    retention_interval: int,
) -> None:
    """Run pagewiki as an HTTP API server (v0.7).

    Scans the vault once at startup, keeps the tree warm in memory,
    and exposes /health, /scan, /ask, /chat endpoints. Chat sessions
    maintain conversation history server-side.

    Requires the optional ``server`` extra:
        pip install 'pagewiki[server]'
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn is not installed.[/]\n"
            "Install the server extra:\n"
            "    pip install 'pagewiki[server]'"
        )
        sys.exit(1)

    from .server import build_initial_state, create_app

    # Resolve primary + extra vaults.
    primary = _resolve_vault(vault)
    all_vaults: list[Path] = [primary]
    for extra in extra_vaults:
        try:
            abs_extra = extra.expanduser().resolve()
        except OSError:
            abs_extra = extra
        if abs_extra not in [v.resolve() for v in all_vaults]:
            all_vaults.append(extra)

    console.print(f"[bold cyan]Starting pagewiki server[/] on http://{host}:{port}")
    console.print(
        f"  vaults={len(all_vaults)}  folder={folder}  model={model}"
    )
    console.print("[dim]Warming up: scanning + summarizing...[/]")

    # v0.9: cumulative UsageTracker. v0.10: optional UsageStore.
    from .usage import UsageTracker

    tracker = UsageTracker()

    usage_store = None
    if usage_db is not None:
        from .usage_store import UsageStore

        usage_store = UsageStore(usage_db)
        console.print(f"[dim]Usage persistence: {usage_db}[/]")

    chat_fn = _make_chat_fn(
        model, num_ctx, tracker=tracker, usage_store=usage_store,
    )

    # v0.16: optionally build a system-split chat_fn so WebSocket
    # clients can request prompt-cache mode on a per-query basis.
    system_chat_fn = None
    if enable_prompt_cache:
        system_chat_fn = _make_system_chat_fn(
            model, num_ctx, tracker=tracker, usage_store=usage_store,
        )
        console.print("[dim]Prompt cache: enabled (clients can opt in per-request)[/]")

    try:
        state = build_initial_state(
            all_vaults,
            folder=folder,
            model=model,
            num_ctx=num_ctx,
            max_workers=max_workers,
            chat_fn=chat_fn,
        )
    except Exception as e:
        console.print(f"[red]Failed to build initial state: {e}[/]")
        sys.exit(1)

    # Attach the same tracker + store + optional system_chat_fn to state.
    state.tracker = tracker
    state.usage_store = usage_store
    state.system_chat_fn = system_chat_fn

    note_count = sum(1 for n in state.root.walk() if n.kind == "note")
    console.print(f"[dim]Ready! {note_count} notes loaded. Press Ctrl+C to stop.[/]")

    # v0.17: optional background retention thread. Validates flags
    # and quietly no-ops when requirements aren't met (e.g. no
    # --usage-db) so misconfiguration doesn't kill the server.
    if retention_days is not None and retention_days > 0:
        if usage_store is None:
            console.print(
                "[yellow]--retention-days set but no --usage-db; "
                "retention disabled.[/]"
            )
        else:
            import threading as _threading

            def _retention_loop() -> None:
                import time as _time
                while True:
                    try:
                        deleted = usage_store.prune_older_than_days(retention_days)
                        if deleted > 0:
                            console.print(
                                f"[dim][retention] pruned {deleted:,} events "
                                f"older than {retention_days}d[/]"
                            )
                    except Exception as e:  # pragma: no cover
                        console.print(f"[yellow][retention] error: {e}[/]")
                    _time.sleep(retention_interval)

            _threading.Thread(
                target=_retention_loop, name="usage-retention", daemon=True,
            ).start()
            console.print(
                f"[dim]Retention: prune >{retention_days}d every "
                f"{retention_interval}s[/]"
            )

    app = create_app(state)
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command("usage-report")
@click.option(
    "--db",
    "db_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the SQLite usage database (written by `serve --usage-db`).",
)
@click.option(
    "--since",
    default=None,
    help="Only include events newer than this ISO date (e.g. 2024-11-01).",
)
@click.option(
    "--until",
    default=None,
    help="Only include events older than this ISO date.",
)
@click.option(
    "--phase",
    default=None,
    help="Filter to a single phase (select/evaluate/final/...).",
)
@click.option(
    "--recent",
    default=0,
    type=int,
    help="Also print the N most recent events as a detail list.",
)
@click.option(
    "--daily",
    is_flag=True,
    help="Rollup and print daily aggregates (v0.12).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "csv", "json"]),
    default="table",
    help="Output format. 'table' (default) uses Rich; csv/json are machine-readable (v0.13).",
)
@click.option(
    "--prune-older-than",
    "prune_days",
    default=None,
    type=int,
    help="Delete raw events older than N days (rollups preserved) (v0.14).",
)
def usage_report(
    db_path: Path,
    since: str | None,
    until: str | None,
    phase: str | None,
    recent: int,
    daily: bool,
    output_format: str,
    prune_days: int | None,
) -> None:
    """Query a SQLite usage database and print a Rich breakdown (v0.11).

    Pair with ``pagewiki serve --usage-db PATH`` which persists every
    LLM call into the same database. Useful for daily/weekly token
    cost audits, cost attribution by phase, and trend analysis
    across many sessions.
    """
    from datetime import datetime

    from .usage_store import UsageStore

    def _parse_iso(s: str | None) -> float | None:
        if s is None:
            return None
        # Accept bare dates or full ISO timestamps.
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            console.print(f"[red]Invalid ISO timestamp: {s}[/]")
            sys.exit(1)
        return dt.timestamp()

    since_ts = _parse_iso(since)
    until_ts = _parse_iso(until)

    store = UsageStore(db_path)

    # v0.14: optional rolling retention. Rolls up affected days to
    # preserve historical aggregates before deleting raw events.
    if prune_days is not None and prune_days > 0:
        deleted = store.prune_older_than_days(prune_days)
        console.print(
            f"[dim]Pruned {deleted:,} events older than {prune_days} days "
            f"(daily rollups preserved).[/]"
        )

    summary = store.query_summary(since=since_ts, until=until_ts)

    by_phase_items = list(summary.by_phase.items())
    if phase:
        by_phase_items = [(p, b) for p, b in by_phase_items if p == phase]

    events = []
    if recent > 0:
        events = store.query_events(
            since=since_ts,
            until=until_ts,
            phase=phase,
            limit=recent,
        )

    daily_rows: list[dict] = []
    days_written = 0
    if daily:
        days_written = store.rollup_range(since=since, until=until)
        daily_rows = store.query_daily(since=since, until=until)

    # v0.13: machine-readable output branches — CSV and JSON emit
    # clean stdout (no Rich markup) so the command pipes cleanly.
    if output_format == "json":
        import json as _json

        payload = {
            "db": str(db_path),
            "filter": {
                "since": since,
                "until": until,
                "phase": phase,
            },
            "total": {
                "calls": summary.total_calls,
                "prompt": summary.total_prompt,
                "completion": summary.total_completion,
                "elapsed": summary.total_elapsed,
            },
            "by_phase": {
                p: {
                    "calls": int(b["calls"]),
                    "prompt": int(b["prompt"]),
                    "completion": int(b["completion"]),
                    "elapsed": float(b["elapsed"]),
                }
                for p, b in by_phase_items
            },
            "recent": [
                {
                    "timestamp": e.timestamp,
                    "phase": e.phase,
                    "prompt": e.prompt,
                    "completion": e.completion,
                    "elapsed": e.elapsed,
                }
                for e in events
            ],
            "daily": daily_rows,
        }
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        store.close()
        return

    if output_format == "csv":
        import csv as _csv
        import io as _io

        buf = _io.StringIO()
        writer = _csv.writer(buf)
        writer.writerow(["section", "key", "calls", "prompt", "completion", "elapsed"])
        writer.writerow(
            [
                "total",
                "-",
                summary.total_calls,
                summary.total_prompt,
                summary.total_completion,
                f"{summary.total_elapsed:.3f}",
            ]
        )
        for p, b in sorted(by_phase_items):
            writer.writerow(
                [
                    "phase",
                    p,
                    int(b["calls"]),
                    int(b["prompt"]),
                    int(b["completion"]),
                    f"{float(b['elapsed']):.3f}",
                ]
            )
        for row in daily_rows:
            writer.writerow(
                [
                    "daily",
                    row["date"],
                    row["total_calls"],
                    row["total_prompt"],
                    row["total_completion"],
                    f"{row['total_elapsed']:.3f}",
                ]
            )
        for e in events:
            writer.writerow(
                [
                    "event",
                    datetime.fromtimestamp(e.timestamp).isoformat(),
                    1,
                    e.prompt,
                    e.completion,
                    f"{e.elapsed:.3f}",
                ]
            )
        click.echo(buf.getvalue().rstrip("\n"))
        store.close()
        return

    # Default: human-friendly Rich table output.
    title_parts = [f"Usage Report ({db_path.name})"]
    if since:
        title_parts.append(f"since={since}")
    if until:
        title_parts.append(f"until={until}")
    if phase:
        title_parts.append(f"phase={phase}")

    console.print(f"[bold cyan]{' | '.join(title_parts)}[/]")
    console.print(
        f"[dim]total_calls={summary.total_calls:,}  "
        f"prompt={summary.total_prompt:,}  "
        f"completion={summary.total_completion:,}  "
        f"elapsed={summary.total_elapsed:.1f}s[/]\n"
    )

    if summary.total_calls == 0:
        console.print("[yellow]No events match the filter.[/]")
        store.close()
        return

    table = Table(title="By Phase")
    table.add_column("Phase", style="bold")
    table.add_column("Calls", justify="right")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Elapsed (s)", justify="right")
    for p, b in sorted(by_phase_items):
        table.add_row(
            p,
            f"{int(b['calls']):,}",
            f"{int(b['prompt']):,}",
            f"{int(b['completion']):,}",
            f"{float(b['elapsed']):.1f}",
        )
    console.print(table)

    if events:
        console.print(f"\n[bold]Most recent {len(events)} events:[/]")
        for e in events:
            ts = datetime.fromtimestamp(e.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            console.print(
                f"  [dim]{ts}[/] [cyan]{e.phase:<10}[/] "
                f"p={e.prompt:,} c={e.completion:,} "
                f"({e.elapsed:.1f}s)"
            )

    if daily:
        if daily_rows:
            daily_table = Table(
                title=f"Daily Rollup (rolled {days_written} new days)"
            )
            daily_table.add_column("Date", style="bold")
            daily_table.add_column("Calls", justify="right")
            daily_table.add_column("Prompt", justify="right")
            daily_table.add_column("Completion", justify="right")
            daily_table.add_column("Elapsed (s)", justify="right")
            for row in daily_rows:
                daily_table.add_row(
                    row["date"],
                    f"{row['total_calls']:,}",
                    f"{row['total_prompt']:,}",
                    f"{row['total_completion']:,}",
                    f"{row['total_elapsed']:.1f}",
                )
            console.print()
            console.print(daily_table)
        else:
            console.print("\n[dim]No daily rollup rows in range.[/]")

    store.close()


if __name__ == "__main__":
    sys.exit(main())
