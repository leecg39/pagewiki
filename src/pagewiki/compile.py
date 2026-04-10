"""v0.3 LLM-Wiki compiler — extract entities from vault notes and
compile a persistent, cross-referenced wiki in ``{vault}/LLM-Wiki/``.

Follows Karpathy's LLM-Wiki pattern:

  * **Raw Sources**: the Obsidian vault notes (read-only).
  * **Wiki Layer**: LLM-generated entity pages + index in ``LLM-Wiki/``.
  * **Schema Layer**: ``index.md`` as the content-oriented catalog.

The compiler works in two passes:

  1. **Extract**: for each note, ask the LLM to identify key entities
     (people, concepts, methods, datasets, etc.) with one-line
     descriptions. Results are aggregated across all notes.

  2. **Compile**: for each unique entity, generate a wiki page that
     synthesizes mentions from all source notes. Cross-references
     between entity pages are written as ``[[entity]]`` links so
     Obsidian indexes them in the graph view.

Design notes
------------

* Entity extraction and page generation both go through the injected
  ``chat_fn`` so tests can script responses without LLM access.
* The compiler is idempotent: re-running overwrites ``LLM-Wiki/``
  with fresh content. Incremental updates are deferred to v0.4.
* All filesystem writes go to ``{vault}/LLM-Wiki/`` — source notes
  are never modified.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .tree import TreeNode

ChatFn = Callable[[str], str]

WIKI_DIRNAME = "LLM-Wiki"
INDEX_FILENAME = "index.md"
LOG_FILENAME = "log.md"
ENTITIES_CACHE_FILENAME = "entities-cache.json"

# Safety cap — skip notes above this size to avoid blowing context
_MAX_NOTE_CHARS = 12_000


@dataclass(frozen=True)
class EntityMention:
    """One mention of an entity found in a specific source note."""

    source_title: str
    source_node_id: str
    description: str


@dataclass
class Entity:
    """Aggregated entity across all source notes."""

    name: str
    category: str = ""
    mentions: list[EntityMention] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────


def _extract_entities_prompt(note_title: str, note_content: str) -> str:
    """Ask the LLM to extract key entities from a single note."""
    return (
        "다음 노트에서 핵심 엔티티(인물, 개념, 방법론, 데이터셋, 조직, 기술 등)를 "
        "추출하세요. 각 엔티티마다 한 줄로 출력합니다.\n\n"
        "출력 형식 (한 줄에 하나):\n"
        "  ENTITY: <이름> | <카테고리> | <한 줄 설명>\n"
        "  ENTITY: <이름> | <카테고리> | <한 줄 설명>\n\n"
        "카테고리는: person, concept, method, dataset, organization, technology, other 중 하나.\n"
        "엔티티가 없으면 NONE 한 줄만 출력하세요.\n"
        "다른 설명은 추가하지 마세요.\n\n"
        f"[노트 제목] {note_title}\n\n"
        f"[본문]\n{note_content[:_MAX_NOTE_CHARS]}"
    )


def _generate_page_prompt(
    entity_name: str,
    category: str,
    mentions: list[EntityMention],
    all_entity_names: list[str],
) -> str:
    """Ask the LLM to write a wiki page for a single entity."""
    mention_lines = []
    for m in mentions:
        mention_lines.append(f"- [{m.source_title}]: {m.description}")
    mentions_text = "\n".join(mention_lines)

    # Suggest cross-references to other entities
    other_entities = [n for n in all_entity_names if n != entity_name]
    xref_hint = ""
    if other_entities:
        xref_hint = (
            "\n\n관련 엔티티가 있으면 [[엔티티이름]] 형식으로 본문 안에 "
            "자연스럽게 교차참조하세요. 참고할 수 있는 다른 엔티티:\n"
            + ", ".join(other_entities[:20])
        )

    return (
        f"다음 엔티티에 대한 위키 페이지를 한국어로 작성하세요.\n\n"
        f"엔티티: {entity_name}\n"
        f"카테고리: {category}\n\n"
        f"[소스 노트에서의 언급]\n{mentions_text}\n\n"
        "위 정보를 종합하여 간결한 위키 페이지를 작성하세요:\n"
        "- 첫 줄은 `# {엔티티이름}` 제목\n"
        "- 2~4 문단으로 핵심 내용 요약\n"
        "- 마지막에 `## 출처` 섹션으로 소스 노트 나열"
        f"{xref_hint}\n\n"
        "본문만 출력하세요."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_RE = re.compile(
    r"^ENTITY:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$", re.MULTILINE
)


def parse_entities(response: str) -> list[tuple[str, str, str]]:
    """Parse ``ENTITY: name | category | description`` lines.

    Returns list of ``(name, category, description)`` tuples.
    Returns empty list if response contains ``NONE``.
    """
    if response.strip().upper() == "NONE":
        return []
    results = []
    for match in _ENTITY_RE.finditer(response):
        name = match.group(1).strip()
        category = match.group(2).strip().lower()
        desc = match.group(3).strip()
        if name:
            results.append((name, category, desc))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Compiler
# ─────────────────────────────────────────────────────────────────────────────


def extract_entities_from_tree(
    root: TreeNode,
    chat_fn: ChatFn,
) -> dict[str, Entity]:
    """Pass 1: extract entities from every note in the tree.

    Returns a dict keyed by normalized entity name (lowercase).
    """
    entities: dict[str, Entity] = {}

    for node in root.walk():
        if node.kind != "note":
            continue
        if node.file_path is None or not node.file_path.exists():
            continue

        content = node.file_path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        prompt = _extract_entities_prompt(node.title, content)
        response = chat_fn(prompt)
        parsed = parse_entities(response)

        for name, category, description in parsed:
            key = name.strip().lower()
            if key not in entities:
                entities[key] = Entity(name=name, category=category)
            entities[key].mentions.append(
                EntityMention(
                    source_title=node.title,
                    source_node_id=node.node_id,
                    description=description,
                )
            )

    return entities


def generate_wiki_pages(
    entities: dict[str, Entity],
    chat_fn: ChatFn,
) -> dict[str, str]:
    """Pass 2: generate a wiki page for each entity.

    Returns a dict of ``{filename: markdown_content}``.
    """
    all_names = [e.name for e in entities.values()]
    pages: dict[str, str] = {}

    for entity in entities.values():
        if not entity.mentions:
            continue

        prompt = _generate_page_prompt(
            entity.name,
            entity.category,
            entity.mentions,
            all_names,
        )
        page_content = chat_fn(prompt).strip()
        # Sanitize filename: replace spaces with hyphens, remove special chars
        safe_name = re.sub(r"[^\w\s-]", "", entity.name).strip()
        safe_name = re.sub(r"\s+", "-", safe_name)
        filename = f"{safe_name}.md"
        pages[filename] = page_content

    return pages


def generate_index(entities: dict[str, Entity]) -> str:
    """Generate ``index.md`` — the content-oriented catalog.

    Groups entities by category and lists each with a one-line
    description drawn from the first mention.
    """
    lines = [
        "# LLM-Wiki Index",
        "",
        f"> Auto-generated by pagewiki v0.3 on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    by_category: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities.values():
        cat = entity.category or "other"
        by_category[cat].append(entity)

    category_labels = {
        "person": "인물 (People)",
        "concept": "개념 (Concepts)",
        "method": "방법론 (Methods)",
        "dataset": "데이터셋 (Datasets)",
        "organization": "조직 (Organizations)",
        "technology": "기술 (Technology)",
        "other": "기타 (Other)",
    }

    for cat in sorted(by_category.keys()):
        label = category_labels.get(cat, cat)
        lines.append(f"## {label}")
        lines.append("")
        for entity in sorted(by_category[cat], key=lambda e: e.name):
            safe_name = re.sub(r"[^\w\s-]", "", entity.name).strip()
            safe_name = re.sub(r"\s+", "-", safe_name)
            first_desc = entity.mentions[0].description if entity.mentions else ""
            lines.append(f"- [[{safe_name}]] — {first_desc}")
        lines.append("")

    return "\n".join(lines)


def generate_log_entry(
    entities: dict[str, Entity],
    source_count: int,
) -> str:
    """Generate a log.md entry for this compilation run."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"## [{ts}] COMPILE\n\n"
        f"- Sources scanned: {source_count}\n"
        f"- Entities extracted: {len(entities)}\n"
        f"- Pages generated: {sum(1 for e in entities.values() if e.mentions)}\n"
    )


def _save_entities_cache(
    wiki_dir: Path,
    entities: dict[str, Entity],
    note_mtimes: dict[str, int],
) -> None:
    """Persist extracted entities + source mtimes for incremental reuse."""
    payload = {
        "mtimes": note_mtimes,
        "entities": {
            key: {
                "name": e.name,
                "category": e.category,
                "mentions": [
                    {
                        "source_title": m.source_title,
                        "source_node_id": m.source_node_id,
                        "description": m.description,
                    }
                    for m in e.mentions
                ],
            }
            for key, e in entities.items()
        },
    }
    cache_file = wiki_dir / ENTITIES_CACHE_FILENAME
    cache_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_entities_cache(
    wiki_dir: Path,
) -> tuple[dict[str, int], dict[str, Entity]] | None:
    """Load cached entities + mtimes. Returns ``None`` on miss."""
    cache_file = wiki_dir / ENTITIES_CACHE_FILENAME
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    mtimes = payload.get("mtimes", {})
    raw_entities = payload.get("entities", {})
    entities: dict[str, Entity] = {}
    for key, data in raw_entities.items():
        entities[key] = Entity(
            name=data["name"],
            category=data.get("category", ""),
            mentions=[
                EntityMention(
                    source_title=m["source_title"],
                    source_node_id=m["source_node_id"],
                    description=m["description"],
                )
                for m in data.get("mentions", [])
            ],
        )
    return mtimes, entities


def _incremental_extract(
    root: TreeNode,
    chat_fn: ChatFn,
    cached_mtimes: dict[str, int],
    cached_entities: dict[str, Entity],
) -> tuple[dict[str, Entity], dict[str, int]]:
    """Re-extract only from notes whose mtime changed since the last run.

    Unchanged notes reuse their cached entity mentions.  Deleted notes
    have their mentions purged.  New/modified notes are re-extracted.
    """
    # Build current mtime map
    current_mtimes: dict[str, int] = {}
    notes_to_extract: list[TreeNode] = []
    for node in root.walk():
        if node.kind != "note" or node.file_path is None:
            continue
        if not node.file_path.exists():
            continue
        node_id = node.node_id
        try:
            mtime = node.file_path.stat().st_mtime_ns
        except OSError:
            continue
        current_mtimes[node_id] = mtime
        if node_id not in cached_mtimes or cached_mtimes[node_id] != mtime:
            notes_to_extract.append(node)

    # Start with cached entities, removing mentions from changed/deleted notes
    changed_ids = {n.node_id for n in notes_to_extract}
    deleted_ids = set(cached_mtimes.keys()) - set(current_mtimes.keys())
    stale_ids = changed_ids | deleted_ids

    entities: dict[str, Entity] = {}
    for key, entity in cached_entities.items():
        kept = [m for m in entity.mentions if m.source_node_id not in stale_ids]
        if kept:
            entities[key] = Entity(
                name=entity.name, category=entity.category, mentions=kept
            )

    # Extract from changed/new notes
    for node in notes_to_extract:
        content = node.file_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
        if not content.strip():
            continue
        prompt = _extract_entities_prompt(node.title, content)
        response = chat_fn(prompt)
        for name, category, description in parse_entities(response):
            norm_key = name.strip().lower()
            if norm_key not in entities:
                entities[norm_key] = Entity(name=name, category=category)
            entities[norm_key].mentions.append(
                EntityMention(
                    source_title=node.title,
                    source_node_id=node.node_id,
                    description=description,
                )
            )

    # Remove entities with no mentions left
    entities = {k: v for k, v in entities.items() if v.mentions}

    return entities, current_mtimes


def compile_wiki(
    root: TreeNode,
    vault_root: Path,
    chat_fn: ChatFn,
    *,
    subfolder: str | None = None,
) -> Path:
    """Run the full two-pass compilation and write to ``{vault}/LLM-Wiki/``.

    Args:
        root: Layer 1 tree root from vault.scan_folder.
        vault_root: Path to the Obsidian vault root.
        chat_fn: LLM callable.
        subfolder: Optional subfolder name (for scoped compilation).

    Returns:
        Path to the generated LLM-Wiki directory.
    """
    wiki_dir = vault_root / WIKI_DIRNAME
    wiki_dir.mkdir(exist_ok=True)

    # Count source notes
    source_count = sum(1 for n in root.walk() if n.kind == "note")

    # Pass 1: extract entities (incremental when cache exists)
    cached = _load_entities_cache(wiki_dir)
    if cached is not None:
        cached_mtimes, cached_entities = cached
        entities, note_mtimes = _incremental_extract(
            root, chat_fn, cached_mtimes, cached_entities
        )
    else:
        entities = extract_entities_from_tree(root, chat_fn)
        note_mtimes = {}
        for node in root.walk():
            if node.kind == "note" and node.file_path and node.file_path.exists():
                with contextlib.suppress(OSError):
                    note_mtimes[node.node_id] = node.file_path.stat().st_mtime_ns

    if not entities:
        # Write a minimal index noting no entities found
        index_path = wiki_dir / INDEX_FILENAME
        index_path.write_text(
            "# LLM-Wiki Index\n\n"
            "> No entities were extracted from the source notes.\n",
            encoding="utf-8",
        )
        return wiki_dir

    # Pass 2: generate pages
    pages = generate_wiki_pages(entities, chat_fn)

    # Write entity pages
    for filename, content in pages.items():
        (wiki_dir / filename).write_text(content, encoding="utf-8")

    # Write index.md
    index_content = generate_index(entities)
    (wiki_dir / INDEX_FILENAME).write_text(index_content, encoding="utf-8")

    # Persist entity cache for incremental reuse
    _save_entities_cache(wiki_dir, entities, note_mtimes)

    # Append to log.md
    log_path = wiki_dir / LOG_FILENAME
    log_entry = generate_log_entry(entities, source_count)
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        log_path.write_text(existing + "\n" + log_entry, encoding="utf-8")
    else:
        log_path.write_text(
            "# LLM-Wiki Compilation Log\n\n" + log_entry,
            encoding="utf-8",
        )

    return wiki_dir
