"""YAML frontmatter parser for Obsidian notes.

Obsidian notes commonly start with a YAML block delimited by ``---``::

    ---
    tags: [research, ml]
    date: 2024-11-15
    aliases: [Transformer Paper]
    ---

This module extracts structured metadata from that block without
pulling in a full YAML library (PyYAML would be a new dependency).
The parser handles the subset of YAML that Obsidian actually produces:
simple scalars, inline lists ``[a, b]``, and dash-prefixed lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Match the opening/closing ``---`` fence.
_FENCE_RE = re.compile(r"^---\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Frontmatter:
    """Parsed frontmatter metadata."""

    tags: list[str] = field(default_factory=list)
    date: str | None = None
    aliases: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)


def _parse_yaml_list(value: str) -> list[str]:
    """Parse an inline YAML list like ``[a, b, c]`` or a bare scalar."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
    # Single value — return as one-element list.
    return [value.strip("\"'")] if value else []


def parse_frontmatter(text: str) -> Frontmatter:
    """Extract frontmatter from a markdown string.

    Returns ``Frontmatter()`` (all defaults) if no valid frontmatter
    block is found — the caller never needs to handle ``None``.
    """
    # Frontmatter must start at the very beginning of the file.
    if not text.startswith("---"):
        return Frontmatter()

    matches = list(_FENCE_RE.finditer(text))
    if len(matches) < 2:
        return Frontmatter()

    # The YAML body sits between the first and second ``---`` lines.
    start = matches[0].end()
    end = matches[1].start()
    yaml_block = text[start:end]

    raw: dict[str, str] = {}
    # Separate storage for keys whose value came from dash-prefixed lists,
    # so we can return them as proper Python lists later without re-parsing.
    list_keys: dict[str, list[str]] = {}
    current_key: str | None = None
    list_values: list[str] = []

    for line in yaml_block.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Dash-prefixed list continuation (e.g. ``  - item``).
        if stripped.startswith("- ") and current_key is not None:
            list_values.append(stripped[2:].strip().strip("\"'"))
            continue

        # Flush any accumulated list values.
        if current_key is not None and list_values:
            list_keys[current_key] = list(list_values)
            raw[current_key] = "[" + ", ".join(list_values) + "]"
            list_values = []

        # ``key: value`` line.
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip().lower()
            val = val.strip()
            current_key = key
            if val:
                raw[key] = val
            else:
                list_values = []
        else:
            current_key = None

    # Flush trailing list.
    if current_key is not None and list_values:
        list_keys[current_key] = list(list_values)
        raw[current_key] = "[" + ", ".join(list_values) + "]"

    # Extract well-known fields.
    tags: list[str] = []
    raw_tags = raw.get("tags", "")
    if raw_tags:
        tags = _parse_yaml_list(raw_tags)
    # Obsidian also supports ``tag:`` (singular).
    raw_tag = raw.get("tag", "")
    if raw_tag and not tags:
        tags = _parse_yaml_list(raw_tag)

    date = raw.get("date")

    aliases: list[str] = []
    raw_aliases = raw.get("aliases", "")
    if raw_aliases:
        aliases = _parse_yaml_list(raw_aliases)

    return Frontmatter(tags=tags, date=date, aliases=aliases, raw=raw)
