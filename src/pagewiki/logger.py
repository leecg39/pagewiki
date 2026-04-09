"""Query log writer — every query's answer + cited_nodes is persisted as
a markdown file inside `{vault}/{folder}/.pagewiki-log/`.

This log folder is intentionally inside the vault so Obsidian indexes it
automatically — users can browse past research via the graph view.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LOG_DIRNAME = ".pagewiki-log"


@dataclass(frozen=True)
class QueryRecord:
    query: str
    answer: str
    cited_nodes: list[str]
    model: str
    elapsed_seconds: float


def write_log(folder: Path, record: QueryRecord) -> Path:
    """Write a markdown log file and return its path."""
    log_dir = folder / LOG_DIRNAME
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{ts}.md"
    path = log_dir / filename

    cited_md = "\n".join(f"- {c}" for c in record.cited_nodes) or "_(no citations)_"

    content = f"""---
query: {record.query!r}
model: {record.model}
elapsed: {record.elapsed_seconds:.1f}s
timestamp: {ts}
---

# Q: {record.query}

## Answer

{record.answer}

## Cited Nodes

{cited_md}
"""
    path.write_text(content, encoding="utf-8")
    return path
