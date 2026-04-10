"""Tests for v0.3 LLM-Wiki compiler.

All LLM calls are mocked via a scripted chat_fn. Tests verify:
  - Entity extraction prompt parsing
  - Wiki page generation
  - Index generation
  - Full compile pipeline (end-to-end)
  - Idempotency (re-running overwrites cleanly)
  - Edge cases (no entities, empty vault)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pagewiki.compile import (
    Entity,
    EntityMention,
    compile_wiki,
    extract_entities_from_tree,
    generate_index,
    generate_wiki_pages,
    parse_entities,
)
from pagewiki.vault import scan_folder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_chat(responses: list[str]) -> Callable[[str], str]:
    """Return a chat_fn that cycles through responses."""
    idx = {"i": 0}

    def _call(prompt: str) -> str:
        if idx["i"] >= len(responses):
            return "NONE"
        reply = responses[idx["i"]]
        idx["i"] += 1
        return reply

    return _call


@pytest.fixture
def research_vault(tmp_path: Path) -> Path:
    """Small vault with 2 notes containing extractable entities."""
    vault = tmp_path / "vault"
    research = vault / "Research"
    research.mkdir(parents=True)

    (research / "rag.md").write_text(
        "# RAG Overview\n\n"
        "Retrieval-Augmented Generation (RAG)는 검색 기반 생성 기법이다. "
        "Lewis et al. (2020)이 제안했으며 FAISS를 벡터 검색에 사용한다. " * 30,
        encoding="utf-8",
    )
    (research / "pageindex.md").write_text(
        "# PageIndex\n\n"
        "VectifyAI가 개발한 vectorless RAG 엔진이다. "
        "트리 기반 추론으로 RAG의 한계를 극복한다. " * 30,
        encoding="utf-8",
    )
    return vault


# ─────────────────────────────────────────────────────────────────────────────
# parse_entities
# ─────────────────────────────────────────────────────────────────────────────


class TestParseEntities:
    def test_basic_parsing(self) -> None:
        response = (
            "ENTITY: RAG | concept | 검색 기반 생성 기법\n"
            "ENTITY: FAISS | technology | 벡터 검색 라이브러리\n"
        )
        result = parse_entities(response)
        assert len(result) == 2
        assert result[0] == ("RAG", "concept", "검색 기반 생성 기법")
        assert result[1] == ("FAISS", "technology", "벡터 검색 라이브러리")

    def test_none_response(self) -> None:
        assert parse_entities("NONE") == []
        assert parse_entities("  NONE  ") == []

    def test_mixed_with_noise(self) -> None:
        response = (
            "Here are the entities:\n"
            "ENTITY: RAG | concept | 검색 기반 생성\n"
            "Some extra text\n"
            "ENTITY: Lewis | person | RAG 제안자\n"
        )
        result = parse_entities(response)
        assert len(result) == 2

    def test_empty_response(self) -> None:
        assert parse_entities("") == []

    def test_whitespace_handling(self) -> None:
        response = "ENTITY:  RAG  |  concept  |  검색 기반 생성  "
        result = parse_entities(response)
        assert len(result) == 1
        assert result[0] == ("RAG", "concept", "검색 기반 생성")


# ─────────────────────────────────────────────────────────────────────────────
# extract_entities_from_tree
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractEntities:
    def test_extracts_from_multiple_notes(
        self, research_vault: Path
    ) -> None:
        root = scan_folder(research_vault, "Research")
        chat = _make_chat([
            # Response for first note (alphabetical: pageindex)
            "ENTITY: PageIndex | technology | vectorless RAG 엔진\n"
            "ENTITY: VectifyAI | organization | PageIndex 개발사\n"
            "ENTITY: RAG | concept | 검색 기반 생성\n",
            # Response for second note (rag)
            "ENTITY: RAG | concept | 검색 기반 생성 기법\n"
            "ENTITY: FAISS | technology | 벡터 검색\n"
            "ENTITY: Lewis | person | RAG 제안자\n",
        ])

        entities = extract_entities_from_tree(root, chat)

        assert "rag" in entities  # normalized key
        assert "faiss" in entities
        assert "pageindex" in entities
        # RAG should have 2 mentions (from both notes)
        assert len(entities["rag"].mentions) == 2

    def test_no_entities_returns_empty(self, research_vault: Path) -> None:
        root = scan_folder(research_vault, "Research")
        chat = _make_chat(["NONE", "NONE"])

        entities = extract_entities_from_tree(root, chat)
        assert len(entities) == 0


# ─────────────────────────────────────────────────────────────────────────────
# generate_wiki_pages
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateWikiPages:
    def test_generates_page_per_entity(self) -> None:
        entities = {
            "rag": Entity(
                name="RAG",
                category="concept",
                mentions=[
                    EntityMention("rag", "Research/rag.md", "검색 기반 생성"),
                ],
            ),
            "faiss": Entity(
                name="FAISS",
                category="technology",
                mentions=[
                    EntityMention("rag", "Research/rag.md", "벡터 검색"),
                ],
            ),
        }
        chat = _make_chat([
            "# RAG\n\nRAG는 검색 기반 생성 기법입니다.\n\n## 출처\n- rag.md",
            "# FAISS\n\nFAISS는 벡터 검색 라이브러리입니다.\n\n## 출처\n- rag.md",
        ])

        pages = generate_wiki_pages(entities, chat)

        assert len(pages) == 2
        assert "RAG.md" in pages
        assert "FAISS.md" in pages
        assert "# RAG" in pages["RAG.md"]


# ─────────────────────────────────────────────────────────────────────────────
# generate_index
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateIndex:
    def test_groups_by_category(self) -> None:
        entities = {
            "rag": Entity(
                name="RAG",
                category="concept",
                mentions=[
                    EntityMention("rag", "Research/rag.md", "검색 기반 생성"),
                ],
            ),
            "lewis": Entity(
                name="Lewis",
                category="person",
                mentions=[
                    EntityMention("rag", "Research/rag.md", "RAG 제안자"),
                ],
            ),
        }

        index = generate_index(entities)

        assert "## 개념 (Concepts)" in index
        assert "## 인물 (People)" in index
        assert "[[RAG]]" in index
        assert "[[Lewis]]" in index

    def test_empty_entities(self) -> None:
        index = generate_index({})
        assert "LLM-Wiki Index" in index


# ─────────────────────────────────────────────────────────────────────────────
# compile_wiki (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────


class TestCompileWiki:
    def test_full_pipeline_creates_wiki_dir(
        self, research_vault: Path
    ) -> None:
        root = scan_folder(research_vault, "Research")
        chat = _make_chat([
            # Extract from pageindex.md
            "ENTITY: PageIndex | technology | vectorless RAG\n",
            # Extract from rag.md
            "ENTITY: RAG | concept | 검색 기반 생성\n",
            # Generate page for PageIndex
            "# PageIndex\n\nvectorless RAG 엔진.\n\n## 출처\n- pageindex.md",
            # Generate page for RAG
            "# RAG\n\n검색 기반 생성.\n\n## 출처\n- rag.md",
        ])

        wiki_dir = compile_wiki(root, research_vault, chat)

        assert wiki_dir.exists()
        assert (wiki_dir / "index.md").exists()
        assert (wiki_dir / "log.md").exists()

        index_text = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "LLM-Wiki Index" in index_text

        log_text = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "COMPILE" in log_text
        assert "Entities extracted: 2" in log_text

    def test_no_entities_writes_minimal_index(
        self, research_vault: Path
    ) -> None:
        root = scan_folder(research_vault, "Research")
        chat = _make_chat(["NONE", "NONE"])

        wiki_dir = compile_wiki(root, research_vault, chat)

        assert (wiki_dir / "index.md").exists()
        text = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "No entities" in text

    def test_idempotent_recompile(self, research_vault: Path) -> None:
        """Re-running compile should overwrite cleanly."""
        root = scan_folder(research_vault, "Research")
        responses = [
            "ENTITY: RAG | concept | 검색 기반 생성\n",
            "NONE",
            "# RAG\n\n검색 기반 생성.\n\n## 출처\n- rag.md",
        ]

        # First run
        compile_wiki(root, research_vault, _make_chat(list(responses)))
        # Second run
        wiki_dir = compile_wiki(root, research_vault, _make_chat(list(responses)))

        # Should still work; log.md should have 2 entries
        log_text = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert log_text.count("COMPILE") == 2

    def test_empty_vault(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        (vault / "Empty").mkdir(parents=True)
        root = scan_folder(vault, "Empty")
        chat = _make_chat([])

        wiki_dir = compile_wiki(root, vault, chat)

        assert (wiki_dir / "index.md").exists()
        text = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "No entities" in text
