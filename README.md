# pagewiki

> Obsidian vault를 PageIndex 트리로 인덱싱하고, 로컬 Gemma 4로 reasoning-based 질의를 수행하는 vectorless RAG 도구.

**핵심 철학**: `From Chunk-and-Embed... to Reason-and-Navigate.`

## 왜 pagewiki인가

기존 Obsidian RAG 플러그인은 대부분 **벡터 임베딩 + 코사인 유사도** 방식입니다. 이 방식은:

- 📉 문서의 논리적 구조(헤딩, 섹션, 상호참조)를 파괴합니다
- 🔍 "비슷한 단어"는 찾지만 "논리적으로 관련된" 섹션을 놓칩니다
- 🕳️ 왜 그 노트가 선택되었는지 추적할 수 없습니다 (블랙박스)

pagewiki는 [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)의 **Reasoning-based Vectorless RAG**를 Obsidian 볼트에 적용합니다:

- ✅ 벡터 DB 없음, 임베딩 없음, 청킹 없음
- ✅ LLM이 직접 "목차 → 섹션 → 세부" 순서로 탐색
- ✅ cited_nodes로 100% 추적 가능한 답변 (Audit Trail)
- ✅ 로컬 Gemma 4 + Ollama → 완전한 프라이버시

## 아키텍처

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Vault Tree  (pagewiki 자체 구현)            │
│   • 폴더 + 노트 제목 + 1-line summary               │
│   • 노트 = leaf (atomic) OR sub-tree root (long)   │
└────────────────┬────────────────────────────────────┘
                 │ (long note에 도달하면)
                 ▼
┌─────────────────────────────────────────────────────┐
│ Layer 2: Note Tree  (PageIndex SDK)                │
│   • markdown #/##/### 자동 인식                     │
│   • max-tokens-per-node=20,000                     │
│   • 동적 컨텍스트 윈도우                             │
└─────────────────────────────────────────────────────┘
```

### 노트 3-tier 분류

| 크기 | 분류 | 처리 |
|---|---|---|
| `< 500 토큰` | micro-leaf | 제목만 사용 |
| `500~3000 토큰` | atomic leaf | Gemma 1-line 요약 |
| `> 3000 토큰` | long note | PageIndex 서브트리 생성 |

상세 설계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 참고.

## 설치

```bash
# 1. Ollama + Gemma 4 설치
brew install ollama
ollama pull gemma4:26b  # 24GB+ VRAM 권장. 16GB면 gemma4:e4b

# 2. pagewiki 설치 (개발 모드)
git clone https://github.com/leecg39/PageIndex.git pagewiki
cd pagewiki
pip install -e .
```

## 사용법

```bash
# 단일 폴더에 대해 질의 (MVP v0.1)
pagewiki ask "2024년 3분기 매출 관련 리서치 요약" \
  --vault ~/Documents/Obsidian \
  --folder Research \
  --model gemma4:26b
```

출력:
- 답변 (stdout)
- 인용된 노드 경로 (page_range, section path)
- `{vault}/Research/.pagewiki-log/{timestamp}.md`에 자동 기록

## 로드맵

- **v0.1** (현재): Literature 폴더 Deep-Research 모드 (단일 폴더)
- **v0.2**: 복수 폴더 + `[[wiki-link]]` cross-reference
- **v0.3**: Karpathy LLM-Wiki compiler 통합
- **v0.4**: 증분 재인덱싱 + 파일 watcher
- **v0.5**: Obsidian 플러그인 UI

## 트레이드오프 (솔직하게)

- ⏱️ **레이턴시**: 쿼리당 60~300초 (로컬 Gemma 4 기준). 실시간 채팅 아님
- 💭 **연산 비용**: 임베딩은 0, 대신 LLM 추론 토큰 소모 큼 (로컬이라 전기요금)
- 📊 **확장성**: 단일 볼트/폴더에 최적. 수백만 문서엔 Vector DB가 더 유리

## 라이선스

MIT. [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)도 MIT.

## Credits

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — Reasoning-based vectorless RAG 엔진
- [Andrej Karpathy's LLM-Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — 파일시스템 기반 지식 컴파일 철학
- [Google Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) — 로컬 추론 엔진
