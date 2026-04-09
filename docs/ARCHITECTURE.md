# pagewiki Architecture

> **TL;DR**: 2-layer tree + PageIndex SDK 위임 + 로컬 Gemma 4. Obsidian 볼트의 짧은 노트와 긴 노트를 각각 다른 전략으로 처리하여 vectorless RAG의 강점을 유지하면서 PKM 환경에 맞게 확장.

## 0. 설계 원칙

1. **짓지 말고 빌려라** — PageIndex SDK가 이미 해결한 문제(트리 생성, multi-hop reasoning, cited_nodes)는 재구현하지 않는다.
2. **로컬 우선** — 모든 LLM 호출은 Ollama(Gemma 4) 경유. 외부 API 의존성 0.
3. **Obsidian 친화** — 볼트 파일을 직접 읽고 쓴다. 별도 DB 불필요.
4. **감사추적** — 모든 쿼리 답변은 cited_nodes + 로그 파일로 기록된다.

## 1. 상위 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                      pagewiki CLI                            │
│         (scan / ask — src/pagewiki/cli.py)                   │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
   ┌──────────────────────┐     ┌──────────────────────────────┐
   │  Layer 1: Vault      │     │  Layer 2: Note Tree          │
   │  vault.py            │     │  pageindex_adapter.py        │
   │                      │     │                              │
   │  폴더 walk           │     │  PageIndex SDK 위임          │
   │  3-tier 분류기       │     │  submit_document → get_tree  │
   │  [[wiki-link]] 추출  │     │                              │
   └──────────┬───────────┘     └──────────┬───────────────────┘
              │                            │
              └───────────┬────────────────┘
                          ▼
           ┌──────────────────────────────┐
           │  Unified TreeNode (tree.py)  │
           │  JSON 트리 저장              │
           └──────────┬───────────────────┘
                      │
                      ▼
           ┌──────────────────────────────┐
           │  Multi-hop Reasoning Loop    │
           │  (v0.1.1에 구현)             │
           └──────────┬───────────────────┘
                      ▼
           ┌──────────────────────────────┐
           │  Ollama / Gemma 4            │
           │  ollama_client.py (LiteLLM)  │
           └──────────┬───────────────────┘
                      ▼
           ┌──────────────────────────────┐
           │  Answer + cited_nodes        │
           │  → .pagewiki-log/*.md        │
           │  (logger.py)                 │
           └──────────────────────────────┘
```

## 2. 2-Layer Tree 구조

### Layer 1: Vault Tree (pagewiki 자체 구현)

폴더 계층과 노트 단위로 Obsidian 볼트 전체를 **얇게** 매핑한다. 이 레이어는 LLM을 거의 호출하지 않는다 (atomic note 요약만).

- **node_id**: vault 루트 기준 상대 경로
- **kind**: `folder` | `note`
- **tier**: `MICRO` | `ATOMIC` | `LONG` (3절 참고)
- **wiki_links**: 노트 본문에서 추출한 `[[target]]` 리스트

### Layer 2: Note Tree (PageIndex SDK 위임)

`tier=LONG`인 노트 각각에 대해 PageIndex의 `submit_document()`를 호출하여 **스마트 목차 트리**를 생성한다. 우리는 아무것도 재구현하지 않는다.

- `max-tokens-per-node=20,000` (Gemma 4 128K에 여유)
- Markdown `#`/`##`/`###` 자동 인식 (Obsidian 노트 직접 소비)
- 동적 컨텍스트 윈도우 (섹션 크기에 맞춰 가변)

### 왜 두 레이어인가

Obsidian 볼트는 **짧은 atomic 노트**와 **긴 논문/리서치 노트**가 혼재한다. 단일 전략으로는:

- 전부 PageIndex에 넘기면 → atomic 노트마다 LLM 호출 → 비용 폭발
- 전부 Layer 1 leaf로 두면 → 긴 노트 내부 탐색 불가
- **해결**: 크기로 분기. PageIndex는 꼭 필요한 long note에만 적용.

## 3. 3-Tier 노트 분류

| Tier | 토큰 범위 | 처리 전략 | Layer 1 필드 |
|---|---|---|---|
| **MICRO** | `< 500` | 제목만 사용. 요약 생성 안 함 | `summary=""` |
| **ATOMIC** | `500 ~ 3000` | Gemma 1-line 요약 1회 생성 | `summary="..."`, LLM 1 call |
| **LONG** | `> 3000` | PageIndex 서브트리 생성 | `children=[PageIndex 노드]` |

**임계값 근거**:
- `500 토큰`: 한글 약 300자. 이 이하는 제목으로 충분히 식별 가능
- `3000 토큰`: PageIndex `max-tokens-per-node=20,000` 기본값의 15%. 이 이상은 최소 1개의 의미 있는 서브트리(헤딩 분할)가 나올 확률이 높음

## 4. 쿼리 루프 (v0.1.1)

PageIndex 공식 Phase 2 Multi-hop Reasoning을 Layer 1 진입점만 추가해서 그대로 준용한다.

```
1. ToC Review
   - Layer 1 루트의 상위 N개 노드 요약을 Gemma에 제시
   - "사용자 질문에 가장 관련된 폴더/노트는?" 선택

2. Select Node
   - 선택된 노드 타입에 따라 분기:
     - folder → 재귀 (1로 돌아가 해당 서브트리에서 다시 Review)
     - note (MICRO/ATOMIC) → 노트 전체 내용을 컨텍스트에 로드
     - note (LONG) → Layer 2 (PageIndex sub-tree) Multi-hop 호출

3. Extract & Evaluate
   - "이 정보로 답변이 충분한가?" (Gemma 자가 평가)
   - wiki-link cross-reference 확인:
     - 선택된 노트가 [[OtherNote]]를 참조 → OtherNote 탐색 후보로 추가

4. Conditional Branch
   - No → 다른 노드 추가 탐색 (1 ~ 3 반복)
   - Yes → 최종 답변 생성 + cited_nodes 수집

5. Persist
   - answer + cited_nodes → .pagewiki-log/{timestamp}.md
```

## 5. 파일 구조

```
pagewiki/
├── README.md
├── LICENSE (MIT)
├── pyproject.toml
├── docs/
│   └── ARCHITECTURE.md  ← 이 파일
├── src/pagewiki/
│   ├── __init__.py
│   ├── cli.py                # Click CLI
│   ├── tree.py               # TreeNode pydantic 모델
│   ├── vault.py              # Layer 1 스캐너 + 분류기
│   ├── pageindex_adapter.py  # Layer 2 어댑터
│   ├── ollama_client.py      # LiteLLM + Ollama 래퍼
│   └── logger.py             # QueryRecord → .pagewiki-log
└── tests/
    └── test_vault.py         # Layer 1 단위 테스트
```

## 6. 의존성

| 패키지 | 역할 | 라이선스 |
|---|---|---|
| `litellm` | LLM 라우터 (Ollama 포함) | MIT |
| `pydantic` | TreeNode 검증 | MIT |
| `click` | CLI | BSD-3 |
| `rich` | 터미널 출력 | MIT |

**Vendored (직접 번들):**
| 모듈 | 역할 | 라이선스 |
|---|---|---|
| `_vendor/pageindex/` | Layer 2 마크다운 트리 빌더 (VectifyAI/PageIndex `@f2dcffc`) | MIT |

v0.1.2부터 PageIndex는 pip 의존성이 아니라 `src/pagewiki/_vendor/pageindex/`에
최소 범위(마크다운 전용, LLM 호출 제거)로 번들된다. 이유:
1. 공식 `pageindex` 패키지는 `PyPDF2`/`pymupdf`/`litellm`을 강제로 끌어오며,
   pagewiki의 Obsidian-markdown-only 파이프라인에는 PDF 의존성이 불필요하다.
2. 업스트림의 LLM 호출 경로(`md_to_tree`)는 `.env` 기반 OpenAI 키를 가정한다.
   pagewiki는 `chat_fn` DI로 Ollama 경유 Gemma 4만 사용하므로 자체 어댑터에서
   요약을 생성한다.

**런타임 외부 의존**:
- Ollama (`brew install ollama`)
- `gemma4:26b` (64GB RAM Mac 기준 기본), `gemma4:e4b` (16GB 폴백)

**Layer 2 캐시**: `{vault}/.pagewiki-cache/trees/{sha1}.json`. 캐시 키는
`(abs_path, mtime_ns, model_id, adapter_version)` 4요소이며 하나라도
바뀌면 재빌드된다. `rm -rf .pagewiki-cache` 안전 (idempotent).

## 7. 로드맵

| 버전 | 범위 |
|---|---|
| v0.1 | 스캐폴딩, Layer 1 스캐너, 3-tier 분류, CLI 구조 |
| v0.1.1 | Multi-hop reasoning 루프 구현, ask 명령 완성 |
| **v0.1.2** (현재) | PageIndex SDK 실제 통합 (vendored), Layer 2 섹션 트리, 디스크 캐시, 섹션 단위 retrieval descend |
| v0.2 | `[[wiki-link]]` cross-reference 탐색 |
| v0.3 | Karpathy LLM-Wiki compiler (entity 추출 → `LLM-Wiki/` 폴더) |
| v0.4 | 증분 재인덱싱 + mtime 기반 watcher |
| v0.5 | Obsidian 플러그인 UI |

## 8. 명시적 비목표 (Non-Goals)

- ❌ 실시간 채팅 UI (쿼리당 60~300초 레이턴시)
- ❌ 수백만 문서 스케일 (볼트 수천~수만 노트 대상)
- ❌ 벡터 DB 하이브리드 (순수 vectorless 유지)
- ❌ 클라우드 LLM 지원 (로컬 프라이버시 우선)
