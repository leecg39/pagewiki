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
│     (scan / ask / chat / compile / watch — cli.py)           │
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
- **tags** / **date** / **aliases**: YAML frontmatter에서 파싱 (v0.6)

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

## 4. 쿼리 루프 (v0.1.1, v0.2 교차참조 확장)

PageIndex 공식 Phase 2 Multi-hop Reasoning을 Layer 1 진입점만 추가해서 그대로 준용한다.

```
1. ToC Review
   - Layer 1 루트의 상위 N개 노드 요약을 Gemma에 제시
   - "사용자 질문에 가장 관련된 폴더/노트는?" 선택
   - (v0.2) 교차참조 후보가 있으면 [교차참조: source → [[target]]] 태그와 함께 표시

2. Select Node
   - 선택된 노드 타입에 따라 분기:
     - folder → 재귀 (1로 돌아가 해당 서브트리에서 다시 Review)
     - note (MICRO/ATOMIC) → 노트 전체 내용을 컨텍스트에 로드
     - note (LONG) → Layer 2 (PageIndex sub-tree) Multi-hop 호출
     - (v0.2) cross-ref 후보 선택 시 → cursor를 root로 리셋 후 해당 노트로 이동

3. Extract & Evaluate
   - "이 정보로 답변이 충분한가?" (Gemma 자가 평가)
   - (v0.2) wiki-link cross-reference 자동 수집:
     - 선택된 노트의 outgoing [[wiki-link]]를 LinkIndex에서 조회
     - 미방문 대상 노트를 cross_ref_pool에 추가
     - 다음 iteration의 후보 목록에 [교차참조] 태그와 함께 병합
     - section-anchor 링크([[Paper#Methods]])는 enclosing note로 promote

4. Conditional Branch
   - No → 다른 노드 + cross-ref 후보 추가 탐색 (1 ~ 3 반복)
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
│   ├── cli.py                # Click CLI (scan/ask/chat/compile/watch/vaults)
│   ├── tree.py               # TreeNode pydantic 모델
│   ├── vault.py              # Layer 1 스캐너 + 분류기 + filter_tree
│   ├── pageindex_adapter.py  # Layer 2 어댑터
│   ├── retrieval.py          # Multi-hop reasoning 루프 (on_event 스트리밍)
│   ├── prompts.py            # 프롬프트 템플릿 (select/evaluate/final/chat)
│   ├── frontmatter.py        # YAML frontmatter 파서 (v0.6)
│   ├── cache.py              # TreeCache + SummaryCache (v0.6)
│   ├── wiki_links.py         # [[wiki-link]] 인덱스 + 교차참조
│   ├── compile.py            # LLM-Wiki 컴파일러 (entity → wiki)
│   ├── watcher.py            # mtime 기반 파일 변경 감지
│   ├── obsidian_config.py    # 볼트 자동 발견 (notesmd-cli / obsidian.json)
│   ├── ollama_client.py      # LiteLLM + Ollama 래퍼
│   ├── logger.py             # QueryRecord → .pagewiki-log
│   ├── server.py             # FastAPI HTTP 서버 (v0.7, optional)
│   ├── usage.py              # 토큰 사용량 추적 (v0.8)
│   ├── ranking.py            # BM25-style 후보 사전 랭킹 (v0.8)
│   └── _vendor/pageindex/    # 번들된 PageIndex SDK (MIT)
├── obsidian-plugin/
│   ├── main.ts               # Obsidian 플러그인 (Scan/Ask/Chat/Compile/Watch)
│   ├── manifest.json
│   └── styles.css
└── tests/                    # 185+ 테스트
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

**캐시** (`{vault}/.pagewiki-cache/`):
- `trees/{sha1}.json` — Layer 2 서브트리. 키: `(abs_path, mtime_ns, model_id, adapter_version)`
- `summaries/{sha1}.json` — ATOMIC 노트 요약 (v0.6). 키: `(abs_path, mtime_ns, model_id)`
- 어느 쪽이든 키 불일치 시 자동 재빌드. `rm -rf .pagewiki-cache` 안전 (idempotent).

## 7. 로드맵

| 버전 | 범위 |
|---|---|
| v0.1 | 스캐폴딩, Layer 1 스캐너, 3-tier 분류, CLI 구조 |
| v0.1.1 | Multi-hop reasoning 루프 구현, ask 명령 완성 |
| v0.1.2 | PageIndex SDK 실제 통합 (vendored), Layer 2 섹션 트리, 디스크 캐시, 섹션 단위 retrieval descend |
| v0.1.3 | h1-title flatten + `(intro)` 보존, vault-relative section id, local validation scripts |
| v0.1.4 | `[[wiki-link]]` resolution **index** + `scan --show-graph` (v0.2 Phase 1, PR #2) |
| v0.1.5 | notesmd-cli 통합 — `--vault` auto-discovery, `pagewiki vaults` 서브커맨드, `ask` 출력에 `notesmd-cli open` 힌트 |
| v0.2 | `[[wiki-link]]` retrieval traversal — 노트 평가 후 outgoing wiki-link 대상을 교차참조 후보로 자동 추가, 프롬프트에 `[교차참조]` 태그 표시, transitive chain following |
| v0.3 | Karpathy LLM-Wiki compiler — `pagewiki compile` 서브커맨드, 2-pass 파이프라인 (entity 추출 → 위키 페이지 생성), `{vault}/LLM-Wiki/` 출력 |
| v0.4 | 증분 재인덱싱 + mtime watcher — `pagewiki watch`로 vault 파일 변경 실시간 감지 |
| v0.5 | Obsidian 플러그인 UI — Command Palette에서 Scan/Ask/Compile/Watch 실행, Settings 탭, 결과 모달 |
| v0.6 | 대화형 모드 + 캐시 + 필터 + 스트리밍 — 아래 §7.1 상세 |
| v0.7 | 병렬 LLM + 멀티쿼리 분해 + 멀티 vault + API 서버 — 아래 §7.2 상세 |
| **v0.8** (현재) | **토큰 사용량 추적 + 파싱 재시도 + BM25 사전 랭킹** — 아래 §7.3 상세 |

### 7.1 v0.6 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| `pagewiki chat` | cli.py, prompts.py, retrieval.py | 대화형 REPL. 후속 질문을 `rewrite_query_with_context()`로 독립 질문으로 재작성, `final_answer_with_history_prompt()`로 이전 대화 맥락 반영 |
| Atomic summary cache | cache.py `SummaryCache` | ATOMIC 노트 요약을 `.pagewiki-cache/summaries/`에 디스크 캐시. TreeCache와 동일한 `(abs_path, mtime_ns, model_id)` 무효화 |
| Frontmatter 필터 | frontmatter.py, vault.py `filter_tree()` | YAML frontmatter(tags, date, aliases) 파싱 → TreeNode 필드. `--tag`/`--after`/`--before` CLI 옵션으로 트리 사전 가지치기 |
| 실시간 스트리밍 | retrieval.py `on_event`, cli.py | `run_retrieval()`에 `EventCallback` 추가. SELECT/EVAL/XREF/DONE 단계를 실시간 표시 |

### 7.2 v0.7 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| 병렬 LLM 호출 | vault.py, compile.py | `summarize_atomic_notes()`와 `extract_entities_from_tree()`/`generate_wiki_pages()`를 `ThreadPoolExecutor`로 병렬화. `--max-workers`로 동시 worker 수 조정 (기본 4). Ollama는 동시 요청 지원. |
| 멀티쿼리 분해 | retrieval.py `run_decomposed_retrieval`, prompts.py | `--decompose` 플래그 활성 시 LLM으로 복합 질문을 서브쿼리로 분해 → 각각 retrieve → synthesize. `SINGLE` 응답 시 단일 쿼리 경로로 fall-through. |
| 멀티 vault 검색 | vault.py `scan_multi_vault()`, cli.py `--extra-vault` | 여러 볼트를 가상 루트 아래 병합. 각 노드의 `node_id`는 `<vault_name>::`로 네임스페이스 prefix. 캐시는 abs_path 기반이라 볼트 간 충돌 없음. |
| HTTP API 서버 | server.py, cli.py `serve` | FastAPI 기반 `pagewiki serve`. 트리/캐시 startup warmup → `/health`, `/scan`, `/ask`, `/chat`(세션 관리) 엔드포인트. FastAPI는 `pip install 'pagewiki[server]'`로 선택적 설치. |

**API 서버 엔드포인트**:

```
GET    /health          # liveness + note count
POST   /scan            # refresh in-memory tree
POST   /ask             # single-shot query (decompose/filter 지원)
POST   /chat            # session-based multi-turn (session_id 자동 발급)
DELETE /chat/{sid}      # 세션 삭제
```

세션은 in-process dict로 관리, 1시간 비활성 시 자동 만료.

### 7.3 v0.8 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| 토큰 사용량 추적 | usage.py `UsageTracker` | 스레드 안전 counter. `chat_fn` 래퍼가 매 호출을 phase(summarize/select/evaluate/final/...)별로 기록. `ask --usage`로 terminal에 Rich 테이블 출력. |
| 파싱 재시도 | prompts.py `build_retry_prompt`, retrieval.py | SELECT 응답이 malformed일 때 stricter format reminder를 append해 한 번 더 시도. 기존에는 loop가 즉시 abort됐지만 이제 자동 복구. |
| BM25 사전 랭킹 | ranking.py | Zero-LLM 비용의 candidate 사전 정렬. 토큰 overlap 기반 BM25-style 스코어 + 짧은 후보 보너스. 정렬된 목록을 SELECT 프롬프트에 전달 → LLM이 적은 iteration으로 정답 도달. Korean/English 혼용 vault 지원. |
| 서버 엔드포인트 테스트 | tests/test_server.py | FastAPI `TestClient` 기반 11개 테스트. /health, /scan, /ask, /chat (세션 생성/누적/삭제/만료) 검증. FastAPI 미설치 시 `pytest.importorskip`로 clean skip. |

### 7.4 v0.9+ 향후 계획

- Pydantic JSON-mode 출력 (LLM이 JSON으로 응답 → strict 파싱)
- 컨텍스트 reuse: 반복 iteration 시 ToC 델타만 재전송
- Usage 예산 제한 (`--max-tokens`로 쿼리당 예산 하드 캡)
- Re-ranking: gathered notes를 LLM에 다시 물어 최종 인용 우선순위 정렬

## 8. 명시적 비목표 (Non-Goals)

- ❌ 수백만 문서 스케일 (볼트 수천~수만 노트 대상)
- ❌ 벡터 DB 하이브리드 (순수 vectorless 유지)
- ❌ 클라우드 LLM 지원 (로컬 프라이버시 우선)
