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
│   ├── retrieval/            # Multi-hop reasoning (v0.13 split)
│   │   ├── __init__.py       #   public API re-exports
│   │   ├── types.py          #   TraceStep, RetrievalResult, ChatFn
│   │   ├── helpers.py        #   tree helpers + _load_note_content
│   │   ├── core.py           #   run_retrieval (main loop)
│   │   ├── decompose.py      #   run_decomposed_retrieval
│   │   └── cross_vault.py    #   run_cross_vault_retrieval
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
│   ├── usage_store.py        # SQLite usage 영속화 + daily rollup (v0.10, retention v0.14)
│   ├── ranking.py            # BM25-style 후보 사전 랭킹 (v0.8)
│   ├── webui.py              # Embedded self-contained Web UI (v0.14)
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
| v0.8 | 토큰 사용량 추적 + 파싱 재시도 + BM25 사전 랭킹 — 아래 §7.3 상세 |
| v0.9 | 토큰 예산 + chat usage + /usage endpoint + cited 재정렬 — 아래 §7.4 상세 |
| v0.10 | JSON-mode + SQLite usage + SSE 스트리밍 + context reuse — 아래 §7.5 상세 |
| v0.11 | /chat/stream + usage 이벤트 + per-vault 캐시 + usage-report CLI — 아래 §7.6 상세 |
| v0.12 | WebSocket + daily 롤업 + cross-vault + 플러그인 server-mode — 아래 §7.7 상세 |
| v0.13 | 폴리싱 + 리팩토링 — chat flags, real tokens, plugin Cancel, cross-vault×decompose, CSV/JSON, retrieval split — 아래 §7.8 상세 |
| v0.14 | DB 정리 + `/usage/history` + Web UI + 예산 분배 + 프롬프트 캐싱 — 아래 §7.9 상세 |
| v0.15 | Cross-vault parallel + cache hit rate + history stream + WS ext + sparkline — 아래 §7.10 상세 |
| v0.16 | Parallel budget waves + inferred cache savings + history UI + WS prompt-cache + allow_partial — 아래 §7.11 상세 |
| v0.17 | /usage cache stats + background retention + retry_failed + WS cancel metadata + plugin history modal |
| v0.18 | CHANGELOG + better error messages + Web UI keyboard shortcuts |
| **v1.0.0** (현재) | **Stable release — 공개 API 고정, 1.x 라인 내에서는 additive-only 변경** |

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

### 7.4 v0.9 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| 토큰 예산 한도 | retrieval.py `run_retrieval`, `run_decomposed_retrieval` | 새 `max_tokens` + `tracker` 매개변수. 매 iteration 시작 전에 `tracker.total_tokens >= max_tokens`를 검사하고 초과 시 루프를 clean abort. 초과 상태에서 answer 합성 비용까지 아끼기 위해 final answer도 raw 근거 concat으로 fall back. CLI `ask --max-tokens N`, `chat --max-tokens N` (per-turn 예산). |
| Chat 세션 usage | cli.py `chat` | `UsageTracker`가 세션 전체에 걸쳐 cumulative 집계. 매 turn마다 pre-turn snapshot과 delta를 비교해 per-turn 토큰 + call 수를 표시. 종료 시 phase별 Rich 테이블 출력. `--max-tokens`는 per-turn 예산 (세션 누적이 아님). |
| Server `/usage` | server.py `GET /usage`, `POST /usage/reset` | `ServerState.tracker`가 서버 lifetime 동안 모든 LLM 호출 누적. FastAPI 엔드포인트로 cumulative counts + phase 분해 반환. reset 엔드포인트로 모니터링 윈도우 초기화 가능. |
| Cited note 재정렬 | retrieval.py + ranking.py | `gathered` 목록을 `rank_candidates`로 쿼리 관련도 순으로 재정렬해 `cited_nodes`가 discovery order가 아닌 relevance order로 반환. Zero-LLM 비용 (기존 BM25 스코어러 재사용). |

### 7.5 v0.10 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| JSON-mode | prompts.py `select_node_prompt_json` / `evaluate_prompt_json` + `parse_*_response_json` | LLM이 `{"action": "SELECT", "node_id": "..."}` 형식의 JSON 객체로 응답. 파서는 markdown code fence + 앞뒤 noise tolerant. `run_retrieval(json_mode=True)` 혹은 `ask --json-mode`. JSON retry가 실패하면 v0.8 text parser로 자동 fall-back. |
| SQLite usage 영속화 | usage_store.py `UsageStore` | SQLite `usage_events` 테이블 (timestamp, phase, prompt, completion, elapsed). WAL 모드 + phase/timestamp 인덱스. `query_summary(since, until)` for 집계. `pagewiki serve --usage-db PATH`로 활성화. `/usage` 엔드포인트가 `persistent_total_*` 필드 추가. |
| SSE 스트리밍 | server.py `POST /ask/stream` | `fastapi.StreamingResponse`로 SSE 이벤트 반환. `trace` 이벤트 (TraceStep당 하나) → `answer` 이벤트 (최종 답변 + cited). 동기 retrieval loop을 `threading.Thread` + `queue.Queue`로 async generator에 bridge. |
| Context reuse | retrieval.py `shown_ids` + `reuse_context` | 각 iteration에서 후보 목록에 등장한 모든 node_id를 추적. `reuse_context=True`일 때 이후 iteration에서는 이미 보여준 후보를 자동 제거 → 프롬프트 길이 감소. path_so_far가 3단계 넘으면 `...(생략)` 프리픽스로 truncate. CLI `ask --reuse-context`. |

### 7.6 v0.11 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| `/chat/stream` SSE | server.py `_stream_retrieval` + `chat_stream` | `/ask/stream`과 같은 threading + queue 패턴을 공유하는 헬퍼로 통합. `ChatSession` 히스토리가 `run_retrieval(history=...)`로 전달. 종료 이벤트에 `session_id`, `turn` 포함. 세션은 서버 state에서 관리, 1시간 비활성 시 만료. |
| Usage 이벤트 SSE | server.py `_stream_retrieval` | 각 trace step 직후 `event: usage` 프레임을 emit. per-request `UsageTracker`로 cumulative counts를 계산해 `{total_calls, prompt_tokens, completion_tokens, total_tokens, elapsed}` JSON 반환. 클라이언트가 실시간 토큰 미터 표시 가능. |
| Per-vault 캐시 | vault.py `vault_for_note`, `build_long_subtrees_multi`, `summarize_atomic_notes_multi` | 멀티 vault에서 각 노트의 `file_path`로 소속 vault를 찾아 해당 vault의 `TreeCache`/`SummaryCache`에 기록. 기존에는 primary vault의 `.pagewiki-cache/`로 통합되던 것을 vault별로 분리. longest-match 라우팅으로 중첩 vault도 올바르게 처리. |
| `pagewiki usage-report` | cli.py `usage_report` 커맨드 | SQLite 사용량 DB 조회 CLI. `--since`/`--until`/`--phase`/`--recent` 필터. phase별 breakdown Rich 테이블 + 최근 이벤트 리스트. `serve --usage-db`와 페어링. |
| compile 토큰 추적 | cli.py `compile` | `--usage`/`--usage-db` 플래그 추가. `_make_chat_fn`에 tracker + store 주입해 entity extraction/page generation 양쪽 phase의 토큰 사용량 집계. |
| Obsidian plugin v0.10 catch-up | obsidian-plugin/main.ts | 누락됐던 v0.8~v0.10 flag (`--usage`, `--max-tokens`, `--json-mode`, `--reuse-context`) 모두 설정 탭 + runAsk/ChatModal에 연결. |

### 7.7 v0.12 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| WebSocket `/ask/ws` | server.py `ask_ws` | FastAPI WebSocket 엔드포인트. 양방향 메시지 프로토콜 (`ask`/`cancel`/`ping`). 서버는 `trace`/`usage`/`answer`/`cancelled`/`error`/`pong` 전송. 클라이언트가 `cancel`을 보내면 `threading.Event`로 retrieval loop에 신호 → 다음 iteration 시작 전 `should_stop` 콜백이 확인하고 clean abort. |
| `should_stop` 콜백 | retrieval.py | `run_retrieval`에 `should_stop: Callable[[], bool] | None` 파라미터 추가. 매 iteration 시작 전 호출되어 True 반환 시 loop abort → `TraceStep("cancel", ...)` emit. WebSocket 외에도 임의 interrupt 구현 가능. |
| Daily 롤업 | usage_store.py `rollup_day`, `rollup_range`, `query_daily`, `usage_daily` 테이블 | 대용량 usage DB에서 날짜별 집계 쿼리 가속. `rollup_day(date)`가 하루 분량을 pre-aggregate → `usage_daily` 테이블에 JSON phase breakdown 저장. `rollup_range(since, until)`로 범위 일괄 처리. `usage-report --daily`로 CLI 노출. 빈 날짜도 zero row를 한 번만 기록해 idempotent. |
| Cross-vault retrieval | retrieval.py `run_cross_vault_retrieval` | 멀티 vault를 하나의 가상 루트로 merge하는 대신 **각 vault에서 독립적으로 full retrieval 실행 → 결과 synthesize**. 각 vault의 wiki-link 인덱스가 스코프 유지되어 spurious cross-vault 링크 해결 방지. `cited_nodes`는 `<vault_name>::` 프리픽스로 attribution. CLI `ask --per-vault` (멀티 vault 시에만 의미). |
| 플러그인 server-mode | obsidian-plugin/main.ts `streamSSE`, `_runAskServerMode`, `_submitServerMode` | 새 `serverUrl` 설정이 비어있지 않으면 `fetch('/ask/stream')` 또는 `/chat/stream`로 직접 POST하고 SSE를 파싱. Node.js `ReadableStream` + 프레임 분리기로 `trace`/`usage`/`answer` 이벤트 처리. ChatModal은 live placeholder를 mutate해 스트리밍 중 진행 표시. session_id 자동 유지로 후속 질문이 서버-side history 활용. |

### 7.8 v0.13 상세 (폴리싱 + 리팩토링)

| 기능 | 모듈 | 설명 |
|---|---|---|
| chat flags 확장 | cli.py `chat` | `--json-mode` / `--reuse-context` 노출. ask에만 있던 v0.10 flags 를 chat에도 forwarding. |
| 실제 LiteLLM 토큰 | server.py `_stream_retrieval`, `ask_ws` | per-request `local_tracker`가 `state.tracker`의 before/after delta를 읽어 char/3 추정치가 아닌 실제 LiteLLM 토큰 카운트를 SSE/WS usage 이벤트로 emit. 공유 tracker가 advance 하지 않으면 char/3 fallback. |
| 플러그인 WebSocket + Cancel | obsidian-plugin/main.ts `connectAskWS`, ChatModal `cancelBtn` | 새 `useWebSocket` 설정이 활성화되면 `/ask/ws`로 접속해 진행 중 취소 가능. ChatModal에 "Cancel" 버튼이 활성 WS 핸들의 cancel() 호출. 서버는 `threading.Event`로 루프에 신호 → `should_stop` 콜백이 clean abort. |
| Cross-vault × decompose | retrieval/cross_vault.py `decompose` 파라미터 | `run_cross_vault_retrieval(decompose=True)` 시 각 vault에서 `run_decomposed_retrieval` 실행 → vault별 분해-합성 → cross-vault 최종 합성. CLI `ask --per-vault --decompose`로 활성화. |
| usage-report CSV/JSON | cli.py `usage_report`, `--format` 옵션 | `--format json` / `--format csv` 머신 출력. Rich 마크업 없이 `click.echo`로 clean stdout. JSON은 total + by_phase + recent + daily 통합 페이로드, CSV는 section-tagged rows. |
| retrieval 패키지 분할 | src/pagewiki/retrieval/{types,helpers,core,decompose,cross_vault}.py | 846줄 단일 파일을 5개 모듈 subpackage로 분할. `__init__.py`가 public API 재노출해 기존 import 호환 유지 (`from pagewiki.retrieval import run_retrieval` 그대로 작동). |

### 7.9 v0.14 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| Usage DB rolling retention | usage_store.py `prune_events_before`, `prune_older_than_days` | 오래된 `usage_events` 행을 삭제하기 전에 영향받는 날짜들을 `usage_daily` 로 rollup해서 historical 집계는 보존. VACUUM으로 파일 크기 즉시 회수. CLI `usage-report --prune-older-than N` 로 노출. |
| `GET /usage/history` | server.py `usage_history` + `UsageHistoryResponse` | `since`/`until`/`phase`/`limit` 쿼리 파라미터로 SQLite store를 직접 조회. 응답에 `summary` (in-memory + persistent), `events` (raw), `daily` (rollup)를 묶어 반환. `--usage-db` 없이 실행된 서버는 503. |
| Web UI | webui.py `build_ui_html`, server.py `web_ui` | 외부 CSS/JS/빌드 없는 self-contained 단일 페이지. `fetch` + `ReadableStream`으로 `/ask/stream` SSE 소비. Decompose/JSON/Reuse 토글 + Cancel 버튼 + 라이브 토큰 미터. `PAGEWIKI_UI_HTML` env로 커스텀 HTML 교체 가능. `GET /` 로 서빙. |
| Budget split 정책 | cli.py `_parse_token_split`, vault.py `summarize_atomic_notes(max_tokens=...)` | `--token-split A:B:C` 로 `--max-tokens`를 summarize/retrieve/synthesis 3단계에 비례 분배. summarize는 soft cap (넘으면 남은 노트 건너뛰고 제목만 사용), retrieve는 기존 hard cap 재사용. 비율은 정규화되므로 `20:60:20`과 `1:3:1`이 동일. |
| Prompt caching | prompts.py `SELECT_NODE_SYSTEM` 등 + `*_user_prompt`, retrieval/core.py `system_chat_fn` | Select/Evaluate/Final 프롬프트를 `(system, user)` 쌍으로 분리. CLI `--prompt-cache` 활성 시 `ollama_client.chat(user, system=SELECT_NODE_SYSTEM)` 으로 호출해 Ollama가 stable prefix의 KV cache를 재사용. history-aware final 답변과 JSON-mode는 기존 concat 경로 유지 (내용이 call 마다 달라서 캐싱 효과 없음). |

### 7.10 v0.15 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| Cross-vault 병렬 실행 | retrieval/cross_vault.py | `run_cross_vault_retrieval(parallel_workers=N)` 시 `ThreadPoolExecutor`로 각 vault를 동시에 retrieval. 인덱스 보존 정렬로 `cited_nodes` 프리픽스 순서 유지. CLI `ask --per-vault`는 자동으로 `--max-workers` 값 사용. |
| Prompt cache 히트율 | usage.py `UsageEvent.cacheable`, `UsageTracker.cacheable_ratio()` | `_make_system_chat_fn` 경로 호출 시 `tracker.record(..., cacheable=True)`. `cacheable_ratio()`는 전체 LLM 호출 중 cacheable 경로로 디스패치된 비율. `ask --usage`가 "Prompt-cache eligible: N/M (X%)" 라인으로 출력. |
| `/usage/history/stream` | server.py `usage_history_stream` | SSE 엔드포인트. 연결 시 `initial` 프레임에 최근 N개 이벤트 스냅샷, 이후 `poll_interval`마다 store 폴링해서 새 이벤트만 `event` 프레임으로 송출. idle 시 `heartbeat`. `max_events` / `max_duration` 양쪽 종료 조건. |
| Plugin WebSocket 확장 | obsidian-plugin/main.ts `connectAskWS(opts)`, server.py `/ask/ws` | `ask` 프레임에 `max_tokens`/`token_split`/`json_mode`/`reuse_context` 수용. 서버는 token_split 비율을 파싱해 retrieve+synth 합산을 `run_retrieval(max_tokens=...)`에 적용. 플러그인 Settings에 `tokenSplit` 텍스트 필드 추가. |
| Web UI sparkline | webui.py `_HTML_TEMPLATE` | SVG `<polyline>` 기반 인라인 스파크라인. `usage` SSE 이벤트마다 `usageSeries` 에 누적 → `renderSparkline()` 호출. 외부 차트 라이브러리 없음, 40×200px. |

### 7.11 v0.16 상세

| 기능 | 모듈 | 설명 |
|---|---|---|
| Parallel summarize 예산 파동 | vault.py `summarize_atomic_notes` | `max_tokens + tracker` 설정 시 병렬 경로도 작동: `ThreadPoolExecutor.map`을 `max_workers` 크기의 wave로 나눠 호출하고, 각 wave 사이에 budget 체크. 기존의 "budget이면 무조건 순차" 제약 제거, 예산 엄수하면서도 병렬 속도 유지. |
| Inferred cache 레이턴시 savings | usage.py `cacheable_latency_savings()` | Ollama가 실제 KV-hit/miss를 노출하지 않으므로 간접 측정: 첫 cacheable 호출을 cold proxy, 이후 호출의 평균을 hit proxy로 가정. `{first_call_seconds, subsequent_mean_seconds, savings_per_call_seconds, inferred_hit_rate, samples}` 반환. `ask --usage`가 savings 라인을 추가로 출력. |
| Web UI historical 뷰 | webui.py | 접을 수 있는 `<details>` 섹션 추가. Start/Stop 버튼으로 `/usage/history/stream` 구독 on/off. `initial`/`event`/`heartbeat`/`done` 프레임 파싱해서 4-column 테이블에 append. 최대 200행으로 capped. 외부 deps 없음. |
| Plugin WS prompt-cache 토글 | server.py `ServerState.system_chat_fn`, cli.py `serve --prompt-cache`, plugin `promptCacheWebSocket` | 서버가 선택적 `system_chat_fn`을 보유. WebSocket `ask` 프레임에 `prompt_cache: true`가 있고 서버에 system_chat_fn이 있으면 해당 경로로 retrieval 실행. `_make_system_chat_fn` 재사용. 플러그인 Settings에 토글 추가. |
| Cross-vault `allow_partial` | retrieval/cross_vault.py | 한 vault의 `_run_one`이 예외를 raise할 때 `allow_partial=False`면 전파, `True`면 잡아서 실패 vault만 제외하고 나머지로 synthesis. 모든 vault가 실패하면 `"전체 실패"` 메시지 반환 (synth 호출 생략). CLI `ask --allow-partial`. |

### 7.12 v0.17+ 향후 계획

- Streaming 중간에 WebSocket cancel → 서버 cleanup 로그 + retry 가이드
- Plugin에 historical view 버튼 (현재는 Web UI 전용)
- Usage DB retention policy를 `serve` 시작 시 주기 실행
- Cross-vault 부분 실패 시 실패 vault 재시도 옵션
- Prompt cache 통계를 `/usage` 엔드포인트에 포함

## 8. 명시적 비목표 (Non-Goals)

- ❌ 수백만 문서 스케일 (볼트 수천~수만 노트 대상)
- ❌ 벡터 DB 하이브리드 (순수 vectorless 유지)
- ❌ 클라우드 LLM 지원 (로컬 프라이버시 우선)
