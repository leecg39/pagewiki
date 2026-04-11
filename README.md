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
# 등록된 Obsidian 볼트 목록 확인 (notesmd-cli 또는 obsidian.json에서 자동 감지)
pagewiki vaults

# --vault 생략하면 기본 볼트 자동 감지 (v0.1.5+)
pagewiki scan --folder Research
pagewiki scan --show-graph                     # 전체 볼트 + 링크 그래프

# 단일 질의 (v0.1+)
pagewiki ask "2024년 3분기 매출 관련 리서치 요약" --folder Research

# 대화형 모드 (v0.6+): 후속 질문이 이전 답변 맥락을 이어받음
pagewiki chat --folder Research

# Frontmatter 필터 (v0.6+): 태그/날짜 기반 검색 범위 제한
pagewiki ask "attention 메커니즘 비교" --tag research --after 2024-01
pagewiki chat --tag ml --before 2025-06

# 멀티쿼리 분해 (v0.7+): 복합 질문을 서브쿼리로 분해 후 종합
pagewiki ask "Transformer와 RNN의 차이점과 장단점은?" --decompose

# 병렬 LLM 호출 (v0.7+): ATOMIC 노트 요약 병렬화
pagewiki ask "query" --max-workers 8

# 멀티 vault 검색 (v0.7+): 여러 볼트 동시 검색
pagewiki ask "query" --vault ~/Research --extra-vault ~/Work --extra-vault ~/Personal

# HTTP API 서버 (v0.7+): 트리를 메모리에 warm 유지, 반복 쿼리 가속
pagewiki serve --vault ~/Research --folder Research --port 8000
# 이후:
# curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
#   -d '{"query": "What is attention?"}'
# curl http://localhost:8000/usage   # 누적 토큰 사용량 (v0.9+)

# 토큰 예산 한도 (v0.9+): 쿼리당 토큰 하드 캡
pagewiki ask "query" --max-tokens 50000 --usage

# Chat 세션 usage 집계 (v0.9+)
pagewiki chat --usage --max-tokens 30000

# LLM-Wiki 컴파일 (v0.3+): entity 추출 → 위키 페이지 자동 생성
pagewiki compile --folder Research             # → {vault}/LLM-Wiki/

# 파일 변경 감시 (v0.4+): 노트 편집 시 자동 감지
pagewiki watch --folder Research --interval 10  # Ctrl+C로 종료

# 명시적으로 vault 지정도 여전히 가능 (공백 경로는 따옴표)
pagewiki ask "query" --vault "~/Documents/Obsidian Vault" --model ollama/gemma4:26b
```

[Yakitrak/notesmd-cli](https://github.com/Yakitrak/notesmd-cli)를 설치해 두면:
- `--vault` 생략 시 `notesmd-cli print-default`로 기본 볼트 자동 감지
- `pagewiki ask` 출력 하단에 `notesmd-cli open <note> --section <anchor>` 명령이
  자동 생성되어 인용된 섹션을 바로 editor에서 열 수 있음

파이프라인은 4단계로 구성됩니다:

1. **Scan** — 볼트 walking + 3-tier 분류 + frontmatter 파싱
2. **Summarize atomic** — ATOMIC 노트에 한 줄 요약 (v0.6부터 on-disk 캐시)
3. **Build long sub-trees** — LONG 노트마다 PageIndex 섹션 트리 생성 (캐시됨)
4. **Multi-hop retrieval** — ToC Review → Select → Descend → Evaluate → Final Answer (실시간 스트리밍)

출력:
- 답변 (stdout) + 실시간 탐색 과정 (SELECT/EVAL/XREF)
- 인용된 노드 경로 (파일명 + 섹션 id, 예: `paper.md#0003`)
- `{vault}/Research/.pagewiki-log/{timestamp}.md` 감사 로그
- 캐시: `{vault}/.pagewiki-cache/trees/` + `summaries/` (자동 invalidate)

## 로드맵

- v0.1: Literature 폴더 Deep-Research 모드 (단일 폴더)
- v0.1.1: Multi-hop reasoning 루프 (ToC Review → Select → Evaluate → Final)
- v0.1.2: PageIndex 실제 통합 — LONG 노트의 섹션 트리 빌드 + 디스크 캐시 + 섹션 단위 descend
- v0.1.3: h1-title flatten + `(intro)` 보존 / vault-relative section id / local validation scripts
- v0.1.4: `[[wiki-link]]` resolution index + `scan --show-graph` (PR #2)
- v0.1.5: notesmd-cli 통합 — `--vault` auto-discovery, `pagewiki vaults` 서브커맨드, `ask` citation을 `notesmd-cli open` 힌트로 표시
- v0.2: `[[wiki-link]]` retrieval traversal — 노트 평가 후 outgoing wiki-link를 교차참조 후보로 자동 추가, transitive chain following
- v0.3: Karpathy LLM-Wiki compiler — `pagewiki compile`로 entity 추출 → 위키 페이지 자동 생성 → `{vault}/LLM-Wiki/`에 교차참조된 위키 출력
- v0.4: 증분 재인덱싱 + mtime watcher — `pagewiki watch`로 파일 변경 실시간 감지
- v0.5: Obsidian 플러그인 UI — Command Palette에서 Scan/Ask/Compile 실행, Settings 탭, 결과 모달
- v0.6: 대화형 `chat` 모드, atomic summary 디스크 캐시, YAML frontmatter 필터 (`--tag`/`--after`/`--before`), 실시간 스트리밍 출력
- v0.7: 병렬 LLM 호출 (`--max-workers`), 멀티쿼리 분해 (`--decompose`), 멀티 vault 검색 (`--extra-vault`), HTTP API 서버 (`pagewiki serve`)
- v0.8: 토큰 사용량 추적 (`--usage`), SELECT 파싱 실패 시 자동 재시도, BM25 기반 후보 사전 랭킹, 서버 엔드포인트 테스트
- **v0.9 (현재)**: 토큰 예산 한도 (`--max-tokens`), chat 세션 usage 집계, 서버 `/usage` 엔드포인트, cited note BM25 재정렬

## Obsidian 플러그인 (v0.6)

터미널 없이 Obsidian 안에서 직접 pagewiki를 사용할 수 있습니다.

```bash
cd obsidian-plugin
npm install && npm run build
# main.js, manifest.json, styles.css를
# {vault}/.obsidian/plugins/pagewiki/ 에 복사
```

Command Palette (`Cmd/Ctrl + P`)에서:
- **PageWiki: Scan vault** — 3-tier 분류 + wiki-link 그래프
- **PageWiki: Ask a question** — 질문 입력 → 답변 모달
- **PageWiki: Chat** — 대화형 모드 (후속 질문 지원, v0.6)
- **PageWiki: Compile LLM-Wiki** — entity 추출 → 위키 생성
- **PageWiki: Toggle watch mode** — 파일 변경 감시 on/off
- **PageWiki: List discovered vaults** — 감지된 볼트 목록

Settings 탭에서 Python 경로, 모델, 기본 폴더 설정 가능.

로컬 머신에서 Ollama + Gemma 4로 실제 품질/성능을 검증하려면
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) 참고.

## 트레이드오프 (솔직하게)

- ⏱️ **레이턴시**: 쿼리당 60~300초 (로컬 Gemma 4 기준). v0.6 스트리밍으로 탐색 과정은 실시간 표시
- 💭 **연산 비용**: 임베딩은 0, 대신 LLM 추론 토큰 소모 큼 (로컬이라 전기요금). v0.6 캐시로 반복 쿼리 시 대폭 절감
- 📊 **확장성**: 단일 볼트/폴더에 최적. 수백만 문서엔 Vector DB가 더 유리

## 라이선스

MIT. [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)도 MIT.

`src/pagewiki/_vendor/pageindex/`에 PageIndex 마크다운 트리 빌더의
최소 부분이 번들되어 있습니다 (MIT, commit `f2dcffc`).
LICENSE 사본은 해당 디렉터리에 그대로 보존되어 있습니다.

## Credits

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — Reasoning-based vectorless RAG 엔진
- [Andrej Karpathy's LLM-Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — 파일시스템 기반 지식 컴파일 철학
- [Google Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) — 로컬 추론 엔진
