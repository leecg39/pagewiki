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

# LLM-Wiki 컴파일 (v0.3+): entity 추출 → 위키 페이지 자동 생성
pagewiki compile --folder Research             # → {vault}/LLM-Wiki/

# 파일 변경 감시 (v0.4+): 노트 편집 시 자동 감지
pagewiki watch --folder Research --interval 10  # Ctrl+C로 종료

# 명시적으로 vault 지정도 여전히 가능 (공백 경로는 따옴표)
pagewiki ask "2024년 3분기 매출 관련 리서치 요약" \
  --vault "~/Documents/Obsidian Vault" \
  --folder Research \
  --model ollama/gemma4:26b
```

[Yakitrak/notesmd-cli](https://github.com/Yakitrak/notesmd-cli)를 설치해 두면:
- `--vault` 생략 시 `notesmd-cli print-default`로 기본 볼트 자동 감지
- `pagewiki ask` 출력 하단에 `notesmd-cli open <note> --section <anchor>` 명령이
  자동 생성되어 인용된 섹션을 바로 editor에서 열 수 있음

파이프라인은 4단계로 구성됩니다 (v0.1.2):

1. **Scan** — 볼트 walking + 3-tier 분류 (MICRO/ATOMIC/LONG)
2. **Summarize atomic** — ATOMIC 노트에 한 줄 요약
3. **Build long sub-trees** — LONG 노트마다 PageIndex 섹션 트리 생성 (캐시됨)
4. **Multi-hop retrieval** — ToC Review → Select → Descend → Evaluate → Final Answer

출력:
- 답변 (stdout)
- 인용된 노드 경로 (파일명 + 섹션 id, 예: `paper.md#0003`)
- `{vault}/Research/.pagewiki-log/{timestamp}.md` 감사 로그
- Layer 2 캐시: `{vault}/.pagewiki-cache/trees/` (자동 invalidate)

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
- **v0.5 (현재)**: Obsidian 플러그인 UI — Command Palette에서 Scan/Ask/Compile 실행, Settings 탭, 결과 모달

## Obsidian 플러그인 (v0.5)

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
- **PageWiki: Compile LLM-Wiki** — entity 추출 → 위키 생성
- **PageWiki: List discovered vaults** — 감지된 볼트 목록

Settings 탭에서 Python 경로, 모델, 기본 폴더 설정 가능.

로컬 머신에서 Ollama + Gemma 4로 실제 품질/성능을 검증하려면
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) 참고.

## 트레이드오프 (솔직하게)

- ⏱️ **레이턴시**: 쿼리당 60~300초 (로컬 Gemma 4 기준). 실시간 채팅 아님
- 💭 **연산 비용**: 임베딩은 0, 대신 LLM 추론 토큰 소모 큼 (로컬이라 전기요금)
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
