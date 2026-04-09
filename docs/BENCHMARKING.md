# pagewiki Benchmarking & Smoke Testing

이 문서는 **로컬 머신에서** pagewiki의 Ollama + Gemma 4 통합을
직접 검증하는 방법을 설명합니다. pytest 단위 테스트는 mocked
`chat_fn`으로 빠르게 돌지만, 실제 모델 품질과 실제 볼트 성능은
여기 있는 스크립트로만 확인할 수 있습니다.

## 사전 준비

```bash
# 1. Ollama 설치
brew install ollama          # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh  (Linux)

# 2. 모델 풀
ollama pull gemma4:26b       # 24GB+ VRAM 권장
# or: ollama pull gemma4:e4b (16GB 머신용 폴백)

# 3. Ollama 데몬 실행 (자동 시작 안 되어 있으면)
ollama serve

# 4. pagewiki 설치 (편집 모드)
cd /path/to/pagewiki
pip install -e .
```

---

## 1. Prompt Smoke Test — `scripts/ollama_smoke.py`

**목적**: pagewiki가 쓰는 5가지 프롬프트 각각에 대해 Gemma 4가
파싱 가능한 응답을 반환하는지 **실제로** 확인합니다. 프롬프트
품질을 바꾸거나, 모델을 swap 하거나, Ollama를 업그레이드했을
때마다 실행하세요.

### 무엇을 검증하는가

| 프롬프트 | 검증 | 실패 의미 |
|---|---|---|
| `atomic_summary_prompt` | 100자 이내 한국어 한 문장 | 모델이 preamble 추가 / 영어 응답 |
| `section_summary_prompt` | 80자 이내, 섹션 본인 주제 요약 | "이 섹션은 ~에 대한..." 같은 메타 응답 |
| `select_node_prompt` | `parse_select_response`가 `("SELECT", node_id)` 반환 | "SELECT:" 포맷 미준수 |
| `evaluate_prompt` | `parse_evaluate_response`가 `(True, reason)` 반환 | "SUFFICIENT:" 포맷 미준수 |
| `final_answer_prompt` | 답변에 증거 중 최소 1개 숫자 포함 | 근거 없이 환각 / 근거 부족 |

### 실행

```bash
# 기본값 (ollama/gemma4:26b, 128K context)
python scripts/ollama_smoke.py

# 작은 모델로
python scripts/ollama_smoke.py --model ollama/gemma4:e4b --num-ctx 32768

# 원격 Ollama
OLLAMA_BASE_URL=http://192.168.1.10:11434 python scripts/ollama_smoke.py
```

### 출력 예시

```
pagewiki Ollama smoke test
  model:     ollama/gemma4:26b
  num_ctx:   131072
  base_url:  http://localhost:11434

[PASS] atomic_summary_prompt  (3.2s)
       detail:   OK (non-empty, under 300 chars)
       response: '2024년 3분기 매출이 1조 2천억원으로 15% 성장했습니다.'

[PASS] select_node_prompt  (4.1s)
       detail:   OK (picked the right note)
       response: 'SELECT: Research/q3.md'

...

────────────────────────────────────────────────────
Summary: 5/5 prompts passed
Total latency: 18.4s
```

### 실패 시 대응

- **`ollama not reachable`**: `ollama serve` 수동 실행 후 재시도
- **`model not pulled`**: 안내에 따라 `ollama pull <name>` 실행
- **`select_node_prompt FAIL`**: 모델이 "SELECT:" prefix 없이
  응답. `src/pagewiki/prompts.py`의 `_SELECT_NODE_SYSTEM`을
  더 엄격하게 수정하거나 temperature를 낮춰보세요.
- **`final_answer_prompt FAIL`**: 모델이 근거를 인용하지 않음.
  볼트가 너무 작아서 evidence가 약할 수 있으므로 다른 모델(e4b→26b)로 재시도.

---

## 2. Real-Vault Benchmark — `scripts/benchmark_vault.py`

**목적**: 실제 본인 Obsidian 볼트에 대해 전체 파이프라인
(Scan → Build → Retrieval × N queries)을 돌리고 per-query
wall-clock / 인용 노드 수 / 답변 길이를 측정합니다.

### 실행

```bash
# 기본 3개 질의
python scripts/benchmark_vault.py \
    --vault ~/Documents/Obsidian \
    --folder Research \
    --model ollama/gemma4:26b

# 자체 질의 파일 (한 줄당 질의 하나, `#`로 시작하는 줄은 주석)
python scripts/benchmark_vault.py \
    --vault ~/Documents/Obsidian \
    --folder Research \
    --queries-file my-queries.txt \
    --json-out                # 회귀 추적용 JSON 파일도 같이 기록
```

### 측정 지표

```
[Scan] 347 notes in 0.12s
       MICRO=120  ATOMIC=185  LONG=42
       total tokens ≈ 548,200
       largest: 'RAG Paper Review' (18,420 tokens)

[Build] 42 LONG notes in 187.3s  (42 built, 0 cached)
        total sections: 312
        largest sub-tree: 'RAG Paper Review' (24 sections)

[Queries]
  1. 이 볼트의 핵심 주제 3가지를 요약해줘
     elapsed=112.4s  iter=4  cited=3  answer=587 chars
     cites: Research/rag.md, Research/llm_wiki.md, Research/pageindex.md
     preview: '이 볼트는 1) vectorless RAG...'
```

### 첫 실행 vs 캐시 실행

Build 단계에서 가장 큰 차이가 납니다. 첫 실행:

```
[Build] 42 LONG notes in 187.3s  (42 built, 0 cached)
```

같은 볼트에 대해 다시 실행:

```
[Build] 42 LONG notes in 0.4s  (0 built, 42 cached)
```

`{vault}/.pagewiki-cache/trees/*.json` 파일들이 캐시 저장소이며,
원본 노트의 mtime이 바뀔 때만 자동으로 재빌드됩니다.
안전하게 지우려면 `rm -rf {vault}/.pagewiki-cache/` 하면 됩니다.

### Regression tracking

`--json-out` 플래그를 주면 `benchmark-<timestamp>.json` 파일이
생성됩니다. 이 파일을 git에 커밋하거나 별도 디렉토리에 쌓으면,
프롬프트/모델/알고리즘을 바꿀 때마다 per-query wall-clock과
cited_count의 변화를 비교할 수 있습니다.

---

## 3. 자주 묻는 트러블슈팅

### "scan은 되는데 build_long이 0개 처리"
- 3-tier 분류에서 LONG 노트가 하나도 없다는 뜻입니다.
- `scan` 명령의 출력에서 `LONG` 카운트를 확인하세요.
- 기본 임계값은 3000 토큰 (≈ 9000자)입니다.

### "build 단계가 20분 걸림"
- 첫 실행 + `chat_fn`을 붙여서 섹션 요약을 생성하는 경우
  LONG 노트 × (long section 수)만큼 LLM 호출이 발생합니다.
  (작은 섹션은 `summary_token_threshold=200` 기본값 아래면 LLM 호출 생략.)
- `--skip-summaries` 플래그로 전부 skip 하면 훨씬 빠릅니다.
- 두 번째 실행부터는 디스크 캐시 덕에 sub-second로 끝납니다.

### "Gemma 4 응답이 한국어가 아니라 영어"
- 모델이 프롬프트의 "한국어" 지시를 따르지 않는 경우가 있음.
- 26B 모델에서는 거의 발생 안 함. e4b에서는 가끔 발생.
- Workaround: `prompts.py`의 템플릿 앞에 `"반드시 한국어로만 답변하세요."`
  한 줄 추가.

### "node_id 충돌 (`Research/paper.md` vs `Archive/paper.md`)"
- v0.1.3부터는 vault-relative path prefix로 구분합니다.
- 재발하면 `test_same_filename_in_different_folders_stays_distinct`
  회귀 테스트가 바로 fail할 것입니다.

### "v0.1.2 caches가 v0.1.3에서 왜 재빌드되는지"
- `cache.ADAPTER_VERSION`이 `"v0.1.2"` → `"v0.1.3"`으로 bump됐기 때문.
- 의도적인 동작입니다 — v0.1.3 트리는 `(intro)` synthetic section과
  vault-relative node_id를 포함하므로 구 cache와 schema가 다릅니다.

---

## 4. 파이프라인 전체 검증 순서 (권장)

새 볼트 / 새 모델을 도입할 때의 권장 순서:

```bash
# Step 1: Ollama 동작 확인
ollama run gemma4:26b "hello"

# Step 2: 프롬프트 품질 smoke test
python scripts/ollama_smoke.py

# Step 3: 볼트 구조 확인 (LLM 호출 없음)
pagewiki scan --vault ~/Documents/Obsidian --folder Research

# Step 4: Layer 2 sub-tree 미리 빌드 (LLM 없이, 캐시 채움)
pagewiki scan --vault ~/Documents/Obsidian --folder Research --build-long

# Step 5: 실제 쿼리 1개로 end-to-end 검증
pagewiki ask "가장 중요한 노트는?" \
    --vault ~/Documents/Obsidian --folder Research

# Step 6: 종합 벤치마크 (여러 쿼리, JSON 기록)
python scripts/benchmark_vault.py \
    --vault ~/Documents/Obsidian --folder Research --json-out
```

각 단계가 fail 하면 바로 다음 단계로 넘어가지 말고, fail한
지점에서 원인을 해결한 뒤 재개하세요.
