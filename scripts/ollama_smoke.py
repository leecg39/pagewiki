#!/usr/bin/env python3
"""Ollama + Gemma 4 smoke test for pagewiki prompts.

This script is meant to be run **on the user's own machine** (which
has Ollama + Gemma 4 locally). The pagewiki CI sandbox does not ship
Ollama, so these checks are deliberately kept out of the unit test
suite — they are reproducible validation the user runs by hand
whenever they upgrade Ollama, swap models, or change a prompt.

What it does
------------
1. Pings the Ollama HTTP endpoint to confirm the daemon is reachable.
2. Verifies the target model is actually pulled (prompts to run
   ``ollama pull <model>`` otherwise).
3. Runs each pagewiki prompt type against the model with canned
   inputs:

     * ``atomic_summary_prompt``      — ATOMIC note → one-line summary
     * ``section_summary_prompt``     — LONG note section → one-line summary
     * ``select_node_prompt``         — ToC review → "SELECT: <id>" or "DONE: …"
     * ``evaluate_prompt``            — evidence check → "SUFFICIENT: …" or "INSUFFICIENT: …"
     * ``final_answer_prompt``        — synthesis over gathered notes

4. Validates that the response parses cleanly with pagewiki's parsers
   (``parse_select_response``, ``parse_evaluate_response``).
5. Reports per-prompt latency and a single PASS/FAIL summary.

Usage
-----
::

    python scripts/ollama_smoke.py
    python scripts/ollama_smoke.py --model ollama/gemma4:26b
    python scripts/ollama_smoke.py --model ollama/gemma4:e4b --num-ctx 32768

Exit code
---------
``0`` if every prompt returned a parseable response, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# pagewiki prompts are pure functions — we can import them without
# touching litellm / ollama, and then run them manually against the
# real model via ollama_client.chat().
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pagewiki.prompts import (  # noqa: E402
    NodeSummary,
    atomic_summary_prompt,
    evaluate_prompt,
    final_answer_prompt,
    parse_evaluate_response,
    parse_select_response,
    section_summary_prompt,
    select_node_prompt,
)


@dataclass
class SmokeResult:
    name: str
    ok: bool
    latency_s: float
    detail: str
    response_preview: str


def _check_ollama_reachable(base_url: str) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[!] Ollama not reachable at {base_url}: {e}", file=sys.stderr)
        return False


def _model_is_pulled(base_url: str, model_name: str) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[!] Could not list Ollama models: {e}", file=sys.stderr)
        return False
    for m in data.get("models", []):
        if m.get("name") == model_name:
            return True
    return False


def _chat(prompt: str, *, model: str, num_ctx: int) -> str:
    from pagewiki.ollama_client import chat

    return chat(prompt, model=model, num_ctx=num_ctx).text


def run_smoke(model: str, num_ctx: int) -> list[SmokeResult]:
    results: list[SmokeResult] = []

    # ─── 1. Atomic summary ───────────────────────────────────────────
    atomic_note = (
        "2024년 3분기 매출은 1조 2천억원으로 전년 동기 대비 15% 성장했습니다. "
        "주력 사업인 반도체 부문이 회복세를 보이며 전체 실적을 견인했습니다."
    )
    t0 = time.time()
    try:
        response = _chat(
            atomic_summary_prompt("Q3 Revenue", atomic_note),
            model=model,
            num_ctx=num_ctx,
        )
        ok = bool(response.strip()) and len(response.strip()) < 300
        detail = (
            "OK (non-empty, under 300 chars)"
            if ok
            else f"response too long or empty ({len(response)} chars)"
        )
    except Exception as e:  # noqa: BLE001
        response = ""
        ok = False
        detail = f"LLM call failed: {e}"
    results.append(
        SmokeResult(
            "atomic_summary_prompt",
            ok,
            time.time() - t0,
            detail,
            response[:200],
        )
    )

    # ─── 2. Section summary ──────────────────────────────────────────
    # Note: pagewiki's section_summary_prompt takes THREE args:
    # (note_title, section_title, section_text).
    section_text = (
        "# Methods\n\n우리는 6개월에 걸쳐 3개 도시에서 참여관찰과 설문조사를 "
        "병행했다. 설문 도구는 사전 파일럿 스터디와 IRB 심사를 통해 검증되었다. "
        "분석은 근거이론 접근을 사용했다."
    )
    t0 = time.time()
    try:
        response = _chat(
            section_summary_prompt("Research Paper", "Methods", section_text),
            model=model,
            num_ctx=num_ctx,
        )
        ok = bool(response.strip()) and len(response.strip()) < 300
        detail = (
            "OK" if ok else f"response too long or empty ({len(response)} chars)"
        )
    except Exception as e:  # noqa: BLE001
        response = ""
        ok = False
        detail = f"LLM call failed: {e}"
    results.append(
        SmokeResult(
            "section_summary_prompt",
            ok,
            time.time() - t0,
            detail,
            response[:200],
        )
    )

    # ─── 3. SELECT (ToC review) ──────────────────────────────────────
    candidates = [
        NodeSummary(
            node_id="Research/q3.md",
            title="Q3 Revenue",
            kind="note",
            summary="2024년 3분기 매출 1조 2천억원, 15% 성장",
        ),
        NodeSummary(
            node_id="Research/risks.md",
            title="Risk Factors",
            kind="note",
            summary="환율/공급망/규제 3대 리스크 요약",
        ),
        NodeSummary(
            node_id="Research/intro.md",
            title="회사 소개",
            kind="note",
            summary="조직 개요 및 연혁",
        ),
    ]
    t0 = time.time()
    try:
        response = _chat(
            select_node_prompt("2024년 Q3 매출 알려줘", candidates),
            model=model,
            num_ctx=num_ctx,
        )
        try:
            action, value = parse_select_response(response)
            ok = action in {"SELECT", "DONE"}
            if ok and action == "SELECT":
                ok = value == "Research/q3.md"
                detail = (
                    "OK (picked the right note)"
                    if ok
                    else f"picked {value!r}, expected Research/q3.md"
                )
            else:
                detail = f"parsed but action={action!r} value={value[:40]!r}"
        except ValueError as e:
            ok = False
            detail = f"response did not parse: {e}"
    except Exception as e:  # noqa: BLE001
        response = ""
        ok = False
        detail = f"LLM call failed: {e}"
    results.append(
        SmokeResult(
            "select_node_prompt",
            ok,
            time.time() - t0,
            detail,
            response[:200],
        )
    )

    # ─── 4. EVALUATE (sufficiency check) ─────────────────────────────
    eval_content = (
        "2024년 3분기 매출은 1조 2천억원으로 전년 동기 대비 15% 성장했습니다. "
        "영업이익은 1,800억원, 영업이익률은 15%로 견조한 수익성을 유지했습니다."
    )
    t0 = time.time()
    try:
        response = _chat(
            evaluate_prompt("Q3 매출과 영업이익률은?", "Q3 Revenue", eval_content),
            model=model,
            num_ctx=num_ctx,
        )
        try:
            sufficient, reason = parse_evaluate_response(response)
            ok = sufficient is True
            detail = (
                "OK (correctly marked sufficient)"
                if ok
                else f"marked {'INSUFFICIENT' if not sufficient else 'SUFFICIENT'}: {reason[:60]}"
            )
        except ValueError as e:
            ok = False
            detail = f"response did not parse: {e}"
    except Exception as e:  # noqa: BLE001
        response = ""
        ok = False
        detail = f"LLM call failed: {e}"
    results.append(
        SmokeResult(
            "evaluate_prompt",
            ok,
            time.time() - t0,
            detail,
            response[:200],
        )
    )

    # ─── 5. Final answer synthesis ───────────────────────────────────
    gathered = [
        (
            "Q3 Revenue",
            "2024년 3분기 매출은 1조 2천억원으로 전년 동기 대비 15% 성장했습니다.",
        ),
        (
            "Operating Income",
            "2024년 3분기 영업이익은 1,800억원, 영업이익률은 15%를 기록했습니다.",
        ),
    ]
    t0 = time.time()
    try:
        response = _chat(
            final_answer_prompt("2024년 Q3 실적을 요약해줘", gathered),
            model=model,
            num_ctx=num_ctx,
        )
        ok = bool(response.strip()) and (
            "1조 2천억" in response or "1,800" in response or "15%" in response
        )
        detail = (
            "OK (cites evidence)"
            if ok
            else "answer did not reference any of the evidence numbers"
        )
    except Exception as e:  # noqa: BLE001
        response = ""
        ok = False
        detail = f"LLM call failed: {e}"
    results.append(
        SmokeResult(
            "final_answer_prompt",
            ok,
            time.time() - t0,
            detail,
            response[:300],
        )
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("PAGEWIKI_MODEL", "ollama/gemma4:26b"),
        help="LiteLLM model id (e.g. ollama/gemma4:26b, ollama/gemma4:e4b)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=131072,
        help="Ollama context window (default: 131072 = Gemma 4 full 128K)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL (default: http://localhost:11434)",
    )
    args = parser.parse_args()

    print("pagewiki Ollama smoke test")
    print(f"  model:     {args.model}")
    print(f"  num_ctx:   {args.num_ctx}")
    print(f"  base_url:  {args.base_url}")
    print()

    if not _check_ollama_reachable(args.base_url):
        print("FAIL: Ollama daemon not reachable. Start it with:")
        print("  ollama serve")
        return 1

    bare_model = args.model.removeprefix("ollama/")
    if not _model_is_pulled(args.base_url, bare_model):
        print(f"FAIL: model '{bare_model}' is not pulled. Run:")
        print(f"  ollama pull {bare_model}")
        return 1

    results = run_smoke(args.model, args.num_ctx)

    total_ok = 0
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}  ({r.latency_s:.1f}s)")
        print(f"       detail:   {r.detail}")
        print(f"       response: {r.response_preview!r}")
        print()
        if r.ok:
            total_ok += 1

    print("─" * 60)
    print(f"Summary: {total_ok}/{len(results)} prompts passed")
    total_time = sum(r.latency_s for r in results)
    print(f"Total latency: {total_time:.1f}s")

    return 0 if total_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
