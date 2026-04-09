"""Thin wrapper around LiteLLM for local Gemma 4 via Ollama.

Why LiteLLM: PageIndex SDK already depends on LiteLLM for multi-model support,
so we reuse the same abstraction and avoid a second HTTP client. This also
means we can swap to any LiteLLM-supported provider by changing one string.

Environment:
  OLLAMA_BASE_URL   default: http://localhost:11434
  PAGEWIKI_MODEL    default: ollama/gemma4:26b
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from litellm import completion

DEFAULT_MODEL = os.getenv("PAGEWIKI_MODEL", "ollama/gemma4:26b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


def chat(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.1,
    num_ctx: int = 131072,
) -> LLMResponse:
    """Single-shot completion. Raises on transport errors.

    num_ctx=131072 targets Gemma 4's 128K context window. Drop to 32768 on
    <24GB VRAM machines to avoid quality degradation under memory pressure.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            api_base=DEFAULT_BASE_URL,
            num_ctx=num_ctx,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed (model={model}, base={DEFAULT_BASE_URL}): {e}"
        ) from e

    choice = response.choices[0].message.content or ""
    usage = response.usage
    return LLMResponse(
        text=choice,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


def summarize_one_line(text: str, *, model: str = DEFAULT_MODEL) -> str:
    """Generate a single-line summary for an atomic note.

    Used by vault Layer 1 to attach summaries to `tier=ATOMIC` leaves.
    """
    prompt = (
        "Summarize the following note in ONE Korean sentence (under 100 chars). "
        "Output only the sentence, no preamble.\n\n"
        f"{text[:4000]}"
    )
    return chat(prompt, model=model, temperature=0.0).text.strip()
