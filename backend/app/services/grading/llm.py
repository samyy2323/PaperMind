"""Locally hosted open-weight LLM clients (no third-party APIs)."""
from __future__ import annotations

from typing import Protocol

import httpx

from app.core.config import get_settings
from app.core.singleton import once


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class OllamaClient:
    """Ollama server on our own infrastructure (e.g. llama3.1:8b, mistral:7b)."""

    def __init__(self, base_url: str, model: str, timeout: float, max_tokens: int):
        self.base_url, self.model, self.timeout, self.max_tokens = base_url.rstrip("/"), model, timeout, max_tokens

    def generate(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": self.max_tokens},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class TransformersClient:
    """In-process Hugging Face `transformers` chat model (e.g. meta-llama/Meta-Llama-3-8B-Instruct)."""

    def __init__(self, model_id: str, max_tokens: int):
        import torch
        from transformers import pipeline

        self._pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        out = self._pipe(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )
        return out[0]["generated_text"][-1]["content"]


@once
def _load(backend: str, url: str, model: str, timeout: float, max_tokens: int) -> LLMClient:
    if backend == "ollama":
        return OllamaClient(url, model, timeout, max_tokens)
    if backend == "transformers":
        return TransformersClient(model, max_tokens)
    raise ValueError(f"Unknown LLM_BACKEND: {backend}")


def get_llm_client() -> LLMClient:
    s = get_settings()
    return _load(s.LLM_BACKEND, s.OLLAMA_URL, s.LLM_MODEL, s.LLM_TIMEOUT_S, s.LLM_MAX_NEW_TOKENS)
