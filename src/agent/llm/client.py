"""Qwen over vLLM's OpenAI-compatible API, returning raw text for the parser.

Spoken over httpx rather than the openai SDK for the same reason as the
ElevenLabs adapters: one small documented endpoint, easy to test against
httpx.MockTransport. The client never parses or validates; nodes pass the
text to `parse_or_ambiguous`, so validation lives in one place (rule 4).

No failure raises. A timeout or an HTTP error comes back as a Completion
with no text, which the parser turns into `ambiguous` like any other invalid
output: a slow model must not crash a live call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from src.common.log import get_logger
from src.common.prompts import Prompt

log = get_logger(__name__)


@dataclass(frozen=True)
class Completion:
    """Raw model text plus what telemetry needs: why it failed and how long it took.

    `latency_ms` is request to complete JSON, the quantity the 500 ms budget
    is set on, not time to first token.
    """

    text: str | None
    latency_ms: float
    error: str | None = None


class LLMClient(Protocol):
    async def structured(self, prompt: Prompt, variables: dict[str, Any]) -> Completion: ...


class QwenClient:
    """Chat completions with greedy decoding and thinking disabled.

    Greedy (temperature 0) so a turn is reproducible, which also means a
    retry on the same input returns the same output: invalid JSON is
    answered by asking the caller again, never by calling the model again.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout_s: float = 2.0,
        max_tokens: int = 256,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = httpx.Timeout(timeout_s)
        self._max_tokens = max_tokens
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def aclose(self) -> None:
        """Close the HTTP client, unless the caller passed one in and owns it."""
        if self._owns_client:
            await self._client.aclose()

    def _body(self, prompt: Prompt, variables: dict[str, Any]) -> dict[str, Any]:
        # A missing variable is a programming error: it raises on every call,
        # including the happy path, instead of reaching the model half-filled.
        user = prompt.body["user_template"].format(**variables)
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt.body["system"]},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self._max_tokens,
            # Qwen3 reasoning tokens would spend the budget before the JSON.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def structured(self, prompt: Prompt, variables: dict[str, Any]) -> Completion:
        body = self._body(prompt, variables)
        started = time.perf_counter()
        error: str | None = None
        text: str | None = None
        try:
            response = await self._client.post(
                self._url, json=body, headers=self._headers, timeout=self._timeout
            )
            if response.status_code != 200:
                error = f"http_{response.status_code}"
            else:
                content = response.json()["choices"][0]["message"]["content"]
                text = content if isinstance(content, str) else None
                error = None if text is not None else "no_content"
        except httpx.TimeoutException:
            error = "timeout"
        except httpx.HTTPError:
            error = "transport"
        except (ValueError, KeyError, IndexError, TypeError):
            error = "malformed_response"
        latency_ms = (time.perf_counter() - started) * 1000
        if error:
            log.warning("llm_failed", prompt=prompt.id, version=prompt.version, error=error)
        return Completion(text=text, latency_ms=latency_ms, error=error)
