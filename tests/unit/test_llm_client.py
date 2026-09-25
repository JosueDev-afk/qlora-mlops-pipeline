"""Qwen client: the vLLM request shape, and failures that come back as no text.

No network: HTTP goes through httpx.MockTransport.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from src.agent.llm.client import Completion, QwenClient
from src.agent.llm.structured import parse_or_ambiguous
from src.common.prompts import load_prompt

PROMPT = load_prompt("extract_entity", version=1)
VARIABLES = {"field": "phone", "transcript": "ocho uno ocho dos", "asr_confidence": 0.91}
ANSWER = '{"field": "phone", "normalized_value": null, "confidence": 0.2, "needs_reprompt": true}'


def _complete(handler: Any, variables: dict = VARIABLES) -> Completion:
    async def run() -> Completion:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = QwenClient("http://vllm:8000/v1/", model="qwen3-4b-contact", client=http)
        try:
            return await client.structured(PROMPT, variables)
        finally:
            await http.aclose()

    return asyncio.run(run())


def _chat(content: object) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_sends_greedy_chat_with_thinking_disabled() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_chat(ANSWER))

    completion = _complete(handler)
    assert str(seen[0].url) == "http://vllm:8000/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body["model"] == "qwen3-4b-contact"
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    system, user = body["messages"]
    assert system == {"role": "system", "content": PROMPT.body["system"]}
    assert "transcripción: ocho uno ocho dos" in user["content"]
    assert completion.text == ANSWER and completion.error is None
    assert completion.latency_ms >= 0


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (lambda r: httpx.Response(503, text="overloaded"), "http_503"),
        (lambda r: httpx.Response(200, text="not json"), "malformed_response"),
        (lambda r: httpx.Response(200, json={"choices": []}), "malformed_response"),
        (lambda r: httpx.Response(200, json=_chat(None)), "no_content"),
    ],
)
def test_failures_come_back_as_no_text(handler: Any, error: str) -> None:
    completion = _complete(handler)
    assert completion.text is None
    assert completion.error == error


@pytest.mark.parametrize(
    ("exc", "error"),
    [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "transport")],
)
def test_transport_failures_do_not_raise(exc: type[httpx.HTTPError], error: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc("boom", request=request)

    assert _complete(handler).error == error


def test_no_text_reads_as_ambiguous() -> None:
    """The contract with the parser: a failed call is an ambiguous extraction."""
    completion = _complete(lambda r: httpx.Response(500))
    result = parse_or_ambiguous(completion.text, "extract_entity", field="phone")
    assert result["needs_reprompt"] is True and result["normalized_value"] is None


def test_missing_template_variable_is_a_programming_error() -> None:
    with pytest.raises(KeyError, match="transcript"):
        _complete(lambda r: httpx.Response(200, json=_chat(ANSWER)), variables={"field": "phone"})
