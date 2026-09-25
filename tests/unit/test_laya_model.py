"""Laya client: Laya's wire protocol, and refusal of anything uncalibrated.

No network: HTTP goes through httpx.MockTransport. The response bodies mirror
what `laya.serve` returns for laya 0.3.20 plus the server's calibration stamp.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from src.agent.decision.base import DecisionError
from src.agent.decision.laya_model import LayaDecisionModel

RUN_ID = "cal-run-7"
QUESTIONS = {
    "interruption": {"type": "noul", "instructions": "¿Quiere tomar el turno?"},
    "kind": {
        "type": "choice",
        "instructions": "¿De qué tipo es?",
        "criteria": {"backchannel": "acusa recibo", "rejection": "no puede atender"},
    },
}


def _answers() -> dict[str, Any]:
    return {
        "interruption": {"type": "noul", "noul": 0.83, "confidence": 0.83,
                         "answer_confidence": 0.83, "action": {}},
        "kind": {"type": "choice", "choice": "rejection",
                 "probabilities": {"backchannel": 0.3, "rejection": 0.7},
                 "confidence": 0.12, "answer_confidence": 0.7, "action": {}},
    }  # fmt: skip


def _payload(**overrides: Any) -> dict[str, Any]:
    body = {
        "model": "laya-multilingual",
        "answers": _answers(),
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "routing": {"model": "multilingual", "reason": "explicit model"},
        "calibration": {"run_id": RUN_ID},
    }
    body.update(overrides)
    return body


def _decide(handler: Any, *, api_key: str | None = None, questions: dict = QUESTIONS) -> Any:
    async def run() -> Any:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        model = LayaDecisionModel(
            "http://laya:8001/", calibration_run_id=RUN_ID, api_key=api_key, client=client
        )
        try:
            return await model.decide({"transcript": "ahorita no puedo"}, questions)
        finally:
            await client.aclose()

    return asyncio.run(run())


def _reply(payload: Any, status: int = 200) -> Any:
    return lambda request: httpx.Response(status, json=payload)


def test_posts_laya_protocol_pinned_to_multilingual() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_payload())

    _decide(handler, api_key="secret")
    request = seen[0]
    assert str(request.url) == "http://laya:8001/v1/systemone"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["model"] == "multilingual"
    assert body["questions"] == QUESTIONS
    assert body["state"] == {"transcript": "ahorita no puedo"}


def test_maps_answers_to_decisions_with_calibrated_confidence() -> None:
    decisions = _decide(_reply(_payload()))
    assert decisions["interruption"].value == 0.83  # P(true), gated on a threshold
    assert decisions["kind"].value == "rejection"
    # answer_confidence, not Laya's entropy-based `confidence` (0.12)
    assert decisions["kind"].confidence == 0.7
    assert decisions["kind"].raw["probabilities"] == {"backchannel": 0.3, "rejection": 0.7}


def test_score_answers_carry_the_expected_level() -> None:
    questions = {"urgency": {"type": "score", "instructions": "¿Qué tan urgente?",
                             "criteria": ["nada", "algo", "mucho"]}}  # fmt: skip
    answers = {"urgency": {"type": "score", "score": 1.4, "answer_confidence": 0.6}}
    decisions = _decide(_reply(_payload(answers=answers)), questions=questions)
    assert decisions["urgency"].value == 1.4


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        (_payload(routing={"model": "english"}), "wrong_checkpoint"),
        (_payload(calibration=None), "uncalibrated"),
        (_payload(calibration={"run_id": "cal-run-6"}), "calibration_mismatch"),
        (_payload(answers=None), "malformed"),
        (_payload(answers={"kind": _answers()["kind"]}), "malformed"),  # a question missing
        (["not", "an", "object"], "malformed"),
    ],
)
def test_untrusted_responses_raise(payload: Any, kind: str) -> None:
    with pytest.raises(DecisionError) as info:
        _decide(_reply(payload))
    assert info.value.kind == kind


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answer_confidence", 1.3),
        ("answer_confidence", None),
        ("noul", -0.1),
        ("noul", True),
        ("type", "choice"),
    ],
)
def test_malformed_noul_answers_raise(field: str, value: Any) -> None:
    answers = _answers()
    answers["interruption"][field] = value
    with pytest.raises(DecisionError, match="malformed"):
        _decide(_reply(_payload(answers=answers)))


def test_choice_outside_the_criteria_raises() -> None:
    answers = _answers()
    answers["kind"]["choice"] = "objection"
    with pytest.raises(DecisionError, match="unknown label"):
        _decide(_reply(_payload(answers=answers)))


def test_http_errors_raise_with_status() -> None:
    with pytest.raises(DecisionError) as info:
        _decide(_reply({"detail": "no fitted temperature for choice:6-10"}, status=422))
    assert info.value.kind == "http_422"
    assert "choice:6-10" in info.value.detail


def test_timeout_raises_decision_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(DecisionError) as info:
        _decide(handler)
    assert info.value.kind == "timeout"


def test_connection_error_raises_decision_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(DecisionError) as info:
        _decide(handler)
    assert info.value.kind == "transport"


def test_calibration_run_id_is_required() -> None:
    with pytest.raises(ValueError, match="calibration_run_id"):
        LayaDecisionModel("http://laya:8001", calibration_run_id="")
