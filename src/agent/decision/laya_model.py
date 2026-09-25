"""Laya-multilingual decision model, reached over HTTP.

The model runs in `src/serving/laya.py`, which owns the model card's rules:
only the multilingual checkpoint (the English root collapses on non-Latin
script while staying confident), preloaded (a lazy router rebuilds it on a
language switch, 7-10 s), and calibrated (the temperatures fitted by
`calibrate_laya` are applied to the logits there, because Laya rounds the
probabilities it returns and re-tempering rounded values is lossy). Keeping
the model out of process keeps `src/agent/` free of torch.

This client trusts none of it: a response routed to another checkpoint, or
stamped with a calibration other than the one the agent's thresholds were
derived with, raises DecisionError. The wire protocol is Laya's own
`POST /v1/systemone`, so the server adds no endpoint of its own.
"""

from __future__ import annotations

import math
from typing import Any, TypeGuard

import httpx

from src.agent.decision.base import Decision, DecisionError, DecisionModel

CHECKPOINT = "multilingual"
DEFAULT_LANGUAGE = "es"
_VALUE_FIELD = {"choice": "choice", "score": "score", "noul": "noul"}


def _probability(value: object) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= value <= 1.0


class LayaDecisionModel(DecisionModel):
    """Client for the Laya server.

    `calibration_run_id` is the `calibrate_laya` run whose temperatures the
    server must be applying: the `call_rejected` threshold comes from the
    same run, so a server on another calibration would make it meaningless.
    """

    def __init__(
        self,
        base_url: str,
        *,
        calibration_run_id: str,
        timeout_s: float = 0.5,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not calibration_run_id:
            raise ValueError("calibration_run_id is required: thresholds need calibrated answers")
        self._url = base_url.rstrip("/") + "/v1/systemone"
        self._calibration_run_id = calibration_run_id
        self._timeout = httpx.Timeout(timeout_s)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def aclose(self) -> None:
        """Close the HTTP client, unless the caller passed one in and owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def decide(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        *,
        language: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Decision]:
        """Ask every question in one request.

        `language` is accepted for the interface and not sent: the server pins
        every request to the multilingual checkpoint, which needs no hint.
        """
        body = {"state": state, "questions": questions, "model": CHECKPOINT}
        try:
            response = await self._client.post(
                self._url, json=body, headers=self._headers, timeout=self._timeout
            )
        except httpx.TimeoutException as exc:
            raise DecisionError("timeout", str(exc)) from exc
        except httpx.HTTPError as exc:
            raise DecisionError("transport", str(exc)) from exc
        if response.status_code != 200:
            raise DecisionError(f"http_{response.status_code}", response.text[:200])
        try:
            payload = response.json()
        except ValueError as exc:
            raise DecisionError("malformed", "response is not JSON") from exc
        return self._decisions(payload, questions)

    def _decisions(self, payload: object, questions: dict[str, Any]) -> dict[str, Decision]:
        if not isinstance(payload, dict):
            raise DecisionError("malformed", "response is not an object")
        routed = (payload.get("routing") or {}).get("model")
        if routed != CHECKPOINT:
            raise DecisionError("wrong_checkpoint", f"answered by {routed!r}")
        calibration = payload.get("calibration")
        if not isinstance(calibration, dict) or not calibration.get("run_id"):
            raise DecisionError("uncalibrated", "the server applies no fitted temperatures")
        if calibration["run_id"] != self._calibration_run_id:
            raise DecisionError(
                "calibration_mismatch",
                f"server {calibration['run_id']!r}, expected {self._calibration_run_id!r}",
            )
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise DecisionError("malformed", "no answers")
        return {qid: self._decision(qid, q, answers.get(qid)) for qid, q in questions.items()}

    @staticmethod
    def _decision(qid: str, question: dict[str, Any], answer: object) -> Decision:
        """`answer_confidence`, not `confidence`: Laya's `confidence` is normalized
        entropy for choice and score, which temperature scaling does not fit."""
        qtype = question.get("type")
        if not isinstance(answer, dict) or answer.get("type") != qtype:
            raise DecisionError("malformed", f"question {qid!r}: missing or wrong type")
        value = answer.get(_VALUE_FIELD.get(str(qtype), ""))
        confidence = answer.get("answer_confidence")
        if not _probability(confidence):
            raise DecisionError("malformed", f"question {qid!r}: bad answer_confidence")
        if qtype == "choice":
            if not isinstance(value, str) or value not in (question.get("criteria") or {}):
                raise DecisionError("malformed", f"question {qid!r}: unknown label {value!r}")
        elif qtype == "noul":
            if not _probability(value):
                raise DecisionError("malformed", f"question {qid!r}: bad noul")
        elif not isinstance(value, int | float) or not math.isfinite(value):
            raise DecisionError("malformed", f"question {qid!r}: bad score")
        return Decision(value=value, confidence=confidence, raw=answer)
