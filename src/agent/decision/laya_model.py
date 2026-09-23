"""Laya-multilingual decision model, reached over HTTP.

The model runs in `src/serving/`, which owns two of the model card's rules:
always the multilingual checkpoint (the English root collapses on non-Latin
script while staying confident) and always preloaded (without it the router
rebuilds the checkpoint on every language switch, 7-10 s). Keeping the model
out of process keeps `src/agent/` free of torch.

This client owns the third rule: the server returns raw probabilities and the
temperatures fitted by `calibrate_laya` are applied here, before anything is
compared against a threshold.
"""

from __future__ import annotations

from src.agent.decision.base import Decision, DecisionModel

DEFAULT_LANGUAGE = "es"


class LayaDecisionModel(DecisionModel):
    def __init__(self, base_url: str, temperatures: dict[str, float], timeout_s: float):
        self._base_url = base_url
        self._temperatures = temperatures
        self._timeout_s = timeout_s

    async def decide(
        self,
        state: dict,
        questions: dict,
        *,
        language: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Decision]:
        raise NotImplementedError

    def apply_temperature(self, question_type: str, n_options: int) -> float:
        return self._temperatures.get(f"{question_type}:{n_options}", 1.0)
