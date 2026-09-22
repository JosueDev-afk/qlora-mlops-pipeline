"""Laya-multilingual decision model.

Three non-negotiables, taken from the model card and encoded here so they
cannot be forgotten at deploy time:

1. Always the MULTILINGUAL checkpoint. The English root collapses on non-Latin
   script while staying confident, so confidence gating cannot save you. For
   Spanish the multilingual checkpoint is the only correct choice.
2. Always preload. Without it the router rebuilds the checkpoint on every
   language switch (measured 7-10 s median). With preload, latency is ~33 ms.
3. Always calibrated. Raw probabilities are over-confident; the temperatures
   fitted by the `calibrate_laya` DAG must be applied before any thresholding.
"""

from __future__ import annotations

from src.agent.decision.base import Decision, DecisionModel

CHECKPOINT = "multilingual"  # never "english": see note 1
DEFAULT_LANGUAGE = "es"


class LayaDecisionModel(DecisionModel):
    def __init__(self, adapter_path: str, temperatures: dict[str, float], device: str = "cuda"):
        self._temperatures = temperatures
        self._adapter_path = adapter_path
        self._device = device
        self._router = None

    def load(self) -> None:
        """Preload the checkpoint. Call once at service start, never per request."""
        raise NotImplementedError

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
