"""Provider-agnostic decision-model interface.

Laya is one implementation. Keeping the interface separate means the
architecture can fall back to the generative model for Tasks B and D if the
frozen eval set does not support Hypothesis 1 — without touching the graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """One typed answer with a calibrated probability."""

    value: str | bool | float
    confidence: float
    raw: dict


class DecisionModel(ABC):
    @abstractmethod
    async def decide(
        self,
        state: dict,
        questions: dict,
        *,
        language: str = "es",
    ) -> dict[str, Decision]:
        """Answer every question in a single forward pass.

        `questions` follows the typed-question schema: each entry declares a
        type ('choice', 'score', 'noul') plus instructions and criteria, so the
        answer space is defined at request time and needs no retraining.
        """

    @abstractmethod
    def apply_temperature(self, question_type: str, n_options: int) -> float:
        """Return the fitted temperature for this question shape.

        Laya ships over-confident: refitting one temperature per
        (question type, option count) moves mean ECE from ~0.31 to ~0.11 on the
        multilingual checkpoint. Calibration is mandatory before the
        probabilities are used to gate a decision — see `calibrate_laya` DAG.
        """
