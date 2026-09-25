"""Provider-agnostic decision-model interface.

Laya is one implementation. Keeping the interface separate means the
architecture can fall back to the generative model for Tasks B and D if the
frozen eval set does not support Hypothesis 1 — without touching the graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Decision:
    """One typed answer with a calibrated probability.

    `value` is the label for `choice`, the expected level for `score` and
    P(true) for `noul`, because a noul is gated on its probability, never on
    a boolean. `confidence` is the probability of the answer given, which is
    what temperature scaling fits and ECE measures.
    """

    value: str | float
    confidence: float
    raw: dict[str, Any] = field(default_factory=dict)


class DecisionError(Exception):
    """Any failure to get a calibrated answer: transport, protocol or calibration.

    One exception type, so each caller maps every failure onto its own
    fallback: `ambiguous` for Task B, an interruption for Task D (rule 4).
    """

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


class DecisionModel(ABC):
    @abstractmethod
    async def decide(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        *,
        language: str = "es",
    ) -> dict[str, Decision]:
        """Answer every question in a single forward pass.

        `questions` follows the typed-question schema: each entry declares a
        type ('choice', 'score', 'noul') plus instructions and criteria, so the
        answer space is defined at request time and needs no retraining.

        Every returned confidence must already be calibrated: Laya ships
        over-confident, and a threshold on raw probabilities has no meaning
        (H6). Raises DecisionError instead of returning an uncalibrated answer.
        """
