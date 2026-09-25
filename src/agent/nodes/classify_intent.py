"""Task B on Laya: the turn's intent, and the global rejection signal.

Runs first on every turn of both flows. `call_rejected` is decided on its own
calibrated probability against the threshold `calibrate_laya` derived for
the target recall, not on the argmax: a caller who says "voy manejando" must
be let go even when another label scores slightly higher. Any failure is
`ambiguous` (rule 4), which never ends a call.
"""

from __future__ import annotations

from typing import Any

from src.agent.decision.base import DecisionError, DecisionModel
from src.agent.llm.structured import ambiguous, parse_or_ambiguous
from src.common.log import get_logger
from src.common.prompts import load_prompt

PROMPT = load_prompt("classify_intent_laya", version=1)
TASK = "classify_intent"

log = get_logger(__name__)


async def classify_intent(
    decision: DecisionModel,
    *,
    flow: str,
    context: str,
    transcript: str,
    rejection_threshold: float,
) -> dict[str, Any]:
    """Return a value valid against `classify_intent.schema.json`."""
    if not 0.0 < rejection_threshold < 1.0:
        raise ValueError(f"rejection_threshold must be in (0, 1), got {rejection_threshold}")
    # Same labels as the Qwen template, so both H1 arms see the same state.
    state = {"flujo": flow, "contexto": context, "transcripción": transcript}
    try:
        answers = await decision.decide(state, PROMPT.body["questions"])
    except DecisionError as exc:
        log.warning("decision_failed", task=TASK, kind=exc.kind)
        return ambiguous(TASK)

    answer = answers["intent"]
    p_rejected = (answer.raw.get("probabilities") or {}).get("call_rejected")
    if isinstance(p_rejected, int | float) and p_rejected >= rejection_threshold:
        return parse_or_ambiguous({"intent": "call_rejected", "confidence": p_rejected}, TASK)
    return parse_or_ambiguous({"intent": answer.value, "confidence": answer.confidence}, TASK)
