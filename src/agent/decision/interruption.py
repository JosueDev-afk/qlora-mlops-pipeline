"""Resolve Task D (real interruption vs. backchannel) toward stopping the TTS.

Task D runs on ASR partials while the agent is speaking, so rule 4's "treat as
ambiguous and retry" does not apply: the next partial arrives in ~100-200 ms and
is the retry, and a boolean has no `ambiguous` value. Every failure therefore
resolves to an interruption. Stopping on a backchannel costs a short pause;
talking over someone who says "ese no es mi correo" or "voy manejando" costs the
call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "is_real_interruption.schema.json"
_VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))

# A failure carries no information, so its confidence is maximal uncertainty
# (0.5). Because the threshold is always below 0.5, that resolves to interrupting
# without the fallback claiming a certainty it does not have.
FALLBACK: dict[str, Any] = {"interruption": True, "confidence": 0.5, "kind": "unclear"}


def parse_interruption(raw: object) -> dict[str, Any]:
    """Return `raw` if it satisfies the schema, else the interrupting fallback.

    `raw` is None on timeout or transport error; that is a failure like any
    other and must not keep the agent talking.
    """
    if isinstance(raw, dict) and _VALIDATOR.is_valid(raw):
        return raw
    return dict(FALLBACK)


def should_interrupt(result: dict[str, Any], threshold: float) -> bool:
    """Decide on the probability that the speech is a real interruption.

    `confidence` is the calibrated confidence of the answer given, so
    P(real) is `confidence` when `interruption` is true and `1 - confidence`
    otherwise. The threshold must stay below 0.5: at or above it, an unsure
    model would keep talking, which inverts the policy.
    """
    if threshold >= 0.5:
        raise ValueError(f"interruption threshold must be < 0.5, got {threshold}")
    if result.get("kind") == "rejection":
        return True
    confidence = result["confidence"]
    p_real = confidence if result["interruption"] else 1.0 - confidence
    return p_real >= threshold
