"""Task A on Qwen: extract and normalize one contact field from a turn.

The node returns structured output only. It never calls a tool; the graph
decides transitions and performs writes (rule 3).

Invalid output is `ambiguous` (rule 4), and the retry is asking the caller
again, never calling the model again: decoding is greedy, so the same input
returns the same invalid output, and a second call would spend another
500 ms of the turn's budget.
"""

from __future__ import annotations

import re
from typing import Any

from src.agent.llm.client import LLMClient
from src.agent.llm.structured import Parsed, parse_structured
from src.common.normalizers import EMAIL_RE, normalize_name
from src.common.prompts import load_prompt

PROMPT = load_prompt("extract_entity", version=1)
TASK = "extract_entity"
PHONE_RE = re.compile(r"^\+52[2-9]\d{9}$")


async def extract_entity(
    llm: LLMClient, *, field: str, transcript: str, asr_confidence: float | None
) -> tuple[Parsed, dict[str, Any]]:
    """Return the parsed extraction and what telemetry records about the call."""
    completion = await llm.structured(
        PROMPT,
        {"field": field, "transcript": transcript, "asr_confidence": asr_confidence},
    )
    parsed = parse_structured(completion.text, TASK, field=field)
    telemetry = {
        "task": TASK,
        "prompt_version": PROMPT.version,
        "prompt_sha256": PROMPT.sha256,
        "json_valid": parsed.valid,
        "latency_ms": completion.latency_ms,
        "error": completion.error or parsed.error,
    }
    return parsed, telemetry


def canonical(field: str, value: str | None) -> str | None:
    """The value to write, or None when it is not a well-formed value for `field`.

    The schema only says "string"; a model can still return "+52 81..." or a
    half-spelled address with high confidence. Nothing malformed reaches
    update_contact, whatever the model's confidence.
    """
    if not value:
        return None
    if field == "phone":
        return value if PHONE_RE.match(value) else None
    if field == "email":
        address = value.strip().lower()
        return address if EMAIL_RE.match(address) else None
    if field == "name":
        return normalize_name(value)
    raise ValueError(f"unknown field {field!r}")


def acceptable(field: str, result: dict[str, Any], threshold: float) -> bool:
    """check_confidence: confident, not flagged by the model, and well-formed."""
    return (
        result["confidence"] >= threshold
        and not result["needs_reprompt"]
        and canonical(field, result["normalized_value"]) is not None
    )
