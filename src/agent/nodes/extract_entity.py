"""LangGraph node: extract and normalize one contact field.

The node returns structured output only. It never calls a tool; the graph
decides transitions and performs writes.
"""

from __future__ import annotations

from typing import Any

from src.agent.llm.client import LLMClient
from src.agent.llm.structured import parse_or_ambiguous
from src.common.prompts import load_prompt

PROMPT = load_prompt("extract_entity", version=1)
CONFIDENCE_THRESHOLD = 0.75
MAX_REPROMPTS = 2


async def extract_entity(state: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    field_name = state["current_field"]

    result = await llm.structured(
        prompt=PROMPT,
        variables={
            "field": field_name,
            "transcript": state["last_transcript"],
            "asr_confidence": state["last_asr_confidence"],
        },
        schema_path=PROMPT.schema,
    )
    # An unparseable output is treated as ambiguous and retried, never crashes the call.
    result = parse_or_ambiguous(result)

    low_confidence = result["confidence"] < CONFIDENCE_THRESHOLD or result["needs_reprompt"]
    attempts = state["reprompts"].get(field_name, 0)

    return {
        **state,
        "extracted": {**state.get("extracted", {}), field_name: result},
        # After MAX_REPROMPTS the field is marked unvalidated and the flow moves on.
        # It does NOT escalate to a human and does NOT end the call.
        "next": "reprompt" if low_confidence and attempts < MAX_REPROMPTS
        else "mark_unvalidated" if low_confidence
        else "confirm",
    }
