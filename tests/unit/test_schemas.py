"""Every model output must validate against its schema.

An invalid output is treated as `ambiguous` and retried — it must never crash a call.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

SCHEMAS = Path("schemas")


@pytest.mark.parametrize("name", ["extract_entity", "classify_intent", "parse_datetime"])
def test_schema_is_valid(name: str) -> None:
    schema = json.loads((SCHEMAS / f"{name}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)


def test_extract_entity_accepts_valid_output() -> None:
    schema = json.loads((SCHEMAS / "extract_entity.schema.json").read_text())
    Draft202012Validator(schema).validate(
        {
            "field": "phone",
            "normalized_value": "+528112345678",
            "confidence": 0.91,
            "needs_reprompt": False,
        }
    )


def test_extract_entity_rejects_unknown_field() -> None:
    schema = json.loads((SCHEMAS / "extract_entity.schema.json").read_text())
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(
            {
                "field": "curp",  # not an allowed field
                "normalized_value": "x",
                "confidence": 0.5,
                "needs_reprompt": True,
            }
        )


def test_call_rejected_is_a_valid_intent() -> None:
    """The global rejection interruption depends on this value existing."""
    schema = json.loads((SCHEMAS / "classify_intent.schema.json").read_text())
    Draft202012Validator(schema).validate({"intent": "call_rejected", "confidence": 0.97})
