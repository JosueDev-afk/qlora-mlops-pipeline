"""Rule 4: invalid model output is `ambiguous`, never an exception.

Each task has its own meaning of "ambiguous", and the fallback itself must be
valid against the task's schema, or the graph would crash on the very value
meant to keep the call alive.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.agent.llm.structured import SCHEMAS_DIR, ambiguous, parse_or_ambiguous, parse_structured

# Task D resolves failures toward interrupting instead (src/agent/decision/interruption.py).
TASKS = sorted(
    p.name.removesuffix(".schema.json")
    for p in SCHEMAS_DIR.glob("*.schema.json")
    if not p.name.startswith("is_real_interruption")
)
CONTEXT = {"extract_entity": {"field": "phone"}}

VALID = {
    "extract_entity": {
        "field": "phone",
        "normalized_value": "+528112345678",
        "confidence": 0.93,
        "needs_reprompt": False,
    },
    "classify_intent": {"intent": "confirmed", "confidence": 0.97},
    "parse_datetime": {
        "datetime_iso": "2026-10-01T16:00:00-06:00",
        "precision": "exact",
        "confidence": 0.88,
    },
}


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_a_fallback_valid_against_its_schema(task: str) -> None:
    schema = json.loads((SCHEMAS_DIR / f"{task}.schema.json").read_text())
    Draft202012Validator(schema).validate(ambiguous(task, **CONTEXT.get(task, {})))


@pytest.mark.parametrize("task", TASKS)
def test_valid_output_passes_through_as_dict_or_json_text(task: str) -> None:
    ctx = CONTEXT.get(task, {})
    for raw in (VALID[task], json.dumps(VALID[task])):
        parsed = parse_structured(raw, task, **ctx)
        assert parsed.valid and parsed.error is None
        assert parsed.value == VALID[task]


@pytest.mark.parametrize(
    "raw",
    [
        None,  # timeout or transport error
        "Claro, el teléfono es 81 1234 5678",  # prose instead of JSON
        '```json\n{"intent": "confirmed", "confidence": 0.9}\n```',  # fenced: not accepted
        ["confirmed"],
        {"intent": "confirmed"},  # missing confidence
        {"intent": "maybe", "confidence": 0.5},  # outside the enum
        {"intent": "confirmed", "confidence": 0.9, "extra": True},
    ],
)
def test_invalid_intent_output_becomes_ambiguous(raw: object) -> None:
    parsed = parse_structured(raw, "classify_intent")
    assert not parsed.valid and parsed.error
    assert parsed.value == {"intent": "ambiguous", "confidence": 0.0}


def test_failed_extraction_asks_again_for_the_same_field() -> None:
    value = parse_or_ambiguous("not json", "extract_entity", field="email")
    assert value["field"] == "email"
    assert value["needs_reprompt"] is True
    assert value["normalized_value"] is None


def test_extraction_for_another_field_is_rejected() -> None:
    """Writing an email into the phone field is worse than asking again."""
    raw = {**VALID["extract_entity"], "field": "email"}
    parsed = parse_structured(raw, "extract_entity", field="phone")
    assert not parsed.valid and "field" in parsed.error
    assert parsed.value["field"] == "phone"


@pytest.mark.parametrize("iso", ["2026-10-01T16:00:00", "mañana a las 4", "2026-13-01T16:00:00Z"])
def test_datetime_without_offset_or_malformed_is_ambiguous(iso: str) -> None:
    """reschedule must never receive a naive or malformed datetime."""
    raw = {**VALID["parse_datetime"], "datetime_iso": iso}
    parsed = parse_structured(raw, "parse_datetime")
    assert not parsed.valid
    assert parsed.value["precision"] == "unresolved"


def test_unknown_task_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="no_such_task"):
        parse_structured({}, "no_such_task")


def test_extraction_requires_the_field_being_captured() -> None:
    with pytest.raises(ValueError, match="field"):
        parse_structured(VALID["extract_entity"], "extract_entity")


def test_schemas_dir_is_the_repository_one() -> None:
    expected = Path(__file__).resolve().parents[2] / "schemas"
    assert expected == SCHEMAS_DIR
