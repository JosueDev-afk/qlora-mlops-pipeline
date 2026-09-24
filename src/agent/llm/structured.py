"""Validate model output and fall back to `ambiguous` (rule 4).

A parse failure must never crash a live call, so every failure becomes the
task's own notion of "ambiguous", which the graph already knows how to handle
(reprompt, reframe). What that means differs per task:

- classify_intent: `intent: "ambiguous"`, so the graph reframes the question
- extract_entity:  `needs_reprompt: true` for the same field, so it asks again
- parse_datetime:  `precision: "unresolved"`, so nothing is ever rescheduled

Retrying is the graph's job, not this module's. JSON wrapped in prose or code
fences is rejected rather than repaired: the JSON-validity KPI (>= 99.5 %) has
to measure what the model actually emits.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import best_match

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

# jsonschema ignores `format` unless told otherwise. `reschedule` must never get
# a naive or malformed datetime, so date-time is checked, offset required.
_FORMATS = FormatChecker(formats=())


@_FORMATS.checks("date-time", raises=ValueError)
def _is_offset_datetime(value: object) -> bool:
    return not isinstance(value, str) or datetime.fromisoformat(value).tzinfo is not None


def _extract_entity(field: str | None = None) -> dict[str, Any]:
    if field is None:
        raise ValueError("extract_entity needs the field being captured (field=...)")
    return {
        "field": field,
        "normalized_value": None,
        "confidence": 0.0,
        "needs_reprompt": True,
        "reprompt_reason": "low_confidence",
    }


_FALLBACKS: dict[str, Callable[..., dict[str, Any]]] = {
    "classify_intent": lambda: {"intent": "ambiguous", "confidence": 0.0},
    "extract_entity": _extract_entity,
    "parse_datetime": lambda: {"datetime_iso": None, "precision": "unresolved", "confidence": 0.0},
}


@dataclass(frozen=True)
class Parsed:
    """The value the graph consumes, plus whether the model produced it.

    `valid` feeds `fact_turn.json_valid`; `error` says why it was replaced.
    """

    value: dict[str, Any]
    valid: bool
    error: str | None = None


@cache
def _validator(task: str) -> Draft202012Validator:
    schema = json.loads((SCHEMAS_DIR / f"{task}.schema.json").read_text())
    return Draft202012Validator(schema, format_checker=_FORMATS)


def ambiguous(task: str, **context: Any) -> dict[str, Any]:
    """The task's safe fallback. Unknown tasks are programming errors, so they raise."""
    if task not in _FALLBACKS:
        raise ValueError(f"no ambiguous fallback for task {task!r}")
    return _FALLBACKS[task](**context)


def parse_structured(raw: object, task: str, **context: Any) -> Parsed:
    """Accept a dict or JSON text; anything else, or anything invalid, is ambiguous.

    The fallback is built first so a missing context argument fails in every
    call, including the happy path, instead of only when the model misbehaves.
    """
    fallback = ambiguous(task, **context)

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return Parsed(fallback, valid=False, error=f"not JSON: {exc.msg}")
    if not isinstance(raw, dict):
        return Parsed(fallback, valid=False, error=f"expected an object, got {type(raw).__name__}")

    error = best_match(_validator(task).iter_errors(raw))
    if error is not None:
        return Parsed(fallback, valid=False, error=error.message)

    if task == "extract_entity" and raw["field"] != context["field"]:
        return Parsed(
            fallback,
            valid=False,
            error=f"field mismatch: expected {context['field']!r}, got {raw['field']!r}",
        )
    return Parsed(raw, valid=True)


def parse_or_ambiguous(raw: object, task: str, **context: Any) -> dict[str, Any]:
    """Shortcut for nodes that only need the value."""
    return parse_structured(raw, task, **context).value
