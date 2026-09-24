"""Logs are structured JSON in English and never carry what the caller dictated.

In this domain a transcript *is* personal data: names, phone numbers and email
addresses spoken out loud. Logs flow to Kafka and the lake, so those fields are
redacted before anything is rendered.
"""

import io
import json

import pytest

from src.common.log import REDACTED, configure_logging, get_logger


def _emit(level: str = "INFO", **fields: object) -> list[dict]:
    stream = io.StringIO()
    configure_logging(level, json=True, stream=stream)
    get_logger("test", flow="validate_contact").info("field_extracted", **fields)
    get_logger("test").debug("noise")
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_emits_one_json_object_per_event_with_bound_context() -> None:
    (record,) = _emit(field="email")
    assert record["event"] == "field_extracted"
    assert record["level"] == "info"
    assert record["flow"] == "validate_contact"
    assert record["field"] == "email"
    assert "timestamp" in record


@pytest.mark.parametrize(
    "key",
    [
        "transcript",
        "asr_partial",
        "utterance",
        "contact_name",
        "email",
        "phone",
        "normalized_value",
    ],
)
def test_personal_data_is_redacted(key: str) -> None:
    (record,) = _emit(**{key: "jota u a ene arroba gmail punto com"})
    assert record[key] == REDACTED


def test_non_personal_fields_pass_through() -> None:
    (record,) = _emit(confidence=0.91, reprompts=1)
    assert record["confidence"] == 0.91
    assert record["reprompts"] == 1


def test_level_filters_debug_at_info() -> None:
    records = _emit("INFO")
    assert [r["event"] for r in records] == ["field_extracted"]


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="VERBOSE"):
        configure_logging("VERBOSE", json=True, stream=io.StringIO())
