"""Structured logging shared by the pipeline and the agent.

One JSON object per event, event names in English (rule 1), so logs can be
shipped to Kafka and queried in the lake without parsing prose.

Personal data is redacted before rendering. In this domain a transcript *is*
personal data: callers dictate names, phone numbers and email addresses, and
the LFPDPPP asks for minimization. Redaction by field name is a floor, not a
guarantee: never put spoken text in the event name or in an unlisted field.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO

import structlog

REDACTED = "[redacted]"
PERSONAL_FIELDS = frozenset(
    {"transcript", "asr_partial", "utterance", "contact_name", "email", "phone", "normalized_value"}
)


def redact_personal_data(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in PERSONAL_FIELDS & event_dict.keys():
        event_dict[key] = REDACTED
    return event_dict


def configure_logging(
    level: str = "INFO", *, json: bool | None = None, stream: TextIO | None = None
) -> None:
    """Configure structlog once per process.

    `json` defaults to True when the stream is not a terminal, so services and
    Airflow tasks emit JSON while a developer at a TTY gets readable output.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"unknown log level {level!r}")
    out = stream if stream is not None else sys.stdout
    use_json = json if json is not None else not out.isatty()
    renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_personal_data,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=out),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None, **context: Any) -> Any:
    """Return a logger with `context` bound to every event it emits."""
    return structlog.get_logger(name).bind(**context)
