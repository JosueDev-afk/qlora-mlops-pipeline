"""Call telemetry to Kafka: the source of the streaming path (Kafka -> Bronze).

Events follow `schemas/events/call_event.schema.json`, keyed by call id so
one call's events stay ordered within a partition. Three topics, as in the
Phase 2 design: transcripts, model inferences, and everything else.

Two rules shape this module:

- **Telemetry never breaks a call.** Producing is fire-and-forget: a full
  buffer, a broker that is down or an unserializable payload drops the event
  and logs it. A live call does not wait on Kafka.
- **Personal data stays out by default.** Transcripts, TTS text and
  extracted values carry names, phones and emails. They are redacted unless
  the trace says otherwise, which only the simulator (synthetic data) does.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from src.common.log import REDACTED, get_logger

SCHEMA_VERSION = 1
TOPICS = {
    "asr_partial": "calls.transcripts",
    "asr_final": "calls.transcripts",
    "decision": "model.inferences",
    "llm_response": "model.inferences",
}
DEFAULT_TOPIC = "calls.events"
# Payload keys that can hold what a caller said or what was extracted from it.
PERSONAL_KEYS = frozenset({"text", "normalized_value", "raw_span"})

log = get_logger(__name__)


def topic_for(event_type: str) -> str:
    return TOPICS.get(event_type, DEFAULT_TOPIC)


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace personal values, one level into nested outputs (`output.normalized_value`)."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if key in PERSONAL_KEYS and value is not None:
            clean[key] = REDACTED
        elif isinstance(value, dict):
            clean[key] = redact(value)
        else:
            clean[key] = value
    return clean


class TelemetrySink(Protocol):
    def send(self, topic: str, key: str, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


@dataclass
class MemorySink:
    """Keeps what was sent. For tests, and for dumping a trace as JSONL."""

    sent: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def send(self, topic: str, key: str, event: dict[str, Any]) -> None:
        self.sent.append((topic, key, event))

    def close(self) -> None:
        return None


class KafkaSink:
    """confluent-kafka producer; every failure drops the event instead of raising.

    `producer` is injectable so tests need neither a broker nor the library.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        *,
        flush_timeout_s: float = 5.0,
        producer: Any = None,
    ) -> None:
        if producer is None:
            from confluent_kafka import Producer

            producer = Producer({
                "bootstrap.servers": bootstrap_servers,
                "client.id": "voice-agent-telemetry",
                "linger.ms": 50,             # batch a turn's events, not one request each
                "enable.idempotence": True,  # a retry never duplicates an event
            })  # fmt: skip
        self._producer = producer
        self._flush_timeout_s = flush_timeout_s
        self.dropped = 0

    def _on_delivery(self, error: Any, message: Any) -> None:
        if error is not None:
            self.dropped += 1
            log.warning("telemetry_delivery_failed", error=str(error))

    def send(self, topic: str, key: str, event: dict[str, Any]) -> None:
        try:
            value = json.dumps(event, ensure_ascii=False).encode()
            self._producer.produce(topic, key=key, value=value, on_delivery=self._on_delivery)
            self._producer.poll(0)  # serve delivery callbacks without blocking
        except Exception as exc:  # noqa: BLE001 -- BufferError, KafkaException, TypeError
            self.dropped += 1
            log.warning("telemetry_dropped", topic=topic, type=event.get("type"), error=repr(exc))

    def close(self) -> None:
        remaining = self._producer.flush(self._flush_timeout_s)
        if remaining:
            self.dropped += remaining
            log.warning("telemetry_unflushed", events=remaining)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CallTrace:
    """Emits one call's events: envelope, sequence number, topic and redaction.

    `clock` and `new_id` are injectable so the simulator produces the same
    trace for the same seed.
    """

    def __init__(
        self,
        call_id: str,
        sink: TelemetrySink,
        *,
        include_text: bool = False,
        clock: Callable[[], datetime] = _utc_now,
        new_id: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        self.call_id = call_id
        self._sink = sink
        self._include_text = include_text
        self._clock = clock
        self._new_id = new_id
        self._seq = 0

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(self._new_id()),
            "call_id": self.call_id,
            "seq": self._seq,
            "ts": self._clock().isoformat(timespec="milliseconds"),
            "type": event_type,
            "payload": payload if self._include_text else redact(payload),
        }
        self._seq += 1
        self._sink.send(topic_for(event_type), self.call_id, event)
        return event
