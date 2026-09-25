"""Call telemetry: the event contract, privacy by default, and never breaking a call.

Every simulated event is validated against the same JSON Schema the
streaming job reads, so the simulator cannot drift from the contract. Kafka
is a fake object: what is pinned is how the sink uses it and survives it.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.agent.telemetry.producer import (
    REDACTED,
    CallTrace,
    KafkaSink,
    MemorySink,
    redact,
    topic_for,
)
from src.agent.telemetry.simulate import main, replay, simulate, simulate_call

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas/events/call_event.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
START = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def stream() -> list[dict[str, Any]]:
    return simulate(40, seed=3, start=START, mean_arrival_s=20)


# ── contract ───────────────────────────────────────────────────────────────


def test_every_simulated_event_matches_the_schema(stream: list[dict[str, Any]]) -> None:
    for event in stream:
        errors = [e.message for e in VALIDATOR.iter_errors(event)]
        assert not errors, (event["type"], errors)


def test_schema_rejects_an_unknown_type_and_a_missing_payload_field() -> None:
    trace = CallTrace("c-1", MemorySink())
    event = trace.emit("latency", turn_idx=0, end_to_end_ms=640.0, endpointing_ms=280.0)
    assert VALIDATOR.is_valid(event)
    assert not VALIDATOR.is_valid({**event, "type": "vibes"})
    assert not VALIDATOR.is_valid({**event, "payload": {"turn_idx": 0, "end_to_end_ms": 640.0}})


def test_each_call_is_complete_and_ordered(stream: list[dict[str, Any]]) -> None:
    by_call: dict[str, list[dict[str, Any]]] = {}
    for event in stream:
        by_call.setdefault(event["call_id"], []).append(event)
    assert len(by_call) == 40
    for events in by_call.values():
        assert events[0]["type"] == "call_started" and events[-1]["type"] == "call_ended"
        assert [e["seq"] for e in events] == list(range(len(events)))


def test_calls_overlap_like_concurrent_calls(stream: list[dict[str, Any]]) -> None:
    order = [e["call_id"] for e in stream]
    switches = sum(a != b for a, b in zip(order, order[1:], strict=False))
    assert switches > 40  # interleaved, not one call after another


def test_simulation_follows_flow_1(stream: list[dict[str, Any]]) -> None:
    ended = [e["payload"] for e in stream if e["type"] == "call_ended"]
    outcomes = {p["outcome"] for p in ended}
    assert {"completed", "partial"} <= outcomes
    for event in stream:
        if event["type"] == "tool_invoked":
            assert event["payload"]["tool"] in {"update_contact", "mark_unvalidated"}
    # Laya runs on every person turn; Qwen only on value turns, never on a rejection.
    person_turns = sum(
        1 for e in stream if e["type"] == "turn_started" and e["payload"]["speaker"] == "person"
    )
    assert sum(1 for e in stream if e["type"] == "decision") == person_turns
    rejected = {e["call_id"] for e in stream if e["type"] == "call_ended"
                and e["payload"]["outcome"] == "rejected"}  # fmt: skip
    assert rejected
    for call_id in rejected:
        events = [e for e in stream if e["call_id"] == call_id]
        decisions = [e for e in events if e["type"] == "decision"]
        assert decisions[-1]["payload"]["output"]["intent"] == "call_rejected"
        assert all(e["type"] != "llm_response" for e in events if e["seq"] > decisions[-1]["seq"])


def test_latencies_sit_around_the_design_budgets(stream: list[dict[str, Any]]) -> None:
    def p50(values: list[float]) -> float:
        return sorted(values)[len(values) // 2]

    laya = [e["payload"]["latency_ms"] for e in stream if e["type"] == "decision"]
    qwen = [e["payload"]["latency_ms"] for e in stream if e["type"] == "llm_response"]
    e2e = [e["payload"]["end_to_end_ms"] for e in stream if e["type"] == "latency"]
    assert 25 < p50(laya) < 45
    assert 250 < p50(qwen) < 400
    assert p50(e2e) < 800


def test_same_seed_same_stream() -> None:
    assert simulate(3, seed=11, start=START) == simulate(3, seed=11, start=START)
    assert simulate(3, seed=11, start=START) != simulate(3, seed=12, start=START)


def test_simulated_calls_say_they_are_synthetic_and_keep_text() -> None:
    import random

    events = simulate_call(random.Random(0), START)
    assert events[0]["payload"]["synthetic"] is True
    finals = [e for e in events if e["type"] == "asr_final"]
    assert finals and all(e["payload"]["text"] != REDACTED for e in finals)


# ── topics and privacy ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("event_type", "topic"),
    [
        ("asr_partial", "calls.transcripts"),
        ("asr_final", "calls.transcripts"),
        ("decision", "model.inferences"),
        ("llm_response", "model.inferences"),
        ("tool_invoked", "calls.events"),
        ("call_ended", "calls.events"),
    ],
)
def test_topics_match_bootstrap(event_type: str, topic: str) -> None:
    assert topic_for(event_type) == topic
    bootstrap = (Path(__file__).resolve().parents[2] / "infra/init/bootstrap.sh").read_text()
    assert topic in bootstrap


def test_live_traces_redact_personal_values_by_default() -> None:
    sink = MemorySink()
    trace = CallTrace("c-1", sink)
    trace.emit("asr_final", turn_idx=1, text="juan punto perez arroba gmail", asr_confidence=0.9)
    trace.emit("llm_response", turn_idx=1, task="extract_entity", prompt_version=1,
               prompt_sha256="x", json_valid=True, latency_ms=300.0,
               output={"field": "email", "normalized_value": "juan.perez@gmail.com",
                       "confidence": 0.9})  # fmt: skip
    (_, key, final), (topic, _, inference) = sink.sent
    assert key == "c-1"
    assert final["payload"]["text"] == REDACTED
    assert final["payload"]["asr_confidence"] == 0.9  # metrics survive
    assert topic == "model.inferences"
    assert inference["payload"]["output"] == {
        "field": "email", "normalized_value": REDACTED, "confidence": 0.9,
    }  # fmt: skip


def test_redact_keeps_nulls_so_missing_values_stay_visible() -> None:
    assert redact({"normalized_value": None, "text": "x"}) == {
        "normalized_value": None,
        "text": REDACTED,
    }


# ── kafka sink ─────────────────────────────────────────────────────────────


class FakeProducer:
    def __init__(self, fail: Exception | None = None, unflushed: int = 0) -> None:
        self.fail = fail
        self.unflushed = unflushed
        self.produced: list[dict[str, Any]] = []
        self.polls = 0

    def produce(self, topic: str, *, key: str, value: bytes, on_delivery: Any) -> None:
        if self.fail:
            raise self.fail
        self.produced.append({"topic": topic, "key": key, "value": json.loads(value)})
        self.on_delivery = on_delivery

    def poll(self, timeout: float) -> int:
        self.polls += 1
        return 0

    def flush(self, timeout: float) -> int:
        return self.unflushed


def test_kafka_sink_keys_by_call_and_polls_without_blocking() -> None:
    producer = FakeProducer()
    sink = KafkaSink("unused:9092", producer=producer)
    CallTrace("c-9", sink).emit("turn_started", turn_idx=0, speaker="agent")
    (message,) = producer.produced
    assert message["topic"] == "calls.events" and message["key"] == "c-9"
    assert message["value"]["type"] == "turn_started"
    assert producer.polls == 1


@pytest.mark.parametrize("failure", [BufferError("queue full"), RuntimeError("broker down")])
def test_kafka_failures_drop_the_event_and_never_raise(failure: Exception) -> None:
    sink = KafkaSink("unused:9092", producer=FakeProducer(fail=failure))
    CallTrace("c-9", sink).emit("turn_started", turn_idx=0, speaker="agent")
    assert sink.dropped == 1


def test_failed_deliveries_and_unflushed_events_are_counted() -> None:
    producer = FakeProducer(unflushed=2)
    sink = KafkaSink("unused:9092", producer=producer)
    CallTrace("c-9", sink).emit("turn_started", turn_idx=0, speaker="agent")
    producer.on_delivery("timed out", None)
    sink.close()
    assert sink.dropped == 3


# ── replay ─────────────────────────────────────────────────────────────────


def test_replay_paces_by_timestamps_at_the_given_speed() -> None:
    events = simulate(1, seed=5, start=START)
    pauses: list[float] = []
    sink = MemorySink()
    replay(events, sink, speed=10, sleep=pauses.append)
    span = (
        datetime.fromisoformat(events[-1]["ts"]) - datetime.fromisoformat(events[0]["ts"])
    ).total_seconds()
    assert sum(pauses) == pytest.approx(span / 10, rel=1e-6)
    assert [e for _, _, e in sink.sent] == events


def test_replay_at_speed_zero_never_sleeps() -> None:
    pauses: list[float] = []
    replay(simulate(2, seed=5, start=START), MemorySink(), speed=0, sleep=pauses.append)
    assert pauses == []


def test_cli_stdout_prints_jsonl(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--calls", "2", "--stdout"])
    lines = capsys.readouterr().out.splitlines()
    assert lines and all(VALIDATOR.is_valid(json.loads(line)) for line in lines)
