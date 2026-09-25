"""Simulated call producer: synthetic validate_contact calls, replayed to Kafka.

There are no real calls in this project, so the streaming path (Kafka ->
Spark Structured Streaming -> Bronze) is exercised with traces that follow
Flow 1's state machine: questions, partials and finals, a Laya decision per
turn, Qwen on value turns, readbacks, reprompts, writes, rejections and
barge-ins. Latencies are drawn around the design's budgets (Laya ~33 ms,
Qwen < 500 ms to complete JSON, end to end < 800 ms), so the dashboards and
the streaming job see realistic numbers before a real call exists.

Everything is synthetic, so text is included; `call_started.synthetic` says
so, and the lake can tell simulated calls from demo calls. The same seed
gives the same events. Calls overlap: arrivals are exponential, and the
merged stream interleaves them as concurrent calls would.

    python -m src.agent.telemetry.simulate --calls 50 --speed 10
    python -m src.agent.telemetry.simulate --calls 3 --stdout   # JSONL, no Kafka
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from src.agent.graphs.flow_validate_contact import (
    ASK,
    CONFIRM,
    DEFERRED,
    FIELDS,
    GREETING,
    REJECTED,
    REPROMPT,
    SAVED,
    readback,
    summary,
)
from src.agent.nodes.extract_entity import PROMPT as EXTRACT_PROMPT
from src.agent.telemetry.producer import (
    CallTrace,
    KafkaSink,
    MemorySink,
    TelemetrySink,
    topic_for,
)
from src.common.log import configure_logging, get_logger

FIRST = ["juan", "maria", "jose", "ana", "luis", "sofia", "carlos", "fernanda", "miguel", "valeria"]
LAST = ["garcia", "hernandez", "lopez", "martinez", "gonzalez", "perez", "ramirez", "flores"]
DOMAINS = ["gmail", "hotmail", "outlook", "yahoo"]
DIGITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
YES = ["sí", "sí, correcto", "así es", "ajá, sí", "correcto"]
NO = ["no", "no, está mal", "no, ese no es"]
REJECTIONS = ["ahorita no puedo, voy manejando", "márqueme después", "estoy en el trabajo"]

# Per-turn behaviour. Rough by design: these shape dashboards, not results.
P_REJECT = 0.03
P_EXTRACTED = {"name": 0.85, "phone": 0.75, "email": 0.6}
P_JSON_VALID = 0.97
P_CONFIRMED = 0.9
P_BARGE_IN = 0.08
MAX_REPROMPTS = 2

log = get_logger(__name__)


class _Clock:
    """Virtual time: the trace's timestamps, advanced by simulated durations."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += timedelta(milliseconds=ms)


class _Call:
    def __init__(self, rng: random.Random, start: datetime, model_version: str) -> None:
        self.rng = rng
        self.clock = _Clock(start)
        self.sink = MemorySink()
        call_id = str(self._uuid())
        self.trace = CallTrace(
            call_id, self.sink, include_text=True, clock=self.clock, new_id=self._uuid
        )
        self.turn = 0
        self.model_version = model_version

    def _uuid(self) -> uuid.UUID:
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)

    def _ms(self, median: float, spread: float, floor: float = 1.0) -> float:
        return round(max(floor, self.rng.lognormvariate(0, spread) * median), 1)

    def agent(self, text: str) -> None:
        self.trace.emit("turn_started", turn_idx=self.turn, speaker="agent")
        duration = self._ms(400 * len(text.split()), 0.1)
        if self.rng.random() < P_BARGE_IN:
            self.clock.advance(duration / 2)
            p_real = round(self.rng.random(), 3)
            self.trace.emit(
                "barge_in", turn_idx=self.turn, interrupted=p_real >= 0.4, p_real=p_real
            )
            self.clock.advance(duration / 2)
        else:
            self.clock.advance(duration)
        self.trace.emit("tts_done", turn_idx=self.turn, text=text, duration_ms=duration)
        self.turn += 1

    def person(self, text: str, *, intent: str, extract: tuple[str, str] | None = None) -> bool:
        """One caller turn through Laya (and Qwen on value turns). True if it parsed."""
        self.trace.emit("turn_started", turn_idx=self.turn, speaker="person")
        words = text.split()
        for k in range(1, len(words) + 1):
            self.clock.advance(self._ms(350, 0.2))
            self.trace.emit("asr_partial", turn_idx=self.turn, text=" ".join(words[:k]))
        endpointing = self._ms(300, 0.2)
        self.clock.advance(endpointing)
        confidence = round(self.rng.betavariate(8, 2), 3)
        self.trace.emit("asr_final", turn_idx=self.turn, text=text, asr_confidence=confidence)

        laya = self._ms(33, 0.15, floor=10)
        self.clock.advance(laya)
        output = {"intent": intent, "confidence": round(self.rng.uniform(0.6, 0.99), 3)}
        self.trace.emit(
            "decision", turn_idx=self.turn, task="classify_intent", output=output, latency_ms=laya
        )
        qwen, valid = 0.0, True
        if extract is not None:
            qwen = self._ms(320, 0.25, floor=80)
            valid = self.rng.random() < P_JSON_VALID
            self.clock.advance(qwen)
            self.trace.emit(
                "llm_response",
                turn_idx=self.turn,
                task="extract_entity",
                prompt_version=EXTRACT_PROMPT.version,
                prompt_sha256=EXTRACT_PROMPT.sha256,
                json_valid=valid,
                latency_ms=qwen,
                output={"field": extract[0], "normalized_value": extract[1] if valid else None},
            )
        # End of turn detected -> first agent audio: STT final, models, TTS first byte.
        end_to_end = round(self._ms(150, 0.2) + laya + qwen + self._ms(75, 0.2), 1)
        self.trace.emit(
            "latency", turn_idx=self.turn, end_to_end_ms=end_to_end, endpointing_ms=endpointing
        )
        self.turn += 1
        return valid

    def tool(self, name: str, field: str) -> None:
        self.trace.emit("tool_invoked", turn_idx=self.turn, tool=name, field=field)


def _spoken(field: str, rng: random.Random) -> tuple[str, str]:
    """(what the caller says, the value it stands for)."""
    first, last = rng.choice(FIRST), rng.choice(LAST)
    if field == "name":
        return f"{first} {last}", f"{first.title()} {last.title()}"
    if field == "phone":
        digits = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))
        return " ".join(DIGITS[int(d)] for d in digits), f"+52{digits}"
    domain = rng.choice(DOMAINS)
    return f"{first} punto {last} arroba {domain} punto com", f"{first}.{last}@{domain}.com"


def simulate_call(
    rng: random.Random, start: datetime, *, model_version: str = "Qwen/Qwen3-4B"
) -> list[dict[str, Any]]:
    """One validate_contact call, from call_started to call_ended."""
    call = _Call(rng, start, model_version)
    call.trace.emit(
        "call_started",
        flow="validate_contact",
        contact_id=f"sim-{call._uuid().hex[:12]}",
        model_version=model_version,
        prompt_hash=EXTRACT_PROMPT.sha256,
        synthetic=True,
    )
    results: dict[str, str] = {}
    pending = list(FIELDS)
    call.agent(f"{GREETING.format(company='')} {ASK[pending[0]]}")

    def end(outcome: str) -> list[dict[str, Any]]:
        call.trace.emit("call_ended", outcome=outcome, turns=call.turn, fields=results)
        return [event for _, _, event in call.sink.sent]

    while pending:
        field = pending.pop(0)
        failures = 0
        said, value = _spoken(field, rng)
        while field not in results:
            if rng.random() < P_REJECT:
                call.person(rng.choice(REJECTIONS), intent="call_rejected")
                call.agent(REJECTED)
                return end("rejected")
            parsed = call.person(said, intent="ambiguous", extract=(field, value))
            ok = parsed and rng.random() < P_EXTRACTED[field]
            if ok:
                call.agent(CONFIRM.format(value=readback(field, value)))
                confirmed = rng.random() < P_CONFIRMED
                call.person(
                    rng.choice(YES if confirmed else NO),
                    intent="confirmed" if confirmed else "ambiguous",
                )
                if confirmed:
                    call.tool("update_contact", field)
                    results[field] = "validated"
                    break
            failures += 1
            if failures > MAX_REPROMPTS:
                call.tool("mark_unvalidated", field)
                results[field] = "unvalidated"
            else:
                call.agent(REPROMPT[field])
        closing = SAVED if results[field] == "validated" else DEFERRED
        call.agent(f"{closing} {ASK[pending[0]]}" if pending else f"{closing} {summary(results)}")
    return end("completed" if all(r == "validated" for r in results.values()) else "partial")


def simulate(
    calls: int, *, seed: int, start: datetime, mean_arrival_s: float = 30.0
) -> list[dict[str, Any]]:
    """`calls` overlapping calls, merged into one stream ordered by time."""
    rng = random.Random(seed)
    events: list[dict[str, Any]] = []
    arrival = start
    for _ in range(calls):
        events.extend(simulate_call(rng, arrival))
        arrival += timedelta(seconds=rng.expovariate(1 / mean_arrival_s))
    return sorted(events, key=lambda e: (e["ts"], e["call_id"], e["seq"]))


def replay(
    events: list[dict[str, Any]],
    sink: TelemetrySink,
    *,
    speed: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Send in order; `speed` x real time, or as fast as possible when 0."""
    previous: datetime | None = None
    for event in events:
        ts = datetime.fromisoformat(event["ts"])
        if speed > 0 and previous is not None and ts > previous:
            sleep((ts - previous).total_seconds() / speed)
        previous = ts
        sink.send(topic_for(event["type"]), event["call_id"], event)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay simulated calls to Kafka.")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed", type=float, default=0.0, help="x real time; 0 = no pauses")
    parser.add_argument("--mean-arrival-s", type=float, default=30.0)
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:29092",
        help="from the host; containers use kafka:9092",
    )
    parser.add_argument("--stdout", action="store_true", help="print JSONL instead of producing")
    args = parser.parse_args(argv)

    configure_logging()
    events = simulate(
        args.calls, seed=args.seed, start=datetime.now(UTC), mean_arrival_s=args.mean_arrival_s
    )
    if args.stdout:
        memory = MemorySink()
        replay(events, memory, speed=args.speed)
        for _, _, event in memory.sent:
            sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        return
    kafka = KafkaSink(args.bootstrap_servers)
    replay(events, kafka, speed=args.speed)
    kafka.close()
    log.info("simulation_sent", calls=args.calls, events=len(events), dropped=kafka.dropped)


if __name__ == "__main__":
    main()
