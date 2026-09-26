"""CallSession: the seam between audio and turns, with scripted models.

What is pinned: turns reach the graph, partials reach Task D only while the
agent speaks, every Task D failure stops the agent, and the telemetry a
real call emits is redacted and valid against the event schema.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from src.agent.decision.base import Decision, DecisionError
from src.agent.decision.calibration import load_agent_calibration
from src.agent.demo.session import CallSession
from src.agent.llm.client import Completion
from src.agent.telemetry.producer import REDACTED, MemorySink

SCHEMA = json.loads(Path("schemas/events/call_event.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


class FakeDecision:
    """Answers intent questions from `intents`, interruption questions from `partials`."""

    def __init__(self, intents: list[Decision], partials: list[Any] | None = None) -> None:
        self.intents = list(intents)
        self.partials = list(partials or [])
        self.interruption_calls = 0

    async def decide(self, state: dict, questions: dict, *, language: str = "es") -> dict:
        if "intent" in questions:
            return {"intent": self.intents.pop(0)}
        self.interruption_calls += 1
        step = self.partials.pop(0)
        if isinstance(step, Exception):
            raise step
        p_real, kind = step
        return {"interruption": Decision(p_real, max(p_real, 1 - p_real)),
                "kind": Decision(kind, 0.9)}  # fmt: skip


class FakeLLM:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    async def structured(self, prompt: Any, variables: dict) -> Completion:
        return Completion(text=self.texts.pop(0), latency_ms=210.0)


def extraction(field: str, value: str) -> str:
    return json.dumps({"field": field, "normalized_value": value, "confidence": 0.95,
                       "needs_reprompt": False})  # fmt: skip


def session(intents: list[Decision], llm: list[str] = (), partials: list[Any] = ()) -> CallSession:
    return CallSession(
        call_id="call-7",
        decision=FakeDecision(intents, list(partials)),
        llm=FakeLLM(list(llm)),
        rejection_threshold=0.4,
        interruption_threshold=0.4,
        telemetry=MemorySink(),
    )


def events(s: CallSession) -> list[dict[str, Any]]:
    assert isinstance(s.telemetry, MemorySink)
    return [event for _, _, event in s.telemetry.sent]


def intent(label: str, p_rejected: float = 0.0) -> Decision:
    return Decision(label, 0.9, {"probabilities": {"call_rejected": p_rejected}})


def test_a_turn_runs_the_graph_and_reports_it() -> None:
    s = session([intent("ambiguous"), intent("confirmed")], llm=[extraction("name", "Ana Ruiz")])
    greeting = asyncio.run(s.start())
    assert "Clínica Demo" in greeting
    assert asyncio.run(s.on_final("ana ruiz", 0.93)) == "Tengo Ana Ruiz. ¿Es correcto?"
    assert asyncio.run(s.on_final("sí")).startswith("Perfecto, quedó registrado.")
    types = [e["type"] for e in events(s)]
    assert types[0] == "call_started"
    assert types.count("llm_response") == 1  # only the value turn reached Qwen
    tool = next(e for e in events(s) if e["type"] == "tool_invoked")
    assert tool["payload"] == {"turn_idx": 2, "tool": "update_contact", "field": "name"}


def test_real_call_telemetry_is_redacted_and_valid() -> None:
    s = session([intent("ambiguous")], llm=[extraction("name", "Ana Ruiz")])
    asyncio.run(s.start())
    asyncio.run(s.on_final("ana ruiz", 0.93))
    for event in events(s):
        assert VALIDATOR.is_valid(event), event
    final = next(e for e in events(s) if e["type"] == "asr_final")
    assert final["payload"]["text"] == REDACTED
    llm = next(e for e in events(s) if e["type"] == "llm_response")
    assert llm["payload"]["output"]["normalized_value"] == REDACTED


def test_rejection_ends_the_call_once() -> None:
    s = session([intent("ambiguous", p_rejected=0.9)])
    asyncio.run(s.start())
    assert "disculpe la molestia" in asyncio.run(s.on_final("voy manejando"))
    assert s.ended
    assert asyncio.run(s.on_final("¿bueno?")) == ""
    ended = [e for e in events(s) if e["type"] == "call_ended"]
    assert [e["payload"]["outcome"] for e in ended] == ["rejected"]


def test_partials_are_ignored_while_the_agent_is_silent() -> None:
    s = session([])
    assert asyncio.run(s.on_partial("ajá")) is False
    assert s.decision.interruption_calls == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("step", "stop"),
    [
        ((0.1, "backchannel"), False),  # "ajá": keep talking
        ((0.9, "correction"), True),  # "ese no es mi correo": stop
        ((0.45, "unclear"), True),  # unsure: P(real) above the < 0.5 threshold
        ((0.1, "rejection"), True),  # a rejection always stops
        ((0.1, "sarcasm"), True),  # invalid output: stop
        (DecisionError("timeout"), True),  # no answer: stop
    ],
)
def test_task_d_while_the_agent_speaks(step: Any, stop: bool) -> None:
    s = session([], partials=[step])
    s.agent_started("Tengo juan punto perez arroba gmail punto com. ¿Es correcto?")
    assert asyncio.run(s.on_partial("mhm")) is stop
    barge = next(e for e in events(s) if e["type"] == "barge_in")
    assert barge["payload"]["interrupted"] is stop


def test_after_playback_partials_go_back_to_being_ignored() -> None:
    s = session([], partials=[(0.1, "backchannel")])
    s.agent_started("¿Me dicta su correo?")
    asyncio.run(s.on_partial("ajá"))
    s.agent_stopped()
    assert asyncio.run(s.on_partial("juan")) is False


def test_interruption_threshold_must_stay_below_half() -> None:
    with pytest.raises(ValueError, match="below 0.5"):
        CallSession("c", FakeDecision([]), FakeLLM([]), 0.4, interruption_threshold=0.5)


# ── calibration file ───────────────────────────────────────────────────────


def test_agent_reads_run_id_and_threshold(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"run_id": "r-cal-1", "thresholds": {"call_rejected": 0.31}}))
    calibration = load_agent_calibration(path)
    assert (calibration.run_id, calibration.call_rejected_threshold) == ("r-cal-1", 0.31)


@pytest.mark.parametrize(
    "data",
    [{"thresholds": {"call_rejected": 0.3}}, {"run_id": "r", "thresholds": {}},
     {"run_id": "r", "thresholds": {"call_rejected": 1.0}}],
)  # fmt: skip
def test_unusable_calibration_fails_at_start_up(tmp_path: Path, data: dict) -> None:
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_agent_calibration(path)


def test_missing_calibration_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make calibrate"):
        load_agent_calibration(tmp_path / "missing.json")
