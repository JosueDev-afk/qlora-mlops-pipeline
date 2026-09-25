"""Flow 1 transitions, turn by turn, with scripted models and an in-memory store.

Each test is a conversation: the graph runs once per caller turn with the
real LangGraph checkpointer, and the assertions are on what the agent says
and what gets written. The models are fakes, so what is pinned here is the
state machine, not model quality.
"""

import asyncio
import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.decision.base import Decision, DecisionError
from src.agent.graphs.flow_validate_contact import (
    ASK,
    DEFERRED,
    PURPOSE,
    REJECTED,
    REPROMPT,
    FlowDeps,
    build_graph,
    keyterms,
    readback,
    run_turn,
    start_call,
    summary,
)
from src.agent.llm.client import Completion
from src.agent.tools.contact import InMemoryContactStore

CONTACT = {"id": "c-1", "name": "Juan Pérez", "phone": "+528100000000",
           "email": "juan@gmail.com", "company_name": "Clínica Norte"}  # fmt: skip
THRESHOLD = 0.35


def intent(label: str, confidence: float = 0.9, p_rejected: float = 0.0) -> Decision:
    return Decision(label, confidence, {"probabilities": {"call_rejected": p_rejected}})


def extraction(field: str, value: str | None, confidence: float = 0.95) -> str:
    return json.dumps({"field": field, "normalized_value": value, "confidence": confidence,
                       "needs_reprompt": value is None})  # fmt: skip


class ScriptedDecision:
    def __init__(self, script: list[Decision | Exception]) -> None:
        self.script = list(script)
        self.states: list[dict] = []

    async def decide(self, state: dict, questions: dict, *, language: str = "es") -> dict:
        self.states.append(state)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"intent": step}


class ScriptedLLM:
    def __init__(self, script: list[str | None]) -> None:
        self.script = list(script)
        self.calls = 0

    async def structured(self, prompt: Any, variables: dict[str, Any]) -> Completion:
        self.calls += 1
        text = self.script.pop(0)
        return Completion(text=text, latency_ms=12.0, error=None if text else "timeout")


class Call:
    """One call against the compiled graph, turn by turn."""

    def __init__(self, intents: list, extractions: list, **deps: Any) -> None:
        self.decision = ScriptedDecision(intents)
        self.llm = ScriptedLLM(extractions)
        self.store = InMemoryContactStore()
        flow = FlowDeps(self.decision, self.llm, self.store, THRESHOLD, **deps)
        self.app = build_graph(flow, checkpointer=InMemorySaver())
        self.state = asyncio.run(start_call(self.app, "call-1", CONTACT))

    def turn(self, transcript: str) -> str:
        self.state = asyncio.run(run_turn(self.app, "call-1", transcript, 0.9))
        return self.state["say"]


def test_happy_path_confirms_and_writes_every_field() -> None:
    call = Call(
        intents=[intent("ambiguous"), intent("confirmed")] * 3,
        extractions=[
            extraction("name", "Juan Pérez"),
            extraction("phone", "+528182345678"),
            extraction("email", "juan.perez@gmail.com"),
        ],
    )
    assert call.state["say"].startswith("Buen día, le llamo de Clínica Norte")
    assert call.state["say"].endswith(ASK["name"])

    assert call.turn("juan pérez") == "Tengo Juan Pérez. ¿Es correcto?"
    assert call.turn("sí").endswith(ASK["phone"])
    assert call.turn("ochenta y uno...") == "Tengo 81 82 34 56 78. ¿Es correcto?"
    assert call.turn("sí").endswith(ASK["email"])
    assert (
        call.turn("juan punto perez...")
        == "Tengo juan punto perez arroba gmail punto com. ¿Es correcto?"
    )
    closing = call.turn("correcto")

    assert "quedaron confirmados su nombre, su teléfono y su correo" in closing
    assert call.state["outcome"] == "completed"
    assert call.store.writes == [
        ("update_contact", "c-1", "name", "Juan Pérez"),
        ("update_contact", "c-1", "phone", "+528182345678"),
        ("update_contact", "c-1", "email", "juan.perez@gmail.com"),
    ]


def test_rejection_ends_the_call_on_its_probability_not_the_argmax() -> None:
    call = Call(intents=[intent("ambiguous", 0.6, p_rejected=0.4)], extractions=[])
    assert call.turn("ahorita voy manejando") == REJECTED
    assert call.state["outcome"] == "rejected"
    assert call.store.writes == [] and call.llm.calls == 0  # Laya first: Qwen never ran


def test_rejection_below_threshold_does_not_end_the_call() -> None:
    call = Call(intents=[intent("ambiguous", 0.6, p_rejected=0.3)],
                extractions=[extraction("name", "Juan Pérez")])  # fmt: skip
    assert call.turn("juan pérez").startswith("Tengo Juan Pérez")


def test_two_failed_reprompts_mark_unvalidated_and_move_on() -> None:
    call = Call(intents=[intent("ambiguous")] * 3, extractions=[extraction("name", None)] * 3)
    assert call.turn("mmm") == REPROMPT["name"]
    assert call.turn("este...") == REPROMPT["name"]
    third = call.turn("no sé")
    assert third == f"{DEFERRED} {ASK['phone']}"  # the call continues with the next field
    assert call.store.writes == [("mark_unvalidated", "c-1", "name", None)]
    assert call.state["outcome"] is None


def test_invalid_json_is_a_failed_try_not_a_crash() -> None:
    call = Call(intents=[intent("ambiguous")] * 2,
                extractions=["Claro, el nombre es Juan", None])  # fmt: skip
    assert call.turn("juan") == REPROMPT["name"]
    assert call.state["last_model_call"]["json_valid"] is False
    assert call.turn("juan") == REPROMPT["name"]  # a timeout reads the same way
    assert call.llm.calls == 2  # one model call per turn: no silent retry


def test_confident_but_malformed_value_is_never_written() -> None:
    call = Call(intents=[intent("ambiguous")], extractions=[extraction("name", "juan 23", 0.99)])
    assert call.turn("juan veintitrés") == REPROMPT["name"]
    assert call.store.writes == []


def test_denied_readback_with_correction_is_read_back_again_and_bounded() -> None:
    call = Call(
        intents=[intent("ambiguous")] + [intent("ambiguous")] * 3,
        extractions=[extraction("name", f"Juana Pérez {n}") for n in "abcd"],
    )
    assert call.turn("juana").startswith("Tengo Juana Pérez A")
    assert call.turn("no, es juana b").startswith("Tengo Juana Pérez B")
    assert call.turn("no, es juana c").startswith("Tengo Juana Pérez C")
    assert call.turn("no, es juana d").startswith(DEFERRED)  # third denial: move on
    assert call.store.writes == [("mark_unvalidated", "c-1", "name", None)]


def test_unconfident_confirmation_is_not_a_write() -> None:
    call = Call(
        intents=[intent("ambiguous"), intent("confirmed", confidence=0.5)],
        extractions=[extraction("name", "Juan Pérez"), extraction("name", None)],
    )
    call.turn("juan pérez")
    assert call.turn("mjm") == REPROMPT["name"]
    assert call.store.writes == []


def test_out_of_scope_is_answered_and_capped() -> None:
    call = Call(intents=[intent("out_of_scope")] * 3, extractions=[extraction("name", None)])
    assert call.turn("¿quién habla?") == f"{PURPOSE} {ASK['name']}"
    assert call.turn("¿de dónde sacaron mi número?") == f"{PURPOSE} {ASK['name']}"
    assert call.turn("¿quién es?") == REPROMPT["name"]  # cap reached: treated as a failed try
    assert call.state["failures"] == {"name": 1}


def test_decision_model_failure_is_ambiguous_and_the_flow_continues() -> None:
    call = Call(intents=[DecisionError("timeout")], extractions=[extraction("name", "Juan Pérez")])
    assert call.turn("juan pérez").startswith("Tengo Juan Pérez")


def test_laya_sees_the_question_the_agent_asked() -> None:
    call = Call(intents=[intent("ambiguous")], extractions=[extraction("name", "Juan Pérez")])
    call.turn("juan pérez")
    assert call.decision.states[0] == {
        "flujo": "validate_contact",
        "contexto": ASK["name"],
        "transcripción": "juan pérez",
    }


def test_turns_after_the_end_do_nothing() -> None:
    call = Call(intents=[intent("call_rejected", p_rejected=0.9)], extractions=[])
    call.turn("no puedo")
    assert call.turn("¿bueno?") == ""
    assert call.decision.script == [] and call.llm.calls == 0


def test_keyterms_carry_stored_values_for_the_stt() -> None:
    call = Call(intents=[], extractions=[])
    assert call.state["keyterms"] == ["Juan", "Pérez"]
    assert keyterms("email", CONTACT) == ["gmail"]
    assert keyterms("phone", CONTACT) == []


@pytest.mark.parametrize(
    ("field", "value", "spoken"),
    [
        ("phone", "+528182345678", "81 82 34 56 78"),
        ("email", "ana_lopez-9@empresa.com.mx",
         "ana guion bajo lopez guion 9 arroba empresa punto com punto mx"),
        ("name", "María de la Luz", "María de la Luz"),
    ],
)  # fmt: skip
def test_readback(field: str, value: str, spoken: str) -> None:
    assert readback(field, value) == spoken


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ({"name": "validated"}, "Listo, quedó confirmado su nombre."),
        ({"name": "validated", "phone": "unvalidated"}, "Lo demás lo revisaremos"),
        ({"name": "unvalidated"}, "Revisaremos sus datos por otro medio."),
    ],
)
def test_summary(results: dict, expected: str) -> None:
    assert expected in summary(results)
