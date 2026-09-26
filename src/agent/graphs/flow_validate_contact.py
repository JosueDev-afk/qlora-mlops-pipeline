"""Flow 1, validate_contact: confirm and correct name, phone and email.

LangGraph runs once per caller turn (layer 2); audio, barge-in and Task D
live in layer 1. Each invocation takes the turn's transcript, walks the
nodes of the Phase 2 state machine (Figure 5) and stops at the agent's next
utterance in `say`. State between turns is kept by the checkpointer under
the call id, which is also where the prompt version is anchored (H3).

    start -> ask_field -> [turn] -> classify_intent -> extract_entity
          -> check_confidence -> confirm -> [turn] -> update_contact -> ask_field ...
                                 -> reprompt (max 2) -> mark_unvalidated -> ask_field ...
    handle_rejection from any turn; end_call when no field is left.

Only two nodes call a model (classify_intent on Laya, extract_entity on Qwen)
and only update_contact / mark_unvalidated write. After two failed reprompts
a field is marked unvalidated and the flow moves on; it never escalates to a
human and never ends the call for one field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.decision.base import DecisionModel
from src.agent.llm.client import LLMClient
from src.agent.nodes.classify_intent import classify_intent
from src.agent.nodes.extract_entity import acceptable, canonical, extract_entity
from src.agent.tools.contact import ContactStore

FLOW = "validate_contact"
FIELDS = ("name", "phone", "email")
Awaiting = Literal["start", "value", "confirmation", "done"]

# ── agent speech (es-MX) ───────────────────────────────────────────────────

FIELD_NAMES = {"name": "su nombre", "phone": "su teléfono", "email": "su correo"}
ASK = {
    "name": "¿Me podría decir su nombre completo, por favor?",
    "phone": "¿Me podría dictar su número de celular a diez dígitos?",
    "email": "¿Me podría decir su correo electrónico?",
}
# Guided: the second try asks for the form the normalizers read best.
REPROMPT = {
    "name": "Disculpe, no le entendí bien. ¿Me lo podría deletrear?",
    "phone": "Disculpe, ¿me lo podría repetir de dos en dos dígitos?",
    "email": "Disculpe, ¿me lo podría deletrear despacio? Por ejemplo: jota, u, a, ene, "
    "arroba, gmail, punto com.",
}
GREETING = "Buen día, le llamo{company} para confirmar sus datos de contacto."
PURPOSE = "Le llamo para confirmar sus datos de contacto."
CONFIRM = "Tengo {value}. ¿Es correcto?"
SAVED = "Perfecto, quedó registrado."
DEFERRED = "No se preocupe, lo revisaremos después."
FAREWELL = "Gracias por su tiempo, que tenga buen día."
REJECTED = "Entiendo, disculpe la molestia. Que tenga buen día."


class ContactState(TypedDict, total=False):
    call_id: str
    contact: dict[str, str]  # prefetched by primary key at dial time
    awaiting: Awaiting
    transcript: str  # this turn's final transcript (turn input)
    asr_confidence: float | None  # (turn input)
    pending: list[str]
    field: str | None
    asked: str  # the agent's last question: Laya's context, clarify repeats it
    intent: dict[str, Any]
    candidate: dict[str, Any] | None
    failures: dict[str, int]
    clarifications: int
    results: dict[str, str]  # field -> validated | unvalidated
    keyterms: list[str]  # for the STT on the next turn
    say: str  # the agent's utterance this turn
    outcome: str | None  # completed | rejected
    last_model_call: dict[str, Any] | None


@dataclass(frozen=True)
class FlowDeps:
    """Everything the flow reaches outside itself, injected so tests use fakes.

    `rejection_threshold` comes from `models/laya_calibration.json` (derived
    by calibrate_laya, never hand-set).
    """

    decision: DecisionModel
    llm: LLMClient
    contacts: ContactStore
    rejection_threshold: float
    confidence_threshold: float = 0.75
    max_reprompts: int = 2
    max_clarifications: int = 2
    fields: tuple[str, ...] = FIELDS


# ── pure helpers ───────────────────────────────────────────────────────────


def _say(state: ContactState, text: str) -> str:
    return f"{state.get('say', '')} {text}".strip()


def readback(field: str, value: str) -> str:
    """How the agent reads a value back so the caller can confirm it by ear."""
    if field == "phone":
        national = value.removeprefix("+52")
        return " ".join(national[i : i + 2] for i in range(0, len(national), 2))
    if field == "email":
        for symbol, word in (("@", " arroba "), (".", " punto "), ("_", " guion bajo "),
                             ("-", " guion ")):  # fmt: skip
            value = value.replace(symbol, word)
        return " ".join(value.split())
    return value


def keyterms(field: str, contact: dict[str, str]) -> list[str]:
    """Stored values that help the STT hear this field (keyterm prompting)."""
    if field == "name":
        return contact.get("name", "").split()
    if field == "email" and "@" in contact.get("email", ""):
        return [contact["email"].split("@")[1].split(".")[0]]
    return []


def summary(results: dict[str, str]) -> str:
    validated = [FIELD_NAMES[f] for f, r in results.items() if r == "validated"]
    if not validated:
        return f"Revisaremos sus datos por otro medio. {FAREWELL}"
    listed = " y ".join([", ".join(validated[:-1]), validated[-1]] if len(validated) > 1
                        else validated)  # fmt: skip
    plural = "quedaron confirmados" if len(validated) > 1 else "quedó confirmado"
    deferred = " Lo demás lo revisaremos por otro medio." if len(validated) < len(results) else ""
    return f"Listo, {plural} {listed}.{deferred} {FAREWELL}"


# ── graph ──────────────────────────────────────────────────────────────────


def build_graph(deps: FlowDeps, checkpointer: Any = None) -> Any:
    """Compile Flow 1. Pass a checkpointer to keep state across turns by call id."""

    def receive(state: ContactState) -> ContactState:
        # Per-turn fields: a turn that never reaches Qwen must not report the
        # previous turn's call to telemetry.
        return {"say": "", "last_model_call": None}

    def route_turn(state: ContactState) -> str:
        awaiting = state.get("awaiting", "start")
        return {"start": "start", "done": END}.get(awaiting, "classify_intent")

    def start(state: ContactState) -> ContactState:
        company = state.get("contact", {}).get("company_name")
        greeting = GREETING.format(company=f" de {company}" if company else "")
        return {
            "pending": list(deps.fields),
            "failures": {},
            "results": {},
            "clarifications": 0,
            "outcome": None,
            "say": greeting,
        }

    def ask_field(state: ContactState) -> ContactState:
        field, *rest = state["pending"]
        question = ASK[field]
        return {
            "field": field,
            "pending": rest,
            "candidate": None,
            "awaiting": "value",
            "asked": question,
            "keyterms": keyterms(field, state.get("contact", {})),
            "say": _say(state, question),
        }

    async def classify(state: ContactState) -> ContactState:
        intent = await classify_intent(
            deps.decision,
            flow=FLOW,
            context=state.get("asked", ""),
            transcript=state.get("transcript", ""),
            rejection_threshold=deps.rejection_threshold,
        )
        return {"intent": intent}

    def route_intent(state: ContactState) -> str:
        intent = state["intent"]
        if intent["intent"] == "call_rejected":
            return "handle_rejection"
        if (
            intent["intent"] == "out_of_scope"
            and state.get("clarifications", 0) < deps.max_clarifications
        ):
            return "clarify"
        if (
            state.get("awaiting") == "confirmation"
            and intent["intent"] == "confirmed"
            and intent["confidence"] >= deps.confidence_threshold
        ):
            return "update_contact"
        # A value turn, or a readback the caller did not confirm: the turn may
        # carry the value or its correction ("no, es ocho uno...").
        return "extract_entity"

    def clarify(state: ContactState) -> ContactState:
        """ "¿Quién habla?": answer with the purpose and repeat the question.

        Does not count as a failed try, but is capped so a model that keeps
        reading dictation as out_of_scope cannot loop forever.
        """
        return {
            "clarifications": state.get("clarifications", 0) + 1,
            "say": _say(state, f"{PURPOSE} {state.get('asked', '')}"),
        }

    async def extract(state: ContactState) -> ContactState:
        parsed, telemetry = await extract_entity(
            deps.llm,
            field=state["field"] or "",
            transcript=state.get("transcript", ""),
            asr_confidence=state.get("asr_confidence"),
        )
        return {"candidate": parsed.value, "last_model_call": telemetry}

    def check_confidence(state: ContactState) -> str:
        """One failed try per turn: a bad extraction, or a denied readback.

        A readback denied with a correction is confirmed again, and counts as
        a failure in `confirm`; without one it goes to `reprompt`, which
        counts it. Either way the third failure marks the field unvalidated.
        """
        field = state["field"] or ""
        failures = state.get("failures", {}).get(field, 0)
        ok = acceptable(field, state["candidate"] or {}, deps.confidence_threshold)
        rereading = state.get("awaiting") == "confirmation"
        if ok and (not rereading or failures < deps.max_reprompts):
            return "confirm"
        if not ok and failures < deps.max_reprompts:
            return "reprompt"
        return "mark_unvalidated"

    def _failed(state: ContactState) -> dict[str, int]:
        field = state["field"] or ""
        failures = dict(state.get("failures", {}))
        failures[field] = failures.get(field, 0) + 1
        return failures

    def confirm(state: ContactState) -> ContactState:
        field = state["field"] or ""
        value = canonical(field, (state["candidate"] or {}).get("normalized_value")) or ""
        question = CONFIRM.format(value=readback(field, value))
        update: ContactState = {
            "awaiting": "confirmation",
            "asked": question,
            "say": _say(state, question),
        }
        if state.get("awaiting") == "confirmation":
            update["failures"] = _failed(state)
        return update

    def reprompt(state: ContactState) -> ContactState:
        question = REPROMPT[state["field"] or ""]
        return {
            "failures": _failed(state),
            "candidate": None,
            "awaiting": "value",
            "asked": question,
            "say": _say(state, question),
        }

    async def update_contact(state: ContactState) -> ContactState:
        field = state["field"] or ""
        value = canonical(field, (state["candidate"] or {}).get("normalized_value"))
        if value is None:  # unreachable: confirm only reads back canonical values
            raise RuntimeError(f"refusing to write a non-canonical {field}")
        await deps.contacts.update_contact(state["contact"]["id"], field, value)
        return {
            "results": {**state.get("results", {}), field: "validated"},
            "say": _say(state, SAVED),
        }

    async def mark_unvalidated(state: ContactState) -> ContactState:
        field = state["field"] or ""
        await deps.contacts.mark_unvalidated(state["contact"]["id"], field)
        return {
            "results": {**state.get("results", {}), field: "unvalidated"},
            "say": _say(state, DEFERRED),
        }

    def route_next(state: ContactState) -> str:
        return "ask_field" if state.get("pending") else "end_call"

    def end_call(state: ContactState) -> ContactState:
        return {
            "field": None,
            "awaiting": "done",
            "outcome": "completed",
            "keyterms": [],
            "say": _say(state, summary(state.get("results", {}))),
        }

    def handle_rejection(state: ContactState) -> ContactState:
        """One rejection signal ends the call: no insisting, no writes."""
        return {"awaiting": "done", "outcome": "rejected", "keyterms": [], "say": REJECTED}

    graph = StateGraph(ContactState)
    for name, node in [
        ("receive", receive),
        ("start", start),
        ("ask_field", ask_field),
        ("classify_intent", classify),
        ("clarify", clarify),
        ("extract_entity", extract),
        ("confirm", confirm),
        ("reprompt", reprompt),
        ("update_contact", update_contact),
        ("mark_unvalidated", mark_unvalidated),
        ("end_call", end_call),
        ("handle_rejection", handle_rejection),
    ]:
        graph.add_node(name, node)

    graph.add_edge(START, "receive")
    graph.add_conditional_edges("receive", route_turn, ["start", "classify_intent", END])
    graph.add_edge("start", "ask_field")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        ["handle_rejection", "clarify", "update_contact", "extract_entity"],
    )
    graph.add_conditional_edges(
        "extract_entity", check_confidence, ["confirm", "reprompt", "mark_unvalidated"]
    )
    graph.add_conditional_edges("update_contact", route_next, ["ask_field", "end_call"])
    graph.add_conditional_edges("mark_unvalidated", route_next, ["ask_field", "end_call"])
    for turn_end in ("ask_field", "clarify", "confirm", "reprompt", "end_call", "handle_rejection"):
        graph.add_edge(turn_end, END)
    return graph.compile(checkpointer=checkpointer)


async def start_call(app: Any, call_id: str, contact: dict[str, str]) -> ContactState:
    """First invocation: greet and ask the first field. `contact` needs `id`."""
    config = {"configurable": {"thread_id": call_id}}
    initial: ContactState = {"call_id": call_id, "contact": contact, "awaiting": "start"}
    result: ContactState = await app.ainvoke(initial, config)
    return result


async def run_turn(
    app: Any, call_id: str, transcript: str, asr_confidence: float | None = None
) -> ContactState:
    """One caller turn in, the agent's next utterance out (in `say`)."""
    config = {"configurable": {"thread_id": call_id}}
    turn: ContactState = {"transcript": transcript, "asr_confidence": asr_confidence}
    result: ContactState = await app.ainvoke(turn, config)
    return result
