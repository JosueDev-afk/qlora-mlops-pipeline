"""One call, between the audio layer and the turn layer.

Layer 1 (Pipecat, or the terminal demo) calls four methods and knows nothing
about graphs; layer 2 (LangGraph) runs once per final transcript and knows
nothing about audio:

- `start()`: the greeting.
- `on_final(text)`: a completed caller turn; returns what the agent says.
- `on_partial(text)`: an ASR partial while the agent speaks; returns whether
  to stop talking (Task D). Every failure stops: talking over someone who
  says "ese no es mi correo" costs more than a pause.
- `agent_spoke(text)` / `agent_interrupted()`: what actually came out.

Telemetry is redacted (real people may be on the line) and never blocks.
The demo writes to an in-memory contact store: nothing reaches Postgres.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from src.agent.decision.base import DecisionError, DecisionModel
from src.agent.decision.calibration import CALIBRATION_PATH, load_agent_calibration
from src.agent.decision.interruption import parse_interruption, should_interrupt
from src.agent.graphs.flow_validate_contact import (
    ContactState,
    FlowDeps,
    build_graph,
    run_turn,
    start_call,
)
from src.agent.llm.client import LLMClient
from src.agent.nodes.extract_entity import PROMPT as EXTRACT_PROMPT
from src.agent.telemetry.producer import CallTrace, MemorySink, TelemetrySink
from src.agent.tools.contact import ContactStore, InMemoryContactStore
from src.common.config import load_params, param
from src.common.log import get_logger
from src.common.prompts import load_prompt

INTERRUPTION = load_prompt("is_real_interruption", version=2)
# A made-up contact: the demo never loads a real record.
DEMO_CONTACT = {
    "id": "demo-contact",
    "name": "Juan Pérez",
    "phone": "+528100000000",
    "email": "juan.perez@gmail.com",
    "company_name": "Clínica Demo",
}

log = get_logger(__name__)


@dataclass
class CallSession:
    call_id: str
    decision: DecisionModel
    llm: LLMClient
    rejection_threshold: float
    interruption_threshold: float
    contacts: ContactStore = field(default_factory=InMemoryContactStore)
    telemetry: TelemetrySink = field(default_factory=MemorySink)
    contact: dict[str, str] = field(default_factory=lambda: dict(DEMO_CONTACT))
    model_version: str = "Qwen/Qwen3-4B"

    def __post_init__(self) -> None:
        if not self.interruption_threshold < 0.5:
            raise ValueError("interruption threshold must stay below 0.5 (rule 4)")
        deps = FlowDeps(self.decision, self.llm, self.contacts, self.rejection_threshold)
        self._app = build_graph(deps, checkpointer=InMemorySaver())
        self._trace = CallTrace(self.call_id, self.telemetry)  # redacted by default
        self._turn = 0
        self._speaking = ""
        self._results: dict[str, str] = {}
        self.ended = False
        self.keyterms: list[str] = []

    # ── turns (layer 2) ────────────────────────────────────────────────────

    async def start(self) -> str:
        self._trace.emit(
            "call_started",
            flow="validate_contact",
            contact_id=self.contact["id"],
            model_version=self.model_version,
            prompt_hash=EXTRACT_PROMPT.sha256,
            synthetic=False,
        )
        state = await start_call(self._app, self.call_id, self.contact)
        return self._after(state)

    async def on_final(self, text: str, asr_confidence: float | None = None) -> str:
        """Run the graph on a completed turn. Empty text or a finished call says nothing."""
        if self.ended or not text.strip():
            return ""
        self._trace.emit("turn_started", turn_idx=self._turn, speaker="person")
        self._trace.emit("asr_final", turn_idx=self._turn, text=text, asr_confidence=asr_confidence)
        state = await run_turn(self._app, self.call_id, text, asr_confidence)
        call = state.get("last_model_call")
        if call:
            self._trace.emit(
                "llm_response",
                turn_idx=self._turn,
                output=state.get("candidate"),
                **{
                    k: call[k]
                    for k in ("task", "prompt_version", "prompt_sha256", "json_valid", "latency_ms")
                },  # fmt: skip
            )
        self._turn += 1
        return self._after(state)

    def _after(self, state: ContactState) -> str:
        results = state.get("results") or {}
        for field_name, result in results.items():
            if field_name not in self._results:
                tool = "update_contact" if result == "validated" else "mark_unvalidated"
                self._trace.emit("tool_invoked", turn_idx=self._turn, tool=tool, field=field_name)
        self._results = dict(results)
        self.keyterms = list(state.get("keyterms") or [])
        if state.get("awaiting") == "done" and not self.ended:
            self.ended = True
            self._trace.emit(
                "call_ended",
                outcome=_outcome(state.get("outcome"), results),
                turns=self._turn,
                fields=results,
            )
        return str(state.get("say") or "")

    # ── audio (layer 1) ────────────────────────────────────────────────────

    def agent_started(self, text: str) -> None:
        """Synthesis began: from here until playback stops, partials go to Task D."""
        self._speaking = text

    def agent_spoke(self, text: str, duration_ms: float) -> None:
        """Synthesis finished (audio may still be playing): telemetry only."""
        self._trace.emit("turn_started", turn_idx=self._turn, speaker="agent")
        self._trace.emit("tts_done", turn_idx=self._turn, text=text, duration_ms=duration_ms)

    def agent_stopped(self) -> None:
        """Playback ended."""
        self._speaking = ""

    async def on_partial(self, text: str) -> bool:
        """Task D: stop talking? Only asked while the agent speaks."""
        if not self._speaking or not text.strip():
            return False
        state = {"agente_dice": self._speaking, "transcripción": text}
        started = time.perf_counter()
        try:
            answers = await self.decision.decide(state, INTERRUPTION.body["questions"])
            p_real = float(answers["interruption"].value)
            raw: object = {
                "interruption": p_real >= 0.5,
                "confidence": max(p_real, 1.0 - p_real),
                "kind": answers["kind"].value,
            }
        except (DecisionError, KeyError, TypeError, ValueError) as exc:
            log.warning("interruption_failed", error=repr(exc))
            raw = None  # resolves to interrupting
        result = parse_interruption(raw)
        stop = should_interrupt(result, self.interruption_threshold)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        p = result["confidence"] if result["interruption"] else 1 - result["confidence"]
        self._trace.emit(
            "decision",
            turn_idx=self._turn,
            task="is_real_interruption",
            output=result,
            latency_ms=latency_ms,
        )
        self._trace.emit("barge_in", turn_idx=self._turn, interrupted=stop, p_real=round(p, 4))
        if stop:
            self._speaking = ""
        return stop

    # ── wiring ─────────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        call_id: str,
        *,
        params_path: str = "params.yaml",
        calibration_path: Path = CALIBRATION_PATH,
        telemetry: TelemetrySink | None = None,
    ) -> CallSession:
        """Real clients from the environment (.env): Laya, Qwen, calibration, thresholds."""
        from src.agent.decision.laya_model import LayaDecisionModel
        from src.agent.llm.client import QwenClient

        params = load_params(params_path)
        calibration = load_agent_calibration(calibration_path)
        model = os.environ.get("QWEN_MODEL", str(param(params, "train.base_model")))
        return cls(
            call_id=call_id,
            decision=LayaDecisionModel(
                os.environ.get("LAYA_BASE_URL", "http://localhost:8001"),
                calibration_run_id=calibration.run_id,
                api_key=os.environ.get("LAYA_API_KEY") or None,
            ),
            llm=QwenClient(
                os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1"), model=model
            ),
            rejection_threshold=calibration.call_rejected_threshold,
            interruption_threshold=float(param(params, "audio.interruption_threshold")),
            telemetry=telemetry or MemorySink(),
            model_version=model,
        )


def _outcome(outcome: str | None, results: dict[str, str]) -> str:
    if outcome == "rejected":
        return "rejected"
    return "completed" if all(r == "validated" for r in results.values()) else "partial"
