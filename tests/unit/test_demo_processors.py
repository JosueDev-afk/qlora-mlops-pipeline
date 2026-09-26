"""The demo's Pipecat processors, inside a real Pipecat pipeline, without WebRTC.

Skipped where pipecat-ai is not installed (it lives in the agent group, not
dev, so CI skips it). Run locally with pipecat-ai==1.11.0 before touching
src/agent/demo/processors.py.
"""

import asyncio
import functools
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    EndWorkerFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.tests.utils import SleepFrame  # noqa: E402
from pipecat.tests.utils import run_test as _run_test  # noqa: E402

from src.agent.demo.processors import SpeechInput, SpeechOutput, TurnProcessor  # noqa: E402
from src.agent.speech.base import SpeechToText, TextToSpeech, TranscriptChunk  # noqa: E402

# A cold first start (imports, bytecode) can exceed run_test's 1 s default.
run_test = functools.partial(_run_test, start_timeout=5.0)


class FakeSTT(SpeechToText):
    """A partial after the first audio chunk, a final after the third."""

    async def stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        language: str = "es",
        keyterms: list[str] | None = None,
    ) -> AsyncIterator[TranscriptChunk]:
        n = 0
        async for _ in audio:
            n += 1
            if n == 1:
                yield TranscriptChunk("ocho", is_final=False, confidence=None)
            if n == 3:
                yield TranscriptChunk("ocho uno", is_final=True, confidence=0.9)


class FakeTTS(TextToSpeech):
    def __init__(self, pieces: list[bytes], delay: float = 0.0) -> None:
        self.pieces, self.delay = pieces, delay

    async def synthesize(self, text: str, *, voice_id: str) -> AsyncIterator[bytes]:
        for piece in self.pieces:
            await asyncio.sleep(self.delay)
            yield piece


class FakeSession:
    def __init__(self, stop_on_partial: bool = False, end_after: bool = False) -> None:
        self.stop_on_partial = stop_on_partial
        self.end_after = end_after
        self.ended = False
        self.finals: list[tuple[str, Any]] = []
        self.spoken: list[tuple[str, float]] = []
        self.started: list[str] = []

    async def start(self) -> str:
        return "Buen día."

    async def on_final(self, text: str, confidence: Any = None) -> str:
        self.finals.append((text, confidence))
        self.ended = self.end_after
        return "Tengo 81. ¿Es correcto?"

    async def on_partial(self, text: str) -> bool:
        return self.stop_on_partial

    def agent_started(self, text: str) -> None:
        self.started.append(text)

    def agent_spoke(self, text: str, duration_ms: float) -> None:
        self.spoken.append((text, duration_ms))

    def agent_stopped(self) -> None:
        pass


def audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)


def test_speech_input_turns_audio_into_partials_and_finals() -> None:
    down, _ = asyncio.run(run_test(
        SpeechInput(FakeSTT()),
        frames_to_send=[audio(), audio(), audio(), SleepFrame(0.2)],
    ))  # fmt: skip
    assert [type(f) for f in down] == [InterimTranscriptionFrame, TranscriptionFrame]
    assert down[1].text == "ocho uno" and down[1].result.confidence == 0.9


def test_turn_processor_greets_and_answers_finals() -> None:
    session = FakeSession()
    final = TranscriptionFrame(
        "ocho uno", "caller", "t", result=TranscriptChunk("ocho uno", True, 0.8)
    )
    down, _ = asyncio.run(run_test(TurnProcessor(session), frames_to_send=[final]))  # type: ignore[arg-type]
    assert [f.text for f in down if isinstance(f, TTSSpeakFrame)] == [
        "Buen día.", "Tengo 81. ¿Es correcto?"
    ]  # fmt: skip
    assert session.finals == [("ocho uno", 0.8)]


def test_task_d_stop_broadcasts_an_interruption() -> None:
    partial = InterimTranscriptionFrame("ese no es", "caller", "t")
    session = FakeSession(stop_on_partial=True)
    down, up = asyncio.run(run_test(TurnProcessor(session), frames_to_send=[partial]))  # type: ignore[arg-type]
    assert any(isinstance(f, InterruptionFrame) for f in down)
    assert any(isinstance(f, InterruptionFrame) for f in up)


def test_a_backchannel_does_not_interrupt() -> None:
    partial = InterimTranscriptionFrame("ajá", "caller", "t")
    down, _ = asyncio.run(run_test(TurnProcessor(FakeSession()), frames_to_send=[partial]))  # type: ignore[arg-type]
    assert not any(isinstance(f, InterruptionFrame) for f in down)


def test_the_last_turn_ends_the_call_after_the_farewell() -> None:
    final = TranscriptionFrame("voy manejando", "caller", "t")
    down, _ = asyncio.run(run_test(
        TurnProcessor(FakeSession(end_after=True)),  # type: ignore[arg-type]
        frames_to_send=[final],
        send_end_frame=False,
    ))  # fmt: skip
    kinds = [type(f) for f in down]
    assert kinds.index(EndWorkerFrame) > max(i for i, k in enumerate(kinds) if k is TTSSpeakFrame)


def test_speech_output_streams_whole_samples() -> None:
    session = FakeSession()
    tts = FakeTTS([b"\x01\x02\x03", b"\x04\x05", b"\x06"])  # 6 bytes in odd pieces
    down, _ = asyncio.run(run_test(
        SpeechOutput(tts, voice_id="v", session=session),  # type: ignore[arg-type]
        frames_to_send=[TTSSpeakFrame("Hola"), SleepFrame(0.2)],
    ))  # fmt: skip
    assert type(down[0]) is TTSStartedFrame and type(down[-1]) is TTSStoppedFrame
    chunks = [f.audio for f in down if isinstance(f, TTSAudioRawFrame)]
    assert all(len(c) % 2 == 0 for c in chunks) and b"".join(chunks) == bytes(range(1, 7))
    assert session.started == ["Hola"]
    assert session.spoken == [("Hola", pytest.approx(3 / 16000 * 1000))]


def test_an_interruption_cuts_the_agent_off() -> None:
    session = FakeSession()
    tts = FakeTTS([b"\x00\x00" * 160] * 20, delay=0.02)  # ~0.4 s of synthesis
    down, _ = asyncio.run(run_test(
        SpeechOutput(tts, voice_id="v", session=session),  # type: ignore[arg-type]
        frames_to_send=[TTSSpeakFrame("Tengo juan punto perez"), SleepFrame(0.1),
                        InterruptionFrame(), SleepFrame(0.3)],
    ))  # fmt: skip
    chunks = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert 0 < len(chunks) < 20
