"""Pipecat processors that put our interfaces into the audio pipeline.

    transport.input -> SpeechInput -> TurnProcessor -> SpeechOutput -> transport.output

Plain FrameProcessors rather than Pipecat's STT/TTS service classes: the
providers stay behind `SpeechToText` / `TextToSpeech` (never the ElevenLabs
SDK in the demo either), and Pipecat's contract is reduced to a handful of
frames. Nothing here decides anything; `CallSession` does.

Turn ends come from Scribe's VAD commits (a final transcript is a turn).
Barge-in comes from Task D on partials, not from Pipecat's VAD: the point of
the project is that "ajá" while the agent talks is not an interruption.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    EndWorkerFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601

from src.agent.demo.session import CallSession
from src.agent.speech.base import SpeechProviderError, SpeechToText, TextToSpeech
from src.common.log import get_logger

SAMPLE_RATE = 16000
USER_ID = "caller"

log = get_logger(__name__)


class SpeechInput(FrameProcessor):
    """Streams caller audio into a SpeechToText and pushes partials and finals."""

    def __init__(
        self, stt: SpeechToText, *, language: str = "es", keyterms: list[str] | None = None
    ) -> None:
        super().__init__()
        self._stt = stt
        self._language = language
        self._keyterms = keyterms
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._audio.put_nowait(frame.audio)  # consumed here, not passed on
            return
        if isinstance(frame, EndFrame | CancelFrame):
            await self._stop()
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._task = self.create_task(self._transcribe())

    async def _chunks(self) -> AsyncIterator[bytes]:
        while (piece := await self._audio.get()) is not None:
            yield piece

    async def _transcribe(self) -> None:
        try:
            async for chunk in self._stt.stream(
                self._chunks(), language=self._language, keyterms=self._keyterms
            ):
                if chunk.is_final:
                    frame: Frame = TranscriptionFrame(
                        chunk.text, USER_ID, time_now_iso8601(), result=chunk, finalized=True
                    )
                else:
                    frame = InterimTranscriptionFrame(
                        chunk.text, USER_ID, time_now_iso8601(), result=chunk
                    )
                await self.push_frame(frame)
        except SpeechProviderError as exc:
            log.warning("stt_failed", kind=exc.kind)
            await self.push_frame(ErrorFrame(error=f"stt: {exc.kind}"), FrameDirection.UPSTREAM)

    async def _stop(self) -> None:
        self._audio.put_nowait(None)
        if self._task:
            await self.cancel_task(self._task)
            self._task = None


class TurnProcessor(FrameProcessor):
    """Finals run a turn through the session; partials may stop the agent (Task D)."""

    def __init__(self, session: CallSession) -> None:
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            confidence = getattr(frame.result, "confidence", None)
            await self._say(await self._session.on_final(frame.text, confidence))
            if self._session.ended:
                # Downstream, behind the farewell: it is spoken before the call ends.
                await self.push_frame(EndWorkerFrame(reason="call ended"))
            return
        if isinstance(frame, InterimTranscriptionFrame):
            if await self._session.on_partial(frame.text):
                await self.broadcast_interruption()
            return
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._session.agent_stopped()
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self._say(await self._session.start())

    async def _say(self, text: str) -> None:
        if text:
            await self.push_frame(TTSSpeakFrame(text, append_to_context=False))


class SpeechOutput(FrameProcessor):
    """Speaks TTSSpeakFrames through a TextToSpeech; an interruption cuts it off."""

    def __init__(self, tts: TextToSpeech, *, voice_id: str, session: CallSession) -> None:
        super().__init__()
        self._tts = tts
        self._voice_id = voice_id
        self._session = session
        self._task: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            await self._finish()  # one utterance at a time, in order
            self._task = self.create_task(self._speak(frame.text))
            return
        if isinstance(frame, InterruptionFrame | CancelFrame):
            await self._cut()
        elif isinstance(frame, EndFrame):
            await self._finish()  # the farewell is spoken before the call ends
        await self.push_frame(frame, direction)

    async def _speak(self, text: str) -> None:
        self._session.agent_started(text)
        await self.push_frame(TTSStartedFrame())
        pending, total = b"", 0
        try:
            async for piece in self._tts.synthesize(text, voice_id=self._voice_id):
                pending += piece
                usable = len(pending) - len(pending) % 2  # whole 16-bit samples only
                if usable:
                    await self.push_frame(TTSAudioRawFrame(pending[:usable], SAMPLE_RATE, 1))
                    total += usable
                    pending = pending[usable:]
        except SpeechProviderError as exc:
            log.warning("tts_failed", kind=exc.kind)
            await self.push_frame(ErrorFrame(error=f"tts: {exc.kind}"), FrameDirection.UPSTREAM)
        finally:
            await self.push_frame(TTSStoppedFrame())
            self._session.agent_spoke(text, duration_ms=total / 2 / SAMPLE_RATE * 1000)

    async def _finish(self) -> None:
        if self._task:
            await self._task
            self._task = None

    async def _cut(self) -> None:
        if self._task:
            await self.cancel_task(self._task)
            self._task = None
