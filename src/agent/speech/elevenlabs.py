"""ElevenLabs behind the provider-agnostic speech interfaces.

Scribe v2 Realtime for STT (WebSocket) and Flash v2.5 for TTS (HTTP stream).
Both protocols are spoken directly instead of through the SDK: the two
endpoints are small and documented, the SDK's surface has changed between
versions, and a thin adapter is easy to test against fakes. Nodes and graphs
never import this module; they receive a SpeechToText / TextToSpeech.

Privacy: ElevenLabs retains audio unless `enable_logging` is false, and zero
retention is enterprise-only. Callers dictate names, phones and emails, so the
demo must not be used with real people's data on a non-enterprise plan.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import urlencode

import httpx
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from src.agent.speech.base import SpeechProviderError, SpeechToText, TextToSpeech, TranscriptChunk

STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
SAMPLE_RATES = {"pcm_8000": 8000, "pcm_16000": 16000, "pcm_22050": 22050,
                "pcm_24000": 24000, "pcm_44100": 44100, "pcm_48000": 48000,
                "ulaw_8000": 8000}  # fmt: skip
ERROR_TYPES = frozenset({
    "error", "auth_error", "quota_exceeded", "commit_throttled", "rate_limited",
    "queue_overflow", "resource_exhausted", "session_time_limit_exceeded",
    "input_error", "chunk_size_exceeded", "insufficient_audio_activity", "transcriber_error",
})  # fmt: skip


class ElevenLabsSTT(SpeechToText):
    """Scribe v2 Realtime.

    Finals come from `committed_transcript_with_timestamps`, because word
    timestamps feed endpointing; the plain `committed_transcript` duplicate is
    ignored. Confidence is exp(mean word logprob); partials carry none.
    """

    def __init__(
        self,
        api_key: str,
        *,
        audio_format: str = "pcm_16000",
        commit_strategy: str = "vad",
        enable_logging: bool = True,
        drain_timeout_s: float = 2.0,
        connect: Callable[..., Any] = ws_connect,
        url: str = STT_URL,
    ) -> None:
        if audio_format not in SAMPLE_RATES:
            raise ValueError(f"unsupported audio_format {audio_format!r}")
        self._api_key = api_key
        self._audio_format = audio_format
        self._commit_strategy = commit_strategy
        self._enable_logging = enable_logging
        self._drain_timeout_s = drain_timeout_s
        self._connect = connect
        self._base_url = url

    def _url(self, language: str, keyterms: list[str] | None) -> str:
        terms = list(dict.fromkeys(t.strip() for t in keyterms or [] if t.strip()))
        query = {
            "model_id": "scribe_v2_realtime",
            "audio_format": self._audio_format,
            "language_code": language,
            "commit_strategy": self._commit_strategy,
            "include_timestamps": "true",
            "enable_logging": str(self._enable_logging).lower(),
        }
        return f"{self._base_url}?{urlencode({**query, 'keyterms': terms}, doseq=True)}"

    def _message(self, audio: bytes, *, commit: bool) -> str:
        return json.dumps({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(audio).decode(),
            "commit": commit,
            "sample_rate": SAMPLE_RATES[self._audio_format],
        })  # fmt: skip

    @staticmethod
    def _chunk(event: dict[str, Any]) -> TranscriptChunk | None:
        kind = event.get("message_type")
        if kind in ERROR_TYPES:
            raise SpeechProviderError(str(kind), str(event.get("error", "")))
        if kind == "partial_transcript" and event.get("text"):
            return TranscriptChunk(text=event["text"], is_final=False, confidence=None)
        if kind == "committed_transcript_with_timestamps":
            words = [
                {k: w[k] for k in ("text", "start", "end", "logprob")}
                for w in event.get("words", [])
                if w.get("type") == "word"
            ]
            confidence = math.exp(sum(w["logprob"] for w in words) / len(words)) if words else None
            return TranscriptChunk(
                text=event.get("text", ""), is_final=True, confidence=confidence, words=words
            )
        return None  # session_started, entities, the untimed committed duplicate

    async def stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        language: str = "es",
        keyterms: list[str] | None = None,
    ) -> AsyncIterator[TranscriptChunk]:
        """Send audio while yielding transcripts; after the final commit, drain and close.

        Draining waits `drain_timeout_s` of silence from the server, so the
        last committed segment is not lost when VAD commits are still in flight.
        """
        headers = {"xi-api-key": self._api_key}
        async with self._connect(self._url(language, keyterms), additional_headers=headers) as ws:
            done_sending = asyncio.Event()

            async def send() -> None:
                async for piece in audio:
                    await ws.send(self._message(piece, commit=False))
                await ws.send(self._message(b"", commit=True))
                done_sending.set()

            sender = asyncio.create_task(send())
            try:
                while True:
                    timeout = self._drain_timeout_s if done_sending.is_set() else None
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout)
                    except (TimeoutError, ConnectionClosed):
                        break
                    if chunk := self._chunk(json.loads(raw)):
                        yield chunk
                if sender.done():
                    sender.result()  # surface an error raised by the audio source
            finally:
                if not sender.done():
                    sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender


class ElevenLabsTTS(TextToSpeech):
    """Flash v2.5 (~75 ms), streamed as raw PCM so playback can start at once."""

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "pcm_16000",
        language_code: str = "es",
        enable_logging: bool = True,
        client: httpx.AsyncClient | None = None,
        url: str = TTS_URL,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._output_format = output_format
        self._language_code = language_code
        self._enable_logging = enable_logging
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0))
        self._url = url

    async def aclose(self) -> None:
        """Close the HTTP client, unless the caller passed one in and owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def synthesize(self, text: str, *, voice_id: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        request = self._client.stream(
            "POST",
            self._url.format(voice_id=voice_id),
            params={
                "output_format": self._output_format,
                "enable_logging": str(self._enable_logging).lower(),
            },
            headers={"xi-api-key": self._api_key},
            json={"text": text, "model_id": self._model_id, "language_code": self._language_code},
        )
        async with request as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:200]
                raise SpeechProviderError(f"http_{response.status_code}", body)
            async for piece in response.aiter_bytes():
                yield piece
