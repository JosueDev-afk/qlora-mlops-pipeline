"""ElevenLabs adapters behind the provider-agnostic speech interfaces.

No network: the WebSocket is a scripted fake and HTTP goes through
httpx.MockTransport. What is pinned here is the protocol contract (URL,
messages, event mapping) and the failure modes a live call must survive.
"""

import asyncio
import base64
import json
import math
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from websockets.exceptions import ConnectionClosedOK

from src.agent.speech.base import SpeechProviderError
from src.agent.speech.elevenlabs import ElevenLabsSTT, ElevenLabsTTS

WORDS = [
    {"text": "ocho", "start": 0.0, "end": 0.3, "type": "word", "logprob": -0.1},
    {"text": " ", "start": 0.3, "end": 0.35, "type": "spacing", "logprob": 0.0},
    {"text": "uno", "start": 0.35, "end": 0.6, "type": "word", "logprob": -0.3},
]


class FakeSocket:
    """Replies only after the client commits, like the real server."""

    def __init__(self, events: list[dict]) -> None:
        self.events = [json.dumps(e) for e in events]
        self.sent: list[dict] = []
        self.committed = asyncio.Event()

    async def __aenter__(self) -> "FakeSocket":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def send(self, message: str) -> None:
        payload = json.loads(message)
        self.sent.append(payload)
        if payload["commit"]:
            self.committed.set()

    async def recv(self) -> str:
        await self.committed.wait()
        if not self.events:
            raise ConnectionClosedOK(None, None)
        return self.events.pop(0)


def _stt(events: list[dict], **kwargs):
    socket = FakeSocket(events)
    calls = {}

    def connect(url, additional_headers):
        calls.update(url=url, headers=additional_headers)
        return socket

    return ElevenLabsSTT(api_key="k", connect=connect, **kwargs), socket, calls


async def _audio(*chunks: bytes):
    for chunk in chunks:
        yield chunk


def _collect(stt, **kwargs):
    async def run():
        return [c async for c in stt.stream(_audio(b"\x01\x02", b"\x03"), **kwargs)]

    return asyncio.run(run())


def test_connects_with_model_format_language_timestamps_and_keyterms() -> None:
    stt, _, calls = _stt([])
    _collect(stt, keyterms=["Pérez", "gmail", " ", "Pérez"])
    query = parse_qs(urlparse(calls["url"]).query)
    assert query["model_id"] == ["scribe_v2_realtime"]
    assert query["audio_format"] == ["pcm_16000"]
    assert query["language_code"] == ["es"]
    assert query["include_timestamps"] == ["true"]
    assert query["keyterms"] == ["Pérez", "gmail"]  # blanks and duplicates dropped
    assert calls["headers"] == {"xi-api-key": "k"}


def test_streams_audio_as_base64_and_commits_at_the_end() -> None:
    stt, socket, _ = _stt([])
    _collect(stt)
    audio = [m for m in socket.sent if not m["commit"]]
    assert [base64.b64decode(m["audio_base_64"]) for m in audio] == [b"\x01\x02", b"\x03"]
    assert all(m["sample_rate"] == 16000 for m in socket.sent)
    assert socket.sent[-1]["commit"] is True


def test_maps_partials_and_timestamped_finals() -> None:
    stt, _, _ = _stt(
        [
            {"message_type": "session_started", "session_id": "s"},
            {"message_type": "partial_transcript", "text": ""},
            {"message_type": "partial_transcript", "text": "ocho"},
            {"message_type": "committed_transcript", "text": "ocho uno"},
            {
                "message_type": "committed_transcript_with_timestamps",
                "text": "ocho uno",
                "words": WORDS,
            },
        ]
    )
    partial, final = _collect(stt)
    assert (partial.text, partial.is_final, partial.confidence) == ("ocho", False, None)
    assert final.is_final and final.text == "ocho uno"
    assert final.confidence == pytest.approx(math.exp((-0.1 - 0.3) / 2))  # words only
    assert [w["text"] for w in final.words] == ["ocho", "uno"]
    assert final.words[0] == {"text": "ocho", "start": 0.0, "end": 0.3, "logprob": -0.1}


@pytest.mark.parametrize("kind", ["auth_error", "quota_exceeded", "input_error"])
def test_provider_errors_raise_a_typed_exception(kind: str) -> None:
    stt, _, _ = _stt([{"message_type": kind, "error": "nope"}])
    with pytest.raises(SpeechProviderError) as err:
        _collect(stt)
    assert err.value.kind == kind


def test_unsupported_audio_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="audio_format"):
        ElevenLabsSTT(api_key="k", audio_format="mp3_44100")


# ── TTS ──────────────────────────────────────────────────────────────────


def _tts(handler, **kwargs) -> ElevenLabsTTS:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ElevenLabsTTS(api_key="k", client=client, **kwargs)


def _speak(tts: ElevenLabsTTS, text: str) -> bytes:
    async def run():
        return b"".join([c async for c in tts.synthesize(text, voice_id="voice123")])

    return asyncio.run(run())


def test_tts_posts_text_with_flash_model_and_streams_audio() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(
            path=request.url.path,
            query=dict(request.url.params),
            key=request.headers["xi-api-key"],
            body=json.loads(request.content),
        )
        return httpx.Response(200, content=b"PCMDATA")

    assert _speak(_tts(handler), "¿Me confirma su correo?") == b"PCMDATA"
    assert seen["path"] == "/v1/text-to-speech/voice123/stream"
    assert seen["query"]["output_format"] == "pcm_16000"
    assert seen["key"] == "k"
    assert seen["body"] == {
        "text": "¿Me confirma su correo?",
        "model_id": "eleven_flash_v2_5",
        "language_code": "es",
    }


def test_tts_http_error_raises_a_typed_exception() -> None:
    tts = _tts(lambda request: httpx.Response(401, json={"detail": "invalid key"}))
    with pytest.raises(SpeechProviderError, match="401"):
        _speak(tts, "hola")


def test_tts_with_empty_text_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert _speak(_tts(handler), "   ") == b""
