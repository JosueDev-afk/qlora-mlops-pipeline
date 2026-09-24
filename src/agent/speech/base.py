"""Provider-agnostic speech interfaces.

The risk register lists 'dependency on an external voice provider' as a medium
risk, mitigated by an in-house abstraction layer. This module *is* that
mitigation: ElevenLabs is one implementation, not the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TranscriptChunk:
    """One partial or final piece of a transcript.

    `confidence` is None when the provider gives none (partials usually do
    not): a made-up 0.0 would read as "certainly wrong" to Task A.
    """

    text: str
    is_final: bool
    confidence: float | None
    words: list[dict] = field(default_factory=list)  # word-level timestamps, feeds endpointing


class SpeechProviderError(RuntimeError):
    """The provider rejected the session or the request (auth, quota, input)."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


class SpeechToText(ABC):
    # Declared without `async`: implementations are async generators, and an
    # `async def` here would type them as coroutines returning an iterator.
    @abstractmethod
    def stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        language: str = "es",
        keyterms: list[str] | None = None,
    ) -> AsyncIterator[TranscriptChunk]:
        """Transcribe streaming audio.

        `keyterms` carries domain vocabulary already known from the database
        (contact name, email domain) to bias recognition of dictated strings.
        """


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str, *, voice_id: str) -> AsyncIterator[bytes]:
        """Synthesize speech, streaming audio out as it is produced."""
