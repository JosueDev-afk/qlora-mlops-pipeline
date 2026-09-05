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
    text: str
    is_final: bool
    confidence: float
    words: list[dict] = field(default_factory=list)  # word-level timestamps, feeds endpointing


class SpeechToText(ABC):
    @abstractmethod
    async def stream(
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
    async def synthesize(self, text: str, *, voice_id: str) -> AsyncIterator[bytes]:
        """Synthesize speech, streaming audio out as it is produced."""
