"""Voice demo: validate_contact in the browser, over WebRTC (Pipecat's runner).

    make demo            # then open http://localhost:7860/client

Needs ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID, the Laya and Qwen servers,
and a calibration file. Demo only: hypotheses are evaluated offline on
transcripts, and nothing here may block that work.

Privacy: ElevenLabs keeps audio unless logging is off, and zero retention is
enterprise-only. Do not run this with real people's data on a standard plan.
Telemetry, when sent, is redacted; contact writes stay in memory.
"""

from __future__ import annotations

import os
import uuid

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from src.agent.demo.processors import SAMPLE_RATE, SpeechInput, SpeechOutput, TurnProcessor
from src.agent.demo.session import DEMO_CONTACT, CallSession
from src.agent.graphs.flow_validate_contact import keyterms
from src.agent.speech.elevenlabs import ElevenLabsSTT, ElevenLabsTTS
from src.agent.telemetry.producer import KafkaSink, MemorySink, TelemetrySink
from src.common.log import get_logger

log = get_logger(__name__)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (see .env.example)")
    return value


async def bot(runner_args: SmallWebRTCRunnerArguments) -> None:
    """One browser connection is one call."""
    api_key = _env("ELEVENLABS_API_KEY")
    retain = os.environ.get("ELEVENLABS_ENABLE_LOGGING", "true").lower() != "false"
    kafka = os.environ.get("DEMO_KAFKA_BOOTSTRAP")
    sink: TelemetrySink = KafkaSink(kafka) if kafka else MemorySink()
    session = CallSession.from_env(str(uuid.uuid4()), telemetry=sink)
    tts = ElevenLabsTTS(api_key, output_format=f"pcm_{SAMPLE_RATE}", enable_logging=retain)
    # Keyterms are fixed per STT connection, so the whole record goes in up front.
    terms = keyterms("name", DEMO_CONTACT) + keyterms("email", DEMO_CONTACT)
    stt = ElevenLabsSTT(api_key, audio_format=f"pcm_{SAMPLE_RATE}", enable_logging=retain)

    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        SpeechInput(stt, keyterms=terms),
        TurnProcessor(session),
        SpeechOutput(tts, voice_id=_env("ELEVENLABS_VOICE_ID"), session=session),
        transport.output(),
    ])  # fmt: skip
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: SmallWebRTCTransport, client: object) -> None:
        await task.cancel()

    log.warning("demo_call", call_id=session.call_id, audio_retained_by_provider=retain)
    try:
        await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)
    finally:
        sink.close()
        await tts.aclose()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
