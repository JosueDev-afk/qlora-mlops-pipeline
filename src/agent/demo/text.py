"""The call in a terminal: type what the caller says, read what the agent says.

Same session, graph and models as the voice demo, without audio: the way to
try a flow when there is no microphone, no ElevenLabs key, or a transcript
to replay. Needs the Laya and Qwen servers and a calibration file.

    make demo-text
    python -m src.agent.demo.text --kafka localhost:29092   # also send telemetry
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Callable

from src.agent.demo.session import CallSession
from src.agent.telemetry.producer import KafkaSink, MemorySink, TelemetrySink
from src.common.log import configure_logging


async def converse(session: CallSession, read: Callable[[str], str] = input) -> None:
    print(f"Agente: {await session.start()}")
    while not session.ended:
        try:
            text = await asyncio.to_thread(read, "Tú: ")
        except EOFError:
            break
        reply = await session.on_final(text)
        if reply:
            session.agent_spoke(reply, duration_ms=0.0)
            print(f"Agente: {reply}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Text demo of validate_contact.")
    parser.add_argument(
        "--kafka", default=None, help="bootstrap servers; telemetry stays local if unset"
    )
    args = parser.parse_args(argv)
    configure_logging(level="WARNING")
    sink: TelemetrySink = KafkaSink(args.kafka) if args.kafka else MemorySink()
    session = CallSession.from_env(str(uuid.uuid4()), telemetry=sink)
    try:
        asyncio.run(converse(session))
    finally:
        sink.close()


if __name__ == "__main__":
    main()
