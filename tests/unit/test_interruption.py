"""Task D fails toward interrupting, never toward talking over someone.

Task D runs on ASR partials while the agent speaks. There is no retry (the next
partial is the retry) and a boolean has no `ambiguous` value, so every failure
must resolve to stopping the TTS.
"""

import pytest

from src.agent.decision.interruption import FALLBACK, parse_interruption, should_interrupt

THRESHOLD = 0.4


def test_valid_output_passes_through() -> None:
    raw = {"interruption": False, "confidence": 0.93, "kind": "backchannel"}
    assert parse_interruption(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        None,  # timeout or transport error
        "no es un objeto",
        {"interruption": "sí", "confidence": 0.9},  # wrong type
        {"interruption": True},  # missing confidence
        {"interruption": True, "confidence": 1.4},  # out of range
        {"interruption": True, "confidence": 0.9, "extra": 1},  # additionalProperties
    ],
)
def test_any_failure_becomes_an_interruption(raw: object) -> None:
    assert parse_interruption(raw) == FALLBACK
    assert should_interrupt(parse_interruption(raw), THRESHOLD)


def test_confident_backchannel_keeps_speaking() -> None:
    assert not should_interrupt({"interruption": False, "confidence": 0.93}, THRESHOLD)


def test_unsure_backchannel_interrupts() -> None:
    """P(real) = 1 - 0.55 = 0.45 >= 0.4: when in doubt, stop talking."""
    assert should_interrupt({"interruption": False, "confidence": 0.55}, THRESHOLD)


def test_real_interruption_interrupts() -> None:
    raw = {"interruption": True, "confidence": 0.8, "kind": "correction"}
    assert should_interrupt(raw, THRESHOLD)


def test_rejection_always_interrupts_even_with_low_confidence() -> None:
    """A rejection short-circuits to handle_rejection; it can never be talked over."""
    raw = {"interruption": False, "confidence": 0.9, "kind": "rejection"}
    assert should_interrupt(raw, THRESHOLD)


def test_threshold_at_or_above_half_is_rejected() -> None:
    """A threshold >= 0.5 would mean 'when in doubt, keep talking'."""
    with pytest.raises(ValueError, match="0.5"):
        should_interrupt({"interruption": True, "confidence": 0.9}, 0.6)
