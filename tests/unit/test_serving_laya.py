"""Laya server rules that hold without a GPU: calibration, pinning, checkpoint.

Laya itself is never imported here; the router and agent are fakes that
record what the server does to them. `build_router` (the only function that
loads a checkpoint) is covered on Colab, not in unit tests.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.serving.laya import (
    Calibration,
    CalibrationError,
    PinnedRouter,
    apply_calibration,
    bucket,
    check_checkpoint,
    load_calibration,
)

CALIBRATION = Calibration("cal-run-7", {"noul:2": 1.8, "choice:3-5": 2.2})
NOUL = {"type": "noul", "instructions": "¿Quiere tomar el turno?"}
FIVE_WAY = {
    "type": "choice",
    "instructions": "¿De qué tipo?",
    "criteria": dict.fromkeys("abcde", ""),
}


class FakeRouter:
    loaded = ["multilingual"]

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, state: Any, questions: dict, model: str | None = None) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions, "model": model})
        return {"answers": {}, "routing": {"model": model}}


class FakeAgent:
    def __init__(self) -> None:
        self.temperature = [0.9, 0.9, 0.9]
        self.temperature_by_options = {"choice:11+": 0.5, "noul:2": 0.7}
        self.lang_temperatures = {"es": {"temperature": [2.0, 2.0, 2.0]}}


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "laya_calibration.json"
    path.write_text(json.dumps(data))
    return path


# ── calibration file ───────────────────────────────────────────────────────


def test_loads_run_id_and_temperatures(tmp_path: Path) -> None:
    path = _write(tmp_path, {"run_id": "cal-run-7", "temperatures": {"noul:2": 1.8},
                             "thresholds": {"call_rejected": 0.41}})  # fmt: skip
    assert load_calibration(path) == Calibration("cal-run-7", {"noul:2": 1.8})


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"temperatures": {"noul:2": 1.8}}, "run_id"),
        ({"run_id": "r", "temperatures": {}}, "temperatures"),
        ({"run_id": "r", "temperatures": {"noul:4": 1.8}}, "bucket"),  # n_options, not a bucket
        ({"run_id": "r", "temperatures": {"choice:3": 1.8}}, "bucket"),
        ({"run_id": "r", "temperatures": {"noul:2": 0.2}}, "outside"),  # Laya would clamp it
        ({"run_id": "r", "temperatures": {"noul:2": 7.0}}, "outside"),
        ({"run_id": "r", "temperatures": {"noul:2": "1.8"}}, "outside"),
    ],
)
def test_calibration_laya_would_alter_is_refused(tmp_path: Path, data: dict, match: str) -> None:
    with pytest.raises(CalibrationError, match=match):
        load_calibration(_write(tmp_path, data))


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (NOUL, "noul:2"),
        ({"type": "choice", "criteria": ["a", "b"]}, "choice:2"),
        (FIVE_WAY, "choice:3-5"),
        ({"type": "choice", "criteria": dict.fromkeys("abcdef", "")}, "choice:6-10"),
        ({"type": "score", "criteria": list("abcdefghijk")}, "score:11+"),
    ],
)
def test_bucket_matches_laya(question: dict, expected: str) -> None:
    assert bucket(question) == expected


def test_fitted_temperatures_replace_the_checkpoint_ones() -> None:
    agent = FakeAgent()
    apply_calibration(agent, CALIBRATION)
    assert agent.temperature == [1.0, 1.0, 1.0]  # no unfitted per-type fallback
    assert agent.temperature_by_options == CALIBRATION.temperatures  # choice:11+ dropped
    assert agent.lang_temperatures == {}


def test_uncalibrated_means_neutral_not_the_checkpoint_ones() -> None:
    agent = FakeAgent()
    apply_calibration(agent, None)
    assert agent.temperature == [1.0, 1.0, 1.0]
    assert agent.temperature_by_options == {}


# ── pinning ────────────────────────────────────────────────────────────────


def test_every_request_is_pinned_to_multilingual_and_stamped() -> None:
    router = FakeRouter()
    result = PinnedRouter(router, CALIBRATION).predict("hola", {"q": NOUL}, model="english")
    assert router.calls[0]["model"] == "multilingual"
    assert result["calibration"] == {"run_id": "cal-run-7"}


def test_unfitted_bucket_is_refused_before_inference() -> None:
    router = FakeRouter()
    six_way = {"type": "choice", "instructions": "?", "criteria": dict.fromkeys("abcdef", "")}
    with pytest.raises(ValueError, match="choice:6-10"):
        PinnedRouter(router, CALIBRATION).predict("hola", {"q": NOUL, "k": six_way})
    assert router.calls == []


def test_uncalibrated_server_stamps_none() -> None:
    result = PinnedRouter(FakeRouter(), None).predict("hola", {"q": FIVE_WAY})
    assert result["calibration"] is None


def test_health_sees_what_is_loaded() -> None:
    assert PinnedRouter(FakeRouter(), None).loaded == ["multilingual"]


@pytest.mark.parametrize(
    "checkpoint",
    ["convaiinnovations/laya", "Convaiinnovations/Laya/", "convaiinnovations/laya-typed-decisions"],
)
def test_english_checkpoints_are_refused(checkpoint: str) -> None:
    with pytest.raises(ValueError, match="English"):
        check_checkpoint(checkpoint)


@pytest.mark.parametrize("checkpoint", ["convaiinnovations/laya-multilingual", "models/laya"])
def test_multilingual_and_fine_tuned_checkpoints_are_accepted(checkpoint: str) -> None:
    check_checkpoint(checkpoint)
