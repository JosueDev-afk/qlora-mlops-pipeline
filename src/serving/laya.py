"""Laya-multilingual server: the model card's three rules, enforced at start-up.

Laya ships its own HTTP server (`laya.serve`, `POST /v1/systemone`), and this
module reuses it rather than writing another. What it adds is a router that
cannot break the rules:

- **Multilingual only.** `Router(preload=True)` builds all three checkpoints,
  English root included, and routes short Latin-script text ("sí, ahí estaré")
  to English by default. Here only the multilingual checkpoint is built, and
  every request is pinned to it whatever the client asks for.
- **Preloaded.** The checkpoint is built before the first request is accepted.
- **Calibrated.** The temperatures fitted by `calibrate_laya` replace the
  checkpoint's own and are applied to the logits by Laya itself, one per
  (question type, option bucket). Every response carries the calibration's
  `run_id`, and a question whose bucket was never fitted is refused rather
  than answered with an uncalibrated probability.

Everything above `build_router` is pure and imports neither laya nor torch, so
it is unit-tested without a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.log import configure_logging, get_logger

CHECKPOINT_NAME = "multilingual"
# English encoders: the root collapses on Spanish while staying confident, and
# typed-decisions is the same ModernBERT fine-tuned on English workflows.
FORBIDDEN_CHECKPOINTS = frozenset(
    {"convaiinnovations/laya", "convaiinnovations/laya-typed-decisions"}
)
# Laya's temperature buckets (laya.common.temp_bucket) and the range it clamps
# fitted values to. A value outside the range would be clamped silently, so
# the calibration no longer being the one that was fitted; refuse it instead.
BUCKET_RE = re.compile(r"^(choice|score|noul):(2|3-5|6-10|11\+)$")
TEMPERATURE_RANGE = (0.5, 5.0)
NEUTRAL = [1.0, 1.0, 1.0]

log = get_logger(__name__)


class CalibrationError(ValueError):
    """The calibration file cannot be applied as fitted."""


@dataclass(frozen=True)
class Calibration:
    """What `calibrate_laya` writes that the server needs: the run and its temperatures."""

    run_id: str
    temperatures: dict[str, float]


def load_calibration(path: Path) -> Calibration:
    """Read `models/laya_calibration.json`, failing on anything Laya would alter."""
    data = json.loads(Path(path).read_text())
    run_id = data.get("run_id")
    temperatures = data.get("temperatures")
    if not isinstance(run_id, str) or not run_id:
        raise CalibrationError(f"{path}: no run_id")
    if not isinstance(temperatures, dict) or not temperatures:
        raise CalibrationError(f"{path}: no temperatures")
    lo, hi = TEMPERATURE_RANGE
    for bucket, value in temperatures.items():
        if not BUCKET_RE.match(bucket):
            raise CalibrationError(f"{path}: {bucket!r} is not a Laya bucket (e.g. 'noul:2')")
        if not isinstance(value, int | float) or not math.isfinite(value) or not lo <= value <= hi:
            raise CalibrationError(f"{path}: {bucket} = {value!r} outside [{lo}, {hi}]")
    return Calibration(run_id, {k: float(v) for k, v in temperatures.items()})


def bucket(question: dict[str, Any]) -> str:
    """The temperature bucket Laya will use for this question."""
    qtype = question.get("type")
    n = 2 if qtype == "noul" else len(question.get("criteria") or ())
    size = "2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"
    return f"{qtype}:{size}"


def apply_calibration(agent: Any, calibration: Calibration | None) -> None:
    """Replace the checkpoint's temperatures with the fitted ones, or neutral ones.

    A fine-tuned checkpoint may carry temperatures from its training notebook,
    fitted on synthetic data; they must never stack with ours (H6). Per-type
    and per-language temperatures are reset so no unfitted value applies.
    """
    agent.temperature = list(NEUTRAL)
    agent.temperature_by_options = dict(calibration.temperatures) if calibration else {}
    agent.lang_temperatures = {}


def check_checkpoint(checkpoint: str) -> None:
    if checkpoint.strip().rstrip("/").lower() in FORBIDDEN_CHECKPOINTS:
        raise ValueError(f"{checkpoint!r} is an English checkpoint; serve laya-multilingual")


class PinnedRouter:
    """The router `laya.serve.create_app` sees: one checkpoint, stamped answers.

    The client's `model` field is ignored, so no request can build the English
    checkpoint. ValueError is what laya.serve turns into a 422 naming the
    problem, which is what an unfitted bucket should be.
    """

    def __init__(self, router: Any, calibration: Calibration | None) -> None:
        self._router = router
        self._calibration = calibration

    @property
    def loaded(self) -> list[str]:
        return list(self._router.loaded)

    def predict(
        self, state: Any, questions: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if self._calibration is not None:
            unfitted = sorted(
                {bucket(q) for q in questions.values() if isinstance(q, dict)}
                - set(self._calibration.temperatures)
            )
            if unfitted:
                raise ValueError(f"no fitted temperature for {', '.join(unfitted)}")
        result = self._router.predict(state, questions, model=CHECKPOINT_NAME, **kwargs)
        result["calibration"] = {"run_id": self._calibration.run_id} if self._calibration else None
        return result


def build_router(
    checkpoint: str, *, calibration: Calibration | None, device: str | None = None
) -> PinnedRouter:
    """Build and preload the multilingual checkpoint before any request."""
    from laya import Router

    check_checkpoint(checkpoint)
    router = Router(
        models={CHECKPOINT_NAME: checkpoint}, default=CHECKPOINT_NAME, max_loaded=1, device=device
    )
    router.preload([CHECKPOINT_NAME])
    apply_calibration(router.load(CHECKPOINT_NAME), calibration)
    return PinnedRouter(router, calibration)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve laya-multilingual, preloaded, calibrated.")
    parser.add_argument("--checkpoint", default="models/laya", help="fine-tuned dir or hub id")
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument("--calibration", default="models/laya_calibration.json")
    calibration_group.add_argument(
        "--uncalibrated",
        action="store_true",
        help="exploration only: neutral temperatures, and the agent's client refuses the answers",
    )
    parser.add_argument("--device", default=None, help="cuda, mps or cpu; auto when omitted")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)

    configure_logging()
    calibration = None if args.uncalibrated else load_calibration(Path(args.calibration))
    router = build_router(args.checkpoint, calibration=calibration, device=args.device)
    log.info(
        "laya_ready",
        checkpoint=args.checkpoint,
        calibration=calibration.run_id if calibration else None,
        device=args.device or "auto",
    )

    import uvicorn
    from laya.serve import create_app

    uvicorn.run(create_app(router), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
