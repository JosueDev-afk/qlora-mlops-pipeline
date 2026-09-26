"""The agent's side of `models/laya_calibration.json`: which run, which threshold.

The Laya server applies the temperatures; the agent needs only the run id,
to check every answer was calibrated by the same run, and the call_rejected
threshold that run derived. Both come from one file so they cannot drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CALIBRATION_PATH = Path("models/laya_calibration.json")


@dataclass(frozen=True)
class AgentCalibration:
    run_id: str
    call_rejected_threshold: float


def load_agent_calibration(path: Path = CALIBRATION_PATH) -> AgentCalibration:
    """Fails at start-up, never mid-call, when the file cannot gate a rejection."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run calibrate_laya (make calibrate RUN_ID=...) first"
        )
    data = json.loads(path.read_text())
    run_id = data.get("run_id")
    threshold = (data.get("thresholds") or {}).get("call_rejected")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{path}: no run_id")
    if not isinstance(threshold, int | float) or not 0.0 < threshold < 1.0:
        raise ValueError(f"{path}: thresholds.call_rejected must be in (0, 1), got {threshold!r}")
    return AgentCalibration(run_id, float(threshold))
