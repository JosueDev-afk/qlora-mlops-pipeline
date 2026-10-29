"""MLflow runs for pipeline stages: what ran, on which commit, with what result.

Every stage opens one run with its params, metrics and manifest, tagged with
the git commit, so a number in a report traces back to the code and data
that produced it (lineage is the deliverable, CLAUDE.md).

Tracking is optional by design: without MLFLOW_TRACKING_URI (unit tests, CI,
a laptop without the stack) or without mlflow installed, `stage_run` yields a
recorder that does nothing. A stage never fails because tracking is down.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.common.log import get_logger

REPO = Path(__file__).resolve().parents[3]
log = get_logger(__name__)


def git_commit(repo: Path = REPO) -> str | None:
    """HEAD's commit read from .git directly: the Airflow container cannot run git."""
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if not ref.startswith("ref: "):
        return ref
    name = ref.removeprefix("ref: ")
    loose = repo / ".git" / name
    if loose.is_file():
        return loose.read_text().strip()
    packed = repo / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + name):
                return line.split()[0]
    return None


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dicts to dotted keys; MLflow params and metrics are flat."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


class Recorder:
    """Collects a stage's results; sends them to MLflow when a run is active."""

    def __init__(self, mlflow: Any | None) -> None:
        self._mlflow = mlflow

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    def params(self, values: Mapping[str, Any]) -> None:
        if self._mlflow:
            flat = {k: str(v)[:500] for k, v in flatten(values).items()}
            self._mlflow.log_params(flat)

    def metrics(self, values: Mapping[str, Any]) -> None:
        if self._mlflow:
            numeric = {k: float(v) for k, v in flatten(values).items()
                       if isinstance(v, int | float) and not isinstance(v, bool)}  # fmt: skip
            self._mlflow.log_metrics(numeric)

    def artifact(self, values: Mapping[str, Any], name: str) -> None:
        if self._mlflow:
            self._mlflow.log_dict(dict(values), name)


def _mlflow() -> Any | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


@contextmanager
def stage_run(stage: str, *, experiment: str = "pipeline") -> Iterator[Recorder]:
    """One MLflow run per stage, timed; a no-op when tracking is unavailable."""
    mlflow = _mlflow()
    started = time.perf_counter()
    if mlflow is None:
        yield Recorder(None)
        return
    try:
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=stage)
    except Exception as exc:  # noqa: BLE001 -- tracking down must not fail the stage
        log.warning("tracking_unavailable", stage=stage, error=repr(exc))
        yield Recorder(None)
        return
    with run:
        mlflow.set_tags({"stage": stage, "git_commit": git_commit() or "unknown"})
        recorder = Recorder(mlflow)
        try:
            yield recorder
        finally:
            mlflow.log_metric("duration_s", round(time.perf_counter() - started, 3))


def log_stage(
    stage: str,
    *,
    seconds: float,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifact: Mapping[str, Any] | None = None,
    experiment: str = "pipeline",
) -> None:
    """Record a finished stage in one call: its wall time, inputs and results."""
    with stage_run(stage, experiment=experiment) as tracked:
        tracked.params(params or {})
        tracked.metrics({**(metrics or {}), "stage_seconds": round(seconds, 3)})
        if artifact:
            tracked.artifact(artifact, f"{stage}.json")
