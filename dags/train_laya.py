"""Fine-tune the decision model (Tasks B and D) and calibrate it.

Runs in parallel with train_qlora against the SAME gold dataset tag, so any
performance difference between the two models is attributable to the model and
not to the data. That is what makes Hypothesis 1 a fair comparison.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.pipeline.calibrate import fit_temperature
from src.pipeline.train import laya

DEFAULT_ARGS = {"owner": "ml", "retries": 1}


@dag(
    dag_id="train_laya",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["training", "decision-model"],
)
def train_laya():
    @task
    def fine_tune() -> str:
        """Fine-tune laya-multilingual on Tasks B and D.

        The base checkpoints sit near chance on typed decisions zero-shot; the
        capability comes from this step. Returns the MLflow run id.

        The gold tag lives in `gold.tag` and is read from params.yaml, not
        passed in: a literal here could drift from the tag train_qlora uses and
        invalidate Hypothesis 1 without anything failing.
        """
        return laya.train(params_path="params.yaml")

    @task
    def calibrate(run_id: str) -> str:
        """Fit one temperature per (question type, option count) on held-out data.

        Fails the DAG if ECE stays above `calibration.max_ece`: an uncalibrated
        model cannot be used to gate `handle_rejection`.
        """
        return fit_temperature.run(run_id=run_id, params_path="params.yaml")

    calibrate(fine_tune())


train_laya()
