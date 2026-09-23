"""Fit Laya's temperatures on the human calibration split and gate on ECE.

Separate from train_laya so temperatures can be re-fitted for an existing run
(`make calibrate RUN_ID=...`) without retraining. Laya ships over-confident:
without this step `rejection_threshold` has no meaning (H6).
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.models.param import Param

from src.pipeline.calibrate import fit_temperature

DEFAULT_ARGS = {"owner": "ml", "retries": 1}


@dag(
    dag_id="calibrate_laya",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={"run_id": Param(type="string", description="MLflow run id produced by train_laya")},
    tags=["training", "decision-model", "calibration"],
)
def calibrate_laya():
    @task
    def calibrate(params: dict | None = None) -> str:
        """Fit one temperature per (question type, option count).

        Uses `calibration.split`, which is human-annotated: gold is mostly
        synthetic, and a calibration fitted there may not transfer to real
        speech. Fails the DAG if ECE stays above `calibration.max_ece`, because
        an uncalibrated model cannot gate `handle_rejection`.
        """
        assert params is not None
        return fit_temperature.run(run_id=params["run_id"], params_path="params.yaml")

    calibrate()


calibrate_laya()
