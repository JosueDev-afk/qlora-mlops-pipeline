"""Baselines on the gold test split: the numbers the fine-tuned models must beat.

Orchestration only. The rule-based normalizers (Task A) and majority classes
(Tasks B and D), logged to MLflow (experiment "evaluation"). Laya's zero-shot
baseline runs on the Mac (src/pipeline/evaluate/laya_zero_shot.py): the
Airflow image does not carry torch.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.pipeline.evaluate import baseline

DEFAULT_ARGS = {"owner": "ml", "retries": 0}


@dag(
    dag_id="evaluate_baselines",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["evaluation"],
)
def evaluate_baselines():
    @task
    def rules_and_majority() -> dict:
        return baseline.run(params_path="params.yaml")

    rules_and_majority()


evaluate_baselines()
