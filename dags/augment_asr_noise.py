"""Inject rule-based ASR noise into the generated examples.

Orchestration only: the task delegates to src/pipeline/augment. Rules, not a
model: CPU only, deterministic, and the tables are the ones plan task 10's
ASR error profile calibrates.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.pipeline.augment import run

DEFAULT_ARGS = {"owner": "data", "retries": 1}


@dag(
    dag_id="augment_asr_noise",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["pipeline", "augment"],
)
def augment_asr_noise():
    @task
    def augment() -> str:
        """Variants share their dialogue's group, so build_gold keeps them in one split."""
        return run.run(params_path="params.yaml")

    augment()


augment_asr_noise()
