"""Curate bronze into silver: normalize, scrub personal data, de-duplicate.

Orchestration only: the task delegates to src/pipeline/curate. Today that is
the replay set (a few thousand rows, plain Python); CallCenterEN's profile
joins it in Spark local mode once its license is decided, which is what the
DAG id refers to.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.pipeline.curate import run

DEFAULT_ARGS = {"owner": "data", "retries": 1}


@dag(
    dag_id="curate_spark",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["pipeline", "silver"],
)
def curate_spark():
    @task
    def curate() -> str:
        """Reads bronze at the pins in params.yaml; never the network."""
        return run.run(params_path="params.yaml")

    curate()


curate_spark()
