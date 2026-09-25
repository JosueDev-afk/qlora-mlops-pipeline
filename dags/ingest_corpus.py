"""Download the pinned public corpora into bronze.

Orchestration only: every task delegates to src/pipeline/ingest. One mapped
task per verified source, so a slow or failing download retries alone.
Sources whose license is not verified yet are logged and skipped.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.pipeline.ingest import run

DEFAULT_ARGS = {"owner": "data", "retries": 2, "retry_delay": pendulum.duration(minutes=2)}


@dag(
    dag_id="ingest_corpus",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["pipeline", "bronze"],
)
def ingest_corpus():
    @task
    def list_sources() -> list[str]:
        return run.verified_sources(params_path="params.yaml")

    @task(max_active_tis_per_dag=2)  # two downloads at a time on a 16 GB laptop
    def ingest(source: str) -> str:
        """Idempotent: a source already in bronze at its pin is not downloaded again."""
        return run.ingest_named(source, params_path="params.yaml")

    ingest.expand(source=list_sources())


ingest_corpus()
