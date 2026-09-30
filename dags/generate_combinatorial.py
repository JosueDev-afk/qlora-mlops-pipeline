"""Generate the code-built corpus: examples written from their labels.

Orchestration only: the task delegates to src/pipeline/synth. No LLM runs here;
the carrier-phrase bank (the only LLM-written text) is built and reviewed
separately.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.pipeline.synth import run

DEFAULT_ARGS = {"owner": "data", "retries": 1}


@dag(
    dag_id="generate_combinatorial",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["pipeline", "synthetic"],
)
def generate_combinatorial():
    @task
    def generate() -> str:
        """Deterministic for synth.seed and the templates file."""
        return run.run(params_path="params.yaml")

    generate()


generate_combinatorial()
