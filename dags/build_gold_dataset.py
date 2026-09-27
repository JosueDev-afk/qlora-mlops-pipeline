"""Build, version and tag the gold dataset.

Orchestration only: every task delegates to src/pipeline. Keeping logic out of
the DAG means the whole pipeline is unit-testable without running Airflow.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.pipeline.datasets import build_gold, quality_gate, tag_release

DEFAULT_ARGS = {"owner": "data", "retries": 1}


@dag(
    dag_id="build_gold_dataset",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["pipeline", "gold"],
)
def build_gold_dataset():
    @task
    def filter_and_balance() -> str:
        """Schema validation, heuristics, stratified balancing, train/val/test split.

        No LLM-as-judge: examples are generated from their labels, so a judge
        would add cost without adding information.
        """
        return build_gold.run(params_path="params.yaml")

    @task
    def validate(gold_path: str) -> str:
        """Great Expectations gate. Fails the DAG rather than shipping bad data."""
        quality_gate.assert_expectations(gold_path)
        return gold_path

    @task
    def version(gold_path: str) -> str:
        """dvc commit + push; returns the gold set's hash.

        The git tag (gold.tag, read by both trainings) is created on the host
        with `make tag-gold`: the container has no business committing to the
        maintainer's branch.
        """
        return tag_release.publish(gold_path)

    version(validate(filter_and_balance()))


build_gold_dataset()
