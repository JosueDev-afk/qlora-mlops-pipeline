"""Fine-tune the decision model (Tasks B and D), then hand off to calibration.

Runs in parallel with train_qlora against the SAME gold dataset tag, so any
performance difference between the two models is attributable to the model and
not to the data. That is what makes Hypothesis 1 a fair comparison.

Calibration is its own DAG (calibrate_laya) so temperatures can be re-fitted
for an existing run without retraining.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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

    run_id = fine_tune()
    run_id >> TriggerDagRunOperator(
        task_id="trigger_calibrate_laya",
        trigger_dag_id="calibrate_laya",
        conf={"run_id": "{{ ti.xcom_pull(task_ids='fine_tune') }}"},
    )


train_laya()
