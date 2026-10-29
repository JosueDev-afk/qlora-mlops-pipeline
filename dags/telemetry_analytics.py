"""Call telemetry, end to end: simulated calls -> bronze -> Spark silver -> warehouse.

Orchestration only. The batch twin of the streaming path (Kafka -> Spark ->
lake): the same events and schema, archived as JSONL and processed by Spark
in local mode. The warehouse tables (dim_call, fact_turn, KPIs) and every
stage's metrics go to data/warehouse and MLflow (experiment "analytics").
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.agent.telemetry.simulate import simulate_to_bronze
from src.pipeline.analytics import telemetry

DEFAULT_ARGS = {"owner": "data", "retries": 0}


@dag(
    dag_id="telemetry_analytics",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Monterrey"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["analytics", "spark", "streaming"],
)
def telemetry_analytics():
    @task
    def simulate_calls() -> str:
        """telemetry.calls simulated calls (~86 events each) as today's bronze partition."""
        return simulate_to_bronze(params_path="params.yaml")

    @task
    def spark_silver(_bronze: str) -> str:
        """Typed, de-duplicated events, partitioned by date and event type."""
        return telemetry.run_silver(params_path="params.yaml")

    @task
    def spark_warehouse(_silver: str) -> dict:
        """dim_call, fact_turn and the KPI tables; logs KPIs and query times to MLflow."""
        return telemetry.run_warehouse(params_path="params.yaml")

    spark_warehouse(spark_silver(simulate_calls()))


telemetry_analytics()
