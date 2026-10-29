"""Call telemetry in Spark: bronze JSONL -> silver Parquet -> warehouse star schema.

The analytics half of the streaming path (Phase 2: Kafka -> Spark -> lake ->
warehouse), run in batch over the archived stream for the MVP: the events
and schema are the ones the agent produces (schemas/events/call_event.schema.json).

- silver: the explicit schema below (inference over a million events is slow
  and guesses), de-duplicated by event_id, timestamps typed, partitioned by
  date and event type.
- warehouse: dim_call (one row per call) and fact_turn (one row per caller
  turn, the grain the Phase 2 design names), plus the KPI tables the
  dashboards read: latency against the 500/800 ms budgets, valid JSON,
  outcomes, barge-ins, validated fields.

Spark runs in local mode inside the Airflow image (the Mac has no JVM); the
master and driver memory come from params.yaml (spark.*).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import get_logger
from src.pipeline.common.tracking import stage_run

BRONZE = Path("data/bronze/telemetry")
SILVER = Path("data/silver/telemetry")
WAREHOUSE = Path("data/warehouse")
BUDGET_MS = {"end_to_end": 800, "generative_json": 500}

log = get_logger(__name__)


def spark_session(params: dict[str, Any], app: str) -> Any:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app)
        .master(str(param(params, "spark.master")))
        .config("spark.driver.memory", str(param(params, "spark.driver_memory")))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")  # local mode: 200 would be 200 tiny tasks
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def event_schema() -> Any:
    """Every payload field any event type carries; absent ones read as null."""
    from pyspark.sql import types as T  # noqa: N812 -- the PySpark idiom

    output = T.StructType(
        [
            T.StructField("intent", T.StringType()),
            T.StructField("confidence", T.DoubleType()),
            T.StructField("interruption", T.BooleanType()),
            T.StructField("kind", T.StringType()),
            T.StructField("field", T.StringType()),
            T.StructField("normalized_value", T.StringType()),
        ]
    )
    payload = T.StructType(
        [
            T.StructField("turn_idx", T.IntegerType()),
            T.StructField("speaker", T.StringType()),
            T.StructField("text", T.StringType()),
            T.StructField("asr_confidence", T.DoubleType()),
            T.StructField("task", T.StringType()),
            T.StructField("latency_ms", T.DoubleType()),
            T.StructField("json_valid", T.BooleanType()),
            T.StructField("prompt_version", T.IntegerType()),
            T.StructField("tool", T.StringType()),
            T.StructField("field", T.StringType()),
            T.StructField("duration_ms", T.DoubleType()),
            T.StructField("end_to_end_ms", T.DoubleType()),
            T.StructField("endpointing_ms", T.DoubleType()),
            T.StructField("interrupted", T.BooleanType()),
            T.StructField("p_real", T.DoubleType()),
            T.StructField("outcome", T.StringType()),
            T.StructField("turns", T.IntegerType()),
            T.StructField("flow", T.StringType()),
            T.StructField("contact_id", T.StringType()),
            T.StructField("model_version", T.StringType()),
            T.StructField("prompt_hash", T.StringType()),
            T.StructField("synthetic", T.BooleanType()),
            T.StructField("output", output),
        ]
    )
    return T.StructType(
        [
            T.StructField("schema_version", T.IntegerType()),
            T.StructField("event_id", T.StringType()),
            T.StructField("call_id", T.StringType()),
            T.StructField("seq", T.LongType()),
            T.StructField("ts", T.StringType()),
            T.StructField("type", T.StringType()),
            T.StructField("payload", payload),
        ]
    )


def _bronze_manifests(bronze: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(bronze.glob("*/_manifest.json"))]


def bronze_to_silver(spark: Any, bronze: Path, silver: Path) -> dict[str, Any]:
    """Typed, de-duplicated events; returns what came in and what went out."""
    from pyspark.sql import functions as F  # noqa: N812 -- the PySpark idiom

    started = time.perf_counter()
    raw = spark.read.schema(event_schema()).json(str(bronze / "*" / "part-*.jsonl"))
    read = raw.count()
    valid = raw.where(
        (F.col("schema_version") == 1) & F.col("event_id").isNotNull() & F.col("type").isNotNull()
    )
    events = (
        valid.dropDuplicates(["event_id"])
        .withColumn("event_ts", F.to_timestamp("ts"))
        .withColumn("event_date", F.to_date("event_ts"))
        .select(
            "event_id",
            "call_id",
            "seq",
            "event_ts",
            "event_date",
            "type",
            *[
                F.col(f"payload.{f}").alias(f)
                for f in (
                    "turn_idx",
                    "speaker",
                    "text",
                    "asr_confidence",
                    "task",
                    "latency_ms",
                    "json_valid",
                    "prompt_version",
                    "tool",
                    "field",
                    "duration_ms",
                    "end_to_end_ms",
                    "endpointing_ms",
                    "interrupted",
                    "p_real",
                    "outcome",
                    "turns",
                    "flow",
                    "contact_id",
                    "model_version",
                    "prompt_hash",
                    "synthetic",
                )
            ],
            *[
                F.col(f"payload.output.{f}").alias(f"output_{f}")
                for f in ("intent", "confidence", "interruption", "kind")
            ],
        )
    )
    events.write.mode("overwrite").partitionBy("event_date", "type").parquet(str(silver))
    written = spark.read.parquet(str(silver)).count()
    seconds = time.perf_counter() - started
    return {
        "rows_read": read,
        "rows_invalid": read - valid.count(),
        "rows_duplicate": valid.count() - written,
        "rows_written": written,
        "seconds": round(seconds, 2),
        "rows_per_second": round(read / seconds) if seconds else 0,
    }


def _percentiles(column: str, name: str) -> list[Any]:
    from pyspark.sql import functions as F  # noqa: N812 -- the PySpark idiom

    return [
        F.percentile_approx(column, 0.5).alias(f"{name}_p50"),
        F.percentile_approx(column, 0.95).alias(f"{name}_p95"),
    ]


def silver_to_warehouse(spark: Any, silver: Path, warehouse: Path) -> dict[str, Any]:
    """Star schema and KPI tables; returns the KPI summary row."""
    from pyspark.sql import functions as F  # noqa: N812 -- the PySpark idiom

    started = time.perf_counter()
    ev = spark.read.parquet(str(silver))
    key = ["call_id", "turn_idx"]

    started_calls = ev.where(F.col("type") == "call_started").select(
        "call_id",
        "flow",
        "contact_id",
        "model_version",
        "prompt_hash",
        "synthetic",
        F.col("event_ts").alias("started_at"),
    )
    ended = ev.where(F.col("type") == "call_ended").select(
        "call_id", "outcome", "turns", F.col("event_ts").alias("ended_at")
    )
    barge = (
        ev.where(F.col("type") == "barge_in")
        .groupBy("call_id")
        .agg(
            F.count("*").alias("barge_ins"),
            F.sum(F.col("interrupted").cast("int")).alias("interruptions"),
        )
    )
    tools = ev.where(F.col("type") == "tool_invoked")
    fields = tools.groupBy("call_id").agg(
        F.sum((F.col("tool") == "update_contact").cast("int")).alias("fields_validated"),
        F.sum((F.col("tool") == "mark_unvalidated").cast("int")).alias("fields_unvalidated"),
    )
    dim_call = (
        started_calls.join(ended, "call_id", "left")
        .join(barge, "call_id", "left")
        .join(fields, "call_id", "left")
        .fillna(0, ["barge_ins", "interruptions", "fields_validated", "fields_unvalidated"])
        .withColumn(
            "duration_s", F.col("ended_at").cast("double") - F.col("started_at").cast("double")
        )
        .withColumn("started_hour", F.date_trunc("hour", "started_at"))
    )

    person = ev.where((F.col("type") == "turn_started") & (F.col("speaker") == "person")).select(
        *key, F.col("event_ts").alias("turn_started_at"), "event_date"
    )
    asr = ev.where(F.col("type") == "asr_final").select(
        *key, "asr_confidence", F.length("text").alias("transcript_chars")
    )
    partials = (
        ev.where(F.col("type") == "asr_partial")
        .groupBy(*key)
        .agg(F.count("*").alias("asr_partials"))
    )
    decision = ev.where(
        (F.col("type") == "decision") & (F.col("task") == "classify_intent")
    ).select(
        *key,
        F.col("output_intent").alias("intent"),
        F.col("output_confidence").alias("intent_confidence"),
        F.col("latency_ms").alias("decision_ms"),
    )
    llm = ev.where(F.col("type") == "llm_response").select(
        *key, F.col("task").alias("llm_task"), "json_valid", F.col("latency_ms").alias("llm_ms")
    )
    latency = ev.where(F.col("type") == "latency").select(*key, "end_to_end_ms", "endpointing_ms")
    turn_tools = tools.select(*key, "tool", F.col("field").alias("tool_field"))
    fact_turn = (
        person.join(asr, key, "left")
        .join(partials, key, "left")
        .join(decision, key, "left")
        .join(llm, key, "left")
        .join(latency, key, "left")
        .join(turn_tools, key, "left")
        .withColumn("within_e2e_budget", F.col("end_to_end_ms") <= BUDGET_MS["end_to_end"])
        .withColumn("within_llm_budget", F.col("llm_ms") <= BUDGET_MS["generative_json"])
    )

    dim_call.write.mode("overwrite").parquet(str(warehouse / "dim_call"))
    fact_turn.write.mode("overwrite").partitionBy("event_date").parquet(
        str(warehouse / "fact_turn")
    )
    dim_call = spark.read.parquet(str(warehouse / "dim_call"))
    fact_turn = spark.read.parquet(str(warehouse / "fact_turn"))

    calls = (
        dim_call.agg(
            F.count("*").alias("calls"),
            F.avg((F.col("outcome") == "completed").cast("double")).alias("completed_rate"),
            F.avg((F.col("outcome") == "partial").cast("double")).alias("partial_rate"),
            F.avg((F.col("outcome") == "rejected").cast("double")).alias("rejected_rate"),
            F.avg("duration_s").alias("call_duration_s_mean"),
            F.avg("turns").alias("turns_mean"),
            (F.sum("interruptions") / F.sum("barge_ins")).alias("barge_in_stop_rate"),
            (
                F.sum("fields_validated")
                / (F.sum("fields_validated") + F.sum("fields_unvalidated"))
            ).alias("field_validation_rate"),
        )
        .first()
        .asDict()
    )
    turns = (
        fact_turn.agg(
            F.count("*").alias("caller_turns"),
            *_percentiles("end_to_end_ms", "end_to_end_ms"),
            *_percentiles("decision_ms", "decision_ms"),
            *_percentiles("llm_ms", "llm_ms"),
            F.avg(F.col("within_e2e_budget").cast("double")).alias("e2e_budget_rate"),
            F.avg(F.col("within_llm_budget").cast("double")).alias("llm_budget_rate"),
            F.avg(F.col("json_valid").cast("double")).alias("json_valid_rate"),
            F.avg("asr_confidence").alias("asr_confidence_mean"),
        )
        .first()
        .asDict()
    )
    summary = {
        k: (round(v, 4) if isinstance(v, float) else v) for k, v in {**calls, **turns}.items()
    }
    # JSON, not Parquet: a KPI can be null (no barge-ins), and a one-row frame
    # cannot infer the type of a null column.
    (warehouse / "kpi_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    hourly = (
        fact_turn.join(dim_call.select("call_id", "started_hour"), "call_id")
        .groupBy("started_hour")
        .agg(
            F.countDistinct("call_id").alias("calls"),
            F.count("*").alias("caller_turns"),
            *_percentiles("end_to_end_ms", "end_to_end_ms"),
            F.avg(F.col("json_valid").cast("double")).alias("json_valid_rate"),
        )
        .orderBy("started_hour")
    )
    hourly.write.mode("overwrite").parquet(str(warehouse / "kpi_hourly"))
    by_intent = fact_turn.groupBy("intent").agg(
        F.count("*").alias("turns"), *_percentiles("end_to_end_ms", "end_to_end_ms")
    )
    by_intent.write.mode("overwrite").parquet(str(warehouse / "kpi_intent"))

    summary["fact_turn_rows"] = fact_turn.count()
    summary["dim_call_rows"] = dim_call.count()
    summary["seconds"] = round(time.perf_counter() - started, 2)
    return summary


QUERIES = {
    "p95_latency_by_outcome": """
        SELECT c.outcome, percentile_approx(t.end_to_end_ms, 0.95) AS p95_ms, count(*) AS turns
        FROM fact_turn t JOIN dim_call c USING (call_id) GROUP BY c.outcome""",
    "rejections_by_hour": """
        SELECT started_hour, avg(CASE WHEN outcome = 'rejected' THEN 1 ELSE 0 END) AS rate
        FROM dim_call GROUP BY started_hour ORDER BY started_hour""",
    "invalid_json_by_field": """
        SELECT tool_field, avg(CASE WHEN json_valid THEN 0 ELSE 1 END) AS invalid_rate,
               count(*) AS n
        FROM fact_turn WHERE llm_task IS NOT NULL GROUP BY tool_field""",
}


def time_queries(spark: Any, warehouse: Path, repeats: int = 3) -> dict[str, float]:
    """Median wall time of the dashboards' questions: the cost of an analytical query."""
    for table in ("fact_turn", "dim_call"):
        spark.read.parquet(str(warehouse / table)).createOrReplaceTempView(table)
    out = {}
    for name, sql in QUERIES.items():
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            spark.sql(sql).collect()
            times.append((time.perf_counter() - started) * 1000)
        out[name] = round(sorted(times)[len(times) // 2], 1)
    return out


def run_silver(
    params_path: str = "params.yaml", *, bronze: Path = BRONZE, silver: Path = SILVER
) -> str:
    params = load_params(params_path)
    manifests = _bronze_manifests(bronze)
    if not manifests:
        raise FileNotFoundError(f"{bronze} has no partitions: run the simulator first")
    with stage_run("telemetry_silver", experiment="analytics") as tracked:
        spark = spark_session(params, "telemetry_silver")
        stats = bronze_to_silver(spark, bronze, silver)
        tracked.params(
            {
                "bronze": {
                    "partitions": len(manifests),
                    "calls": sum(m["calls"] for m in manifests),
                },
                "spark": param(params, "spark"),
            }
        )
        tracked.metrics(stats)
    log.info("telemetry_silver", **stats)
    return str(silver)


def run_warehouse(
    params_path: str = "params.yaml", *, silver: Path = SILVER, warehouse: Path = WAREHOUSE
) -> dict[str, Any]:
    params = load_params(params_path)
    with stage_run("telemetry_warehouse", experiment="analytics") as tracked:
        spark = spark_session(params, "telemetry_warehouse")
        summary = silver_to_warehouse(spark, silver, warehouse)
        queries = time_queries(spark, warehouse)
        tracked.params({"spark": param(params, "spark"), "budget_ms": BUDGET_MS})
        tracked.metrics({**summary, "query_ms": queries})
        tracked.artifact({"kpi_summary": summary, "query_ms": queries}, "kpi_summary.json")
    log.info("telemetry_warehouse", **summary)
    return {**summary, "query_ms": queries}
