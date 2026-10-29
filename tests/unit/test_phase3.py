"""Phase 3 MVP: stage tracking, telemetry in bronze, baselines.

Spark itself needs a JVM, which CI and the Mac lack: the Spark job is
exercised inside the Airflow image (make analytics), and its test here is
skipped unless pyspark and java are both available.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from src.agent.telemetry.simulate import simulate, write_bronze
from src.pipeline.common import tracking
from src.pipeline.common.tracking import Recorder, flatten, git_commit, log_stage, stage_run
from src.pipeline.evaluate.baseline import evaluate, majority, rules_on_task_a
from src.pipeline.evaluate.laya_zero_shot import evaluate as laya_evaluate
from src.pipeline.evaluate.laya_zero_shot import laya_state, macro_f1

START = datetime(2026, 10, 28, 12, tzinfo=UTC)
SCHEMA = Draft202012Validator(json.loads(Path("schemas/events/call_event.schema.json").read_text()))


# ── tracking ───────────────────────────────────────────────────────────────


class FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def set_experiment(self, name: str) -> None:
        self.calls.append(("experiment", name))

    def start_run(self, run_name: str) -> "FakeMlflow":
        self.calls.append(("run", run_name))
        return self

    def __enter__(self) -> "FakeMlflow":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def set_tags(self, tags: dict) -> None:
        self.calls.append(("tags", tags))

    def log_params(self, params: dict) -> None:
        self.calls.append(("params", params))

    def log_metrics(self, metrics: dict) -> None:
        self.calls.append(("metrics", metrics))

    def log_metric(self, key: str, value: float) -> None:
        self.calls.append(("metric", key))

    def log_dict(self, values: dict, name: str) -> None:
        self.calls.append(("artifact", name))


def test_tracking_is_a_no_op_without_a_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with stage_run("x") as tracked:
        tracked.metrics({"rows": 3})
        assert not tracked.active


def test_tracking_logs_flat_numeric_metrics_and_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMlflow()
    monkeypatch.setattr(tracking, "_mlflow", lambda: fake)
    log_stage("build_gold", seconds=1.5, params={"splits": {"train": 0.8}},
              metrics={"rows": {"train": 10}, "label": "skip me"}, artifact={"a": 1})  # fmt: skip
    kinds = dict(fake.calls[:3])
    assert kinds["experiment"] == "pipeline" and kinds["run"] == "build_gold"
    assert "git_commit" in kinds["tags"]
    metrics = next(v for k, v in fake.calls if k == "metrics")
    assert metrics == {"rows.train": 10.0, "stage_seconds": 1.5}  # strings are not metrics
    assert ("params", {"splits.train": "0.8"}) in fake.calls
    assert ("artifact", "build_gold.json") in fake.calls


def test_flatten_and_commit() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}}) == {"a.b": 1, "a.c.d": 2}
    commit = git_commit()
    assert commit is None or len(commit) == 40
    assert Recorder(None).active is False


# ── telemetry in bronze ────────────────────────────────────────────────────


def test_bronze_matches_the_in_memory_stream(tmp_path: Path) -> None:
    out = tmp_path / "ingest_date=2026-10-28"
    manifest = write_bronze(out, 25, seed=3, start=START, rows_per_file=500)
    events = [json.loads(line) for f in sorted(out.glob("part-*.jsonl"))
              for line in f.read_text().splitlines()]  # fmt: skip
    assert len(events) == manifest["events"] and len(manifest["files"]) > 1  # chunked
    expected = simulate(25, seed=3, start=START)
    assert sorted(e["event_id"] for e in events) == sorted(e["event_id"] for e in expected)
    assert all(SCHEMA.is_valid(e) for e in events)
    assert manifest["synthetic"] is True and not list(tmp_path.glob(".*"))


def test_bronze_partition_is_replaced_whole(tmp_path: Path) -> None:
    out = tmp_path / "ingest_date=2026-10-28"
    write_bronze(out, 10, seed=1, start=START)
    (out / "stale.jsonl").write_text("{}\n")
    write_bronze(out, 10, seed=1, start=START)
    assert not (out / "stale.jsonl").exists()


# ── baselines ──────────────────────────────────────────────────────────────


def row(task: str, inputs: dict, output: dict, wer: float = 0.0) -> dict[str, Any]:
    return {"task": task, "input": inputs, "output": output, "meta": {"wer_target": wer}}


def task_a(transcript: str, label: str | None, field: str = "phone", wer: float = 0.0) -> dict:
    return row("extract_entity", {"field": field, "transcript": transcript},
               {"normalized_value": label}, wer)  # fmt: skip


def test_rules_baseline_counts_wrong_values_apart_from_refusals() -> None:
    rows = [
        task_a("ocho uno ocho dos tres cuatro cinco seis siete ocho", "+528182345678"),
        task_a("ocho uno ocho dos tres cuatro cinco seis siete ocho", "+528182345679", wer=0.1),
        task_a("ocho uno siente dos tres cuatro cinco seis siete ocho", "+528172345678", wer=0.1),
        task_a("juan punto perez arroba gmail punto com", "juan.perez@gmail.com", "email"),
        task_a("me llamo juan", "Juan", "name"),  # names have no rule reader
    ]
    results = rules_on_task_a(rows)
    assert results["all"]["n"] == 4
    assert results["phone"]["exact_match"] == pytest.approx(1 / 3, abs=1e-4)
    assert results["phone"]["wrong_value_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert results["wer_0.1"]["coverage"] == 0.5  # "siente" is refused, not guessed
    assert results["email"]["exact_match"] == 1.0


def test_oracle_span_reads_only_the_value() -> None:
    noisy = task_a(
        "con gusto, ocho uno ocho dos tres cuatro cinco seis siete ocho", "+528182345678"
    )
    noisy["output"]["raw_span"] = "ocho uno ocho dos tres cuatro cinco seis siete ocho"
    assert rules_on_task_a([noisy])["all"]["exact_match"] == 0.0  # "con gusto" blocks the rule
    assert rules_on_task_a([noisy], oracle_span=True)["all"]["exact_match"] == 1.0


def test_majority_is_the_floor() -> None:
    rows = [
        row("classify_intent", {}, {"intent": i}) for i in ["confirmed"] * 3 + ["call_rejected"]
    ]
    assert majority(rows, "classify_intent", "intent") == {
        "n": 4, "majority_label": "confirmed", "accuracy": 0.75}  # fmt: skip
    assert evaluate(rows)["task_d_majority"] == {"n": 0}


# ── Laya zero-shot, with a fake scorer ─────────────────────────────────────


def test_laya_sees_the_runtime_state_keys() -> None:
    b = row("classify_intent", {"flow": "f", "context": "c", "transcript": "t"}, {})
    d = row(
        "is_real_interruption", {"agent_utterance": "a", "agent_said": "x", "transcript": "t"}, {}
    )
    assert laya_state(b) == {"flujo": "f", "contexto": "c", "transcripción": "t"}
    assert laya_state(d) == {"agente_dice": "a", "transcripción": "t"}


def test_laya_evaluation_scores_accuracy_f1_and_ece() -> None:
    rows = [row("classify_intent", {"flow": "f", "context": "c", "transcript": "t"},
                {"intent": label}) for label in ("confirmed", "call_rejected")]  # fmt: skip
    rows += [row("is_real_interruption", {"agent_utterance": "a", "transcript": "t"},
                 {"interruption": True})]  # fmt: skip

    def scorer(state: dict, questions: dict) -> dict:
        if "intent" in questions:
            probabilities = {"confirmed": 0.9, "call_rejected": 0.1}
            return {"intent": {"choice": "confirmed", "probabilities": probabilities}}
        return {"interruption": {"noul": 0.8}}

    results = laya_evaluate(rows, scorer)
    assert results["classify_intent"]["accuracy"] == 0.5
    assert results["classify_intent"]["macro_f1"] == pytest.approx(0.3333, abs=1e-4)
    assert results["is_real_interruption"]["accuracy"] == 1.0
    assert 0 <= results["classify_intent"]["ece_raw"] <= 1


def test_macro_f1_perfect_and_empty() -> None:
    assert macro_f1([("a", "a"), ("b", "b")]) == 1.0
    assert macro_f1([]) == 0.0


# ── Spark (only where a JVM exists) ────────────────────────────────────────


def test_spark_telemetry_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    if not shutil.which("java"):
        pytest.skip("no JVM: Spark runs in the Airflow image (make analytics)")
    from src.pipeline.analytics.telemetry import (
        bronze_to_silver,
        silver_to_warehouse,
        spark_session,
    )

    write_bronze(tmp_path / "bronze" / "ingest_date=2026-10-28", 40, seed=2, start=START)
    spark = spark_session({"spark": {"master": "local[2]", "driver_memory": "1g"}}, "test")
    stats = bronze_to_silver(spark, tmp_path / "bronze", tmp_path / "silver")
    assert stats["rows_written"] == stats["rows_read"] and stats["rows_duplicate"] == 0
    summary = silver_to_warehouse(spark, tmp_path / "silver", tmp_path / "warehouse")
    assert summary["calls"] == 40 and summary["dim_call_rows"] == 40
    assert 0 < summary["end_to_end_ms_p95"] < 2000
