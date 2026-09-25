"""Temperature scaling for Laya: recovers a known temperature, gates on held-out ECE,
derives the rejection threshold, and writes what the server and agent read.

The data is a synthetic over-confident model: labels are drawn from
softmax(z), the model reports softmax(3z). A correct fit finds T close to 3,
and calibration must bring held-out ECE under the gate.
"""

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.pipeline.calibrate.fit_temperature import (
    CalibrationGateError,
    CalibrationSplitError,
    Example,
    cross_validated_ece,
    ece,
    examples_from,
    fit_t,
    rejection_threshold,
    run,
    temper,
)
from src.serving.laya import load_calibration

# The option order of prompts/classify_intent_laya/v1.yaml.
INTENTS = ["confirmed", "cannot_attend", "call_rejected", "ambiguous", "out_of_scope"]


def softmax(z: list[float]) -> list[float]:
    top = max(z)
    e = [math.exp(v - top) for v in z]
    return [v / sum(e) for v in e]


def overconfident(rng: random.Random, k: int, true_t: float = 3.0) -> tuple[list[float], int]:
    """(reported probabilities, gold index): gold ~ softmax(z), reported softmax(true_t z)."""
    z = [rng.gauss(0, 1.2) for _ in range(k)]
    gold = rng.choices(range(k), weights=softmax(z))[0]
    return [round(p, 4) for p in softmax([true_t * v for v in z])], gold


def examples(n: int, k: int = 5, seed: int = 0, true_t: float = 3.0) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        probs, gold = overconfident(rng, k, true_t)
        out.append(Example(f"r{i}", "classify_intent_laya", "intent", f"choice:{k}",
                           tuple(INTENTS[:k]), tuple(probs), gold))  # fmt: skip
    return out


# ── math ───────────────────────────────────────────────────────────────────


def test_temper_is_identity_at_one_and_flattens_above() -> None:
    p = [0.7, 0.2, 0.1]
    assert temper(p, 1.0) == pytest.approx(p)
    flat = temper(p, 3.0)
    assert flat[0] < 0.7 and flat[2] > 0.1 and sum(flat) == pytest.approx(1.0)


def test_rounded_zeros_are_floored_not_infinite() -> None:
    assert all(v > 0 for v in temper([1.0, 0.0], 2.0))


def test_fit_recovers_a_known_temperature() -> None:
    assert fit_t(examples(3000)) == pytest.approx(3.0, rel=0.1)


def test_fit_stays_inside_the_range_laya_accepts() -> None:
    assert fit_t(examples(1000, true_t=9.0)) <= 5.0


@pytest.mark.parametrize(
    ("scored", "expected"),
    [
        ([(1.0, True)] * 10, 0.0),
        ([(0.9, False)] * 10, 0.9),
        ([(0.75, True)] * 3 + [(0.75, False)], 0.0),
    ],
)
def test_ece(scored: list[tuple[float, bool]], expected: float) -> None:
    assert ece(scored) == pytest.approx(expected)


def test_calibration_brings_held_out_ece_under_the_gate() -> None:
    data = examples(1500)
    raw = ece([(max(e.probs), e.probs.index(max(e.probs)) == e.gold) for e in data])
    assert raw > 0.10
    assert cross_validated_ece(data, folds=5, seed=1) < 0.05


# ── rejection threshold ────────────────────────────────────────────────────


def _binary(scores: list[tuple[float, bool]]) -> list[Example]:
    return [Example(f"r{i}", "classify_intent_laya", "intent", "choice:2", ("other", "rej"),
                    (1 - s, s), 1 if pos else 0) for i, (s, pos) in enumerate(scores)]  # fmt: skip


def test_threshold_is_the_highest_that_reaches_the_target_recall() -> None:
    positives = [(0.9, True), (0.8, True), (0.7, True), (0.2, True)]
    negatives = [(0.85, False), (0.5, False), (0.1, False)]
    result = rejection_threshold(_binary(positives + negatives), 1.0, 1, target_recall=0.75)
    assert result["threshold"] == pytest.approx(0.7)
    assert result["recall"] == 0.75
    assert result["precision"] == 0.75  # 3 of the 4 above 0.7 are rejections
    assert result["positives"] == 4


def test_full_recall_takes_the_lowest_positive() -> None:
    data = _binary([(0.9, True), (0.2, True), (0.1, False)])
    assert rejection_threshold(data, 1.0, 1, target_recall=1.0)["threshold"] == pytest.approx(0.2)


def test_no_rejections_means_no_threshold() -> None:
    with pytest.raises(CalibrationGateError, match="no call_rejected"):
        rejection_threshold(_binary([(0.3, False)]), 1.0, 1, target_recall=0.95)


# ── split and scoring ──────────────────────────────────────────────────────


def _intent_answer(probs: list[float]) -> dict[str, Any]:
    return {"type": "choice", "probabilities": dict(zip(INTENTS, probs, strict=True))}


def test_rows_are_scored_against_their_own_question_file() -> None:
    rows = [
        {"id": "a", "questions": "classify_intent_laya/v1", "state": {"t": "voy manejando"},
         "labels": {"intent": "call_rejected"}},
        {"id": "b", "questions": "is_real_interruption/v2", "state": {"t": "ese no es"},
         "labels": {"interruption": True, "kind": "correction"}},
    ]  # fmt: skip
    seen: list[dict] = []

    def scorer(state: dict, questions: dict) -> dict:
        seen.append(questions)
        if "intent" in questions:
            return {"intent": _intent_answer([0.1, 0.1, 0.6, 0.1, 0.1])}
        return {"interruption": {"type": "noul", "noul": 0.8},
                "kind": {"type": "choice", "probabilities": {
                    "backchannel": 0.1, "correction": 0.6, "objection": 0.1,
                    "rejection": 0.1, "unclear": 0.1}}}  # fmt: skip

    found = {(e.qid, e.bucket, e.gold) for e in examples_from(rows, scorer)}
    assert found == {("intent", "choice:3-5", 2), ("interruption", "noul:2", 1),
                     ("kind", "choice:3-5", 1)}  # fmt: skip
    assert set(seen[1]) == {"interruption", "kind"}  # only the labelled questions are asked


def test_every_annotation_problem_is_reported_at_once() -> None:
    def row(row_id: str, questions: str, labels: dict) -> dict:
        return {"id": row_id, "questions": questions, "state": {}, "labels": labels}

    rows = [
        row("a", "classify_intent_laya/v1", {"intent": "rechazo"}),
        row("b", "classify_intent_laya/v9", {"intent": "confirmed"}),
        row("c", "is_real_interruption/v2", {"urgency": True}),
    ]
    with pytest.raises(CalibrationSplitError) as info:
        examples_from(rows, lambda s, q: {})
    message = str(info.value)
    assert "'rechazo' is not an option" in message
    assert "v9" in message
    assert "no question 'urgency'" in message


# ── end to end ─────────────────────────────────────────────────────────────


def _write_split(root: Path, n: int, seed: int = 0) -> dict[str, list[float]]:
    rng = random.Random(seed)
    table: dict[str, list[float]] = {}
    lines = []
    for i in range(n):
        probs, gold = overconfident(rng, 5)
        table[str(i)] = probs
        lines.append(json.dumps({"id": f"cal-{i}", "questions": "classify_intent_laya/v1",
                                 "state": {"row": str(i)}, "labels": {"intent": INTENTS[gold]}},
                                ensure_ascii=False))  # fmt: skip
    root.mkdir(parents=True, exist_ok=True)
    (root / "intent.jsonl").write_text("\n".join(lines) + "\n")
    return table


def _params(tmp_path: Path, **calibration: Any) -> str:
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["calibration"].update({"split": str(tmp_path / "calibration"), **calibration})
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params, allow_unicode=True))
    return str(path)


def _run(tmp_path: Path, n: int, **calibration: Any) -> tuple[Path, Path]:
    table = _write_split(tmp_path / "calibration", n)
    out, report = tmp_path / "models/laya_calibration.json", tmp_path / "reports/calibration.json"

    def scorer(state: dict, questions: dict) -> dict:
        return {"intent": _intent_answer(table[state["row"]])}

    run("train-run-1", _params(tmp_path, **calibration), model_dir=tmp_path / "models/laya",
        out=out, report=report, scorer=scorer)  # fmt: skip
    return out, report


def test_run_writes_what_the_server_and_agent_read(tmp_path: Path) -> None:
    out, report = _run(tmp_path, 1200)
    written = json.loads(out.read_text())
    assert written["run_id"].startswith("train-run-1-cal-")
    assert written["temperatures"]["choice:3-5"] == pytest.approx(3.0, rel=0.15)
    assert 0 < written["thresholds"]["call_rejected"] < 1
    assert written["rejection"]["recall"] >= 0.95
    assert written["buckets"]["choice:3-5"]["ece_calibrated_cv"] <= 0.10
    assert written["buckets"]["choice:3-5"]["ece_raw"] > 0.10
    # The contract: the Laya server accepts this file as is.
    served = load_calibration(out)
    assert served.run_id == written["run_id"]
    assert json.loads(report.read_text())["failures"] == []


def test_same_inputs_same_calibration_id(tmp_path: Path) -> None:
    first, _ = _run(tmp_path / "a", 400, min_per_bucket=50)
    again, _ = _run(tmp_path / "b", 400, min_per_bucket=50)
    assert json.loads(first.read_text())["run_id"] == json.loads(again.read_text())["run_id"]


def test_a_failed_gate_writes_the_report_but_no_calibration(tmp_path: Path) -> None:
    with pytest.raises(CalibrationGateError, match="examples < 500"):
        _run(tmp_path, 200, min_per_bucket=500)
    assert not (tmp_path / "models/laya_calibration.json").exists()
    report = json.loads((tmp_path / "reports/calibration.json").read_text())
    assert report["failures"]


def test_held_out_ece_above_the_gate_fails(tmp_path: Path) -> None:
    with pytest.raises(CalibrationGateError, match="held-out ECE"):
        _run(tmp_path, 300, max_ece=0.001)
