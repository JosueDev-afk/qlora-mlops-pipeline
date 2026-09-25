"""Fit Laya's temperatures on the human calibration split, gate on ECE, derive thresholds.

Laya ships over-confident, so its probabilities are rescaled with one
temperature per bucket (question type x option count, Laya's own buckets).
Three choices make the numbers defensible in the paper (H6):

- **Held-out ECE.** The gate is the ECE of k-fold cross-validation: each
  fold is scored with a temperature fitted on the others. ECE measured on
  the data the temperature was fitted to would flatter it.
- **Per bucket, with a floor.** Task B (choice) and Task D (noul) are gated
  separately, so a good bucket cannot hide a bad one, and a bucket with too
  few examples is an error rather than a temperature fitted on noise.
- **The rejection threshold is derived.** `call_rejected` fires at the
  highest calibrated probability that still reaches `target_recall` on the
  split: the most precise threshold that lets go of enough people.

Pure Python on purpose: a few thousand small softmaxes need no numpy, and
the fit then runs in CI, Airflow and a notebook alike. Scoring is injected;
the default loads the fine-tuned checkpoint in process with neutral
temperatures.

Laya returns probabilities rounded to 4 decimals, so they are floored at
1e-4 before tempering. That only moves answers that were already certain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger
from src.common.prompts import load_prompt

T_RANGE = (0.5, 5.0)  # the range Laya clamps temperatures to (laya.common.clamp_temperature)
FLOOR = 1e-4  # Laya's 4-decimal rounding
ECE_BINS = 15
# Task B's question on Laya and the label that ends a call.
REJECTION = ("classify_intent_laya", "intent", "call_rejected")
MODEL_DIR = Path("models/laya")
OUT = Path("models/laya_calibration.json")
REPORT = Path("evaluation/reports/calibration.json")

Scorer = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
log = get_logger(__name__)


class CalibrationSplitError(ValueError):
    """The human calibration split has rows that cannot be scored as annotated."""


class CalibrationGateError(RuntimeError):
    """Calibration failed its gate; the model must not gate decisions."""


@dataclass(frozen=True)
class Example:
    """One annotated answer: Laya's raw probabilities and the human label's index."""

    row_id: str
    question_set: str
    qid: str
    bucket: str
    options: tuple[str, ...]
    probs: tuple[float, ...]
    gold: int


# ── math ───────────────────────────────────────────────────────────────────


def bucket(qtype: str, n_options: int) -> str:
    """Mirrors laya.common.temp_bucket."""
    size = (
        "2" if n_options <= 2 else "3-5" if n_options <= 5 else "6-10" if n_options <= 10 else "11+"
    )
    return f"{qtype}:{size}"


def temper(probs: Iterable[float], t: float) -> list[float]:
    """softmax(log p / T): what Laya computes from logits, recovered from probabilities."""
    logs = [math.log(max(p, FLOOR)) / t for p in probs]
    top = max(logs)
    exps = [math.exp(v - top) for v in logs]
    total = sum(exps)
    return [e / total for e in exps]


def nll(examples: list[Example], t: float) -> float:
    return -sum(math.log(max(temper(e.probs, t)[e.gold], 1e-12)) for e in examples) / len(examples)


def fit_t(examples: list[Example], lo: float = T_RANGE[0], hi: float = T_RANGE[1]) -> float:
    """The NLL-minimizing temperature. Golden-section on log T: NLL is convex in 1/T."""
    a, b = math.log(lo), math.log(hi)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(examples, math.exp(c)), nll(examples, math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(examples, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(examples, math.exp(d))
    return round(math.exp((a + b) / 2), 4)


def ece(scored: list[tuple[float, bool]], bins: int = ECE_BINS) -> float:
    """Top-label expected calibration error over (confidence, correct) pairs."""
    totals = [[0, 0.0, 0.0] for _ in range(bins)]  # n, sum(conf), sum(correct)
    for confidence, correct in scored:
        slot = totals[min(int(confidence * bins), bins - 1)]
        slot[0] += 1
        slot[1] += confidence
        slot[2] += correct
    n = len(scored)
    return sum(abs(s[2] - s[1]) / n for s in totals if s[0])


def _scored(examples: list[Example], t: float) -> list[tuple[float, bool]]:
    out = []
    for e in examples:
        p = temper(e.probs, t)
        top = max(range(len(p)), key=p.__getitem__)
        out.append((p[top], top == e.gold))
    return out


def cross_validated_ece(examples: list[Example], folds: int, seed: int) -> float:
    """ECE with every example scored by a temperature fitted without it."""
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    held_out: list[tuple[float, bool]] = []
    for k in range(folds):
        test = [examples[i] for i in order[k::folds]]
        train = [examples[i] for j, i in enumerate(order) if j % folds != k]
        held_out += _scored(test, fit_t(train))
    return ece(held_out)


def rejection_threshold(
    examples: list[Example], t: float, label_index: int, target_recall: float
) -> dict[str, float | int]:
    """Highest threshold on calibrated P(label) whose recall reaches the target."""
    scores = [(temper(e.probs, t)[label_index], e.gold == label_index) for e in examples]
    positives = sorted((s for s, pos in scores if pos), reverse=True)
    if not positives:
        raise CalibrationGateError("no call_rejected examples: the threshold cannot be derived")
    threshold = positives[math.ceil(target_recall * len(positives)) - 1]
    fired = [pos for s, pos in scores if s >= threshold]
    return {
        "threshold": round(threshold, 6),
        "recall": round(sum(fired) / len(positives), 4),
        "precision": round(sum(fired) / len(fired), 4),
        "positives": len(positives),
    }


# ── data ───────────────────────────────────────────────────────────────────


def _options(question: dict[str, Any]) -> list[Any]:
    if question["type"] == "noul":
        return [False, True]
    criteria = question["criteria"]
    return list(criteria) if isinstance(criteria, dict | list) else []


def _answer_probs(question: dict[str, Any], answer: dict[str, Any]) -> tuple[float, ...]:
    if question["type"] == "noul":
        p = float(answer["noul"])
        return (1 - p, p)
    probs = answer["probabilities"]
    if question["type"] == "score":
        return tuple(float(probs[str(i)]) for i in range(len(question["criteria"])))
    return tuple(float(probs[label]) for label in question["criteria"])


def load_rows(split: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(split.glob("*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                rows.append({**json.loads(line), "_where": f"{path.name}:{n}"})
    if not rows:
        raise CalibrationSplitError(f"{split} has no *.jsonl rows")
    return rows


def split_digest(split: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(split.glob("*.jsonl")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def examples_from(rows: list[dict[str, Any]], scorer: Scorer) -> list[Example]:
    """Score each row once and pair every labelled question with its human label.

    Rows name their question file (`classify_intent_laya/v1`), so a split
    annotated against one version is never scored against another. Every
    labelling problem is reported at once, since the split is typed by hand.
    """
    problems: list[str] = []
    examples: list[Example] = []
    for row in rows:
        where = row.get("_where", row.get("id"))
        try:
            prompt_id, version = str(row["questions"]).split("/v")
            questions = load_prompt(prompt_id, int(version)).body["questions"]
            labels = row["labels"]
            state = row["state"]
        except (KeyError, ValueError, FileNotFoundError) as exc:
            problems.append(f"{where}: {exc!r}")
            continue
        for qid, label in labels.items():
            if qid not in questions:
                problems.append(f"{where}: no question {qid!r} in {row['questions']}")
            elif label not in _options(questions[qid]) and not (
                questions[qid]["type"] == "score" and label in range(len(_options(questions[qid])))
            ):
                problems.append(f"{where}: {label!r} is not an option of {qid!r}")
        if problems:
            continue
        answers = scorer(state, {qid: questions[qid] for qid in labels})
        for qid, label in labels.items():
            question = questions[qid]
            options = _options(question)
            gold = label if question["type"] == "score" else options.index(label)
            examples.append(Example(
                row_id=str(row.get("id", where)),
                question_set=prompt_id,
                qid=qid,
                bucket=bucket(question["type"], len(options)),
                options=tuple(str(o) for o in options),
                probs=_answer_probs(question, answers[qid]),
                gold=gold,
            ))  # fmt: skip
    if problems:
        raise CalibrationSplitError("calibration split:\n  " + "\n  ".join(problems))
    return examples


# ── fit ────────────────────────────────────────────────────────────────────


def fit(
    examples: list[Example],
    *,
    max_ece: float,
    target_recall: float,
    folds: int,
    min_per_bucket: int,
    seed: int,
) -> dict[str, Any]:
    """Temperatures, held-out ECE and the rejection threshold, plus `passed`."""
    by_bucket: dict[str, list[Example]] = {}
    for e in examples:
        by_bucket.setdefault(e.bucket, []).append(e)
    temperatures: dict[str, float] = {}
    report: dict[str, Any] = {}
    failures: list[str] = []
    for name, group in sorted(by_bucket.items()):
        if len(group) < min_per_bucket:
            failures.append(f"{name}: {len(group)} examples < {min_per_bucket}")
            report[name] = {"n": len(group)}
            continue
        temperatures[name] = fit_t(group)
        calibrated = cross_validated_ece(group, folds, seed)
        report[name] = {
            "n": len(group),
            "temperature": temperatures[name],
            "ece_raw": round(ece(_scored(group, 1.0)), 4),
            "ece_calibrated_cv": round(calibrated, 4),
        }
        if calibrated > max_ece:
            failures.append(f"{name}: held-out ECE {calibrated:.3f} > {max_ece}")

    set_id, qid, label = REJECTION
    intent = [e for e in examples if e.question_set == set_id and e.qid == qid]
    rejection: dict[str, Any] = {}
    if intent and intent[0].bucket in temperatures:
        index = intent[0].options.index(label)
        rejection = rejection_threshold(
            intent, temperatures[intent[0].bucket], index, target_recall
        )
        if rejection["positives"] < 100:
            log.warning("few_rejections", positives=rejection["positives"], wanted=100)
    else:
        failures.append(f"no calibrated {set_id}/{qid} examples: call_rejected has no threshold")
    return {
        "passed": not failures,
        "failures": failures,
        "temperatures": temperatures,
        "thresholds": {label: rejection["threshold"]} if rejection else {},
        "rejection": {**rejection, "target_recall": target_recall} if rejection else {},
        "buckets": report,
    }


def laya_scorer(checkpoint: str, device: str | None = None) -> Scorer:
    """Score with the checkpoint in process, multilingual only, temperatures neutral."""
    from laya import Router

    router = Router(
        models={"multilingual": checkpoint}, default="multilingual", max_loaded=1, device=device
    )
    router.preload(["multilingual"])
    agent = router.load("multilingual")
    agent.temperature = [1.0, 1.0, 1.0]  # score raw: the checkpoint's own must not stack
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}

    def score(state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        answers: dict[str, Any] = router.predict(state, questions, model="multilingual")["answers"]
        return answers

    return score


def _train_run_id(model_dir: Path, run_id: str | None) -> str:
    if run_id:
        return run_id
    marker = model_dir / "train_run.json"
    if marker.is_file():
        return str(json.loads(marker.read_text())["run_id"])
    raise ValueError(f"pass --run-id, or have train_laya write {marker}")


def run(
    run_id: str | None,
    params_path: str = "params.yaml",
    *,
    model_dir: Path = MODEL_DIR,
    out: Path = OUT,
    report: Path = REPORT,
    scorer: Scorer | None = None,
) -> str:
    """Fit, write the report always and the calibration only if the gate passes."""
    params = load_params(params_path)
    train_run_id = _train_run_id(model_dir, run_id)
    split = Path(param(params, "calibration.split"))
    rows = load_rows(split)
    examples = examples_from(rows, scorer or laya_scorer(str(model_dir)))
    result = fit(
        examples,
        max_ece=float(param(params, "calibration.max_ece")),
        target_recall=float(param(params, "calibration.target_recall")),
        folds=int(param(params, "calibration.folds")),
        min_per_bucket=int(param(params, "calibration.min_per_bucket")),
        seed=int(param(params, "calibration.seed")),
    )
    digest = hashlib.sha256(
        json.dumps(
            [train_run_id, split_digest(split), result["temperatures"], result["thresholds"]],
            sort_keys=True,
        ).encode()  # fmt: skip
    ).hexdigest()
    calibration = {
        # Stamped by the Laya server on every answer and checked by the agent's
        # client; it moves whenever the checkpoint, the split or the fit does.
        "run_id": f"{train_run_id}-cal-{digest[:8]}",
        "train_run_id": train_run_id,
        "checkpoint": str(model_dir),
        "split": {"path": str(split), "sha256": split_digest(split), "rows": len(rows)},
        "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "max_ece": float(param(params, "calibration.max_ece")),
        **{k: v for k, v in result.items() if k not in ("passed", "failures")},
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({**calibration, "failures": result["failures"]}, indent=2) + "\n")
    if not result["passed"]:
        raise CalibrationGateError("calibration gate failed:\n  " + "\n  ".join(result["failures"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibration, indent=2) + "\n")
    log.info("calibrated", run_id=calibration["run_id"], temperatures=result["temperatures"])
    return str(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit Laya temperatures on the calibration split.")
    parser.add_argument("--model", type=Path, default=MODEL_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--run-id", default=None, help="train_laya's MLflow run id")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    run(args.run_id, args.params, model_dir=args.model, out=args.out, report=args.report)


if __name__ == "__main__":
    main()
