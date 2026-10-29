"""Laya-multilingual before fine-tuning, on the gold test split (Tasks B and D).

The starting point of H1 and H6: how the base checkpoint decides intent and
interruption with no training, and how over-confident it is (raw ECE). It
runs on the Mac's CPU (laya is in the train group, which the Airflow image
does not install), against the same multilingual checkpoint and neutral
temperatures as calibration, with the same state keys the agent sends.

    python -m src.pipeline.evaluate.laya_zero_shot --per-task 300

Latency here is CPU latency on the Mac: the H1 figure is measured on an L4.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.log import configure_logging, get_logger
from src.common.prompts import load_prompt
from src.pipeline.calibrate.fit_temperature import Scorer, ece, laya_scorer
from src.pipeline.common.tracking import stage_run
from src.pipeline.datasets.examples import read_jsonl

GOLD = Path("data/gold")
CHECKPOINT = "convaiinnovations/laya-multilingual"

log = get_logger(__name__)


def laya_state(row: dict[str, Any]) -> dict[str, str]:
    """The keys the runtime sends (src/agent/nodes, src/agent/demo/session)."""
    inputs = row["input"]
    if row["task"] == "classify_intent":
        return {"flujo": inputs["flow"], "contexto": inputs["context"],
                "transcripción": inputs["transcript"]}  # fmt: skip
    return {"agente_dice": inputs["agent_utterance"], "transcripción": inputs["transcript"]}


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({gold for gold, _ in pairs})
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def evaluate(rows: list[dict[str, Any]], scorer: Scorer) -> dict[str, Any]:
    intent_q = load_prompt("classify_intent_laya", 1).body["questions"]
    interruption_q = load_prompt("is_real_interruption", 2).body["questions"]
    results: dict[str, Any] = {}
    for task, questions, gold_key in (
        ("classify_intent", {"intent": intent_q["intent"]}, "intent"),
        ("is_real_interruption", {"interruption": interruption_q["interruption"]}, "interruption"),
    ):
        pairs, confidences, latencies = [], [], []
        for row in (r for r in rows if r["task"] == task):
            started = time.perf_counter()
            answer = scorer(laya_state(row), questions)[gold_key]
            latencies.append((time.perf_counter() - started) * 1000)
            if gold_key == "intent":
                predicted, confidence = answer["choice"], max(answer["probabilities"].values())
            else:
                p_true = float(answer["noul"])
                predicted, confidence = str(p_true >= 0.5), max(p_true, 1 - p_true)
            gold = str(row["output"][gold_key])
            pairs.append((gold, str(predicted)))
            confidences.append((confidence, gold == str(predicted)))
        if not pairs:
            continue
        latencies.sort()
        results[task] = {
            "n": len(pairs),
            "accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 4),
            "macro_f1": macro_f1(pairs),
            "ece_raw": round(ece(confidences), 4),
            "cpu_latency_ms_p50": round(latencies[len(latencies) // 2], 1),
            "predicted": dict(Counter(p for _, p in pairs).most_common()),
        }
    return results


def sample(rows: list[dict[str, Any]], per_task: int, seed: int = 7) -> list[dict[str, Any]]:
    out = []
    for task in ("classify_intent", "is_real_interruption"):
        pool = sorted((r for r in rows if r["task"] == task), key=lambda r: r["id"])
        out += random.Random(f"{seed}:{task}").sample(pool, min(per_task, len(pool)))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Laya zero-shot on the gold test split.")
    parser.add_argument("--per-task", type=int, default=300)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    configure_logging()
    rows = [row for _, row in read_jsonl([GOLD / f"{args.split}.jsonl"])]
    chosen = sample(rows, args.per_task)
    results = evaluate(chosen, laya_scorer(CHECKPOINT, args.device))
    with stage_run("laya_zero_shot", experiment="evaluation") as tracked:
        tracked.params({"checkpoint": CHECKPOINT, "split": args.split, "per_task": args.per_task,
                        "device": args.device, "temperatures": "neutral"})  # fmt: skip
        tracked.metrics({task: {k: v for k, v in r.items() if k != "predicted"}
                         for task, r in results.items()})  # fmt: skip
        tracked.artifact(results, "laya_zero_shot.json")
    for task, r in results.items():
        log.info("laya_zero_shot", task=task, **{k: v for k, v in r.items() if k != "predicted"})


if __name__ == "__main__":
    main()
