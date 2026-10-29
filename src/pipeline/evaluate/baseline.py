"""Baselines on the gold test split: the numbers the fine-tuned models must beat.

- Task A, the traditional method: the rule-based normalizers
  (src/common/normalizers.py) reading phones and emails straight from the
  noisy transcript. Two numbers matter: exact match, and wrong values
  accepted (a well-formed value that is not the label), because a wrong value
  written to the contact record is worse than asking again. Names have no
  rule reader, so they are left to the model comparison.
- Tasks B and D: the majority class, the floor any classifier must clear.

These run on the synthetic test split, not the frozen eval set (which does
not exist yet): they size the gap, they do not test a hypothesis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger
from src.common.normalizers import normalize_email, spoken_to_e164
from src.pipeline.common.tracking import stage_run
from src.pipeline.datasets.examples import read_jsonl

GOLD = Path("data/gold")
READERS = {"phone": spoken_to_e164, "email": normalize_email}

log = get_logger(__name__)


def rules_on_task_a(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact match and wrong-value rate of the normalizers, by field and noise level."""
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["task"] != "extract_entity" or row["input"]["field"] not in READERS:
            continue
        field = row["input"]["field"]
        predicted = READERS[field](row["input"]["transcript"])
        label = row["output"]["normalized_value"]
        level = f"wer_{row.get('meta', {}).get('wer_target', 0.0)}"
        for bucket in ("all", field, level):
            totals[bucket]["n"] += 1
            totals[bucket]["exact"] += predicted == label
            totals[bucket]["answered"] += predicted is not None
            totals[bucket]["wrong_value"] += predicted is not None and predicted != label
    return {
        bucket: {
            "n": c["n"],
            "exact_match": round(c["exact"] / c["n"], 4),
            "coverage": round(c["answered"] / c["n"], 4),
            "wrong_value_rate": round(c["wrong_value"] / c["n"], 4),
        }
        for bucket, c in sorted(totals.items())
    }


def majority(rows: list[dict[str, Any]], task: str, key: str) -> dict[str, Any]:
    labels = [str(r["output"][key]) for r in rows if r["task"] == task]
    if not labels:
        return {"n": 0}
    label, count = Counter(labels).most_common(1)[0]
    return {"n": len(labels), "majority_label": label, "accuracy": round(count / len(labels), 4)}


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_a_rules": rules_on_task_a(rows),
        "task_b_majority": majority(rows, "classify_intent", "intent"),
        "task_d_majority": majority(rows, "is_real_interruption", "interruption"),
    }


def run(
    params_path: str = "params.yaml", *, gold: Path = GOLD, split: str = "test"
) -> dict[str, Any]:
    params = load_params(params_path)
    path = gold / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: build the gold set first (make gold)")
    rows = [row for _, row in read_jsonl([path])]
    results = evaluate(rows)
    with stage_run("baselines", experiment="evaluation") as tracked:
        tracked.params({"gold_tag": param(params, "gold.tag"), "split": split, "rows": len(rows)})
        tracked.metrics(results)
        tracked.artifact(results, "baselines.json")
    log.info("baselines", split=split, task_a=results["task_a_rules"].get("all"))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baselines on the gold test split.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    configure_logging()
    run(args.params, split=args.split)


if __name__ == "__main__":
    main()
