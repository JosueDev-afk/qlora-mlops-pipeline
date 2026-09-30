"""Inject rule-based ASR noise: data/synthetic (+ replay) -> data/augmented.

    python -m src.pipeline.augment.run --out data/augmented

Each example yields `augment.variants_per_dialogue` variants sharing its
group: variant 0 is clean, the rest cycle through `augment.wer_levels`.
Replay rows from data/silver/replay pass through untouched: general
instruction data is not speech. Everything build_gold draws from lands here.

The output is deterministic (same inputs, params and confusion tables, same
bytes), and the manifest records what relabelling did per task, so a table
change that silently nulls half the phones shows up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger
from src.pipeline.augment.noise import CONFUSIONS, load_confusions
from src.pipeline.augment.variants import make_variant
from src.pipeline.datasets.examples import REPLAY, invalid_reason, read_jsonl, write_jsonl

SYNTHETIC = Path("data/synthetic")
REPLAY_DIR = Path("data/silver/replay")
OUT = Path("data/augmented")
MANIFEST = "manifest.json"

log = get_logger(__name__)


class AugmentError(RuntimeError):
    """Inputs or settings the stage cannot turn into valid examples."""


def _rows(root: Path) -> list[dict[str, Any]]:
    rows, bad = [], []
    for where, row in read_jsonl(p for p in root.glob("*.jsonl")):
        reason = invalid_reason(row)
        (bad.append(f"{where}: {reason}") if reason else rows.append(row))
    if bad:
        raise AugmentError(f"{root}: {len(bad)} invalid rows, first: {bad[:3]}")
    return rows


def augment(
    examples: list[dict[str, Any]],
    *,
    wer_levels: list[float],
    variants: int,
    seed: int,
    emphasis: bool,
    confusions_path: Path = CONFUSIONS,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Variants per task, and what happened to their labels."""
    confusions = load_confusions(confusions_path)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status: dict[str, Counter[str]] = defaultdict(Counter)
    applied: dict[str, list[float]] = defaultdict(list)
    for example in sorted(examples, key=lambda e: e["id"]):
        for k in range(variants):
            wer = 0.0 if k == 0 else wer_levels[(k - 1) % len(wer_levels)]
            variant, outcome = make_variant(
                example, k, wer, confusions, seed=seed, emphasis=emphasis
            )
            status[example["task"]][outcome] += 1
            if variant is None:
                continue
            reason = invalid_reason(variant)
            if reason:
                raise AugmentError(f"{variant['id']}: {reason}")  # a bug in the noise, not data
            out[example["task"]].append(variant)
            applied[str(wer)].append(variant["meta"]["wer"])
    stats = {
        "labels": {task: dict(sorted(c.items())) for task, c in sorted(status.items())},
        "wer_applied_mean": {
            level: round(sum(v) / len(v), 4) for level, v in sorted(applied.items())
        },
        "confusions_version": confusions.version,
    }
    return dict(out), stats


def run(
    params_path: str = "params.yaml",
    *,
    synthetic: Path = SYNTHETIC,
    replay: Path = REPLAY_DIR,
    out: Path = OUT,
    confusions: Path = CONFUSIONS,
) -> str:
    params = load_params(params_path)
    method = str(param(params, "augment.method"))
    if method != "rules":
        raise AugmentError(f"augment.method {method!r}: only 'rules' is implemented")
    examples = _rows(synthetic)
    if not examples:
        raise AugmentError(f"{synthetic} has no examples: run the generate stage first")
    parts, stats = augment(
        examples,
        wer_levels=[float(w) for w in param(params, "augment.wer_levels")],
        variants=int(param(params, "augment.variants_per_dialogue")),
        seed=int(param(params, "augment.seed")),
        emphasis=str(param(params, "augment.emphasis")) == "alphanumeric",
        confusions_path=confusions,
    )
    replay_rows = _rows(replay) if replay.is_dir() else []
    if [r for r in replay_rows if r["task"] != REPLAY]:
        raise AugmentError(f"{replay} holds rows that are not replay")
    if not replay_rows:
        log.warning("augment_no_replay", path=str(replay), hint="the gold set needs 10-20% replay")

    staging = out.with_name(f".{out.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for task, rows in sorted(parts.items()):
        write_jsonl(staging / f"{task}.jsonl", rows)
    if replay_rows:
        write_jsonl(staging / "replay.jsonl", sorted(replay_rows, key=lambda r: r["id"]))
    manifest = {
        "schema_version": 1,
        "confusions_sha256": hashlib.sha256(confusions.read_bytes()).hexdigest(),
        "rows": {**{t: len(r) for t, r in sorted(parts.items())}, REPLAY: len(replay_rows)},
        **stats,
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    log.info("augmented", out=str(out), **manifest["rows"])
    return str(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inject rule-based ASR noise.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    run(args.params, out=args.out)


if __name__ == "__main__":
    main()
