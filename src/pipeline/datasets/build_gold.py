"""Filter, balance and split data/augmented into the gold set.

    python -m src.pipeline.datasets.build_gold --out data/gold

Quotas are exact, never approximate: H4 sweeps `gold.template_ratio` and H5b
depends on `gold.replay_ratio`, so a mix that quietly shrinks when a pool runs
short would change the experiment without failing. A shortfall is an error
that names every pool it hit.

- `replay_ratio` of `target_size` is general instruction data (H5b).
- The rest is split evenly across the tasks present, and within each task
  `template_ratio` comes from code templates, the remainder from the
  human-reviewed carrier-phrase bank (H4).
- Splits are per dialogue group (see examples.py), seeded, and stable.

The output is deterministic: same inputs and params, same bytes. The manifest
has no timestamp for that reason; DVC records when.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger
from src.pipeline.datasets.examples import (
    REPLAY,
    SPLITS,
    invalid_reason,
    read_jsonl,
    split_of,
    write_jsonl,
)

INPUTS = Path("data/augmented")
OUT = Path("data/gold")
MANIFEST = "manifest.json"
MAX_REPORTED_ERRORS = 20

log = get_logger(__name__)


class GoldBuildError(RuntimeError):
    """The inputs cannot fill the requested mix."""


def load(inputs: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Valid, de-duplicated rows and what was dropped on the way."""
    files = sorted(inputs.rglob("*.jsonl"))
    if not files:
        raise GoldBuildError(f"{inputs} has no *.jsonl files")
    rows: dict[str, dict[str, Any]] = {}
    contents: set[str] = set()
    errors: list[str] = []
    stats = Counter[str]()
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    for where, row in read_jsonl(files):
        stats["read"] += 1
        reason = invalid_reason(row)
        if reason:
            stats["invalid"] += 1
            if len(errors) < MAX_REPORTED_ERRORS:
                errors.append(f"{where}: {reason}")
            continue
        content = json.dumps([row["task"], row["input"], row["output"]], sort_keys=True)
        if row["id"] in rows or content in contents:
            stats["duplicates"] += 1
            continue
        rows[row["id"]] = row
        contents.add(content)
    if stats["invalid"]:
        log.warning("gold_invalid_rows", invalid=stats["invalid"], first=errors[:3])
    return list(rows.values()), {
        "files": [str(f.relative_to(inputs)) for f in files],
        "sha256": digest.hexdigest(),
        **{k: stats[k] for k in ("read", "invalid", "duplicates")},
        "errors": errors,
    }


def quotas(
    tasks: list[str], target_size: int, template_ratio: float, replay_ratio: float
) -> dict[tuple[str, str], int]:
    """(task, source) -> rows. Remainders go to the first tasks in name order."""
    replay = round(target_size * replay_ratio)
    per_task, extra = divmod(target_size - replay, len(tasks))
    out: dict[tuple[str, str], int] = {(REPLAY, REPLAY): replay} if replay else {}
    for i, task in enumerate(sorted(tasks)):
        n = per_task + (1 if i < extra else 0)
        template = round(n * template_ratio)
        out[(task, "template")] = template
        out[(task, "carrier")] = n - template
    return out


def build(
    rows: list[dict[str, Any]],
    *,
    target_size: int,
    template_ratio: float,
    replay_ratio: float,
    splits: dict[str, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Sample every quota exactly, then assign splits by group."""
    if abs(sum(splits.values()) - 1.0) > 1e-9 or set(splits) != set(SPLITS):
        raise GoldBuildError(f"gold.splits must be {SPLITS} summing to 1, got {splits}")
    pools: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        pools.setdefault((row["task"], row["source"]), []).append(row)
    tasks = sorted({task for task, _ in pools if task != REPLAY})
    if not tasks:
        raise GoldBuildError("no task examples in the inputs, only replay")
    wanted = quotas(tasks, target_size, template_ratio, replay_ratio)
    short = [
        f"{task}/{source}: need {n}, have {len(pools.get((task, source), []))}"
        for (task, source), n in sorted(wanted.items())
        if len(pools.get((task, source), [])) < n
    ]
    if short:
        raise GoldBuildError("not enough examples:\n  " + "\n  ".join(short))
    chosen: list[dict[str, Any]] = []
    for (task, source), n in sorted(wanted.items()):
        # Seeded per pool: growing one pool never reshuffles another.
        pool = pools.get((task, source), [])  # absent is fine when its quota is 0
        chosen += random.Random(f"{seed}:{task}:{source}").sample(pool, n)
    out: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in chosen:
        out[split_of(row["group"], seed, splits)].append(row)
    return {split: sorted(part, key=lambda r: r["id"]) for split, part in out.items()}


def counts(parts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            "rows": len(part),
            "tasks": dict(sorted(Counter(r["task"] for r in part).items())),
            "sources": dict(sorted(Counter(r["source"] for r in part).items())),
        }
        for split, part in parts.items()
    }


def run(params_path: str = "params.yaml", *, inputs: Path = INPUTS, out: Path = OUT) -> str:
    """Build data/gold (replacing it whole) and return its path."""
    params = load_params(params_path)
    target_size = int(param(params, "gold.target_size"))
    template_ratio = float(param(params, "gold.template_ratio"))
    replay_ratio = float(param(params, "gold.replay_ratio"))
    splits = {str(k): float(v) for k, v in param(params, "gold.splits").items()}
    seed = int(param(params, "gold.seed"))
    rows, input_stats = load(inputs)
    parts = build(rows, target_size=target_size, template_ratio=template_ratio,
                  replay_ratio=replay_ratio, splits=splits, seed=seed)  # fmt: skip
    settings = {"tag": str(param(params, "gold.tag")), "target_size": target_size,
                "template_ratio": template_ratio, "replay_ratio": replay_ratio,
                "splits": splits, "seed": seed}  # fmt: skip
    staging = out.with_name(f".{out.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for split, part in parts.items():
        write_jsonl(staging / f"{split}.jsonl", part)
    manifest = {
        "schema_version": 1,
        "params": settings,
        "inputs": input_stats,
        "counts": counts(parts),
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    log.info("gold_built", out=str(out), **{s: len(p) for s, p in parts.items()})
    return str(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the gold set from data/augmented.")
    parser.add_argument("--in", dest="inputs", type=Path, default=INPUTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    run(args.params, inputs=args.inputs, out=args.out)


if __name__ == "__main__":
    main()
