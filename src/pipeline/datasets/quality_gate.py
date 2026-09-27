"""Gate the gold set before it is versioned: fail the DAG, never ship bad data.

Two layers, one report (`evaluation/reports/gold_quality.json`):

- **Structural checks, in Python.** The ones that decide whether an
  experiment means anything: no dialogue group in two splits (leakage), the
  exact replay and template shares H5b and H4 depend on, every task in every
  split, and no example that also appears in the frozen eval set or the human
  calibration split (contamination). Unit-tested in CI.
- **Column expectations, in Great Expectations**, run where it is installed
  (the Airflow image): unique ids, no null groups, closed vocabularies for
  task, source and split, the exact row count.

Either layer failing raises QualityGateError after the report is written.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import get_logger
from src.pipeline.datasets.build_gold import MANIFEST
from src.pipeline.datasets.examples import (
    REPLAY,
    SPLITS,
    TASKS,
    invalid_reason,
    normalize_text,
    read_jsonl,
)

EVAL_SET = Path("evaluation/eval_set")
REPORT = Path("evaluation/reports/gold_quality.json")
SHARE_TOLERANCE = 0.005  # rounding of quotas, nothing more

log = get_logger(__name__)


class QualityGateError(RuntimeError):
    """The gold set failed its gate."""


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def load_gold(gold: Path) -> dict[str, list[dict[str, Any]]]:
    parts: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = gold / f"{split}.jsonl"
        parts[split] = [row for _, row in read_jsonl([path])] if path.is_file() else []
    return parts


def _texts(row: dict[str, Any]) -> set[str]:
    """Every string the model reads in a row: what contamination is measured on."""
    values = row.get("input", {})
    return {normalize_text(v) for v in values.values() if isinstance(v, str) and v.strip()}


def _reference_texts(*dirs: Path) -> set[str]:
    texts: set[str] = set()
    for root in dirs:
        if not root.is_dir():
            continue
        for _, row in read_jsonl(root.rglob("*.jsonl")):
            if isinstance(row, dict):
                for key in ("input", "state"):
                    block = row.get(key)
                    if isinstance(block, dict):
                        texts |= {normalize_text(v) for v in block.values() if isinstance(v, str)}
    return {t for t in texts if len(t.split()) >= 3}  # "sí" in both is not contamination


def structural_checks(
    parts: dict[str, list[dict[str, Any]]],
    *,
    target_size: int,
    replay_ratio: float,
    template_ratio: float,
    held_out: set[str],
) -> list[Check]:
    rows = [row for part in parts.values() for row in part]
    checks = [Check("splits_present", all(parts[s] for s in SPLITS),
                    str({s: len(p) for s, p in parts.items()}))]  # fmt: skip
    checks.append(Check("row_count", len(rows) == target_size, f"{len(rows)} of {target_size}"))

    invalid = [(r.get("id"), reason) for r in rows if (reason := invalid_reason(r))]
    checks.append(Check("rows_valid", not invalid, str(invalid[:5])))

    ids = [r.get("id") for r in rows]
    checks.append(
        Check("ids_unique", len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} repeated")
    )

    groups: dict[str, set[str]] = defaultdict(set)
    for split, part in parts.items():
        for row in part:
            groups[row.get("group", "")].add(split)
    leaked = sorted(str(g) for g, s in groups.items() if len(s) > 1)
    checks.append(Check("no_group_leakage", not leaked, f"{len(leaked)} groups: {leaked[:5]}"))

    replay = sum(r.get("task") == REPLAY for r in rows) / max(len(rows), 1)
    replay_ok = abs(replay - replay_ratio) <= SHARE_TOLERANCE and 0.10 <= replay <= 0.20
    checks.append(Check("replay_share", replay_ok, f"{replay:.4f} (want {replay_ratio})"))
    tasks = [r for r in rows if r.get("task") != REPLAY]
    template = sum(r.get("source") == "template" for r in tasks) / max(len(tasks), 1)
    checks.append(Check("template_share", abs(template - template_ratio) <= SHARE_TOLERANCE,
                        f"{template:.4f} (want {template_ratio})"))  # fmt: skip

    present = {str(r.get("task")) for r in tasks}
    missing = [f"{t}/{s}" for t in sorted(present) for s in SPLITS
               if not any(r.get("task") == t for r in parts[s])]  # fmt: skip
    checks.append(
        Check("tasks_in_every_split", not missing and present <= set(TASKS), str(missing))
    )

    overlap = sorted({str(r.get("id")) for r in rows if _texts(r) & held_out})
    checks.append(Check("no_eval_or_calibration_overlap", not overlap,
                        f"{len(overlap)} rows: {overlap[:5]}"))  # fmt: skip
    return checks


def gx_checks(parts: dict[str, list[dict[str, Any]]], *, target_size: int) -> list[Check]:
    """Column expectations in Great Expectations; skipped where it is not installed."""
    try:
        import great_expectations as gx
        import pandas as pd
    except ImportError:
        return [Check("great_expectations", True, "skipped: not installed here")]

    frame = pd.DataFrame(
        [
            {
                "id": r.get("id"),
                "group": r.get("group"),
                "task": r.get("task"),
                "source": r.get("source"),
                "split": split,
            }
            for split, part in parts.items()
            for r in part
        ]  # fmt: skip
    )
    context = gx.get_context(mode="ephemeral")
    asset = context.data_sources.add_pandas("gold").add_dataframe_asset("rows")
    batch = asset.add_batch_definition_whole_dataframe("all").get_batch(
        batch_parameters={"dataframe": frame}
    )
    exp = gx.expectations
    expectations = [
        exp.ExpectTableRowCountToEqual(value=target_size),
        exp.ExpectColumnValuesToBeUnique(column="id"),
        exp.ExpectColumnValuesToNotBeNull(column="group"),
        exp.ExpectColumnValuesToBeInSet(column="task", value_set=[*TASKS, REPLAY]),
        exp.ExpectColumnValuesToBeInSet(column="source", value_set=["template", "carrier", REPLAY]),
        exp.ExpectColumnValuesToBeInSet(column="split", value_set=list(SPLITS)),
    ]
    checks = []
    for expectation in expectations:
        result = batch.validate(expectation)
        column = getattr(expectation, "column", "table")
        checks.append(Check(f"gx:{type(expectation).__name__}:{column}", bool(result.success),
                            json.dumps(result.result, default=str)[:300]))  # fmt: skip
    return checks


def assert_expectations(
    gold_path: str,
    params_path: str = "params.yaml",
    *,
    eval_set: Path = EVAL_SET,
    report: Path = REPORT,
) -> None:
    """Write the report; raise if any check failed."""
    params = load_params(params_path)
    gold = Path(gold_path)
    parts = load_gold(gold)
    target_size = int(param(params, "gold.target_size"))
    held_out = _reference_texts(eval_set, Path(param(params, "calibration.split")))
    checks = structural_checks(
        parts,
        target_size=target_size,
        replay_ratio=float(param(params, "gold.replay_ratio")),
        template_ratio=float(param(params, "gold.template_ratio")),
        held_out=held_out,
    )
    checks.append(Check("manifest_present", (gold / MANIFEST).is_file()))
    checks += gx_checks(parts, target_size=target_size)
    failed = [c for c in checks if not c.passed]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "gold": str(gold),
        "held_out_texts": len(held_out),
        "passed": not failed,
        "checks": [asdict(c) for c in checks],
    }, indent=2, ensure_ascii=False) + "\n")  # fmt: skip
    if failed:
        raise QualityGateError(
            "gold quality gate failed:\n  " + "\n  ".join(f"{c.name}: {c.detail}" for c in failed)
        )
    log.info("gold_gate_passed", checks=len(checks))
