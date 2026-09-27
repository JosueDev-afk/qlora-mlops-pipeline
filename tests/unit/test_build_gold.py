"""Gold set: exact mixes, group-level splits, a gate that catches leakage and
contamination, and a release that never lets the container touch git.

Inputs are synthetic rows written the way generate/augment will write them:
several noise variants per dialogue group, from templates and carrier phrases.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.pipeline.datasets import build_gold, quality_gate, tag_release
from src.pipeline.datasets.build_gold import GoldBuildError, build, quotas
from src.pipeline.datasets.examples import invalid_reason, normalize_text, split_of
from src.pipeline.datasets.quality_gate import QualityGateError, structural_checks
from src.pipeline.datasets.tag_release import ReleaseError

OUTPUTS: dict[str, dict[str, Any]] = {
    "extract_entity": {"field": "phone", "normalized_value": "+528182345678",
                       "confidence": 0.9, "needs_reprompt": False},
    "parse_datetime": {"datetime_iso": "2026-10-01T16:00:00-06:00", "precision": "exact",
                       "confidence": 0.8},
    "classify_intent": {"intent": "call_rejected", "confidence": 0.9},
    "is_real_interruption": {"interruption": True, "confidence": 0.8, "kind": "correction"},
}  # fmt: skip


def row(task: str, source: str, group: str, variant: int) -> dict[str, Any]:
    text = f"{task} {source} dialogo {group} variante {variant}"
    return {
        "schema_version": 1,
        "id": f"{group}-{variant}",
        "group": group,
        "task": task,
        "source": source,
        "input": {"field": "phone", "transcript": text}
        if task != "replay"
        else {"instruction": text},
        "output": OUTPUTS[task] if task != "replay" else {"response": "Claro, aquí tiene."},
        "meta": {"wer": 0.1},
    }


def pool(groups_per_pool: int = 60, variants: int = 4) -> list[dict[str, Any]]:
    rows = []
    for task in OUTPUTS:
        for source in ("template", "carrier"):
            for g in range(groups_per_pool):
                rows += [
                    row(task, source, f"{task[:4]}-{source[:3]}-{g}", v) for v in range(variants)
                ]
    rows += [
        row("replay", "replay", f"rep-{g}", v)
        for g in range(groups_per_pool)
        for v in range(variants)
    ]
    return rows


SETTINGS = {"target_size": 400, "template_ratio": 0.75, "replay_ratio": 0.15,
            "splits": {"train": 0.8, "val": 0.1, "test": 0.1}, "seed": 42}  # fmt: skip


# ── contract ───────────────────────────────────────────────────────────────


def test_a_well_formed_row_is_valid() -> None:
    assert invalid_reason(row("classify_intent", "template", "g", 0)) is None
    assert invalid_reason(row("replay", "replay", "g", 0)) is None


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"output": {"intent": "maybe", "confidence": 0.5}}, "classify_intent output"),
        ({"source": "replay"}, "example"),  # a task row claiming to be replay
        ({"group": ""}, "example"),
        ({"output": {**OUTPUTS["classify_intent"], "extra": 1}}, "classify_intent output"),
    ],
)
def test_rows_the_agent_would_reject_are_invalid(change: dict, reason: str) -> None:
    assert reason in (
        invalid_reason({**row("classify_intent", "template", "g", 0), **change}) or ""
    )


def test_extraction_labels_must_answer_the_field_asked() -> None:
    bad = row("extract_entity", "template", "g", 0)
    bad["input"]["field"] = "email"
    assert "field" in (invalid_reason(bad) or "")


def test_split_is_stable_and_follows_the_ratios() -> None:
    ratios = SETTINGS["splits"]
    assert split_of("g-1", 42, ratios) == split_of("g-1", 42, ratios)
    counts = {s: 0 for s in ratios}
    for g in range(5000):
        counts[split_of(f"g-{g}", 42, ratios)] += 1
    assert counts["train"] / 5000 == pytest.approx(0.8, abs=0.02)


def test_normalize_text_folds_accents_but_keeps_enie() -> None:
    assert normalize_text("¡Sí, AÑO pasado!  ") == "si año pasado"


# ── build ──────────────────────────────────────────────────────────────────


def test_quotas_are_exact() -> None:
    q = quotas(["classify_intent", "extract_entity", "is_real_interruption", "parse_datetime"],
               400, 0.75, 0.15)  # fmt: skip
    assert q[("replay", "replay")] == 60
    assert sum(q.values()) == 400
    assert q[("classify_intent", "template")] == 64 and q[("classify_intent", "carrier")] == 21


def test_build_hits_every_quota_and_splits_by_group() -> None:
    parts = build(pool(), **SETTINGS)
    rows = [r for p in parts.values() for r in p]
    assert len(rows) == 400
    assert sum(r["task"] == "replay" for r in rows) == 60
    splits_per_group: dict[str, set[str]] = {}
    for split, part in parts.items():
        for r in part:
            splits_per_group.setdefault(r["group"], set()).add(split)
    assert all(len(s) == 1 for s in splits_per_group.values())


def test_same_inputs_same_gold() -> None:
    assert build(pool(), **SETTINGS) == build(list(reversed(pool())), **SETTINGS)


def test_a_short_pool_is_an_error_naming_it() -> None:
    thin = [r for r in pool() if not (r["task"] == "parse_datetime" and r["source"] == "carrier")]
    with pytest.raises(GoldBuildError, match="parse_datetime/carrier: need 21, have 0"):
        build(thin, **SETTINGS)


def test_template_only_is_allowed_for_the_h4_sweep() -> None:
    thin = [r for r in pool() if r["source"] != "carrier"]
    parts = build(thin, **{**SETTINGS, "template_ratio": 1.0})
    assert {r["source"] for p in parts.values() for r in p} == {"template", "replay"}


# ── run, gate and release on disk ──────────────────────────────────────────


def write_inputs(root: Path, rows: list[dict[str, Any]], extra_lines: list[str] = ()) -> Path:
    inputs = root / "augmented"
    inputs.mkdir(parents=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows] + list(extra_lines)
    (inputs / "part-0.jsonl").write_text("\n".join(lines) + "\n")
    return inputs


def write_params(root: Path, **gold: Any) -> str:
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["gold"].update({"target_size": 400, **gold})
    params["calibration"]["split"] = str(root / "calibration")
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(params, allow_unicode=True))
    return str(path)


def test_run_writes_deterministic_splits_and_manifest(tmp_path: Path) -> None:
    bad = json.dumps({"id": "broken"})
    inputs = write_inputs(tmp_path, pool() + pool()[:3], extra_lines=[bad, "not json"])
    params = write_params(tmp_path)
    out = Path(build_gold.run(params, inputs=inputs, out=tmp_path / "gold"))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["inputs"]["duplicates"] == 3 and manifest["inputs"]["invalid"] == 2
    assert sum(c["rows"] for c in manifest["counts"].values()) == 400
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    build_gold.run(params, inputs=inputs, out=tmp_path / "gold")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first  # same bytes, same DVC hash


def test_gate_passes_a_clean_gold_set(tmp_path: Path) -> None:
    params = write_params(tmp_path)
    gold = build_gold.run(params, inputs=write_inputs(tmp_path, pool()), out=tmp_path / "gold")
    report = tmp_path / "report.json"
    quality_gate.assert_expectations(gold, params, eval_set=tmp_path / "eval", report=report)
    assert json.loads(report.read_text())["passed"] is True


def test_gate_catches_eval_set_contamination(tmp_path: Path) -> None:
    params = write_params(tmp_path)
    rows = pool()
    gold = build_gold.run(params, inputs=write_inputs(tmp_path, rows), out=tmp_path / "gold")
    leaked = json.loads((Path(gold) / "test.jsonl").read_text().splitlines()[0])
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "dialogues.jsonl").write_text(
        json.dumps({"input": leaked["input"]}) + "\n"
    )
    report = tmp_path / "report.json"
    with pytest.raises(QualityGateError, match="no_eval_or_calibration_overlap"):
        quality_gate.assert_expectations(gold, params, eval_set=tmp_path / "eval", report=report)
    assert json.loads(report.read_text())["passed"] is False  # the report is written anyway


def _parts() -> dict[str, list[dict[str, Any]]]:
    return build(pool(), **SETTINGS)


def _checks(parts: dict[str, list[dict[str, Any]]], **overrides: Any) -> dict[str, bool]:
    kwargs = {"target_size": 400, "replay_ratio": 0.15, "template_ratio": 0.75, "held_out": set()}
    return {c.name: c.passed for c in structural_checks(parts, **{**kwargs, **overrides})}


def test_structural_checks_catch_leakage_between_splits() -> None:
    parts = _parts()
    parts["test"].append({**parts["train"][0], "id": "leaked-copy"})
    assert _checks(parts)["no_group_leakage"] is False


def test_structural_checks_catch_a_wrong_mix() -> None:
    assert _checks(_parts(), replay_ratio=0.10)["replay_share"] is False
    assert _checks(_parts(), template_ratio=0.5)["template_share"] is False


def test_structural_checks_catch_a_task_missing_from_a_split() -> None:
    parts = _parts()
    parts["val"] = [r for r in parts["val"] if r["task"] != "is_real_interruption"]
    assert _checks(parts)["tasks_in_every_split"] is False


# ── release ────────────────────────────────────────────────────────────────


class FakeShell:
    def __init__(self, outputs: dict[str, str] | None = None) -> None:
        self.outputs = outputs or {}
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        out = next((v for k, v in self.outputs.items() if " ".join(command).startswith(k)), "")
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")


def write_lock(repo: Path, md5: str = "abcdef0123456789.dir") -> None:
    lock = {
        "schema": "2.0",
        "stages": {"build_gold": {"outs": [{"path": "data/gold", "md5": md5}]}},
    }
    (repo / "dvc.lock").write_text(yaml.safe_dump(lock))


def test_publish_commits_and_pushes_with_dvc_only(tmp_path: Path) -> None:
    write_lock(tmp_path)
    shell = FakeShell()
    md5 = tag_release.publish(str(tmp_path / "data/gold"), repo=tmp_path, run=shell)
    assert md5 == "abcdef0123456789.dir"
    assert shell.commands == [["dvc", "commit", "--force", "build_gold"],
                              ["dvc", "push", "--remote", "local", "build_gold"]]  # fmt: skip
    assert not any(c[0] == "git" for c in shell.commands)


def test_publish_refuses_anything_but_the_stage_output(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="not data/gold"):
        tag_release.publish(str(tmp_path / "elsewhere"), repo=tmp_path, run=FakeShell())


def test_tag_commits_the_lock_and_tags_once(tmp_path: Path) -> None:
    write_lock(tmp_path)
    params = write_params(tmp_path, tag="gold-v7")
    shell = FakeShell({"git status": " M dvc.lock"})
    assert tag_release.tag(params, repo=tmp_path, run=shell) == "gold-v7"
    assert [
        "git",
        "commit",
        "-m",
        "data: gold-v7 (gold abcdef01)",
        "--",
        "dvc.lock",
    ] in shell.commands
    assert shell.commands[-1] == [
        "git",
        "tag",
        "-a",
        "gold-v7",
        "-m",
        "gold set abcdef0123456789.dir",
    ]


def test_an_existing_tag_is_never_moved(tmp_path: Path) -> None:
    write_lock(tmp_path)
    params = write_params(tmp_path, tag="gold-v1")
    shell = FakeShell({"git tag --list": "gold-v1"})
    with pytest.raises(ReleaseError, match="already exists"):
        tag_release.tag(params, repo=tmp_path, run=shell)
    assert not any(c[:2] == ["git", "commit"] for c in shell.commands)


def test_no_lock_means_nothing_to_tag(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="dvc.lock"):
        tag_release.gold_md5(tmp_path)
