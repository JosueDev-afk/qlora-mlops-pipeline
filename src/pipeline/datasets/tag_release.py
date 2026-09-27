"""Version the gold set with DVC; tag it in git on the host.

Two halves, because they run in different places:

- `publish` (the DAG, in the Airflow container): `dvc commit` the build_gold
  stage, which records the gold set's hash in dvc.lock, and `dvc push` it to
  the local remote. No git: the container has neither the maintainer's
  identity nor any business committing to whatever branch is checked out.
- `tag` (the host, `make tag-gold`): commit dvc.lock and create the annotated
  tag `gold.tag`, the name train_qlora and train_laya both read (H1).

A tag is never moved: re-tagging would let two trainings claim the same data.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger

STAGE = "build_gold"
GOLD = "data/gold"
REMOTE = "local"

Runner = Callable[..., subprocess.CompletedProcess[str]]
log = get_logger(__name__)


class ReleaseError(RuntimeError):
    """The gold set cannot be versioned or tagged as it stands."""


def _run(command: Sequence[str], repo: Path, run: Runner) -> str:
    result = run(list(command), cwd=repo, check=True, capture_output=True, text=True)
    return str(result.stdout).strip()


def gold_md5(repo: Path) -> str:
    """The gold set's hash as dvc.lock records it for the build_gold stage."""
    lock = repo / "dvc.lock"
    if not lock.is_file():
        raise ReleaseError("no dvc.lock: run the build_gold_dataset DAG first")
    stage: dict[str, Any] = (yaml.safe_load(lock.read_text()).get("stages") or {}).get(STAGE) or {}
    for out in stage.get("outs") or []:
        if out.get("path") == GOLD and out.get("md5"):
            return str(out["md5"])
    raise ReleaseError(f"dvc.lock has no hash for {GOLD}: the build_gold stage was never committed")


def publish(gold_path: str, *, repo: Path = Path("."), remote: str = REMOTE,
            run: Runner = subprocess.run) -> str:  # fmt: skip
    """Record and push the gold set; return its hash. Git is left to `tag`."""
    if Path(gold_path).resolve() != (repo / GOLD).resolve():
        raise ReleaseError(f"{gold_path} is not {GOLD}, the build_gold stage output")
    _run(["dvc", "commit", "--force", STAGE], repo, run)
    _run(["dvc", "push", "--remote", remote, STAGE], repo, run)
    md5 = gold_md5(repo)
    log.info("gold_versioned", md5=md5, next_step="make tag-gold (on the host)")
    return md5


def tag(params_path: str = "params.yaml", *, repo: Path = Path("."),
        run: Runner = subprocess.run) -> str:  # fmt: skip
    """Commit dvc.lock if it changed and create the annotated tag `gold.tag`."""
    name = str(param(load_params(params_path), "gold.tag"))
    if _run(["git", "tag", "--list", name], repo, run):
        raise ReleaseError(f"tag {name} already exists; bump gold.tag instead of moving it")
    md5 = gold_md5(repo)
    if _run(["git", "status", "--porcelain", "--", "dvc.lock"], repo, run):
        _run(["git", "add", "dvc.lock"], repo, run)
        _run(["git", "commit", "-m", f"data: {name} (gold {md5[:8]})", "--", "dvc.lock"], repo, run)
    _run(["git", "tag", "-a", name, "-m", f"gold set {md5}"], repo, run)
    log.info("gold_tagged", tag=name, md5=md5)
    return name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Version (DVC) or tag (git) the gold set.")
    parser.add_argument("action", choices=["publish", "tag"])
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    if args.action == "publish":
        publish(GOLD)
    else:
        print(tag(args.params))


if __name__ == "__main__":
    main()
