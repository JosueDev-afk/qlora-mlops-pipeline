"""Curate bronze into silver.

    python -m src.pipeline.curate.run --in data/bronze --out data/silver

Reads only from bronze, at the partition holding the pin in params.yaml, so
silver is rebuilt from exactly the bytes ingest verified. Today it curates
the replay set (Aya, Spanish) into data/silver/replay, which augment passes
through to build_gold. CallCenterEN's ASR profile will join here, in Spark,
once its license is decided.

The output is deterministic and replaced whole; a manifest records the pin
and what every rule dropped.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger
from src.pipeline.curate.replay import curate
from src.pipeline.datasets.examples import invalid_reason, write_jsonl
from src.pipeline.ingest.run import find_existing
from src.pipeline.ingest.sources import load_sources

BRONZE = Path("data/bronze")
SILVER = Path("data/silver")
MANIFEST = "manifest.json"

log = get_logger(__name__)


class CurateError(RuntimeError):
    """Bronze does not hold what params.yaml pins."""


def read_parquet_rows(files: list[Path]) -> list[dict[str, Any]]:
    """pyarrow comes with the pipeline group (through datasets); imported here only."""
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in sorted(files):
        rows += pq.read_table(path).to_pylist()
    return rows


def curate_replay(params: dict[str, Any], bronze: Path, out: Path) -> dict[str, Any]:
    settings = param(params, "curate.replay")
    source = load_sources(params)[settings["source"]]
    partition = find_existing(source, bronze)
    if partition is None:
        raise CurateError(f"{source.name} is not in {bronze} at {source.pin[:12]}: run make ingest")
    files = sorted(partition.rglob("*.parquet"))
    if not files:
        raise CurateError(f"{partition} has no parquet files")
    examples, stats = curate(
        read_parquet_rows(files),
        language=str(settings["language"]),
        max_words=int(settings["max_words"]),
        dataset=str(source.repo_id),
        revision=source.pin,
        license=source.license,
    )
    bad = [(e["id"], reason) for e in examples if (reason := invalid_reason(e))]
    if bad:
        raise CurateError(f"curated rows break the gold contract: {bad[:3]}")
    target = out / "replay"
    staging = target.with_name(".replay.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    write_jsonl(staging / f"{source.name}.jsonl", examples)
    manifest = {"source": source.name, "partition": str(partition.relative_to(bronze)),
                "pin": source.pin, "settings": dict(settings), "rows": stats}  # fmt: skip
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    log.info("replay_curated", out=str(target), **stats)
    return manifest


def run(params_path: str = "params.yaml", *, bronze: Path = BRONZE, out: Path = SILVER) -> str:
    curate_replay(load_params(params_path), bronze, out)
    return str(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Curate bronze into silver.")
    parser.add_argument("--in", dest="bronze", type=Path, default=BRONZE)
    parser.add_argument("--out", type=Path, default=SILVER)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    run(args.params, bronze=args.bronze, out=args.out)


if __name__ == "__main__":
    main()
