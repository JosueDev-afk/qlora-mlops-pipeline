"""Download public corpora, raw and pinned, into bronze.

Bronze is immutable and append-only: `data/bronze/<source>/ingest_date=<d>/`
holds the files exactly as published plus `_manifest.json` (origin, pin,
license, sha256 of every file). Curation reads from there and never from
the network.

- **Licenses first.** A source is downloaded only once its `verified` flag
  is true in params.yaml, the maintainer's check against DATA_PROVENANCE.md.
- **Idempotent.** A source whose pin is already in bronze, with its files
  intact, is not downloaded again.
- **Atomic.** Files land in a hidden temporary directory that is renamed
  into place only after the checksum and manifest are written, so a crash
  or a bad download never leaves a partition that looks complete.

    python -m src.pipeline.ingest.run --out data/bronze [--source massive]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.common.config import load_params
from src.common.log import configure_logging, get_logger
from src.pipeline.ingest.sources import Source, load_sources

MANIFEST = "_manifest.json"
BRONZE = Path("data/bronze")
CHUNK = 1 << 20

Fetcher = Callable[[Source, Path], None]
log = get_logger(__name__)


class LicenseNotVerifiedError(RuntimeError):
    """The source's license has not been checked by the maintainer yet."""


class ChecksumMismatchError(RuntimeError):
    """The download is not the file params.yaml pinned."""


class BronzeConflictError(RuntimeError):
    """A partition already exists with other contents; bronze is never overwritten."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(source: Source, dest: Path, timeout_s: float = 60.0) -> None:
    """Stream the file to `dest`; the name is the URL's last segment."""
    assert source.url is not None
    target = dest / source.url.rstrip("/").rsplit("/", 1)[-1]
    with urllib.request.urlopen(source.url, timeout=timeout_s) as response, target.open("wb") as f:
        shutil.copyfileobj(response, f, CHUNK)


def fetch_hf(source: Source, dest: Path) -> None:
    """Snapshot of a Hub dataset at the pinned commit (imported lazily: pipeline group)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        revision=source.revision,
        local_dir=dest,
        allow_patterns=list(source.allow_patterns) if source.allow_patterns else None,
    )
    shutil.rmtree(dest / ".cache", ignore_errors=True)  # hub bookkeeping, not data


FETCHERS: dict[str, Fetcher] = {"url": fetch_url, "hf_dataset": fetch_hf}


def _files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name != MANIFEST
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def _intact(partition: Path, manifest: dict[str, Any]) -> bool:
    """Every listed file present with its size. Sizes, not hashes: a 1 GB rehash per run
    would cost more than the download check is worth; hashes are checked on ingest."""
    for entry in manifest.get("files", []):
        path = partition / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            return False
    return bool(manifest.get("files"))


def find_existing(source: Source, bronze: Path) -> Path | None:
    for manifest_path in sorted((bronze / source.name).glob(f"*/{MANIFEST}")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("pin") == source.pin and _intact(manifest_path.parent, manifest):
            return manifest_path.parent
    return None


def ingest_source(
    source: Source,
    bronze: Path = BRONZE,
    *,
    today: date | None = None,
    fetch: Fetcher | None = None,
) -> Path:
    """Bring one source into bronze, or return the partition that already holds it."""
    if not source.verified:
        raise LicenseNotVerifiedError(
            f"{source.name} ({source.license}): set verified: true in params.yaml after "
            "checking its license in docs/DATA_PROVENANCE.md"
        )
    existing = find_existing(source, bronze)
    if existing is not None:
        log.info("ingest_skipped", source=source.name, partition=str(existing))
        return existing

    today = today or datetime.now(UTC).date()
    partition = bronze / source.name / f"ingest_date={today.isoformat()}"
    if partition.exists():
        raise BronzeConflictError(
            f"{partition} exists with another pin; bronze is never overwritten"
        )
    staging = partition.with_name(f".{partition.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)  # a previous crash
    staging.mkdir(parents=True)
    try:
        (fetch or FETCHERS[source.kind])(source, staging)
        files = _files(staging)
        if not files:
            raise RuntimeError(f"{source.name}: the download produced no files")
        entries: list[dict[str, Any]] = [
            {"path": str(p.relative_to(staging)), "bytes": p.stat().st_size, "sha256": sha256_of(p)}
            for p in files
        ]
        if source.kind == "url" and [e["sha256"] for e in entries] != [source.sha256]:
            raise ChecksumMismatchError(f"{source.name}: expected {source.sha256}, got {entries}")
        manifest = {
            "source": source.name,
            "kind": source.kind,
            "origin": source.origin,
            "pin": source.pin,
            "license": source.license,
            "purpose": source.purpose,
            "ingest_date": today.isoformat(),
            "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "total_bytes": sum(e["bytes"] for e in entries),
            "files": entries,
        }
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(partition)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log.info(
        "ingested", source=source.name, partition=str(partition), bytes=manifest["total_bytes"]
    )
    return partition


def verified_sources(params_path: str = "params.yaml") -> list[str]:
    """The sources the DAG maps over; unverified ones are reported, not downloaded."""
    sources = load_sources(load_params(params_path))
    for source in sources.values():
        if not source.verified:
            log.warning("ingest_unverified_license", source=source.name, license=source.license)
    return [name for name, source in sources.items() if source.verified]


def ingest_named(name: str, params_path: str = "params.yaml", bronze: Path = BRONZE) -> str:
    """One source by name: what each mapped DAG task calls."""
    sources = load_sources(load_params(params_path))
    if name not in sources:
        raise KeyError(f"no source {name!r} in ingest.sources")
    return str(ingest_source(sources[name], bronze))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download pinned public corpora into bronze.")
    parser.add_argument("--out", type=Path, default=BRONZE)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--source", action="append", help="only these; default: every verified")
    args = parser.parse_args(argv)

    configure_logging()
    names = args.source or verified_sources(args.params)
    for name in names:
        ingest_named(name, args.params, args.out)


if __name__ == "__main__":
    main()
