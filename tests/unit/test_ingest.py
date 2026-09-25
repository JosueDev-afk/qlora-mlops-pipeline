"""Bronze ingest: pinned, license-gated, idempotent, atomic.

No network: fetchers are fakes that write files, so what is pinned here is
what reaches bronze and what never does.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.common.config import load_params
from src.pipeline.ingest.run import (
    MANIFEST,
    BronzeConflictError,
    ChecksumMismatchError,
    LicenseNotVerifiedError,
    ingest_source,
)
from src.pipeline.ingest.sources import Source, SourceError, load_sources

PAYLOAD = b"es-ES\tjsonl\n" * 100
DAY = date(2026, 9, 25)


def url_source(**overrides: Any) -> Source:
    spec = {
        "name": "massive", "kind": "url", "license": "cc-by-4.0", "purpose": "intents",
        "verified": True, "url": "https://example.org/amazon-massive-dataset-1.1.tar.gz",
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    }  # fmt: skip
    return Source(**{**spec, **overrides})


def hf_source(**overrides: Any) -> Source:
    spec = {
        "name": "ciempiess_light", "kind": "hf_dataset", "license": "cc-by-sa-4.0",
        "purpose": "audio", "verified": True, "repo_id": "ciempiess/ciempiess_light",
        "revision": "3d6afb2b3b8dd00ad8f5b1288fe4c18f5882faaa",
    }  # fmt: skip
    return Source(**{**spec, **overrides})


class FakeFetch:
    def __init__(self, files: dict[str, bytes] | None = None, fail: Exception | None = None):
        self.files = files if files is not None else {"amazon-massive-dataset-1.1.tar.gz": PAYLOAD}
        self.fail = fail
        self.calls = 0

    def __call__(self, source: Source, dest: Path) -> None:
        self.calls += 1
        for name, data in self.files.items():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).write_bytes(data)
        if self.fail:
            raise self.fail


# ── sources ────────────────────────────────────────────────────────────────


def test_committed_params_describe_valid_pinned_sources() -> None:
    sources = load_sources(load_params("params.yaml"))
    assert {"massive", "callcenter_en", "ciempiess_light"} <= set(sources)
    assert all(s.pin for s in sources.values())


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        ({"kind": "hf_dataset", "repo_id": "a/b", "revision": "main"}, "40-char commit"),
        ({"kind": "url", "url": "https://x/y.tgz"}, "sha256"),
        ({"kind": "ftp"}, "unknown kind"),
        ({"kind": "url", "url": "https://x", "sha256": "0" * 64, "license": ""}, "license"),
        ({"kind": "url", "url": "https://x", "sha256": "0" * 64, "verified": "yes"}, "verified"),
    ],
)
def test_unpinned_or_unaudited_sources_are_refused(spec: dict, problem: str) -> None:
    base = {"license": "cc-by-4.0", "purpose": "p", "verified": False}
    with pytest.raises(SourceError, match=problem):
        load_sources({"ingest": {"sources": {"demo": {**base, **spec}}}})


# ── ingest ─────────────────────────────────────────────────────────────────


def test_writes_files_and_manifest_into_a_dated_partition(tmp_path: Path) -> None:
    partition = ingest_source(url_source(), tmp_path, today=DAY, fetch=FakeFetch())
    assert partition == tmp_path / "massive" / "ingest_date=2026-09-25"
    manifest = json.loads((partition / MANIFEST).read_text())
    assert manifest["pin"] == url_source().sha256
    assert manifest["license"] == "cc-by-4.0"
    assert manifest["files"] == [{
        "path": "amazon-massive-dataset-1.1.tar.gz",
        "bytes": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    }]  # fmt: skip
    assert not list((tmp_path / "massive").glob(".*"))  # no staging left behind


def test_unverified_license_is_never_downloaded(tmp_path: Path) -> None:
    fetch = FakeFetch()
    with pytest.raises(LicenseNotVerifiedError, match="DATA_PROVENANCE"):
        ingest_source(url_source(verified=False), tmp_path, today=DAY, fetch=fetch)
    assert fetch.calls == 0 and not (tmp_path / "massive").exists()


def test_same_pin_is_not_downloaded_again_even_on_another_day(tmp_path: Path) -> None:
    fetch = FakeFetch()
    first = ingest_source(url_source(), tmp_path, today=DAY, fetch=fetch)
    again = ingest_source(url_source(), tmp_path, today=date(2026, 10, 1), fetch=fetch)
    assert again == first and fetch.calls == 1


def test_a_damaged_partition_is_downloaded_again_elsewhere(tmp_path: Path) -> None:
    fetch = FakeFetch()
    first = ingest_source(url_source(), tmp_path, today=DAY, fetch=fetch)
    (first / "amazon-massive-dataset-1.1.tar.gz").write_bytes(b"truncated")
    second = ingest_source(url_source(), tmp_path, today=date(2026, 10, 1), fetch=fetch)
    assert second != first and fetch.calls == 2


def test_checksum_mismatch_leaves_nothing_in_bronze(tmp_path: Path) -> None:
    fetch = FakeFetch({"amazon-massive-dataset-1.1.tar.gz": b"tampered"})
    with pytest.raises(ChecksumMismatchError):
        ingest_source(url_source(), tmp_path, today=DAY, fetch=fetch)
    assert list((tmp_path / "massive").iterdir()) == []


def test_interrupted_download_leaves_nothing_in_bronze(tmp_path: Path) -> None:
    fetch = FakeFetch(fail=ConnectionError("reset by peer"))
    with pytest.raises(ConnectionError):
        ingest_source(hf_source(), tmp_path, today=DAY, fetch=fetch)
    assert list((tmp_path / "ciempiess_light").iterdir()) == []


def test_empty_download_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no files"):
        ingest_source(hf_source(), tmp_path, today=DAY, fetch=FakeFetch(files={}))


def test_bronze_is_never_overwritten(tmp_path: Path) -> None:
    ingest_source(hf_source(), tmp_path, today=DAY, fetch=FakeFetch({"a.parquet": b"1"}))
    newer = hf_source(revision="f" * 40)
    with pytest.raises(BronzeConflictError):
        ingest_source(newer, tmp_path, today=DAY, fetch=FakeFetch({"a.parquet": b"2"}))


def test_hub_snapshots_keep_their_layout_and_drop_hub_bookkeeping(tmp_path: Path) -> None:
    files = {"ciempiess_light/train-00000-of-00004.parquet": b"x" * 10,
             ".cache/huggingface/download/meta": b"hub"}  # fmt: skip
    partition = ingest_source(hf_source(), tmp_path, today=DAY, fetch=FakeFetch(files))
    manifest = json.loads((partition / MANIFEST).read_text())
    assert [f["path"] for f in manifest["files"]] == [
        "ciempiess_light/train-00000-of-00004.parquet"
    ]
    assert manifest["origin"] == "hf://datasets/ciempiess/ciempiess_light"
    assert manifest["pin"] == hf_source().revision
