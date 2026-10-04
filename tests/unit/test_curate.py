"""Replay curation: Spanish, short, clean of personal data, de-duplicated,
and in the gold contract. The rules run on row dicts; the parquet path is
covered where pyarrow is installed.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.pipeline.curate.replay import curate, drop_reason
from src.pipeline.datasets.examples import invalid_reason

ROWS = [
    {"language": "Spanish", "inputs": "¿Qué es un volcán?", "targets": "Una abertura.",
     "annotation_type": "original-annotations"},
    {"language": "Spanish", "inputs": "¿Qué  es un   volcán?", "targets": "Una abertura."},
    {"language": "English", "inputs": "What is a volcano?", "targets": "An opening."},
    {"language": "Spanish", "inputs": "Escríbeme a juan@correo.com", "targets": "Claro."},
    {"language": "Spanish", "inputs": "Llama al 81 8234 5678", "targets": "Listo."},
    {"language": "Spanish", "inputs": "¿Cuándo fue la guerra?", "targets": "Entre 1914-1918."},
    {"language": "Spanish", "inputs": "Mira www.ejemplo.mx", "targets": "Ok."},
    {"language": "Spanish", "inputs": "Cuenta", "targets": "palabra " * 300},
    {"language": "Spanish", "inputs": "  ", "targets": "vacío"},
]  # fmt: skip


def run(rows=ROWS) -> tuple[list[dict], dict[str, int]]:
    return curate(rows, language="Spanish", max_words=250, dataset="CohereLabs/aya_dataset",
                  revision="f" * 40, license="apache-2.0")  # fmt: skip


def test_keeps_clean_spanish_and_counts_every_drop() -> None:
    examples, stats = run()
    assert sorted(e["output"]["response"] for e in examples) == [
        "Entre 1914-1918.",
        "Una abertura.",
    ]
    assert stats == {"read": 9, "in_language": 8, "duplicate": 1, "email": 1, "phone": 1,
                     "link": 1, "too_long": 1, "empty": 1, "kept": 2}  # fmt: skip


def test_a_year_range_is_not_a_phone() -> None:
    assert drop_reason("¿Cuándo?", "Entre 1914-1918", 250) is None
    assert drop_reason("Llama al 818 234 5678", "Ok", 250) == "phone"


def test_examples_meet_the_gold_contract_and_record_provenance() -> None:
    examples, _ = run()
    assert not [e["id"] for e in examples if invalid_reason(e)]
    first = examples[0]
    assert first["task"] == first["source"] == "replay" and first["group"] == first["id"]
    assert first["meta"]["license"] == "apache-2.0" and first["meta"]["revision"] == "f" * 40


def test_ids_are_stable_and_whitespace_does_not_make_a_new_row() -> None:
    first, _ = run()
    again, _ = run(list(reversed(ROWS)))
    assert [e["id"] for e in first] == [e["id"] for e in again]


def test_curate_reads_bronze_at_the_pin(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from src.pipeline.curate.run import CurateError, curate_replay
    from src.pipeline.ingest.run import ingest_source
    from src.pipeline.ingest.sources import load_sources

    params = yaml.safe_load(Path("params.yaml").read_text())
    params["ingest"]["sources"]["aya_es"]["verified"] = True
    with pytest.raises(CurateError, match="make ingest"):
        curate_replay(params, tmp_path / "bronze", tmp_path / "silver")

    def fetch(source, dest: Path) -> None:
        (dest / "data").mkdir()
        pq.write_table(pa.Table.from_pylist(ROWS), dest / "data" / "train-00000-of-00001.parquet")

    ingest_source(load_sources(params)["aya_es"], tmp_path / "bronze", fetch=fetch)
    manifest = curate_replay(params, tmp_path / "bronze", tmp_path / "silver")
    assert manifest["rows"]["kept"] == 2
    lines = (tmp_path / "silver" / "replay" / "aya_es.jsonl").read_text().splitlines()
    assert all(json.loads(line)["task"] == "replay" for line in lines)
