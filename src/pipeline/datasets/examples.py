"""Gold examples: the row contract, and the helpers build and gate share.

The contract is `schemas/data/example.schema.json`; a task's `output` must also
be valid against `schemas/<task>.schema.json`, so a label that the agent would
reject never reaches training.

Splits are assigned per `group`, never per row: augmentation writes ~10 noisy
variants of each dialogue, and a dialogue in both train and test would score
memorisation as generalisation.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"
SPLITS = ("train", "val", "test")
TASKS = ("extract_entity", "parse_datetime", "classify_intent", "is_real_interruption")
REPLAY = "replay"


@cache
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMAS_DIR / name).read_text()))


def invalid_reason(row: object) -> str | None:
    """Why `row` cannot be a gold example, or None if it can."""
    if not isinstance(row, dict):
        return "not an object"
    errors = sorted(_validator("data/example.schema.json").iter_errors(row), key=str)
    if errors:
        return f"example: {errors[0].message}"
    task = row["task"]
    if task == REPLAY:
        return None
    errors = sorted(_validator(f"{task}.schema.json").iter_errors(row["output"]), key=str)
    if errors:
        return f"{task} output: {errors[0].message}"
    if task == "extract_entity" and row["output"]["field"] != row["input"].get("field"):
        return "extract_entity output.field differs from input.field"
    return None


def read_jsonl(paths: Iterable[Path]) -> Iterator[tuple[str, Any]]:
    """(where, row) for every line; unparseable lines come back as the raw text."""
    for path in sorted(paths):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    yield f"{path.name}:{n}", json.loads(line)
                except json.JSONDecodeError:
                    yield f"{path.name}:{n}", line


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """One key-sorted row per line, so the same rows are the same bytes (DVC hashes)."""
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def split_of(group: str, seed: int, ratios: dict[str, float]) -> str:
    """Stable split for a group: its hash, not its position, so adding groups moves none."""
    point = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for split in SPLITS:
        edge += ratios[split]
        if point <= edge:
            return split
    return SPLITS[-1]


def normalize_text(text: str) -> str:
    """For contamination checks: case, accents (ñ kept), punctuation and spacing folded."""
    text = text.lower().replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s\0]", " ", text.replace("\0", "ñ"))
    return " ".join(text.split())
