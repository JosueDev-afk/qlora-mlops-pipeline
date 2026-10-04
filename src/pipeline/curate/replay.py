"""Aya (Spanish) -> replay examples in the gold row contract.

Replay keeps general instruction following alive while the model narrows
onto two call flows (H5b). It is written by people (Aya is human-annotated),
so curation only has to keep what fits and drop what must not train:

- Spanish rows only (Aya is multilingual);
- whitespace normalized, empty sides dropped;
- longer than `max_words` on either side dropped: a voice agent's turn is
  short, and long answers would dominate the token budget of a 4B model;
- anything carrying an email, a phone-like run of 9+ digits or a link dropped, not
  redacted: replay teaches instruction following, and a redacted answer
  teaches redaction;
- exact duplicates (after normalization) dropped.

Pure Python over row dicts, so it is unit-tested without pyarrow; the reader
lives in run.py.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from src.pipeline.datasets.examples import normalize_text

EMAIL = re.compile(r"\S+@\S+\.\w+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
LINK = re.compile(r"https?://|www\.", re.I)


def _clean(text: object) -> str:
    return " ".join(str(text or "").split())


def drop_reason(instruction: str, response: str, max_words: int) -> str | None:
    if not instruction or not response:
        return "empty"
    if max(len(instruction.split()), len(response.split())) > max_words:
        return "too_long"
    for text in (instruction, response):
        if EMAIL.search(text):
            return "email"
        if LINK.search(text):
            return "link"
        # 9+ digits: a phone has 10; a year range ("1914-1918") has 8.
        if any(sum(c.isdigit() for c in m.group()) >= 9 for m in PHONE.finditer(text)):
            return "phone"
    return None


def curate(
    rows: Iterable[dict[str, Any]],
    *,
    language: str,
    max_words: int,
    dataset: str,
    revision: str,
    license: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replay examples sorted by id, and how many rows each rule dropped."""
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    out = []
    for row in rows:
        stats["read"] += 1
        if row.get("language") != language:
            continue
        stats["in_language"] += 1
        instruction, response = _clean(row.get("inputs")), _clean(row.get("targets"))
        reason = drop_reason(instruction, response, max_words)
        key = normalize_text(instruction) + "\0" + normalize_text(response)
        if reason or key in seen:
            stats[reason or "duplicate"] += 1
            continue
        seen.add(key)
        example_id = "rep-aya-" + hashlib.sha256(key.encode()).hexdigest()[:12]
        out.append({
            "schema_version": 1,
            "id": example_id,
            "group": example_id,
            "task": "replay",
            "source": "replay",
            "input": {"instruction": instruction},
            "output": {"response": response},
            "meta": {"dataset": dataset, "revision": revision, "license": license,
                     "annotation_type": row.get("annotation_type")},
        })  # fmt: skip
    stats["kept"] = len(out)
    return sorted(out, key=lambda r: r["id"]), dict(stats)
