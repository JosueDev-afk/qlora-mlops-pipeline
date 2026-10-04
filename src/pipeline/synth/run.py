"""Generate the code-built corpus into data/synthetic.

    python -m src.pipeline.synth.run --out data/synthetic

`synth.n_dialogues` examples, split evenly across Tasks A, B and D. Every row
is checked against the gold contract before it is written: a generator bug
fails here, not three stages later in the quality gate. The output is
deterministic (same params and templates, same bytes) and replaced whole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger
from src.pipeline.datasets.examples import invalid_reason, write_jsonl
from src.pipeline.synth.carrier import approved
from src.pipeline.synth.generate import TASKS, TEMPLATES, Generator, load_templates

OUT = Path("data/synthetic")
MANIFEST = "manifest.json"

log = get_logger(__name__)


class GenerationError(RuntimeError):
    """The generator produced a row the gold contract rejects: a bug, not data."""


def per_task(n: int) -> dict[str, int]:
    base, extra = divmod(n, len(TASKS))
    return {task: base + (1 if i < extra else 0) for i, task in enumerate(TASKS)}


def capped(
    generator: Generator, requested: dict[str, int], source: str = "template"
) -> dict[str, int]:
    """Never more rows than distinct examples: the rest would be duplicates."""
    counts = {}
    for task, n in requested.items():
        capacity = generator.capacity(task, source)
        counts[task] = n if capacity is None else min(n, capacity)
        if counts[task] < n and source == "template":
            log.warning("template_space_exhausted", task=task, requested=n, distinct=capacity,
                        hint="more volume needs the carrier-phrase bank and augment")  # fmt: skip
    return counts


def generate(
    generator: Generator, counts: dict[str, int], source: str = "template"
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for task, n in counts.items():
        rows = [generator.generate(task, i, source) for i in range(n)]
        bad = [(r["id"], reason) for r in rows if (reason := invalid_reason(r))]
        if bad:
            raise GenerationError(f"{task}: {len(bad)} invalid rows, first: {bad[:3]}")
        out[task] = rows
    return out


def summary(parts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def count(task: str, key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(r["meta"].get(key)) for r in parts[task]).items()))

    return {
        task: {
            "rows": len(rows),
            **({"persona": count(task, "persona")} if task != "is_real_interruption" else {}),
            **(
                {
                    "style": count(task, "style"),
                    "field": dict(sorted(Counter(r["input"]["field"] for r in rows).items())),
                }
                if task == "extract_entity"
                else {}
            ),
            **(
                {"label": dict(sorted(Counter(r["output"]["intent"] for r in rows).items()))}
                if task == "classify_intent"
                else {}
            ),
            **(
                {"kind": dict(sorted(Counter(r["output"]["kind"] for r in rows).items()))}
                if task == "is_real_interruption"
                else {}
            ),
        }  # fmt: skip
        for task, rows in parts.items()
    }


def run(
    params_path: str = "params.yaml",
    *,
    out: Path = OUT,
    templates: Path = TEMPLATES,
    bank: Path | None = None,
) -> str:
    """Template examples always; carrier examples from the approved bank, if any."""
    params = load_params(params_path)
    bank = bank or Path(param(params, "synth.carrier_phrases.bank"))
    phrases = approved(bank)
    if not phrases:
        log.warning(
            "carrier_bank_empty",
            bank=str(bank),
            hint="template examples only; build_gold needs carrier ones unless template_ratio is 1",
        )
    personas = list(param(params, "synth.personas"))
    loaded = load_templates(
        personas,
        list(param(params, "synth.backchannels")),
        list(param(params, "synth.real_interruptions")),
        templates,
    )
    seed = int(param(params, "synth.seed"))
    generator = Generator(loaded, personas, seed, carrier=tuple(phrases))
    requested = per_task(int(param(params, "synth.n_dialogues")))
    parts = generate(generator, capped(generator, requested))
    carrier_parts = generate(generator, capped(generator, requested, "carrier"), "carrier")
    for task, rows in carrier_parts.items():
        parts[task] = parts[task] + rows

    staging = out.with_name(f".{out.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for task, rows in parts.items():
        write_jsonl(staging / f"{task}.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "templates": {
            "file": templates.name,
            "version": loaded.version,
            "sha256": hashlib.sha256(templates.read_bytes()).hexdigest(),
        },  # fmt: skip
        "requested": requested,
        "template_space": {t: generator.capacity(t) for t in requested},
        "carrier": {
            "bank": str(bank),
            "sha256": hashlib.sha256(bank.read_bytes()).hexdigest() if bank.is_file() else None,
            "approved": len(phrases),
            "space": {t: generator.capacity(t, "carrier") for t in requested},
            "rows": {t: len(r) for t, r in carrier_parts.items()},
        },
        "counts": summary(parts),
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    log.info("synthetic_generated", out=str(out), **{t: len(r) for t, r in parts.items()})
    return str(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate code-built examples by their labels.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    run(args.params, out=args.out)


if __name__ == "__main__":
    main()
