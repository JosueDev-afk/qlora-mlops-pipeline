"""The carrier-phrase bank: the only LLM-written text in the corpus.

Three steps, and only the last one lets a phrase into the corpus:

1. `generate` (on Colab, against Qwen3-8B served by vLLM): requests spread
   evenly over task, label and persona, from prompts/carrier_phrases/v1.yaml.
   The model never writes a value: Task A phrases carry `{value}`, which
   generate fills with a spoken value whose label is known.
2. Code filters every candidate (one `{value}` in Task A and none elsewhere;
   no digits, emails or links; a sane length) and drops duplicates of what
   the bank already holds, so reruns only add.
3. `review` (in a terminal): a native speaker approves, rejects or edits each
   pending phrase. Only approved phrases reach the corpus, with who approved
   them and when; the bank file is the record (DATA_PROVENANCE.md).

    vllm serve Qwen/Qwen3-8B --max-model-len 4096 &          # Colab, L4
    python -m src.pipeline.synth.carrier generate --base-url http://localhost:8000/v1
    python -m src.pipeline.synth.carrier review               # the Mac, by hand

The bank cannot be regenerated (the review is human), so it is versioned with
DVC and copied off this disk (CLAUDE.md: the local remote is not a backup).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import re
import subprocess
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.common.config import load_params, param
from src.common.log import configure_logging, get_logger
from src.common.prompts import Prompt, load_prompt
from src.pipeline.datasets.examples import normalize_text
from src.pipeline.synth.generate import TEMPLATES

PROMPT_ID, PROMPT_VERSION = "carrier_phrases", 1
PLACEHOLDER = "{value}"
FIELDS = ("name", "phone", "email")
KINDS = ("backchannel", "correction", "objection", "rejection")
MAX_WORDS = {"extract_entity": 30, "classify_intent": 25, "is_real_interruption": 8}
FORBIDDEN = re.compile(r"\d|@|https?://|www\.")

Complete = Callable[[list[dict[str, str]], int], str]
log = get_logger(__name__)


class CarrierError(RuntimeError):
    """The bank or its settings cannot be used as asked."""


@dataclass(frozen=True)
class Request:
    task: str
    label: str  # field for A, intent for B, kind for D
    persona: str
    n: int
    seed: int
    context: str | None = None


# ── what the bank means by a label ─────────────────────────────────────────


def descriptions() -> dict[str, dict[str, str]]:
    """Intent and kind descriptions from the Laya prompts the corpus trains."""
    intents = load_prompt("classify_intent_laya", 1).body["questions"]["intent"]["criteria"]
    kinds = load_prompt("is_real_interruption", 2).body["questions"]["kind"]["criteria"]
    prompt = load_prompt(PROMPT_ID, PROMPT_VERSION).body
    return {
        "classify_intent": {k: " ".join(str(v).split()) for k, v in intents.items()},
        "is_real_interruption": {
            "backchannel": " ".join(prompt["backchannel"].split()),
            **{k: str(kinds[k]) for k in KINDS if k != "backchannel"},
        },  # fmt: skip
    }


def contexts() -> dict[str, list[str]]:
    """Agent lines each intent can answer, from the generator's flows."""
    flows = yaml.safe_load(TEMPLATES.read_text())["flows"]
    out: dict[str, list[str]] = {}
    for spec in flows.values():
        for label in spec["labels"]:
            out.setdefault(label, []).extend(spec["context"])
    return out


# ── step 1: requests ───────────────────────────────────────────────────────


def plan(settings: dict[str, Any], personas: list[str]) -> list[Request]:
    """Requests that over-generate each (task, label) share of target_count."""
    labels = {"extract_entity": list(FIELDS),
              "classify_intent": sorted(descriptions()["classify_intent"]),
              "is_real_interruption": list(KINDS)}  # fmt: skip
    lines = contexts()
    per_request = int(settings["per_request"])
    requests = []
    for task, share in sorted(settings["shares"].items()):
        wanted = settings["target_count"] * share * settings["overgenerate"] / len(labels[task])
        for label in labels[task]:
            for i in range(math.ceil(wanted / per_request)):
                seed = int(hashlib.sha256(f"{task}:{label}:{i}".encode()).hexdigest()[:8], 16)
                context = lines[label][i % len(lines[label])] if task == "classify_intent" else None
                requests.append(Request(task, label, personas[i % len(personas)], per_request,
                                        seed, context))  # fmt: skip
    return requests


def messages(request: Request, prompt: Prompt) -> list[dict[str, str]]:
    body = prompt.body
    if request.task == "extract_entity":
        instruction = body["instructions"]["extract_entity"].format(
            field=body["fields"][request.label]
        )
    else:
        description = descriptions()[request.task][request.label]
        instruction = body["instructions"][request.task].format(
            context=request.context or "", description=description
        )
    user = body["user_template"].format(
        instruction=" ".join(instruction.split()),
        persona=body["personas"][request.persona],
        n=request.n,
    )
    return [{"role": "system", "content": body["system"]}, {"role": "user", "content": user}]


def openai_complete(base_url: str, model: str, temperature: float) -> Complete:
    """Chat completions on an OpenAI-compatible server (vLLM), thinking off."""

    def complete(chat: list[dict[str, str]], seed: int) -> str:
        body = json.dumps({
            "model": model, "messages": chat, "temperature": temperature, "top_p": 0.95,
            "seed": seed, "max_tokens": 1200,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()  # fmt: skip
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return str(json.load(response)["choices"][0]["message"]["content"])

    return complete


# ── step 2: filter ─────────────────────────────────────────────────────────


def parse(raw: str) -> list[str]:
    """The phrases in a reply; tolerant of a code fence, since this runs offline."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return []
    try:
        phrases = json.loads(match.group(0)).get("phrases", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [p for p in phrases if isinstance(p, str)] if isinstance(phrases, list) else []


def reject_reason(task: str, text: str) -> str | None:
    """Why a candidate never reaches review, or None if it may."""
    words = len(text.replace(PLACEHOLDER, "x").split())
    if task == "extract_entity" and text.count(PLACEHOLDER) != 1:
        return "needs exactly one {value}"
    if task != "extract_entity" and PLACEHOLDER in text:
        return "unexpected {value}"
    if FORBIDDEN.search(text.replace(PLACEHOLDER, "")):
        return "contains a digit, email or link"
    if not 1 <= words <= MAX_WORDS[task]:
        return f"{words} words"
    return None


def _clean(text: str) -> str:
    return " ".join(text.strip().strip("\"'«»“”").split())


def _key(task: str, label: str, text: str) -> str:
    return f"{task}|{label}|{normalize_text(text.replace(PLACEHOLDER, 'VALOR'))}"


def candidates(
    request: Request, raw: str, prompt: Prompt, seen: set[str], model: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejected: Counter[str] = Counter()
    out = []
    for text in map(_clean, parse(raw)):
        reason = reject_reason(request.task, text)
        key = _key(request.task, request.label, text)
        if reason or key in seen:
            rejected[reason or "duplicate"] += 1
            continue
        seen.add(key)
        out.append({
            "id": "cp-" + hashlib.sha256(key.encode()).hexdigest()[:12],
            "task": request.task, "label": request.label, "text": text,
            "persona": request.persona, "context": request.context, "status": "pending",
            "generator": model, "seed": request.seed,
            "prompt": {"id": prompt.id, "version": prompt.version, "sha256": prompt.sha256},
            "reviewed_by": None, "reviewed_at": None,
        })  # fmt: skip
    return out, rejected


# ── the bank file ──────────────────────────────────────────────────────────


def load_bank(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_bank(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    """Atomic: a review interrupted mid-write never loses decisions already made."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in entries)
    )
    tmp.replace(path)


def approved(path: Path) -> list[dict[str, Any]]:
    """What generate may use: approved by a person, and still passing the filter."""
    return [e for e in load_bank(path)
            if e["status"] == "approved" and not reject_reason(e["task"], e["text"])]  # fmt: skip


def generate_bank(
    path: Path,
    requests: list[Request],
    complete: Complete,
    *,
    model: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run requests, append new candidates as pending; returns a summary."""
    prompt = load_prompt(PROMPT_ID, PROMPT_VERSION)
    bank = load_bank(path)
    seen = {_key(e["task"], e["label"], e["text"]) for e in bank}
    rejected: Counter[str] = Counter()
    added = 0
    for request in requests[:limit]:
        try:
            raw = complete(messages(request, prompt), request.seed)
        except OSError as exc:  # one failed request does not lose the others
            rejected["request_failed"] += 1
            log.warning(
                "carrier_request_failed", task=request.task, label=request.label, error=repr(exc)
            )
            continue
        new, dropped = candidates(request, raw, prompt, seen, model)
        bank += new
        added += len(new)
        rejected += dropped
        save_bank(path, bank)
    summary = {"requests": len(requests[:limit]), "added": added, "rejected": dict(rejected),
               "bank": dict(Counter(e["status"] for e in bank))}  # fmt: skip
    log.info("carrier_generated", **summary)
    return summary


# ── step 3: human review ───────────────────────────────────────────────────


def _reviewer() -> str:
    try:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout
    except OSError:
        name = ""
    return name.strip() or getpass.getuser()


def review(
    path: Path,
    *,
    reviewer: str,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> Counter[str]:
    """Walk pending phrases: [a]pprove, [r]eject, [e]dit, [s]kip, [q]uit. Resumable."""
    bank = load_bank(path)
    done: Counter[str] = Counter()
    pending = [e for e in bank if e["status"] == "pending"]
    for i, entry in enumerate(pending, 1):
        say(f"\n[{i}/{len(pending)}] {entry['task']} · {entry['label']} · {entry['persona']}")
        if entry.get("context"):
            say(f"  agente: {entry['context']}")
        say(f"  frase:  {entry['text']}")
        while True:
            choice = ask("  [a]probar [r]echazar [e]ditar [s]altar [q]uitar > ").strip().lower()
            if choice in ("a", "r"):
                entry["status"] = "approved" if choice == "a" else "rejected"
            elif choice == "e":
                text = _clean(ask("  nueva frase > "))
                reason = reject_reason(entry["task"], text)
                if reason:
                    say(f"  no: {reason}")
                    continue
                entry.setdefault("original_text", entry["text"])
                entry["text"], entry["status"] = text, "approved"
            elif choice == "s":
                break
            elif choice == "q":
                say(f"guardado: {dict(done)}")
                return done
            else:
                continue
            entry["reviewed_by"] = reviewer
            entry["reviewed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            done[entry["status"]] += 1
            save_bank(path, bank)
            break
    say(f"listo: {dict(done)}")
    return done


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate or review the carrier-phrase bank.")
    sub = parser.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("generate", help="on Colab, against an OpenAI-compatible server")
    gen.add_argument("--base-url", required=True)
    gen.add_argument("--limit", type=int, default=None, help="first N requests only (a dry run)")
    rev = sub.add_parser("review", help="approve, reject or edit pending phrases")
    rev.add_argument("--reviewer", default=None)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    params = load_params(args.params)
    settings = dict(param(params, "synth.carrier_phrases"))
    bank = Path(settings["bank"])
    if args.action == "generate":
        model = str(settings["generator"])
        complete = openai_complete(args.base_url, model, float(settings["temperature"]))
        requests = plan(settings, list(param(params, "synth.personas")))
        print(json.dumps(generate_bank(bank, requests, complete, model=model, limit=args.limit),
                         ensure_ascii=False, indent=2))  # fmt: skip
    else:
        review(bank, reviewer=args.reviewer or _reviewer())


if __name__ == "__main__":
    main()
