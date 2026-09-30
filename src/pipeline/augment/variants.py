"""One noisy variant of an example, relabelled so the label stays true.

Noise on carrier words never changes a label. Noise on a dictated value
changes it only when it changes what was heard, which depends on the edit
kind and the field:

- homophones ("sero" for "cero", "a roba") and fillers leave a phone or an
  email recoverable: the label stays the true value. This is the robustness
  the fine-tuned model must learn; labelling it "ask again" would cap the
  model at the rule baseline.
- In a phone, a respelled word ("sinco") is still the same digit, and a
  dropped "y" ("treinta cuatro") is still 34: recoverable too.
- In a name or an email, spelling is the value: a respelling (b/v, z/s)
  sounds identical, so neither the model nor the readback can tell. Ask again.
- Mishearings ("setenta" for "sesenta", "ene" for "eme") and, in an email,
  dropped letters change the value: the label becomes what was heard when it
  is still well formed, and "ask again" when it is not. The readback catches
  a wrong but well-formed value, as in a real call.
- A value-less example never gains a value: if noise made one readable, the
  variant is dropped, since no label is right for it.

Tasks B and D keep their labels: protected words (no, sí, ya...) are never
edited, so noise cannot flip an intent.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from src.common.normalizers import normalize_email, spoken_to_e164
from src.pipeline.augment.noise import (
    Confusions,
    Noised,
    apply_noise,
    heard_value,
    measured_wer,
    tokenize,
)

UNRESOLVED_CONFIDENCE = 0.5  # templates_es_mx.yaml confidence.ambiguous


def asr_confidence(wer: float, rng: random.Random) -> float:
    """PROVISIONAL: falls with the applied WER. Plan task 10 replaces it with the
    measured relation between Scribe's confidence and its errors."""
    return round(min(0.99, max(0.30, 1.0 - 1.4 * wer + rng.gauss(0.0, 0.03))), 3)


def _read(field: str, raw: str) -> str | None:
    return spoken_to_e164(raw) if field == "phone" else normalize_email(raw)


def _unresolved(output: dict[str, Any]) -> None:
    output.update({"normalized_value": None, "confidence": UNRESOLVED_CONFIDENCE,
                   "needs_reprompt": True, "reprompt_reason": "low_confidence"})  # fmt: skip


# Edit kinds after which the true value can still be read from the transcript.
RECOVERABLE = {
    "phone": frozenset({"homophone", "respelled", "filler", "deleted"}),
    "email": frozenset({"homophone", "filler"}),
    "name": frozenset({"filler"}),
}


def relabel(variant: dict[str, Any], original: dict[str, Any], noised: Noised) -> str:
    """Fix variant's Task A label in place; return kept | heard | unresolved | dropped."""
    out = variant["output"]
    field, style = variant["input"]["field"], original["meta"].get("style")
    truth = original["output"]["normalized_value"]
    kinds = noised.value_kinds()
    if kinds <= RECOVERABLE[field]:
        return "kept"
    unknowable = (
        field == "name"
        or style in ("correction", "said_spelled")
        or (field == "email" and "respelled" in kinds)
    )
    if unknowable:
        if truth is not None:
            _unresolved(out)
            return "unresolved"
        return "kept"
    heard = _read(field, heard_value(noised, keep=set(kinds - RECOVERABLE[field])))
    if truth is None:  # never gains a value it was not given
        return "dropped" if heard else "kept"
    if heard == truth:
        return "kept"
    if heard is None:
        _unresolved(out)
        return "unresolved"
    out["normalized_value"] = heard
    return "heard"


def make_variant(
    example: dict[str, Any],
    k: int,
    wer: float,
    confusions: Confusions,
    *,
    seed: int,
    emphasis: bool,
) -> tuple[dict[str, Any] | None, str]:
    """Variant k of example at the target WER, or None when it has no true label."""
    rng = random.Random(f"{seed}:{example['id']}:{k}")
    meta = example.get("meta") or {}
    tokens = tokenize(example["input"]["transcript"], meta.get("value_span"))
    noised = apply_noise(tokens, wer, confusions, rng, emphasis=emphasis)
    variant = copy.deepcopy(example)
    variant["id"] = f"{example['id']}-v{k:02d}"
    text = noised.text()
    variant["input"]["transcript"] = text
    applied = measured_wer(noised, len(tokens))
    variant["meta"] = {
        **meta,
        "wer": min(applied, 1.0),
        "wer_target": wer,
        "augment": {"ops": noised.ops, "confusions": confusions.version},
    }
    if example["task"] != "extract_entity":
        return variant, "kept"
    span = noised.value_span()
    variant["meta"]["value_span"] = list(span) if span else None
    if example["output"]["raw_span"] is not None:
        variant["output"]["raw_span"] = text[span[0] : span[1]] if span else None
    variant["input"]["asr_confidence"] = asr_confidence(applied, rng)
    status = relabel(variant, example, noised)
    return (None, status) if status == "dropped" else (variant, status)
