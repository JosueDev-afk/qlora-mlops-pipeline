"""Code-generated examples for Tasks A, B and D, each built from its label.

The label is chosen first and the transcript is written from it, never the
other way round, so labels are correct by construction and no LLM judges
them. Phones and emails are spoken by spoken_forms, the inverse of the
normalizers; names, intents and interruptions come from the reviewed seed
templates in templates_es_mx.yaml.

Every example gets its own RNG, seeded by (seed, task, index): the corpus is
reproducible, and changing one task's count never reshuffles another's.

Task C (parse_datetime) is not generated yet: it belongs to confirm_appointment,
whose prompt and label conventions ("en la tarde" -> which hour?) are still to
be decided.
"""

from __future__ import annotations

import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.common.normalizers import name_key, normalize_name
from src.pipeline.synth.spoken_forms import (
    DIGIT_WORDS,
    EMAIL_STYLES,
    LETTER_NAMES,
    PHONE_STYLES,
    random_phone,
    speak_email,
    speak_phone,
)

TEMPLATES = Path(__file__).with_name("templates_es_mx.yaml")
TASKS = ("extract_entity", "classify_intent", "is_real_interruption")
FIELDS = ("name", "phone", "email")
KINDS = frozenset({"backchannel", "correction", "objection", "rejection", "unclear"})
INTENTS = frozenset({"confirmed", "cannot_attend", "ambiguous", "call_rejected", "out_of_scope"})
ACCENTED = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
NEGATIVES = {"name": ("not_provided", "partial"), "phone": ("not_provided", "partial"),
             "email": ("not_provided", "invalid_format")}  # fmt: skip


class TemplateError(ValueError):
    """The templates cannot produce valid examples for these params."""


@dataclass(frozen=True)
class Templates:
    version: int
    raw: dict[str, Any]
    real_interruptions: dict[str, str]  # phrase -> kind
    backchannels: tuple[str, ...]
    ambiguous_tokens: frozenset[str]  # name tokens with a homophone in the lists

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_templates(
    personas: list[str],
    backchannels: list[str],
    real_interruptions: list[str],
    path: Path = TEMPLATES,
) -> Templates:
    """Load and cross-check against params, reporting every mismatch at once."""
    raw = yaml.safe_load(path.read_text())
    problems = []
    missing = sorted(set(personas) - set(raw["personas"]))
    if missing:
        problems.append(f"personas without templates: {missing}")
    kinds = dict(raw.get("interruption_kinds") or {}) | dict(
        raw.get("extra_real_interruptions") or {}
    )
    unlabelled = [p for p in real_interruptions if p not in kinds]
    if unlabelled:
        problems.append(
            f"synth.real_interruptions without a kind in interruption_kinds: {unlabelled}"
        )
    bad_kinds = sorted({k for k in kinds.values() if k not in KINDS or k == "backchannel"})
    if bad_kinds:
        problems.append(f"invalid interruption kinds: {bad_kinds}")
    bad_intents = sorted(set(raw["intents"]) - INTENTS)
    for flow in raw["flows"].values():
        bad_intents += [label for label in flow["labels"] if label not in raw["intents"]]
    if bad_intents:
        problems.append(f"unknown intent labels: {sorted(set(bad_intents))}")
    if problems:
        raise TemplateError(f"{path.name}:\n  " + "\n  ".join(problems))

    homophones: dict[str, set[str]] = defaultdict(set)
    for name in raw["names"]["first"] + raw["names"]["last"]:
        for token in name.split():
            if token[0].isupper():  # particles ("de", "la") are not names
                homophones[name_key(token)].add(token)
    ambiguous = frozenset(
        t for spellings in homophones.values() if len(spellings) > 1 for t in spellings
    )
    real = {p: kinds[p] for p in real_interruptions} | dict(
        raw.get("extra_real_interruptions") or {}
    )
    return Templates(int(raw["version"]), raw, real, tuple(backchannels), ambiguous)


def _ascii(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )


def spell(word: str) -> str:
    """How a caller spells a name aloud: "equis, i, eme, e, ene, a"."""
    names = []
    for char in word.lower():
        if char in ACCENTED:
            names.append(f"{ACCENTED[char]} con acento")
        elif char == "ñ":
            names.append("eñe")
        elif char in LETTER_NAMES:
            names.append(LETTER_NAMES[char][0])
    return ", ".join(names)


@dataclass
class Generator:
    templates: Templates
    personas: list[str]
    seed: int
    carrier: tuple[dict[str, Any], ...] = ()  # approved bank entries (carrier.approved)
    _spaces: dict[str, list[tuple[Any, ...]]] = field(default_factory=dict, repr=False)

    def _rng(self, task: str, index: int) -> random.Random:
        return random.Random(f"{self.seed}:{task}:{index}")

    def _example(self, task: str, index: int, inputs: dict[str, Any], output: dict[str, Any],
                 *, source: str = "template", **meta: Any) -> dict[str, Any]:  # fmt: skip
        example_id = f"syn-{task}-{'c' if source == 'carrier' else ''}{index:06d}"
        generator = (f"templates_es_mx@v{self.templates.version}" if source == "template"
                     else f"carrier_bank:{meta.get('carrier_id')}")  # fmt: skip
        return {
            "schema_version": 1,
            "id": example_id,
            "group": example_id,  # augment's noise variants will share it
            "task": task,
            "source": source,
            "input": inputs,
            "output": output,
            "meta": {"generator": generator, "seed": self.seed, **meta},
        }

    # ── Task A ─────────────────────────────────────────────────────────────

    def _name(self, rng: random.Random) -> tuple[str, str, bool]:
        """(spoken, canonical, ambiguous): said aloud, or said and spelled."""
        names = self.templates["names"]
        parts = [rng.choice(names["first"]), rng.choice(names["last"]), rng.choice(names["last"])]
        full = " ".join(parts)
        tokens = full.split()
        ambiguous = [t for t in tokens if t in self.templates.ambiguous_tokens]
        if ambiguous and rng.random() < 0.5:
            token = ambiguous[0]
            return f"{full.lower()}, {token.lower()} se escribe {spell(token)}", full, False
        return full.lower(), full, bool(ambiguous)

    def _email_address(self, rng: random.Random) -> str:
        names = self.templates["names"]
        first = _ascii(rng.choice(names["first"]).split()[0])
        last = _ascii(rng.choice(names["last"]).split()[-1])
        digits = str(rng.randint(1, 99)) if rng.random() < 0.4 else ""
        return f"{first}.{last}{digits}@{rng.choice(self.templates['email_domains'])}"

    def _negative(self, rng: random.Random, index: int, field: str, persona: str) -> dict[str, Any]:
        """No usable value: never given, cut short, or malformed. The value is null
        in every case: the model must ask again, never complete it by guessing."""
        reason = rng.choice(NEGATIVES[field])
        spoken: str | None = None
        span: list[int] | None = None
        if reason == "not_provided":
            transcript = rng.choice(self.templates["negatives"]["not_provided"][field])
        else:
            if field == "name":  # first name only, for "nombre completo"
                spoken = rng.choice(self.templates["names"]["first"]).lower()
            elif field == "phone":  # 7-9 digits
                digits = str(rng.randint(2, 9)) + "".join(
                    str(rng.randint(0, 9)) for _ in range(rng.randint(6, 8))
                )
                spoken = spoken_digits(digits)
            else:  # no top-level domain: "juan arroba gmail"
                head, _, domain = speak_email(self._email_address(rng), "words", rng).partition(
                    " arroba "
                )
                spoken = f"{head} arroba {domain.split(' punto ')[0]}"
            wrap = rng.choice(self.templates["personas"][persona]["wrap"])
            start = wrap.index("{value}")
            span = [start, start + len(spoken)]
            transcript = wrap.replace("{value}", spoken)
        output = {"field": field, "normalized_value": None, "raw_span": spoken,
                  "confidence": self.templates["confidence"]["absent"], "needs_reprompt": True,
                  "reprompt_reason": reason}  # fmt: skip
        return self._example("extract_entity", index,
                             {"field": field, "transcript": transcript, "asr_confidence": None},
                             output, persona=persona, style=reason, value_span=span)  # fmt: skip

    def extract_entity(self, index: int) -> dict[str, Any]:
        rng = self._rng("extract_entity", index)
        field = FIELDS[index % len(FIELDS)]
        persona = rng.choice(self.personas)
        if rng.random() < self.templates["negatives_share"]:
            return self._negative(rng, index, field, persona)

        frame = rng.choice(self.templates["personas"][persona]["wrap"])
        return self._dictated(rng, index, field, persona, frame)

    def _dictated(self, rng: random.Random, index: int, field: str, persona: str, frame: str,
                  *, source: str = "template", **meta: Any) -> dict[str, Any]:  # fmt: skip
        """A spoken value inside a frame: a persona wrap, or an approved carrier phrase."""
        conf = self.templates["confidence"]
        base = {"field": field, "normalized_value": None, "raw_span": None}
        if field == "name":
            spoken, canonical, ambiguous = self._name(rng)
            style = "said" if ", " not in spoken else "said_spelled"
        elif field == "phone":
            canonical = random_phone(rng)
            style = rng.choice(PHONE_STYLES)
            spoken, ambiguous = speak_phone(canonical, style, rng), False
        else:
            canonical = self._email_address(rng)
            style = rng.choice(EMAIL_STYLES)
            spoken, ambiguous = speak_email(canonical, style, rng), False

        start = frame.index("{value}")
        transcript = frame.replace("{value}", spoken)
        if ambiguous:  # heard, but the spelling cannot be known: ask, never guess
            output = {**base, "raw_span": spoken, "confidence": conf["ambiguous"],
                      "needs_reprompt": True, "reprompt_reason": "low_confidence"}  # fmt: skip
        else:
            value = normalize_name(canonical) if field == "name" else canonical
            output = {**base, "normalized_value": value, "raw_span": spoken,
                      "confidence": conf["clear"], "needs_reprompt": False,
                      "reprompt_reason": None}  # fmt: skip
        return self._example("extract_entity", index,
                             {"field": field, "transcript": transcript, "asr_confidence": None},
                             output, source=source, persona=persona, style=style,
                             value_span=[start, start + len(spoken)], **meta)  # fmt: skip

    # ── Task B ─────────────────────────────────────────────────────────────

    # Tasks B and D have a small, finite template space: every combination is
    # emitted once, in a seeded order, instead of sampling duplicates that
    # build_gold would drop. Volume beyond it comes from the carrier-phrase
    # bank and from augment's noise variants, not from repetition.

    def _space(self, task: str) -> list[tuple[Any, ...]]:
        if task not in self._spaces:
            combos = sorted(set(self._combinations(task)))
            random.Random(f"{self.seed}:{task}:order").shuffle(combos)
            self._spaces[task] = combos
        return self._spaces[task]

    def _combinations(self, task: str) -> list[tuple[Any, ...]]:
        t = self.templates
        if task == "carrier_classify_intent":
            return [
                (flow, context, entry["label"], entry["text"], entry["persona"], entry["id"])
                for entry in self._bank("classify_intent")
                for flow, spec in sorted(t["flows"].items())
                if entry["label"] in spec["labels"]
                for context in spec["context"]
            ]
        if task == "carrier_is_real_interruption":
            return [
                (utterance, position, entry["text"], entry["label"], entry["id"])
                for entry in self._bank("is_real_interruption")
                for utterance in t["agent_utterances"]
                for position in range(1, len(utterance.split()))
            ]
        if task == "classify_intent":
            leads = sorted({(lead, p) for p in self.personas for lead in t["personas"][p]["lead"]})
            return [
                (flow, context, label, lead + phrase, persona)
                for flow, spec in t["flows"].items()
                for context in spec["context"]
                for label in spec["labels"]
                for phrase in t["intents"][label]
                for lead, persona in leads
            ]
        phrases = [(p, kind) for p, kind in t.real_interruptions.items()]
        phrases += [(p, "backchannel") for p in t.backchannels]
        return [
            (utterance, position, phrase, kind)
            for utterance in t["agent_utterances"]
            for position in range(1, len(utterance.split()))
            for phrase, kind in phrases
        ]

    def capacity(self, task: str, source: str = "template") -> int | None:
        """How many distinct examples are possible; None when unbounded."""
        if source == "carrier":
            if task == "extract_entity":
                return None if self._bank(task) else 0
            return len(self._space(f"carrier_{task}"))
        return (
            len(self._space(task)) if task in ("classify_intent", "is_real_interruption") else None
        )

    def _bank(self, task: str) -> list[dict[str, Any]]:
        return sorted((e for e in self.carrier if e["task"] == task), key=lambda e: e["id"])

    def classify_intent(self, index: int) -> dict[str, Any]:
        flow, context, label, transcript, persona = self._space("classify_intent")[index]
        inputs = {"flow": flow, "context": context, "transcript": transcript}
        output = {"intent": label, "confidence": self.templates["confidence"]["clear"]}
        return self._example("classify_intent", index, inputs, output, persona=persona, flow=flow)

    # ── Task D ─────────────────────────────────────────────────────────────

    def is_real_interruption(self, index: int) -> dict[str, Any]:
        """`agent_said` is what the agent had spoken when the caller overlapped:
        the context the model sees in production (synth.real_interruptions)."""
        utterance, position, phrase, kind = self._space("is_real_interruption")[index]
        output = {"interruption": kind != "backchannel",
                  "confidence": self.templates["confidence"]["clear"], "kind": kind}  # fmt: skip
        inputs = {
            "agent_utterance": utterance,
            "agent_said": " ".join(utterance.split()[:position]),
            "transcript": phrase,
        }
        return self._example("is_real_interruption", index, inputs, output, position=position)

    # ── carrier phrases: approved by a person, values still filled by code ──

    def carrier_extract_entity(self, index: int) -> dict[str, Any]:
        entries = self._bank("extract_entity")
        entry = entries[index % len(entries)]  # every phrase in turn, a new value each time
        rng = self._rng("carrier_extract_entity", index)
        return self._dictated(rng, index, entry["label"], entry["persona"], entry["text"],
                              source="carrier", carrier_id=entry["id"])  # fmt: skip

    def carrier_classify_intent(self, index: int) -> dict[str, Any]:
        flow, context, label, text, persona, carrier_id = self._space("carrier_classify_intent")[
            index
        ]
        inputs = {"flow": flow, "context": context, "transcript": text}
        output = {"intent": label, "confidence": self.templates["confidence"]["clear"]}
        return self._example("classify_intent", index, inputs, output, source="carrier",
                             persona=persona, flow=flow, carrier_id=carrier_id)  # fmt: skip

    def carrier_is_real_interruption(self, index: int) -> dict[str, Any]:
        utterance, position, text, kind, carrier_id = self._space("carrier_is_real_interruption")[
            index
        ]
        output = {"interruption": kind != "backchannel",
                  "confidence": self.templates["confidence"]["clear"], "kind": kind}  # fmt: skip
        inputs = {
            "agent_utterance": utterance,
            "agent_said": " ".join(utterance.split()[:position]),
            "transcript": text,
        }
        return self._example("is_real_interruption", index, inputs, output, source="carrier",
                             position=position, carrier_id=carrier_id)  # fmt: skip

    def generate(self, task: str, index: int, source: str = "template") -> dict[str, Any]:
        method = f"carrier_{task}" if source == "carrier" else task
        return dict(getattr(self, method)(index))


def spoken_digits(digits: str) -> str:
    """Digit by digit, for partial numbers that must not form a valid phone."""
    return " ".join(DIGIT_WORDS[int(d)] for d in digits)
