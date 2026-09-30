"""Rule-based ASR noise at a target word error rate.

Edits are substitutions from the confusion tables, deletions of function
words, and inserted fillers, counted against the clean transcript so the
applied WER is measured, not assumed. Every token keeps what it was and what
kind of edit it received, because the kinds relabel differently:

    homophone   same sound, another spelling, from the curated table
    respelled   a spelling rule (b/v, z/s...): same sound, new letters
    mishearing  another word, so another value
    filler      inserted hesitation
    deleted     a dropped function word (kept, hidden, so it can be restored)

Tokens also remember whether they belong to the dictated value (the
generator's value_span); with emphasis on alphanumerics, value tokens are
three times as likely to be edited. Protected words (no, sí, ya...) are never
touched: noise must not turn "no puedo" into "puedo".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

CONFUSIONS = Path(__file__).with_name("confusions_es_mx.yaml")
PUNCTUATION = ",.;:¿?¡!…"
VALUE_WEIGHT = 3.0

Kind = Literal["homophone", "respelled", "mishearing", "filler", "deleted"]


@dataclass(frozen=True)
class Confusions:
    version: int
    operations: dict[str, float]
    homophones: dict[str, list[str]]
    mishearings: dict[str, list[str]]
    characters: list[tuple[str, str]]
    deletable: frozenset[str]
    protected: frozenset[str]
    fillers: list[str]


def load_confusions(path: Path = CONFUSIONS) -> Confusions:
    raw = yaml.safe_load(path.read_text())
    return Confusions(
        version=int(raw["version"]),
        operations={k: float(v) for k, v in raw["operations"].items()},
        homophones={str(k): list(v) for k, v in raw["homophones"].items()},
        mishearings={str(k): list(v) for k, v in raw["mishearings"].items()},
        characters=[(str(a), str(b)) for a, b in raw["characters"]],
        deletable=frozenset(raw["deletable"]),
        protected=frozenset(str(w) for w in raw["protected"]),
        fillers=list(raw["fillers"]),
    )


@dataclass
class Token:
    text: str
    in_value: bool = False
    original: str | None = None  # before a substitution; None when untouched
    kind: Kind | None = None

    @property
    def shown(self) -> bool:
        return self.kind != "deleted"


@dataclass
class Noised:
    tokens: list[Token]
    ops: dict[str, int] = field(default_factory=lambda: {"substitute": 0, "delete": 0, "insert": 0})

    @property
    def edits(self) -> int:
        return sum(self.ops.values())

    def value_tokens(self) -> list[Token]:
        return [t for t in self.tokens if t.in_value]

    def value_kinds(self) -> set[str]:
        return {t.kind for t in self.value_tokens() if t.kind}

    def text(self) -> str:
        return " ".join(t.text for t in self.tokens if t.shown)

    def value_span(self) -> tuple[int, int] | None:
        """Character span of the value in text(), or None when noise erased it."""
        offset, start, end = 0, None, None
        for token in (t for t in self.tokens if t.shown):
            if token.in_value:  # punctuation stuck to a word is not part of the value
                lead, _, trail = _core(token.text)
                start = offset + len(lead) if start is None else start
                end = offset + len(token.text) - len(trail)
            offset += len(token.text) + 1
        return (start, end) if start is not None and end is not None else None


def tokenize(text: str, value_span: list[int] | None) -> list[Token]:
    tokens, offset = [], 0
    for word in text.split(" "):
        lead, core, trail = _core(word)
        start, end = offset + len(lead), offset + len(word) - len(trail)
        inside = (
            bool(core)
            and value_span is not None
            and start >= value_span[0]
            and end <= value_span[1]
        )
        if word:
            tokens.append(Token(word, inside))
        offset += len(word) + 1
    return tokens


def _core(word: str) -> tuple[str, str, str]:
    """(leading punctuation, lowercase core, trailing punctuation)."""
    stripped = word.strip(PUNCTUATION)
    lead = word[: len(word) - len(word.lstrip(PUNCTUATION))]
    trail = word[len(word.rstrip(PUNCTUATION)) :]
    return lead, stripped.lower(), trail


def _substitute(word: str, confusions: Confusions, rng: random.Random) -> tuple[str, Kind] | None:
    lead, core, trail = _core(word)
    if not core or core in confusions.protected:
        return None
    table: list[tuple[list[str], Kind]] = []
    if core in confusions.homophones:
        table.append((confusions.homophones[core], "homophone"))
    if core in confusions.mishearings:
        table.append((confusions.mishearings[core], "mishearing"))
    if table:
        options, kind = rng.choice(table)
        return lead + rng.choice(options) + trail, kind
    rules = [(a, b) for a, b in confusions.characters if a in core]
    if not rules:
        return None
    a, b = rng.choice(rules)
    at = rng.choice([i for i in range(len(core)) if core.startswith(a, i)])
    return lead + core[:at] + b + core[at + len(a) :] + trail, "respelled"


def apply_noise(
    tokens: list[Token],
    wer: float,
    confusions: Confusions,
    rng: random.Random,
    *,
    emphasis: bool = True,
) -> Noised:
    """Edit about wer * words times; a word is edited at most once.

    The count is rounded at random (2.3 edits: 2, or 3 with probability 0.3),
    so short turns still get noise and the mean WER matches the target.
    """
    out = Noised([Token(t.text, t.in_value) for t in tokens])
    exact = wer * len(tokens)
    target = int(exact) + (1 if rng.random() < exact - int(exact) else 0)
    ops, weights = zip(*confusions.operations.items(), strict=True)
    for _ in range(target * 4):  # attempts: not every token allows every edit
        if out.edits >= target:
            break
        candidates = [i for i, t in enumerate(out.tokens) if t.kind is None]
        if not candidates:
            break
        pick = rng.choices(
            candidates,
            weights=[
                VALUE_WEIGHT if emphasis and out.tokens[i].in_value else 1.0 for i in candidates
            ],
        )[0]
        token = out.tokens[pick]
        op = rng.choices(ops, weights=weights)[0]
        if op == "substitute":
            new = _substitute(token.text, confusions, rng)
            if new is None:
                continue
            token.original, (token.text, token.kind) = token.text, new
        elif op == "delete":
            core = _core(token.text)[1]
            if core not in confusions.deletable or core in confusions.protected:
                continue
            token.kind = "deleted"
        else:  # a filler inside the value only when both neighbours are in it
            inside = token.in_value and pick > 0 and out.tokens[pick - 1].in_value
            out.tokens.insert(pick, Token(rng.choice(confusions.fillers), inside, kind="filler"))
        out.ops[op] += 1
    return out


def measured_wer(noised: Noised, reference_words: int) -> float:
    return round(noised.edits / max(reference_words, 1), 4)


def heard_value(noised: Noised, *, keep: set[str]) -> str:
    """The value as heard, keeping only the edit kinds in `keep`: others are undone
    (original text restored, fillers dropped). What a relabel should read."""
    words = []
    for token in noised.value_tokens():
        if token.kind == "filler":
            if "filler" in keep:
                words.append(token.text)
        elif token.kind == "deleted":
            if "deleted" not in keep:
                words.append(token.text)
        elif token.kind in keep or token.kind is None:
            words.append(token.text)
        else:
            words.append(token.original or token.text)
    return " ".join(words)


def describe(confusions: Confusions) -> dict[str, Any]:
    return {"version": confusions.version, "operations": confusions.operations}
