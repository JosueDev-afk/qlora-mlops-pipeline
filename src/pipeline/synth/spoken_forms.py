"""Canonical value -> how a caller in Mexico would say it.

The inverse of src/common/normalizers.py, and the reason synthetic labels are
correct by construction: the generator starts from the value (the label) and
produces the transcript, never the other way round. Round-trip tests hold
every readable style to normalize(speak(value)) == value.

Styles are variation axes from the Phase 2 design (grouping by ones, twos or
threes, corrections mid-phrase, spelled vs spoken emails). Every random choice
goes through the `rng` passed in, so synth.seed regenerates the exact corpus.

Word tables are imported from the normalizers, so the two sides cannot drift.
"""

from __future__ import annotations

import random
import re
from typing import Literal

from src.common.normalizers import (
    AMBIGUOUS_LETTERS,
    EMAIL_PREFIX,
    EMAIL_RE,
    EMAIL_SKIP,
    HUNDREDS,
    LETTERS,
    REPEAT,
    TEENS,
    TENS,
    UNITS,
)

PhoneStyle = Literal["digits", "pairs", "triples", "mixed", "correction"]
EmailStyle = Literal["words", "spelled", "mixed"]
PHONE_STYLES: tuple[PhoneStyle, ...] = ("digits", "pairs", "triples", "mixed", "correction")
EMAIL_STYLES: tuple[EmailStyle, ...] = ("words", "spelled", "mixed")

DIGIT_WORDS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
TEEN_WORDS = {
    10: "diez", 11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince",
    16: "dieciséis", 17: "diecisiete", 18: "dieciocho", 19: "diecinueve", 20: "veinte",
    21: "veintiuno", 22: "veintidós", 23: "veintitrés", 24: "veinticuatro", 25: "veinticinco",
    26: "veintiséis", 27: "veintisiete", 28: "veintiocho", 29: "veintinueve",
}  # fmt: skip
TENS_WORDS = {
    3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta",
    7: "setenta", 8: "ochenta", 9: "noventa",
}  # fmt: skip
HUNDRED_WORDS = {
    1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos",
    6: "seiscientos", 7: "setecientos", 8: "ochocientos", 9: "novecientos",
}  # fmt: skip
GROUPINGS = {"digits": [1] * 10, "pairs": [2] * 5, "triples": [3, 3, 2, 2]}
PHONE_PREFIXES = ["", "es el ", "mi número es ", "a ver, es el ", "mi celular es ", "sería el "]
COUNTRY_CODES = ["más cincuenta y dos ", "cincuenta y dos "]

LETTER_NAMES = {
    "a": ["a"], "b": ["be", "be grande", "be de burro", "be larga"], "c": ["ce"], "d": ["de"],
    "e": ["e"], "f": ["efe"], "g": ["ge"], "h": ["hache"], "i": ["i"], "j": ["jota"],
    "k": ["ka"], "l": ["ele"], "m": ["eme"], "n": ["ene"], "o": ["o"], "p": ["pe"],
    "q": ["cu"], "r": ["ere", "erre"], "s": ["ese"], "t": ["te"], "u": ["u"],
    "v": ["ve", "uve", "ve chica", "ve de vaca", "ve corta"], "w": ["doble ve", "doble u"],
    "x": ["equis"], "y": ["i griega", "ye"], "z": ["zeta"],
}  # fmt: skip
EMAIL_PREFIXES = ["", "mi correo es ", "es ", "sería ", "mi correo electrónico es "]
# Words the normalizer would read as something else: spell them instead.
RESERVED = (
    set(LETTERS) | set(AMBIGUOUS_LETTERS) | set(UNITS) | set(TEENS) | set(TENS)
    | set(HUNDREDS) | set(REPEAT) | EMAIL_SKIP | EMAIL_PREFIX
    | {"arroba", "punto", "guion", "underscore", "bajo", "medio", "griega",
       "grande", "larga", "alta", "chica", "corta", "baja", "burro", "vaca", "seria"}
)  # fmt: skip
PHONE_RE = re.compile(r"^\+52[2-9]\d{9}$")


# ── numbers ───────────────────────────────────────────────────────────────


def _below_100(n: int) -> str:
    if n < 10:
        return DIGIT_WORDS[n]
    if n < 30:
        return TEEN_WORDS[n]
    tens, unit = divmod(n, 10)
    return TENS_WORDS[tens] + (f" y {DIGIT_WORDS[unit]}" if unit else "")


def _one_by_one(digits: str, rng: random.Random) -> str:
    """Digit by digit, sometimes collapsing repeats into "doble" / "triple"."""
    words, i = [], 0
    while i < len(digits):
        run = len(digits[i:]) - len(digits[i:].lstrip(digits[i]))
        name = DIGIT_WORDS[int(digits[i])]
        if run >= 3 and rng.random() < 0.5:
            words.append(f"triple {name}")
            i += 3
        elif run >= 2 and rng.random() < 0.5:
            words.append(f"doble {name}")
            i += 2
        else:
            words.append(name)
            i += 1
    return " ".join(words)


def _group(g: str, rng: random.Random, *, last: bool) -> str:
    """One dictation group of 1-3 digits, read back by the normalizer as exactly `g`."""
    if len(g) == 1:
        return DIGIT_WORDS[int(g)]
    if g[0] == "0":
        return "cero " + _group(g[1:], rng, last=last) if len(g) == 3 else _one_by_one(g, rng)
    if len(g) == 2:
        return _below_100(int(g))
    hundreds, rest = int(g[0]), int(g[1:])
    if rest:
        return f"{HUNDRED_WORDS[hundreds]} {_below_100(rest)}"
    if hundreds == 1:
        return "cien"
    # "doscientos, treinta y cuatro" and "doscientos treinta y cuatro" transcribe
    # the same, so a bare hundred is only safe as the last group.
    return HUNDRED_WORDS[hundreds] if last else _one_by_one(g, rng)


def _speak_digits(digits: str, sizes: list[int], rng: random.Random, sep: str) -> str:
    groups, i = [], 0
    for size in sizes:
        groups.append(digits[i : i + size])
        i += size
    return sep.join(_group(g, rng, last=k == len(groups) - 1) for k, g in enumerate(groups))


def _random_sizes(total: int, rng: random.Random) -> list[int]:
    sizes: list[int] = []
    while sum(sizes) < total:
        sizes.append(rng.randint(1, min(3, total - sum(sizes))))
    return sizes


# ── phone ─────────────────────────────────────────────────────────────────


def random_phone(rng: random.Random) -> str:
    """A valid Mexican E.164 number: 10 digits, never starting with 0 or 1."""
    return "+52" + str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9))


def speak_phone(e164: str, style: PhoneStyle, rng: random.Random) -> str:
    """How a caller dictates `e164`.

    "correction" restates a wrong group, says "no, perdón" and dictates the
    whole number again. The normalizer returns None for it on purpose: which
    group a correction replaces cannot be known from text, so those examples
    teach the model, not the rule-based baseline.
    """
    if not PHONE_RE.match(e164):
        raise ValueError(f"not a canonical Mexican phone: {e164!r}")
    if style not in PHONE_STYLES:
        raise ValueError(f"unknown phone style {style!r}")
    digits = e164[3:]
    sep = rng.choice([", ", " "])
    prefix = rng.choice(PHONE_PREFIXES)
    if rng.random() < 0.15:
        prefix += rng.choice(COUNTRY_CODES)

    if style == "correction":
        sizes = GROUPINGS["pairs"]
        cut = rng.randint(1, len(sizes)) * 2
        wrong = digits[: cut - 1] + str((int(digits[cut - 1]) + rng.randint(1, 9)) % 10)
        said_wrong = _speak_digits(wrong, sizes[: cut // 2], rng, sep)
        return f"{prefix}{said_wrong}, no, perdón, {_speak_digits(digits, sizes, rng, sep)}"

    sizes = _random_sizes(10, rng) if style == "mixed" else GROUPINGS[style]
    if style == "digits":
        return prefix + _one_by_one(digits, rng)
    return prefix + _speak_digits(digits, sizes, rng, sep)


# ── email ─────────────────────────────────────────────────────────────────


def _spell(chunk: str, rng: random.Random) -> str:
    names = []
    for char in chunk:
        options = LETTER_NAMES[char]
        if char == "y" and len(chunk) == 1:
            options = ["i griega"]  # a lone "ye" could be read as a word
        names.append(rng.choice(options))
    return " ".join(names)


def _speakable(word: str) -> bool:
    return len(word) >= 2 and word.isalpha() and word.isascii() and word not in RESERVED


def _speak_chunk(chunk: str, spell: bool, rng: random.Random) -> str:
    if chunk == ".":
        return "punto"
    if chunk == "_":
        return "guion bajo"
    if chunk == "-":
        return rng.choice(["guion", "guion medio"])
    if chunk.isdigit():
        if len(chunk) <= 3 and rng.random() < 0.5:
            return _group(chunk, rng, last=True)
        if rng.random() < 0.5:
            return _one_by_one(chunk, rng)
        sizes = [2] * (len(chunk) // 2) + [1] * (len(chunk) % 2)
        return _speak_digits(chunk, sizes, rng, " ")
    return _spell(chunk, rng) if spell or not _speakable(chunk) else chunk


def speak_email(address: str, style: EmailStyle, rng: random.Random) -> str:
    """How a caller says `address`.

    "words" says pronounceable chunks as words, "spelled" spells the whole
    local part letter by letter, "mixed" chooses per chunk. The domain is
    always said as words when it can be ("arroba gmail punto com"), because
    that is how people say it. Spelling an all-ambiguous chunk ("ed") or a
    lone d, t or s is genuinely ambiguous, so the normalizer refuses those.
    """
    if not EMAIL_RE.match(address) or re.search(r"[^a-z0-9._@-]", address):
        raise ValueError(f"not a canonical address: {address!r}")
    if style not in EMAIL_STYLES:
        raise ValueError(f"unknown email style {style!r}")
    local, domain = address.split("@")

    def spell_local() -> bool:
        return style == "spelled" or (style == "mixed" and rng.random() < 0.5)

    chunks = re.findall(r"[a-z]+|\d+|[._-]", local)
    said_local = " ".join(_speak_chunk(c, spell_local(), rng) for c in chunks)
    said_domain = " punto ".join(
        " ".join(_speak_chunk(c, False, rng) for c in re.findall(r"[a-z]+|\d+|-", label))
        for label in domain.split(".")
    )
    return f"{rng.choice(EMAIL_PREFIXES)}{said_local} arroba {said_domain}"
