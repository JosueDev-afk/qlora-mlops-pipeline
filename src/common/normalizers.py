"""Turn what a caller says in es-MX into canonical values, or refuse.

These are the project's hardest pure functions: the real difficulty of both
flows is capturing alphanumeric data dictated over a noisy channel. They serve
three consumers: a deterministic baseline for Task A, the scorer that compares
predictions with gold after normalization, and the synthetic generator, which
must produce spoken forms these functions read back (round-trip tests).

They never guess. A correction mid-dictation, an unknown word, a phone with the
wrong number of digits or an address without a TLD returns None, and the agent
asks again. A wrong value written to the database costs more than a reprompt.
"""

from __future__ import annotations

import re
import unicodedata

# ── shared: folding, tokens, spoken numbers ───────────────────────────────

UNITS = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}  # fmt: skip
TEENS = {
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintiun": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29,
}  # fmt: skip
TENS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
}  # fmt: skip
HUNDREDS = {
    "cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300,
    "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600,
    "setecientos": 700, "ochocientos": 800, "novecientos": 900,
}  # fmt: skip
REPEAT = {"doble": 2, "triple": 3}


def _fold(text: str) -> str:
    """Lowercase and strip accents, keeping ñ: it is a different letter, not an accent."""
    text = text.lower().replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return text.replace("\0", "ñ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zñ']+|\d+|[@._+\-]", _fold(text))


def _below_100(tokens: list[str], i: int) -> tuple[int, int] | None:
    token = tokens[i]
    if token in TENS:
        value, j = TENS[token], i + 1
        if j + 1 < len(tokens) and tokens[j] == "y" and UNITS.get(tokens[j + 1], 0) > 0:
            value, j = value + UNITS[tokens[j + 1]], j + 2
        return value, j
    if token in TEENS:
        return TEENS[token], i + 1
    if token in UNITS:
        return UNITS[token], i + 1
    return None


def _spoken_number(tokens: list[str], i: int) -> tuple[str, int] | None:
    """Read one spoken number at tokens[i] as the digit string it stands for.

    "ochenta y uno" -> "81", "ocho" -> "8", "cero" -> "0", "doble ocho" -> "88",
    "ochocientos dieciocho" -> "818". Callers concatenate the strings, which is
    how people group digits when dictating.
    """
    token = tokens[i]
    if token in REPEAT and i + 1 < len(tokens) and tokens[i + 1] in UNITS:
        return str(UNITS[tokens[i + 1]]) * REPEAT[token], i + 2
    if token in HUNDREDS:
        value, j = HUNDREDS[token], i + 1
        if token != "cien" and j < len(tokens) and (rest := _below_100(tokens, j)):
            value, j = value + rest[0], rest[1]
        return str(value), j
    if below := _below_100(tokens, i):
        return str(below[0]), below[1]
    return None


# ── phone → E.164 ─────────────────────────────────────────────────────────

PHONE_FILLERS = frozenset({
    "es", "el", "la", "mi", "su", "numero", "telefono", "celular", "cel", "whatsapp",
    "de", "a", "ver", "este", "eh", "em", "mmm", "pues", "seria", "son", "y", "con",
    "lada", "codigo", "area", "movil", "casa", "oficina", "ok", "okey", "si", "claro",
    "le", "lo", "paso", "doy",
})  # fmt: skip
CORRECTION_MARKERS = frozenset({
    "no", "perdon", "perdona", "corrijo", "digo", "mejor", "espera", "espere",
    "esperese", "equivoque",
})  # fmt: skip


def spoken_to_e164(transcript: str) -> str | None:
    """Mexican phone transcript -> "+52XXXXXXXXXX", or None.

    Accepts digits, spoken numbers in any grouping, "doble"/"triple", a
    spoken +52, the legacy mobile prefixes (52 1, 044, 045) and filler words.
    A correction ("no, perdón") returns None: whether it replaces one group or
    the whole number cannot be known from text.
    """
    tokens, digits, i = _tokens(transcript), "", 0
    while i < len(tokens):
        token = tokens[i]
        if token.isdigit():
            digits, i = digits + token, i + 1
        elif token in CORRECTION_MARKERS:
            return None
        elif token in PHONE_FILLERS or token in {"mas", "+", "-", "."}:
            i += 1
        elif number := _spoken_number(tokens, i):
            digits, i = digits + number[0], number[1]
        else:
            return None

    if digits.startswith("00"):
        digits = digits[2:]
    if len(digits) == 13 and digits.startswith(("521", "044", "045")):
        digits = digits[3:]
    elif len(digits) == 12 and digits.startswith("52"):
        digits = digits[2:]
    if len(digits) != 10 or digits[0] in "01":
        return None
    return f"+52{digits}"


# ── spelled email → address ───────────────────────────────────────────────

# Letter names that are not ordinary words: always a letter.
LETTERS = {
    "be": "b", "ce": "c", "efe": "f", "ge": "g", "hache": "h", "jota": "j", "ka": "k",
    "ele": "l", "eme": "m", "ene": "n", "eñe": "ñ", "pe": "p", "cu": "q", "ere": "r",
    "erre": "r", "equis": "x", "zeta": "z", "ceta": "z", "ve": "v", "uve": "v",
}  # fmt: skip
# Letter names that are also words ("a", "de", "ese"): a letter only next to a letter.
AMBIGUOUS_LETTERS = {
    "a": "a", "de": "d", "e": "e", "i": "i", "o": "o", "u": "u", "te": "t", "ese": "s", "ye": "y",
}  # fmt: skip
# Multi-token letters and the descriptors people add to disambiguate b and v.
MULTI_LETTERS = [
    (("be", "de", "burro"), "b"), (("ve", "de", "vaca"), "v"),
    (("doble", "ve"), "w"), (("doble", "u"), "w"), (("doble", "uve"), "w"),
    (("i", "griega"), "y"),
    (("be", "grande"), "b"), (("be", "larga"), "b"), (("be", "alta"), "b"),
    (("ve", "chica"), "v"), (("ve", "corta"), "v"), (("ve", "baja"), "v"),
]  # fmt: skip
SYMBOLS = [
    (("guion", "bajo"), "_"), (("guion", "medio"), "-"),
    (("guion",), "-"), (("underscore",), "_"), (("arroba",), "@"), (("punto",), "."),
    (("@",), "@"), ((".",), "."), (("_",), "_"), (("-",), "-"), (("+",), "+"),
]  # fmt: skip
EMAIL_SKIP = frozenset({
    "todo", "junto", "pegado", "sin", "espacios", "espacio", "minuscula", "minusculas",
    "letra", "letras",
})  # fmt: skip
EMAIL_PREFIX = frozenset({
    "mi", "el", "su", "correo", "email", "e", "mail", "electronico", "pues", "a", "ver",
    "este", "okey", "ok", "si", "claro", "le", "paso", "doy",
})  # fmt: skip
EMAIL_RE = re.compile(
    r"^(?!\.)(?!.*\.\.)[a-z0-9._%+-]+(?<!\.)@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$"
)


def _strip_email_prefix(tokens: list[str]) -> list[str]:
    """Drop "mi correo es", "es", "sería"... but only when everything before is filler."""
    for k in range(min(6, len(tokens))):
        if tokens[k] in {"es", "seria"} and all(t in EMAIL_PREFIX for t in tokens[:k]):
            return tokens[k + 1 :]
    return tokens


def _match(
    tokens: list[str], i: int, table: list[tuple[tuple[str, ...], str]]
) -> tuple[str, int] | None:
    for pattern, out in table:
        if tuple(tokens[i : i + len(pattern)]) == pattern:
            return out, i + len(pattern)
    return None


def normalize_email(transcript: str) -> str | None:
    """Spelled or spoken email transcript -> lowercase address, or None.

    Handles letter names ("jota u a ene"), whole words ("juan punto perez"),
    spoken numbers, "arroba", "punto", "guion bajo/medio", "todo junto" and the
    b/v descriptors ("be de burro", "ve chica"). An address without a TLD is
    not completed ("arroba gmail" alone returns None): completing it is a guess.
    """
    tokens = _strip_email_prefix(_tokens(transcript))
    items: list[list[str]] = []  # [kind, text], kind in sym|digits|letter|ambig|word
    i = 0
    while i < len(tokens):
        if hit := _match(tokens, i, MULTI_LETTERS):
            items.append(["letter", hit[0]])
            i = hit[1]
        elif hit := _match(tokens, i, SYMBOLS):
            items.append(["sym", hit[0]])
            i = hit[1]
        elif tokens[i] in EMAIL_SKIP:
            i += 1
        elif tokens[i].isdigit():
            items.append(["digits", tokens[i]])
            i += 1
        elif number := _spoken_number(tokens, i):
            items.append(["digits", number[0]])
            i = number[1]
        elif tokens[i] in LETTERS:
            items.append(["letter", LETTERS[tokens[i]]])
            i += 1
        elif tokens[i] in AMBIGUOUS_LETTERS:
            items.append(["ambig", tokens[i]])
            i += 1
        else:
            items.append(["word", tokens[i]])
            i += 1

    changed = True
    while changed:  # an ambiguous letter name next to a letter is a letter too
        changed = False
        for k, (kind, text) in enumerate(items):
            neighbours = items[max(k - 1, 0) : k] + items[k + 1 : k + 2]
            if kind == "ambig" and any(n[0] == "letter" for n in neighbours):
                items[k] = ["letter", AMBIGUOUS_LETTERS[text]]
                changed = True

    for k, (kind, text) in enumerate(items):
        # "punto de punto" is the letter d or the word "de" with nothing to tell
        # them apart. Vowels are safe (the letter and the word are the same text);
        # de/te/ese/ye are not, unless a neighbouring word shows they are words.
        neighbours = items[max(k - 1, 0) : k] + items[k + 1 : k + 2]
        is_word_like = AMBIGUOUS_LETTERS.get(text) != text
        if kind == "ambig" and is_word_like and not any(n[0] == "word" for n in neighbours):
            return None

    address = "".join(text for _, text in items)
    return address if EMAIL_RE.match(address) else None


# ── names ─────────────────────────────────────────────────────────────────

PARTICLES = frozenset({
    "de", "del", "la", "las", "los", "y", "van", "von", "da", "di",
})  # fmt: skip


def normalize_name(raw: str) -> str | None:
    """Format a name without changing its spelling, or None if it is not a name.

    Spelling is the caller's to confirm, never ours to fix: Ximena and Jimena
    are both correct and different. Only case and spacing are normalized.
    """
    if not raw.strip() or re.search(r"[\d_]|[^\w\s'.-]", raw):
        return None
    words = raw.lower().split()
    return " ".join(w if k and w in PARTICLES else w.title() for k, w in enumerate(words))


def _phonetic(word: str) -> str:
    word = word.replace("ch", "§").replace("ll", "y").replace("qu", "k")
    word = re.sub(r"gu(?=[ei])", "G", word)
    word = re.sub(r"c(?=[ei])", "s", word).replace("c", "k").replace("z", "s")
    word = re.sub(r"g(?=[ei])", "j", word)
    if word.startswith("x"):
        word = "j" + word[1:]  # Ximena, Xavier; internal x reads as ks
    word = word.replace("x", "ks").replace("h", "").replace("v", "b").replace("w", "u")
    word = re.sub(r"y(?=[^aeiou]|$)", "i", word)  # Ybarra, Rey
    word = re.sub(r"(.)\1+", r"\1", word)  # Ibarra/Ibara, Anna/Ana
    return word.replace("§", "ch").replace("G", "g")


def name_key(name: str) -> str:
    """Sound-alike key: equal keys mean the agent must confirm spelling letter by letter.

    Maps the es-MX homophones that ASR cannot tell apart (b/v, s/c/z, j/g/x,
    silent h, ll/y, qu/k). Used for fuzzy scoring and to decide when to ask for
    spelling; never to rewrite what the caller said.
    """
    return " ".join(_phonetic(w) for w in re.findall(r"[a-zñ]+", _fold(name)))
