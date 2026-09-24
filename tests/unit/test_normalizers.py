"""Spoken es-MX -> canonical values. The hard part of the project.

SEED CASES. Plan task 4 belongs to the maintainer: extend these with real
Scribe v2 transcripts (notebook 02) and any phrasing a native speaker hears on
real calls. Normalizers never guess: when a transcript is ambiguous they return
None, and the agent asks again.
"""

import pytest

from src.common.normalizers import name_key, normalize_email, normalize_name, spoken_to_e164

# ── phone → E.164 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("ocho uno ocho dos tres cuatro cinco seis siete ocho", "+528182345678"),
        (
            "ochenta y uno, ochenta y dos, treinta y cuatro, cincuenta y seis, setenta y ocho",
            "+528182345678",
        ),
        ("81 82 34 56 78", "+528182345678"),
        ("8182345678", "+528182345678"),
        ("ocho uno, 82, treinta y cuatro, cinco seis, 78", "+528182345678"),
        ("es el ocho uno ocho dos tres cuatro cinco seis siete ocho", "+528182345678"),
        (
            "mi celular es cincuenta y cinco, doce, treinta y cuatro, cero cinco, noventa",
            "+525512340590",
        ),
        ("ocho uno doble ocho tres cuatro cinco seis siete ocho", "+528188345678"),
        (
            "más cincuenta y dos ochenta y uno ochenta y dos treinta y cuatro "
            "cincuenta y seis setenta y ocho",
            "+528182345678",
        ),
        ("52 1 81 82 34 56 78", "+528182345678"),  # legacy mobile prefix 1
        (
            "cero cuatro cuatro ocho uno ocho dos tres cuatro cinco seis siete ocho",
            "+528182345678",
        ),  # legacy 044
        (
            "ochocientos dieciocho, doscientos treinta y cuatro, cincuenta y seis, setenta y ocho",
            "+528182345678",
        ),
        ("treinta y tres dieciséis veintitrés cuarenta y cinco sesenta y siete", "+523316234567"),
    ],
)
def test_phone_transcripts_normalize_to_e164(transcript: str, expected: str) -> None:
    assert spoken_to_e164(transcript) == expected


@pytest.mark.parametrize(
    "transcript",
    [
        "ocho uno ocho dos tres cuatro cinco seis siete",  # 9 digits
        "ocho uno ocho dos tres cuatro cinco seis siete ocho nueve",  # 11 digits
        # a correction mid-dictation: which group it replaces is unknowable
        "ocho uno ocho dos, no, perdón, ocho uno ocho tres cuatro cinco seis siete ocho nueve",
        "ocho uno oso dos tres cuatro cinco seis siete ocho",  # ASR garbage mid-number
        # national numbers never start with 0 or 1
        "cero uno ocho dos tres cuatro cinco seis siete ocho",
        "no me acuerdo",
        "",
    ],
)
def test_ambiguous_phone_transcripts_return_none(transcript: str) -> None:
    assert spoken_to_e164(transcript) is None


# ── spelled email → address ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        (
            "jota u a ene punto pe e ere e zeta arroba ge eme a i ele punto com",
            "juan.perez@gmail.com",
        ),
        ("juan punto perez arroba gmail punto com", "juan.perez@gmail.com"),
        ("juan pérez todo junto arroba hotmail punto com", "juanperez@hotmail.com"),
        ("mi correo es juan guion bajo perez arroba outlook punto com", "juan_perez@outlook.com"),
        ("ana guion medio lopez arroba empresa punto com punto mx", "ana-lopez@empresa.com.mx"),
        ("juan noventa y ocho arroba gmail punto com", "juan98@gmail.com"),
        ("juan nueve ocho arroba gmail punto com", "juan98@gmail.com"),
        ("ele u ce i a doble ve arroba yahoo punto com", "luciaw@yahoo.com"),
        ("be de burro e te o arroba gmail punto com", "beto@gmail.com"),
        ("ve chica i ce te o ere arroba gmail punto com", "victor@gmail.com"),
        ("i griega o ele a arroba gmail punto com", "yola@gmail.com"),
        ("juan de la rosa arroba gmail punto com", "juandelarosa@gmail.com"),
        ("JUAN.PEREZ@GMAIL.COM", "juan.perez@gmail.com"),
    ],
)
def test_spelled_email_transcripts_normalize(transcript: str, expected: str) -> None:
    assert normalize_email(transcript) == expected


@pytest.mark.parametrize(
    "transcript",
    [
        "juan punto perez arroba gmail",  # no TLD: never complete it by guessing
        "juan punto perez gmail punto com",  # no arroba
        "juan arroba perez arroba gmail punto com",  # two arrobas
        "punto juan arroba gmail punto com",  # leading dot
        "juan punto punto perez arroba gmail punto com",  # consecutive dots
        "jota u a eñe o arroba gmail punto com",  # ñ is not valid in an address
        "juan punto de punto perez arroba gmail punto com",  # letter d or the word "de"?
        "",
    ],
)
def test_malformed_emails_return_none(transcript: str) -> None:
    assert normalize_email(transcript) is None


# ── names ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  maría   de la luz  ", "María de la Luz"),
        ("JOSÉ LUIS GARCÍA-LÓPEZ", "José Luis García-López"),
        ("de la rosa ibarra", "De la Rosa Ibarra"),
        ("o'connor", "O'Connor"),
    ],
)
def test_names_are_formatted_without_changing_their_spelling(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "juan 23", "juan@perez"])
def test_non_names_return_none(raw: str) -> None:
    assert normalize_name(raw) is None


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Ximena", "Jimena"),
        ("Ibarra", "Ybarra"),
        ("Valeria", "Baleria"),
        ("Gisela", "Jisela"),
        ("Cecilia", "Sesilia"),
        ("Hernández", "Ernandez"),
        ("Yolanda", "Llolanda"),
        ("Enrique", "Enrike"),
    ],
)
def test_homophone_spellings_share_a_key(a: str, b: str) -> None:
    """Same key means the agent must confirm the spelling, never pick one."""
    assert name_key(a) == name_key(b)


@pytest.mark.parametrize(("a", "b"), [("Juan", "Juana"), ("Luis", "Luisa"), ("Mario", "María")])
def test_different_names_keep_different_keys(a: str, b: str) -> None:
    assert name_key(a) != name_key(b)
