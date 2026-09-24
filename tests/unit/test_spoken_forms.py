"""Spoken forms are the inverse of the normalizers: labels correct by construction.

The round-trip property is the contract: every readable style must normalize
back to exactly the value it was generated from, for many random seeds. A
style the normalizer cannot read on purpose (corrections) must read as None,
never as a different number.
"""

import random

import pytest

from src.common.normalizers import normalize_email, spoken_to_e164
from src.pipeline.synth.spoken_forms import (
    EMAIL_STYLES,
    PHONE_STYLES,
    random_phone,
    speak_email,
    speak_phone,
)

SEEDS = range(300)
READABLE_PHONE_STYLES = [s for s in PHONE_STYLES if s != "correction"]

# Realistic addresses. All-ambiguous spelled chunks ("ed") and lone d/t/s are
# genuinely ambiguous when spelled, so the normalizer refuses them by design.
ADDRESSES = [
    "juan.perez@gmail.com",
    "maria_lopez98@hotmail.com",
    "ana-garcia@empresa.com.mx",
    "luciaw@yahoo.com",
    "victor.b@outlook.com",
    "jose2024@icloud.com",
    "x.ramirez07@live.com.mx",
    "rocio.ibarra@prodigy.net.mx",
    "beto.v@gmail.com",
    "yolanda.zamora@gmail.com",
    "carlos_88@gmail.com",
    "eduardo.quiroz100@gmail.com",
]


@pytest.mark.parametrize("style", READABLE_PHONE_STYLES)
def test_phone_round_trips_for_every_readable_style(style: str) -> None:
    for seed in SEEDS:
        rng = random.Random(seed)
        phone = random_phone(rng)
        spoken = speak_phone(phone, style, rng)
        assert spoken_to_e164(spoken) == phone, (seed, spoken)


@pytest.mark.parametrize(
    "phone",
    [
        "+528200345678",  # "doscientos" alone would swallow the next group
        "+525500000000",  # zeros everywhere
        "+523310010005",  # ciento / cien / leading zeros inside groups
        "+529999999999",  # repeated digits: doble / triple
    ],
)
@pytest.mark.parametrize("style", READABLE_PHONE_STYLES)
def test_phone_edge_cases_round_trip(phone: str, style: str) -> None:
    for seed in range(50):
        spoken = speak_phone(phone, style, random.Random(seed))
        assert spoken_to_e164(spoken) == phone, (seed, spoken)


def test_correction_style_is_never_read_as_a_wrong_number() -> None:
    for seed in SEEDS:
        rng = random.Random(seed)
        spoken = speak_phone(random_phone(rng), "correction", rng)
        assert "perdón" in spoken
        assert spoken_to_e164(spoken) is None, spoken


@pytest.mark.parametrize("address", ADDRESSES)
@pytest.mark.parametrize("style", EMAIL_STYLES)
def test_email_round_trips_for_every_style(address: str, style: str) -> None:
    for seed in range(100):
        spoken = speak_email(address, style, random.Random(seed))
        assert normalize_email(spoken) == address, (seed, spoken)


def test_spelled_style_spells_the_local_part_and_says_the_domain() -> None:
    spoken = speak_email("juan.perez@gmail.com", "spelled", random.Random(0))
    assert spoken.split(" arroba ")[1] == "gmail punto com"
    assert "jota" in spoken and "juan" not in spoken


def test_same_seed_gives_the_same_transcript() -> None:
    """Reproducibility: synth.seed must regenerate the exact corpus."""
    first = speak_phone("+528182345678", "mixed", random.Random(7))
    again = speak_phone("+528182345678", "mixed", random.Random(7))
    assert first == again


def test_random_phone_is_a_valid_mexican_number() -> None:
    for seed in SEEDS:
        phone = random_phone(random.Random(seed))
        assert phone.startswith("+52") and len(phone) == 13 and phone[3] not in "01"


@pytest.mark.parametrize("phone", ["8182345678", "+520812345678", "+52818234567"])
def test_non_canonical_phones_are_rejected(phone: str) -> None:
    with pytest.raises(ValueError, match="phone"):
        speak_phone(phone, "pairs", random.Random(0))


@pytest.mark.parametrize("address", ["Juan@gmail.com", "juan+tag@gmail.com", "juan@gmail"])
def test_non_canonical_emails_are_rejected(address: str) -> None:
    with pytest.raises(ValueError, match="address"):
        speak_email(address, "words", random.Random(0))


def test_unknown_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="style"):
        speak_phone("+528182345678", "whispered", random.Random(0))
