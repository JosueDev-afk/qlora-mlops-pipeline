"""ASR noise that keeps labels true.

Hand-built cases pin each relabel rule; invariants over real generated
variants pin that no label is ever a guess; the chain generate -> augment ->
build_gold -> quality gate pins that the stages fit together.
"""

import json
import random
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.normalizers import EMAIL_RE
from src.pipeline.augment import run as augment_run
from src.pipeline.augment.noise import Noised, apply_noise, load_confusions, measured_wer, tokenize
from src.pipeline.augment.run import AugmentError, augment
from src.pipeline.augment.variants import make_variant, relabel
from src.pipeline.datasets import build_gold, quality_gate
from src.pipeline.datasets.examples import invalid_reason
from src.pipeline.synth import run as synth_run

CONFUSIONS = load_confusions()
PHONE_SAID = "ochenta y uno, ochenta y dos, treinta y cuatro, sesenta y cinco, setenta y ocho"


def phone_example(style: str = "pairs", label: str | None = "+528182346578") -> dict[str, Any]:
    text = f"mi número es {PHONE_SAID}"
    start = text.index(PHONE_SAID)
    return {
        "schema_version": 1, "id": "syn-x-000001", "group": "syn-x-000001",
        "task": "extract_entity", "source": "template",
        "input": {"field": "phone", "transcript": text, "asr_confidence": None},
        "output": {"field": "phone", "normalized_value": label, "raw_span": PHONE_SAID,
                   "confidence": 0.95, "needs_reprompt": label is None,
                   "reprompt_reason": None if label else "partial"},
        "meta": {"style": style, "value_span": [start, start + len(PHONE_SAID)]},
    }  # fmt: skip


def edited(example: dict[str, Any], **edits: tuple[str, str]) -> tuple[dict[str, Any], Noised]:
    """Apply hand-picked edits: word -> (new text, kind)."""
    tokens = tokenize(example["input"]["transcript"], example["meta"]["value_span"])
    for token in tokens:
        core = token.text.strip(",")
        if core in edits:
            new, kind = edits.pop(core)
            token.original, token.text, token.kind = token.text, token.text.replace(core, new), kind  # type: ignore[assignment]
    variant = json.loads(json.dumps(example))
    noised = Noised(tokens)
    variant["input"]["transcript"] = noised.text()
    return variant, noised


# ── relabel rules ──────────────────────────────────────────────────────────


def test_homophones_in_a_phone_keep_the_true_value() -> None:
    variant, noised = edited(
        phone_example(), cinco=("sinco", "respelled"), ocho=("ocho", "homophone")
    )
    assert relabel(variant, phone_example(), noised) == "kept"
    assert variant["output"]["normalized_value"] == "+528182346578"


def test_a_misheard_number_becomes_what_was_heard() -> None:
    variant, noised = edited(phone_example(), sesenta=("setenta", "mishearing"))
    assert relabel(variant, phone_example(), noised) == "heard"
    assert variant["output"]["normalized_value"] == "+528182347578"


def test_a_mishearing_that_breaks_the_number_asks_again() -> None:
    example = phone_example()
    example["input"]["transcript"] = example["input"]["transcript"].replace(
        "ochenta y dos", "ochenta dos"
    )
    example["output"]["raw_span"] = example["output"]["raw_span"].replace(
        "ochenta y dos", "ochenta dos"
    )
    example["meta"]["value_span"][1] -= 2
    variant, noised = edited(example, dos=("doce", "mishearing"))  # 11 digits
    assert relabel(variant, example, noised) == "unresolved"
    assert variant["output"]["normalized_value"] is None
    assert variant["output"]["reprompt_reason"] == "low_confidence"


def test_a_correction_hit_by_a_mishearing_asks_again() -> None:
    variant, noised = edited(phone_example(style="correction"), sesenta=("setenta", "mishearing"))
    assert relabel(variant, phone_example(style="correction"), noised) == "unresolved"


def test_a_value_less_example_never_gains_a_value() -> None:
    partial = phone_example(label=None)
    variant, noised = edited(partial, sesenta=("setenta", "mishearing"))
    # The transcript still reads as a full number, so no label is right: dropped.
    assert relabel(variant, partial, noised) == "dropped"


def test_names_and_emails_ask_again_when_a_spelling_changes() -> None:
    name = {**phone_example(), "input": {"field": "name", "transcript": "valeria ibarra",
            "asr_confidence": None}, "meta": {"style": "said", "value_span": [0, 14]}}  # fmt: skip
    name["output"] = {**name["output"], "field": "name", "normalized_value": "Valeria Ibarra",
                      "raw_span": "valeria ibarra"}  # fmt: skip
    variant, noised = edited(name, valeria=("baleria", "respelled"))
    assert relabel(variant, name, noised) == "unresolved"
    variant, noised = edited(name, valeria=("valeria eh", "filler"))
    assert relabel(variant, name, noised) == "kept"


# ── the noise itself ───────────────────────────────────────────────────────


def test_applied_wer_follows_the_target_on_average() -> None:
    text = " ".join([PHONE_SAID] * 4)
    for wer in (0.05, 0.15, 0.25):
        applied = []
        for seed in range(200):
            tokens = tokenize(text, None)
            noised = apply_noise(tokens, wer, CONFUSIONS, random.Random(seed))
            applied.append(measured_wer(noised, len(tokens)))
        assert sum(applied) / len(applied) == pytest.approx(wer, abs=0.02)


def test_protected_words_are_never_touched() -> None:
    text = "no, ahorita no puedo, ya voy manejando, márqueme después, sí"
    for seed in range(300):
        noised = apply_noise(tokenize(text, None), 0.5, CONFUSIONS, random.Random(seed))
        shown = [t.text.strip(",") for t in noised.tokens if t.kind != "deleted"]
        for word in ("no", "ahorita", "puedo", "ya", "voy", "después", "sí"):
            assert shown.count(word) == text.replace(",", "").split().count(word), (seed, shown)


# ── invariants over real generated examples ────────────────────────────────


@pytest.fixture(scope="module")
def variants(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[dict, dict]]:
    root = tmp_path_factory.mktemp("synth")
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["synth"]["n_dialogues"] = 1500
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(params, allow_unicode=True))
    out = Path(synth_run.run(str(path), out=root / "synthetic"))
    examples = [
        json.loads(line) for line in (out / "extract_entity.jsonl").read_text().splitlines()
    ]
    pairs = []
    for example in examples:
        for k, wer in enumerate((0.0, 0.1, 0.25, 0.25)):
            variant, _ = make_variant(example, k, wer, CONFUSIONS, seed=42, emphasis=True)
            if variant:
                pairs.append((example, variant))
    return pairs


def test_every_variant_is_a_valid_example(variants: list[tuple[dict, dict]]) -> None:
    assert not [v["id"] for _, v in variants if invalid_reason(v)]


def test_labels_are_the_truth_what_was_heard_or_ask_again(
    variants: list[tuple[dict, dict]],
) -> None:
    for original, variant in variants:
        truth, label = original["output"]["normalized_value"], variant["output"]["normalized_value"]
        field = variant["input"]["field"]
        if truth is None:
            assert label is None  # never gains a value
        elif field == "name":
            assert label in (truth, None)  # names are never "heard" differently
        elif label not in (truth, None):
            assert (
                (label.startswith("+52") and len(label) == 13)
                if field == "phone"
                else EMAIL_RE.match(label)
            )
        if variant["meta"]["value_span"]:
            a, b = variant["meta"]["value_span"]
            assert variant["input"]["transcript"][a:b] == variant["output"]["raw_span"]


def test_the_clean_variant_keeps_everything(variants: list[tuple[dict, dict]]) -> None:
    for original, variant in variants:
        if variant["id"].endswith("-v00"):
            assert variant["output"] == original["output"]
            assert variant["input"]["transcript"] == original["input"]["transcript"]
            assert 0.9 <= variant["input"]["asr_confidence"] <= 0.99


def test_variants_share_the_group_and_are_reproducible() -> None:
    example = phone_example()
    first, _ = make_variant(example, 3, 0.25, CONFUSIONS, seed=1, emphasis=True)
    again, _ = make_variant(example, 3, 0.25, CONFUSIONS, seed=1, emphasis=True)
    assert first == again and first is not None
    assert first["group"] == example["group"] and first["id"] == "syn-x-000001-v03"


# ── the stage ──────────────────────────────────────────────────────────────


def _params(root: Path, **sections: dict) -> str:
    params = yaml.safe_load(Path("params.yaml").read_text())
    for name, values in sections.items():
        params[name].update(values)
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(params, allow_unicode=True))
    return str(path)


def _replay(root: Path, n: int, task: str = "replay") -> Path:
    replay = root / "silver" / "replay"
    replay.mkdir(parents=True)
    rows = [{"schema_version": 1, "id": f"rep-{i}", "group": f"rep-{i}", "task": task,
             "source": "replay", "input": {"instruction": f"pregunta {i}"},
             "output": {"response": "respuesta"}} for i in range(n)]  # fmt: skip
    (replay / "part-0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return replay


def test_generate_augment_gold_and_gate_fit_together(tmp_path: Path) -> None:
    params = _params(tmp_path, synth={"n_dialogues": 900}, augment={"variants_per_dialogue": 4},
                     gold={"target_size": 1500, "template_ratio": 1.0},
                     calibration={"split": str(tmp_path / "calibration")})  # fmt: skip
    synthetic = Path(synth_run.run(params, out=tmp_path / "synthetic"))
    augmented = Path(augment_run.run(params, synthetic=synthetic, replay=_replay(tmp_path, 400),
                                     out=tmp_path / "augmented"))  # fmt: skip
    first = {p.name: p.read_bytes() for p in augmented.iterdir()}
    augment_run.run(params, synthetic=synthetic, replay=tmp_path / "silver" / "replay",
                    out=tmp_path / "augmented")  # fmt: skip
    assert {p.name: p.read_bytes() for p in augmented.iterdir()} == first  # deterministic
    gold = build_gold.run(params, inputs=augmented, out=tmp_path / "gold")
    report = tmp_path / "gold_quality.json"
    quality_gate.assert_expectations(gold, params, eval_set=tmp_path / "eval", report=report)
    assert json.loads(report.read_text())["passed"]


def test_replay_passes_through_and_must_be_replay(tmp_path: Path) -> None:
    params = _params(tmp_path, synth={"n_dialogues": 30}, augment={"variants_per_dialogue": 2})
    synthetic = Path(synth_run.run(params, out=tmp_path / "synthetic"))
    bad = _replay(tmp_path, 3, task="classify_intent")
    with pytest.raises(AugmentError):
        augment_run.run(params, synthetic=synthetic, replay=bad, out=tmp_path / "augmented")


def test_only_rules_are_implemented(tmp_path: Path) -> None:
    params = _params(tmp_path, augment={"method": "llm"})
    with pytest.raises(AugmentError, match="only 'rules'"):
        augment_run.run(params, synthetic=tmp_path, out=tmp_path / "augmented")


def test_augment_reports_what_relabelling_did() -> None:
    parts, stats = augment([phone_example()], wer_levels=[0.25], variants=6, seed=3, emphasis=True)
    assert sum(stats["labels"]["extract_entity"].values()) == 6
    assert len(parts["extract_entity"]) + stats["labels"]["extract_entity"].get("dropped", 0) == 6
