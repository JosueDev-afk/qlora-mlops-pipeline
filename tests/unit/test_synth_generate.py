"""Code-generated examples: valid, reproducible, and labelled by construction.

The central property: a label is never a guess. Where a rule-based normalizer
exists, it reads the spoken value back to exactly the label; where the value
cannot be known from speech (homophone names, partial numbers), the label is
null and asks again.
"""

import json
import random
from pathlib import Path

import pytest
import yaml

from src.common.normalizers import normalize_email, normalize_name, spoken_to_e164
from src.pipeline.datasets import build_gold, quality_gate
from src.pipeline.datasets.examples import invalid_reason
from src.pipeline.synth import run as synth_run
from src.pipeline.synth.generate import Generator, TemplateError, load_templates, spell
from src.pipeline.synth.run import capped, generate, per_task

PARAMS = yaml.safe_load(Path("params.yaml").read_text())["synth"]
PERSONAS = PARAMS["personas"]


def templates():
    return load_templates(PERSONAS, PARAMS["backchannels"], PARAMS["real_interruptions"])


def make(seed: int, n: int) -> dict[str, list[dict]]:
    g = Generator(templates(), PERSONAS, seed=seed)
    return generate(g, capped(g, per_task(n)))


@pytest.fixture(scope="module")
def corpus() -> dict[str, list[dict]]:
    return make(42, 3000)


def test_every_example_meets_the_gold_contract(corpus: dict[str, list[dict]]) -> None:
    rows = [r for part in corpus.values() for r in part]
    assert len(corpus["extract_entity"]) == len(corpus["classify_intent"]) == 1000
    assert not [r["id"] for r in rows if invalid_reason(r)]
    assert len({r["id"] for r in rows}) == len(rows)


def test_same_seed_same_corpus_and_growing_moves_nothing() -> None:
    small = make(7, 30)
    large = make(7, 90)
    for task in small:
        assert small[task] == large[task][: len(small[task])]
    other = make(8, 30)
    assert other != small


# ── Task A: labels by construction ─────────────────────────────────────────


def values(corpus: dict[str, list[dict]], field: str) -> list[dict]:
    return [r for r in corpus["extract_entity"] if r["input"]["field"] == field]


def test_value_span_points_at_the_spoken_value(corpus: dict[str, list[dict]]) -> None:
    for r in corpus["extract_entity"]:
        span = r["meta"]["value_span"]
        if span:
            assert r["input"]["transcript"][span[0] : span[1]] == r["output"]["raw_span"]


def test_phones_read_back_to_their_label(corpus: dict[str, list[dict]]) -> None:
    for r in values(corpus, "phone"):
        heard = spoken_to_e164(r["output"]["raw_span"] or "")
        if r["meta"]["style"] == "correction":
            assert heard is None  # the rule baseline refuses; the model learns the final number
            assert r["output"]["normalized_value"].startswith("+52")
        elif r["output"]["normalized_value"]:
            assert heard == r["output"]["normalized_value"], r["output"]["raw_span"]


def test_emails_read_back_to_their_label(corpus: dict[str, list[dict]]) -> None:
    for r in values(corpus, "email"):
        if r["output"]["normalized_value"]:
            assert normalize_email(r["output"]["raw_span"]) == r["output"]["normalized_value"]


def test_names_keep_their_spelling_or_ask(corpus: dict[str, list[dict]]) -> None:
    ambiguous = templates().ambiguous_tokens
    for r in values(corpus, "name"):
        if r["meta"]["style"] not in ("said", "said_spelled"):
            continue  # negatives: covered below
        out = r["output"]
        tokens = (out["normalized_value"] or normalize_name(out["raw_span"]) or "").split()
        if r["meta"]["style"] == "said" and any(t in ambiguous for t in tokens):
            assert out["normalized_value"] is None and out["reprompt_reason"] == "low_confidence"
        elif out["normalized_value"]:
            assert normalize_name(out["raw_span"].split(",")[0]) == out["normalized_value"]


def test_homophones_said_aloud_are_never_guessed() -> None:
    t = templates()
    assert {"Ximena", "Jimena", "Ibarra", "Ybarra"} <= t.ambiguous_tokens
    corpus = make(3, 6000)["extract_entity"]
    homophones = {"ximena", "jimena", "ibarra", "ybarra"}
    said = [
        r
        for r in corpus
        if r["meta"]["style"] == "said" and homophones & set(r["output"]["raw_span"].split())
    ]
    spelled = [r for r in corpus if r["meta"]["style"] == "said_spelled"]
    assert said and spelled
    assert all(r["output"]["normalized_value"] is None for r in said)
    assert all(r["output"]["normalized_value"] for r in spelled)


def test_negatives_never_carry_a_value(corpus: dict[str, list[dict]]) -> None:
    kinds = ("not_provided", "partial", "invalid_format")
    negatives = [r for r in corpus["extract_entity"] if r["meta"]["style"] in kinds]
    assert {r["meta"]["style"] for r in negatives} == {"not_provided", "partial", "invalid_format"}
    for r in negatives:
        assert r["output"]["normalized_value"] is None and r["output"]["needs_reprompt"]
        assert r["output"]["reprompt_reason"] == r["meta"]["style"]
        if r["meta"]["style"] == "partial" and r["input"]["field"] == "phone":
            assert spoken_to_e164(r["output"]["raw_span"]) is None
        if r["meta"]["style"] == "invalid_format":
            assert normalize_email(r["output"]["raw_span"]) is None


def test_spelling_names_letters_accents_and_enie() -> None:
    assert spell("Ximena") == "equis, i, eme, e, ene, a"
    assert spell("Ibáñez") == "i, be, a con acento, eñe, e, zeta"


# ── Tasks B and D ──────────────────────────────────────────────────────────


def test_intents_fit_their_flow(corpus: dict[str, list[dict]]) -> None:
    rows = corpus["classify_intent"]
    assert {r["output"]["intent"] for r in rows} == {
        "confirmed", "cannot_attend", "call_rejected", "ambiguous", "out_of_scope"}  # fmt: skip
    # cannot_attend is about an appointment: it never answers a contact question.
    assert not [r for r in rows if r["input"]["flow"] == "validate_contact"
                and r["output"]["intent"] == "cannot_attend"]  # fmt: skip


def test_interruptions_agree_with_their_kind(corpus: dict[str, list[dict]]) -> None:
    rows = corpus["is_real_interruption"]
    for r in rows:
        out = r["output"]
        assert out["interruption"] is (out["kind"] != "backchannel")
        words = r["input"]["agent_utterance"].split()
        assert 1 <= r["meta"]["position"] < len(words)
        assert r["input"]["agent_said"] == " ".join(words[: r["meta"]["position"]])
    assert {r["input"]["transcript"] for r in rows if not r["output"]["interruption"]} <= set(
        PARAMS["backchannels"]
    )


# ── templates and the stage ────────────────────────────────────────────────


def test_templates_must_cover_the_params() -> None:
    with pytest.raises(TemplateError, match="personas without templates: \\['whispering'\\]"):
        load_templates(
            [*PERSONAS, "whispering"], PARAMS["backchannels"], PARAMS["real_interruptions"]
        )
    with pytest.raises(TemplateError, match="without a kind"):
        load_templates(PERSONAS, PARAMS["backchannels"], [*PARAMS["real_interruptions"], "¡alto!"])


def _params(tmp_path: Path, **overrides: dict) -> str:
    params = yaml.safe_load(Path("params.yaml").read_text())
    for section, values in overrides.items():
        params[section].update(values)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params, allow_unicode=True))
    return str(path)


def test_run_is_deterministic_on_disk(tmp_path: Path) -> None:
    params = _params(tmp_path, synth={"n_dialogues": 300})
    out = Path(synth_run.run(params, out=tmp_path / "synthetic"))
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    synth_run.run(params, out=tmp_path / "synthetic")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first
    manifest = json.loads((out / "manifest.json").read_text())
    assert sum(t["rows"] for t in manifest["counts"].values()) == 300


def test_generated_corpus_passes_the_gold_gate(tmp_path: Path) -> None:
    """generate -> (augment, still missing) -> build_gold -> quality_gate, with template only."""
    params = _params(tmp_path, synth={"n_dialogues": 3000},
                     gold={"target_size": 1200, "template_ratio": 1.0},
                     calibration={"split": str(tmp_path / "calibration")})  # fmt: skip
    synthetic = Path(synth_run.run(params, out=tmp_path / "augmented"))
    replay = [{"schema_version": 1, "id": f"rep-{i}", "group": f"rep-{i}", "task": "replay",
               "source": "replay", "input": {"instruction": f"pregunta {i}"},
               "output": {"response": "respuesta"}} for i in range(300)]  # fmt: skip
    (synthetic / "replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replay))
    gold = build_gold.run(params, inputs=synthetic, out=tmp_path / "gold")
    report = tmp_path / "gold_quality.json"
    quality_gate.assert_expectations(gold, params, eval_set=tmp_path / "eval", report=report)
    assert json.loads(report.read_text())["passed"]


def test_finite_template_spaces_are_emitted_once_and_capped(tmp_path: Path) -> None:
    g = Generator(templates(), PERSONAS, seed=42)
    space = g.capacity("is_real_interruption")
    assert space and g.capacity("extract_entity") is None
    rows = [g.generate("is_real_interruption", i) for i in range(space)]
    assert len({json.dumps([r["input"], r["output"]], sort_keys=True) for r in rows}) == space
    params = _params(tmp_path, synth={"n_dialogues": 3 * (space + 100)})
    out = Path(synth_run.run(params, out=tmp_path / "synthetic"))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["is_real_interruption"]["rows"] == space  # capped, not padded
    assert manifest["requested"]["is_real_interruption"] == space + 100


def test_rng_is_per_example() -> None:
    """Two generators on the same seed agree example by example, in any order."""
    g = Generator(templates(), PERSONAS, seed=1)
    forward = [g.generate("classify_intent", i) for i in range(5)]
    backward = [g.generate("classify_intent", i) for i in reversed(range(5))]
    assert forward == list(reversed(backward))
    assert random.Random(0).random()  # the module never touches the global RNG state
