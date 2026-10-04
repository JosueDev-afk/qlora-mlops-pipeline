"""The carrier-phrase bank: planned evenly, filtered by code, approved by a person.

No model and no network: the completion function is a fake, and review reads
scripted keystrokes. What is pinned is that nothing reaches the corpus
without passing the filter and a human decision.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.common.prompts import load_prompt
from src.pipeline.synth.carrier import (
    approved,
    generate_bank,
    load_bank,
    messages,
    parse,
    plan,
    reject_reason,
    review,
)

PARAMS = yaml.safe_load(Path("params.yaml").read_text())["synth"]
SETTINGS = PARAMS["carrier_phrases"]
PROMPT = load_prompt("carrier_phrases", 1)


def reply(*phrases: str) -> str:
    return json.dumps({"phrases": list(phrases)}, ensure_ascii=False)


# ── plan and prompts ───────────────────────────────────────────────────────


def test_plan_overgenerates_every_label_evenly() -> None:
    requests = plan(SETTINGS, PARAMS["personas"])
    asked = sum(r.n for r in requests)
    assert asked >= SETTINGS["target_count"] * SETTINGS["overgenerate"]
    per_label = Counter((r.task, r.label) for r in requests)
    assert set(per_label) >= {("extract_entity", "phone"), ("classify_intent", "call_rejected"),
                              ("is_real_interruption", "backchannel")}  # fmt: skip
    for task in SETTINGS["shares"]:
        assert len({n for (t, _), n in per_label.items() if t == task}) == 1  # even within a task
    assert {r.persona for r in requests} == set(PARAMS["personas"])


def test_task_a_prompts_ask_for_the_placeholder_never_the_value() -> None:
    request = next(r for r in plan(SETTINGS, PARAMS["personas"]) if r.label == "email")
    user = messages(request, PROMPT)[1]["content"]
    assert "{value}" in user and "correo electrónico" in user


def test_intent_prompts_use_the_laya_label_definitions() -> None:
    request = next(r for r in plan(SETTINGS, PARAMS["personas"]) if r.label == "call_rejected")
    user = messages(request, PROMPT)[1]["content"]
    assert "voy manejando" in user  # from classify_intent_laya/v1, not a copy
    assert request.context and request.context in user


# ── filter ─────────────────────────────────────────────────────────────────


def test_parse_tolerates_fences_and_refuses_garbage() -> None:
    assert parse(reply("hola")) == ["hola"]
    assert parse("```json\n" + reply("a", "b") + "\n```") == ["a", "b"]
    assert parse("lo siento, no puedo") == []
    assert parse('{"phrases": "no es lista"}') == []


@pytest.mark.parametrize(
    ("task", "text", "reason"),
    [
        ("extract_entity", "claro, es el {value}", None),
        ("extract_entity", "claro, es el 8182345678", "{value}"),
        ("extract_entity", "{value} y también {value}", "{value}"),
        ("classify_intent", "ahorita no puedo, voy manejando", None),
        ("classify_intent", "mándeme un correo a juan@gmail.com", "digit, email"),
        ("classify_intent", "llámeme a las 5", "digit"),
        ("is_real_interruption", "no, espérese", None),
        (
            "is_real_interruption",
            "no espérese tantito porque ese no es mi correo para nada",
            "words",
        ),
        ("is_real_interruption", "ajá {value}", "unexpected"),
    ],
)
def test_the_filter(task: str, text: str, reason: str | None) -> None:
    result = reject_reason(task, text)
    assert result is None if reason is None else reason in result


# ── generation ─────────────────────────────────────────────────────────────


def test_generation_appends_new_pending_candidates_only(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    requests = [r for r in plan(SETTINGS, PARAMS["personas"]) if r.label == "phone"][:2]
    answers = iter([
        reply("es el {value}", "ahí le va: {value}", "es el {value}", "8181818181"),
        reply("ES EL {value}", "mire, {value}"),  # case-insensitive duplicate
    ])  # fmt: skip
    summary = generate_bank(bank, requests, lambda chat, seed: next(answers), model="Qwen/Qwen3-8B")
    entries = load_bank(bank)
    assert [e["text"] for e in entries] == ["es el {value}", "ahí le va: {value}", "mire, {value}"]
    assert all(e["status"] == "pending" and e["reviewed_by"] is None for e in entries)
    assert entries[0]["prompt"]["sha256"] == PROMPT.sha256
    assert summary["rejected"] == {"duplicate": 2, "needs exactly one {value}": 1}
    again = generate_bank(bank, requests[:1], lambda chat, seed: reply("es el {value}"), model="m")
    assert again["added"] == 0 and len(load_bank(bank)) == 3  # reruns only add


def test_a_failed_request_does_not_lose_the_others(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    requests = [r for r in plan(SETTINGS, PARAMS["personas"]) if r.label == "name"][:2]
    calls = iter([OSError("connection reset"), reply("me llamo {value}")])

    def complete(chat: list, seed: int) -> str:
        step = next(calls)
        if isinstance(step, Exception):
            raise step
        return step

    summary = generate_bank(bank, requests, complete, model="m")
    assert summary["rejected"] == {"request_failed": 1} and summary["added"] == 1


# ── review ─────────────────────────────────────────────────────────────────


def seeded_bank(tmp_path: Path) -> Path:
    bank = tmp_path / "bank.jsonl"
    requests = [r for r in plan(SETTINGS, PARAMS["personas"]) if r.label == "phone"][:1]
    reply_text = reply("es el {value}", "a ver, {value}", "pues {value}", "mire {value}")
    generate_bank(bank, requests, lambda chat, seed: reply_text, model="m")
    return bank


def test_review_records_who_decided_and_only_approved_phrases_count(tmp_path: Path) -> None:
    bank = seeded_bank(tmp_path)
    keys = iter(["a", "r", "e", "sin valor", "e", "pues, sería {value}", "s"])
    done = review(bank, reviewer="Josué", ask=lambda _: next(keys), say=lambda _: None)
    assert done == Counter({"approved": 2, "rejected": 1})
    entries = {e["text"]: e for e in load_bank(bank)}
    edited = entries["pues, sería {value}"]
    assert edited["original_text"] == "pues {value}" and edited["reviewed_by"] == "Josué"
    assert entries["mire {value}"]["status"] == "pending"  # skipped stays pending
    assert sorted(e["text"] for e in approved(bank)) == ["es el {value}", "pues, sería {value}"]


def test_review_resumes_where_it_stopped(tmp_path: Path) -> None:
    bank = seeded_bank(tmp_path)
    first = iter(["a", "q"])
    review(bank, reviewer="J", ask=lambda _: next(first), say=lambda _: None)
    shown: list[str] = []
    review(bank, reviewer="J", ask=lambda _: "q", say=shown.append)
    assert any("[1/3]" in line for line in shown)  # the first decision was kept


# ── generate uses approved phrases only ────────────────────────────────────


def approved_bank(tmp_path: Path) -> Path:
    """A bank as review leaves it: some approved, one rejected, one pending."""
    bank = tmp_path / "carrier.jsonl"
    rows = [("extract_entity", "phone", "fíjese que es el {value}", "approved"),
            ("extract_entity", "email", "anótele, es {value}", "approved"),
            ("extract_entity", "name", "me llamo {value}", "approved"),
            ("classify_intent", "call_rejected", "ando en el súper, luego me marca", "approved"),
            ("classify_intent", "confirmed", "sí, todo bien", "approved"),
            ("classify_intent", "confirmed", "ándele, así quedamos", "approved"),
            ("classify_intent", "ambiguous", "pues fíjese que no sé", "approved"),
            ("classify_intent", "out_of_scope", "oiga, ¿y ustedes quiénes son?", "approved"),
            ("classify_intent", "call_rejected", "ando manejando, al rato", "approved"),
            ("classify_intent", "cannot_attend", "ese día no voy a poder llegar", "approved"),
            ("is_real_interruption", "correction", "no, así no", "approved"),
            ("is_real_interruption", "backchannel", "aja pues", "approved"),
            ("classify_intent", "confirmed", "esta frase fue rechazada", "rejected"),
            ("classify_intent", "confirmed", "esta sigue pendiente", "pending")]  # fmt: skip
    entries = [{"id": f"cp-{i:03d}", "task": task, "label": label, "text": text,
                "persona": "cooperative", "context": None, "status": status}
               for i, (task, label, text, status) in enumerate(rows)]  # fmt: skip
    bank.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    return bank


def _params(root: Path, **sections: dict) -> str:
    params = yaml.safe_load(Path("params.yaml").read_text())
    for name, values in sections.items():
        params[name].update(values)
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(params, allow_unicode=True))
    return str(path)


def test_generate_turns_approved_phrases_into_carrier_examples(tmp_path: Path) -> None:
    from src.common.normalizers import normalize_email, spoken_to_e164
    from src.pipeline.datasets.examples import invalid_reason
    from src.pipeline.synth import run as synth_run

    params = _params(tmp_path, synth={"n_dialogues": 300})
    out = Path(synth_run.run(params, out=tmp_path / "synthetic", bank=approved_bank(tmp_path)))
    rows = [
        json.loads(line) for f in sorted(out.glob("*.jsonl")) for line in f.read_text().splitlines()
    ]
    carrier = [r for r in rows if r["source"] == "carrier"]
    assert {r["task"] for r in carrier} == {
        "extract_entity",
        "classify_intent",
        "is_real_interruption",
    }
    assert not [r for r in carrier if invalid_reason(r)]
    texts = " ".join(r["input"]["transcript"] for r in carrier)
    assert "rechazada" not in texts and "pendiente" not in texts  # only approved phrases
    for r in carrier:
        if r["task"] == "extract_entity" and r["input"]["field"] == "phone":
            assert r["input"]["transcript"].startswith("fíjese que es el")
            if r["meta"]["style"] != "correction":
                assert spoken_to_e164(r["output"]["raw_span"]) == r["output"]["normalized_value"]
        if r["task"] == "extract_entity" and r["input"]["field"] == "email":
            assert normalize_email(r["output"]["raw_span"]) == r["output"]["normalized_value"]
        if r["task"] == "classify_intent" and "súper" in r["input"]["transcript"]:
            assert r["output"]["intent"] == "call_rejected"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["carrier"]["approved"] == 12


def test_the_full_mix_builds_once_carrier_and_replay_exist(tmp_path: Path) -> None:
    """generate (with a bank) -> augment -> build_gold at template_ratio 0.75 -> gate."""
    from src.pipeline.augment import run as augment_run
    from src.pipeline.datasets import build_gold, quality_gate
    from src.pipeline.synth import run as synth_run

    params = _params(tmp_path, synth={"n_dialogues": 900}, augment={"variants_per_dialogue": 4},
                     gold={"target_size": 1200, "template_ratio": 0.75},
                     calibration={"split": str(tmp_path / "calibration")})  # fmt: skip
    synthetic = Path(
        synth_run.run(params, out=tmp_path / "synthetic", bank=approved_bank(tmp_path))
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "r.jsonl").write_text("".join(json.dumps({
        "schema_version": 1, "id": f"rep-{i}", "group": f"rep-{i}", "task": "replay",
        "source": "replay", "input": {"instruction": f"pregunta {i}"},
        "output": {"response": "respuesta"}}) + "\n" for i in range(300)))  # fmt: skip
    augmented = augment_run.run(params, synthetic=synthetic, replay=replay, out=tmp_path / "aug")
    gold = build_gold.run(params, inputs=Path(augmented), out=tmp_path / "gold")
    report = tmp_path / "q.json"
    quality_gate.assert_expectations(gold, params, eval_set=tmp_path / "eval", report=report)
    assert json.loads(report.read_text())["passed"]
