# CLAUDE.md

Project context and rules for Claude Code. Read this before writing any code.

*(Written in English because it governs code. Conversation with the maintainer is in Spanish.)*

---

## What this project is

`qlora-mlops-pipeline` builds a training corpus and fine-tunes a small LLM with
QLoRA for a **task-oriented voice agent in Mexican Spanish**, with the full
lifecycle orchestrated, versioned and reproducible.

**This repo is the pipeline, not the agent.** `src/agent/` is a reference runtime
that consumes the adapter so the model can be evaluated and demonstrated. When in
doubt about where code belongs, ask whether it produces the model (`src/pipeline/`)
or consumes it (`src/agent/`).

Two phone flows:

1. **`validate_contact`** — confirm and correct name, phone and email, field by field.
2. **`confirm_appointment`** — call 30 min before an appointment; reschedule if needed.

The deliverable is **the pipeline**, not the model. Reproducibility and lineage
matter more than squeezing the last point of accuracy.

Full design rationale lives in `docs/` (Phase 2 document). When a decision seems
arbitrary, it is probably documented there — ask before overriding it.

---

## Hard rules

These are not preferences. Violating them breaks the project's guarantees.

### 1. Language convention

| Scope | Language |
|---|---|
| Code, DAG ids, table/column names, graph nodes, JSON schemas, branches, commits, logs, docstrings | **English** |
| Training corpus, prompt bodies, model outputs, agent speech | **Spanish (es-MX)** |

No accented characters or `ñ` in identifiers, ever. Prompt *files* have English
ids and filenames; the prompt *text* inside is Spanish.

### 2. DAGs contain no logic

Every Airflow task calls a function in `src/`. A DAG file may only wire tasks
together. This exists so the whole pipeline is unit-testable without Airflow.

If you find yourself writing a loop or a transformation inside `dags/`, move it
to `src/pipeline/` and import it.

### 3. The model never invokes a tool

The LLM emits **structured JSON only**, validated against `schemas/`. LangGraph
interprets that output, decides the transition, and performs any write
(`update_contact`, `mark_confirmed`, `reschedule`).

Never give the model tool-calling access to those functions. A model error must
not be able to write to the database or move an appointment.

### 4. Invalid model output is `ambiguous`, never an exception

If output fails schema validation, treat it as `ambiguous` and retry. A parse
failure must never crash a live call.

### 5. The eval set is frozen

`evaluation/eval_set/` is immutable after week 3. Do not add, edit, remove or
regenerate its contents. Do not generate eval data with an LLM. If you think the
eval set needs to change, stop and ask.

### 6. Notebooks are never imported

`notebooks/` is exploration only. Nothing in `src/` or `dags/` may import from
it. When notebook code proves useful, move it to `src/` with a test.

### 7. Prompts are versioned artifacts

Never edit a prompt file in place. Create a new version (`v2.yaml`) and register
its hash. A prompt change is a new experiment — this is what Hypothesis 3
measures.

### 8. Large files go to DVC, not Git

Anything in `data/`, model checkpoints, adapters, audio. Pre-commit blocks files
over 500 KB.

---

## Architecture

```
dags/           orchestration only
src/pipeline/   data construction + training  (Spark, GPU)
src/agent/      runtime                       (latency-sensitive, no Spark, no torch)
prompts/        versioned prompt artifacts
schemas/        JSON Schema per model output
evaluation/     frozen eval set + reports
```

The two `src/` zones have separate dependency groups in `pyproject.toml`
(`pipeline`, `train`, `agent`). Do not import across them: `src/agent/` must
never import `pyspark`, `torch` or `datasets`.

`src/agent/speech/base.py` defines provider-agnostic `SpeechToText` /
`TextToSpeech` interfaces. ElevenLabs is one implementation. Never call the
ElevenLabs SDK directly from a node or graph.

---

## Key design decisions (do not silently reverse)

| Decision | Reason |
|---|---|
| Base model is **Qwen3-4B** | Apache-2.0. Qwen2.5-3B is research-license, non-commercial |
| Second arm is **Qwen2.5-1.5B-Instruct** | Minimum viable size (H5a) |
| **QLoRA**, r=16, 1–2 epochs, early stopping | Narrow tasks risk rigidity via catastrophic forgetting, not classic overfitting |
| **10–20 % general-instruction replay** in the mix | Preserves out-of-scope handling (H5b). Not optional |
| STT/TTS are **bought**, not built | See `docs/` — build vs. buy is settled |
| Context is **prefetched by primary key**, not retrieved | 1–5 ms vs 100–300 ms; `contact_id` is known at dial time |
| Vector search reserved for unpredictable Q&A only | Filtered by `company_id` |
| After 2 failed reprompts: **mark unvalidated and move on** | Does not escalate to a human, does not end the call |
| `call_rejected` ends the call immediately | One rejection signal is enough. The agent never insists |

**Rejection disambiguation:** `cannot_attend` refers to the *appointment*;
`call_rejected` refers to the *call*. When ambiguous between the two, choose
`call_rejected` — cutting a call short is far better than pressing someone who
is driving.

---

## Working style

- **Small vertical slices.** One module plus its tests per change. Do not
  scaffold ten empty files.
- **Tests first for contracts.** Schema validation, node transitions, and
  normalizers (phone → E.164, spelled email → address) are pure functions and
  should be tested before implementation.
- **Run `make test` and `make lint` before declaring a task done.**
- **Ask when a decision is not covered here.** Guessing at a convention costs
  more than a question.
- Do not add dependencies without saying which group they belong to and why.

## Commands

```bash
make up      make init     make test     make lint
make ingest  make curate   make localize make generate
make augment make gold     make train    make eval
```

## Definition of done

- [ ] Type hints on public functions
- [ ] Docstring says *why*, not *what*
- [ ] Unit test covering the happy path and one failure mode
- [ ] `make lint` and `make test` pass
- [ ] No new large files in Git
- [ ] Identifiers in English
