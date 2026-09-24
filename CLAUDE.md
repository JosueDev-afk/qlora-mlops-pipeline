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

Two flows:

1. **`validate_contact`** — confirm and correct name, phone and email, field by field.
2. **`confirm_appointment`** — call 30 min before an appointment; reschedule if needed.

**Two models, on purpose.** Deciding and generating are different problems:

| Model | Tasks | Latency |
|---|---|---|
| **Laya-multilingual** (322M, non-autoregressive) | B `classify_intent`, D `is_real_interruption` | ~33 ms |
| **Qwen3-4B + QLoRA** (generative) | A `extract_entity`, C `parse_datetime` | < 500 ms |

Laya runs first on every turn and decides whether Qwen is needed at all.

**Two layers, on purpose.** LangGraph is not real-time — it runs once per turn.
Pauses, interruptions and barge-in happen in milliseconds, inside and between
turns. Layer 1 (Pipecat over WebRTC) owns the audio; layer 2 (LangGraph) owns
the turn. Code for the two lives in `src/agent/` but never blurs the boundary.

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

**Exception — Task D (`is_real_interruption`).** It runs on ASR partials while
the agent speaks: there is no retry (the next partial is the retry) and a
boolean has no `ambiguous` value. Any failure — invalid output, timeout, error —
resolves to an interruption (`src/agent/decision/interruption.py`). The
interruption threshold must stay below 0.5, so an unsure model stops talking
instead of talking over someone.

### 5. Laya: multilingual, preloaded, calibrated

Three rules, all from the model card, all easy to get wrong:

- **Never the English root checkpoint.** It collapses on non-Latin script *while
  staying confident*, so confidence gating cannot save you. Spanish always uses
  `laya-multilingual`.
- **Never lazy-load in a service.** Without preload the router rebuilds the
  checkpoint on every language switch (7–10 s measured). Always
  `Router(preload=True)` at start-up, never per request.
- **Never threshold on raw probabilities.** Laya ships over-confident. Apply the
  temperatures fitted by the `calibrate_laya` DAG before any comparison against
  a threshold.
- **Never hand-set the `call_rejected` threshold.** `calibrate_laya` derives it
  from the calibrated PR curve on the calibration split for
  `calibration.target_recall`, and stores it with the temperatures. Only the
  demo's `audio.early_rejection_threshold` (ASR partials, never evaluated) is
  set by hand.

### 6. The eval set is frozen

`evaluation/eval_set/` is immutable after week 3. Do not add, edit, remove or
regenerate its contents. Do not generate eval data with an LLM. If you think the
eval set needs to change, stop and ask.

### 7. Notebooks are never imported

`notebooks/` is exploration only, in Jupyter (`make notebook`). Nothing in
`src/` or `dags/` may import from it. When notebook code proves useful, move it
to `src/` with a test.

### 8. Prompts are versioned artifacts

Never edit a prompt file in place. Create a new version (`v2.yaml`) and register
its hash. A prompt change is a new experiment — this is what Hypothesis 3
measures.

### 9. Large files go to DVC, not Git

Anything in `data/`, model checkpoints, adapters, audio. Pre-commit blocks files
over 500 KB.

---

## Architecture

```
dags/               orchestration only
src/pipeline/       data construction + training  (Spark, GPU)
src/agent/          runtime  (latency-sensitive, no Spark, no torch)
  ├─ speech/        layer 1: STT/TTS behind a provider-agnostic interface
  ├─ decision/      layer 1+2: Laya client behind a provider-agnostic interface
  ├─ llm/           layer 2: Qwen via an OpenAI-compatible client
  ├─ graphs/        layer 2: the two LangGraph state machines
  ├─ nodes/, tools/ layer 2: nodes emit JSON; tools perform writes
  ├─ telemetry/     call events to Kafka; feeds the analytics path
  └─ demo/          layer 1 entrypoint (Pipecat/WebRTC). Demo only, recutable
src/serving/        model servers: Qwen via vLLM, Laya preloaded  (GPU, torch)
src/common/         shared by all zones: prompts, config, logging, normalizers
prompts/            versioned prompt and typed-question artifacts
schemas/            JSON Schema per model output
evaluation/         frozen eval set + reports
```

The `src/` zones have separate dependency groups in `pyproject.toml`
(`pipeline`, `train`, `agent`, `serve`). Do not import across them: `src/agent/`
must never import `pyspark`, `torch` or `datasets`, and reaches both models over
HTTP. `src/serving/` is where torch runs at inference time. `src/common/` sits
outside the zones and may only use `[project].dependencies`, so any zone can
import it.

`src/agent/speech/base.py` defines provider-agnostic `SpeechToText` /
`TextToSpeech` interfaces. ElevenLabs is one implementation. Never call the
ElevenLabs SDK directly from a node or graph.

---

## Key design decisions (do not silently reverse)

| Decision | Reason |
|---|---|
| Base model is **Qwen3-4B** | Apache-2.0. Qwen2.5-3B is research-license, non-commercial |
| Second arm is **Qwen3-1.7B** | Minimum viable size (H5a). Same family as the 4B, so only size changes; Qwen2.5-1.5B would confound size with model generation |
| **QLoRA**, r=16, 1–2 epochs, early stopping | Narrow tasks risk rigidity via catastrophic forgetting, not classic overfitting |
| **10–20 % general-instruction replay** in the mix | Preserves out-of-scope handling (H5b). Not optional |
| STT/TTS are **bought**, not built | See `docs/` — build vs. buy is settled |
| Context is **prefetched by primary key**, not retrieved | 1–5 ms vs 100–300 ms; `contact_id` is known at dial time |
| Vector search reserved for unpredictable Q&A only | Filtered by `company_id` |
| After 2 failed reprompts: **mark unvalidated and move on** | Does not escalate to a human, does not end the call |
| `call_rejected` ends the call immediately | One rejection signal is enough. The agent never insists |
| Tasks B and D go to **Laya**, A and C to **Qwen** | Deciding and generating have opposite latency profiles |
| Laya runs **before** Qwen on every turn | A plain "sí, ahí estaré" costs ~33 ms instead of ~500 ms |
| Temperature calibration is **mandatory**, not optional | Without it the rejection threshold has no meaning (H6) |
| Audio layer is **demo only** | Hypotheses are evaluated offline on transcripts. Never let demo work block evaluation work |
| Transport is **WebRTC**, not telephony | Removes an external dependency that teaches nothing. Migrating is a Pipecat transport swap |
| Synthetic data is generated **by code** from its label | Labels are correct by construction and cost nothing per example, so there is no LLM-as-judge filter |
| **No translation or localization** of English corpora | Their domains (hotels, restaurants) do not teach alphanumeric capture, they add translationese, and they were most of the GPU/API cost |
| The only LLM-written corpus text is the **carrier-phrase bank** (1,000–2,000, human-reviewed) | Written with **Qwen3-8B**: open, no Cloud billing, larger than the arms under test. Never a free-tier API for corpus text: providers may train on what they receive |
| Laya temperatures are fitted on a **human calibration split**, never on gold | Gold is mostly synthetic; a calibration fitted there may not transfer to real speech (H6) |
| The eval set includes **Task D** and **≥ 100 `call_rejected` turns** before freezing | Nothing can be added after week 3; with 40 rejections a 0.95 recall has a CI of ~0.84–0.99 |
| Qwen3-4B is also trained on **Task B**, only as the H1 arm | Without it H1 has no generative arm. At runtime Task B stays on Laya |
| Latency is measured **to complete JSON** on a fixed **L4** | LangGraph needs the whole output to act; p95s are only comparable on the same hardware |
| Laya and Qwen are both **served over HTTP** from `src/serving/` | Keeps `src/agent/` torch-free, and H1 compares latencies over the same serving path |
| Qwen3 runs with **thinking disabled** | Reasoning tokens would spend the 500 ms budget before the JSON |
| **Everything runs locally** (MacBook Air M4, 16 GB) except GPU work | No Cloud billing, no Dataproc: Spark runs in local mode. Colab (Google AI Pro CCU) runs QLoRA, Laya fine-tuning and latency measurements, because the Mac has no CUDA for vLLM or bitsandbytes |
| Docker services run **by profile**, never the whole stack at once | 16 GB does not fit Airflow, HDFS, Kafka, Postgres, MLflow and Metabase next to a Spark job. Only Postgres runs by default |
| **No object store**: the lake is `data/{bronze,silver,gold}` on disk | MinIO stopped publishing images (Oct 2025) and Spark runs in local mode; DVC already versions `data/`. HDFS is an optional, amd64-only profile for the rubric |
| The DVC remote is a **local directory**; the human-made artifacts are copied elsewhere | Eval set, calibration split and carrier-phrase bank cannot be regenerated, and a remote on the same disk is not a backup |
| The H1/H1b frontier arm uses the **Gemini API free tier** | The only non-local component. Only the frozen eval set (no personal data) is sent, and the paper discloses it |

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
make ingest  make curate   make generate make augment
make gold    make train    make eval
make train-laya   make calibrate RUN_ID=...   make demo   make demo-text
make notebook
```

## Definition of done

- [ ] Type hints on public functions
- [ ] Docstring says *why*, not *what*
- [ ] Unit test covering the happy path and one failure mode
- [ ] `make lint` and `make test` pass
- [ ] No new large files in Git
- [ ] Identifiers in English
