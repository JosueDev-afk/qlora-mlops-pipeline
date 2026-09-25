# Working with Claude Code on this repo

## Before you start

```bash
git init && git add -A && git commit -m "chore: scaffold"
git branch -M main
```

Commit the scaffold **first**. That gives Claude Code a stable base and gives you
a diff to review against.

## Rules of engagement

- One task per session, one vertical slice: a module plus its tests.
- Never "implement the pipeline". Always name the file.
- Review every diff before committing. If it invents a convention, point at
  `CLAUDE.md` instead of accepting it.
- When a task depends on a design decision, paste the relevant section of the
  Phase 2 document into the prompt.

## Suggested order (Ola 1, weeks 1–4)

Contracts and pure functions first — they are testable without infrastructure
and they pin down the interfaces everything else depends on.

| # | Task | Why first |
|---|---|---|
| 1 | `src/common/`: config loader, structured logging, prompt loader with hash | Everything imports this; lives outside both zones so the agent can use it |
| 2 | `src/agent/llm/structured.py`: schema validation, `parse_or_ambiguous` | Rule 4 lives here |
| 3 | `src/common/normalizers.py`: spelled-email → address, spoken digits → E.164, name variants | Pure functions, high test value, **the actual hard part of the project** |
| 4 | `tests/unit/test_normalizers.py` with es-MX cases | Seed cases exist; extend them yourself with real Scribe transcripts. You are the native speaker |
| 4b | `src/pipeline/synth/spoken_forms.py`: canonical value → spoken es-MX variants | The inverse of item 3. Generates the corpus with labels by construction; round-trip tests against the normalizers |
| 5 | `infra/docker-compose.yml`: postgres, airflow (Spark in-process), mlflow, kafka, metabase, optional hdfs, with a **profile per stage** | Unblocks everything else. On a 16 GB Mac the whole stack never runs at once |
| 6 | `src/agent/speech/elevenlabs.py` implementing the base interfaces | Keeps the provider behind the abstraction |
| 6b | `src/agent/decision/laya_model.py`: HTTP client implementing `DecisionModel` | Refuses answers not stamped with the expected calibration; keeps the agent torch-free |
| 6c | `src/serving/laya.py`: Laya server, multilingual checkpoint only, preloaded, fitted temperatures applied to the logits | Same serving path as Qwen on vLLM, so H1's latency comparison is fair |
| 7 | `src/agent/graphs/flow_validate_contact.py` + node stubs | Flow 1 end to end with the base model |
| 8 | `src/agent/telemetry/producer.py` + simulated call producer | Feeds the Kafka path |
| 9 | `dags/ingest_corpus.py` + `src/pipeline/ingest/` | First real data |
| 10 | `notebooks/02_asr_error_profile.ipynb` | Calibrates augmentation. **Do this yourself**, it is judgement work |
| 11 | `src/pipeline/calibrate/fit_temperature.py` + tests | Temperature scaling per question shape on the human calibration split; the ECE gate |
| 12 | `src/agent/demo/` (Pipecat over WebRTC) | Last, and recutable — the audio layer never blocks evaluation |

## Tasks to keep for yourself

Claude Code is good at code, not at these:

- Building and freezing the eval set — needs a native speaker's judgement
- Reviewing the carrier-phrase bank for naturalness — the whole point is that a human decides
- Annotating the human calibration split for Laya, separate from the eval set
- Choosing the personas and variation axes for synthetic generation
- Reading and interpreting license terms
- Deciding when a hypothesis is answered
- Deciding whether Laya actually beats Qwen on Task B — that call comes from the
  frozen eval set, and reverting to a single model is a legitimate outcome

## Prompt shape that works

> Implement `src/pipeline/common/normalizers.py`: a function that converts a
> spelled-out Spanish email transcription into an address. Handle "arroba",
> "punto", "guion bajo", "todo junto", and ASR fragmentation. Pure function, no
> I/O. Write `tests/unit/test_normalizers.py` first with at least 10 cases in
> es-MX, including three that should fail to parse. Follow CLAUDE.md.

Specific file, specific behaviour, tests first, explicit reference to the rules.
