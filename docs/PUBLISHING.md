# Publishing checklist

## Before publishing anything

- [ ] Read the IP / invention-assignment clause of your employment contract
- [ ] Confirm the framing is generic (contact validation and appointment confirmation
      by phone — a domain problem, not any company's product)
- [ ] Confirm no employer data, code, prompts or flows are present
- [ ] License audit complete (`DATA_PROVENANCE.md`)

## Dual-use considerations

This model is fine-tuned to sound natural on outbound calls and to extract names,
phone numbers and emails. That capability profile overlaps with phone fraud and
non-consented data collection. Publication is reasonable, but must be deliberate:

- [ ] Model card states intended use and **out-of-scope use** explicitly
- [ ] Document that the agent identifies itself as automated at the start of every call
- [ ] Document that a single rejection signal ends the call — measurable
      (rejection recall ≥ 0.95), not just a claim
- [ ] Consider open license for the dataset, more restrictive for the adapter

## Artifacts

| Artifact | Repo | License |
|---|---|---|
| Dataset | `<user>/es-mx-contact-dialogues` | TBD |
| LoRA adapter | `<user>/qwen3-4b-voice-agent-es-mx` | Apache-2.0 (base is Apache-2.0) |
| Code | this repo | Apache-2.0 |

Publish only stable DVC tags (`v1.0`, `v1.1`) so the public dataset and the project
lineage stay in sync.
