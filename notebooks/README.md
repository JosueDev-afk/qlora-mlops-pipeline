# Notebooks

Notebooks are for **exploration only**. Nothing here is ever imported by `src/` or `dags/`.
When code proves useful, move it into `src/` with a test.

`nbstripout` runs in pre-commit: outputs are never committed, so diffs stay readable.

| Notebook | Purpose | Ola |
|---|---|---|
| `01_explore_corpora.ipynb` | Profile public corpora: turn length, dialogue length, slot coverage | 1 |
| `02_asr_error_profile.ipynb` | Measure Scribe v2 WER on es-MX audio; characterize alphanumeric errors. **Feeds the augmentation calibration.** | 1 |
| `03_baseline_eval.ipynb` | Prompt-only baseline vs frontier model on the frozen eval set | 1 |
| `04_localization_qa.ipynb` | Manual review of translated/localized samples (naturalness gate) | 2 |
| `05_synthetic_review.ipynb` | Diversity metrics and spot checks on generated dialogues | 2 |
| `06_training_analysis.ipynb` | Loss curves, hyperparameter sweep, error taxonomy | 3 |
| `07_hypotheses.ipynb` | Figures and statistical tests for H1–H5 | 4 |
