---
title: What Exists in the Repo Today
audience: beginner
page_type: project-status
status: implemented
last_reviewed: 2026-05-05
repo_sources:
  - docs/execution/SPRINTS.md
  - docs/execution/NOTEBOOKS.md
  - src/contracts.py
  - src/evaluation/runner.py
  - src/evaluation/prompt_sweeps.py
  - src/inference/submission.py
  - src/data/synthetic.py
  - src/training/sft_trainer.py
  - scripts/hpc/
  - notebooks/08_synthetic_data_recipe.ipynb
  - docs/architecture/SYNTHETIC_PROVENANCE.md
external_sources: []
---

# What Exists in the Repo Today

## Why This Page Exists

This page is the safest answer to "What has actually been built already?" It
focuses on committed repo state rather than hopes, future phases, or marketing
language.

## What Exists Today

As of the latest review date on this page, the repo contains delivered Wave A/B
work in six broad areas:

- execution and review harness docs
- canonical data contracts and schema utilities
- validation and golden-set evaluation gates
- baseline evaluation and submission-packaging code
- prompt and decode sweep helpers (deterministic run-id construction, sparse grid
  expansion, Best-of-N majority vote, aggregate CSV writing, findings markdown
  rendering)
- a bounded external-baseline review artifact for Tong (`tonghuikang`) and
  konbu17, with Adopt / Reject / Gate decisions tied back to the verified
  Kaggle/NVIDIA contract
- synthetic data generation with solver-first teacher policy, quality filters,
  cost caps, and SHA-256 fingerprinting (`src/data/synthetic.py` with config
  integration via `configs/synthetic_prompts.yaml`)
- training infrastructure and masking for SFT (`src/training/sft_trainer.py` with
  `apply_loss_mask()` function, plus training config files: `configs/lora_baseline.yaml`,
  `configs/lora_qlora.yaml`, `configs/smoke_sft.yaml`)
- HPC queue submission and checkpoint management scripts
  (`scripts/hpc/` directory with 8 production scripts for preflight validation,
  dataset tokenization, SFT/RL submission, checkpoint policy, regression gating,
  adapter packaging, and checkpoint resumption)
- fully executable synthetic data recipe notebook
  (`notebooks/08_synthetic_data_recipe.ipynb` with generation loop, stub teacher,
  fingerprint verification, and summary table)
- artifact provenance schema documentation
  (`docs/architecture/SYNTHETIC_PROVENANCE.md` defining layout, fingerprint algorithm,
  and provenance ledger)

That means the repo is no longer just planning documents. It now contains real
Python modules for contracts, evaluation, packaging, prompt sweeping, synthetic
data generation, and training, along with tests for each and documented runbooks
for HPC execution.

## Current Repo Evidence

- The phase map in `docs/execution/SPRINTS.md` shows Waves A and B as the
  foundation for the project.
- The notebook registry in `docs/execution/NOTEBOOKS.md` marks the early
  notebooks as validated or active rather than scaffold-only. Notebook 08
  is now `active` with full executable implementations.
- `src/contracts.py` defines the shared project contracts.
- `src/evaluation/runner.py` and related modules show that the eval pipeline now
  exists in code.
- `src/evaluation/prompt_sweeps.py` contains the fully implemented sweep helpers
  introduced in issue #21. The module is tested and importable; the notebook
  that calls it is `active` in NOTEBOOKS.md but has not been executed end-to-end
  because the required split artifacts (`data/eval/validation_200.jsonl` and
  `data/eval/golden_20.jsonl`) are not yet present in the repo.
- `src/inference/submission.py` and `scripts/package_submission.py` implement the
  submission packaging path in code (not only in planning docs). The packaging
  contract matches the verified competition layout: `submission.zip` contains
  **only** `adapter_config.json` + `adapter_model.safetensors`, and a sibling
  `submission.manifest.json` is written beside the zip for provenance (recommended
  output directory: `experiments/submissions/<run_id>/`).
- `docs/analysis/EXTERNAL_BASELINE_REVIEW.md` records the current external
  baseline review and keeps public reference choices subordinate to the official
  `#14` competition contract.
- `src/data/synthetic.py` implements the synthetic data pipeline with
  solver-first teacher policy. The module integrates with `configs/synthetic_prompts.yaml`
  via `prompt_config_path` field in `SyntheticConfig`, and enforces quality
  filters with `solver_confidence_threshold` gate. Tests are in
  `tests/data/test_synthetic.py`.
- `src/training/sft_trainer.py` introduces the `apply_loss_mask()` function for
  completion-only masking during SFT. Training config files exist at
  `configs/lora_baseline.yaml` (r=32), `configs/lora_qlora.yaml` (r=16 QLoRA),
  and `configs/smoke_sft.yaml` (100-step smoke test). Tests are in
  `tests/training/test_masking.py` with 4 masking test cases.
- `notebooks/08_synthetic_data_recipe.ipynb` is now fully executable with a
  complete generation loop, stub teacher implementation, fingerprint verification,
  and summary table output.
- `docs/architecture/SYNTHETIC_PROVENANCE.md` documents the artifact layout,
  provenance schema, and SHA-256 fingerprint algorithm for reproducibility.
- `scripts/hpc/` directory contains 8 production-ready scripts: `preflight.sh`
  (validation), `tokenize_dataset.py` (dataset prep), `submit_sft.sbatch` and
  `submit_rl.sbatch` (HPC batch submission), `checkpoint_policy.py` (checkpoint
  selection), `regression_gate.py` (validation gates), `package_adapter.py`
  (submission packaging), and `resume_from_latest.py` (checkpoint resumption).

## Why This Matters

For a non-technical reader, this means the team has moved past "we have a plan"
and into "we have a working foundation." That foundation is what later training
and solver work will build on.

## Sources

- Repo: [docs/execution/SPRINTS.md](../../execution/SPRINTS.md)
- Repo: [docs/execution/NOTEBOOKS.md](../../execution/NOTEBOOKS.md)
- Repo: [src/contracts.py](../../../src/contracts.py)
- Repo: [src/evaluation/runner.py](../../../src/evaluation/runner.py)
- Repo: [src/inference/submission.py](../../../src/inference/submission.py)
- Repo: [docs/analysis/EXTERNAL_BASELINE_REVIEW.md](../../analysis/EXTERNAL_BASELINE_REVIEW.md)
