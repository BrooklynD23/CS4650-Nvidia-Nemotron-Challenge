# Notebook 10: Submission Packaging and Provenance

**Parent Issue**: `#20`
**Plan Phase**: Phase 8 (Final Submission)
**Scaffold**: `notebooks/10_submission_packaging_and_provenance.ipynb`
**Status**: `active` (packager code exists; notebook is a thin runbook scaffold)
**Dependencies (upstream)**: `#14` (constraints frozen), `#25` (real adapter output)

---

## 1. Objective

Provide a reproducible, verifiable **packaging entrypoint** that turns a LoRA adapter directory into a Kaggle-compliant `submission.zip`, plus an out-of-band provenance manifest (`submission.manifest.json`) so every submission attempt is auditable.

## 2. What already exists in the repo

The packaging path is implemented and tested:

- **Library**: `src/inference/submission.py`
  - validates adapter directory
  - writes `submission.zip` with **exactly**:
    - `adapter_config.json`
    - `adapter_model.safetensors`
    at the **zip root**
  - writes `submission.manifest.json` **beside** the zip (never inside)
  - validates zip shape (no nested paths, no directories, no extras)
- **CLI wrapper**: `scripts/package_submission.py`
- **Tests**: `tests/inference/test_submission_packaging.py`

## 3. Canonical usage

Recommended output location (kept out of git):

- `experiments/submissions/<run_id>/submission.zip`
- `experiments/submissions/<run_id>/submission.manifest.json`

Example command:

```bash
python scripts/package_submission.py \
  --adapter-dir adapters/<your_adapter_dir> \
  --output-dir experiments/submissions/<run_id> \
  --base-model-id metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \
  --adapter-rank 32 \
  --dataset-version <dataset_version> \
  --eval-sha <eval_sha>
```

## 4. What is intentionally deferred (future work)

This notebook’s earlier plan text mentioned “dry-run” evaluation gates (golden set + validation slice) and a Kaggle upload wrapper.

Those are **not** part of the implemented packaging module yet, and should be treated as separate follow-on work once the real eval artifacts exist:

- `#18`: real `data/eval/validation_200.jsonl` + `data/eval/golden_20.jsonl`
- `#19`: a real baseline eval run ID + normalizer ID

When those exist, we can add:

- a **dry-run script** that loads the base model + adapter and executes:
  - a smoke test
  - golden regression gate
  - small validation slice
- a “submit checklist” document

## 5. Definition of done for Notebook 10

This notebook is “done enough” when it:

- points to the real packaging entrypoint (`scripts/package_submission.py`)
- explains the minimal zip contract and the out-of-band manifest policy
- links to the acceptance tests

## 6. References

- `docs/execution/plans/issue-20-submission-packaging-and-provenance.md`
- `docs/architecture/COMPETITION.md` (verified submission contract)
- Kaggle submission demo: https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
