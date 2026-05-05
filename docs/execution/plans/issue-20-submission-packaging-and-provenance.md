# Issue 20 — Submission Packaging and Provenance Implementation Plan

> **Status:** Implemented in code (packager + tests). Remaining items are docs alignment + future “dry-run” gates.

**Goal:** Produce a Kaggle-valid `submission.zip` for the Nemotron adapter and a separate provenance record that can be reviewed without contaminating the submission archive.

**Architecture:** Treat packaging as two artifacts with one source of truth.

- The submission zip is intentionally minimal and mirrors the confirmed konbu17 packaging pattern:
  - `adapter_config.json` and `adapter_model.safetensors` at the zip root
- All provenance lives beside the zip, not inside it:
  - `submission.manifest.json`

Packaging must fail closed if either required adapter file is missing, if the zip contains extra paths or nested folders, or if required manifest fields are missing.

**Implemented in:**
- `src/inference/submission.py` (packager + validators)
- `scripts/package_submission.py` (CLI wrapper)
- `tests/inference/test_submission_packaging.py` (acceptance tests)

---

## Evidence Basis

- `docs/architecture/ARCHITECTURE.md` defines `PackageManifest` and makes provenance a first-class artifact.
- `docs/architecture/COMPETITION.md` “Verified” section (snapshot 2026-04-29) freezes the submission constraints:
  - `r <= 32` (evaluator enforces `max_lora_rank=32`)
  - zip root contains only `adapter_config.json` + `adapter_model.safetensors`
- konbu17 baseline writes the same minimal zip shape.

## Decisions

- `submission.zip` root contents are exactly:
  - `adapter_config.json`
  - `adapter_model.safetensors`
- No `README.md`, no manifest, no logs, and no nested directory structure inside `submission.zip`.
- Provenance is written out-of-band as `submission.manifest.json` beside the zip.
- Recommended run output location: `experiments/submissions/<run_id>/` (kept out of git).
- Manifest contents include (at minimum) the canonical `PackageManifest` fields:
  - `base_model_id`, `adapter_rank`, `dataset_version`, `eval_sha`, `artifact_paths`, `created_at`
  and packaging augments the manifest JSON with:
  - `artifact_sha256`, `submission_zip_sha256`, `submission_zip_bytes`, `git_sha`

## Tasks (implemented)

### Task 1: Define the submission contract

**Files:**
- Create: `src/inference/submission.py`

- [x] Reuse the canonical `PackageManifest` dataclass from `src/contracts.py` (single source of truth).
- [x] Add a `SubmissionBundle` dataclass with `adapter_dir`, `submission_zip`, and `manifest_path`.
- [x] Implement `validate_adapter_dir()` so it fails if either required file is missing (and rejects symlinks / path traversal).
- [x] Implement `validate_submission_zip()` so it rejects extra entries, nested paths, directory entries, and missing root files.

### Task 2: Add the packager entrypoint

**Files:**
- Create: `scripts/package_submission.py`

- [x] Parse `--adapter-dir`, `--output-dir`, `--base-model-id`, `--adapter-rank`, `--dataset-version`, and `--eval-sha`.
- [x] Write `submission.zip` with root-only archive names.
- [x] Write `submission.manifest.json` beside the zip (never inside).
- [x] Print a single-line success summary including zip path, manifest path, and zip size.

### Task 3: Add acceptance tests

**Files:**
- Create: `tests/inference/test_submission_packaging.py`

- [x] Test that the zip contains exactly `adapter_config.json` and `adapter_model.safetensors` at the zip root.
- [x] Test that the manifest file is outside the zip and references the zip hash.
- [x] Test that extracting the zip yields a directory accepted by `PeftConfig.from_pretrained(...)` (skipped unless explicitly enabled).
- [x] Test a cached-base-model smoke load path behind an explicit opt-in env var so CI stays offline.

## Acceptance Tests

- Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/inference/test_submission_packaging.py -q -p no:cacheprovider`
- Run: `python -m zipfile -l experiments/submissions/<run_id>/submission.zip`
- Run: `python -m zipfile -t experiments/submissions/<run_id>/submission.zip`

## Non-Goals

- Do not embed provenance inside the Kaggle submission zip.
- Do not require network access during tests.
- “Dry-run” evaluation gates (golden/validation execution) are separate work and should depend on real `#18/#19` artifacts.
