# Issue 20 — Submission Packaging and Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Kaggle-valid `submission.zip` for the Nemotron adapter and a separate provenance record that can be reviewed without contaminating the submission archive.

**Architecture:** Treat packaging as two artifacts with one source of truth. The submission zip is intentionally minimal and mirrors the confirmed konbu17 packaging pattern: only `adapter_config.json` and `adapter_model.safetensors` at the zip root. All provenance lives beside the zip, not inside it, so the archive stays Kaggle-clean while the run remains auditable and restorable. Packaging must fail closed if either required adapter file is missing, if the zip contains extra paths or nested folders, or if the extracted adapter cannot be reloaded.

**Tech Stack:** Python 3.11, `zipfile`, `json`, `hashlib`, `pathlib`, `safetensors`, `peft`, `transformers`, `pytest`.

---

## Evidence Basis

- `docs/architecture/ARCHITECTURE.md` defines `PackageManifest` and makes provenance a first-class artifact.
- `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) freezes the submission constraints: `r <= 32` (evaluator enforces `max_lora_rank=32`) and zip root containing only `adapter_config.json` + `adapter_model.safetensors`.
- `data/external/konbu17/cells/cell04.py` zips exactly two files at the archive root: `adapter_config.json` and `adapter_model.safetensors`.
- `data/external/konbu17/cells/cell15.py` enforces those two files as required adapter inputs.
- `data/external/konbu17/cells/cell17.py` packages either a freshly trained adapter or a pretrained adapter path using the same two-file contract.

## Decisions

- `submission.zip` root contents are exactly:
  - `adapter_config.json`
  - `adapter_model.safetensors`
- No `README.md`, no manifest, no logs, and no nested directory structure inside `submission.zip`.
- Provenance is written out-of-band as `submission.manifest.json` beside the zip in the same run directory.
- Canonical run output location: `experiments/submissions/<run_id>/`, so the zip and manifest stay outside source control.
- Manifest contents include `base_model_id`, `adapter_rank`, `dataset_version`, `eval_sha`, `artifact_paths`, `created_at`, source git commit, and SHA256 hashes for the adapter files and final zip.

## Tasks

### Task 1: Define the submission contract

**Files:**
- Create: `src/inference/submission.py`

- [ ] Add a frozen `PackageManifest` dataclass that matches `docs/architecture/ARCHITECTURE.md` and extends it with hashes and git provenance.
- [ ] Add a `SubmissionBundle` dataclass with `adapter_dir`, `submission_zip`, and `manifest_path`.
- [ ] Implement `validate_adapter_dir()` so it fails if either required file is missing.
- [ ] Implement `validate_submission_zip()` so it rejects extra entries, nested paths, and missing root files.

### Task 2: Add the packager entrypoint

**Files:**
- Create: `scripts/package_submission.py`

- [ ] Parse `--adapter-dir`, `--output-dir`, `--base-model-id`, `--adapter-rank`, `--dataset-version`, and `--eval-sha`.
- [ ] Copy the two required adapter files into a staging directory and write `submission.zip` with root-only archive names.
- [ ] Write `submission.manifest.json` beside the zip and include the file hashes, git SHA, and artifact paths.
- [ ] Print a single-line success summary with the zip path, manifest path, and zip size.

### Task 3: Add acceptance tests

**Files:**
- Create: `tests/inference/test_submission_packaging.py`

- [ ] Test that the zip contains exactly `["adapter_config.json", "adapter_model.safetensors"]`.
- [ ] Test that the manifest file is outside the zip and references the zip hash.
- [ ] Test that extracting the zip yields a directory accepted by `PeftConfig.from_pretrained(...)`.
- [ ] Test that a cached-base-model smoke load via `PeftModel.from_pretrained(...)` works when the competition base model is available locally; otherwise skip with a clear message.

## Acceptance Tests

- Run: `pytest tests/inference/test_submission_packaging.py -q`
  - Expected: zip layout test passes; manifest location test passes; loadability test passes or skips only if the base model cache is unavailable.
- Run: `python -m zipfile -l experiments/submissions/<run_id>/submission.zip`
  - Expected: exactly two entries at the archive root, no folders.
- Run: `python -m zipfile -t experiments/submissions/<run_id>/submission.zip`
  - Expected: archive integrity check passes.
- Run: extract the zip and load the adapter from the extracted directory with PEFT.
  - Expected: adapter config parses and the `safetensors` file opens without corruption.

## Non-Goals

- Do not embed provenance inside the Kaggle submission zip.
- Do not add notebook changes for this issue.
- Do not depend on network access during the zip-shape test.
