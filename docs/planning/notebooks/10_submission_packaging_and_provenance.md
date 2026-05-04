# Notebook 10: Submission Packaging, Dry-Run, and Provenance

**Parent Issue**: `#20`
**Plan Phase**: Phase 8 (Final Submission)
**Scaffold**: `notebooks/10_submission_packaging_and_provenance.ipynb`
**Status**: `planned`
**Dependencies (upstream)**: `#14` (constraints frozen), `#19` (eval contract)
**Consumers (downstream)**: none — terminal notebook before Kaggle submit

---

## 1. Objective

Implement a reproducible, verifiable submission pipeline that packages a LoRA adapter into a Kaggle-compliant bundle, validates it end-to-end via dry-run (smoke test + golden_20 + validation_50 slice), computes provenance metadata (git SHA, adapter SHA256, training config hash, eval scores), and produces a signed manifest so every submission is auditable and restorable from a frozen state.

## 2. Why It Matters

- **Competition requirement**: Kaggle submission format must match frozen constraints from `#14` exactly; any drift breaks the submission.
- **Reproducibility & Trust**: A manifest ties the submitted adapter to training config, dataset version, git state, and eval baseline so results are reviewable.
- **Risk mitigation**: Dry-run validation prevents invalid adapters reaching Kaggle; golden_20 regression test catches accidental breakage.
- **Capstone learning goal**: End-to-end traceability from training → packaging → evaluation → submission demonstrates rigorous ML engineering.

## 3. Strategy — How We Aim To Accomplish It

1. **Enumerate competition-required files** from `#14` frozen constraints: `adapter_config.json` and `adapter_model.safetensors` at the zip root. Keep README, manifest, and provenance metadata outside the submitted zip.
2. **Implement `package(adapter_path, output_dir, config_dict)`**: copy adapter → compute SHA256 hash → write manifest.json (schema validated) → zip bundle.
3. **Implement `dry_run(base_model, adapter_path, dry_run_config)`**: load base + adapter, run Phase 1.2 smoke test (2^10 mod 7), verify output format.
4. **Run golden_20 validation**: load golden_20 from `data/eval/golden_20.jsonl`, verify all 20 pass (100% accuracy requirement); fail immediately if any drop.
5. **Run validation_50 slice**: load first 50 of `data/eval/validation_200.jsonl`, compute mean accuracy, compare to `#19` baseline within ±2σ; alert if drift.
6. **Generate manifest JSON**: include git SHA (fail if uncommitted changes), adapter SHA256, base model id + revision, training config hash, eval scores (golden_20, validation_50 mean), decode config, author, timestamp.
7. **Submit checklist**: document pre-Kaggle validation steps (manifest schema check, zip integrity, git status clean, adapter size within limit).

## 4. MVP (Minimum Viable Notebook)

**Scope**: Dry-run a packaged dummy (null identity) adapter end-to-end; produce valid manifest JSON; verify all artifacts present.

- **Inputs**: `data/eval/golden_20.jsonl`, `data/eval/validation_200.jsonl`, base model id from config, git repo state
- **Cells**:
  1. Import utilities, set paths, load config
  2. Define `PackageManifest` dataclass (git_sha, adapter_sha256, base_model, training_config_hash, eval_scores, decode_config, author, date)
  3. Implement `compute_file_hash(path)`: SHA256 of adapter file
  4. Implement `package()`: writes manifest, zips bundle to `submissions/<YYYY-MM-DD>_<tag>/`
  5. Implement `dry_run_smoke_test()`: load base + adapter, run 2^10 mod 7, verify `\boxed{2}` format
  6. Implement `dry_run_golden_set()`: load golden_20, verify 20/20 pass, print accuracy
  7. Implement `dry_run_validation_slice()`: load validation_200[:50], compute mean accuracy vs baseline
  8. Run all dry-run checks; print manifest JSON; verify zip unzips cleanly
  9. Write `SUBMISSION_CHECKLIST.md` (10-item pre-submit validation)

- **Outputs**:
  - `src/inference/submission.py` (packaging + dry-run classes/functions)
  - `submissions/<YYYY-MM-DD>_<tag>/adapter_model.safetensors` (null adapter for MVP)
  - `submissions/<YYYY-MM-DD>_<tag>/manifest.json` (signed provenance)
  - `submissions/<YYYY-MM-DD>_<tag>/adapter_model.zip` (Kaggle-submittable bundle)
  - `docs/execution/SUBMISSION_CHECKLIST.md` (pre-submit gates)
  - `scripts/submit.py` (wrapper to upload manifest + zip to Kaggle)

- **Verification**: Manifest is valid JSON Schema; adapter SHA256 matches re-hashed file; golden_20 = 20/20; validation_50 mean within recorded baseline ±2σ; zip unzips without error; git SHA matches current HEAD.

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: Base model loaded, dummy (null identity) adapter created, golden_20.jsonl and validation_200.jsonl present
- **Action**: Call `package(adapter, "submissions/2026-04-20_test/")` → call `dry_run_smoke_test()` → call `dry_run_golden_set()` → call `dry_run_validation_slice()`
- **Expected**: Manifest JSON written outside the zip; all three dry-run checks pass; golden_20 accuracy = 20/20; validation_50 mean accuracy within +/-2 sigma of `#19` baseline; zip contains only `adapter_config.json` and `adapter_model.safetensors` at root.

### 5.2 Alternative / Fallback

- **Setup**: Real trained adapter (Phase 4 or Phase 6 output) instead of dummy
- **Action**: Same packaging + dry-run pipeline
- **Expected**: Manifest reflects actual training config hash + adapter SHA256; golden_20 and validation_50 may show improvement over baseline; regression test still requires golden_20 ≥ 19/20 (allow 1 question drop from stochasticity)

### 5.3 Regression Guardrails

- **Manifest schema**: Must validate against frozen JSON Schema (include in `docs/schemas/manifest_v1.0.json`); fail packaging if invalid
- **Adapter SHA256**: Re-hash adapter file after zip creation; compare to manifest entry; must match exactly (prevent silent corruption)
- **Golden set**: Any final submitted adapter must have golden_20 ≥ 19/20 (allow max 1 drop); fail submission if this drops
- **Git cleanliness**: Refuse to package if `git status` shows uncommitted changes; include git SHA in manifest so future reviewers can checkout exact training state

## 6. Success Criteria (Done When)

- [ ] `src/inference/submission.py` exports `PackageManifest`, `package()`, `dry_run_smoke_test()`, `dry_run_golden_set()`, `dry_run_validation_slice()`
- [ ] Manifest schema (JSON Schema v7) defined in `docs/schemas/manifest_v1.0.json`
- [ ] Dummy adapter packaged, dry-run passes, manifest JSON is valid, zip unzips cleanly
- [ ] Golden_20 test passes (20/20); validation_50 mean within ±2σ of `#19` baseline
- [ ] Adapter SHA256 verified (re-hash matches manifest entry)
- [ ] Git SHA from `git rev-parse HEAD` included in manifest; fail if working tree dirty
- [ ] `scripts/submit.py` wrapper defined (loads manifest, prints submit checklist, prompts for Kaggle upload)
- [ ] `docs/execution/SUBMISSION_CHECKLIST.md` written with 10-item pre-submit validation list
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`

## 7. Risks & Open Questions

- **Risk**: Competition format changes between `#14` freeze (April 2026) and submission deadline → **Mitigation**: Pluggable packager abstraction in submission.py; implement CompetitionFormatV1 class; v2 format can be added without breaking training code.
- **Risk**: Timezone drift on deadline (uploaded manifest timestamp vs Kaggle server clock) → **Mitigation**: Use UTC timestamps, include timezone in manifest, document Kaggle deadline in submission checklist.
- **Risk**: Adapter size exceeds unconfirmed competition max (no explicit file-size limit found in `#14`) → **Mitigation**: Pre-flight check in packaging; warn if adapter is unusually large and monitor real submission feedback.
- **Risk**: Final submitted adapter differs in SHA256 from dry-run (accidental overwrite, file corruption) → **Mitigation**: Compute hash before zipping; store in manifest; re-hash after zip and verify; prevent re-upload of same adapter without explicit confirmation.
- **Resolved question**: Kaggle demo requires root `adapter_config.json` and `adapter_model.safetensors`; do not include optional metadata or README inside `submission.zip` unless the official contract changes.

## 8. Artifacts & Handoff

- **Produces**:
  - `src/inference/submission.py` (packaging + dry-run module)
  - `submissions/<YYYY-MM-DD>_<tag>/adapter_model.safetensors` (LoRA weights from Phase 4/6)
  - `submissions/<YYYY-MM-DD>_<tag>/manifest.json` (signed provenance metadata)
  - `submissions/<YYYY-MM-DD>_<tag>/adapter_model.zip` (Kaggle submission bundle)
  - `submissions/<YYYY-MM-DD>_<tag>/README.md` (bundle contents + training summary)
  - `docs/schemas/manifest_v1.0.json` (JSON Schema for validation)
  - `docs/execution/SUBMISSION_CHECKLIST.md` (pre-Kaggle gates)
  - `scripts/submit.py` (Kaggle upload wrapper)

- **Consumed by**: Manual Kaggle leaderboard submission (human reviews checklist, uploads zip, records leaderboard score)
- **External references cited**:
  - https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
  - Issue `#14` (constraints), Issue `#19` (eval contract), Issue `#20` (submission spec)

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| MVP (cells 1–9, dummy adapter, basic dry-run) | 4 | Local (any) |
| Alternative path (real Phase 4/6 adapter testing) | 2 | Colab Pro or local RTX 3080 |
| Full polish (schema v1.0, checklist docs, submit.py wrapper) | 3 |  Local (any) |
