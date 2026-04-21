# Code Review: Wave A/B foundation commits (`6772065..af9a258`)

Date: 2026-04-20 (America/Los_Angeles)

## Scope

- Branch: `main`
- Commits reviewed:
  - `6772065` docs: land sprint planning + wave issue plans (pre-exec context)
  - `5a59f4d` feat(#17): canonical schema layer + tests
  - `95357a2` feat(#14): constraints-freeze notebook 00 + BLOCKED gate
  - `d22208c` feat(#20): submission packaging + provenance tests
  - `dc7a502` feat(#18): validation + golden regression gate
  - `af9a258` feat(#16): external-baselines delta notebook 01 (gated on #14)

## Test status

- Command: `.venv/bin/python -m pytest tests/ -q`
- Original review result: `71 passed, 1 skipped`

## Remediation status

Status updated: 2026-04-20

- Fixed: submission packaging now rejects symlinked required files and duplicate zip entries.
- Fixed: split reservation now fails closed unless every input row has `split == "train"`.
- Fixed: Issue #19 planning docs now use `normalizer_id`, matching the frozen `EvalRecord` contract.
- Fixed: schema mapping now rejects non-mapping `metadata` and missing-sentinel text values instead of silently coercing them.
- Fixed: `SFTExample` now enforces string `role` / `content` values.
- Fixed: golden gate now rejects duplicate golden `example_id` values.
- Fixed: the split seed test is now deterministic.
- Fixed: the optional PEFT smoke fixture now uses a minimally valid synthetic config instead of xfail-on-`ValueError`.
- Verification after remediation: `.venv/bin/python -m pytest tests/ -q` → `78 passed, 1 skipped`.

---

## Findings

### HIGH — Security: Submission packaging accepts symlinked adapter files

**File**: `src/inference/submission.py:68`

**Issue**: `validate_adapter_dir()` only checks `Path.is_file()`. A symlink named `adapter_model.safetensors` (or `adapter_config.json`) that points outside the adapter directory will pass, and `_write_submission_zip()` will package the symlink target bytes.

**Why it matters**: Accidental leakage of arbitrary local files into `submission.zip`, and/or packaging the wrong adapter bytes while the manifest implies correctness.

**Suggestion**: Reject symlinks for required files *or* require that `fpath.resolve()` stays within `adapter_dir.resolve()`. Add a test that creates a symlink to an external file and asserts packaging fails closed.

---

### HIGH — Bug risk: Split reservation doesn’t enforce `ReasoningExample.split == "train"`

**File**: `src/evaluation/splits.py:95`

**Issue**: `_validate_inputs()` validates types/IDs/category/prompt but never checks `ex.split`. If a caller passes mixed rows (e.g., `test`, already-reserved `val/golden`), those rows can be reserved into new validation/golden artifacts.

**Why it matters**: The “golden” gate is only meaningful if artifacts are guaranteed disjoint from non-train / previously-reserved rows.

**Suggestion**: Require `split=="train"` for all inputs (or take an explicit `allowed_splits` param) and add a unit test that passing a non-train row fails closed.

---

### HIGH — Contract drift: `normalization_version` vs `normalizer_id` (Issue #19)

**Files**:
- `docs/execution/plans/issue-19-baseline-eval-and-normalization.md:71`
- `src/contracts.py:145`

**Issue**: The #19 plan doc uses `normalization_version` as a record/config field, but the canonical shipped eval contract is `EvalRecord.normalizer_id`.

**Why it matters**: Implementing #19 as written risks producing artifacts incompatible with `EvalRecord` validation and downstream modules (`src/evaluation/records.py`, `src/evaluation/golden_gate.py`).

**Suggestion**: Standardize on **`normalizer_id`** everywhere, and treat the version string (e.g. `exact_v1`) as the `normalizer_id` value. Update the plan doc and any handoff prompt text accordingly.

---

### MEDIUM — Bug risk: Zip validator can miss duplicate zip entries

**File**: `src/inference/submission.py:125`

**Issue**: `validate_submission_zip()` compares `set(zf.namelist())` to the required set. A zip with duplicate entries can still pass.

**Why it matters**: Duplicate zip members are a real edge case; validation should be strict for a submission artifact.

**Suggestion**: Validate list length + duplicates explicitly (e.g. `Counter(names)`), or require `sorted(names) == sorted(REQUIRED_ADAPTER_FILES)` and `len(names) == 2`.

---

### MEDIUM — Data integrity: Schema mapping can mask invalid metadata and silently coerce “missing” values

**File**: `src/data/schema_mapping.py:104`

**Issue**:
- `provided_metadata = canonical.get("metadata") or {}` can bypass the mapping-type check for falsy non-mapping values.
- `str(...)` coercion on `prompt/answer/category` can silently turn NaN/None into `"nan"`/`"None"`.

**Why it matters**: Silent ingest corruption can poison splits, eval, and SFT data without tripping the schema’s type checks.

**Suggestion**: Treat `"metadata"` presence separately (don’t use `or {}`), and consider rejecting non-`str` values for required text fields (or at least reject common “missing” sentinels post-coercion).

---

### MEDIUM — Contract validation gap: SFT message value types aren’t checked

**File**: `src/contracts.py:113`

**Issue**: `SFTExample.__post_init__` enforces message keys `role`/`content` exist but does not enforce their values are `str`.

**Why it matters**: Training/chat-template code will fail later (and noisier) if messages contain non-strings.

**Suggestion**: Validate `msg["role"]` and `msg["content"]` are `str`, plus a targeted test in `tests/test_contracts.py`.

---

### MEDIUM — Test flake: probabilistic seed-difference assertion

**File**: `tests/evaluation/test_splits.py:78`

**Issue**: Asserts two seeds produce different golden IDs; extremely unlikely, but still probabilistic.

**Why it matters**: Flaky tests reduce confidence and slow iteration.

**Suggestion**: Make deterministic by pinning expected IDs for known seeds on the fixed synthetic corpus (or test an invariant about RNG usage instead of inequality).

---

### LOW — Robustness: Golden gate doesn’t validate unique golden IDs

**File**: `src/evaluation/golden_gate.py:112`

**Issue**: No check for duplicate `row.example_id` in the `golden` rows input.

**Why it matters**: Duplicate golden rows inflate totals and can hide artifact-construction bugs.

**Suggestion**: Add explicit uniqueness validation (fail closed).

---

### LOW — Test signal: Optional PEFT smoke test can xfail on `ValueError`

**File**: `tests/inference/test_submission_packaging.py:215`

**Issue**: Optional PEFT smoke test `xfail`s on `ValueError` from `PeftConfig.from_pretrained`, weakening its value when enabled.

**Suggestion**: If you want it to be meaningful, make the synthetic `adapter_config.json` satisfy PEFT’s minimal schema so the test must pass when enabled.

---

## Notes

### Issue duplication check (Wave vs Sprint)

The current GitHub structure is consistent with “no duplication” so long as you treat:

- Sprint issues (`#1–#12`) as **epic/tracking** issues (`type:epic`) that do not execute work directly
- Wave issues (`#13–#25`) as the **canonical execution** issues that own deliverables

This mapping is already present in multiple epics (e.g., #1 → #14, #4 → #18/#19, #5 → #20).

### Issue #19 handoff prompt correction (minimum)

Replace `normalization_version` with `normalizer_id` everywhere in the prompt/spec, to match the canonical `EvalRecord` contract.

## Suggested next steps

1. Fix the #19 contract drift in `docs/execution/plans/issue-19-baseline-eval-and-normalization.md` before implementation starts.
2. Harden packaging validation (symlink policy + duplicate zip entries).
3. Enforce `split=="train"` (or explicit allowed splits) in `reserve_splits()` + add tests.
4. De-flake the seed test in `tests/evaluation/test_splits.py`.
