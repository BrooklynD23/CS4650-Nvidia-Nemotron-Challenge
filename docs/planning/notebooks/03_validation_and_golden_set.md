# Notebook 03: Validation Split and Golden Set

**Parent Issue**: `#18`
**Plan Phase**: Phase 3.1 (Validation Set Reservation — must happen BEFORE any training)
**Scaffold**: `notebooks/03_validation_and_golden_set.ipynb`
**Status**: `scaffolded`
**Dependencies (upstream)**: `#14` (constraints), `#17` (schema)
**Consumers (downstream)**: `#19`, `#21`, `#22`, `#23`, `#24`, `#25`, `#20` (all training and eval notebooks)

---

## 1. Objective

Define and commit two frozen evaluation sets before any training occurs: a stratified validation split of 200 diverse problems (held out from training to monitor progress) and a "golden set" of 20 problems that the base model solves with high confidence (≥90% consistency), used as a regression tripwire. Any adapter that breaks these golden-set problems is rejected.

## 2. Why It Matters

- **Leakage prevention**: Validation must be held out before training data is prepared, not after.
- **Regression detection**: Golden set catches regressions that aggregate metrics hide (a 1% accuracy drop on MATH500 might mask golden-set failures).
- **Downstream gates**: Every training notebook (#19-#25) depends on these files locked and hash-verified to prevent accidental mutation.
- **Competition fairness**: Prevents overfitting to a small, visible test set; stratification ensures representative coverage.

## 3. Strategy — How We Aim To Accomplish It

1. **Reserve 200 validation problems** via stratified random sampling (seed=42) from eligible held-out source data, ensuring ≥4 categories (math, code, science, logic). Document source provenance.
2. **Select golden set (20 problems)** by running base model 5x with temperature=0 on candidate problems, keeping only those where ALL 5 runs produce the correct `\boxed{}` answer.
3. **Commit JSONL files** (`data/eval/validation_200.jsonl`, `data/eval/golden_20.jsonl`) with schema matching `#17` and record content hash (SHA-256).
4. **Create `scripts/build_eval_sets.py`** to make the selection process reproducible; document that re-running with seed=42 must yield identical files.
5. **Write `docs/execution/EVAL_POLICY.md`** documenting the no-touch rule: these files are locked; only authorized re-runs (e.g., with new base model version) can regenerate.
6. **Implement hash-check guards** in all eval notebooks (04, 05, 07, 09, 10) so any mutation is caught at load time.

## 4. MVP (Minimum Viable Notebook)

**Inputs**: 
- Base model `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (inference only)
- Eligible source data (e.g., Puzzle-KD Dataset v2, competition benchmark problems, publicly sourced math/code problems — NOT training-set samples)

**Cells**:
1. Load base model in BF16, prepare tokenizer with `enable_thinking=True`
2. Stratified sample 200 problems → save `data/eval/validation_200.jsonl`
3. Infer base model on candidates with temperature=0, 5 runs each
4. Filter to high-confidence problems (≥90% consistency) → take first 20 → save `data/eval/golden_20.jsonl`
5. Compute SHA-256 hash of both files, record in manifest
6. Assert: validation_200 has exactly 200 items; golden_20 has exactly 20 items; both match schema
7. Print hash summary and confirmation

**Outputs**: 
- `data/eval/validation_200.jsonl` (200 problems, locked)
- `data/eval/golden_20.jsonl` (20 problems, locked)
- `scripts/build_eval_sets.py` (reproducible selection with seed=42)
- `docs/execution/EVAL_POLICY.md` (or updated entry if exists)

**Verification**: Content hashes printed and committed; notebooks load and check hashes on startup.

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: Base model loads successfully; eligible source data available; seed=42 set globally
- **Action**: Run cells 1–7 (stratified sample, golden-set infer, hash computation)
- **Expected**:
  - `validation_200.jsonl` contains exactly 200 items with fields matching schema `#17`
  - Category distribution: math ≥50, code ≥40, science ≥40, logic ≥30 (≥4 categories)
  - `golden_20.jsonl` contains exactly 20 items, all with correct answer on 5/5 inference runs
  - Content hash (SHA-256) printed and recorded in manifest; hash stable across re-runs with seed=42

### 5.2 Alternative / Fallback

If base model inference is unavailable or too slow for 5 runs per candidate:
- **Setup**: Same source data; downgrade golden-set confidence threshold to ≥80% correct (3 inference runs instead of 5)
- **Action**: Run cells with `NUM_INFERENCE_RUNS=3` and `GOLDEN_CONFIDENCE_THRESHOLD=0.8`
- **Expected**: `golden_20.jsonl` still has 20 items, but flagged in `EVAL_POLICY.md` as "relaxed confidence (3 runs, ≥80%)" with date and justification

### 5.3 Regression Guardrails

- **Hash check on load**: Every eval notebook reads these files and asserts hash matches manifest; mutation fails with clear error
- **No regeneration during training**: If training code accidentally imports these files, it must use them in eval-only context (asserted by downstream notebooks)
- **Golden-set must-pass gate**: After training, golden_20 is re-evaluated; any drop below 20/20 correct is flagged as regression and blocks submission

## 6. Success Criteria (Done When)

- [x] Validation set (200 problems) reserved with documented source and stratification strategy
- [x] Golden set (20 problems) selected with base model ≥90% consistency (5 runs, temp=0)
- [x] Both JSONL files committed to `data/eval/` with SHA-256 hash recorded
- [x] `scripts/build_eval_sets.py` written and reproducible (seed=42)
- [x] `docs/execution/EVAL_POLICY.md` documents no-touch policy, regeneration approval process, and fallback thresholds
- [x] All eval notebooks (04, 05, 07, 09, 10) implement hash-check guards on load
- [x] Hash mismatch raises clear error with instructions for approval process
- [x] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`

## 7. Risks & Open Questions

- **Risk**: Source data for validation/golden set overlaps with Kaggle-provided test set (unknown to us). **Mitigation**: Use only public-domain source problems (published competition benchmarks, open datasets); document provenance in manifest.
- **Risk**: File normalization (e.g., whitespace, JSON key order) changes hash during git operations. **Mitigation**: Use canonical JSON serialization (sorted keys, no trailing spaces); verify hash post-clone on CI.
- **Risk**: Golden set is too easy (all base model answers are already >90%) or too hard (none qualify). **Mitigation**: Set confidence threshold as fallback (see Alternative path); adjust candidate pool size if needed.
- **Open question**: Should golden set be problem-disjoint from validation set? **Who answers**: #17 (schema), #18 (owner). Recommend: yes, disjoint (20 from 200 candidates, then sample separately).

## 8. Artifacts & Handoff

- **Produces**:
  - `data/eval/validation_200.jsonl` (200 problems, locked content)
  - `data/eval/golden_20.jsonl` (20 problems, locked content)
  - `scripts/build_eval_sets.py` (reproducible selection script with seed=42)
  - `docs/execution/EVAL_POLICY.md` (no-touch policy, approval, fallback thresholds)
  - Manifest with SHA-256 hashes (embedded in EVAL_POLICY.md or separate `data/eval/MANIFEST.json`)
- **Consumed by**: `#19` (data curation), `#21` (SFT LoRA), `#22` (QLoRA), `#23` (GRPO RL), `#24` (eval comparison), `#25` (submission), `#20` (backup eval)
- **External references cited**: 
  - [plan_v0.2.md](../plan_v0.2.md) Phase 3.1
  - [#17](../issues/17) schema definition
  - [Puzzle-KD Dataset v2](https://huggingface.co/datasets/nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2)

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| MVP (stratified sample + golden-set infer) | 2–3 | Colab Pro or RTX 3080 |
| Alternative path (if base model slow) | 1 | CPU (adjust params) |
| Full polish (docs, CI hash checks, approval flow) | 2 | Local |
| **Total** | **4–6** | Colab or Local |
