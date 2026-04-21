# Notebook 04: Baseline Evaluation and Answer Normalization

**Parent Issue**: `#19`
**Plan Phase**: Phase 1.2 (Smoke Test) + Phase 7.2 (Evaluation Protocol)
**Scaffold**: `notebooks/04_baseline_eval_and_normalization.ipynb`
**Status**: `planned`
**Dependencies (upstream)**: `#14`, `#17`, `#18`
**Consumers (downstream)**: `#21`, `#22`, `#20`, `#25`

---

## 1. Objective

Lock the answer-normalization contract and produce the first reproducible baseline on validation_200 + golden_20 sets. Publish `src/evaluation/normalize.py`, `src/evaluation/harness.py`, and an initial baseline score card with deterministic evaluation (temperature=0, n_runs=3, seeds {42, 43, 44}) to establish the metric that all downstream phases measure against.

## 2. Why It Matters

- **Competition leaderboard**: Mismatch between local scorer and Kaggle scorer will invalidate all subsequent phase results; this notebook locks the definition first.
- **Capstone learning**: Writing a deterministic eval harness is a core system design skill; demonstrates reproducibility.
- **Downstream blockers**: Notebooks `#20`, `#21`, `#22`, `#25` (prompting, SFT, RL, final submission) all depend on the `extract_boxed_answer()`, `normalize()`, and `equal()` functions defined here.

## 3. Strategy — How We Aim To Accomplish It

1. **Implement normalization functions** (`extract_boxed_answer`, `normalize`, `equal`) in `src/evaluation/normalize.py` using the regex from plan_v0.2: `r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'` (handles one level of nested braces).
2. **Unit test normalization** in `tests/test_normalize.py` with test cases: boxed extraction, edge cases (whitespace, nested braces, missing boxes), fallback to plain EM if #14 specifies strict exact-match.
3. **Run Phase 1.2 smoke test**: Load base model, generate answer to "What is 2^10 mod 7?", assert `extract_boxed_answer(output) == "2"` passes inline.
4. **Baseline on golden_20**: Evaluate base model on golden set (20 problems) with deterministic generation (temperature=0, seed=42); must achieve 100% accuracy or trigger investigation of golden_20 data quality.
5. **Baseline on validation_200**: Run 3 deterministic evaluation passes (seeds 42, 43, 44) on validation_200; compute mean accuracy and std; record per-problem predictions to `experiments/baseline_<date>.json`.
6. **Log to WandB** with `config={approach:"baseline", enable_thinking:True, temperature:0}` and artifacts (normalize.py, harness.py, baseline scores).

## 4. MVP (Minimum Viable Notebook)

**Inputs**: base model `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (downloaded), validation_200 and golden_20 (from Phase 3), tokenizer config

**Cells**:
1. Environment setup, imports, seed initialization
2. Load model + tokenizer, smoke test smoke test (2^10 mod 7)
3. Implement + test `extract_boxed_answer`, `normalize`, `equal` functions
4. Baseline evaluation on golden_20 (n_runs=1, temperature=0)
5. Baseline evaluation on validation_200 (n_runs=3, seeds {42, 43, 44}, temperature=0)
6. Save baseline scores to `experiments/baseline_<date>.json`
7. Log results to WandB, print summary stats (mean±std accuracy)

**Outputs**:
- `src/evaluation/normalize.py` (extract, normalize, equal functions)
- `src/evaluation/harness.py` (deterministic eval harness)
- `tests/test_normalize.py` (unit tests)
- `experiments/baseline_<date>.json` (per-seed per-problem predictions and scores)
- WandB run link with baseline metrics

**Verification**: Smoke test assertion passes inline; golden_20 accuracy printed and logged; validation_200 mean±std accuracy reported to WandB.

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: Base model loaded, golden_20 and validation_200 in `data/eval/`, seeds {42, 43, 44} set
- **Action**: Run smoke test (2^10 mod 7); evaluate on golden_20 with temperature=0; run 3-seed deterministic pass on validation_200
- **Expected**: 
  - Smoke test prints `assert extract_boxed_answer("...\\boxed{2}.") == "2"` → PASS
  - Golden_20 accuracy = 100% (or investigation required)
  - Validation_200 mean±std logged to WandB (e.g., 0.72±0.01)
  - `experiments/baseline_<date>.json` contains per-problem predictions with scores

### 5.2 Alternative / Fallback

If #14 (competition rules verification) reveals strict exact-match without boxed extraction:
- **Setup**: Confirm competition uses plain exact-match (no special parsing)
- **Action**: Add `strict_em` branch in normalize.py; keep boxed path alongside for ablation
- **Expected**: Code includes conditional `if competition_uses_plain_em: use_strict_em_path()` else `use_boxed_path()`; all downstream code branches on this switch; documentation updated

### 5.3 Regression Guardrails

- **Determinism check**: Run baseline twice with same seed; assert identical per-problem scores
- **Golden set regression**: Any phase that modifies eval code must re-run golden_20 and verify still 100%
- **Format correctness**: Spot-check 5 random predictions from baseline; verify all outputs contain valid `\boxed{}` syntax (or plain answers if fallback active)

## 6. Success Criteria (Done When)

- [ ] `src/evaluation/normalize.py` implements `extract_boxed_answer`, `normalize`, `equal` with docstrings
- [ ] `src/evaluation/harness.py` implements deterministic eval harness with `evaluate(model, tokenizer, eval_set, n_runs, temperature)` signature
- [ ] `tests/test_normalize.py` includes unit tests for extract (boxed, nested, missing), normalization edge cases, determinism
- [ ] Smoke test (2^10 mod 7) passes inline with assertion
- [ ] Golden_20 baseline = 100%; if not, document reason or rebuild golden_20
- [ ] Validation_200 baseline (3 seeds) computed; mean±std logged to WandB
- [ ] `experiments/baseline_<date>.json` committed with per-problem results
- [ ] WandB run tagged `baseline` and linked in docs/execution/NOTEBOOKS.md
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`
- [ ] WandB run (if applicable) tagged and linked

## 7. Risks & Open Questions

- **Risk**: Regex for nested braces may fail on rare edge cases (triple nesting, malformed boxes) | **Mitigation**: Add fallback to strict EM for unparseable outputs; log unparseable rates per eval run
- **Risk**: temperature=0 may not be deterministic on Mamba-2 kernels (reported in some systems) | **Mitigation**: Record any nondeterminism (>0.1% variance between re-runs); if detected, use fixed random seed at generation kernel level if supported
- **Open question**: Does competition use strict exact-match or boxed extraction? | **Who answers**: #14 (competition rules verification) must close before finalizing normalize.py; keep both paths configurable
- **Risk**: golden_20 data quality issues may cause baseline != 100% | **Mitigation**: If baseline < 100% on golden_20, investigate and either rebuild golden_20 with lower confidence threshold or mark specific problems as "expected failures" with justification

## 8. Artifacts & Handoff

- **Produces**:
  - `src/evaluation/normalize.py` — extract_boxed_answer, normalize, equal functions
  - `src/evaluation/harness.py` — deterministic eval harness with n_runs, seed control
  - `tests/test_normalize.py` — unit test suite for normalization
  - `experiments/baseline_<date>.json` — per-seed per-problem predictions and aggregated scores
  - WandB run link (tagged `baseline`, config `{approach: "baseline", enable_thinking: True, temperature: 0}`)
- **Consumed by**: 
  - `#20` (prompting strategies — uses harness to measure deltas vs baseline)
  - `#21`, `#22` (SFT, GRPO — uses golden_20 for regression gates)
  - `#25` (final submission — compares final adapter vs baseline)
- **External references cited**: 
  - Plan v0.2 Phase 1.2, Phase 7.2
  - [NVIDIA Nemotron Submission Demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
  - Competition data format (once verified in #14)

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| Implement normalize.py + harness.py | 2 | Local (CPU) |
| Unit tests + smoke test | 1 | Local (CPU) |
| Baseline eval (golden_20 + validation_200 × 3 runs) | 3-4 | Colab Pro or RTX 3080 |
| Log to WandB + document | 0.5 | Local |
| **MVP total** | **6.5–7.5** | **Colab Pro / RTX 3080** |
| Fallback (strict EM branch if #14 changes) | +1 | Local |
| Full polish (edge case handling, nondeterminism mitigation) | +1 | Colab |
