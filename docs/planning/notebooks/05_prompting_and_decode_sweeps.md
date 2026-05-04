# Notebook 05: Prompting and Decode Sweeps

**Parent Issue**: `#21`
**Plan Phase**: Phase 2 (Prompting Strategies — zero compute cost vs training)
**Scaffold**: `notebooks/05_prompting_and_decode_sweeps.ipynb`
**Status**: `planned`
**Dependencies (upstream)**: `#19` (eval harness + baseline)
**Consumers (downstream)**: `#22` (trajectories/failure slices), `#25` (informs SFT data bias)

---

## 1. Objective

Measure accuracy deltas from prompting strategies and decode parameter changes vs the Phase 1 baseline, using the validation_200 set. Deliver a comparison table of {strategy × decode config × accuracy mean±std × compute cost}, with statistically significant deltas flagged. No literature numbers are validated — all sweeps are measured on Nemotron-3-Nano-4B.

## 2. Why It Matters

- **Leaderboard**: Prompting gains compound with training; identifying +3% improvements here cuts training data requirements.
- **Capstone learning**: Zero-cost exploration teaches prompt design patterns before investing compute in LoRA.
- **Downstream**: Findings inform whether Phase 4 SFT should use reasoning-on vs reasoning-off training data (`#25`). Best strategy feeds `#22` (error analysis).

## 3. Strategy — How We Aim To Accomplish It

1. **Reuse eval harness from `#19`**: Load baseline predictions and ground truth from `validation_200.jsonl` and `golden_20.jsonl`.
2. **Sparse grid sweep** (don't run full factorial): Start with zero-shot CoT + few-shot CoT × {temp 0.6, 1.0} × {top_p 0.9, 0.95} = 8 seeds. Run 3 seeds each. Early exit if NVIDIA defaults (temp=1.0, top_p=0.95) outperform.
3. **Best-of-N sampling**: Test N={8, 32} by resampling from the validation_200 pool (3 runs each). Compute majority vote (maj@N) from single sampled pool.
4. **Log to WandB**: Tag each run with strategy name, seed, decode params. Enable offline mode for Colab.
5. **Rank strategies**: Compare via paired t-test; report winners with delta > 2σ vs baseline.

## 4. MVP (Minimum Viable Notebook)

### Inputs
- `data/eval/validation_200.jsonl` (200 problems with ground truth, from `#19`)
- `data/eval/golden_20.jsonl` (regression guardrail)
- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (BF16, trust_remote_code=True)

### Cells
1. **Setup**: Load model, tokenizer, eval harness (20 lines)
2. **Zero-shot CoT**: "Let's think step by step" + extraction (30 lines)
3. **Few-shot CoT**: 3-5 solved examples injected (40 lines)
4. **Decode sweep**: For each {strategy, temp, top_p} tuple, run N=3 evals, collect accuracy (50 lines)
5. **Best-of-N**: Resample N=8, N=32 from one seed pool, majority vote (40 lines)
6. **Results table**: DataFrame with columns [Strategy, Temp, Top_p, Accuracy Mean, Accuracy Std, Delta vs Baseline, Sig?] (30 lines)
7. **WandB logging**: Log final table + summary to project (20 lines)

### Outputs
- `experiments/prompting_sweep_<date>.csv` (one row per {strategy, decode config, seed})
- WandB run(s) tagged `prompting-sweep`, `phase-2`
- `docs/analysis/prompting_findings.md` (markdown summary: best strategy, recommendations, risks)

### Verification
Assertion: Table has ≥8 rows (sparse sweep completed). Each row has `accuracy_mean`, `accuracy_std`. Exactly one row beats baseline with delta > 2σ.

## 5. Test Cases

### 5.1 Primary (MVP path)

**Setup**: validation_200 loaded, 3 CPU seeds, model in BF16 on RTX 3080 (10GB).

**Action**: Run sparse sweep: zero-shot CoT + few-shot CoT, 2 temperatures, 2 top_p values, 3 runs per config. Majority vote N=8 on one sampled pool. Log to WandB.

**Expected**: 
- Table ≥8 rows (one per config, averaged over 3 seeds).
- Each row: accuracy_mean ∈ [0, 1], accuracy_std ≤ 0.05.
- At least one row has delta > baseline + 2σ (statistically stronger).
- Golden_20 accuracy ≥ 100% under best strategy (no regression).

### 5.2 Alternative / Fallback

**Setup**: If inference too slow on RTX 3080 (>15 min per run), downscale validation_200 to stratified random 100 subsample.

**Action**: Run same sweep on subset. Mark results as "noisy — to be re-run on Colab A100". Drop N=64 best-of-N; keep N=8, N=32.

**Expected**: Table with same structure. Accuracy std may increase (±0.05 to ±0.08). No golden_20 regression.

### 5.3 Regression Guardrails

- **Golden set**: Base model + best prompting strategy → 100% on golden_20 (all 20 correct under best config). Fail if <100%.
- **Schema**: CSV has columns [strategy, temperature, top_p, seed, accuracy, elapsed_seconds]. No NaN values.
- **Baseline**: Phase 1 baseline (from `#19`) accuracy on validation_200 recorded; sweep results compare against it.

## 6. Success Criteria (Done When)

- [ ] Sparse sweep completed: ≥8 unique {strategy, temp, top_p} configs × 3 seeds each.
- [ ] Best-of-N experiments (N=8, N=32) run with majority voting.
- [ ] Results table generated with mean±std, delta vs baseline, statistical significance flagged.
- [ ] Golden set validation: ≥100% under best strategy (no regression from prompting-only changes).
- [ ] Early exit check: If NVIDIA defaults (temp=1.0, top_p=0.95) beat all others, note as "convergence" and skip further sweeps.
- [ ] WandB runs tagged with phase, strategy, seed; offline mode confirmed for Colab.
- [ ] `experiments/prompting_sweep_<date>.csv` committed.
- [ ] `docs/analysis/prompting_findings.md` written (strategy ranking, compute cost, risks).
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`.

## 7. Risks & Open Questions

- **Risk**: Overfitting to validation_200 by cherry-picking best-of-all-runs. | **Mitigation**: Report mean±std across 3 seeds; use paired t-test for significance; reserve golden_20 for regression only.
- **Risk**: Inference latency balloons with best-of-N (N=64). | **Mitigation**: Start with N=8, N=32; measure wall-clock time per problem; cap total experiment to 4 GPU-hours.
- **Risk**: NVIDIA recommended decode (temp=1.0, top_p=0.95) is globally optimal, rendering sweeps low-value. | **Mitigation**: Define early exit criterion: if no config beats baseline by >2σ after sparse grid, mark as "converged" and stop.
- **Open question**: Does few-shot CoT with 3 vs 5 examples differ significantly? | **Who answers**: Determined in Phase 2.2 notebook iteration; defer if time-constrained.

## 8. Artifacts & Handoff

**Produces**:
- `experiments/prompting_sweep_<date>.csv` — CSV with columns [strategy, temperature, top_p, seed, accuracy_mean, accuracy_std, elapsed_seconds, delta_vs_baseline, significant].
- WandB project runs tagged `prompting-sweep`, `phase-2`, with summary plots (accuracy vs decode param).
- `docs/analysis/prompting_findings.md` — Markdown report: best strategy, compute cost, recommended next steps, failure modes.

**Consumed by**: 
- Notebook `#22` (failure slices) — uses best prompting strategy for error analysis.
- Notebook `#25` (data bias) — findings about reasoning-on vs reasoning-off inform training data composition.

**External references cited**:
- plan_v0.2.md Phase 2.2-2.3 (sweep matrix definition).
- Notebook `#19` (eval harness, baseline accuracy, golden_20).
- DeepSeek-R1 paper (literature numbers, not validated).

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| Setup + zero-shot CoT | 1 | RTX 3080 or Colab |
| Few-shot CoT iteration | 1 | RTX 3080 or Colab |
| Decode sweep (sparse grid, 3 seeds) | 2 | RTX 3080 or Colab |
| Best-of-N (N=8, N=32, 3 seeds) | 1.5 | RTX 3080 or Colab |
| Results table + WandB logging | 0.5 | Local CPU |
| Findings doc + early exit decision | 0.5 | Local CPU |
| **MVP total** | **6.5** | **RTX 3080 (or Colab if OOM)** |
| Alternative path (downscaled val_100) | 3 | RTX 3080 local |
| Full sweep (all 48 combos, 3 seeds) | 12+ | Colab A100 (out of MVP scope) |
