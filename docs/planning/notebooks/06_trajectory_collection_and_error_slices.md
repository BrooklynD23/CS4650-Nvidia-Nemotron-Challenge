# Notebook 06: Trajectory Collection and Error Slices

**Parent Issue**: `#22`
**Plan Phase**: Supports Phase 3.3 (Quality Filtering) and Phase 5 (Synthetic Data targeting)
**Scaffold**: `notebooks/06_trajectory_collection_and_error_slices.ipynb`
**Status**: `scaffolded`
**Dependencies (upstream)**: `#19` (eval harness), `#21` (prompting sweep results)
**Consumers (downstream)**: `#23` (solver design), `#24` (synthetic data recipe), `#25` (SFT masking)

---

## 1. Objective

Collect reasoning trajectories from the base model on `validation_200` plus a held-out 1k stratified slice from training data, cluster failures into error types (arithmetic slip, format miss, hallucinated reasoning, refusal, truncation), and export both a JSONL trajectory dataset and a markdown error-slice taxonomy report so downstream notebooks can target their data curation and loss masking at the dominant error modes.

## 2. Why It Matters

- **Leaderboard**: Understanding where Nemotron-3-Nano-4B fails reveals highest-ROI engineering: synthetic data generation (#24) can target the top 3 failure modes, and masked SFT loss (#25) can deprioritize already-correct reasoning traces.
- **Capstone learning**: Error classification reinforces model debugging discipline and teaches learners to prioritize by impact rather than guessing.
- **Downstream dependencies**: Notebooks #23, #24, and #25 all consume the error taxonomy to shape their work; without this inventory, they lack concrete failure-mode targets.

## 3. Strategy — How We Aim To Accomplish It

1. **Load base model** (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) with best prompting config from `#21` (temperature, top_p, system prompt).
2. **Run inference** on `validation_200.jsonl` + a stratified 1k-sample slice from `data/processed/training_curated.jsonl` (held out from Phase 3.2 curation) with `enable_thinking=True`.
3. **Collect raw outputs** and extract structured fields: `input`, `raw_output`, `extracted_answer`, `correct` (bool), `token_length`.
4. **Classify errors** using heuristic patterns (regex for format misses, truncation detection, empty reasoning) + optional LLM-judge pass to label ambiguous cases.
5. **Group trajectories** by error type: `arithmetic_slip`, `format_miss`, `hallucinated_reasoning`, `refusal`, `truncation`, `correct`.
6. **Cluster by category × error-type** (compute accuracy breakdowns per benchmark category).
7. **Export artifacts**: `data/analysis/trajectories_<date>.jsonl` (all trajectories), `docs/analysis/error_slices.md` (taxonomy + stats), `data/analysis/retry_candidates.jsonl` (failures ranked by recoverability).

## 4. MVP (Minimum Viable Notebook)

**Scope**: End-to-end error collection + classification in one focused session (~4 hours).

### Inputs
- `data/eval/validation_200.jsonl` (from Phase 3.1)
- `data/processed/training_curated.jsonl` (curated training data; reserve 1k examples)
- Best prompting config from `#21` (YAML or dict)
- Base model checkpoint: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

### Cells (numbered)
1. **Setup**: Load env, model, tokenizer, best config from `#21`.
2. **Data prep**: Load validation_200 + sample 1k from training (stratified by category).
3. **Inference loop**: Generate with `enable_thinking=True`, collect raw output, extract `<think>...</think>` and final answer.
4. **Field extraction**: Parse `extracted_answer` from `\boxed{}`, compute `correct` against ground truth.
5. **Error classification (heuristic)**: Regex patterns for format, truncation, empty reasoning.
6. **Optional LLM-judge**: If budget allows, classify 10% of ambiguous cases with a judge prompt.
7. **Clustering**: Group by error type, compute per-category accuracy.
8. **Export**: Write `trajectories.jsonl`, markdown report, retry candidates.
9. **Verification**: Assert schema, count checks, top-3 error types have ≥5 examples each.

### Outputs
- **Primary**: `data/analysis/trajectories_<date>.jsonl` — one row per problem: `{input, raw_output, extracted_answer, correct, error_type, token_length}`
- **Report**: `docs/analysis/error_slices.md` — category-level accuracy, top-3 error modes with examples, per-type failure count
- **Retry set**: `data/analysis/retry_candidates.jsonl` — errors ranked by "recoverability" (format-miss > arithmetic slip > hallucination)

### Verification
- Print: "✓ Trajectories JSONL: N rows, schema valid"
- Assert: `len(correct_rows) + len(incorrect_rows) == N`
- Assert: Top 3 error types each have ≥5 instances
- Print: Error type distribution as JSON

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: Base model loads without error; `validation_200.jsonl` has 200 rows; curated training has ≥1k rows.
- **Action**: Run inference on full validation_200 + 1k training slice; classify all outputs.
- **Expected**:
  - JSONL has 1,200 rows (200 + 1k).
  - Every row has keys: `input`, `raw_output`, `extracted_answer`, `correct`, `error_type`, `token_length`.
  - `correct` is boolean; `error_type` is one of: `correct`, `arithmetic_slip`, `format_miss`, `hallucinated_reasoning`, `refusal`, `truncation`.
  - Printed summary: "Correct: 752/1200 (62.7%) | Incorrect: 448/1200 (37.3%)"
  - Top-3 error types: e.g., format_miss (156), arithmetic_slip (142), hallucinated_reasoning (95) — all ≥5.

### 5.2 Alternative / Fallback

- **Setup**: LLM-judge API rate limit hit or budget exhausted; fall back to heuristic-only classification.
- **Action**: Skip LLM-judge cell; re-run classification using regex/heuristic alone.
- **Expected**:
  - Heuristic classifier still produces valid error taxonomy.
  - Ambiguous cases labeled as `ambiguous_classification` instead of being delegated.
  - Accuracy distribution changes slightly but ≥3 named error types with ≥5 examples each remain.

### 5.3 Regression Guardrails

- **Run on same seeds + model + validation_200**: Error taxonomy counts must match prior run within ±1 per bucket (stochasticity tolerance).
- **Golden set not leaked**: Verify `data/eval/golden_20.jsonl` is NOT in trajectories (separate file).
- **No golden-set data in training slice**: Confirm 1k sample is drawn from `training_curated.jsonl`, not golden or val sets.

## 6. Success Criteria (Done When)

- [ ] Base model inference completes without GPU OOM on RTX 3080 (token_length ≤ 16k per sample).
- [ ] All 1,200 trajectories classified into one of 6 error types.
- [ ] Top-3 error types each have ≥5 examples with representative samples shown in markdown report.
- [ ] `data/analysis/trajectories_<date>.jsonl` written and schema-valid (test load with pandas).
- [ ] `docs/analysis/error_slices.md` generated with category-level accuracy breakdown and error-type histograms.
- [ ] `data/analysis/retry_candidates.jsonl` ranked and ready for #24 (synthetic data recipe).
- [ ] Regression test passes: re-run on same validation_200 + same model → error counts within ±1.
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`.

## 7. Risks & Open Questions

- **Risk**: Error-type taxonomy drifts per session (stochasticity in model outputs). | **Mitigation**: Fix `temperature=0.0` for deterministic classification, or run 3x and vote on error type.
- **Risk**: LLM-judge introduces bias in labeling ambiguous cases. | **Mitigation**: Use heuristic-only fallback; compare judge labels on 5% hold-out.
- **Risk**: Golden-set content leaks into training_1k slice. | **Mitigation**: Verify data lineage; golden_20 reserved BEFORE any sampling.
- **Risk**: Token length 16384 exceeds RTX 3080 VRAM; OOM mid-run. | **Mitigation**: Reduce batch_size or max_new_tokens; estimate memory per sample.
- **Open question**: How to score "recoverability" of errors for retry ranking? | **Answer**: Format-miss (fixable with post-processing) > arithmetic-slip (solvable with better data/RL) > hallucination (requires major retraining).

## 8. Artifacts & Handoff

### Produces
- `data/analysis/trajectories_2026-04-20.jsonl` — 1,200 trajectories with error labels and metadata
- `docs/analysis/error_slices.md` — Markdown report with category-level accuracy, error taxonomy, top-3 modes, example failures
- `data/analysis/retry_candidates.jsonl` — Subset of incorrect trajectories ranked by recoverability for #24

### Consumed by
- **Notebook #23** (solver design): Uses error taxonomy to design recovery strategies (e.g., format-correction post-processor for format_miss).
- **Notebook #24** (synthetic data recipe): Targets generation at top-3 error modes; uses retry_candidates to know which problem types need synthetic examples.
- **Notebook #25** (SFT masking): Uses error_type to mask loss on already-correct reasoning traces, boosting gradient signal on failures.

### External references cited
- [NVIDIA Nemotron Blog](https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/) — baseline prompting config
- `docs/planning/plan_v0.2.md` — Phase 3.3, Phase 5 context
- `notebooks/05_prompting_and_decode_sweeps.ipynb` (#21) — best prompting parameters

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| Setup + data load | 0.5 | RTX 3080 |
| Inference loop (1,200 samples) | 2.5 | RTX 3080 (batch inference) |
| Error classification + heuristic | 0.5 | CPU |
| Optional LLM-judge (10% budget) | 0.5 | API + CPU |
| Export + report generation | 0.3 | CPU |
| Regression testing + polish | 0.2 | RTX 3080 |
| **Total MVP** | **4.0** | **RTX 3080** |
| Alternative (heuristic only) | **3.5** | **RTX 3080** |
