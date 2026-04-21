# Notebook 08: Synthetic Data Recipe (Teacher, Filter, Provenance)

**Parent Issue**: `#24`
**Plan Phase**: Phase 5 (Synthetic Data Generation) + Phase 3 (curation integration)
**Scaffold**: `notebooks/08_synthetic_data_recipe.ipynb`
**Status**: `planned`
**Dependencies (upstream)**: `#15` (review harness), `#22` (error slices to target), `#23` (solver design)
**Consumers (downstream)**: `#25` (SFT runbook)

---

## 1. Objective

Write a repeatable recipe that specifies teacher model(s), filter rules, and provenance metadata for generating synthetic reasoning data from error slices identified in #22. The notebook produces a `configs/synthetic_recipe.yaml`, a `src/data/synthetic.py` stub, and documentation of the generation pipeline so that a 500-sample pilot run can be executed with transparent cost estimation and answer verification.

## 2. Why It Matters

- **Competition**: Synthetic data targeted at error slices from test set weaknesses directly addresses leaderboard gaps identified in Phase 3 curation.
- **Capstone learning**: Distillation from stronger teacher models is a core multi-approach strategy; this notebook operationalizes it with cost and quality controls.
- **Downstream**: Notebook 09 SFT data loader expects synthetic samples in canonical schema with provenance fields intact for deduplication and filtering.

## 3. Strategy — How We Aim To Accomplish It

1. **Teacher selection** (2 cells): List candidate teachers (DeepSeek-R1 self-hosted on HPC; optionally Qwen3-235B API for small batches), estimate inference cost per 1K samples, compare latency. Select primary (DeepSeek-R1 open-weight on HF) and optional fallback (API if quota allows).
2. **Problem sourcing** (1 cell): Load error slices from notebook 06 (`data/errors/category_slices.jsonl`), cross-check coverage with public math benchmarks (AIME, MATH dataset), deduplicate, sample 500 problems balanced by difficulty.
3. **Generation wrapper** (2 cells): Implement `generate_with_reasoning()` function using `transformers.AutoModelForCausalLM` or inference API; format prompts with Nemotron chat template (`<|im_start|>user`, `<|im_end|>`, `<|im_start|>assistant`); enforce `enable_thinking=True` or equivalent reasoning mode; log per-sample cost and latency.
4. **Verification pipeline** (2 cells): Implement `verify_answer()` function that programmatically checks final `\boxed{...}` answer: math via `sympy` or numeric comparison, logic via exact-match. Record pass/fail per sample.
5. **Provenance tagging** (1 cell): Attach metadata (teacher_id, teacher_version, prompt_hash, generation_date, verifier_pass) to every sample in JSONL output.
6. **Deduplication & integration** (1 cell): Check for overlap with curated data from notebook 03, remove exact duplicates, format output in canonical schema, verify schema compliance against notebook 02 normalizer.

## 4. MVP (Minimum Viable Notebook)

**Scope**: 500-sample pilot run with cost cap $20, ≥90% verifier pass rate.

- **Inputs**:
  - Error slices from `data/errors/category_slices.jsonl` (Phase 3, notebook 06)
  - HPC access to DeepSeek-R1 or HF Inference API key (optional fallback)
  - Base model tokenizer: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

- **Cells**:
  1. Load problem corpus and sample 500 by category
  2. Initialize teacher model (DeepSeek-R1 via HF transformers)
  3. Define generation + reasoning function with cost tracking
  4. Generate 500 synthetic responses with error handling and checkpoint every 50 samples
  5. Verify answers programmatically; report pass rate and cost per sample
  6. Tag with provenance; deduplicate against curated set
  7. Save to `data/synthetic/pilot_<YYYYMMDD>.jsonl`
  8. Validate output schema; print summary stats

- **Outputs**:
  - `data/synthetic/pilot_<date>.jsonl` — 500 samples with provenance
  - `configs/synthetic_recipe.yaml` — teacher, prompt template, verify rules, cost ceiling
  - `src/data/synthetic.py` — stub with `SyntheticDataGenerator` class
  - `docs/architecture/SYNTHETIC_PROVENANCE.md` — schema and pipeline docs

- **Verification**: Print assertions:
  - `assert len(samples) == 500`
  - `assert pass_rate >= 0.90`
  - `assert total_cost <= 20.0`
  - `assert all(s["teacher_id"] and s["verifier_pass"] is not None for s in samples)`

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: DeepSeek-R1 or compatible model available (HF API or self-hosted HPC); error slices from #22 in `data/errors/category_slices.jsonl`; cost budget $20.
- **Action**: Generate 500 samples from error slices with `enable_thinking=True`; verify answers via `sympy` or exact-match; filter to pass-only; attach provenance fields.
- **Expected**: `pass_rate >= 0.90`, `total_cost <= $20`, `all 5 provenance fields present per sample`, output valid JSON Lines, schema matches notebook 02 canonical.

### 5.2 Alternative / Fallback

- **Setup**: Teacher inference quota exhausted or HPC unavailable; SYNTHETIC-1 (1.4M verified R1 traces) accessible on HF.
- **Action**: Subsample SYNTHETIC-1 by error slice categories (filter to problems matching #22 error slice topics); apply same verification pipeline to verify provenance integrity; deduplicate against curated data.
- **Expected**: ≥500 samples sourced from SYNTHETIC-1, `pass_rate >= 0.90` (likely >99% since pre-verified), zero generation cost, output schema matches primary.

### 5.3 Regression Guardrails

- **Schema compliance**: Output must load via notebook 02 normalizer without errors.
- **No data leakage**: No sample from pilot set appears in validation_200 or golden_20 (Phase 3).
- **Provenance immutability**: Once written to JSONL, provenance fields are never modified (new samples appended to new file, not in-place edits).
- **Cost tracking**: Per-sample cost logged; any run exceeding $20 cap is rolled back and reported to orchestrator.

## 6. Success Criteria (Done When)

- [ ] Teacher model identified and cost estimate ≤$0.04/sample (500×$0.04=$20)
- [ ] Generation function implemented with deterministic seeding for reproducibility
- [ ] Verification function deployed; ≥90% pass rate achieved on 500-sample pilot
- [ ] All 5 provenance fields (teacher_id, teacher_version, prompt_hash, generation_date, verifier_pass) present on every sample
- [ ] `configs/synthetic_recipe.yaml` checked in with teacher config, prompt template, and cost ceiling
- [ ] `src/data/synthetic.py` stub with `SyntheticDataGenerator` class skeleton checked in
- [ ] `data/synthetic/pilot_<date>.jsonl` saved with 500 samples and summary logged to WandB
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`

## 7. Risks & Open Questions

- **Risk**: Cost blowout (inference or API calls exceed budget). | **Mitigation**: Estimate cost per sample before running; track cumulative cost in loop; break early if budget exceeded. Use open-weight DeepSeek-R1 on HPC (free GPU time) as primary to minimize API calls.

- **Risk**: Teacher-student mismatch (4B Nemotron cannot replicate reasoning style of 70B+ teacher). | **Mitigation**: Verify that distilled traces pass verification on final model. Include pilot_<date>.jsonl evaluation in Phase 7 comparison to measure downstream SFT gain.

- **Risk**: License conflict on teacher outputs (if using restricted model). | **Mitigation**: Prefer DeepSeek-R1 (MIT license) or SYNTHETIC-1 (open, pre-verified). Document license for each teacher in `configs/synthetic_recipe.yaml`.

- **Risk**: Synthetic data leakage into golden or validation sets. | **Mitigation**: Cross-check problem IDs/hashes against `data/eval/validation_200.jsonl` and `data/eval/golden_20.jsonl` in Phase 3; reject any matching problems before generation.

- **Open question**: Can HPC cluster inference run DeepSeek-R1 in time budget? | **Who answers**: HPC admin or parallel Phase 4 run confirms feasibility.

## 8. Artifacts & Handoff

- **Produces**:
  - `data/synthetic/pilot_<YYYYMMDD>.jsonl` — 500 verified synthetic samples with provenance
  - `configs/synthetic_recipe.yaml` — teacher selection, prompt template, verification rules, cost cap
  - `src/data/synthetic.py` — `SyntheticDataGenerator` class (stub or minimal implementation)
  - `docs/architecture/SYNTHETIC_PROVENANCE.md` — schema definition, provenance field rationale, integration guide

- **Consumed by**: 
  - Notebook 09 (SFT data loader) — reads `pilot_<date>.jsonl`, deduplicates, merges with curated data
  - Phase 7 evaluation — compares SFT trained on curated-only vs curated+synthetic
  - Notebook 04 (normalizer) — validates schema of all synthetic samples

- **External references cited**:
  - NVIDIA blog: https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/
  - DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
  - SYNTHETIC-1: https://www.primeintellect.ai/blog/synthetic-1
  - Plan v0.2 Phase 5: [Phase 5](plan_v0.2.md#phase-5-synthetic-data-generation)

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| Teacher selection + cost estimation | 1.5 | Local CPU |
| Problem sourcing and sampling | 1 | Local CPU |
| Generation wrapper and inference setup | 2 | HPC (optional) or Colab |
| Verification pipeline implementation | 1.5 | Local CPU |
| Pilot run (500 samples) | 3 | HPC or Colab |
| Provenance tagging and deduplication | 1 | Local CPU |
| Artifact finalization and handoff docs | 1 | Local CPU |
| **MVP total** | **11** | Mixed |
| Alternative path (SYNTHETIC-1 subsample) | 2 | Local CPU (download only) |
| Full polish (>500 samples, API fallback) | 6 | HPC + API budget |
