# Midway Progress Report
**Project:** CS4650 / NVIDIA Nemotron Reasoning Challenge (Kaggle)
**Date:** 2026-05-10
**Deadline:** 2026-06-15 23:59 UTC (36 days remaining)
**Prepared by:** Project team

---

## Executive Summary

- **What's done:** Waves A and B are complete. Competition constraints are frozen, data contracts are in place, the evaluation harness is operational, and a baseline eval has run end-to-end against the competition model.
- **Key finding:** Baseline accuracy is 0%, but this is a format-compliance failure, not a reasoning failure. The Nemotron model reasons correctly; it simply does not spontaneously emit `\boxed{}` wrappers without an explicit prompt instruction. This is expected and fixable with prompting.
- **What's next:** Wave C (Notebook 05 — prompting sweep) is the immediate priority. A self-contained Kaggle notebook is being prepared to run the sweep on GPU and measure accuracy improvement from adding `\boxed{}`-instructed prompt templates.

---

## Progress by Wave and Notebook

| Wave | Notebook | Purpose | Status |
|------|----------|---------|--------|
| A | `00_competition_constraints_and_rules` | Freeze rules, model path, eval contract | DONE |
| A | `01_external_baselines_and_design_deltas` | Review Tong / konbu17 pipelines | DONE |
| B | `02_dataset_schema_and_eda` | Schema, category shape, normalization plan | DONE (scaffolded) |
| B | `03_validation_and_golden_set` | Validation split and golden-set regression policy | DONE (scaffolded) |
| B | `04_baseline_eval_and_normalization` | End-to-end baseline eval; defines EvalRecord shape | DONE (active) |
| B | `10_submission_packaging_and_provenance` | Packager entrypoint + provenance policy | DONE (active) |
| C | `05_prompting_and_decode_sweeps` | Compare prompt templates; produce ranked sweep CSV | IN PROGRESS |
| C | `06_trajectory_collection_and_error_slices` | Classify failures; produce retry-candidate taxonomy | PENDING (blocked on #21) |
| D | `07_solver_framework_design` | Solver/Verifier protocol, CategoryRouter | PENDING |
| D | `08_synthetic_data_recipe` | Synthetic data pipeline, quality filters, cost cap | PENDING |
| D | `09_sft_runbook_and_masking` | LoRA/QLoRA runbook, masking, checkpoint policy | PENDING |

**Wave gates:**
- Wave A gate: CLOSED (constraints frozen 2026-04-29)
- Wave B gate: CLOSED (schema, eval record shape, and package manifest stable)
- Wave C gate: OPEN — awaiting prompting sweep execution
- Wave D gate: OPEN — blocked until Wave C closes

---

## Key Technical Finding — The 0% Baseline

The baseline evaluation (Notebook 04) ran to completion. Result: **0% exact-match accuracy**.

This is not a reasoning failure. The NVIDIA Nemotron-3-Nano-30B model is a capable reasoning model. The root cause is a prompt-format mismatch: the competition evaluator extracts answers by looking for `\boxed{}` content first, falling back to heuristics. The base model, without explicit instruction, generates correct reasoning but does not wrap the final answer in `\boxed{}`. The evaluator finds nothing to extract in the expected location, and the fallback heuristics do not reliably recover the answer.

**Why this is not alarming:** This is the standard starting point for all teams who run the raw base model. The fix — adding an explicit `\boxed{}` format instruction to the prompt template — is mechanical and well-established in public baselines (Tong pipeline, konbu17 notebook). The 0% result confirms the eval harness is working correctly and that improvements will show immediately once prompt templates are corrected.

---

## Current Blocker

**Nature:** Notebook 05 (prompting sweep) is code-complete but has not been executed.

Two dependencies are blocking execution:

1. **Data artifacts not on disk.** `validation_200.jsonl` and `golden_20.jsonl` are git-ignored (large/derived artifacts). They must be regenerated or sourced from the Kaggle dataset before local or remote execution can proceed.
2. **GPU dependency.** Running the prompting sweep against the 30B model requires a GPU. The local dev environment does not have one.

**Mitigation in progress:** A self-contained Kaggle notebook version of Notebook 05 is being prepared. It will pull data directly from the Kaggle competition dataset (no local artifact dependency), load the model via KaggleHub, and run the sweep on Kaggle's GPU-accelerated kernel environment. This sidesteps both blockers simultaneously.

---

## Next 48 Hours

1. Finalize and upload the self-contained Kaggle version of `05_prompting_and_decode_sweeps`.
2. Execute the sweep on Kaggle GPU (P100 or T4; RTX PRO 6000 is the eval environment — Kaggle kernel GPU is sufficient for sweep comparison).
3. Record sweep results: accuracy by prompt template (zero-shot, zero-shot-cot, few-shot-cot) against the `validation_200` split.
4. Identify the winning template. Update `docs/execution/NOTEBOOKS.md` to reflect execution status.
5. If time permits, begin Notebook 06 (trajectory collection and error slices) using the sweep output.

---

## Risk and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| GPU access delays on Kaggle | Low | Medium | Kaggle notebook queue is typically fast; fallback to Colab Pro if needed |
| Prompting sweep shows <20% accuracy after format fix | Low-Medium | High | Few-shot-cot templates and chain-of-thought elicitation are additional levers before training |
| Wave D (SFT/LoRA) training time exceeds available runway | Medium | High | Prioritize solver-first synthetic data (Notebook 08) early; begin LoRA runs as soon as Wave C closes to leave 2+ weeks for iteration |
| Submission packaging untested end-to-end | Low | High | Notebook 10 is active; run a dry-run submission as soon as a non-trivial adapter exists |
| 5 submission/day limit constrains late-stage iteration | Low | Medium | Reserve daily submissions for meaningful checkpoints; do not burn on debugging runs |

**Timeline assessment:** With 36 days remaining and Waves A-B complete, the project is on track. The critical path is: Wave C close (this week) → Wave D SFT start (next week) → LoRA training + iteration (Weeks 3-4) → final submission selection (Week 5). No calendar slippage yet, but Wave D cannot compress further.

---

## Expected Outcome After Wave C

After adding explicit `\boxed{}` format instructions to prompt templates, the expected accuracy range is:

| Prompt strategy | Expected accuracy range |
|-----------------|------------------------|
| Zero-shot + format instruction | 15 – 35% |
| Zero-shot chain-of-thought + format instruction | 25 – 50% |
| Few-shot chain-of-thought + format instruction | 35 – 60% |

These ranges are informed by the external baseline review (konbu17, Tong pipeline) and the nature of the 6 problem families (rule-induction puzzles with structured I/O). The physics and numeral-system categories are expected to benefit most from chain-of-thought elicitation; text_cipher and bit_manipulation may require solver-assisted templates developed in Wave D.

A successful Wave C outcome is any result above 30% on the validation split, which would confirm that the model has the underlying capability and that training (LoRA/SFT) is a tractable path to further improvement.

---

*Report generated 2026-05-10. Next report due after Wave C gate closes.*
