# Pending Architecture Review

**Triggered by:** 2ae1253 — feat(#21, #13): implement prompt and decode sweep workflow
**Date:** 2026-04-21
**Agent:** architecture-sync

## Proposed Change

**Target file:** `docs/architecture/ARCHITECTURE.md`

In Section 5 (Evaluation Layer), add a note clarifying that prompt sweep runs introduce a pre-scoring extraction step (`final-answer-line-v1`) that is applied before the `strip_v1` normalizer, and that this step is part of the evaluation contract when comparing sweep configurations against the baseline:

```
### Output Parser Contract (Prompt Sweeps)

When `output_parser: final-answer-line-v1` is set in a sweep run's `decode_config`,
the raw model generation is passed through `extract_final_answer()` before the
`strip_v1` normalizer sees it. This extraction step selects the last non-empty line
of the generation (stripping any leading "final answer:" or "answer:" label).

Baseline runs from notebook 04 do not use this extraction step. Delta comparisons in
`docs/analysis/prompting_findings.md` are therefore comparisons of
(last-line extraction + strip_v1) vs. (strip_v1 only). This difference must be
acknowledged in any promotion decision.
```

## Why This Is Flagged as Critical

**Trigger: Change to evaluation contract (what counts as a correct answer)**

`src/evaluation/prompt_sweeps.py` introduces `extract_final_answer()` as a
pre-scoring step applied in the notebook predictor before calling `run_baseline_eval()`.
The function selects the last non-empty line of the raw generation and optionally
strips a leading "final answer:" or "answer:" label.

This is a new answer extraction step that is absent from the baseline evaluation
in notebook 04. The baseline passes raw model output directly to `run_baseline_eval()`,
which applies only `strip_v1` normalization before exact-match scoring.

`ARCHITECTURE.md` Section 2 (Prompting + Formatting Layer) states:

> "Output normalization: exact-match string normalization rules for evaluation"

and Section 5 (Evaluation Layer) does not describe any pre-normalizer extraction step.

The `validate_baseline_compatibility()` function in `src/evaluation/prompt_sweeps.py`
enforces `model_id` and `normalizer_id` matching against the baseline, but does not
check whether the baseline used the same `output_parser`. Consequently, sweep accuracy
deltas in `experiments/prompting_sweep_<date>.csv` reflect a mixed comparison:
sweep runs use `final-answer-line-v1 + strip_v1`; the baseline uses `strip_v1` alone.

This affects what "a correct answer" means across the comparison and should be
explicitly acknowledged in `ARCHITECTURE.md` before the promoted config from
notebook 05 informs downstream decisions (trajectory collection in notebook 06,
SFT data curation in notebook 08).

File references:
- `src/evaluation/prompt_sweeps.py` lines 840-851 (DEFAULT_STRATEGIES, decode defaults)
- `notebooks/05_prompting_and_decode_sweeps.ipynb` cell defining `extract_final_answer()`
  and `make_predictor()`
- `docs/architecture/ARCHITECTURE.md` Section 2 and Section 5
- `docs/planning/plan_v0.2.md` Phase 2 (Prompting Strategies) and Phase 7 (Evaluation
  Protocol) — no output parser step is described in either

## Recommended Action

- [ ] Accept: apply the proposed change to `docs/architecture/ARCHITECTURE.md`
- [ ] Reject: remove `final-answer-line-v1` from the sweep predictor and align sweep
      runs with the same extraction contract as the baseline (no pre-normalizer step),
      then re-run sweeps
- [ ] Defer: note explicitly in `ARCHITECTURE.md` that the output parser contract for
      sweep runs is under review, and block promotion of any sweep config until
      resolved

## Auto-applied Routine Updates

None — the `docs/execution/NOTEBOOKS.md` change (notebook 05 status from `scaffolded`
to `active`) was already applied directly in the commit and is outside the
`docs/architecture/` and `docs/analysis/` write scope of this agent.
