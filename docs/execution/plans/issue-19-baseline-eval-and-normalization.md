# Issue #19: Baseline Evaluation and Normalization Versioning Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable baseline eval pipeline that ingests frozen examples, predicts, normalizes, scores, and reports record-level artifacts with versioned normalization rules.

**Architecture:** The eval runner should be pipeline-shaped: ingest → predict → normalize → score → report. Raw predictions stay untouched, normalization is versioned and swappable, and each scored row links back to the example plus the exact run configuration. This keeps competition-style exact-match scoring reproducible while still allowing future parser variants without breaking historical comparisons.

**Tech Stack:** Python, JSONL, pandas, pytest, optional Parquet export, deterministic seeding, markdown reporting.

---

## Issue Metadata

| Field | Value |
|---|---|
| Parent epic | `#13-#25` planning umbrella |
| Deliverable path | `docs/execution/plans/issue-19-baseline-eval-and-normalization.md` |
| Dependencies | `#14` constraints freeze, `#17` schema/EDA contract, `#18` validation and golden set |
| Agent owner | Agent 3 |
| Human reviewer | Capstone reviewer |
| Architecture reviewer | Required, because this defines shared eval artifacts and normalization policy |

## Decision Summary

- Treat exact-answer normalization as the default scoring contract.
- Version the normalization rules so a scoring change is an explicit artifact change, not an invisible code drift.
- Record the full eval run config alongside every scored prediction.
- Keep baseline evaluation fully deterministic for a fixed model, prompt template, and seed.

## Pipeline Steps

1. **Ingest** frozen validation or benchmark rows from the canonical schema.
2. **Predict** with the selected base model and prompt template.
3. **Normalize** the raw prediction using a named normalization version.
4. **Score** normalized predictions against gold answers.
5. **Report** record-level results plus aggregate metrics and provenance.

## File Responsibilities

**Future implementation files**
- `src/evaluation/runner.py`: end-to-end eval pipeline orchestration.
- `src/evaluation/normalization.py`: versioned normalization and parsing rules.
- `src/evaluation/scoring.py`: exact-match and per-category scoring helpers.
- `src/evaluation/reporting.py`: record-level artifact writing and summary generation.
- `src/evaluation/config.py`: run config schema and attribution helpers.
- `tests/evaluation/test_normalization.py`: drift-catching normalization tests.
- `tests/evaluation/test_scoring.py`: parsing and scoring tests.
- `tests/evaluation/test_runner.py`: pipeline integration tests.

**Future artifact outputs**
- `data/eval/runs/<run_id>/predictions.jsonl`
- `data/eval/runs/<run_id>/eval_records.jsonl`
- `data/eval/runs/<run_id>/run_config.json`
- `data/eval/runs/<run_id>/summary.json`

## Task 1: Define the run config and record schema

**Files:**
- Create: `src/evaluation/config.py`
- Create: `src/evaluation/reporting.py`
- Create: `tests/evaluation/test_runner.py`

**Record contract**

```json
{
  "run_id": "string",
  "model_id": "string",
  "prompt_template_id": "string",
  "normalization_version": "string",
  "example_id": "string",
  "category": "string",
  "gold": "string",
  "prediction": "string",
  "normalized_prediction": "string",
  "correct": true,
  "latency_ms": 0.0,
  "tokens_in": 0,
  "tokens_out": 0,
  "seed": 42
}
```

- [ ] Require every record to carry `run_id`, `model_id`, `prompt_template_id`, `normalization_version`, and `example_id`.
- [ ] Require every run to emit a `run_config.json` snapshot that matches the fields stored in each record.
- [ ] Require aggregate summaries to be derivable from record-level JSONL without notebook state.

**Tests that must fail first**
- A record missing `normalization_version` is rejected.
- A run config that does not match the record-level attribution is rejected.
- A summary that cannot be reconstructed from row-level artifacts is rejected.

## Task 2: Implement versioned normalization and parsing

**Files:**
- Create: `src/evaluation/normalization.py`
- Create: `tests/evaluation/test_normalization.py`

**Normalization policy**
- Default to exact string normalization compatible with Kaggle scoring assumptions.
- Keep normalization version names explicit, such as `exact_v1`.
- Allow category-specific parsers only when the category contract requires them.
- Preserve the raw prediction in every case so score disputes can be audited later.

**Drift-catching test cases**
- Leading and trailing whitespace normalizes away when the version says it should.
- Interior whitespace only changes behavior when the version explicitly permits collapsing.
- Reasoning text before or after the final answer is stripped only if the version allows it.
- A parser change that turns `"A\n"` into `"A"` is detected as a version bump, not a silent behavior change.
- Category-specific formats such as numeric strings, symbols, or booleans use their own named parser, not the default parser.

**Acceptance for this task**
- Normalization behavior changes only when the version string changes.
- The test suite demonstrates at least one case where a naive parser would score incorrectly.

## Task 3: Build the baseline eval runner

**Files:**
- Create: `src/evaluation/runner.py`
- Create: `src/evaluation/scoring.py`
- Create: `tests/evaluation/test_scoring.py`

**Runner behavior**
- Load frozen examples from the validation or benchmark split.
- Generate predictions with a fixed seed and prompt template ID.
- Normalize every raw prediction with the selected normalization version.
- Score normalized predictions using exact match by default.
- Write record-level outputs before aggregate summaries so partial failures are inspectable.

**Scoring drift tests**
- The same raw prediction scores differently under two normalization versions, and the runner records which version was used.
- A prediction with extra reasoning text fails under the strict version and passes only under an explicitly permissive version.
- A blank prediction is always incorrect.
- A category-specific parser mismatch is surfaced in the row-level artifact, not hidden in the aggregate score.

**Acceptance for this task**
- The runner emits `eval_records.jsonl` plus a summary file for every run.
- The summary includes overall accuracy, per-category accuracy, latency, token usage, and the active normalization version.

## Task 4: Add pipeline integration checks

**Files:**
- Create: `tests/evaluation/test_runner.py`
- Create: `tests/evaluation/test_reporting.py`

**Integration checks**
- Ingest, predict, normalize, score, and report can run end-to-end on a tiny frozen fixture.
- The report references `run_id`, `model_id`, `prompt_template_id`, `normalization_version`, and `seed`.
- The output is stable when rerun with the same seed and same normalization version.

**Acceptance for this task**
- A clean checkout can reproduce the same baseline report from the same frozen inputs.
- The pipeline can distinguish model quality changes from normalization-version changes.

## Acceptance Criteria

- The eval pipeline is explicit and linear: ingest → predict → normalize → score → report.
- Record-level artifacts contain enough provenance to reproduce a run and debug a bad row.
- Normalization rules are versioned, tested, and visible in the reported artifacts.
- The issue-level checklist can be reviewed against the generated artifacts without notebook context.

## Harness Alignment

This issue must satisfy the harness fields in `docs/execution/ISSUE_REVIEW_HARNESS.md`:
- Parent epic
- Deliverable path
- Dependencies
- Agent owner
- Human reviewer
- Architecture reviewer
- Acceptance checklist
- Sources to verify
- Risks / open questions

## Sources to Verify

- `docs/architecture/ARCHITECTURE.md` for the eval pipeline, `EvalRecord`, and exact-match direction.
- `docs/architecture/COMPETITION.md` for competition scoring uncertainty and category context.
- `docs/execution/ISSUE_REVIEW_HARNESS.md` for required issue metadata and review workflow.
- `docs/planning/plan_v0.2.md` for the baseline eval protocol and validation flow.
- `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` for normalization, eval-record, and score-drift risks.

## Risks / Open Questions

- The exact Kaggle normalization contract is still not fully verified, so version `exact_v1` should be treated as a placeholder name until the rules are frozen.
- Some categories may need specialized parsing, but the default path should stay exact-match first.
- If the base model or prompt template changes, those changes must be isolated from normalization changes in the record-level attribution.

