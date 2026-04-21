# Issue #18: Validation Reservation and Golden Set Regression Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the held-out validation split and the immutable golden regression set before any training starts, so every later eval run can detect regression instead of silently drifting.

**Architecture:** Use the canonical `ReasoningExample` contract from `docs/architecture/ARCHITECTURE.md` as the source of truth, then derive two tracked eval artifacts: a validation split for model selection and an immutable golden set for regression gates. Both artifacts should be versioned, seed-attributed, and loadable without notebook state so the harness can compare runs deterministically.

**Tech Stack:** Python, JSONL, pandas, pytest, deterministic sampling utilities, markdown documentation.

---

## Issue Metadata

| Field | Value |
|---|---|
| Parent epic | `#13-#25` planning umbrella |
| Deliverable path | `docs/execution/plans/issue-18-validation-and-golden-set.md` |
| Dependencies | `#14` constraints freeze, `#17` schema/EDA contract, `#19` baseline eval + normalization |
| Agent owner | Agent 3 |
| Human reviewer | Capstone reviewer |
| Architecture reviewer | Required, because this defines shared eval-set and regression policy |

## Decision Summary

- Reserve validation data before any training data transformation touches the corpus.
- Keep the golden set immutable after selection; only a new versioned artifact can replace it.
- Treat any golden-set miss as a regression gate failure, even if aggregate validation accuracy improves.
- Store enough provenance to reproduce selection: seed, source split, selection rule, and dataset version.

## File Responsibilities

**Future data artifacts**
- `data/eval/validation_200.jsonl`: held-out validation split for model selection.
- `data/eval/golden_20.jsonl`: immutable regression set used as a gate.

**Future implementation files**
- `src/evaluation/splits.py`: split selection, validation, and serialization helpers.
- `src/evaluation/golden_gate.py`: golden regression policy and gate evaluation.
- `src/evaluation/artifacts.py`: artifact schema and provenance helpers.
- `tests/evaluation/test_splits.py`: split contract and immutability tests.
- `tests/evaluation/test_golden_gate.py`: regression-gate behavior tests.

## Task 1: Lock the split contract

**Files:**
- Create: `src/evaluation/splits.py`
- Create: `tests/evaluation/test_splits.py`

**Artifacts to define**

```json
{
  "example_id": "string",
  "category": "string",
  "prompt": "string",
  "gold": "string",
  "source": "string",
  "split": "val|golden",
  "dataset_version": "string",
  "selection_seed": 42,
  "selection_rule": "string",
  "selection_reason": "string"
}
```

- [ ] Define a sampler that can reserve exactly 200 validation rows and exactly 20 golden rows from the canonical dataset.
- [ ] Require unique `example_id` values across `train`, `val`, and `golden`.
- [ ] Require `split == "golden"` rows to be excluded from training and validation sampling.
- [ ] Preserve category coverage so the held-out sets are not dominated by one family.

**Tests that must fail first**
- A duplicate `example_id` across split files is rejected.
- A row with missing `category` or `prompt` is rejected.
- A golden row that also appears in `train` is rejected.

**Acceptance for this task**
- Validation and golden selection are deterministic for a fixed seed.
- Selection metadata is recorded with each row, not only in a separate README.

## Task 2: Define the golden regression policy

**Files:**
- Create: `src/evaluation/golden_gate.py`
- Create: `tests/evaluation/test_golden_gate.py`

**Policy**

- A model run passes only if every golden example is correct under the frozen scoring contract.
- Any single miss blocks promotion, even if the overall validation score is higher than the previous best.
- If the golden set is replaced, it must be a new artifact version, never an in-place edit.

**Regression cases to codify**
- A run that improves validation but misses one golden example fails the gate.
- A run with nondeterministic output on the same seed fails the gate.
- A run that changes formatting but keeps the same normalized answer still passes, because formatting drift is handled in issue `#19`.

**Acceptance for this task**
- Golden gate output is a simple pass/fail plus a per-example failure list.
- Gate output can be consumed by later baseline eval reports without notebook context.

## Task 3: Publish the artifact contract and review checklist

**Files:**
- Create: `src/evaluation/artifacts.py`
- Create: `docs/execution/README.md` update is **not allowed in this issue**; keep the contract self-contained in this plan and the future code files above.
- Create: `tests/evaluation/test_artifact_manifest.py`

**Artifact requirements**
- Store selection artifacts as JSONL for row-level diffability.
- Attach `dataset_version`, `selection_seed`, and `selection_rule` to every record.
- Keep the golden set immutable after the first approved version.
- Support a future Parquet mirror only if it preserves the same row fields.

**Review checklist**
- Validation set is clearly separated from the golden set.
- The golden set is small enough for every eval run to check.
- The split files can be reloaded on a clean machine and produce the same row order.
- The plan states what happens when a golden example must be replaced: create a new version and re-run review.

## Acceptance Criteria

- Validation split and golden split are both versioned and reproducible.
- Golden regression gating is strict enough to stop accidental score regressions.
- The artifact schema contains provenance fields sufficient for later eval run attribution.
- The issue can be reviewed without needing notebook execution.

## Sources to Verify

- `docs/architecture/ARCHITECTURE.md` for `ReasoningExample`, `EvalRecord`, and the golden-set requirement.
- `docs/architecture/COMPETITION.md` for competition constraints and the hidden-benchmark context.
- `docs/execution/ISSUE_REVIEW_HARNESS.md` for required issue fields and review lanes.
- `docs/planning/plan_v0.2.md` for the reserved validation and golden set sizes.
- `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` for the drift/risk rationale and contract gaps.

## Risks / Open Questions

- The exact class balance for the validation and golden sets is still a judgment call until the dataset schema issue is frozen.
- If the hidden benchmark emphasizes a category that is underrepresented in the held-out data, the golden set may need a versioned refresh.
- The repository currently lacks implementation files, so the first code pass must be careful not to create notebook-only logic.

