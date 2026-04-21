# Notebook Plans

One markdown plan per foundation notebook under `notebooks/00_*.ipynb` through `notebooks/10_*.ipynb`. Each plan follows [`_TEMPLATE.md`](./_TEMPLATE.md) and states **Objective → Strategy → MVP → Test Cases (Primary + Alternative + Regression) → Success Criteria → Risks → Artifacts → Effort**.

These are planning artifacts. The living registry of notebook state (scaffolded / active / validated) remains [`docs/execution/NOTEBOOKS.md`](../../execution/NOTEBOOKS.md). The master plan is [`../plan_v0.2.md`](../plan_v0.2.md).

## How to use this folder

- Start a notebook by reading its plan here, then open the corresponding `.ipynb`.
- If reality diverges from the plan, update this file (not just the notebook) so downstream consumers see the change.
- If a plan's Test Cases cannot be satisfied, escalate before writing code — do not silently relax the contract.

## Index

| # | Plan | Parent Issue | Wave | Plan-v0.2 Phase | Upstream Deps | Downstream Consumers |
|---|---|---|---|---|---|---|
| 00 | [Competition Constraints and Rules](./00_competition_constraints_and_rules.md) | `#14` | A | pre-Phase 0 gate | `#13` | all (01-10) |
| 01 | [External Baselines and Design Deltas](./01_external_baselines_and_design_deltas.md) | `#16` | A | informs Phase 1 + 4 | `#13` | 04, 07, 09 |
| 02 | [Dataset Schema and EDA](./02_dataset_schema_and_eda.md) | `#17` | B | Phase 1.3 + Phase 3 prep | `#14` | 03, 04, 06, 08, 09 |
| 03 | [Validation Split and Golden Set](./03_validation_and_golden_set.md) | `#18` | B | Phase 3.1 | `#14`, `#17` | 04, 05, 06, 07, 08, 09, 10 |
| 04 | [Baseline Eval and Normalization](./04_baseline_eval_and_normalization.md) | `#19` | B | Phase 1.2 + Phase 7.2 | `#14`, `#17`, `#18` | 05, 06, 09, 10 |
| 05 | [Prompting and Decode Sweeps](./05_prompting_and_decode_sweeps.md) | `#21` | C | Phase 2 | `#19` | 06, 09 |
| 06 | [Trajectory Collection and Error Slices](./06_trajectory_collection_and_error_slices.md) | `#22` | C | Phase 3.3 + Phase 5 | `#19`, `#21` | 07, 08, 09 |
| 07 | [Solver Framework Design](./07_solver_framework_design.md) | `#23` | D | informs Phase 4 + 8 | `#15`, `#22` | 08, 09, 10 |
| 08 | [Synthetic Data Recipe](./08_synthetic_data_recipe.md) | `#24` | D | Phase 5 | `#15`, `#22`, `#23` | 09 |
| 09 | [SFT LoRA Runbook and Masking](./09_sft_runbook_and_masking.md) | `#25` | D | Phase 4 | `#14`, `#15`, `#19`, `#23`, `#24` | 10 |
| 10 | [Submission Packaging and Provenance](./10_submission_packaging_and_provenance.md) | `#20` | B/D | Phase 8 | `#14`, `#19` | — (terminal) |

## Dependency Graph (notebook-level)

```
00 ─┬─► 01
    ├─► 02 ─┬─► 03 ─┬─► 04 ─► 05 ─► 06 ─┬─► 07 ─► 08 ─► 09 ─► 10
    │       │       │                    │
    │       │       └────────────────────┼──────────────────► 10
    │       └────────────────────────────┘
    └─► (all 01-10 inherit constraint facts from 00)
```

## Review status

| Plan | First draft | Adversarial review | Status |
|---|---|---|---|
| 00 | ✅ 2026-04-20 | see [ADVERSARIAL_REVIEW.md](./ADVERSARIAL_REVIEW.md) | draft |
| 01 | ✅ 2026-04-20 | — | draft |
| 02 | ✅ 2026-04-20 | — | draft |
| 03 | ✅ 2026-04-20 | — | draft |
| 04 | ✅ 2026-04-20 | — | draft |
| 05 | ✅ 2026-04-20 | — | draft |
| 06 | ✅ 2026-04-20 | — | draft |
| 07 | ✅ 2026-04-20 | — | draft |
| 08 | ✅ 2026-04-20 | — | draft |
| 09 | ✅ 2026-04-20 | — | draft |
| 10 | ✅ 2026-04-20 | — | draft |

## Authoring conventions

- **One plan per notebook**. Do not bundle multiple notebooks into one plan.
- Every plan carries **Primary + Alternative + Regression** test cases. If the alternative path is "none", say so explicitly and justify.
- Every Success Criteria item must be **mechanically verifiable** (a file exists, a metric crosses a threshold, a hash matches). No aspirational wording.
- Cite external sources with a URL + the date the link was checked.
