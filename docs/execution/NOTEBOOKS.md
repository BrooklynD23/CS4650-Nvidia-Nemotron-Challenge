# Notebook Registry

This registry tracks the notebook-first execution artifacts for the foundation phase of the project.

## Operating Rules

There are two notebook classes:

1. `Foundation notebooks` under `notebooks/00_*.ipynb` through `notebooks/10_*.ipynb`
2. `External reference notebooks` that we review, cite, and compare against our architecture

If a notebook materially changes model behavior or project-wide assumptions, update this file with:

- dataset version or source snapshot
- base model id
- eval scores or intended outputs
- artifact pointers
- parent GitHub issue

Every local notebook must include the following sections inside the notebook:

- `Audience and Why It Matters`
- `Decision / Hypothesis`
- `Environment and Reproduction`
- `Method and Outputs`
- `Results / Open Risks`
- `Sources`

## Foundation Notebook Inventory

| Notebook | Parent issue | Purpose | Status |
|---|---|---|---|
| `notebooks/00_competition_constraints_and_rules.ipynb` | `#14` | Verify rules, model constraints, evaluation rules, and deadlines | scaffolded |
| `notebooks/01_external_baselines_and_design_deltas.ipynb` | `#16` | Compare Tong, `konbu17`, and current repo assumptions | scaffolded |
| `notebooks/02_dataset_schema_and_eda.ipynb` | `#17` | Explain dataset schema, category shape, and normalization plan | scaffolded |
| `notebooks/03_validation_and_golden_set.ipynb` | `#18` | Define validation split and golden-set regression policy | scaffolded |
| `notebooks/04_baseline_eval_and_normalization.ipynb` | `#19` | Define exact-match eval records and normalization rules | scaffolded |
| `notebooks/05_prompting_and_decode_sweeps.ipynb` | `#21` | Compare prompt templates and decode settings | scaffolded |
| `notebooks/06_trajectory_collection_and_error_slices.ipynb` | `#22` | Document failure slices and targeted follow-up loops | scaffolded |
| `notebooks/07_solver_framework_design.ipynb` | `#23` | Specify the category-aware solver interface and fallback policy | scaffolded |
| `notebooks/08_synthetic_data_recipe.ipynb` | `#24` | Specify teacher, filter, and provenance rules for synthetic data | scaffolded |
| `notebooks/09_sft_runbook_and_masking.ipynb` | `#25` | Specify LoRA/QLoRA runbook, masking, and checkpoint policy | scaffolded |
| `notebooks/10_submission_packaging_and_provenance.ipynb` | `#20` | Specify packaging, dry-run validation, and provenance metadata | scaffolded |

## Cross-Notebook Architectural Decisions

- Treat the benchmark as category-specific rule induction until `#14` proves otherwise.
- Keep base model id, LoRA rank cap, and answer normalization configurable until officially verified.
- Prefer strict exact-match evaluation contracts over math-specific formatting assumptions.
- Keep explanation and citations inside the notebook so non-technical reviewers can read one artifact end-to-end.

## External Reference Work

| Source | Why it matters |
|---|---|
| https://github.com/tonghuikang/nemotron | Winner-adjacent public pipeline; strong source for masking and augmentation ideas |
| https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2 | Requested review target for reproduction and delta analysis |
| https://www.kaggle.com/datasets/kishanvavdara/nemotron-reasoning-traj | Useful for failure-slice and trajectory analysis patterns |
| https://aitherium.com/blog/nemotron-reasoning-challenge-mirothinker-distillation/ | Public distillation notes and design pitfalls |

## Update Procedure

- Update this file whenever a notebook changes state from `scaffolded` to `active`, `validated`, or `superseded`.
- Link the corresponding issue and artifact pointers in the issue description and closing comment.
- If a notebook causes a contract change, also update `docs/architecture/ARCHITECTURE.md` and reference the change in `docs/execution/SPRINTS.md`.
