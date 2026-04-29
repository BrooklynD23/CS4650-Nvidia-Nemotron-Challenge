# Sprint Plan (Issue-Driven Dependency Waves)

This project now uses a two-level issue model:

- Existing GitHub issues `#1-#12` remain the project epics.
- Child issues `#13-#25` define the notebook-first foundation work that agents and human reviewers can run in parallel.

The team should execute these as dependency waves, not calendar sprints. This keeps work parallelizable while preserving the phase gates from `docs/planning/plan_v0.2.md`.

## Epic Map

| Epic | Goal |
|---|---|
| `#1` | Verify Kaggle constraints and freeze competition assumptions |
| `#2` | Review external notebooks and extract transferable ideas |
| `#3` | Normalize the dataset into a canonical schema |
| `#4` | Build trustworthy local evaluation and regression gates |
| `#5` | Produce valid submission artifacts with provenance |
| `#6` | Find prompt and decode improvements before training |
| `#7` | Collect trajectories and failure slices for targeted iteration |
| `#8` | Define a category-aware solver and teacher framework |
| `#9` | Build first solver implementation |
| `#10` | Generate synthetic data with quality controls |
| `#11` | Train LoRA/QLoRA adapters with masking and provenance |
| `#12` | Document reproducible submission and review workflow |

## Dependency Waves

### Wave A: Constraints, references, and review harness

| Child issue | Parent | Deliverable | Depends on |
|---|---|---|---|
| `#13` | `#12` | Notebook template, citation rubric, and issue checklist | None |
| `#14` | `#1` | `notebooks/00_competition_constraints_and_rules.ipynb` | `#13` |
| `#15` | `#12` | Agent and human review harness (`.github/ISSUE_TEMPLATE/*`, execution doc) | `#13` |
| `#16` | `#2` | `notebooks/01_external_baselines_and_design_deltas.ipynb` | `#13` |

**Gate:** official constraints, notebook standards, and the review workflow are frozen.

### Wave B: Data, validation, evaluation, and packaging

| Child issue | Parent | Deliverable | Depends on |
|---|---|---|---|
| `#17` | `#3` | `notebooks/02_dataset_schema_and_eda.ipynb` | `#14` |
| `#18` | `#4` | `notebooks/03_validation_and_golden_set.ipynb` | `#14`, `#17` |
| `#19` | `#4` | `notebooks/04_baseline_eval_and_normalization.ipynb` | `#14`, `#17`, `#18` |
| `#20` | `#5` | `notebooks/10_submission_packaging_and_provenance.ipynb` | `#14`, `#19` |

**Gate:** the schema, held-out split policy, eval record shape, and package manifest are stable.

### Wave C: Prompting and failure analysis

| Child issue | Parent | Deliverable | Depends on |
|---|---|---|---|
| `#21` | `#6` | `notebooks/05_prompting_and_decode_sweeps.ipynb` | `#19` |
| `#22` | `#7` | `notebooks/06_trajectory_collection_and_error_slices.ipynb` | `#19`, `#21` |

**Gate:** the team has a measured prompting baseline and a failure analysis dataset for targeted iteration.

### Wave D: Future-phase notebook specs

| Child issue | Parent | Deliverable | Depends on |
|---|---|---|---|
| `#23` | `#8` | `notebooks/07_solver_framework_design.ipynb` | `#15`, `#22` |
| `#24` | `#10` | `notebooks/08_synthetic_data_recipe.ipynb` | `#15`, `#22`, `#23` |
| `#25` | `#11` | `notebooks/09_sft_runbook_and_masking.ipynb` | `#14`, `#15`, `#19`, `#23`, `#24` |

**Gate:** the next training epics are implementation-ready, but model training stays blocked until Waves A-C are complete.

## Ownership Model

- Each child issue must have one `agent owner`.
- Each child issue must have one `human reviewer`.
- Issues `#14`, `#19`, `#20`, and `#25` also require an `architecture reviewer`.
- Child issues remain open until the notebook or harness artifact exists, cites its sources, and passes the acceptance checklist.

## Recommended Labels

- `wave:A`, `wave:B`, `wave:C`, `wave:D`
- `lane:notebook`, `lane:harness`
- `owner:agent`
- `needs-human-review`
- `architecture-impact`
- `blocked`

## Acceptance Rules

- A notebook issue closes only when the notebook exists under `notebooks/`, follows the required documentation structure, and records source links in a `Sources` section.
- A harness issue closes only when the associated Markdown or GitHub template artifact exists and links back to its parent epic.
- `#14` is now verified (snapshot 2026-04-29, see `docs/architecture/COMPETITION.md`): the competition uses `\\boxed{}` extraction with exact match (or `1e-3` numeric tolerance), reasoning text is allowed, and the base model is KaggleHub `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`. All later child issues must adopt this `\\boxed{}` + exact-match (with numeric tolerance) contract before they can close.
