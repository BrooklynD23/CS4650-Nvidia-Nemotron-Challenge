# Notebooks

Keep notebooks focused on exploration and visualization. When something becomes repeatable, move it into `src/` + a small CLI/script entry point.

Every notebook should have a corresponding entry in `docs/execution/NOTEBOOKS.md` (purpose, inputs, outputs, last-run metrics).

## Required Internal Structure

Every notebook in this repo must include these sections inside the notebook:

1. `Audience and Why It Matters`
2. `Decision / Hypothesis`
3. `Environment and Reproduction`
4. `Method and Outputs`
5. `Results / Open Risks`
6. `Sources`

## Foundation Notebook Order

The initial notebook execution order is:

1. `00_competition_constraints_and_rules.ipynb`
2. `01_external_baselines_and_design_deltas.ipynb`
3. `02_dataset_schema_and_eda.ipynb`
4. `03_validation_and_golden_set.ipynb`
5. `04_baseline_eval_and_normalization.ipynb`
6. `05_prompting_and_decode_sweeps.ipynb`
7. `06_trajectory_collection_and_error_slices.ipynb`
8. `07_solver_framework_design.ipynb`
9. `08_synthetic_data_recipe.ipynb`
10. `09_sft_runbook_and_masking.ipynb`
11. `10_submission_packaging_and_provenance.ipynb`

## Scaffolding

Use `python scripts/scaffold_notebooks.py` to regenerate the notebook skeletons after updating the metadata in that script. The generator is idempotent and safe to re-run.
