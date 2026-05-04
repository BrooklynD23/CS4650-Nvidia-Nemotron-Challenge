# AGENTS.md (Router)

This repo is a CS4650 capstone for the **NVIDIA Nemotron Model Reasoning Challenge** (Kaggle). Treat this file as a **router** to the canonical project docs and working conventions.

## Start Here (Canonical Docs)

1. **Execution Plan (source of truth):** `docs/planning/plan_v0.2.md`
2. **Plan review / rationale:** `docs/planning/plan_review.md`
3. **Docs index:** `docs/README.md`
4. **Competition facts (dates, dataset, constraints):** `docs/architecture/COMPETITION.md`
5. **System design:** `docs/architecture/ARCHITECTURE.md`
6. **Notebook registry (local + external references):** `docs/execution/NOTEBOOKS.md`
7. **Sprint plan + issue mapping:** `docs/execution/SPRINTS.md`
8. **Adversarial review (plan + MVP gaps):** `docs/analysis/ADVERSARIAL_REVIEW.md`
9. **Current repo state:** `docs/learn/project/implemented-today.md`

## Repo Reality Check (as of today)

Wave A/B foundation is present. The repository now contains:

- `src/` — contracts, evaluation harness, golden-gate, normalization, prompt
  sweeps, submission packaging (~2 500 LOC across 11 modules)
- `tests/` — 19 test files, ~1 957 LOC
- `notebooks/` — 11 Jupyter notebooks (00–10), registered in
  `docs/execution/NOTEBOOKS.md`

See `docs/learn/project/implemented-today.md` for the current state summary.
Wave C/D work (trajectory analysis, solvers, synthetic data, SFT training) is
planned but not yet implemented.

## Working Conventions (so work stays parallelizable)

- Put reusable code in `src/`. Keep notebooks thin.
- Every notebook gets an entry in `docs/execution/NOTEBOOKS.md` (owner, purpose, inputs, outputs, last-run results).
- Keep large artifacts out of git:
  - datasets under `data/`
  - adapters under `adapters/`
  - experiment logs under `experiments/` or `wandb/`
- Prefer “small, composable scripts” over monolithic notebooks once a workflow stabilizes.

## Codex CLI ECC (project-local expectations)

### Model Recommendations

| Task Type | Recommended Model |
|----------|-------------------|
| Routine coding, tests, formatting | o4-mini |
| Complex features, architecture | o3 |
| Debugging, refactoring | o4-mini |
| Security review | o3 |

### Skills Discovery

Skills are auto-loaded from `.agents/skills/` when present.

### Security Without Hooks (minimum bar)

1. Validate inputs at system boundaries (file paths, dataset rows, CLI args).
2. Never hardcode secrets. Use environment variables; keep `.env` out of git.
3. Run dependency audits before pushing (language-appropriate).
4. Review diffs before pushing.
