# CLAUDE.md (Router)

Route your task to the right docs. Keep context minimal — open only what your task needs.

| Task | Go to |
|------|-------|
| Competition rules, dataset, scoring | `docs/architecture/COMPETITION.md` |
| System / model design | `docs/architecture/ARCHITECTURE.md` |
| Execution plan (source of truth) | `docs/planning/plan_v0.2.md` |
| Sprint / issue tracking | `docs/execution/SPRINTS.md` |
| Notebook registry & status | `docs/execution/NOTEBOOKS.md` |
| Current repo state | `docs/learn/project/implemented-today.md` |
| Analysis & adversarial review | `docs/analysis/` |
| All docs index | `docs/README.md` |

## Working rules

- Reusable code → `src/`; keep notebooks thin.
- Large artifacts (`data/`, `adapters/`, `experiments/`) stay out of git.
- Every notebook must have a row in `docs/execution/NOTEBOOKS.md`.
