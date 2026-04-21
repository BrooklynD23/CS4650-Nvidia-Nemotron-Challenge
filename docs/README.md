# Docs Router

Last Updated: 2026-04-21

Use this file as the entry point for agents and contributors. Route to the subfolder that matches the task.

## Routing By Need

- Planning, scope, phase gates, dependencies:
  `docs/planning/`
- Competition constraints and system design:
  `docs/architecture/`
- Critical reviews and risk pressure-tests:
  `docs/analysis/`
- Sprint execution, issues, notebooks:
  `docs/execution/`
- Beginner-friendly explainer hub with citations and status pages:
  `docs/learn/`

## Canonical Docs

- Execution plan (source of truth):
  `docs/planning/plan_v0.2.md`
- Plan review rationale:
  `docs/planning/plan_review.md`
- Competition constraints:
  `docs/architecture/COMPETITION.md`
- System architecture:
  `docs/architecture/ARCHITECTURE.md`
- Plan/architecture drift review plan (resolution checklist):
  `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`
- Non-technical concepts explainer:
  `docs/learn/README.md`
- Adversarial review:
  `docs/analysis/ADVERSARIAL_REVIEW.md`
- Sprint-to-issue map:
  `docs/execution/SPRINTS.md`
- Notebook registry:
  `docs/execution/NOTEBOOKS.md`
- Issue review workflow:
  `docs/execution/ISSUE_REVIEW_HARNESS.md`

## Folder Index

- `docs/planning/README.md`
- `docs/architecture/README.md`
- `docs/analysis/README.md`
- `docs/execution/README.md`
- `docs/learn/README.md`

## Drift Guard

- Whenever you modify any doc under `docs/`, update `Last Updated` in this file.
- If you move docs, update routing paths and canonical paths in this file and in `AGENTS.md`.
