---
title: What Is Still In Progress
audience: beginner
page_type: project-status
status: in_progress
last_reviewed: 2026-04-21
repo_sources:
  - docs/execution/SPRINTS.md
  - docs/execution/NOTEBOOKS.md
  - docs/architecture/COMPETITION.md
  - docs/execution/plans/issue-25-hpc-queue-runbook.md
  - docs/analysis/prompting_findings.md
  - src/evaluation/prompt_sweeps.py
external_sources: []
---

# What Is Still In Progress

## Why This Page Exists

Beginner readers need a place where uncertainty is stated plainly. This page is
for work that the repo is actively reasoning about but has not yet turned into a
finished, stable delivery.

## What Is In Progress

The biggest unfinished work falls into three groups:

- external facts that still need confirmation, especially around competition
  constraints and exact model assumptions
- future implementation waves that have plan docs or notebook scaffolds, but not
  finished production code
- the prompt/decode sweep workflow: the implementation is complete and tested,
  but actual execution is blocked because the required dataset split artifacts
  have not yet been produced by the earlier pipeline step (issue #18)

## Current Repo Evidence

- `docs/architecture/COMPETITION.md` still marks some competition facts as open
  questions or assumptions rather than fully frozen truths.
- `docs/execution/SPRINTS.md` shows Waves C and D as downstream work that
  depends on the Wave B foundation.
- `docs/execution/NOTEBOOKS.md` marks notebook 05 (`05_prompting_and_decode_sweeps.ipynb`)
  as `active`, meaning the code is real and importable, but the notebook has
  not been executed end-to-end yet. Several later notebooks remain `scaffolded`.
- `docs/analysis/prompting_findings.md` records the current blocked state
  explicitly: the notebook requires `data/eval/validation_200.jsonl` and
  `data/eval/golden_20.jsonl` from issue #18, which are not yet committed to the
  repo. Execution will produce ranked sweep results once those files exist.
- `docs/execution/plans/issue-25-hpc-queue-runbook.md` exists as a planning
  artifact, but the corresponding training work is still gated on earlier
  phases.

## What This Means For Readers

The repo is mature enough to explain its core foundation clearly, but it is not
yet at the stage where full solver design, synthetic-data generation, and
training execution should be described as completed work.

## Sources

- Repo: [docs/execution/SPRINTS.md](../../execution/SPRINTS.md)
- Repo: [docs/execution/NOTEBOOKS.md](../../execution/NOTEBOOKS.md)
- Repo: [docs/architecture/COMPETITION.md](../../architecture/COMPETITION.md)
- Repo: [docs/execution/plans/issue-25-hpc-queue-runbook.md](../../execution/plans/issue-25-hpc-queue-runbook.md)
- Repo: [docs/analysis/prompting_findings.md](../../analysis/prompting_findings.md)
- Repo: [src/evaluation/prompt_sweeps.py](../../../src/evaluation/prompt_sweeps.py)
