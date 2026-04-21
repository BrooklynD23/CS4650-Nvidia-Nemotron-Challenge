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
external_sources: []
---

# What Is Still In Progress

## Why This Page Exists

Beginner readers need a place where uncertainty is stated plainly. This page is
for work that the repo is actively reasoning about but has not yet turned into a
finished, stable delivery.

## What Is In Progress

The biggest unfinished work falls into two groups:

- external facts that still need confirmation, especially around competition
  constraints and exact model assumptions
- future implementation waves that have plan docs or notebook scaffolds, but not
  finished production code

## Current Repo Evidence

- `docs/architecture/COMPETITION.md` still marks some competition facts as open
  questions or assumptions rather than fully frozen truths.
- `docs/execution/SPRINTS.md` shows Waves C and D as downstream work that
  depends on the Wave B foundation.
- `docs/execution/NOTEBOOKS.md` still marks several later notebooks as
  scaffolded rather than validated.
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
