---
title: What Is Planned Next
audience: beginner
page_type: roadmap
status: planned
last_reviewed: 2026-04-21
repo_sources:
  - docs/execution/SPRINTS.md
  - docs/planning/plan_v0.2.md
  - docs/planning/notebooks/README.md
external_sources: []
---

# What Is Planned Next

## Why This Page Exists

This page keeps the roadmap separate from the already-built foundation. That
separation matters because good projects do not blur future ambition with
present capability.

## What Is Planned Next

The next major steps in the repo’s roadmap are:

- failure-slice and trajectory collection (notebook 06, scaffolded)
- solver-framework design (notebook 07, scaffolded)
- synthetic-data design (notebook 08, scaffolded)
- SFT and masking runbooks for training (notebook 09, scaffolded), using
  `r<=32`, explicit completion-only masking, and the verified `#14` base
  model/package contract

Note: prompt and decode experiments are no longer purely planned. The
implementation in `src/evaluation/prompt_sweeps.py` and notebook 05 exist and
are tested. Execution is blocked only on split artifact availability from issue
#18. See the In Progress page for the current status.

These remaining items are meaningful next steps, but they should be described as
planned work until the repo contains finished deliverables and validation for
them.

External baseline ideas from Tong (`tonghuikang`) and konbu17 are now reviewed,
but implementation remains planned work unless the idea is validated locally and
does not drift from the official Kaggle/NVIDIA contract.

## Current Repo Evidence

- `docs/execution/SPRINTS.md` maps Waves C and D explicitly.
- `docs/planning/plan_v0.2.md` still contains the broader multi-phase execution
  strategy.
- `docs/planning/notebooks/README.md` shows the notebook sequence beyond the
  already-delivered Wave A/B work.

## Why This Matters

For a beginner reader, the roadmap explains where the project is going without
creating false confidence that every later phase already exists in code.

## Sources

- Repo: [docs/execution/SPRINTS.md](../../execution/SPRINTS.md)
- Repo: [docs/planning/plan_v0.2.md](../../planning/plan_v0.2.md)
- Repo: [docs/planning/notebooks/README.md](../../planning/notebooks/README.md)
