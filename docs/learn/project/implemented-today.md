---
title: What Exists in the Repo Today
audience: beginner
page_type: project-status
status: implemented
last_reviewed: 2026-04-21
repo_sources:
  - docs/execution/SPRINTS.md
  - docs/execution/NOTEBOOKS.md
  - src/contracts.py
  - src/evaluation/runner.py
  - src/inference/submission.py
external_sources: []
---

# What Exists in the Repo Today

## Why This Page Exists

This page is the safest answer to "What has actually been built already?" It
focuses on committed repo state rather than hopes, future phases, or marketing
language.

## What Exists Today

As of the latest review date on this page, the repo contains delivered Wave A/B
work in four broad areas:

- execution and review harness docs
- canonical data contracts and schema utilities
- validation and golden-set evaluation gates
- baseline evaluation and submission-packaging code

That means the repo is no longer just planning documents. It now contains real
Python modules for contracts, evaluation, and packaging, along with tests.

## Current Repo Evidence

- The phase map in `docs/execution/SPRINTS.md` shows Waves A and B as the
  foundation for the project.
- The notebook registry in `docs/execution/NOTEBOOKS.md` marks the early
  notebooks as validated or active rather than scaffold-only.
- `src/contracts.py` defines the shared project contracts.
- `src/evaluation/runner.py` and related modules show that the eval pipeline now
  exists in code.
- `src/inference/submission.py` shows that the packaging path is implemented in
  code, not only discussed in planning docs.

## Why This Matters

For a non-technical reader, this means the team has moved past "we have a plan"
and into "we have a working foundation." That foundation is what later training
and solver work will build on.

## Sources

- Repo: [docs/execution/SPRINTS.md](../../execution/SPRINTS.md)
- Repo: [docs/execution/NOTEBOOKS.md](../../execution/NOTEBOOKS.md)
- Repo: [src/contracts.py](../../../src/contracts.py)
- Repo: [src/evaluation/runner.py](../../../src/evaluation/runner.py)
- Repo: [src/inference/submission.py](../../../src/inference/submission.py)
