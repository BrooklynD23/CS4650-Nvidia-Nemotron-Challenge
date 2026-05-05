---
title: Adapters, Evaluation, and Why This Repo Cares About Both
audience: beginner
page_type: concept
status: conceptual
last_reviewed: 2026-04-21
repo_sources:
  - docs/architecture/ARCHITECTURE.md
  - docs/execution/SPRINTS.md
  - docs/execution/plans/issue-19-baseline-eval-and-normalization.md
external_sources:
  - https://huggingface.co/papers/2106.09685
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
---

# Adapters, Evaluation, and Why This Repo Cares About Both

## Why This Page Exists

Non-technical readers often assume the hard part of an AI project is training.
In reality, this repo treats evaluation and packaging almost as seriously as
training because a stronger model is only useful if the team can measure and
submit it correctly.

## What An Adapter Is

An adapter is a smaller set of learned weights attached to a larger base model.
The LoRA paper is the standard reference for this idea: instead of changing the
entire model, you train a low-rank update that is lighter and cheaper to ship.

## What Evaluation Means Here

Evaluation in this repo is not just "did the model get a lot right?" The
project architecture and Wave B plans separate evaluation into explicit steps:

- reserve comparison data
- normalize answers carefully
- score runs repeatably
- keep row-level artifacts so the team can inspect failures

## Why The Repo Separates Validation, Golden Tests, and Packaging

- Validation answers "Are we getting better overall?"
- Golden tests answer "Did we accidentally break something we already solved?"
- Packaging answers "Can Kaggle load what we built?" (implemented in
  `src/inference/submission.py`, with a CLI wrapper at
  `scripts/package_submission.py`).

That separation is one of the healthiest parts of the repo because it prevents
the team from confusing "interesting experiments" with "submission-ready work."

## Sources

- Repo: [docs/architecture/ARCHITECTURE.md](../../architecture/ARCHITECTURE.md)
- Repo: [docs/execution/SPRINTS.md](../../execution/SPRINTS.md)
- Repo: [docs/execution/plans/issue-19-baseline-eval-and-normalization.md](../../execution/plans/issue-19-baseline-eval-and-normalization.md)
- External: https://huggingface.co/papers/2106.09685
- External: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
