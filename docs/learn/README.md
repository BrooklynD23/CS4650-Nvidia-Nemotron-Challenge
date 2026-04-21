---
title: Learn the Repo
audience: beginner
page_type: index
status: implemented
last_reviewed: 2026-04-21
repo_sources:
  - docs/README.md
  - docs/execution/SPRINTS.md
external_sources:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
  - https://research.nvidia.com/labs/nemotron/Nemotron-3/
---

# Learn the Repo

## Why This Page Exists

This folder is the beginner-friendly front door for the project. It explains
what the repository is doing, why the NVIDIA Kaggle challenge matters, how
large language models (LLMs) fit into the work, and which parts of the repo are
already implemented versus still planned.

## How To Use This Hub

If you are new to the project, read in this order:

1. [LLM basics](foundations/llm-basics.md)
2. [Kaggle challenge explained](foundations/kaggle-challenge-explained.md)
3. [Nemotron architecture](foundations/nemotron-architecture.md)
4. [Adapters and evaluation](foundations/adapters-and-evaluation.md)
5. [What exists today](project/implemented-today.md)
6. [What is still in progress](project/in-progress.md)
7. [What is planned next](project/planned-roadmap.md)

## What You Will Learn

- What an LLM is, in plain language
- Why this repo focuses on adapters instead of retraining a full model
- What NVIDIA Nemotron is, and why its architecture matters to the challenge
- How the team measures progress with validation, golden tests, and submission
  packaging
- Which phases are already present in the repo and which phases remain roadmap
  items

## Hub Map

- `foundations/`: background explainers
- `project/`: repo status pages, separated by implemented, in-progress, and
  planned work
- `sources/`: source policy, citation ledger, and Tavily research workflow
- `_templates/`: the markdown contract every learn page should follow

## Current Repo Evidence

- The canonical docs router already lives in `docs/README.md`.
- The current phase map and child issues live in `docs/execution/SPRINTS.md`.
- The repo is currently carrying Wave A/B implementation work, with later waves
  still described as dependent future phases.

## Sources

- Repo: [docs/README.md](../README.md)
- Repo: [docs/execution/SPRINTS.md](../execution/SPRINTS.md)
- External: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- External: https://research.nvidia.com/labs/nemotron/Nemotron-3/
