---
title: Source Policy and Research Workflow
audience: beginner
page_type: source-policy
status: implemented
last_reviewed: 2026-04-21
repo_sources:
  - docs/README.md
  - docs/analysis/PLAN_V0_2_REVIEW_PLAN.md
  - ~/.codex/config.toml
external_sources:
  - https://docs.tavily.com/documentation/mcp
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
---

# Source Policy and Research Workflow

## Why This Page Exists

The `docs/learn` folder is meant to teach, not just summarize. That means every
external claim needs a visible source trail, and every project-status claim
needs a repo-local evidence trail.

## Source Rules

The learn hub follows four rules:

1. Prefer official sources first.
2. Use strong secondary sources when they make beginner explanations clearer.
3. Keep repo-local evidence separate from external sources.
4. Do not make local validation depend on live web access.

## How Tavily Fits In

Tavily is useful as a research helper because it gives an agent live web search
and extraction. It should help authors gather sources, compare official pages,
and refresh outdated explanations.

It should **not** be required during commit-time validation. The repo should be
able to validate page structure and citation presence even when the network or
MCP setup is unavailable.

## Tavily MCP Setup Notes

At the time this page was written, the example local Codex config did not yet
include a Tavily MCP entry. A safe next step is to add Tavily as an optional
research server in local config, then keep the resulting citations inside the
repo so later reviews stay offline-safe.

## Sources

- Repo: [docs/README.md](../../README.md)
- Repo: [docs/analysis/PLAN_V0_2_REVIEW_PLAN.md](../../analysis/PLAN_V0_2_REVIEW_PLAN.md)
- Repo: `~/.codex/config.toml`
- External: https://docs.tavily.com/documentation/mcp
- External: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
