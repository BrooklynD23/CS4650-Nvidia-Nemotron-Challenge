---
title: Prompt Sweeps, Decode Parameters, and Best-of-N
audience: beginner
page_type: concept
status: conceptual
last_reviewed: 2026-04-21
repo_sources:
  - src/evaluation/prompt_sweeps.py
  - notebooks/05_prompting_and_decode_sweeps.ipynb
  - docs/analysis/prompting_findings.md
  - docs/execution/NOTEBOOKS.md
external_sources: []
---

# Prompt Sweeps, Decode Parameters, and Best-of-N

## Why This Page Exists

Before spending GPU time on fine-tuning, this project tries cheaper
improvements first. This page explains the two main tools for that: changing
the prompt wording ("prompt strategies") and adjusting how the model picks its
answer ("decode parameters"). It also explains Best-of-N, a simple technique
that asks the model multiple times and picks the most common answer.

## What a Prompt Strategy Is

A prompt is the text you send to the model before it starts writing. Small
wording changes can meaningfully affect how well the model answers, even with
no change to the model weights.

This repo uses two strategies:

- **zero-shot-cot** ("zero-shot chain-of-thought"): asks the model to reason
  step by step before giving a final answer. No worked examples are included.
- **few-shot-cot** ("few-shot chain-of-thought"): includes a small number of
  worked examples (question + reasoning + answer) before the real question. The
  model learns the expected format by imitation.

The code in `src/evaluation/prompt_sweeps.py` defines these as named strategy
strings (`"zero-shot-cot"`, `"few-shot-cot"`) so the sweep can run them
systematically and log which one worked better.

## What Decode Parameters Are

After reading your prompt, the model generates text one token at a time. Decode
parameters control how it picks each next token. The two most relevant ones
here are:

- **temperature**: higher values make the model more willing to pick unusual
  words (more creative, more variable); lower values make it more conservative
  and repetitive. A temperature of 0 would always pick the single most likely
  next token.
- **top_p** (nucleus sampling): instead of considering all possible next tokens,
  the model only considers the smallest set whose combined probability adds up
  to `top_p`. This trims very unlikely options.

The sweep in this repo tests four combinations of these two values:
`(0.6, 0.9)`, `(0.6, 0.95)`, `(1.0, 0.9)`, and `(1.0, 0.95)`. Each combination
is tried with three different random seeds to measure how stable the result is.

## What Best-of-N Is

Best-of-N is a simple trick: ask the model the same question N times
independently, then take the most common answer (majority vote). This costs
more compute (N times as many model calls) but can improve accuracy on tasks
where individual answers are noisy.

This repo tests `N=8` and `N=32` as follow-up experiments after identifying the
best prompt/decode combination from the sparse grid.

The majority vote function in this repo breaks ties by picking the answer that
comes first alphabetically. This keeps results reproducible across runs.

## Why a Sparse Grid?

Testing every possible combination of strategies, temperatures, top-p values,
and seeds is expensive. A "sparse grid" means picking a small representative
set of combinations rather than exhaustively searching all of them. The goal
is to get useful signal about which region of the parameter space looks
promising, then focus Best-of-N follow-ups on the winner.

## What the Sweep Produces

Each run in the sweep uses the same evaluation harness as the baseline (notebook
04), so results are directly comparable. The sweep writes:

- one artifact directory per run under `data/eval/runs/<run_id>/`
- an aggregate CSV (`experiments/prompting_sweep_<date>.csv`) with one row per
  unique configuration, including mean accuracy, standard deviation, and delta
  versus the baseline
- a findings markdown (`docs/analysis/prompting_findings.md`) with a ranked
  table and a promotion decision

## Current Status in This Repo

The sweep implementation exists in code and is fully tested. The notebook
(`notebooks/05_prompting_and_decode_sweeps.ipynb`) is marked `active` in the
registry. However, execution is blocked: the required input files
(`data/eval/val.jsonl` and `data/eval/golden.jsonl`) are not yet committed to
the repo. Those files will be produced by issue #18.

No sweep results exist yet. The `docs/analysis/prompting_findings.md` file
currently contains the blocked-state description, not final findings.

## Sources

- Repo: [src/evaluation/prompt_sweeps.py](../../../src/evaluation/prompt_sweeps.py)
- Repo: [notebooks/05_prompting_and_decode_sweeps.ipynb](../../../notebooks/05_prompting_and_decode_sweeps.ipynb)
- Repo: [docs/analysis/prompting_findings.md](../../analysis/prompting_findings.md)
- Repo: [docs/execution/NOTEBOOKS.md](../../execution/NOTEBOOKS.md)
