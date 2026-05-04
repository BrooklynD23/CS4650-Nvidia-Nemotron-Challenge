---
title: LLM Basics for This Project
audience: beginner
page_type: concept
status: conceptual
last_reviewed: 2026-04-21
repo_sources:
  - docs/architecture/ARCHITECTURE.md
  - docs/planning/plan_v0.2.md
external_sources:
  - https://huggingface.co/course
  - https://huggingface.co/papers/1706.03762
  - https://huggingface.co/papers/2106.09685
---

# LLM Basics for This Project

## Why This Page Exists

This repo uses terms like model, inference, fine-tuning, adapter, and context
window constantly. If those words are unfamiliar, the rest of the project can
sound more mysterious than it really is.

## The Short Version

A large language model is a system trained to predict the next piece of text
based on everything that came before it. In practice, that lets it answer
questions, complete instructions, and follow patterns it has learned from data.

## The Core Ideas

### 1. Base model

The base model is the starting AI model before the team customizes it. In this
repo, the project is trying to improve an NVIDIA Nemotron model rather than
build a new model from zero.

### 2. Inference

Inference is the moment when the model is asked a question and produces an
answer. It is the "use the model" phase, not the "train the model" phase.

### 3. Fine-tuning

Fine-tuning means taking a model that already knows a lot of general language
patterns and teaching it to behave better on a narrower task. In this project,
that narrower task is solving the Nemotron reasoning challenge more reliably.

### 4. Adapters

Instead of changing every parameter in the full model, teams often train a
smaller add-on. This repo uses that idea because it is cheaper, easier to test,
and closer to the competition submission format.

## Why This Matters Here

The technical plan in `docs/planning/plan_v0.2.md` and the architecture doc in
`docs/architecture/ARCHITECTURE.md` both assume the project will learn through
small, repeatable improvements: measure a baseline, improve data or prompting,
train an adapter, evaluate carefully, then package a submission.

That workflow only makes sense if you understand the difference between:

- using a model (`inference`)
- teaching a model (`fine-tuning`)
- and attaching a smaller learned component (`adapter`)

## Sources

- Repo: [docs/architecture/ARCHITECTURE.md](../../architecture/ARCHITECTURE.md)
- Repo: [docs/planning/plan_v0.2.md](../../planning/plan_v0.2.md)
- External: https://huggingface.co/course
- External: https://huggingface.co/papers/1706.03762
- External: https://huggingface.co/papers/2106.09685
