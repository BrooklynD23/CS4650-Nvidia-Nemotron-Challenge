# Adversarial Review (MVP + Approach)

This is an intentionally critical review of our current MVP and the architecture implied by `docs/planning/plan_v0.2.md`, grounded in what we can observe about the competition task shape and what has worked for top community solutions.

## What The Competition Actually Looks Like (High-Signal Clues)

From public dataset mirrors, the task is not “classic math CoT”:

- Prompts describe a *hidden rule* in a themed story (“Alice’s Wonderland”) and provide multiple **input -> output** examples.
- You must infer the transformation rule and produce the output for a new input.
- Categories include: `bit_manipulation`, `text_cipher`, `unit_conversion`, `numeral_system`, `physics_gravity`, and `equation_*`.

This makes the competition closer to **program induction / rule inference** than “solve math word problems.”

## Critical Review of Our Current “MVP”

### MVP Status: We Don’t Have One Yet

Current repo (before this init work) was effectively:

- `README.md`
- planning docs (`docs/planning/plan_v0.2.md` etc.)

No ingestion, no baseline eval, no reproducible training, no packaging.

### The Plan’s Biggest Mismatch Risks

1. **Base model ambiguity**
   - Planning doc assumes a 4B Nano checkpoint.
   - Public competition mentions “Nemotron-3 Nano baseline,” which usually refers to the 30B-A3B Nano.
   - If we pick the wrong base model locally, we will optimize the wrong tokenizer/chat template/module names and waste weeks.

2. **LoRA rank constraints vs plan**
   - Several public references mention **LoRA rank <= 32**.
   - `plan_v0.2.md` recommends `r=64` (invalid if the constraint holds).
   - This is a hard constraint; we need to treat it as a first-class requirement and design around it.

3. **Evaluation format mismatch**
   - `plan_v0.2.md` heavily assumes math benchmarks and `\\boxed{}` extraction.
   - The competition answers look like plain strings (binary, cipher text, numeric results) and likely use strict exact match.
   - If we keep `\\boxed{}`-centric plumbing, we risk training the model into the wrong output format.

4. **Over-indexing on generic reasoning**
   - Prompting/CoT improvements matter, but the win condition likely hinges on:
     - category-specific structure learning
     - algorithmic reliability (bit ops, ciphers, conversions)
     - in-distribution synthetic augmentation

## What “Good” Looks Like (Observed from a Winner-Adjacent Open Pipeline)

Tong Hui Kang’s public repo declares it was the **Progress Prize winning** submission.

The high-level pattern is:

1. generate/solve problems (or “investigate” rules)
2. augment in-distribution examples
3. build a masked training corpus with careful token accounting
4. SFT a LoRA adapter
5. package/upload adapter

Notably, it emphasizes **token-level masking** and category tracking.

## The Fastest Path to a Competitive Architecture

### 1) Treat This as “Category-Specific Program Induction”

Do not build a single monolithic “reasoning prompt.”

Instead:

- Build a **router** that detects category and selects:
  - a solver (preferred) or
  - a teacher prompting template tuned for that category

### 2) Build Teachers That Are Actually Correct

SFT only helps if targets are correct. For this benchmark, correctness can often be verified:

- bit manipulation: exact string match
- unit conversion: numeric answer with strict formatting; consider canonicalization
- ciphers: deterministic decode
- numeral systems: deterministic base conversion
- equations: symbolic/numeric solve

This suggests a hybrid teacher:

- algorithmic solver first
- LLM teacher fallback only when solver fails (budget capped)

### 3) Make Loss Masking a First-Class Feature

If you don’t mask prompts/user tokens, you waste gradient budget.

Also consider masking *parts of the reasoning* if it causes verbosity or overfitting.

### 4) Don’t Fine-Tune MoE Routers Blindly (30B-A3B)

For MoE base models, router/gating layers are fragile.

Default to LoRA on well-understood projections only; avoid router modules unless you have evidence it helps.

## Review of konbu17 Notebook (Blocking Limitation)

I could not fetch the notebook source from Kaggle directly in this environment (the Kaggle page did not render via our web tool).

What we can still do immediately:

- Pull the notebook via Kaggle API (`kaggle kernels pull konbu17/nemotron-tong-style-cot-sft-updated-v2`) once credentials are configured.
- Extract:
  - base model id
  - dataset used (official vs mirror vs synthetic)
  - LoRA config (rank, targets)
  - masking strategy
  - any category-specific tricks
  - eval protocol (how they got ~0.85)

## Action Items (What To Build Next)

1. dataset ingestion + schema normalization (Kaggle + mirror)
2. baseline eval harness (per-category accuracy + strict answer normalization)
3. notebook pull + reproduction for:
   - konbu17 notebook
   - Tong Hui Kang repo ideas (masking + augmentation)
4. teacher + augmentation pipeline per category (small, verifiable solvers)
5. LoRA SFT pipeline constrained to rank <= 32
6. packaging + submission dry-run
