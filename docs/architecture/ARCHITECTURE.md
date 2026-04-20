# Architecture (Target MVP)

This is the intended architecture for an MVP that can:

1. load the competition dataset
2. run a repeatable baseline evaluation
3. generate synthetic teacher traces (optional but recommended)
4. fine-tune a LoRA adapter
5. evaluate locally and package a Kaggle submission artifact

## Top-Level Components

### 1) Data Layer

**Goal:** a single source of truth for dataset versions and splits.

- Ingest:
  - Kaggle dataset (official)
  - Optional mirrors (Hugging Face) for local dev
- Normalize to internal schema:
  - `id: str`
  - `category: str`
  - `prompt: str`
  - `answer: str`
  - `source: str` (kaggle|hf|synthetic)
- Split strategy:
  - `train`: official training rows
  - `val`: held-out set (stratified by category)
  - `golden`: 20-50 handpicked regression prompts (must never regress)

### 2) Prompting + Formatting Layer

**Goal:** make “what we ask the model” explicit and versioned.

- Prompt templates live in code (and mirrored in docs).
- Model-specific knobs are configurable:
  - reasoning/thinking enablement mechanism (varies by Nemotron variant)
  - max thinking tokens / budget
  - decoding params (temperature, top_p)
- Output normalization:
  - exact-match string normalization rules for evaluation
  - category-specific parsers if needed (e.g., binary strings)

### 3) Teacher / Solver Layer (Competitive Edge)

**Goal:** produce high-quality *correct* targets, optionally with structured reasoning traces.

Two complementary tracks:

1. **Algorithmic solvers per category** (program induction / search / constraint solving)
2. **LLM-based teachers** for cases where solvers are unreliable (budget-capped)

Outputs:

- `teacher_answer`
- optional `teacher_reasoning` in a consistent “style” (for SFT)
- metadata:
  - confidence
  - rule_found vs rule_unknown
  - runtime and failure mode

### 4) Training Layer (LoRA / QLoRA)

**Goal:** train a small adapter that mimics the teacher behavior and generalizes.

- Training data variants (for ablation):
  - answer-only (no reasoning)
  - reasoning + answer
  - mixed (curriculum: answer-only -> reasoning+answer)
- Loss masking:
  - always mask prompt/user tokens
  - optionally mask parts of reasoning (if it harms generalization)
- Adapter config:
  - base model configurable
  - target_modules discovered programmatically (no hard-coded names)

### 5) Evaluation Layer

**Goal:** make improvements measurable and avoid self-deception.

- Metrics:
  - overall exact-match accuracy
  - per-category accuracy
  - “format validity” rate (if applicable)
  - latency / token usage (thinking budget sensitivity)
- Required checks:
  - golden set regression gate
  - seed-controlled determinism for evaluation runs

### 6) Packaging + Submission Layer

**Goal:** always produce a valid submission artifact.

- “export adapter” step that writes:
  - adapter weights (`.safetensors`)
  - adapter config
  - a metadata card (git commit, dataset version, eval scores)
- A “submission dry run” that validates:
  - expected files exist
  - sizes are within competition constraints

## Non-Goals (for MVP)

- GRPO/RLVR: keep on the roadmap, but don’t block a first strong SFT baseline.
- Large-scale external datasets: only after we can win against our own held-out split.

