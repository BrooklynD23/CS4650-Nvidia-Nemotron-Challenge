# Architecture (Target MVP)

This is the intended architecture for an MVP that can:

1. load the competition dataset
2. run a repeatable baseline evaluation
3. generate synthetic teacher traces (optional but recommended)
4. fine-tune a LoRA adapter
5. evaluate locally and package a Kaggle submission artifact

## Architectural Direction

The foundation phase should assume the competition is closer to **category-specific rule induction** than generic chain-of-thought math. That affects four design choices:

- keep output normalization centered on exact-match contracts rather than `\\boxed{}` formatting
- keep the base model id and tokenizer behavior configurable until the Kaggle rules are verified
- treat solver and teacher interfaces as category-aware plugins rather than one monolithic reasoning prompt
- make provenance a first-class artifact so reviewers can trace every notebook, evaluation, and submission assumption

## Shared Contracts

### Canonical data contract

```python
ReasoningExample = {
    "id": str,
    "category": str,
    "prompt": str,
    "answer": str,
    "source": str,
    "split": str,
    "metadata": dict,
}
```

### Evaluation artifact contract

```python
EvalRecord = {
    "run_id": str,
    "model_id": str,
    "prompt_template_id": str,
    "category": str,
    "gold": str,
    "prediction": str,
    "normalized_prediction": str,
    "correct": bool,
    "latency_ms": float,
    "tokens_in": int,
    "tokens_out": int,
    "seed": int,
}
```

### Packaging contract

```python
PackageManifest = {
    "base_model_id": str,
    "adapter_rank": int,
    "dataset_version": str,
    "eval_sha": str,
    "artifact_paths": dict,
    "created_at": str,
}
```

### Future solver contract

```python
def solve(prompt: str) -> dict:
    return {
        "answer": str,
        "confidence": float,
        "metadata": dict,
    }

def verify(pred: str, gold: str) -> bool:
    ...
```

## Top-Level Components

### 1) Data Layer

**Goal:** a single source of truth for dataset versions and splits.

- Ingest:
  - Kaggle dataset (official)
  - Optional mirrors (Hugging Face) for local dev
- Normalize to internal schema:
  - use the `ReasoningExample` contract above
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
  - all eval outputs should be serializable as `EvalRecord`

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
  - a metadata card shaped like `PackageManifest`
- A “submission dry run” that validates:
  - expected files exist
  - sizes are within competition constraints

## Notebook-First Foundation

Before the heavier training phases begin, the repo should maintain a notebook-first documentation harness:

- every notebook must explain its purpose to non-technical readers
- every notebook must cite its sources inline
- every notebook that changes shared assumptions must link back to a GitHub issue and update the relevant contract here

This keeps architecture review grounded in concrete artifacts rather than only planning prose.

## Non-Goals (for MVP)

- GRPO/RLVR: keep on the roadmap, but don’t block a first strong SFT baseline.
- Large-scale external datasets: only after we can win against our own held-out split.
