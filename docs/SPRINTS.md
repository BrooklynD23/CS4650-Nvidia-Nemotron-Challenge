# Sprint Plan (Issue-Driven)

We’ll run this project as a sequence of small sprints where every deliverable maps to a GitHub issue.

## Issue Map

- Sprint 0
  - #1 Confirm Kaggle constraints
  - #2 Pull + review konbu17 notebook
  - #3 Dataset ingestion + canonical schema
  - #4 Baseline eval harness + answer normalization
  - #5 Adapter packaging + submission dry-run
- Sprint 1
  - #6 Prompt template + decode sweep harness
  - #7 Trajectory collection + error analysis dataset
- Sprint 2
  - #8 Teacher/solver framework
  - #9 Bit-manipulation solver
  - #10 Synthetic data generator (Tong-style)
- Sprint 3
  - #11 SFT training pipeline (TRL+PEFT), rank<=32, masking
- Sprint 4
  - #12 Submission runbook + provenance

## Sprint 0: Repo + Baseline Harness

**Goal:** make it possible to run a baseline eval locally and have a place for artifacts.

- Deliverables:
  - repo scaffold (`src/`, `notebooks/`, `configs/`, `data/`, `adapters/`)
  - dataset ingestion (Kaggle + optional mirror)
  - baseline eval script with per-category accuracy
  - submission packaging dry-run (even if not yet accepted)

## Sprint 1: Prompting + Decode Controls

**Goal:** squeeze free performance by prompt and decoding changes (no training).

- Deliverables:
  - standardized prompt template
  - parameter sweeps (temperature/top_p/thinking budget)
  - eval report with confidence / variance notes

## Sprint 2: Teacher Generation (Tong-Style Track)

**Goal:** build a teacher that is correct and produces consistent reasoning traces.

- Deliverables:
  - category-by-category solver prototypes (with a failure dashboard)
  - synthetic dataset generator (reasoning + answer)
  - quality filters and stratified splits

## Sprint 3: LoRA SFT + Ablations

**Goal:** train and validate adapters that outperform prompting-only baselines.

- Deliverables:
  - reproducible training config(s)
  - ablation matrix: answer-only vs reasoning+answer vs mixed; masking variants
  - golden-set regression gates

## Sprint 4: Submission Hardening

**Goal:** repeatable submissions with provenance and rollback.

- Deliverables:
  - adapter metadata + versioning
  - automated packaging checks
  - submission checklist and runbook
