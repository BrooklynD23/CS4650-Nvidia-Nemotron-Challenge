# Competition Facts (Working Notes)

This document is our **best-effort snapshot** of the competition constraints. If anything here conflicts with the Kaggle competition page, **Kaggle wins**.

## Key Dates

- **Entry deadline:** 2026-06-08 (public Kaggle announcement copy on LinkedIn; **UNVERIFIED** here)
- **Final deadline:** 2026-06-15 23:59:00 (verified via Kaggle CLI on 2026-04-20)

## Official Competition Data Files (via Kaggle API)

Verified via Kaggle CLI on 2026-04-20:

- `train.csv`
- `test.csv`

## Compute Environment

- Hosted evaluation compute is reported as **Google Cloud G4 VMs** with **NVIDIA RTX PRO 6000 Blackwell GPUs**.

## Adapter Constraints (Likely)

- **LoRA adapter only**
- **LoRA rank <= 32** (reported publicly; confirm on Kaggle Rules page)

## Dataset Shape (Observed from Public Mirrors)

Public mirrors show a dataset that looks like:

- Columns: `id`, `prompt`, `answer`
- Categories / splits (examples):
  - `bit_manipulation`
  - `unit_conversion`
  - `text_cipher`
  - `numeral_system`
  - `physics_gravity`
  - `equation_numeric`
  - `equation_symbolic`

The prompts appear to include **multiple I/O examples** (“input -> output”) and ask for the output of a new input. This makes the benchmark closer to *program induction / rule inference* than classic math word problems.

## Base Model (Open Question)

The competition is described as starting with a “Nemotron-3 Nano baseline”.

The most likely base checkpoint is the Nemotron 3 Nano model:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (Nemotron-3-Nano-30B)

However, this repo has also referenced a smaller Nano checkpoint (`NVIDIA-Nemotron-3-Nano-4B-BF16`) in planning docs. We must confirm the exact base model used by Kaggle evaluation and then standardize around it.

- Candidate A: `nvidia/nemotron-3-nano-30b-a3b` (Nemotron 3 Nano)
- Candidate B: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (Nemotron 3 Nano 4B)

Until verified, design code to make the base model a config knob and avoid hard-coding:

- chat template quirks
- “thinking” control mechanism
- tokenizer special tokens

## Submission Format (Assumed)

Current assumption (from public descriptions + common competition pattern): submit a **LoRA adapter** that is merged into the base model during evaluation. Confirm the exact expected directory structure / filenames on Kaggle before implementing the final packaging script.

## Evaluation Metric (Assumed)

Likely **accuracy** on a hidden test set of the same puzzle families, but confirm:

- whether scoring is overall accuracy vs macro-averaged by category
- whether partial credit exists
- whether exact string match rules apply (whitespace/case/formatting)

## External Reference Solutions / Artifacts

- Tong Hui Kang “Progress Prize” repo (public): https://github.com/tonghuikang/nemotron
- A public Hugging Face mirror dataset (for local development/EDA): https://huggingface.co/datasets/Taurine511/nvidia-nemotron-model-reasoning-challenge
