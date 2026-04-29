# Competition Facts

If anything here conflicts with the Kaggle competition page, **Kaggle wins**.

---

## Verified (snapshot: 2026-04-29)

Facts confirmed directly from the Kaggle competition overview page and the official Submission Demo notebook (`ryanholbrook/nvidia-nemotron-submission-demo`). These are safe to hard-code in downstream code and configs.

### Base Model

- **KaggleHub artifact path:** `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
- **Model family:** NVIDIA Nemotron-3-Nano-30B (BF16)
- **HF slug:** unconfirmed — use the KaggleHub path above for Kaggle evaluation environments.

### Base-Model Load Recipe (from submission demo)

```python
import kagglehub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
```

**Not confirmed from demo** (evaluator-internal only): tokenizer special-token handling, chat template, attention backend, thinking-control flags.

### Scoring / Answer-Normalization Contract (from competition overview)

- Model is instructed to place its final answer in `\boxed{}` LaTeX.
- Metric extracts the final answer: prefers `\boxed{}` content, falls back to other heuristic patterns, then last numeric value.
- **Correct if:** exact string match **OR** within relative numerical tolerance of 1e-3.
- Final score = proportion of correctly answered questions (no partial credit indicated).
- Reasoning text in the response is **not penalized** — only the extracted final answer is graded.

### Evaluator Parameters (from competition overview)

| Parameter              | Value  |
|------------------------|--------|
| max_lora_rank          | 32     |
| max_tokens             | 7680   |
| top_p                  | 1.0    |
| temperature            | 0.0    |
| max_num_seqs           | 64     |
| gpu_memory_utilization | 0.85   |
| max_model_len          | 8192   |

### LoRA Constraints

- **LoRA adapter only** — no full-weight submissions.
- **Rank cap:** `r ≤ 32` (enforced by evaluator via `max_lora_rank=32`).
- **Target modules (from submission demo):** `.*\.(in_proj|out_proj|up_proj|down_proj)$` — 4 modules.
  - Note: konbu17 baseline used 9 modules. Competition rules do **not** appear to restrict target modules beyond rank cap. Demo is the reference.
- **Adapter dtype:** `bf16` (demo trains and saves in bfloat16; evaluator loads in bfloat16).
- **Competition-enforced dtype limits:** unconfirmed — treat bf16 as required.

### Submission Packaging (from submission demo)

```bash
model.save_pretrained("/kaggle/working")
cd /kaggle/working && zip -m submission.zip *
```

**Zip root must contain:**
- `adapter_config.json`
- `adapter_model.safetensors`

**Not verified:** whether Kaggle enforces a file-size limit on the zip or adapter file.

### Deadlines

- **Final deadline:** 2026-06-15 23:59 (timezone unconfirmed; treat as UTC until verified).
- **Entry deadline:** 2026-05-10 (confirmed by team).
- **Daily submission limit:** 5 submissions per day (confirmed by team).
- **Final-submission selection behavior:** unconfirmed — out of collaborators' access scope.

### Compute Environment

- Evaluation runs on **Google Cloud G4 VMs** with **NVIDIA RTX PRO 6000 Blackwell GPUs**.
- Inference engine: **vLLM**.

---

## Out of Scope / Unable to Confirm (as of 2026-04-29)

The following items are not accessible to the team via the submission demo, competition overview, or available public sources. They are noted here so downstream work does not block on them.

- **Exact HuggingFace slug** — use KaggleHub path `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` for all Kaggle-environment code.
- **Evaluator tokenizer special-token handling, chat template, attention backend, thinking-control flags** — evaluator-internal; cannot be confirmed without running the Kaggle grader. Design code to be configurable.
- **Whether competition rules restrict LoRA target modules** — no explicit restriction found. Treat the demo's 4-module set as the reference; deviation is at the team's discretion.
- **Whether adapters must be exactly bf16** — treat bf16 as required based on demo evidence.
- **Submission zip / adapter file-size limit** — no limit found; monitor submission size and flag if adapter exceeds ~1 GB.
- **Final-submission selection behavior** — not accessible from available sources; assume the last submission counts unless stated otherwise.
- **Final deadline timezone** — treat 2026-06-15 23:59 as UTC.

---

## Historical Working Notes (pre-verification)

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
