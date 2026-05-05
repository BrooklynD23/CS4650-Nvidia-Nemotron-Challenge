# NVIDIA Nemotron Model Reasoning Challenge - Plan v0.2

**Version**: 0.2 (Post Expert Review)
**Date**: 2026-04-19
**Status**: Active
**Previous**: v0.1 (retired) | **Review**: [plan_review.md](plan_review.md)

---

## Context

This is a **CS4650 Capstone project** to compete in the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle. The competition requires improving the reasoning accuracy of **Nemotron-3-Nano-4B** (a 3.97B parameter hybrid Mamba-2 + Transformer model) on a novel benchmark. Participants submit a **LoRA adapter** that gets merged with the base model for evaluation on G4 VMs (NVIDIA RTX PRO 6000 Blackwell GPUs).

The goal is to explore **multiple approaches** (prompting, data curation, synthetic data, RL, fine-tuning) for maximum learning and competitive performance. If time/compute is constrained, see [Scope-Cutting Priority](#scope-cutting-priority) for what to drop.

---

## Competition Summary

| Detail | Value |
|--------|-------|
| **Platform** | Kaggle |
| **Base Model** | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| **Architecture** | Hybrid Mamba-2 + Transformer (only 4 attention layers) |
| **Parameters** | 3.97B total, BF16 precision |
| **Context Window** | 262K tokens |
| **Submission** | LoRA adapter weights (safetensors) |
| **Infrastructure** | G4 VMs with NVIDIA RTX PRO 6000 Blackwell GPUs |
| **Eval** | Reasoning accuracy on novel benchmark |
| **Reasoning Control** | `enable_thinking` param / `<think>` (token 12) `</think>` (token 13) |
| **Submission Demo** | [Kaggle Notebook](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo) |

### Baseline Benchmarks (Reasoning ON)
| Benchmark | Score |
|-----------|-------|
| MATH500 | 95.4% |
| AIME25 | 78.5% |
| GPQA | 53.2% |
| LCB | 51.8% |

---

## Compute & Constraints

### Available Hardware

| Resource | VRAM | Best For | Notes |
|----------|------|----------|-------|
| **Colab Pro** | A100 (40GB) | Prototyping, experiments, QLoRA | Extended sessions, but still has limits |
| **RTX 3080 local** | 10GB | Local dev, small inference, QLoRA | Primary local GPU |
| **RTX 3060 campus** | 12GB | Fallback local GPU | Backup option |
| **HPC cluster** | Varies | Long training runs (SFT, GRPO) | Job queue — submit and wait |

### Phase-to-Hardware Mapping

| Phase | Recommended Hardware | Fallback |
|-------|---------------------|----------|
| 0 (Setup) | Local (any) | — |
| 1 (Baseline) | Colab Pro / Local RTX 3080 | RTX 3060 |
| 2 (Prompting) | Colab Pro | Local RTX 3080 |
| 3 (Data Curation) | Local CPU (no GPU needed) | Colab CPU |
| 4 (SFT LoRA) | **HPC cluster** | Colab Pro (QLoRA) |
| 5 (Synthetic Data) | Local CPU + APIs | Colab |
| 6 (GRPO RL) | **HPC cluster** | Colab Pro (reduced group size) |
| 7 (Evaluation) | Colab Pro | Local RTX 3080 |
| 8 (Submission) | Local (any) | Colab |

### Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Llama-Nemotron Post-Training Dataset | ~130 GB | Can download subsets only |
| Puzzle-KD Dataset v2 | ~2.7 GB | Pre-filtered, start here |
| Model weights (BF16) | ~8 GB | Downloaded once |
| Adapters + checkpoints | ~5 GB | Grows with experiments |
| Processed/synthetic data | ~10 GB | Estimated |
| **Total** | **~155 GB** | Verify WSL2 disk allocation |

> **WSL2 Note**: Check available disk with `df -h /mnt/c`. If tight, store datasets on a separate drive or use the HPC cluster's storage.

### Open Questions (Remaining)

1. **Competition deadline**: Need to anchor the timeline — check Kaggle competition page
2. **Team size**: Solo or team capstone? Affects parallelization of approaches
3. **HPC job limits**: Max GPU time per job? Max concurrent jobs?

---

## Phase 0: Project Setup & Infrastructure

### 0.1 Repository Structure
```
CS4650-Nvidia-Nemotron-Challenge/
├── README.md
├── requirements.txt
├── .env.example                  # HF_TOKEN, WANDB_API_KEY, KAGGLE_KEY
├── .gitignore                    # Exclude data/, adapters/*.safetensors, .env
│
├── configs/                      # Training configurations
│   ├── lora_baseline.yaml        # LoRA r=32, alpha=32 (#14 rank cap)
│   ├── lora_qlora.yaml           # QLoRA 4-bit for consumer GPUs
│   ├── grpo_config.yaml          # GRPO RL training
│   └── data_curation.yaml        # NeMo Curator config
│
├── data/                         # Data directory (gitignored)
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Curated/filtered data
│   ├── synthetic/                # Generated synthetic data
│   └── eval/                     # Held-out validation + golden test set
│
├── notebooks/                    # Jupyter notebooks (one per approach)
│   ├── 01_baseline_inference.ipynb
│   ├── 02_prompting_strategies.ipynb
│   ├── 03_data_exploration.ipynb
│   ├── 04_data_curation.ipynb
│   ├── 05_synthetic_data_gen.ipynb
│   ├── 06_sft_lora_training.ipynb
│   ├── 07_qlora_consumer_gpu.ipynb
│   ├── 08_grpo_rl_training.ipynb
│   ├── 09_evaluation.ipynb
│   └── 10_submission.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # Dataset loading utilities
│   │   ├── curator.py            # Data curation/filtering
│   │   └── synthetic.py          # Synthetic data generation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py        # SFT LoRA training
│   │   ├── grpo_trainer.py       # GRPO RL training
│   │   └── utils.py              # Training utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py          # Evaluation on benchmarks
│   │   └── metrics.py            # Accuracy metrics
│   └── inference/
│       ├── __init__.py
│       ├── prompting.py          # Prompting strategies
│       └── submission.py         # Kaggle submission format
│
├── experiments/                  # Experiment tracking
│   └── README.md
├── adapters/                     # Saved LoRA adapter weights
│   └── .gitkeep
└── docs/
    ├── planning/
    │   ├── plan_v0.2.md          # This file
    │   └── plan_review.md        # Expert panel review
    ├── architecture/
    ├── analysis/
    └── execution/
```

### 0.2 Core Dependencies
```
torch==2.4.0                     # Pin for reproducibility
transformers>=4.48.3
datasets>=2.20.0
peft>=0.14.0
trl>=0.16.0
accelerate>=1.0.0
bitsandbytes>=0.45.0             # QLoRA quantization
unsloth                          # Efficient fine-tuning (Nemotron support)
vllm>=0.15.1                     # Fast inference (verify Mamba-2 support)
wandb                            # Experiment tracking
jupyter
pandas numpy matplotlib
kaggle                           # Kaggle CLI
```

> **Note on vLLM**: Verify that vLLM supports the Mamba-2 hybrid architecture before relying on it. If not, fall back to HuggingFace `generate()` for inference.

### 0.3 Reproducibility Setup
```python
import torch
import random
import numpy as np
from transformers import set_seed

SEED = 42

def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
```

### 0.4 WandB Configuration
```python
import wandb

wandb.init(
    project="nemotron-reasoning-challenge",
    config={
        "model": "NVIDIA-Nemotron-3-Nano-4B-BF16",
        "approach": "baseline",  # Update per experiment
        "seed": 42,
    },
    tags=["cs4650", "capstone"],
)
```

### 0.5 Git & Environment Setup
- `.gitignore`: Python/ML standard (exclude `data/`, `adapters/*.safetensors`, `.env`)
- `.env.example`: `HF_TOKEN`, `WANDB_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
- Download competition data: `kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge`

### Done When
- [ ] All imports succeed: `python -c "import torch, transformers, peft, trl; print('OK')"`
- [ ] `.env` configured with valid tokens
- [ ] WandB test run logs successfully
- [ ] Competition data downloaded
- [ ] Storage verified: `df -h` shows sufficient space

---

## Phase 1: Baseline & Exploration (MVP)

**Requires**: Phase 0 complete
**Hardware**: Colab Pro or local RTX 3080

### 1.1 Baseline Inference
- Load `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` with `trust_remote_code=True`
- Run inference with reasoning ON (`enable_thinking=True`) and OFF
- Evaluate on competition benchmark questions
- Record baseline accuracy as reference point

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

messages = [{"role": "user", "content": "Solve: ..."}]
tokenized = tokenizer.apply_chat_template(
    messages, tokenize=True, enable_thinking=True,
    add_generation_prompt=True, return_tensors="pt"
).to(model.device)
```

### 1.2 Smoke Test

Run this concrete test to verify the pipeline works end-to-end:

**Problem**: "What is 2^10 mod 7?"

**Expected output** (with reasoning ON):
```
<think>
I need to compute 2^10 mod 7.
2^1 = 2
2^2 = 4
2^3 = 8 ≡ 1 (mod 7)
Since 2^3 ≡ 1 (mod 7), then 2^10 = 2^(3×3+1) = (2^3)^3 × 2^1 ≡ 1^3 × 2 = 2 (mod 7)
</think>

The answer is \boxed{2}.
```

**Verification script**:
```python
import re

def extract_boxed_answer(text):
    """Extract answer from \\boxed{}, handling one level of nested braces."""
    match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    return match.group(1).strip() if match else None

def verify_answer(model_output, expected):
    """Check if model output contains the correct boxed answer."""
    answer = extract_boxed_answer(model_output)
    return answer == expected

# Test
output = generate(model, tokenizer, "What is 2^10 mod 7?")
assert verify_answer(output, "2"), f"Expected \\boxed{{2}}, got: {extract_boxed_answer(output)}"
print("Smoke test PASSED")
```

### 1.3 Dataset Exploration
- Explore competition-provided data
- Analyze `nvidia/Llama-Nemotron-Post-Training-Dataset` (32M+ samples, 130GB)
  - Subsets: math (22M), code (10M), science (708K), chat (39K), safety (31K)
- Analyze `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` (851K samples, 2.74GB)
  - Pre-filtered, English-only, reasoning=off, 95/5 train/val split
- Identify most relevant subsets for reasoning improvement

### 1.4 Submission Pipeline
- Study [submission demo notebook](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
- Build end-to-end: load base model -> apply LoRA adapter -> generate answers -> format submission
- Submit "dummy" adapter to verify pipeline works

### Done When
- [ ] Baseline accuracy recorded on MATH500, AIME25, GPQA (minimum 3 benchmarks)
- [ ] Smoke test passes (correct boxed answer extracted and verified)
- [ ] Dummy adapter submission accepted on Kaggle
- [ ] Dataset subsets identified and prioritized
- [ ] Results logged to WandB

---

## Phase 2: Prompting Strategies (Zero Compute Cost)

**Requires**: Phase 1 baseline accuracy recorded
**Hardware**: Colab Pro or local

### 2.1 System Prompt Engineering
- `enable_thinking=True` with `temperature=1.0, top_p=0.95` (NVIDIA recommended)
- Custom system prompts for math domains
- Structured output forcing `\boxed{}` format

### 2.2 Chain-of-Thought Prompting

| Strategy | Description | Literature Estimate* |
|----------|-------------|---------------------|
| Zero-shot CoT | "Let's think step by step" | +5-10% |
| Few-shot CoT | 3-5 solved examples as context | +10-15% |
| Program of Thoughts | Generate code to solve problems | +20% on arithmetic |
| Self-consistency | Sample N=64, majority vote | +15-20% |

> *\*Literature estimates from general LLM research. Not validated on Nemotron-3-Nano-4B. Actual gains on this model may differ significantly. Validate each strategy against Phase 1 baseline.*

### 2.3 Inference-Time Compute Scaling
- Best-of-N sampling (N=8, 16, 32, 64)
- Majority voting: `pass@1` vs `maj@64`
- Temperature sweep: [0.6, 0.8, 1.0, 1.2]
- Top-p sweep: [0.9, 0.95, 0.99]

### Done When
- [ ] Each prompting strategy tested on 3+ benchmarks
- [ ] Comparison table with measured deltas vs Phase 1 baseline (not literature estimates)
- [ ] Best prompting strategy identified with statistical significance (run each 3x, report mean +/- std)
- [ ] Results logged to WandB with tags per strategy

---

## Phase 3: Data Curation & Filtering

**Requires**: Phase 1 dataset exploration complete
**Hardware**: Local CPU (no GPU needed)

### 3.1 Validation Set Reservation (DO THIS FIRST)

Before any training data is prepared, reserve evaluation data:

```python
# Reserve held-out validation set BEFORE any training
import random

random.seed(42)

# 200 diverse problems for validation during training
val_set = sample_diverse_problems(n=200, categories=["math", "code", "science", "logic"])
save_jsonl(val_set, "data/eval/validation_200.jsonl")

# 20 "golden" regression test problems that must ALWAYS pass
golden_set = select_golden_problems(n=20, criteria="high_confidence_solvable")
save_jsonl(golden_set, "data/eval/golden_20.jsonl")
```

**Golden set criteria**: Problems the base model answers correctly with >90% consistency. Any adapter that breaks these is regressing.

### 3.2 NeMo Curator Pipeline
Following the [NVIDIA blog recipe](https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/):
1. Select from `math_v1.1` and `chat` subsets
2. Filter by language (English only)
3. Enforce answer format (`\boxed{}` for math)
4. Exclude refusal samples with empty `<think></think>` tags
5. Apply consistent chat template
6. Sort by completion length (curriculum learning)

**Target**: ~500K curated samples from 32M+ total

```bash
python main.py \
    --input-dir "/path/to/dataset" \
    --tokenizer "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" \
    --max-token-count 8192 \
    --n-workers 8
```

### 3.3 Quality Filtering
- Remove incorrect/inconsistent solutions
- Verify math answers programmatically
- Filter non-empty, coherent reasoning traces
- Deduplicate similar problems

### 3.4 Curriculum Design
- Progressive difficulty: GSM8K-level -> MATH-level -> Olympiad-level
- Domain mixing: 60% math, 20% code, 15% science, 5% chat
- **Critical ratio**: 75% reasoning-on / 25% reasoning-off data

### Done When
- [ ] Validation set (200 problems) and golden set (20 problems) saved to `data/eval/`
- [ ] Curated dataset saved with stats: total size, category distribution, avg token length
- [ ] Quality metrics: % of samples with valid `\boxed{}` answers, % with non-empty reasoning
- [ ] Deduplication complete (report % removed)
- [ ] Curriculum ordering verified (difficulty ascending)

---

## Phase 4: Supervised Fine-Tuning (SFT) with LoRA

**Requires**: Phase 3 curated dataset ready, validation set reserved
**Hardware**: **HPC cluster** (primary) or Colab Pro with QLoRA (fallback)

### 4.1 LoRA Configuration (Verified #14 Contract)
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                          # Max rank allowed by verified #14 contract
    lora_alpha=32,                 # Match konbu17-safe baseline; tune only after PM signoff
    target_modules=[
        "q_proj", "v_proj",        # Transformer attention layers
        "x_proj", "in_proj", "out_proj"  # Mamba-2 layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

> **Verify target modules**: Run `model.named_modules()` to confirm exact module names for the hybrid architecture before training.

### 4.2 Training Hyperparameters
| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 1e-4 | NVIDIA blog |
| Scheduler | Cosine | NVIDIA blog |
| Warmup | 5% of total steps | NVIDIA blog |
| Batch size | 256 (gradient accumulation) | NVIDIA blog |
| Max tokens | 8192 | NVIDIA blog |
| Weight decay | 0.001 | NVIDIA blog |
| Training steps | 2000+ | NVIDIA blog |
| Max completion tokens | 16384 | NVIDIA blog |
| **Random seed** | **42** | Reproducibility |
| **Eval steps** | **100** | Checkpoint + validation |
| **Save steps** | **500** | Adapter snapshots |

### 4.3 Framework Comparison
| Framework | VRAM (4B) | Speed | Nemotron Support | Best For | Our Hardware |
|-----------|-----------|-------|------------------|----------|-------------|
| **Unsloth** | 8-16GB (QLoRA) | Fastest | Official | Rapid prototyping | RTX 3080/3060, Colab |
| **TRL SFTTrainer** | 16-60GB | Good | Yes | HF ecosystem | HPC cluster |
| **NeMo Framework** | 60-80GB | Good | Official recipes | Production runs | HPC cluster (if 80GB GPUs) |
| **HF PEFT** | 16-60GB | Good | Manual config | Custom setups | HPC cluster |

### 4.4 QLoRA for Consumer GPUs
- 4-bit quantization via `bitsandbytes`
- **RTX 3080 (10GB)**: QLoRA with Unsloth, batch_size=16-32, gradient_accumulation=8
- **RTX 3060 (12GB)**: QLoRA with Unsloth, batch_size=16-32, gradient_accumulation=8
- **Colab Pro A100 (40GB)**: Standard LoRA, batch_size=64-128
- **HPC cluster**: Full LoRA training (framework-dependent)

### 4.5 Mamba-2 Specific Target Modules
For the hybrid architecture, target both layer types:
- **Mamba-2 layers**: `x_proj`, `embeddings`, `in_proj`, `out_proj`
- **Transformer layers**: `q_proj`, `v_proj`

### Done When
- [ ] Training loss converges (final loss < initial loss by >50%)
- [ ] Validation loss on held-out 200 problems decreases or stabilizes (no divergence)
- [ ] Golden set: all 20 problems still answered correctly (no regression)
- [ ] Adapter generates correct `\boxed{}` format on 5+ test problems
- [ ] Adapter saved as `.safetensors` and loads correctly with base model
- [ ] Accuracy on MATH500, AIME25, GPQA compared to Phase 1 baseline
- [ ] Results logged to WandB with loss curves and eval metrics

---

## Phase 5: Synthetic Data Generation

**Requires**: Phase 3 data pipeline working
**Hardware**: Local CPU + API calls (or open-weight models on HPC)

### 5.1 Distillation from Stronger Models

**Cost cap: $20 per generation run. Estimate cost before running.**

Generate reasoning traces using:
| Model | Use Case | Access | Cost |
|-------|----------|--------|------|
| DeepSeek-R1 / R1-0528 | Math CoT generation | HF (free, self-hosted) | GPU time only |
| Qwen3-235B | Diverse reasoning styles | HF / API | Varies |
| Claude / GPT-4 | High-quality explanations | API | ~$5-15/1K problems |

> **Prefer open-weight models** (DeepSeek-R1 on HuggingFace) over paid APIs to control costs. Use paid APIs only for small, high-quality batches.

Pipeline:
1. Collect challenging math problems (AIME, AMC, MATH dataset)
2. Generate solutions with `<think>...</think>` reasoning traces
3. Verify final answers programmatically
4. Filter incorrect solutions
5. Format in Nemotron chat template (`<|im_start|>` format)

### 5.2 Available Synthetic Datasets
| Dataset | Size | Description |
|---------|------|-------------|
| [SYNTHETIC-1](https://www.primeintellect.ai/blog/synthetic-1) | 1.4M traces | Verified reasoning from DeepSeek-R1 |
| [Llama-Nemotron-Post-Training](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) | 32M+ | NVIDIA official post-training data |
| [Puzzle-KD Dataset v2](https://huggingface.co/datasets/nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2) | 851K | Pre-filtered, English-only |

### 5.3 Data Format
```json
{
  "input": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "Solve: What is 2^10 mod 7?"}
  ],
  "output": "<think>I need to compute 2^10 mod 7...</think>\n\nThe answer is \\boxed{2}.",
  "category": "math",
  "reasoning": "on"
}
```

### Done When
- [ ] Synthetic data generated with verified answer accuracy > 90%
- [ ] API costs stayed within $20 cap per run
- [ ] Data formatted in Nemotron chat template
- [ ] Quality comparison: synthetic vs curated data on sample eval

---

## Phase 6: Reinforcement Learning (GRPO)

**Requires**: Phase 4 SFT adapter as starting point
**Hardware**: **HPC cluster** (primary) — GRPO is compute-intensive

### 6.1 GRPO Overview
**GRPO (Group Relative Policy Optimization)** - the algorithm used by DeepSeek-R1:
- No separate value model needed (memory efficient vs PPO)
- Group-based advantage estimation across N generations
- Ideal for tasks with verifiable answers (math)
- Available in [TRL library](https://huggingface.co/docs/trl/main/en/grpo_trainer)

### 6.2 Implementation
```python
from trl import GRPOTrainer, GRPOConfig
import re

config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,          # Group size
    max_new_tokens=4096,
    temperature=1.0,
    learning_rate=1e-5,
)

def extract_boxed_answer(text):
    """Extract answer from \\boxed{}, handling one level of nested braces."""
    match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    return match.group(1).strip() if match else ""

def accuracy_reward(completions, ground_truth):
    """Binary reward: 1 if boxed answer matches, 0 otherwise."""
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        answer = extract_boxed_answer(completion)
        rewards.append(1.0 if answer == truth.strip() else 0.0)
    return rewards

def format_reward(completions):
    """Reward correct use of <think> tags and \\boxed{} format."""
    rewards = []
    for c in completions:
        has_think = "<think>" in c and "</think>" in c
        has_boxed = "\\boxed{" in c
        rewards.append(0.5 * has_think + 0.5 * has_boxed)
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[accuracy_reward, format_reward],
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

### 6.3 Reward Design
| Reward | Weight | Description |
|--------|--------|-------------|
| Accuracy | 0.7 | Binary correctness vs ground truth |
| Format | 0.2 | Correct `<think>` + `\boxed{}` usage |
| Length | 0.1 | Penalty for excessively long traces |

### 6.4 NeMo Gym (Advanced)
- NVIDIA's microservice architecture for RL training
- Pre-built environments: math, code, science, Sudoku
- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
- [Nemotron-3-Nano guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/nemotron-3-nano.md)
- Synchronous GRPO: ~48 hours on single GPU for 4B model

### Done When
- [ ] RL reward curve shows sustained improvement (reward increasing over 500+ steps)
- [ ] Adapter accuracy on MATH500/AIME25 exceeds SFT baseline from Phase 4
- [ ] Golden set: all 20 problems still answered correctly (no regression)
- [ ] No degenerate outputs (empty reasoning, repeated tokens, format collapse)
- [ ] Results logged to WandB with reward curves

---

## Phase 7: Evaluation & Comparison

**Requires**: At least one trained adapter (Phase 4 minimum)
**Hardware**: Colab Pro or local RTX 3080

### 7.1 Benchmarks

| Benchmark | Baseline | Eval Set Size | Min Detectable Effect | Purpose |
|-----------|----------|---------------|----------------------|---------|
| MATH500 | 95.4% | 500 | ~2% | General math |
| AIME25 | 78.5% | ~30 | ~7% (1 question) | Competition math |
| GPQA | 53.2% | ~100 | ~5% | Science reasoning |
| GSM8K | ~85% | 1319 | ~1.5% | Grade school math |
| Golden set | 100% | 20 | Any drop = regression | Regression test |
| Competition | TBD | TBD | TBD | Actual leaderboard |

> **Statistical note**: AIME25 has ~30 problems. A single question is ~3.3%. Report improvements below this threshold as "within noise" and do not make decisions based on them. Use pass@k (k=1,5,10) with 3 evaluation runs per configuration.

### 7.2 Evaluation Protocol

```python
# Deterministic evaluation
def evaluate(model, tokenizer, eval_set, n_runs=3, temperature=0.0):
    """Evaluate with deterministic generation for reproducibility."""
    results = []
    for run in range(n_runs):
        set_all_seeds(SEED + run)
        run_results = []
        for problem in eval_set:
            output = generate(model, tokenizer, problem["question"],
                            temperature=temperature, max_new_tokens=4096)
            answer = extract_boxed_answer(output)
            correct = (answer == problem["expected_answer"])
            run_results.append(correct)
        results.append(sum(run_results) / len(run_results))
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    return {"mean": mean_acc, "std": std_acc, "runs": results}
```

### 7.3 Approach Comparison Matrix
Track for each approach:
- Accuracy on competition benchmark (mean +/- std across 3 runs)
- Training compute cost (GPU-hours, dollar cost)
- Adapter size (MB)
- Inference latency
- Regression check (golden set pass/fail)

### 7.4 Adapter Merging (Advanced — defer if time-constrained)
- Combine best adapters from different approaches
- Methods to investigate: weight averaging, DARE, TIES
- Validate merged adapter doesn't regress on golden set
- Select single best adapter for final submission

### Done When
- [ ] All approaches evaluated on all benchmarks with 3 runs each
- [ ] Comparison table complete with mean +/- std accuracy
- [ ] Golden set passes for all candidate adapters
- [ ] Best adapter identified with justification
- [ ] Results logged to WandB

---

## Phase 8: Final Submission

**Requires**: Phase 7 best adapter identified
**Hardware**: Local (any)

### 8.1 Steps
1. Export best LoRA adapter as `.safetensors`
2. Validate adapter loads correctly with base model
3. Run smoke test (Phase 1.2) to verify end-to-end
4. Run golden set to verify no regression
5. Run full evaluation on competition test set
6. Format answers per competition requirements
7. Submit to Kaggle leaderboard

### 8.2 Capstone Documentation
- `docs/approach_comparison.md`: All approaches compared with data
- `experiments/README.md`: Experiment log with WandB links
- Updated `README.md`: Project overview + findings + what we learned

### Done When
- [ ] Adapter submitted to Kaggle and accepted
- [ ] Leaderboard score recorded
- [ ] Capstone documentation complete
- [ ] All experiment results reproducible with documented seeds

---

## Failure Recovery

### Checkpoint Strategy
- **SFT (Phase 4)**: Save adapter every 500 steps. Keep last 3 checkpoints + best (lowest val loss).
- **GRPO (Phase 6)**: Save every 200 steps. Keep last 3 + best (highest reward).
- **Colab Pro**: Mount Google Drive, save checkpoints there. Resume from latest on session restart.
- **HPC**: Save to job output directory. Download best adapters after job completes.

### Early Stopping
| Condition | Action |
|-----------|--------|
| Val loss increases for 3 consecutive evals | Stop training, use best checkpoint |
| Training loss plateaus (< 0.01 change over 500 steps) | Stop, evaluate current adapter |
| Golden set accuracy drops below 90% during training | Stop immediately, investigate regression |
| GPU OOM | Reduce batch size (see ladder below) |

### OOM Recovery Ladder
If training crashes with OOM, try these in order:

| Step | Effective Batch Size | Per-GPU Batch | Grad Accum | Notes |
|------|---------------------|---------------|------------|-------|
| 1 | 256 | 8 | 32 | Default (NVIDIA recommended) |
| 2 | 128 | 4 | 32 | First reduction |
| 3 | 64 | 2 | 32 | Works on most GPUs |
| 4 | 32 | 1 | 32 | Minimum viable |
| 5 | 32 + shorter seqs | 1 | 32 | Reduce max_tokens to 4096 |
| 6 | Switch to QLoRA | 1 | 32 | 4-bit quantization as last resort |

### Adapter Backup
- Tag every saved adapter: `{approach}_{step}_{val_loss}.safetensors`
- Back up best adapters to Google Drive or cloud storage
- Never overwrite — always save alongside previous versions

---

## Scope-Cutting Priority

If time or compute runs short, drop phases in this order:

| Priority | Drop | Minimum Viable Submission |
|----------|------|--------------------------|
| 1st to drop | Phase 6 (GRPO RL) | SFT adapter only |
| 2nd to drop | Phase 5 (Synthetic Data) | Use curated data only |
| 3rd to drop | Phase 7.4 (Adapter Merging) | Single best adapter |
| 4th to drop | Phase 2 (Prompting — partial) | Keep best-of-N only |
| **Never drop** | Phase 0, 1, 3, 4, 8 | Setup + Baseline + Data + SFT + Submit |

**Minimum viable submission**: Phase 0 + 1 + 3 + 4 + 8 (SFT on curated data, no RL, no synthetic data).

---

## Phase Dependencies

```
Phase 0 (Setup)
    └── Phase 1 (Baseline) ─── required by all subsequent phases
            ├── Phase 2 (Prompting) ─── independent, can run anytime after Phase 1
            └── Phase 3 (Data Curation) ─── requires Phase 1 dataset exploration
                    ├── Phase 4 (SFT LoRA) ─── requires Phase 3 curated data
                    │       └── Phase 6 (GRPO RL) ─── requires Phase 4 trained adapter
                    └── Phase 5 (Synthetic Data) ─── requires Phase 3 pipeline
            Phase 7 (Evaluation) ─── requires at least one adapter (Phase 4+)
                    └── Phase 8 (Submission) ─── requires Phase 7 best adapter
```

---

## Execution Timeline

| Step | Phase | Duration | Hardware | Priority | Depends On |
|------|-------|----------|----------|----------|------------|
| 1 | Setup (Phase 0) | 2 hours | Local | CRITICAL | — |
| 2 | Baseline (Phase 1) | 3 hours | Colab/Local | CRITICAL | Phase 0 |
| 3 | Prompting (Phase 2) | 4 hours | Colab | HIGH | Phase 1 |
| 4 | Data Curation (Phase 3) | 6 hours | Local CPU | HIGH | Phase 1 |
| 5 | SFT LoRA (Phase 4) | 12-48 hours | HPC | CRITICAL | Phase 3 |
| 6 | Synthetic Data (Phase 5) | 8 hours | Local + API | MEDIUM | Phase 3 |
| 7 | GRPO RL (Phase 6) | 24-48 hours | HPC | MEDIUM | Phase 4 |
| 8 | Evaluation (Phase 7) | 4 hours | Colab | HIGH | Phase 4+ |
| 9 | Submission (Phase 8) | 2 hours | Local | CRITICAL | Phase 7 |

> **Note**: Phases 2 and 3 can run in parallel after Phase 1 completes. Phase 5 can run in parallel with Phase 4.

---

## Key Resources

| Resource | URL |
|----------|-----|
| Competition Page | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge |
| Submission Demo | https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo |
| Base Model (HF) | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 |
| Post-Training Dataset | https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset |
| Puzzle Dataset v2 | https://huggingface.co/datasets/nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2 |
| NVIDIA Training Blog | https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/ |
| NeMo Reasoning Tutorials | https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/reasoning |
| NeMo Gym | https://github.com/NVIDIA-NeMo/Gym |
| NeMo RL + Nemotron Guide | https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/nemotron-3-nano.md |
| Unsloth Nemotron Guide | https://unsloth.ai/docs/models/nemotron-3 |
| TRL GRPO Docs | https://huggingface.co/docs/trl/main/en/grpo_trainer |
| SYNTHETIC-1 Dataset | https://www.primeintellect.ai/blog/synthetic-1 |
| DeepSeek-R1 Paper | https://arxiv.org/abs/2501.12948 |
| PEFT LoRA Docs | https://huggingface.co/docs/peft/en/package_reference/lora |

---

## Verification Summary

| Phase | Verification | Gate Type |
|-------|-------------|-----------|
| 0 | All imports succeed + WandB logs + storage verified | Checklist |
| 1 | Baseline on 3+ benchmarks + smoke test passes + dummy submission | Checklist |
| 2 | Measured deltas vs baseline with 3 runs per strategy | Quantitative |
| 3 | Validation set reserved + curated data stats + quality metrics | Checklist |
| 4 | Loss converges + golden set passes + adapter loads + benchmark comparison | Quantitative + Regression |
| 5 | Synthetic accuracy > 90% + costs within cap | Quantitative + Budget |
| 6 | Reward increases + beats SFT baseline + golden set passes | Quantitative + Regression |
| 7 | All approaches compared with mean +/- std + best identified | Quantitative |
| 8 | Kaggle submission accepted + leaderboard score recorded | End-to-end |

---

## Changes from v0.1

| Change | Addresses |
|--------|-----------|
| Added Compute & Constraints section with hardware mapping | CR-2 (blocking questions) |
| Added "Done When" gates to every phase | CR-1 (phase transitions) |
| Added Failure Recovery section (checkpoints, early stopping, OOM ladder) | CR-3 (no rollback) |
| Added Smoke Test in Phase 1.2 | CR-4 (no concrete test) |
| Added Validation Set Reservation in Phase 3.1 | CR-5 (no validation strategy) |
| Fixed boxed regex to handle nested braces | MJ-7 (regex bug) |
| Relabeled "Expected Gain" to "Literature Estimate" | MJ-6 (ungrounded estimates) |
| Added Phase Dependencies diagram | MJ-1 (no dependency mapping) |
| Added Scope-Cutting Priority | MJ-2 (explore ALL vs constraints) |
| Added API cost caps in Phase 5 | MJ-4 (cost risk) |
| Added Storage Requirements table | MJ-5 (disk space) |
| Added statistical significance notes to Phase 7 | MJ-9 (significance testing) |
| Added regression testing with golden set | MJ-8 (no regression testing) |
| Added reproducibility setup (seeds, eval protocol) | MN-1, MN-2, MN-5 |
| Added WandB configuration template | MN-3 |
| Added adapter backup strategy | MN-4 |
| Added vLLM Mamba-2 compatibility note | MJ-7 (untested assumption) |
