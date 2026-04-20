# NVIDIA Nemotron Model Reasoning Challenge - Research & MVP Plan v0.1

**Version**: 0.1 (Draft for Review)
**Date**: 2026-04-19
**Status**: Pending Agent Review

---

## Context

This is a **CS4650 Capstone project** to compete in the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle. The competition requires improving the reasoning accuracy of **Nemotron-3-Nano-4B** (a 3.97B parameter hybrid Mamba-2 + Transformer model) on a novel benchmark. Participants submit a **LoRA adapter** that gets merged with the base model for evaluation on G4 VMs (NVIDIA RTX PRO 6000 Blackwell GPUs).

The goal is to explore **ALL available approaches** (prompting, data curation, synthetic data, RL, fine-tuning) for maximum learning and competitive performance.

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
│   ├── lora_baseline.yaml        # LoRA r=64, alpha=128
│   ├── lora_qlora.yaml           # QLoRA 4-bit for consumer GPUs
│   ├── grpo_config.yaml          # GRPO RL training
│   └── data_curation.yaml        # NeMo Curator config
│
├── data/                         # Data directory (gitignored)
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Curated/filtered data
│   └── synthetic/                # Generated synthetic data
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
    ├── plan_v0.1.md              # This file
    ├── research_notes.md
    └── approach_comparison.md
```

### 0.2 Core Dependencies
```
torch>=2.2.0
transformers>=4.48.3
datasets>=2.20.0
peft>=0.14.0
trl>=0.16.0
accelerate>=1.0.0
bitsandbytes>=0.45.0        # QLoRA quantization
unsloth                      # Efficient fine-tuning (Nemotron support)
vllm>=0.15.1                 # Fast inference
wandb                        # Experiment tracking
jupyter
pandas numpy matplotlib
kaggle                       # Kaggle CLI
```

### 0.3 Git & Environment Setup
- `.gitignore`: Python/ML standard (exclude `data/`, `adapters/*.safetensors`, `.env`)
- `.env.example`: `HF_TOKEN`, `WANDB_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
- Download competition data: `kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge`

---

## Phase 1: Baseline & Exploration (MVP)

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

### 1.2 Dataset Exploration
- Explore competition-provided data
- Analyze `nvidia/Llama-Nemotron-Post-Training-Dataset` (32M+ samples, 130GB)
  - Subsets: math (22M), code (10M), science (708K), chat (39K), safety (31K)
- Analyze `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` (851K samples, 2.74GB)
  - Pre-filtered, English-only, reasoning=off, 95/5 train/val split
- Identify most relevant subsets for reasoning improvement

### 1.3 Submission Pipeline
- Study [submission demo notebook](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
- Build end-to-end: load base model -> apply LoRA adapter -> generate answers -> format submission
- Submit "dummy" adapter to verify pipeline works

---

## Phase 2: Prompting Strategies (Zero Compute Cost)

### 2.1 System Prompt Engineering
- `enable_thinking=True` with `temperature=1.0, top_p=0.95` (NVIDIA recommended)
- Custom system prompts for math domains
- Structured output forcing `\boxed{}` format

### 2.2 Chain-of-Thought Prompting
| Strategy | Description | Expected Gain |
|----------|-------------|---------------|
| Zero-shot CoT | "Let's think step by step" | +5-10% |
| Few-shot CoT | 3-5 solved examples as context | +10-15% |
| Program of Thoughts | Generate code to solve problems | +20% on arithmetic |
| Self-consistency | Sample N=64, majority vote | +15-20% |

### 2.3 Inference-Time Compute Scaling
- Best-of-N sampling (N=8, 16, 32, 64)
- Majority voting: `pass@1` vs `maj@64`
- Temperature sweep: [0.6, 0.8, 1.0, 1.2]
- Top-p sweep: [0.9, 0.95, 0.99]

---

## Phase 3: Data Curation & Filtering

### 3.1 NeMo Curator Pipeline
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

### 3.2 Quality Filtering
- Remove incorrect/inconsistent solutions
- Verify math answers programmatically
- Filter non-empty, coherent reasoning traces
- Deduplicate similar problems

### 3.3 Curriculum Design
- Progressive difficulty: GSM8K-level -> MATH-level -> Olympiad-level
- Domain mixing: 60% math, 20% code, 15% science, 5% chat
- **Critical ratio**: 75% reasoning-on / 25% reasoning-off data

---

## Phase 4: Supervised Fine-Tuning (SFT) with LoRA

### 4.1 LoRA Configuration (NVIDIA Recommended)
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=64,                          # Rank (sweet spot per NVIDIA)
    lora_alpha=128,                # Alpha = 2x rank
    target_modules=[
        "q_proj", "v_proj",        # Transformer attention layers
        "x_proj", "in_proj", "out_proj"  # Mamba-2 layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

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

### 4.3 Framework Comparison
| Framework | VRAM (4B) | Speed | Nemotron Support | Best For |
|-----------|-----------|-------|------------------|----------|
| **Unsloth** | 8-16GB (QLoRA) | Fastest | Official | Rapid prototyping on Colab |
| **TRL SFTTrainer** | 16-60GB | Good | Yes | HF ecosystem integration |
| **NeMo Framework** | 60-80GB | Good | Official recipes | Production H100 runs |
| **HF PEFT** | 16-60GB | Good | Manual config | Custom setups |

### 4.4 QLoRA for Consumer GPUs
- 4-bit quantization via `bitsandbytes`
- **Free Colab T4 (16GB)**: Feasible with Unsloth + QLoRA, batch_size=32
- **RTX 4090 (24GB)**: Standard LoRA with batch_size=64-128
- **RunPod H100 (80GB)**: Full training, ~$2/hr, 48 hours = ~$96

### 4.5 Mamba-2 Specific Target Modules
For the hybrid architecture, target both layer types:
- **Mamba-2 layers**: `x_proj`, `embeddings`, `in_proj`, `out_proj`
- **Transformer layers**: `q_proj`, `v_proj`

---

## Phase 5: Synthetic Data Generation

### 5.1 Distillation from Stronger Models
Generate reasoning traces using:
| Model | Use Case | Access |
|-------|----------|--------|
| DeepSeek-R1 / R1-0528 | Math CoT generation | HF / API |
| Qwen3-235B | Diverse reasoning styles | HF / API |
| Claude / GPT-4 | High-quality explanations | API |

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

---

## Phase 6: Reinforcement Learning (GRPO)

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

def accuracy_reward(completions, ground_truth):
    """Binary reward: 1 if boxed answer matches, 0 otherwise."""
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        match = re.search(r'\\boxed\{(.+?)\}', completion)
        answer = match.group(1) if match else ""
        rewards.append(1.0 if answer.strip() == truth.strip() else 0.0)
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

---

## Phase 7: Evaluation & Comparison

### 7.1 Benchmarks
| Benchmark | Baseline | Target | Purpose |
|-----------|----------|--------|---------|
| MATH500 | 95.4% | 96%+ | General math |
| AIME25 | 78.5% | 82%+ | Competition math |
| GPQA | 53.2% | 58%+ | Science reasoning |
| GSM8K | ~85% | 90%+ | Grade school math |
| Competition benchmark | TBD | Top 50% | Actual leaderboard |

### 7.2 Approach Comparison Matrix
Track for each approach:
- Accuracy on competition benchmark
- Training compute cost (GPU-hours, dollar cost)
- Adapter size (MB)
- Inference latency
- Reproducibility

### 7.3 Adapter Merging (Advanced)
- Combine best adapters from different approaches
- LoRA adapter weight averaging
- Select single best adapter for final submission

---

## Phase 8: Final Submission

### 8.1 Steps
1. Export best LoRA adapter as `.safetensors`
2. Validate adapter loads correctly with base model
3. Run full evaluation on competition test set
4. Format answers per competition requirements
5. Submit to Kaggle leaderboard

### 8.2 Capstone Documentation
- `docs/approach_comparison.md`: All approaches compared
- `experiments/README.md`: Experiment log with results
- Updated `README.md`: Project overview + findings

---

## Execution Timeline

| Step | Phase | Duration | Compute | Priority |
|------|-------|----------|---------|----------|
| 1 | Setup (Phase 0) | 2 hours | Local | CRITICAL |
| 2 | Baseline (Phase 1) | 3 hours | Colab/Local | CRITICAL |
| 3 | Prompting (Phase 2) | 4 hours | Colab | HIGH |
| 4 | Data Curation (Phase 3) | 6 hours | Local/Colab | HIGH |
| 5 | SFT LoRA (Phase 4) | 12-48 hours | Colab/H100 | CRITICAL |
| 6 | Synthetic Data (Phase 5) | 8 hours | API calls | MEDIUM |
| 7 | GRPO RL (Phase 6) | 24-48 hours | H100 | MEDIUM |
| 8 | Evaluation (Phase 7) | 4 hours | Colab | HIGH |
| 9 | Submission (Phase 8) | 2 hours | Local | CRITICAL |

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

## Verification Plan

| Phase | Verification |
|-------|-------------|
| 0 | `python -c "import torch, transformers, peft, trl; print('OK')"` |
| 1 | Baseline accuracy recorded; dummy submission accepted on Kaggle |
| 2 | Prompting accuracy comparison table with deltas vs baseline |
| 3 | Curated dataset stats: size, distribution, quality metrics |
| 4 | Training loss converges; adapter loads and generates correct format |
| 5 | Synthetic data passes quality checks; verified answer accuracy > 90% |
| 6 | RL reward curve increases; adapter accuracy > SFT baseline |
| 7 | Comparison table with all approaches ranked |
| 8 | Successful Kaggle submission with leaderboard score |

---

## Open Questions for Review

1. **Competition dataset specifics**: Need to download and inspect the actual competition data to understand exact evaluation format
2. **Mamba-2 LoRA targets**: Need to verify exact target module names for the hybrid architecture via model inspection
3. **Compute budget**: What GPU resources are available? Free Colab vs paid cloud?
4. **Timeline**: When is the competition deadline? This affects which phases we can complete
5. **Team size**: Is this solo or team capstone? Affects parallelization of approaches
