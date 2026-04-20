# Notebook Registry

This repo will have two kinds of notebooks:

1. **Local notebooks** under `notebooks/` (our experiments)
2. **External reference notebooks** (Kaggle/community) we reproduce and document

If a notebook materially changes model behavior, update this file with:

- dataset version (Kaggle / mirror)
- base model id
- eval scores (overall + per-category)
- artifact pointers (adapter name/hash)

## Planned Local Notebooks (Skeleton)

- `notebooks/01_data_eda.ipynb`
  - Purpose: inspect dataset schema, categories, prompt patterns, answer normalization.
- `notebooks/02_baseline_inference.ipynb`
  - Purpose: baseline accuracy with the base model; sweep decoding params.
- `notebooks/03_prompting_sweeps.ipynb`
  - Purpose: prompt-template variants and reasoning budget sensitivity.
- `notebooks/04_teacher_solver_dev.ipynb`
  - Purpose: prototype per-category teacher/solver logic + failure analysis.
- `notebooks/05_synthetic_cot_generation.ipynb`
  - Purpose: generate synthetic “reasoning style” traces for SFT.
- `notebooks/06_sft_lora_training.ipynb`
  - Purpose: launch/monitor LoRA training runs (ideally calling scripts).
- `notebooks/07_eval_ablation.ipynb`
  - Purpose: ablations (masking, curricula, data mixes) + regression gates.
- `notebooks/08_submission.ipynb`
  - Purpose: package adapter and validate submission format.

## External Reference Work (Track + Reproduce)

- Tong Hui Kang (Progress Prize) public repo:
  - https://github.com/tonghuikang/nemotron
- Konbu17 Kaggle notebook (requested review target):
  - https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2
- Kaggle dataset with baseline trajectories (useful for error analysis patterns):
  - https://www.kaggle.com/datasets/kishanvavdara/nemotron-reasoning-traj
- Public write-up on distillation-style approach (config ideas, pitfalls):
  - https://aitherium.com/blog/nemotron-reasoning-challenge-mirothinker-distillation/
