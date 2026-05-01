# External Baseline Review

**Date:** 2026-04-29  
**Scope:** TODO.md `#2` / `#16` bounded review of Tong Hui Kang's public
`tonghuikang/nemotron` repository and konbu17's public Kaggle notebook.  
**Official contract source:** `#14`, captured in
`docs/architecture/COMPETITION.md`. External baselines may inform implementation
choices but do not override the official Kaggle/NVIDIA contract.

## Source Identity

| Source | Identity | Review status |
|---|---|---|
| Tong Hui Kang (`tonghuikang/nemotron`) | Public Progress Prize reference repository, not an organizer source | Reviewed from public GitHub at `82bd1880aa8a8986ad572ccd17ae35b2b5c7da85` |
| konbu17 Kaggle notebook | Public competitor/reference notebook, not an organizer source | Reviewed from repo-local notebook evidence and public notebook URL; live Kaggle cells were not extractable in this environment |
| Kaggle/NVIDIA `#14` source | Official competition/demo contract source | Frozen defaults already captured in `docs/architecture/COMPETITION.md` |

## Official Defaults From `#14`

Use these as defaults whenever external baselines differ:

- Base model path: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`.
- Load recipe: `trust_remote_code=True`, `torch.bfloat16`,
  `device_map="auto"`.
- Scoring: final answer inside `\boxed{}`; exact string match or relative
  numeric tolerance `1e-3`.
- LoRA rank: `r <= 32`.
- Demo target modules: `in_proj`, `out_proj`, `up_proj`, `down_proj`.
- Submission zip root: `adapter_config.json` and
  `adapter_model.safetensors`.

## Baseline Facts

| Dimension | Tong Hui Kang public repo | konbu17 notebook evidence |
|---|---|---|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` in `train_sft.py` | KaggleHub `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` in notebook evidence |
| Load recipe | Tokenizer uses `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`; training client receives `base_model=cfg.model_name` | `trust_remote_code=True`, bf16, `attn_implementation="eager"`, pad-token fallback in local notebook evidence |
| LoRA rank/config | `lora_rank=32`; Tinker client flags `train_mlp=True`, `train_attn=True`, `train_unembed=True`; no explicit PEFT target-module list found | `r=32`, `lora_alpha=32`, `lora_dropout=0.0`, `bias="none"`, `task_type=CAUSAL_LM` |
| Target modules | Coarse Tinker flags only; no explicit module list found | 9-module set in prior local notebook evidence: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `out_proj`, `up_proj`, `down_proj`, `lm_head` |
| Masking | Explicit token mask: prompt/user tokens masked, completion/reasoning tokens unmasked for loss | Implicit TRL formatting path in notebook evidence; no explicit completion-only collator or response template captured |
| Dataset/corpus | `train.csv`, `problems.jsonl`, deterministic `reasoning/`, synthetic `augmentations/`, and `corpus.jsonl` segment files | Kaggle notebook evidence focuses on Tong-style CoT SFT formatting and Kaggle-hosted inputs |
| Eval/metrics | Per-epoch logprobs and token-weighted metrics on unmasked tokens; deterministic solver script reports per-category accuracy/runtime | Local notebook evidence captures training recipe and packaging checks, not a reusable repo-local eval normalizer |
| Packaging | Modal script downloads Tinker checkpoint, finds `adapter_config.json` and `adapter_model.safetensors`, then uploads a Kaggle model version | Minimal `submission.zip` with `adapter_config.json` and `adapter_model.safetensors` at root |
| Augmentation/weighting | Generators for spelling, concatenation, splitting, matching, and lstrip; supports CE, weighted CE, importance sampling, PPO, CISPO, and DRO-style losses | Priority/upweighting ideas are present in notebook evidence but tied to Kaggle input paths |
| Solver/teacher pattern | Category reasoners generate deterministic natural-language CoT traces and keep traces only when extracted answer matches the stored answer | Tong-style CoT formatting is the reusable pattern; source/commit/license capture still required before porting |

## Adopt / Reject / Gate

| Decision | Item | Rationale | Downstream |
|---|---|---|---|
| Adopt | `#14` official facts as defaults | Official Kaggle/NVIDIA facts override public baselines. | All downstream issues |
| Adopt | konbu17 minimal zip layout | Matches the frozen root-only adapter package contract. Keep provenance outside the zip. | `#20` |
| Adopt | `r=32` as SFT-safe LoRA rank | Matches konbu17 and the official `r <= 32` cap. Removes stale `r=64` plan assumptions. | `#25` |
| Reject | Implicit/default masking | TRL defaults can drift by version. Require explicit completion-only masking in our SFT implementation. | `#25` |
| Gate | konbu17 9-module target set | Useful experiment candidate, but official demo references only 4 modules. Requires PM signoff before expanding. | `#25` |
| Gate | Tong augmentation and weighting ideas | Promising but must capture exact source files, commit SHA, and license/provenance before implementation. | `#24`, `#25` |
| Gate | Tong solver/teacher patterns | Solver-first deterministic CoT is relevant, but should be translated into project-owned interfaces and tests. | `#23`, `#24`, `#25` |

## Evidence Notes

- Tong repo commit used for review:
  `82bd1880aa8a8986ad572ccd17ae35b2b5c7da85`.
- Public Tong README identifies the repository as a Progress Prize submission and
  lists the training sequence:
  <https://github.com/tonghuikang/nemotron>.
- Key Tong files reviewed:
  `train_sft.py`, `train_common.py`, `corpus.py`, `reasoning.py`,
  `augmentation.py`, `loss_config.py`, `upload_adapter.py`.
- konbu17 public URL:
  <https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2>.
  The live Kaggle page did not expose notebook cell text in this environment,
  so konbu17 details above come from existing repo-local notebook evidence.
- Official Kaggle/NVIDIA discussion/demo URL:
  <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687547>.
  Use `docs/architecture/COMPETITION.md` as the repo's frozen `#14` artifact.
