# Issue 25: HPC Queue / Runbook Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SLURM-ready runbook that can launch SFT on HPC, optionally launch RL after SFT is frozen, and recover safely through checkpoints, early stopping, and hard regression gates.

**Architecture:** Split the workflow into short preflight jobs and long training jobs. Preflight validates the frozen contracts, tokenization, storage, and environment; training jobs only consume frozen configs and write checkpoints to durable storage; eval and packaging are separate jobs so no checkpoint promotion happens without passing gates.

**Tech Stack:** SLURM, Bash, Python, `transformers`, `peft`, `trl`, optional `accelerate`, optional `wandb`, JSONL/Parquet artifacts, shared scratch storage, repo-local docs.

---

## Non-Negotiable Freeze Gate

`#14` is now verified (snapshot 2026-04-29 in `docs/architecture/COMPETITION.md` "Verified" section). The runbook must consume these frozen inputs verbatim:

- Base model: KaggleHub `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
- Load recipe: `trust_remote_code=True`, `torch.bfloat16`, `device_map="auto"`
- LoRA constraints: `r <= 32` (evaluator enforces `max_lora_rank=32`); demo target modules `in_proj|out_proj|up_proj|down_proj`
- Answer / normalization contract: `\\boxed{}` extraction, exact match or `1e-3` numeric tolerance, reasoning allowed
- Submission artifact layout: `adapter_config.json` + `adapter_model.safetensors` at zip root
- Evaluator decode params: `max_tokens=7680`, `temperature=0.0`, `top_p=1.0`, `max_model_len=8192`

Preflight must reject any run whose `BASE_MODEL_ID`, `LORA_RANK`, `LORA_TARGET_MODULES`, or `NORMALIZER_ID` deviates from the verified values above.

## Scope

This plan covers the operational runbook for:

- SFT queue submission
- optional RL queue submission after SFT
- checkpointing and resume rules
- early stopping and rollback rules
- regression gates for eval promotion
- artifact storage conventions for HPC outputs

This plan does **not** define notebook cells or notebook execution flows.

## Job Types

| Job type | Purpose | Typical resources | Inputs | Outputs | Stop / retry policy |
|---|---|---:|---|---|---|
| `prep` | Validate env, freeze config, create run directories, snapshot git SHA, confirm storage quotas | 1 CPU node, 2-4 cores, 8-16 GB RAM, 5-15 min | frozen base model config, frozen LoRA config, `requirements.txt`, env vars | run manifest, directory tree, config snapshot, preflight log | retry only after config mismatch or missing env vars is fixed |
| `tokenize` | Build tokenized shards or cached dataset fingerprints for SFT / RL | 1 CPU node, 4-16 cores, 16-64 GB RAM, 15-90 min | raw / processed datasets, tokenizer revision | tokenized cache, fingerprint manifest, dataset stats | retry on cache corruption or schema mismatch |
| `train` (SFT) | Run supervised fine-tuning with checkpointing, eval hooks, and early stopping | 1 GPU node, 1-4 GPUs, 64-128 GB RAM, 8-48 GPU-hours | frozen base model, frozen LoRA config, tokenized train split, validation slice | checkpoints, best adapter, training metrics, eval artifacts | stop on gate failure, NaN, OOM, or stalled validation |
| `train` (RL, optional) | Run RL only after SFT passes and the reward / eval contract is frozen | 1-4 GPU nodes depending cluster, 64-256 GB RAM, 24-96 GPU-hours | best SFT adapter, reward config, frozen eval contract | RL checkpoints, reward traces, best adapter candidate | skip entirely unless SFT gate is green |
| `eval` | Score checkpoint candidates against golden and validation slices | 1 GPU node or cached CPU eval if supported, 16-64 GB RAM, 30 min-4 h | candidate checkpoint, golden set, validation slice, normalizer | `EvalRecord`-style outputs, summary metrics, pass/fail status | no promotion if any regression gate fails |
| `package` | Copy the selected adapter into a Kaggle-safe bundle and write provenance | 1 CPU node, 2-4 cores, 4-8 GB RAM, 5-20 min | best adapter, manifest inputs, frozen submission layout | `submission.zip`, manifest, checksum file, handoff log | fail fast on size/layout mismatch |

## Resource Estimates

| Stage | Range | Assumptions |
|---|---:|---|
| `prep` | 5-15 minutes | config validation only; no model loads |
| `tokenize` | 15-90 minutes | one pass over the selected train/val data, cached tokenizer, CPU parallelism available |
| SFT smoke run | 0.5-2 GPU-hours | 100-500 steps, small subset, checkpointing enabled, no sweep fan-out |
| SFT production run | 8-48 GPU-hours | `r <= 32` (evaluator enforces `max_lora_rank=32`), frozen target modules `in_proj|out_proj|up_proj|down_proj`, gradient checkpointing on, validation every few hundred steps |
| Optional RL run | 24-96 GPU-hours | only after SFT gate passes, reward contract frozen, smaller sweep count than SFT |
| `eval` | 30 minutes-4 hours | golden set plus validation slice, plus per-checkpoint comparisons |
| `package` | 5-20 minutes | adapter copy, hash, zip, manifest generation |

## Required Env Vars

| Category | Variables | Purpose |
|---|---|---|
| Authentication | `HF_TOKEN`, `WANDB_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY` | model access, experiment tracking, submission tooling |
| SLURM routing | `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS`, `SLURM_TIME_LIMIT`, `SLURM_MEM_GB`, `SLURM_GPUS`, `SLURM_CPUS_PER_TASK` | queue placement and resource allocation |
| Frozen model contract | `BASE_MODEL_ID`, `BASE_MODEL_REVISION`, `TOKENIZER_ID`, `ENABLE_THINKING` | make the base checkpoint explicit and reproducible |
| LoRA / training contract | `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES`, `MAX_SEQ_LEN`, `MAX_NEW_TOKENS`, `SEED` | lock the adapter and decoding config |
| Data / cache roots | `DATA_ROOT`, `RUN_ROOT`, `CHECKPOINT_ROOT`, `ARTIFACT_ROOT`, `LOG_ROOT`, `HF_HOME`, `TRANSFORMERS_CACHE`, `WANDB_DIR`, `TMPDIR` | keep all heavy outputs off git and on durable scratch |
| Evaluation contract | `NORMALIZER_ID`, `GOLDEN_SET_PATH`, `VALIDATION_SET_PATH`, `RUN_TAG` | bind a run to the exact gate inputs |

## Storage Conventions

| Path | Contents | Retention |
|---|---|---|
| `${DATA_ROOT}/raw/` | raw competition and mirror datasets | read-only after ingest |
| `${DATA_ROOT}/processed/` | filtered / tokenized / cached datasets | keep by dataset fingerprint |
| `${RUN_ROOT}/${RUN_TAG}/logs/` | `slurm-*.out`, stderr, env snapshots | keep until final handoff |
| `${RUN_ROOT}/${RUN_TAG}/checkpoints/` | `step-XXXXX/`, `last/`, `best/` | keep last 3 + best |
| `${RUN_ROOT}/${RUN_TAG}/eval/` | JSONL / Parquet eval records, summaries | keep all runs that passed gates |
| `${RUN_ROOT}/${RUN_TAG}/package/` | final zip, manifest, hashes | keep all promoted artifacts |
| `adapters/issue-25/${RUN_TAG}/` | selected final adapter mirror | copy only after eval gate passes |
| `experiments/issue-25/${RUN_TAG}/` | run manifest, metrics, comparison notes | keep for review and reproducibility |

## Checkpointing and Recovery Policy

- Save SFT checkpoints every 500 steps or every 30 minutes, whichever comes first.
- Save RL checkpoints every 200 steps or every 20 minutes, whichever comes first.
- Keep the last 3 checkpoints plus one `best/` checkpoint chosen by validation score.
- Write checkpoints atomically: stage into a temporary directory, then rename into the final checkpoint path.
- Always write these sidecar files next to each checkpoint:
  - `trainer_state.json`
  - `run_config.json`
  - `metrics.jsonl`
  - `git_sha.txt`
  - `dataset_fingerprint.txt`
- Resume only from the latest checkpoint that passed the most recent regression gate.

## Early Stopping Rules

- Stop SFT if validation loss fails to improve for 3 consecutive evals.
- Stop SFT if the golden set drops below the frozen baseline.
- Stop SFT immediately on NaN loss, repeated OOM, corrupt checkpoint, or broken tokenizer load.
- Stop optional RL immediately if:
  - the SFT baseline adapter is not the current `best/`
  - reward metrics move in the wrong direction for 3 evals
  - golden set regression appears at any checkpoint
- If early stopping triggers, the runbook must keep the best checkpoint and archive the failing state for inspection.

## Regression Gates

| Gate | Required result | Action on failure |
|---|---|---|
| Preflight gate | frozen config matches `#14`, paths exist, env vars are present | do not queue training |
| Smoke gate | tiny SFT smoke run completes and adapter reloads | stop and debug before production |
| Golden gate | golden set accuracy does not regress from the frozen baseline | reject checkpoint promotion |
| Validation gate | validation slice remains within the frozen tolerance band | keep prior best checkpoint |
| Packaging gate | final adapter fits the Kaggle bundle and matches the required zip root | fail packaging and do not submit |

## Planned Scripts and Commands

The runbook expects these scripts to be created later:

- `scripts/hpc/preflight.sh`
- `scripts/hpc/tokenize_dataset.py`
- `scripts/hpc/submit_sft.sbatch`
- `scripts/hpc/submit_rl.sbatch`
- `scripts/hpc/checkpoint_policy.py`
- `scripts/hpc/regression_gate.py`
- `scripts/hpc/package_adapter.py`
- `scripts/hpc/resume_from_latest.py`

Planned command shapes:

```bash
sbatch --job-name=nemotron-sft \
  --export=ALL,RUN_TAG=${RUN_TAG},BASE_MODEL_ID=${BASE_MODEL_ID},LORA_RANK=${LORA_RANK} \
  scripts/hpc/submit_sft.sbatch
```

```bash
sbatch --job-name=nemotron-rl \
  --dependency=afterok:${SFT_JOBID} \
  --export=ALL,RUN_TAG=${RUN_TAG},ALLOW_RL=1 \
  scripts/hpc/submit_rl.sbatch
```

```bash
python scripts/hpc/regression_gate.py \
  --checkpoint "${CHECKPOINT_ROOT}/${RUN_TAG}/best" \
  --golden "${GOLDEN_SET_PATH}" \
  --validation "${VALIDATION_SET_PATH}" \
  --normalizer "${NORMALIZER_ID}"
```

```bash
python scripts/hpc/package_adapter.py \
  --checkpoint "${CHECKPOINT_ROOT}/${RUN_TAG}/best" \
  --output "${RUN_ROOT}/${RUN_TAG}/package"
```

## Task Plan

### Task 1: Freeze queue inputs

**Files:**
- Create: `scripts/hpc/preflight.sh`
- Modify: `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` gate notes only if the blocker remains open in future work

- [ ] **Step 1: Validate frozen inputs**

Run: `bash scripts/hpc/preflight.sh --show`
Expected: prints `BASE_MODEL_ID`, `BASE_MODEL_REVISION`, `LORA_RANK`, `LORA_TARGET_MODULES`, and refuses to continue if any are unset.

- [ ] **Step 2: Lock the run config**

Run: `bash scripts/hpc/preflight.sh --write-config`
Expected: writes `run_config.json` into `${RUN_ROOT}/${RUN_TAG}/`.

### Task 2: Add SLURM submission wrappers

**Files:**
- Create: `scripts/hpc/submit_sft.sbatch`
- Create: `scripts/hpc/submit_rl.sbatch`

- [ ] **Step 1: Encode resource requests**

Run: `sbatch scripts/hpc/submit_sft.sbatch --help`
Expected: wrapper documents partition, GPU count, memory, and time limit requirements.

- [ ] **Step 2: Wire the dependency chain**

Run: `sbatch --dependency=afterok:${SFT_JOBID} scripts/hpc/submit_rl.sbatch`
Expected: RL job stays blocked until SFT exits cleanly.

### Task 3: Add checkpoint and resume policy

**Files:**
- Create: `scripts/hpc/checkpoint_policy.py`
- Create: `scripts/hpc/resume_from_latest.py`

- [ ] **Step 1: Write checkpoint rotation logic**

Run: `python scripts/hpc/checkpoint_policy.py --dry-run`
Expected: prints the keep-last-3-plus-best policy and the atomic checkpoint path layout.

- [ ] **Step 2: Verify resume behavior**

Run: `python scripts/hpc/resume_from_latest.py --run-tag ${RUN_TAG}`
Expected: selects the latest checkpoint that passed the most recent regression gate.

### Task 4: Add regression gates

**Files:**
- Create: `scripts/hpc/regression_gate.py`

- [ ] **Step 1: Score smoke / golden / validation slices**

Run: `python scripts/hpc/regression_gate.py --dry-run`
Expected: emits pass/fail thresholds for smoke, golden, and validation checks.

- [ ] **Step 2: Block bad checkpoints**

Run: `python scripts/hpc/regression_gate.py --checkpoint ...`
Expected: exits non-zero if the golden gate or validation gate fails.

### Task 5: Add packaging and artifact promotion

**Files:**
- Create: `scripts/hpc/package_adapter.py`

- [ ] **Step 1: Package only the promoted adapter**

Run: `python scripts/hpc/package_adapter.py --checkpoint ... --output ...`
Expected: writes a Kaggle-safe bundle and a provenance manifest outside the zip root.

- [ ] **Step 2: Mirror final artifacts**

Run: `rsync -a "${RUN_ROOT}/${RUN_TAG}/package/" "adapters/issue-25/${RUN_TAG}/"`
Expected: final adapter mirror exists for review without putting large files in git.

## Verification Checklist

- Preflight refuses to run if `BASE_MODEL_ID`, `LORA_RANK`, `LORA_TARGET_MODULES`, or `NORMALIZER_ID` deviates from the verified `#14` freeze (snapshot 2026-04-29 in `docs/architecture/COMPETITION.md`).
- SFT can resume from the latest checkpoint after interruption.
- Golden and validation gates prevent promotion of a degraded checkpoint.
- Optional RL stays disabled until the SFT gate is green.
- Package output keeps provenance outside the Kaggle submission zip and matches the verified zip layout (`adapter_config.json` + `adapter_model.safetensors` at root).

## Handoff

When implemented, this plan should leave the repo with a repeatable HPC queue path for SFT, a gated optional RL path, and a stable artifact convention that supports later submission packaging.
