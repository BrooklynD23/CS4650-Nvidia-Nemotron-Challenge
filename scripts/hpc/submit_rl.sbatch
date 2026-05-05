#!/usr/bin/env bash
# SLURM batch script for the optional Nemotron RL phase.
#
# This job must only run after a green SFT gate. Typically submitted with:
#
#   sbatch --dependency=afterok:${SFT_JOBID} \
#     --export=ALL,RUN_TAG=${RUN_TAG},ALLOW_RL=1 \
#     scripts/hpc/submit_rl.sbatch
#
# If ALLOW_RL is not set to "1" the job aborts immediately so it cannot
# be triggered accidentally without an explicit opt-in.
#
# Required env vars:
#   RUN_TAG, BASE_MODEL_ID, LORA_RANK, LORA_TARGET_MODULES, NORMALIZER_ID,
#   DATA_ROOT, RUN_ROOT, ALLOW_RL=1
#
# Optional env vars:
#   VENV_PATH, CHECKPOINT_ROOT, LOG_ROOT, HF_HOME, RL_CONFIG_PATH

#SBATCH --job-name=nemotron-rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment summary
# ---------------------------------------------------------------------------
echo "=== Nemotron RL Job ==="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-<not in SLURM>}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-<not in SLURM>}"
echo "RUN_TAG        = ${RUN_TAG:-<unset>}"
echo "BASE_MODEL_ID  = ${BASE_MODEL_ID:-<unset>}"
echo "LORA_RANK      = ${LORA_RANK:-<unset>}"
echo "ALLOW_RL       = ${ALLOW_RL:-<unset>}"
echo "Started        = $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# RL opt-in gate
# ---------------------------------------------------------------------------
if [[ "${ALLOW_RL:-0}" != "1" ]]; then
  echo "ERROR: ALLOW_RL is not set to '1'. RL phase is disabled."
  echo "       Set ALLOW_RL=1 only after confirming the SFT golden gate passed."
  exit 1
fi

: "${RUN_TAG:?RUN_TAG must be set}"
: "${BASE_MODEL_ID:?BASE_MODEL_ID must be set}"
: "${LORA_RANK:?LORA_RANK must be set}"
: "${LORA_TARGET_MODULES:?LORA_TARGET_MODULES must be set}"
: "${NORMALIZER_ID:?NORMALIZER_ID must be set}"
: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${RUN_ROOT:?RUN_ROOT must be set}"

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
VENV_PATH="${VENV_PATH:-${RUN_ROOT}/.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "INFO: Activated venv: ${VENV_PATH}"
else
  echo "WARNING: venv not found at ${VENV_PATH}; using system Python"
fi

export HF_HOME="${HF_HOME:-${RUN_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${HF_HOME}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 1: Preflight gate
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Preflight ==="
bash scripts/hpc/preflight.sh || {
  echo "ERROR: Preflight failed — aborting RL job"
  exit 1
}

# ---------------------------------------------------------------------------
# Step 2: Verify SFT best checkpoint exists
# ---------------------------------------------------------------------------
SFT_CHECKPOINT_DIR="${CHECKPOINT_ROOT:-${RUN_ROOT}}/${RUN_TAG}/checkpoints"
SFT_BEST_DIR="${SFT_CHECKPOINT_DIR}/best"

echo ""
echo "=== Step 2: Verify SFT best checkpoint ==="
if [[ ! -d "${SFT_BEST_DIR}" ]]; then
  echo "ERROR: SFT best checkpoint not found: ${SFT_BEST_DIR}"
  echo "       Ensure SFT completed and the golden gate passed."
  exit 1
fi
echo "INFO: SFT best checkpoint: ${SFT_BEST_DIR}"

# ---------------------------------------------------------------------------
# RL run directories
# ---------------------------------------------------------------------------
RL_CHECKPOINT_DIR="${CHECKPOINT_ROOT:-${RUN_ROOT}}/${RUN_TAG}/rl_checkpoints"
LOG_DIR="${LOG_ROOT:-${RUN_ROOT}/${RUN_TAG}/logs}"

mkdir -p "${RL_CHECKPOINT_DIR}" "${LOG_DIR}"

git rev-parse HEAD > "${RL_CHECKPOINT_DIR}/git_sha.txt" 2>/dev/null \
  || echo "unknown" > "${RL_CHECKPOINT_DIR}/git_sha.txt"

if [[ -f "${SFT_CHECKPOINT_DIR}/dataset_fingerprint.txt" ]]; then
  cp "${SFT_CHECKPOINT_DIR}/dataset_fingerprint.txt" \
     "${RL_CHECKPOINT_DIR}/dataset_fingerprint.txt"
fi

# ---------------------------------------------------------------------------
# Step 3: RL training loop stub
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: RL training ==="

CHECKPOINT_ROOT_VAL="${CHECKPOINT_ROOT:-${RUN_ROOT}}"

python - <<PYEOF
"""Stub RL training loop.

Replace with a real RL trainer, e.g.:
    from trl import PPOTrainer
    trainer = PPOTrainer(model=..., config=..., reward_model=...)
    trainer.train()
"""
import datetime, json, os, pathlib

rl_ckpt = pathlib.Path("${CHECKPOINT_ROOT_VAL}") / "${RUN_TAG}" / "rl_checkpoints"
sft_best = pathlib.Path("${SFT_BEST_DIR}")

print(f"RL training stub — initialising from SFT best: {sft_best}")
print(f"BASE_MODEL_ID       = ${BASE_MODEL_ID}")
print(f"LORA_RANK           = ${LORA_RANK}")
print(f"LORA_TARGET_MODULES = ${LORA_TARGET_MODULES}")
print(f"RL checkpoint dir   = {rl_ckpt}")

stub_record = {
    "step": 0,
    "reward": None,
    "train_loss": None,
    "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "note": "stub — replace with real RL reward metrics",
}
metrics_path = rl_ckpt / "metrics.jsonl"
with metrics_path.open("a") as fh:
    fh.write(json.dumps(stub_record) + "\n")

print(f"Stub metrics written: {metrics_path}")
print("RL training stub complete.")
PYEOF

# ---------------------------------------------------------------------------
# Step 4: Apply checkpoint policy for RL checkpoints
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Checkpoint policy ==="
python scripts/hpc/checkpoint_policy.py \
  --checkpoint-dir "${RL_CHECKPOINT_DIR}" \
  --run-config "${SFT_CHECKPOINT_DIR}/run_config.json" || true

echo ""
echo "=== RL job complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
