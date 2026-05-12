#!/usr/bin/env bash
# SLURM batch script for Nemotron SFT fine-tuning.
#
# Resources: 4 GPUs, 64 GB RAM, 24 h wall time.
#
# Usage:
#   sbatch --job-name=nemotron-sft \
#     --export=ALL,RUN_TAG=${RUN_TAG},BASE_MODEL_ID=${BASE_MODEL_ID},LORA_RANK=${LORA_RANK} \
#     scripts/hpc/submit_sft.sbatch
#
# Required env vars (set before sbatch or via --export):
#   RUN_TAG, BASE_MODEL_ID, LORA_RANK, LORA_TARGET_MODULES, NORMALIZER_ID,
#   DATA_ROOT, RUN_ROOT
#
# Optional env vars (sensible defaults shown in script):
#   SLURM_PARTITION, VENV_PATH, TOKENIZED_DIR, CONFIG_PATH,
#   INPUT_JSONL, CHECKPOINT_ROOT, LOG_ROOT, HF_HOME

#SBATCH --job-name=nemotron-sft
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
echo "=== Nemotron SFT Job ==="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-<not in SLURM>}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-<not in SLURM>}"
echo "RUN_TAG        = ${RUN_TAG:-<unset>}"
echo "BASE_MODEL_ID  = ${BASE_MODEL_ID:-<unset>}"
echo "LORA_RANK      = ${LORA_RANK:-<unset>}"
echo "Started        = $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Fail fast if frozen contract vars are missing.
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
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# Step 1: Preflight gate
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Preflight ==="
bash scripts/hpc/preflight.sh || {
  echo "ERROR: Preflight failed — aborting SFT job"
  exit 1
}

# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------
LOG_DIR="${LOG_ROOT:-${RUN_ROOT}/${RUN_TAG}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT:-${RUN_ROOT}}/${RUN_TAG}/checkpoints"
TOKENIZED_DIR="${TOKENIZED_DIR:-${DATA_ROOT}/processed/tokenized_${RUN_TAG}}"
CONFIG_PATH="${CONFIG_PATH:-configs/lora_sft.yaml}"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${TOKENIZED_DIR}" "${RUN_DIR}"

git rev-parse HEAD > "${CHECKPOINT_DIR}/git_sha.txt" 2>/dev/null \
  || echo "unknown" > "${CHECKPOINT_DIR}/git_sha.txt"

# ---------------------------------------------------------------------------
# Step 2: Tokenize dataset
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Tokenize dataset ==="
INPUT_JSONL="${INPUT_JSONL:-${DATA_ROOT}/processed/train.jsonl}"

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "ERROR: Input JSONL not found: ${INPUT_JSONL}"
  exit 1
fi

python scripts/hpc/tokenize_dataset.py \
  --input  "${INPUT_JSONL}" \
  --output "${TOKENIZED_DIR}" \
  --config "${CONFIG_PATH}"

if [[ -f "${TOKENIZED_DIR}/dataset_fingerprint.json" ]]; then
  cp "${TOKENIZED_DIR}/dataset_fingerprint.json" \
     "${CHECKPOINT_DIR}/dataset_fingerprint.txt"
fi

# ---------------------------------------------------------------------------
# Step 3: SFT training
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: SFT training ==="
bash scripts/hpc/preflight.sh --write-config
if [[ -f "${RUN_DIR}/run_config.json" ]]; then
  cp "${RUN_DIR}/run_config.json" "${CHECKPOINT_DIR}/run_config.json"
fi

RESUME_ARG=()
if LATEST=$(python scripts/hpc/resume_from_latest.py --checkpoint-dir "$CHECKPOINT_DIR" 2>/dev/null); then
    [ -n "$LATEST" ] && RESUME_ARG=(--resume-from-checkpoint "$LATEST")
fi

python scripts/hpc/run_sft.py \
    --config "$CONFIG_PATH" \
    --data-dir "$TOKENIZED_DIR" \
    --output-dir "$CHECKPOINT_DIR" \
    "${RESUME_ARG[@]}" \
    2>&1 | tee -a "$RUN_DIR/train.log"

# ---------------------------------------------------------------------------
# Step 4: Apply checkpoint policy
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Checkpoint policy ==="
echo "INFO: Checkpoint policy was applied by run_sft.py."

echo ""
echo "=== SFT job complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
