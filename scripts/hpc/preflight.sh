#!/usr/bin/env bash
# Preflight gate for the Nemotron SFT/RL pipeline.
#
# Checks:
#   1. Frozen contract vars match verified #14 values
#   2. CUDA is available (nvidia-smi)
#   3. MODEL_WEIGHTS_PATH (if set) exists
#   4. Dataset .sha256 sidecar is valid (if DATASET_PATH set)
#   5. Scratch disk has >50 GB free
#   6. Required Python packages are importable
#
# Exit 0 on pass, 1 on fail.
# Usage:
#   bash scripts/hpc/preflight.sh [--show] [--write-config] [--help]

set -euo pipefail

# ---------------------------------------------------------------------------
# Frozen #14 values — do not change without updating COMPETITION.md.
# ---------------------------------------------------------------------------
FROZEN_BASE_MODEL_ID="metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
FROZEN_LORA_RANK_MAX=32
FROZEN_NORMALIZER_ID="boxed_exact_or_numeric"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
err()  { echo "ERROR: $*" >&2; }
info() { echo "INFO:  $*"; }
pass() { echo "PASS:  $*"; }
fail() { err "$*"; EXIT_CODE=1; }

EXIT_CODE=0
SHOW=0
WRITE_CONFIG=0
LOCAL=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 [--show] [--write-config] [--local] [--help]

  --show          Print frozen contract vars and exit.
  --write-config  Write run_config.json to \${RUN_ROOT}/\${RUN_TAG}/.
  --local         Local smoke mode: skip frozen BASE_MODEL_ID and model-weights-path
                  checks; still runs CUDA, disk, package, and dataset SHA256 checks.
  --help          Show this message.

Required env vars for full preflight:
  BASE_MODEL_ID, LORA_RANK, LORA_TARGET_MODULES, NORMALIZER_ID

Optional env vars:
  MODEL_WEIGHTS_PATH, DATASET_PATH, RUN_ROOT, RUN_TAG,
  BASE_MODEL_REVISION, MAX_SEQ_LEN, SEED
EOF
}

for arg in "$@"; do
  case "$arg" in
    --show) SHOW=1 ;;
    --write-config) WRITE_CONFIG=1 ;;
    --local) LOCAL=1 ;;
    --help|-h) usage; exit 0 ;;
    *) err "Unknown argument: $arg"; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# --show: dump frozen contract vars then exit
# ---------------------------------------------------------------------------
if [[ $SHOW -eq 1 ]]; then
  echo "BASE_MODEL_ID         = ${BASE_MODEL_ID:-<unset>}"
  echo "BASE_MODEL_REVISION   = ${BASE_MODEL_REVISION:-<unset>}"
  echo "LORA_RANK             = ${LORA_RANK:-<unset>}"
  echo "LORA_TARGET_MODULES   = ${LORA_TARGET_MODULES:-<unset>}"
  echo "NORMALIZER_ID         = ${NORMALIZER_ID:-<unset>}"
  if [[ -z "${BASE_MODEL_ID:-}" || -z "${LORA_RANK:-}" || -z "${LORA_TARGET_MODULES:-}" ]]; then
    err "One or more required vars are unset."
    exit 1
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Frozen contract validation
# ---------------------------------------------------------------------------
info "Checking frozen contract vars..."

if [[ $LOCAL -eq 1 ]]; then
  info "LOCAL mode: skipping frozen BASE_MODEL_ID check (got '${BASE_MODEL_ID:-<unset>}')"
elif [[ -z "${BASE_MODEL_ID:-}" ]]; then
  fail "BASE_MODEL_ID is not set"
elif [[ "${BASE_MODEL_ID}" != "${FROZEN_BASE_MODEL_ID}" ]]; then
  fail "BASE_MODEL_ID mismatch: got '${BASE_MODEL_ID}', expected '${FROZEN_BASE_MODEL_ID}'"
else
  pass "BASE_MODEL_ID matches frozen value"
fi

if [[ -z "${LORA_RANK:-}" ]]; then
  fail "LORA_RANK is not set"
elif [[ "${LORA_RANK}" -gt "${FROZEN_LORA_RANK_MAX}" ]]; then
  fail "LORA_RANK=${LORA_RANK} exceeds max=${FROZEN_LORA_RANK_MAX}"
else
  pass "LORA_RANK=${LORA_RANK} within limit (<= ${FROZEN_LORA_RANK_MAX})"
fi

if [[ -z "${LORA_TARGET_MODULES:-}" ]]; then
  fail "LORA_TARGET_MODULES is not set"
else
  pass "LORA_TARGET_MODULES='${LORA_TARGET_MODULES}'"
fi

if [[ -z "${NORMALIZER_ID:-}" ]]; then
  fail "NORMALIZER_ID is not set"
elif [[ "${NORMALIZER_ID}" != "${FROZEN_NORMALIZER_ID}" ]]; then
  fail "NORMALIZER_ID mismatch: got '${NORMALIZER_ID}', expected '${FROZEN_NORMALIZER_ID}'"
else
  pass "NORMALIZER_ID matches frozen value"
fi

# ---------------------------------------------------------------------------
# 2. CUDA availability
# ---------------------------------------------------------------------------
info "Checking CUDA..."
if command -v nvidia-smi &>/dev/null; then
  if nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
    pass "nvidia-smi ok — ${GPU_COUNT} GPU(s) visible"
  else
    fail "nvidia-smi returned non-zero; CUDA may be unavailable"
  fi
else
  fail "nvidia-smi not found — CUDA driver not installed or not in PATH"
fi

# ---------------------------------------------------------------------------
# 3. Model weights path
# ---------------------------------------------------------------------------
if [[ $LOCAL -eq 1 ]]; then
  info "LOCAL mode: skipping model weights path check"
else
  info "Checking model weights path..."
  if [[ -z "${MODEL_WEIGHTS_PATH:-}" ]]; then
    info "MODEL_WEIGHTS_PATH is unset — will use HF download at runtime"
  else
    if [[ -d "${MODEL_WEIGHTS_PATH}" ]]; then
      pass "Model weights path exists: ${MODEL_WEIGHTS_PATH}"
    else
      fail "Model weights path does not exist: ${MODEL_WEIGHTS_PATH}"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 4. Dataset SHA256 verification
# ---------------------------------------------------------------------------
info "Checking dataset integrity..."
if [[ -z "${DATASET_PATH:-}" ]]; then
  info "DATASET_PATH not set — skipping dataset check"
else
  if [[ ! -f "${DATASET_PATH}" ]]; then
    fail "Dataset file not found: ${DATASET_PATH}"
  else
    SHA_FILE="${DATASET_PATH}.sha256"
    if [[ ! -f "$SHA_FILE" ]]; then
      fail "Dataset .sha256 sidecar not found: ${SHA_FILE}"
    else
      EXPECTED_SHA=$(awk '{print $1}' "${SHA_FILE}")
      ACTUAL_SHA=$(sha256sum "${DATASET_PATH}" 2>/dev/null | awk '{print $1}')
      if [[ "${EXPECTED_SHA}" == "${ACTUAL_SHA}" ]]; then
        pass "Dataset SHA256 ok: ${DATASET_PATH}"
      else
        fail "Dataset SHA256 mismatch for ${DATASET_PATH}"
        err "  expected: ${EXPECTED_SHA}"
        err "  got:      ${ACTUAL_SHA}"
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 5. Disk space (>50 GB free on RUN_ROOT or /)
# ---------------------------------------------------------------------------
info "Checking disk space..."
CHECK_DIR="${RUN_ROOT:-}"
if [[ -z "$CHECK_DIR" || ! -d "$CHECK_DIR" ]]; then
  CHECK_DIR="/"
fi
FREE_KB=$(df -k "$CHECK_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
FREE_GB=$(( FREE_KB / 1024 / 1024 ))
if [[ $FREE_GB -ge 50 ]]; then
  pass "Disk free: ${FREE_GB} GB on ${CHECK_DIR}"
else
  fail "Insufficient disk space: ${FREE_GB} GB free on ${CHECK_DIR} (need >= 50 GB)"
fi

# ---------------------------------------------------------------------------
# 6. Required Python packages
# ---------------------------------------------------------------------------
info "Checking required Python packages..."
PACKAGES=("torch" "transformers" "peft" "datasets" "trl")
for pkg in "${PACKAGES[@]}"; do
  if python3 -c "import ${pkg}" &>/dev/null 2>&1; then
    pass "Python package importable: ${pkg}"
  else
    fail "Python package NOT importable: ${pkg}"
  fi
done

# Check project src (best-effort; requires running from repo root)
if python3 -c "from src.training.sft_trainer import apply_loss_mask" &>/dev/null 2>&1; then
  pass "src.training.sft_trainer.apply_loss_mask importable"
else
  fail "src.training.sft_trainer.apply_loss_mask NOT importable (run from repo root with PYTHONPATH set)"
fi

# ---------------------------------------------------------------------------
# --write-config: emit run_config.json
# ---------------------------------------------------------------------------
if [[ $WRITE_CONFIG -eq 1 ]]; then
  if [[ -z "${RUN_ROOT:-}" || -z "${RUN_TAG:-}" ]]; then
    err "--write-config requires RUN_ROOT and RUN_TAG to be set"
    EXIT_CODE=1
  else
    CONFIG_DIR="${RUN_ROOT}/${RUN_TAG}"
    mkdir -p "$CONFIG_DIR"
    python3 - <<PYEOF
import json, os, subprocess, datetime, pathlib

def git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"

cfg = {
    "BASE_MODEL_ID": os.environ.get("BASE_MODEL_ID", ""),
    "BASE_MODEL_REVISION": os.environ.get("BASE_MODEL_REVISION", ""),
    "LORA_RANK": os.environ.get("LORA_RANK", ""),
    "LORA_TARGET_MODULES": os.environ.get("LORA_TARGET_MODULES", ""),
    "NORMALIZER_ID": os.environ.get("NORMALIZER_ID", ""),
    "MAX_SEQ_LEN": os.environ.get("MAX_SEQ_LEN", ""),
    "SEED": os.environ.get("SEED", ""),
    "RUN_TAG": os.environ.get("RUN_TAG", ""),
    "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_sha": git_sha(),
}
out = pathlib.Path(os.path.join("${CONFIG_DIR}", "run_config.json"))
out.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
print(f"Written: {out}")
PYEOF
    pass "run_config.json written to ${CONFIG_DIR}"
  fi
fi

# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "PREFLIGHT PASS — all checks OK"
else
  err "PREFLIGHT FAIL — one or more checks failed (see above)"
fi
exit $EXIT_CODE
