#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Load secrets from repo-local .env (ignored by git).
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-data/.kaggle}"
if [[ "${KAGGLE_CONFIG_DIR}" != /* ]]; then
  KAGGLE_CONFIG_DIR="${ROOT_DIR}/${KAGGLE_CONFIG_DIR}"
fi
export KAGGLE_CONFIG_DIR
mkdir -p "${KAGGLE_CONFIG_DIR}"

exec "${ROOT_DIR}/.venv/bin/kaggle" "$@"

