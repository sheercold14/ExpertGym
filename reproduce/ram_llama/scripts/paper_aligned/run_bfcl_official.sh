#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: run_bfcl_official.sh MODEL_PATH MODEL_NAME RUN_ID}"
MODEL_NAME="${2:?usage: run_bfcl_official.sh MODEL_PATH MODEL_NAME RUN_ID}"
RUN_ID="${3:?usage: run_bfcl_official.sh MODEL_PATH MODEL_NAME RUN_ID}"

STORAGE_ROOT="${STORAGE_ROOT:-/tmp/shared-storage/ExpertGym/LLaMA}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/results}"
BFCL_REPO="${BFCL_REPO:-/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard}"
PY="${PY:-python}"
GPU="${GPU:-0}"
BFCL_MODEL_ID="${BFCL_MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct-FC}"
TEST_CATEGORY="${TEST_CATEGORY:-live,non_live}"
BACKEND="${BACKEND:-vllm}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-${RESULT_ROOT}/${RUN_ID}/${MODEL_NAME}/bfcl_project}"
RESULT_DIR="${RESULT_DIR:-${BFCL_PROJECT_ROOT}/result}"
SCORE_DIR="${SCORE_DIR:-${BFCL_PROJECT_ROOT}/score}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${SCORE_DIR}/data_overall.csv" ]; then
  printf 'skip existing BFCL results: %s\n' "${SCORE_DIR}/data_overall.csv"
  exit 0
fi

if [ ! -d "${BFCL_REPO}/bfcl_eval" ]; then
  printf 'BFCL repo not found: %s\n' "${BFCL_REPO}" >&2
  exit 1
fi

mkdir -p "${BFCL_PROJECT_ROOT}"
export BFCL_PROJECT_ROOT
export PYTHONPATH="${BFCL_REPO}:${PYTHONPATH:-}"
PY_BIN_DIR="$("${PY}" - <<'PY'
import sys
from pathlib import Path
print(Path(sys.executable).resolve().parent)
PY
)"
export PATH="${PY_BIN_DIR}:${PATH}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -m bfcl_eval generate \
  --model "${BFCL_MODEL_ID}" \
  --test-category "${TEST_CATEGORY}" \
  --backend "${BACKEND}" \
  --num-gpus 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --local-model-path "${MODEL_PATH}" \
  --result-dir "${RESULT_DIR}" \
  --allow-overwrite

"${PY}" -m bfcl_eval evaluate \
  --model "${BFCL_MODEL_ID}" \
  --test-category "${TEST_CATEGORY}" \
  --result-dir "${RESULT_DIR}" \
  --score-dir "${SCORE_DIR}"
