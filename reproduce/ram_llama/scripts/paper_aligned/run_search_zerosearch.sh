#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: run_search_zerosearch.sh MODEL_PATH MODEL_NAME RUN_ID}"
MODEL_NAME="${2:?usage: run_search_zerosearch.sh MODEL_PATH MODEL_NAME RUN_ID}"
RUN_ID="${3:?usage: run_search_zerosearch.sh MODEL_PATH MODEL_NAME RUN_ID}"

STORAGE_ROOT="${STORAGE_ROOT:-/tmp/shared-storage/ExpertGym/LLaMA}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/results}"
PY="${PY:-python}"
GPU="${GPU:-0}"
BENCHMARKS="${BENCHMARKS:-nq_open,two_wiki}"
SEARCH_BACKEND="${SEARCH_BACKEND:-wiki}"
SEARCH_URL="${SEARCH_URL:-http://localhost:6002/retrieve}"
SEARCH_TOPK="${SEARCH_TOPK:-5}"
SEARCH_MAX_TURNS="${SEARCH_MAX_TURNS:-5}"
SEARCH_INFO_MAX_CHARS="${SEARCH_INFO_MAX_CHARS:-3500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

ARGS=(
  reproduce/ram_llama/scripts/evaluate_model_vllm.py
  --model "${MODEL_PATH}"
  --model-name "${MODEL_NAME}"
  --run-id "${RUN_ID}"
  --output-root "${RESULT_ROOT}"
  --benchmarks "${BENCHMARKS}"
  --search-mode zerosearch
  --search-backend "${SEARCH_BACKEND}"
  --search-url "${SEARCH_URL}"
  --search-topk "${SEARCH_TOPK}"
  --search-max-turns "${SEARCH_MAX_TURNS}"
  --search-info-max-chars "${SEARCH_INFO_MAX_CHARS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --dtype "${DTYPE:-bfloat16}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
)

if [ -n "${MAX_SAMPLES}" ]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ARGS[@]}"
