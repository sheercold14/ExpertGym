#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-paper-aligned-$(date +%Y%m%d-%H%M%S)}"
STORAGE_ROOT="${STORAGE_ROOT:-/tmp/shared-storage/ExpertGym/LLaMA}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/results}"
MODEL_SET="${MODEL_SET:-base,math,tool,search,ram,ram_plus}"
DO_PREPARE="${DO_PREPARE:-1}"
DO_MERGE="${DO_MERGE:-1}"
RUN_MATH="${RUN_MATH:-1}"
RUN_TOOL="${RUN_TOOL:-1}"
RUN_SEARCH="${RUN_SEARCH:-1}"

BASE="${BASE:-${STORAGE_ROOT}/models/base/Llama-3.2-3B-Instruct}"
MATH="${MATH:-${STORAGE_ROOT}/models/experts/Llama-3.2-3B-Instruct-GRPO-MATH-1EPOCH}"
TOOL="${TOOL:-${STORAGE_ROOT}/models/experts/ToolRL-Llama3.2-3B}"
SEARCH="${SEARCH:-${STORAGE_ROOT}/models/experts/ZeroSearch_google_V2_Llama_3.2_3B_Instruct}"
RAM="${RAM:-${STORAGE_ROOT}/models/merged/ram_math_tool_search}"
RAM_PLUS="${RAM_PLUS:-${STORAGE_ROOT}/models/merged/ram_plus_math_tool_search}"

if [ "${DO_PREPARE}" = "1" ]; then
  bash reproduce/ram_llama/scripts/paper_aligned/prepare_paper_aligned.sh
fi

if [ "${DO_MERGE}" = "1" ]; then
  bash reproduce/ram_llama/scripts/paper_aligned/run_merge_paper_aligned.sh both
fi

mkdir -p "${RESULT_ROOT}/${RUN_ID}"
{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'storage_root=%s\n' "${STORAGE_ROOT}"
  printf 'model_set=%s\n' "${MODEL_SET}"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RESULT_ROOT}/${RUN_ID}/run_manifest.txt"

model_path() {
  case "$1" in
    base) printf '%s\n' "${BASE}" ;;
    math) printf '%s\n' "${MATH}" ;;
    tool) printf '%s\n' "${TOOL}" ;;
    search) printf '%s\n' "${SEARCH}" ;;
    ram) printf '%s\n' "${RAM}" ;;
    ram_plus) printf '%s\n' "${RAM_PLUS}" ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a MODELS <<< "${MODEL_SET}"
for name in "${MODELS[@]}"; do
  path="$(model_path "${name}")"
  if [ "${RUN_MATH}" = "1" ]; then
    bash reproduce/ram_llama/scripts/paper_aligned/run_math_lmeval.sh "${path}" "${name}" "${RUN_ID}"
  fi
  if [ "${RUN_TOOL}" = "1" ]; then
    bash reproduce/ram_llama/scripts/paper_aligned/run_bfcl_official.sh "${path}" "${name}" "${RUN_ID}"
  fi
  if [ "${RUN_SEARCH}" = "1" ]; then
    bash reproduce/ram_llama/scripts/paper_aligned/run_search_zerosearch.sh "${path}" "${name}" "${RUN_ID}"
  fi
done

python reproduce/ram_llama/scripts/paper_aligned/summarize_paper_aligned.py "${RESULT_ROOT}/${RUN_ID}"
