#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/tmp/shared-storage/ExpertGym/LLaMA}"
PY="${PY:-python}"
MODE="${1:-both}"

BASE="${BASE:-${STORAGE_ROOT}/models/base/Llama-3.2-3B-Instruct}"
MATH="${MATH:-${STORAGE_ROOT}/models/experts/Llama-3.2-3B-Instruct-GRPO-MATH-1EPOCH}"
TOOL="${TOOL:-${STORAGE_ROOT}/models/experts/ToolRL-Llama3.2-3B}"
SEARCH="${SEARCH:-${STORAGE_ROOT}/models/experts/ZeroSearch_google_V2_Llama_3.2_3B_Instruct}"
RAM_OUT="${RAM_OUT:-${STORAGE_ROOT}/models/merged/ram_math_tool_search}"
RAM_PLUS_OUT="${RAM_PLUS_OUT:-${STORAGE_ROOT}/models/merged/ram_plus_math_tool_search}"
EPS="${EPS:-1e-5}"
ALPHA="${ALPHA:-2.0}"
OVERWRITE="${OVERWRITE:-0}"

run_merge() {
  local out="$1"
  local r="$2"
  if [ -e "${out}/ram_merge_config.json" ] && [ "${OVERWRITE}" != "1" ]; then
    printf 'skip existing merged model: %s\n' "${out}"
    return
  fi
  mkdir -p "$(dirname "${out}")"
  "${PY}" reproduce/ram_llama/scripts/ram_merge.py \
    --base "${BASE}" \
    --experts "${MATH}" "${TOOL}" "${SEARCH}" \
    --out "${out}" \
    --eps "${EPS}" \
    --r "${r}" \
    --alpha "${ALPHA}"
}

case "${MODE}" in
  ram) run_merge "${RAM_OUT}" "0" ;;
  ram_plus) run_merge "${RAM_PLUS_OUT}" "0.10" ;;
  both)
    run_merge "${RAM_OUT}" "0"
    run_merge "${RAM_PLUS_OUT}" "0.10"
    ;;
  *)
    printf 'usage: %s [ram|ram_plus|both]\n' "$0" >&2
    exit 2
    ;;
esac
