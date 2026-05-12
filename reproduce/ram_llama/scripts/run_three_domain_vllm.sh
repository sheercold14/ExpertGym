#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
RUN_ID="${1:-llama-three-domain-full-vllm}"
MAX_SAMPLES="${2:-}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"

ROOT=/tmp/shared-storage/ExpertGym/LLaMA/models
declare -A MODELS=(
  [base]="${ROOT}/base/Llama-3.2-3B-Instruct"
  [math]="${ROOT}/experts/Llama-3.2-3B-Instruct-GRPO-MATH-1EPOCH"
  [tool]="${ROOT}/experts/ToolRL-Llama3.2-3B"
  [search]="${ROOT}/experts/ZeroSearch_google_V2_Llama_3.2_3B_Instruct"
  [ram]="${ROOT}/merged/ram_math_tool_search"
  [ram_plus]="${ROOT}/merged/ram_plus_math_tool_search"
)

run_model() {
  local name="$1"
  local path="$2"
  local extra=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    extra+=(--max-samples "${MAX_SAMPLES}")
  fi

  echo "==== ${name}: math ===="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" reproduce/ram_llama/scripts/evaluate_model_vllm.py \
    --model "${path}" --model-name "${name}" --run-id "${RUN_ID}" \
    --benchmarks gsm8k,math500 \
    --max-new-tokens 512 --temperature 0 --dtype bfloat16 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    "${extra[@]}"

  echo "==== ${name}: tool ===="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" reproduce/ram_llama/scripts/evaluate_model_vllm.py \
    --model "${path}" --model-name "${name}" --run-id "${RUN_ID}" \
    --benchmarks bfcl_live,bfcl_non_live \
    --max-new-tokens 512 --temperature 0 --dtype bfloat16 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    "${extra[@]}"

  echo "==== ${name}: local search ===="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" reproduce/ram_llama/scripts/evaluate_model_vllm.py \
    --model "${path}" --model-name "${name}" --run-id "${RUN_ID}" \
    --benchmarks two_wiki \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.95 --dtype bfloat16 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --search-mode zerosearch --search-backend local_context --search-topk 5 --search-max-turns 5 \
    --search-info-max-chars 3500 \
    "${extra[@]}"
}

for name in base math tool search ram ram_plus; do
  run_model "${name}" "${MODELS[$name]}"
done
