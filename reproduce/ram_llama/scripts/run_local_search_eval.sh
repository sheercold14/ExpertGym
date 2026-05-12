#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: run_local_search_eval.sh MODEL_PATH [RUN_ID] [MAX_SAMPLES]}"
RUN_ID="${2:-local-search-$(date +%Y%m%d-%H%M%S)}"
MAX_SAMPLES="${3:-}"

ARGS=(
  --model "${MODEL_PATH}"
  --run-id "${RUN_ID}"
  --benchmarks two_wiki
  --dtype auto
  --temperature 0.7
  --max-new-tokens 1024
  --search-mode zerosearch
  --search-backend local_context
  --search-topk 5
  --search-max-turns 5
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python "$(dirname "$0")/evaluate_model.py" "${ARGS[@]}"
