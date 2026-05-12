#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: run_eval_suite.sh MODEL_PATH [RUN_ID] [MAX_SAMPLES]}"
RUN_ID="${2:-eval-$(date +%Y%m%d-%H%M%S)}"
MAX_SAMPLES="${3:-}"

ARGS=(
  --model "${MODEL_PATH}"
  --run-id "${RUN_ID}"
  --benchmarks gsm8k,math500,nq_open,two_wiki,bfcl_live,bfcl_non_live
  --dtype auto
  --temperature 0
  --max-new-tokens 512
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python "$(dirname "$0")/evaluate_model.py" "${ARGS[@]}"
