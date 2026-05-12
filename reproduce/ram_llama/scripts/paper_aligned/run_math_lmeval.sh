#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: run_math_lmeval.sh MODEL_PATH MODEL_NAME RUN_ID}"
MODEL_NAME="${2:?usage: run_math_lmeval.sh MODEL_PATH MODEL_NAME RUN_ID}"
RUN_ID="${3:?usage: run_math_lmeval.sh MODEL_PATH MODEL_NAME RUN_ID}"

STORAGE_ROOT="${STORAGE_ROOT:-/tmp/shared-storage/ExpertGym/LLaMA}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/results}"
PY="${PY:-python}"
GPU="${GPU:-0}"
TASKS="${TASKS:-gsm8k,minerva_math500}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"
OUT="${RESULT_ROOT}/${RUN_ID}/${MODEL_NAME}/math_lmeval"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ "${SKIP_EXISTING}" = "1" ] && find "${OUT}" -name 'results*.json' -print -quit 2>/dev/null | grep -q .; then
  printf 'skip existing Math results: %s\n' "${OUT}"
  exit 0
fi

if ! "${PY}" -c "import lm_eval" >/dev/null 2>&1; then
  printf 'lm-evaluation-harness is missing in %s. Install with: pip install lm-eval[vllm]\n' "${PY}" >&2
  exit 1
fi

ARGS=(
  -m lm_eval
  --model vllm
  --model_args "pretrained=${MODEL_PATH},dtype=${DTYPE},tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUT}"
  --log_samples
)

if [ "${APPLY_CHAT_TEMPLATE}" = "1" ]; then
  ARGS+=(--apply_chat_template)
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ARGS[@]}"
