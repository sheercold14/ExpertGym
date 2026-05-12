#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash skill/command/run_full_eval_suite.sh /path/to/baked_model [model_name]

Purpose:
  Run the external full-evaluation harnesses for a baked OP-VEC checkpoint.
  The current cleaned worktree does not keep the old Evaluation_all directory,
  so this wrapper calls the maintained harness copies under AgentMerging.

Common examples:
  RUN_TOOL=1 RUN_MEMORY=0 RUN_CODE=0 bash skill/command/run_full_eval_suite.sh $ROOT/checkpoints/fixed_frontier7_global_continue_e3b_20260511 global-e3b
  RUN_TOOL=0 RUN_MEMORY=1 RUN_CODE=0 MEMORY_DATASETS="eval_50 eval_100" bash skill/command/run_full_eval_suite.sh $ROOT/checkpoints/fixed_frontier7_global_continue_e3b_20260511 global-e3b
  RUN_TOOL=0 RUN_MEMORY=0 RUN_CODE=1 CODE_GPU_GROUPS="[[0,1]]" bash skill/command/run_full_eval_suite.sh $ROOT/checkpoints/fixed_frontier7_global_continue_e3b_20260511 global-e3b

Environment switches:
  RUN_TOOL=1|0
  RUN_MEMORY=1|0
  RUN_CODE=1|0

Tool/BFCL:
  TOOL_GPU=0
  TOOL_PORT=8001
  TOOL_CATEGORIES=parallel,parallel_multiple,live_parallel,live_parallel_multiple

Memory/HotpotQA:
  MEMORY_GPU_IDS=0
  MEMORY_TP=1
  MEMORY_DATASETS="eval_50 eval_100 eval_qa_1_32768 eval_qa_1_65536"
  MEMORY_MAX_NEW_TOKENS=2048
  MEMORY_MAX_INPUT_LENGTH=32768
  MEMORY_VLLM_MAX_MODEL_LEN=32768

Code/CURE:
  CODE_GPU_GROUPS="[[0,1]]"
  CODE_MAX_TEST=8
  CODE_MAX_GENERATION_TOKEN=10000
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${1:?Missing model path. Use --help for examples.}"
MODEL_NAME="${2:-$(basename "$MODEL_PATH")}"

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opvec-gated-grpo-full-eval}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"

AGENTMERGING_ROOT="${AGENTMERGING_ROOT:-/mnt/cache/wuruixiao/users/lsc/AgentMerging}"
EVAL_ALL_SCRIPTS="${EVAL_ALL_SCRIPTS:-$AGENTMERGING_ROOT/skill/Evaluation_all/scripts}"

BFCL_HARNESS="${BFCL_HARNESS:-$EVAL_ALL_SCRIPTS/run_bfcl_tool_harness.py}"
MEMORY_HARNESS="${MEMORY_HARNESS:-$EVAL_ALL_SCRIPTS/run_hotpotqa_memory_harness.py}"
CURE_HARNESS="${CURE_HARNESS:-$EVAL_ALL_SCRIPTS/run_cure_full_harness.sh}"

RUN_TOOL="${RUN_TOOL:-1}"
RUN_MEMORY="${RUN_MEMORY:-1}"
RUN_CODE="${RUN_CODE:-1}"

TOOL_GPU="${TOOL_GPU:-0}"
TOOL_PORT="${TOOL_PORT:-8001}"
TOOL_CATEGORIES="${TOOL_CATEGORIES:-parallel,parallel_multiple,live_parallel,live_parallel_multiple}"

MEMORY_GPU_IDS="${MEMORY_GPU_IDS:-0}"
MEMORY_TP="${MEMORY_TP:-1}"
MEMORY_DATASETS="${MEMORY_DATASETS:-eval_50 eval_100 eval_qa_1_32768 eval_qa_1_65536}"
MEMORY_MAX_NEW_TOKENS="${MEMORY_MAX_NEW_TOKENS:-2048}"
MEMORY_MAX_INPUT_LENGTH="${MEMORY_MAX_INPUT_LENGTH:-32768}"
MEMORY_VLLM_MAX_MODEL_LEN="${MEMORY_VLLM_MAX_MODEL_LEN:-32768}"
MEMORY_VLLM_BATCH_SIZE="${MEMORY_VLLM_BATCH_SIZE:-64}"

CODE_GPU_GROUPS="${CODE_GPU_GROUPS:-[[0,1]]}"
CODE_MAX_TEST="${CODE_MAX_TEST:-8}"
CODE_MAX_GENERATION_TOKEN="${CODE_MAX_GENERATION_TOKEN:-10000}"

SUMMARY_DIR="${SUMMARY_DIR:-$ROOT/eval/full_suite/$MODEL_NAME/$RUN_ID}"
mkdir -p "$SUMMARY_DIR/logs"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "[error] missing $label: $path" >&2
    exit 2
  fi
}

require_path "$MODEL_PATH/config.json" "model config"
require_path "$MODEL_PATH/tokenizer_config.json" "tokenizer config"

if [[ "$RUN_TOOL" == "1" ]]; then
  require_path "$BFCL_HARNESS" "BFCL harness"
fi
if [[ "$RUN_MEMORY" == "1" ]]; then
  require_path "$MEMORY_HARNESS" "memory harness"
fi
if [[ "$RUN_CODE" == "1" ]]; then
  require_path "$CURE_HARNESS" "CURE harness"
fi

MANIFEST="$SUMMARY_DIR/full_eval_manifest.env"
{
  echo "MODEL_PATH=$MODEL_PATH"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "RUN_ID=$RUN_ID"
  echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
  echo "SUMMARY_DIR=$SUMMARY_DIR"
  echo "RUN_TOOL=$RUN_TOOL"
  echo "RUN_MEMORY=$RUN_MEMORY"
  echo "RUN_CODE=$RUN_CODE"
  echo "BFCL_HARNESS=$BFCL_HARNESS"
  echo "MEMORY_HARNESS=$MEMORY_HARNESS"
  echo "CURE_HARNESS=$CURE_HARNESS"
} > "$MANIFEST"

echo "[run] model: $MODEL_PATH"
echo "[run] model_name: $MODEL_NAME"
echo "[run] run_id: $RUN_ID"
echo "[run] summary_dir: $SUMMARY_DIR"

if [[ "$RUN_TOOL" == "1" ]]; then
  echo "[tool] BFCL categories=$TOOL_CATEGORIES gpu=$TOOL_GPU port=$TOOL_PORT"
  BFCL_MODEL_NAME="${MODEL_NAME}-tool-${RUN_ID}"
  BFCL_MODEL_NAME="${BFCL_MODEL_NAME//_/-}"
  BFCL_MODEL_NAME="${BFCL_MODEL_NAME//\//-}"
  BFCL_MODEL_NAME="${BFCL_MODEL_NAME// /-}"
  "$PY" "$BFCL_HARNESS" \
    --model-path "$MODEL_PATH" \
    --model-name "$BFCL_MODEL_NAME" \
    --display-name "$MODEL_NAME" \
    --categories "$TOOL_CATEGORIES" \
    --gpu "$TOOL_GPU" \
    --port "$TOOL_PORT" \
    --run-id "$RUN_ID" \
    --no-memory \
    2>&1 | tee "$SUMMARY_DIR/logs/tool_bfcl.log"
fi

if [[ "$RUN_MEMORY" == "1" ]]; then
  echo "[memory] HotpotQA datasets=$MEMORY_DATASETS gpu_ids=$MEMORY_GPU_IDS tp=$MEMORY_TP"
  read -r -a MEMORY_DATASET_ARGS <<< "$MEMORY_DATASETS"
  "$PY" "$MEMORY_HARNESS" \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --experiment-name "$EXPERIMENT_NAME-memory" \
    --run-id "$RUN_ID" \
    --datasets "${MEMORY_DATASET_ARGS[@]}" \
    --gpu-ids "$MEMORY_GPU_IDS" \
    --tensor-parallel-size "$MEMORY_TP" \
    --max-new-tokens "$MEMORY_MAX_NEW_TOKENS" \
    --max-input-length "$MEMORY_MAX_INPUT_LENGTH" \
    --vllm-max-model-len "$MEMORY_VLLM_MAX_MODEL_LEN" \
    --vllm-batch-size "$MEMORY_VLLM_BATCH_SIZE" \
    2>&1 | tee "$SUMMARY_DIR/logs/memory_hotpotqa.log"
fi

if [[ "$RUN_CODE" == "1" ]]; then
  echo "[code] CURE gpu_groups=$CODE_GPU_GROUPS"
  RUN_ID="$RUN_ID" \
  GPU_GROUPS="$CODE_GPU_GROUPS" \
  MAX_TEST="$CODE_MAX_TEST" \
  MAX_GENERATION_TOKEN="$CODE_MAX_GENERATION_TOKEN" \
  FEEDBACK_ROOT="$ROOT/eval/cure_feedback" \
  bash "$CURE_HARNESS" "$MODEL_PATH" "$MODEL_NAME" "$EXPERIMENT_NAME-code" \
    2>&1 | tee "$SUMMARY_DIR/logs/code_cure.log"
fi

echo "[done] summary_dir: $SUMMARY_DIR"
