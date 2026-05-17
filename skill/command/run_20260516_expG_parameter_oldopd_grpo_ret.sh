#!/usr/bin/env bash
set -euo pipefail

# Experiment G for 2026-05-16.
# Direct parameter gate ablation: learn all 196*3 module-level task-vector
# coefficients with OPD + GRPO + retention, using only the original expert
# rollouts and no code augmentation pool.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gated_grpo}"
RUN_NAME="${RUN_NAME:-expG_param_oldopd_grpo_ret_20260516}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_NAME}"
CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
BASE_MODE="${BASE_MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"

TOOL_EXPERT="$EXPERT_DIR/tool_expert_paper96_s2_seed20260514.jsonl"
MEMORY_EXPERT="$EXPERT_DIR/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT="$EXPERT_DIR/code_expert_paper96_s2_seed20260514.jsonl"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

old_expert_rollouts_csv() {
  local paths=("$TOOL_EXPERT" "$MEMORY_EXPERT" "$CODE_EXPERT")
  local path
  for path in "${paths[@]}"; do
    require_file "$path"
  done
  local IFS=,
  echo "${paths[*]}"
}

dynamic_opd_rollouts="$(old_expert_rollouts_csv)"

env \
  STRATEGY=parameter \
  INIT_VALUE=0.3333333333333333 \
  CONFIG=configs/gated_grpo.yaml \
  MODE="$BASE_MODE" \
  CALIBRATION="$CALIBRATION" \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  GPU_LIST="${GPU_LIST:-6,7}" \
  ROLLOUT_GPUS="${ROLLOUT_GPUS:-${GPU_LIST:-6,7}}" \
  NUM_ITERS="${NUM_ITERS:-20}" \
  NUM_PROMPTS=96 \
  SAMPLES_PER_PROMPT=4 \
  STORE_TOKEN_LOGPROBS=0 \
  OPTIMIZER=sgd \
  SGD_MOMENTUM=0.2 \
  PERSIST_OPTIMIZER_STATE=1 \
  LR="${LR:-0.1876}" \
  PRIOR_LOSS_WEIGHT=0.0 \
  MAX_COEFF_DELTA=1.0 \
  UPDATE_EPOCHS=1 \
  UPDATE_BATCH_SIZE=4 \
  BATCH_LOSS_REDUCTION=mean \
  OPTIMIZER_STEP_SCOPE=epoch \
  LOSS_GRANULARITY=sequence \
  FRONTIER_ORDER=task-interleaved \
  PPO_LOSS_WEIGHT=1.0 \
  BEST_RESPONSE_LOSS_WEIGHT=0.0 \
  PAIRWISE_LOSS_WEIGHT=0.0 \
  USE_RETENTION=1 \
  RETENTION_OBJECTIVE=nll \
  RETENTION_POSITIVE_REWARD_THRESHOLD=1.0 \
  RETENTION_TASK_BALANCED_LOSS_SCALE=1 \
  RETENTION_SCALE_TARGET=0.5 \
  OPD_LOSS_WEIGHT=1.0 \
  OPD_POSITIVE_REWARD_THRESHOLD=1.0 \
  OPD_LENGTH_NORMALIZE_LOGPROB=1 \
  RETENTION_LENGTH_NORMALIZE_LOGPROB=1 \
  OPD_TASK_BALANCED_LOSS_SCALE=1 \
  LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
  LENGTH_NORMALIZE_LOGPROB=0 \
  TASK_NORMALIZE_ADVANTAGES=0 \
  ADVANTAGE_NORMALIZATION=centered \
  USE_FRONTIER_WEIGHT=0 \
  FRONTIER_TOOL_QUOTA=32 \
  FRONTIER_MEMORY_QUOTA=32 \
  FRONTIER_CODE_QUOTA=32 \
  DYNAMIC_OPD_EXPERT_ROLLOUT="$dynamic_opd_rollouts" \
  DYNAMIC_OPD_TASKS=tool,memory,code \
  DYNAMIC_OPD_KEY=prompt_id \
  DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0 \
  DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0 \
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1 \
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2 \
  DYNAMIC_OPD_PER_TASK=32 \
  TASK_WEIGHT_TOOL=1.0 \
  TASK_WEIGHT_MEMORY=1.0 \
  TASK_WEIGHT_CODE=1.0 \
  MAX_NEW_TOKENS=1024 \
  TOOL_MAX_NEW_TOKENS=512 \
  CODE_MAX_NEW_TOKENS=4096 \
  MEMORY_UPDATE_MAX_NEW_TOKENS=2048 \
  MEMORY_FINAL_MAX_NEW_TOKENS=2048 \
  MAX_PROMPT_TOKENS=8192 \
  MAX_MODEL_LEN=12288 \
  MAX_LOGPROB_TOKENS=12288 \
  ROLLOUT_BATCH_SIZE=32 \
  ROLLOUT_SHARDS=auto \
  TENSOR_PARALLEL_SIZE=1 \
  GPU_MEMORY_UTILIZATION=0.82 \
  TEMPERATURE=0.7 \
  TOP_P=0.95 \
  SEED_VALUE=20260516 \
  PROGRESS_EVERY=10 \
  DRY_RUN="${DRY_RUN:-0}" \
  bash skill/command/run_qbank_c033333_gate_strategy.sh
