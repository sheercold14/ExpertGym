#!/usr/bin/env bash
set -euo pipefail

# Experiment F for 2026-05-16.
# New-B ablation: keep B-style global-parameter OPD-only + retention, but use
# only code augmentation expert trajectories for dynamic code OPD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gated_grpo}"
RUN_NAME="${RUN_NAME:-expF_gp_code_aug_only_code_opd_20260516}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_NAME}"
CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
BASE_MODE="${BASE_MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
CODE_AUG_DIR="${CODE_AUG_DIR:-$ROOT/data/calibration/20260516_code_opd_aug}"

CODE_EXPERT_REASONFLUX="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl"
CODE_EXPERT_REASONFLUX2="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl"
CODE_EXPERT_DEEPSEEK="$CODE_AUG_DIR/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl"
CODE_EXPERT_MEMORY="$CODE_AUG_DIR/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

code_aug_rollouts_csv() {
  local paths=(
    "$CODE_EXPERT_REASONFLUX"
    "$CODE_EXPERT_REASONFLUX2"
    "$CODE_EXPERT_DEEPSEEK"
    "$CODE_EXPERT_MEMORY"
  )
  local path
  for path in "${paths[@]}"; do
    require_file "$path"
  done
  local IFS=,
  echo "${paths[*]}"
}

dynamic_opd_rollouts="$(code_aug_rollouts_csv)"

env \
  STRATEGY=global-parameter \
  INIT_VALUE=0.3333333333333333 \
  CONFIG=configs/gated_grpo.yaml \
  MODE="$BASE_MODE" \
  CALIBRATION="$CALIBRATION" \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  GPU_LIST="${GPU_LIST:-2,3}" \
  ROLLOUT_GPUS="${ROLLOUT_GPUS:-${GPU_LIST:-2,3}}" \
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
  PPO_LOSS_WEIGHT=0.0 \
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
  FRONTIER_TOOL_QUOTA=0 \
  FRONTIER_MEMORY_QUOTA=0 \
  FRONTIER_CODE_QUOTA=0 \
  DYNAMIC_OPD_EXPERT_ROLLOUT="$dynamic_opd_rollouts" \
  DYNAMIC_OPD_TASKS=code \
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
