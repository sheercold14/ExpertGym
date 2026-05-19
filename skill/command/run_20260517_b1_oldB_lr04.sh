#!/usr/bin/env bash
set -euo pipefail

# b1: reproduce the old B setting with only LR changed to 0.4.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-b1_oldB_lr04_20260517}"
RUN_DIR="${RUN_DIR:-/tmp/shared-storage/OnPolicy/runs/gated_grpo/$RUN_NAME}"
GPU_LIST="${GPU_LIST:-0,1}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$GPU_LIST}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "$DRY_RUN" != "1" && -d "$RUN_DIR" ]]; then
  mapfile -t EXISTING_RUN_FILES < <(
    find "$RUN_DIR" -mindepth 1 -maxdepth 1 \
      ! -name train.log \
      ! -name monitor_8794.log \
      -print 2>/dev/null
  )
  if (( ${#EXISTING_RUN_FILES[@]} > 0 )); then
    if [[ "$OVERWRITE" != "1" ]]; then
      echo "[error] RUN_DIR already exists and is not empty: $RUN_DIR" >&2
      echo "[error] set OVERWRITE=1 only if you intentionally want to remove it" >&2
      exit 2
    fi
    for path in "${EXISTING_RUN_FILES[@]}"; do
      rm -rf "$path"
    done
  fi
fi
mkdir -p "$RUN_DIR"

CALIBRATION="${CALIBRATION:-/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
EXPERT_ROLLOUTS="${EXPERT_ROLLOUTS:-/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl}"

env \
  DRY_RUN="$DRY_RUN" \
  GPU_LIST="$GPU_LIST" \
  ROLLOUT_GPUS="$ROLLOUT_GPUS" \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  STRATEGY=global-parameter \
  INIT_VALUE=0.3333333333333333 \
  CALIBRATION="$CALIBRATION" \
  NUM_ITERS="${NUM_ITERS:-15}" \
  NUM_PROMPTS=96 \
  SAMPLES_PER_PROMPT=4 \
  STORE_TOKEN_LOGPROBS=0 \
  OPTIMIZER=sgd \
  SGD_MOMENTUM=0.2 \
  PERSIST_OPTIMIZER_STATE=1 \
  LR="${LR:-0.4}" \
  PRIOR_LOSS_WEIGHT=0.0 \
  MAX_COEFF_DELTA=1.0 \
  UPDATE_EPOCHS=1 \
  UPDATE_BATCH_SIZE=4 \
  BATCH_LOSS_REDUCTION=mean \
  OPTIMIZER_STEP_SCOPE=epoch \
  LOSS_GRANULARITY=sequence \
  FRONTIER_ORDER=task-interleaved \
  FRONTIER_TOOL_QUOTA=0 \
  FRONTIER_MEMORY_QUOTA=0 \
  FRONTIER_CODE_QUOTA=0 \
  USE_RETENTION=1 \
  RETENTION_OBJECTIVE=nll \
  RETENTION_LOSS_WEIGHT=0.5 \
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
  PPO_LOSS_WEIGHT=0.0 \
  DYNAMIC_OPD_EXPERT_ROLLOUT="$EXPERT_ROLLOUTS" \
  DYNAMIC_OPD_TASKS=tool,memory,code \
  DYNAMIC_OPD_KEY=prompt_id \
  DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0 \
  DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0 \
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1 \
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2 \
  DYNAMIC_OPD_PER_TASK=32 \
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
  SEED_VALUE=20260515 \
  PROGRESS_EVERY=10 \
  bash skill/command/run_qbank_c033333_gate_strategy.sh
