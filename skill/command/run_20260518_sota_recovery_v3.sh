#!/usr/bin/env bash
set -euo pipefail

# Train on SOTA Recovery Calibration v3.
#
# Phases:
#   PHASE=train_gc     global-coefficient
#   PHASE=train_gp     global-parameter
#   PHASE=train_hier   layer-band-parameter
#
# This launcher is intentionally explicit about OPD rollout sources because
# train128 mixes sota_v2 and code_p0 recoverable rows.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PHASE="${PHASE:-train_hier}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gated_grpo}"
BANK="${BANK:-$ROOT/data/calibration/sota_recovery_calib_v3_20260518}"
CALIBRATION="${CALIBRATION:-$BANK/train128.prompts.jsonl}"
MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
SEED_VALUE="${SEED_VALUE:-20260518}"

V2_EXPERT_DIR="${V2_EXPERT_DIR:-$ROOT/data/calibration/sota_calib_v2_20260518/expert_rollouts}"
CODE_P0_EXPERT_DIR="${CODE_P0_EXPERT_DIR:-$ROOT/data/calibration/code_p0_v3_20260518/expert_rollouts}"

TOOL_EXPERT="${TOOL_EXPERT:-$V2_EXPERT_DIR/tool_expert_toolrl_qwen25_7b_sota_v2_train128_s4_seed20260518.jsonl}"
MEMORY_EXPERT="${MEMORY_EXPERT:-$V2_EXPERT_DIR/memory_expert_rl_memoryagent7b_sota_v2_train128_s4_seed20260518.jsonl}"
V2_CODE_REASONFLUX="${V2_CODE_REASONFLUX:-$V2_EXPERT_DIR/code_expert_reasonflux_coder7b_sota_v2_train128_s8_seed20260518.jsonl}"
V2_CODE_DEEPSEEK="${V2_CODE_DEEPSEEK:-$V2_EXPERT_DIR/code_expert_deepseek_r1_distill_qwen7b_sota_v2_train128_s8_seed20260518.jsonl}"
P0_CODE_REASONFLUX="${P0_CODE_REASONFLUX:-$CODE_P0_EXPERT_DIR/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl}"
P0_CODE_DEEPSEEK="${P0_CODE_DEEPSEEK:-$CODE_P0_EXPERT_DIR/code_expert_deepseek_r1_distill_qwen7b_code_p0_v3_train64_s8_seed20260518.merged.jsonl}"
EXTRA_OPD_ROLLOUTS="${EXTRA_OPD_ROLLOUTS:-}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

opd_rollouts_csv() {
  local paths=(
    "$TOOL_EXPERT"
    "$MEMORY_EXPERT"
    "$V2_CODE_REASONFLUX"
    "$V2_CODE_DEEPSEEK"
    "$P0_CODE_REASONFLUX"
    "$P0_CODE_DEEPSEEK"
  )
  if [[ -n "$EXTRA_OPD_ROLLOUTS" ]]; then
    local old_ifs="$IFS"
    IFS=','
    local extra
    for extra in $EXTRA_OPD_ROLLOUTS; do
      if [[ -n "$extra" ]]; then
        paths+=("$extra")
      fi
    done
    IFS="$old_ifs"
  fi
  local path
  for path in "${paths[@]}"; do
    require_file "$path"
  done
  local IFS=,
  echo "${paths[*]}"
}

run_strategy() {
  local strategy="$1"
  local default_name="$2"
  local default_gpus="$3"
  local dynamic_opd_rollouts
  dynamic_opd_rollouts="$(opd_rollouts_csv)"
  env \
    STRATEGY="$strategy" \
    CONFIG="${CONFIG:-configs/gated_grpo.yaml}" \
    QB="${QB:-$BANK}" \
    MODE="$MODE" \
    CALIBRATION="$CALIBRATION" \
    RUN_NAME="${RUN_NAME:-$default_name}" \
    RUN_DIR="$RUN_ROOT/${RUN_NAME:-$default_name}" \
    GPU_LIST="${GPU_LIST:-$default_gpus}" \
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-${GPU_LIST:-$default_gpus}}" \
    INIT_VALUE="${INIT_VALUE:-1.0}" \
    NUM_ITERS="${NUM_ITERS:-12}" \
    NUM_PROMPTS="${NUM_PROMPTS:-128}" \
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}" \
    STORE_TOKEN_LOGPROBS=0 \
    OPTIMIZER=sgd \
    SGD_MOMENTUM="${SGD_MOMENTUM:-0.2}" \
    PERSIST_OPTIMIZER_STATE=1 \
    LR="${LR:-0.1876}" \
    PRIOR_LOSS_WEIGHT=0.0 \
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-1.0}" \
    UPDATE_EPOCHS=1 \
    UPDATE_BATCH_SIZE=4 \
    BATCH_LOSS_REDUCTION=mean \
    OPTIMIZER_STEP_SCOPE=epoch \
    LOSS_GRANULARITY=sequence \
    FRONTIER_ORDER=task-interleaved \
    FRONTIER_SAMPLE_BEFORE_LIMIT=1 \
    FRONTIER_ROWS_PER_TASK="${FRONTIER_ROWS_PER_TASK:-4}" \
    USE_RETENTION=1 \
    RETENTION_OBJECTIVE=nll \
    RETENTION_POSITIVE_REWARD_THRESHOLD=1.0 \
    MAX_RETENTION_ROWS_PER_TASK="${MAX_RETENTION_ROWS_PER_TASK:-8}" \
    RETENTION_SAMPLE_BEFORE_LIMIT=1 \
    RETENTION_DYNAMIC_SCALE=1 \
    RETENTION_TASK_BALANCED_LOSS_SCALE=1 \
    RETENTION_SCALE_TARGET="${RETENTION_SCALE_TARGET:-0.5}" \
    PPO_LOSS_WEIGHT="${PPO_LOSS_WEIGHT:-1.0}" \
    OPD_LOSS_WEIGHT="${OPD_LOSS_WEIGHT:-1.0}" \
    OPD_POSITIVE_REWARD_THRESHOLD=1.0 \
    OPD_LENGTH_NORMALIZE_LOGPROB=1 \
    OPD_TASK_BALANCED_LOSS_SCALE=1 \
    OPD_DYNAMIC_SCALE=1 \
    OPD_SCALE_TARGET_HIGH="${OPD_SCALE_TARGET_HIGH:-5.0}" \
    OPD_SCALE_TARGET_MID="${OPD_SCALE_TARGET_MID:-3.0}" \
    OPD_SCALE_TARGET_LOW="${OPD_SCALE_TARGET_LOW:-1.0}" \
    OPD_SCALE_TARGET_TAIL="${OPD_SCALE_TARGET_TAIL:-0.33}" \
    LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
    LENGTH_NORMALIZE_LOGPROB=0 \
    TASK_NORMALIZE_ADVANTAGES=0 \
    ADVANTAGE_NORMALIZATION=centered \
    USE_FRONTIER_WEIGHT=0 \
    DYNAMIC_OPD_EXPERT_ROLLOUT="$dynamic_opd_rollouts" \
    DYNAMIC_OPD_TASKS=tool,memory,code \
    DYNAMIC_OPD_KEY=prompt_id \
    DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0 \
    DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0 \
    DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1 \
    DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2 \
    DYNAMIC_OPD_PER_TASK="${DYNAMIC_OPD_PER_TASK:-32}" \
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
    SEED_VALUE="$SEED_VALUE" \
    PROGRESS_EVERY=10 \
    bash skill/command/run_qbank_c033333_gate_strategy.sh
}

require_file "$CALIBRATION"
require_file "$MODE"

case "$PHASE" in
  train_gc)
    run_strategy global-coefficient "sota_recovery_v3_gc_init1_grpo_opd_ret_20260518" "0,1"
    ;;
  train_gp)
    run_strategy global-parameter "sota_recovery_v3_gp_init1_grpo_opd_ret_20260518" "2,3"
    ;;
  train_hier)
    run_strategy layer-band-parameter "sota_recovery_v3_hier_init1_grpo_opd_ret_20260518" "4,5"
    ;;
  *)
    echo "[error] unknown PHASE=$PHASE; use train_gc, train_gp, or train_hier" >&2
    exit 2
    ;;
esac
