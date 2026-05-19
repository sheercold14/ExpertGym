#!/usr/bin/env bash
set -euo pipefail

# P0 SOTA-oriented main runs on sota_calib_v2_20260518.
# Run phases:
#   PHASE=train_gc  global-coefficient, 3 trainable expert coefficients
#   PHASE=train_gp  global-parameter, common+residual-style gate space

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PHASE="${PHASE:-train_gc}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gated_grpo}"
CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/sota_calib_v2_20260518/train128.prompts.jsonl}"
MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/sota_calib_v2_20260518/expert_rollouts}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
SEED_VALUE="${SEED_VALUE:-20260518}"

TOOL_EXPERT="$EXPERT_DIR/tool_expert_toolrl_qwen25_7b_sota_v2_train128_s${SAMPLES_PER_PROMPT}_seed${SEED_VALUE}.jsonl"
MEMORY_EXPERT="$EXPERT_DIR/memory_expert_rl_memoryagent7b_sota_v2_train128_s${SAMPLES_PER_PROMPT}_seed${SEED_VALUE}.jsonl"
CODE_EXPERT_REASONFLUX="$EXPERT_DIR/code_expert_reasonflux_coder7b_sota_v2_train128_s${SAMPLES_PER_PROMPT}_seed${SEED_VALUE}.jsonl"
CODE_EXPERT_DEEPSEEK="$EXPERT_DIR/code_expert_deepseek_r1_distill_qwen7b_sota_v2_train128_s${SAMPLES_PER_PROMPT}_seed${SEED_VALUE}.jsonl"
EXTRA_OPD_ROLLOUTS="${EXTRA_OPD_ROLLOUTS:-}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

require_opd_file() {
  if [[ ! -f "$1" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[dry-run warning] missing OPD rollout, command will still be printed: $1" >&2
      return 0
    fi
    echo "[error] missing required OPD rollout: $1" >&2
    echo "[hint] generate it with: POLICY=all bash skill/command/run_20260518_sota_v2_expert_rollouts.sh" >&2
    exit 2
  fi
}

opd_rollouts_csv() {
  local paths=("$TOOL_EXPERT" "$MEMORY_EXPERT" "$CODE_EXPERT_REASONFLUX" "$CODE_EXPERT_DEEPSEEK")
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
    require_opd_file "$path"
  done
  local IFS=,
  echo "${paths[*]}"
}

run_strategy() {
  local strategy="$1"
  local run_name="$2"
  local gpu_list="$3"
  shift 3
  local dynamic_opd_rollouts
  dynamic_opd_rollouts="$(opd_rollouts_csv)"
  env \
    STRATEGY="$strategy" \
    CONFIG="${CONFIG:-configs/gated_grpo.yaml}" \
    MODE="$MODE" \
    CALIBRATION="$CALIBRATION" \
    RUN_NAME="$run_name" \
    RUN_DIR="$RUN_ROOT/$run_name" \
    GPU_LIST="$gpu_list" \
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-$gpu_list}" \
    INIT_VALUE="${INIT_VALUE:-1.0}" \
    NUM_ITERS="${NUM_ITERS:-12}" \
    NUM_PROMPTS="${NUM_PROMPTS:-128}" \
    SAMPLES_PER_PROMPT="$SAMPLES_PER_PROMPT" \
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
    "$@" \
    bash skill/command/run_qbank_c033333_gate_strategy.sh
}

require_file "$CALIBRATION"

case "$PHASE" in
  train_gc)
    run_strategy global-coefficient "${RUN_NAME:-sota_v2_gc_init${INIT_VALUE:-1}_grpo_opd_ret_20260518}" "${GPU_LIST:-0,1}"
    ;;
  train_gp)
    run_strategy global-parameter "${RUN_NAME:-sota_v2_gp_init${INIT_VALUE:-1}_grpo_opd_ret_20260518}" "${GPU_LIST:-2,3}"
    ;;
  *)
    echo "[error] unknown PHASE=$PHASE; use train_gc or train_gp" >&2
    exit 2
    ;;
esac
