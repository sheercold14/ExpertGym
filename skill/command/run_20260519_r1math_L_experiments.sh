#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PHASE="${PHASE:-L1}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gated_grpo}"
CONFIG="${CONFIG:-configs/gated_grpo_4expert_r1math_layer28.yaml}"
MODE="${MODE:-$ROOT/modes/opvec4_r1math_scaled_20260519/mode_manifest.json}"
CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
INIT_GATE_DIR="${INIT_GATE_DIR:-$ROOT/data/calibration/20260519_r1math_init_gates}"

EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
CODE_AUG_DIR="${CODE_AUG_DIR:-$ROOT/data/calibration/20260516_code_opd_aug}"

TOOL_EXPERT="$EXPERT_DIR/tool_expert_paper96_s2_seed20260514.jsonl"
MEMORY_EXPERT="$EXPERT_DIR/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT_OLD="$EXPERT_DIR/code_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT_REASONFLUX="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl"
CODE_EXPERT_REASONFLUX2="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl"
CODE_EXPERT_DEEPSEEK="$CODE_AUG_DIR/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl"
CODE_EXPERT_MEMORY="$CODE_AUG_DIR/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl"
L4_CALIBRATION_DEFAULT="$ROOT/data/calibration/20260519_l4_bfcl_tool_aug/qbank_c033333_paper96_plus_bfcl_tool16_seed20260519.prompts.jsonl"
L4_BFCL_TOOL_EXPERT_DEFAULT="$ROOT/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_official_answer_expert_rollouts_seed20260519.jsonl"
L5_CALIBRATION_DEFAULT="$ROOT/data/calibration/20260519_l5_cure_eval_code16/qbank_c033333_paper96_plus_cure_code16_seed20260519.prompts.jsonl"
L5_CURE_CODE_EXPERT_DEFAULT="$ROOT/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_expert_success_rollouts_seed20260519.jsonl"
L6_CALIBRATION_DEFAULT="$ROOT/data/calibration/20260519_l6_bfcl_tool16_cure_code16/qbank_c033333_paper96_plus_bfcl_tool16_cure_code16_seed20260519.prompts.jsonl"
L6_TOOL_CODE_EXPERT_DEFAULT="$ROOT/data/calibration/20260519_l6_bfcl_tool16_cure_code16/bfcl_tool16_cure_code16_extra_expert_rollouts_seed20260519.jsonl"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

expert_rollouts_csv() {
  local paths=(
    "$TOOL_EXPERT"
    "$MEMORY_EXPERT"
    "$CODE_EXPERT_OLD"
    "$CODE_EXPERT_REASONFLUX"
    "$CODE_EXPERT_DEEPSEEK"
    "$CODE_EXPERT_MEMORY"
  )
  if [[ -f "$CODE_EXPERT_REASONFLUX2" ]]; then
    paths+=("$CODE_EXPERT_REASONFLUX2")
  fi
  if [[ -n "${EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT:-}" ]]; then
    local extra_path
    IFS=',' read -r -a extra_paths <<< "$EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT"
    for extra_path in "${extra_paths[@]}"; do
      if [[ -n "$extra_path" ]]; then
        paths+=("$extra_path")
      fi
    done
  fi
  local path
  for path in "${paths[@]}"; do
    require_file "$path"
  done
  local IFS=,
  echo "${paths[*]}"
}

build_init_gate() {
  local output="$1"
  if [[ -f "$output" ]]; then
    echo "[init-gate] reusing: $output"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  "$PY" scripts/modes/build_constant_gate_checkpoint.py \
    --config "$CONFIG" \
    --mode-manifest "$MODE" \
    --gate-parameterization layer-band-parameter \
    --value 0.3333333333333333 \
    --expert-value reasoning=0.0 \
    --output "$output" >/dev/null
  echo "[init-gate] built: $output"
}

run_one() {
  local run_name="$1"
  local gpu_list="$2"
  local train_coefficients="${3:-}"
  local dynamic_opd_rollouts
  dynamic_opd_rollouts="$(expert_rollouts_csv)"

  require_file "$MODE"
  require_file "$CALIBRATION"

  local init_gate="$INIT_GATE_DIR/init_layer_band_parameter_28layer_tmc033_r0.json"
  build_init_gate "$init_gate"
  local effective_init_gate="${INIT_GATE_CHECKPOINT:-$init_gate}"

  local extra_train_coeff=()
  if [[ -n "$train_coefficients" ]]; then
    extra_train_coeff=(TRAIN_COEFFICIENTS="$train_coefficients")
  fi

  env \
    CONFIG="$CONFIG" \
    MODE="$MODE" \
    RUN_NAME="$run_name" \
    RUN_DIR="$RUN_ROOT/$run_name" \
    STRATEGY=layer-band-parameter \
    INIT_VALUE=0.3333333333333333 \
    INIT_GATE_CHECKPOINT="$effective_init_gate" \
    CALIBRATION="$CALIBRATION" \
    GPU_LIST="$gpu_list" \
    ROLLOUT_GPUS="$gpu_list" \
    NUM_ITERS="${NUM_ITERS:-20}" \
    START_ITERATION="${START_ITERATION:-1}" \
    NUM_PROMPTS="${NUM_PROMPTS:-96}" \
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}" \
    STORE_TOKEN_LOGPROBS=0 \
    OPTIMIZER=sgd \
    SGD_MOMENTUM=0.2 \
    PERSIST_OPTIMIZER_STATE=1 \
    LR="${LR:-0.25}" \
    PRIOR_LOSS_WEIGHT=0.0 \
    MAX_COEFF_DELTA=1.0 \
    UPDATE_EPOCHS=1 \
    UPDATE_BATCH_SIZE=4 \
    BATCH_LOSS_REDUCTION=mean \
    OPTIMIZER_STEP_SCOPE=epoch \
    LOSS_GRANULARITY=sequence \
    FRONTIER_ORDER=task-interleaved \
    FRONTIER_TOOL_QUOTA=32 \
    FRONTIER_MEMORY_QUOTA=32 \
    FRONTIER_CODE_QUOTA=32 \
    USE_RETENTION=1 \
    RETENTION_OBJECTIVE=nll \
    RETENTION_POSITIVE_REWARD_THRESHOLD=1.0 \
    RETENTION_TASK_BALANCED_LOSS_SCALE=1 \
    RETENTION_SCALE_TARGET=0.5 \
    PPO_LOSS_WEIGHT="${PPO_LOSS_WEIGHT:-0.0}" \
    OPD_LOSS_WEIGHT="${OPD_LOSS_WEIGHT:-1.0}" \
    OPD_POSITIVE_REWARD_THRESHOLD=1.0 \
    OPD_LENGTH_NORMALIZE_LOGPROB=1 \
    RETENTION_LENGTH_NORMALIZE_LOGPROB=1 \
    OPD_TASK_BALANCED_LOSS_SCALE=1 \
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
    DYNAMIC_OPD_PER_TASK=32 \
    DYNAMIC_OPD_REQUIRE_ALL_TASKS="${DYNAMIC_OPD_REQUIRE_ALL_TASKS:-0}" \
    TASK_WEIGHT_TOOL="${TASK_WEIGHT_TOOL:-0.5}" \
    TASK_WEIGHT_MEMORY="${TASK_WEIGHT_MEMORY:-2.0}" \
    TASK_WEIGHT_CODE="${TASK_WEIGHT_CODE:-1.5}" \
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
    SEED_VALUE="${SEED_VALUE:-20260519}" \
    PROGRESS_EVERY=10 \
    "${extra_train_coeff[@]}" \
    bash skill/command/run_qbank_c033333_gate_strategy.sh
}

case "$PHASE" in
  L1)
    run_one "${RUN_NAME:-expL1_r1math_layer28_hier_20it_20260519}" "${GPU_LIST:-0,1}" ""
    ;;
  L2)
    run_one "${RUN_NAME:-expL2_r1math_layer28_hier_freezeR1_20it_20260519}" "${GPU_LIST:-2,3}" "*.tool,*.memory,*.code"
    ;;
  L3)
    run_one "${RUN_NAME:-expL3_r1math_adaptive_20it_20260519}" "${GPU_LIST:-4,5}" "${TRAIN_COEFFICIENTS:-}"
    ;;
  L4)
    CALIBRATION="${L4_CALIBRATION:-$L4_CALIBRATION_DEFAULT}"
    NUM_PROMPTS="${NUM_PROMPTS:-112}"
    DYNAMIC_OPD_REQUIRE_ALL_TASKS="${DYNAMIC_OPD_REQUIRE_ALL_TASKS:-1}"
    EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT="${EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT:-$L4_BFCL_TOOL_EXPERT_DEFAULT}"
    run_one "${RUN_NAME:-expL4_r1math_layer28_hier_bfcltool16_reqallopd_20it_20260519}" "${GPU_LIST:-0,1}" ""
    ;;
  L5)
    CALIBRATION="${L5_CALIBRATION:-$L5_CALIBRATION_DEFAULT}"
    NUM_PROMPTS="${NUM_PROMPTS:-112}"
    EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT="${EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT:-$L5_CURE_CODE_EXPERT_DEFAULT}"
    run_one "${RUN_NAME:-expL5_r1math_layer28_hier_curecode16_20it_20260519}" "${GPU_LIST:-4,5}" ""
    ;;
  L6)
    CALIBRATION="${L6_CALIBRATION:-$L6_CALIBRATION_DEFAULT}"
    NUM_PROMPTS="${NUM_PROMPTS:-128}"
    TASK_WEIGHT_TOOL="${TASK_WEIGHT_TOOL:-1.0}"
    TASK_WEIGHT_MEMORY="${TASK_WEIGHT_MEMORY:-1.0}"
    TASK_WEIGHT_CODE="${TASK_WEIGHT_CODE:-1.0}"
    EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT="${EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT:-$L6_TOOL_CODE_EXPERT_DEFAULT}"
    run_one "${RUN_NAME:-expL6_r1math_layer28_hier_bfcltool16_curecode16_20it_20260519}" "${GPU_LIST:-6,7}" ""
    ;;
  *)
    echo "[error] PHASE must be L1, L2, L3, L4, L5, or L6; got $PHASE" >&2
    exit 2
    ;;
esac
