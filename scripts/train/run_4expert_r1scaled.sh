#!/usr/bin/env bash
set -euo pipefail

# 4-expert R1-scaled layer-band experiments (3-band and 28-layer).
# Uses OPD + Retention only (no GRPO policy gradient).
# Init: T/M/C = 1/3, R = 0.  Target: memory >= 0.55 within 10 iterations.
#
# Usage:
#   PHASE=3band  GPU_LIST=0,1 bash scripts/train/run_4expert_r1scaled.sh
#   PHASE=layer28 GPU_LIST=2,3 bash scripts/train/run_4expert_r1scaled.sh
#   PHASE=layer28_hier GPU_LIST=2,3 bash scripts/train/run_4expert_r1scaled.sh
#   PHASE=both   GPU_LIST_3BAND=0,1 GPU_LIST_LAYER28=2,3 bash scripts/train/run_4expert_r1scaled.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PHASE="${PHASE:-3band}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gated_grpo}"
CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
MODE="${MODE:-$ROOT/modes/opvec4_r1scaled_20260518/mode_manifest.json}"

# Expert rollout paths (augmented code OPD from expC)
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
CODE_AUG_DIR="${CODE_AUG_DIR:-$ROOT/data/calibration/20260516_code_opd_aug}"

TOOL_EXPERT="$EXPERT_DIR/tool_expert_paper96_s2_seed20260514.jsonl"
MEMORY_EXPERT="$EXPERT_DIR/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT_OLD="$EXPERT_DIR/code_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT_REASONFLUX="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl"
CODE_EXPERT_REASONFLUX2="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl"
CODE_EXPERT_DEEPSEEK="$CODE_AUG_DIR/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl"
CODE_EXPERT_MEMORY="$CODE_AUG_DIR/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl"

INIT_GATE_DIR="${INIT_GATE_DIR:-$ROOT/data/calibration/20260518_r1scaled_init_gates}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

build_init_gate() {
  local config="$1"
  local strategy="$2"
  local output="$3"
  if [[ -f "$output" ]]; then
    echo "[init-gate] reusing existing: $output"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  echo "[init-gate] building: $output"
  "$PY" scripts/modes/build_constant_gate_checkpoint.py \
    --config "$config" \
    --mode-manifest "$MODE" \
    --gate-parameterization "$strategy" \
    --value 0.3333333333333333 \
    --expert-value "reasoning=0.0" \
    --output "$output" >/dev/null
  echo "[init-gate] done: $output"
}

augmented_expert_rollouts_csv() {
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
  local path
  for path in "${paths[@]}"; do
    require_file "$path"
  done
  local IFS=,
  echo "${paths[*]}"
}

# Common env vars shared between 3-band and 28-layer experiments.
# Matches expC settings except: STRATEGY=layer-band, PPO=0, task weights adjusted.
COMMON_ENV=(
  STRATEGY=layer-band
  INIT_VALUE=0.3333333333333333
  CALIBRATION="$CALIBRATION"
  NUM_ITERS="${NUM_ITERS:-10}"
  NUM_PROMPTS=96
  SAMPLES_PER_PROMPT=4
  STORE_TOKEN_LOGPROBS=0
  OPTIMIZER=sgd
  SGD_MOMENTUM=0.2
  PERSIST_OPTIMIZER_STATE=1
  PRIOR_LOSS_WEIGHT=0.0
  MAX_COEFF_DELTA=1.0
  UPDATE_EPOCHS=1
  UPDATE_BATCH_SIZE=4
  BATCH_LOSS_REDUCTION=mean
  OPTIMIZER_STEP_SCOPE=epoch
  LOSS_GRANULARITY=sequence
  FRONTIER_ORDER=task-interleaved
  FRONTIER_TOOL_QUOTA=32
  FRONTIER_MEMORY_QUOTA=32
  FRONTIER_CODE_QUOTA=32
  USE_RETENTION=1
  RETENTION_OBJECTIVE=nll
  RETENTION_POSITIVE_REWARD_THRESHOLD=1.0
  RETENTION_TASK_BALANCED_LOSS_SCALE=1
  RETENTION_SCALE_TARGET=0.5
  PPO_LOSS_WEIGHT=0.0
  OPD_LOSS_WEIGHT=1.0
  OPD_POSITIVE_REWARD_THRESHOLD=1.0
  OPD_LENGTH_NORMALIZE_LOGPROB=1
  RETENTION_LENGTH_NORMALIZE_LOGPROB=1
  OPD_TASK_BALANCED_LOSS_SCALE=1
  LENGTH_NORMALIZE_POLICY_LOGPROB=1
  LENGTH_NORMALIZE_LOGPROB=0
  TASK_NORMALIZE_ADVANTAGES=0
  ADVANTAGE_NORMALIZATION=centered
  USE_FRONTIER_WEIGHT=0
  DYNAMIC_OPD_TASKS=tool,memory,code
  DYNAMIC_OPD_KEY=prompt_id
  DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0
  DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2
  DYNAMIC_OPD_PER_TASK=32
  TASK_WEIGHT_TOOL=0.5
  TASK_WEIGHT_MEMORY=2.0
  TASK_WEIGHT_CODE=1.5
  MAX_NEW_TOKENS=1024
  TOOL_MAX_NEW_TOKENS=512
  CODE_MAX_NEW_TOKENS=4096
  MEMORY_UPDATE_MAX_NEW_TOKENS=2048
  MEMORY_FINAL_MAX_NEW_TOKENS=2048
  MAX_PROMPT_TOKENS=8192
  MAX_MODEL_LEN=12288
  MAX_LOGPROB_TOKENS=12288
  ROLLOUT_BATCH_SIZE=32
  ROLLOUT_SHARDS=auto
  TENSOR_PARALLEL_SIZE=1
  GPU_MEMORY_UTILIZATION=0.82
  TEMPERATURE=0.7
  TOP_P=0.95
  SEED_VALUE=20260518
  PROGRESS_EVERY=10
  DRY_RUN="${DRY_RUN:-0}"
)

run_strategy() {
  env "${COMMON_ENV[@]}" "$@" bash skill/command/run_qbank_c033333_gate_strategy.sh
}

run_3band() {
  local gpu_list="${GPU_LIST_3BAND:-${GPU_LIST:-0,1}}"
  local dynamic_opd_rollouts
  dynamic_opd_rollouts="$(augmented_expert_rollouts_csv)"

  local init_gate="${INIT_GATE_CHECKPOINT_3BAND:-$INIT_GATE_DIR/init_layer_band_3band_tmc033_r0.json}"
  if [[ -z "${INIT_GATE_CHECKPOINT_3BAND:-}" ]]; then
    build_init_gate configs/gated_grpo_4expert_r1scaled.yaml layer-band "$init_gate"
  else
    require_file "$init_gate"
    echo "[init-gate] using 3-band continuation checkpoint: $init_gate"
  fi

  echo "=== 4-Expert R1-Scaled 3-Band Experiment ==="
  echo "GPU: $gpu_list  LR: 0.25  Iters: ${NUM_ITERS:-10}"

  run_strategy \
    CONFIG=configs/gated_grpo_4expert_r1scaled.yaml \
    MODE="$MODE" \
    RUN_NAME="${RUN_NAME_3BAND:-expD_r1scaled_3band_20260518}" \
    RUN_DIR="$RUN_ROOT/${RUN_NAME_3BAND:-expD_r1scaled_3band_20260518}" \
    GPU_LIST="$gpu_list" \
    ROLLOUT_GPUS="$gpu_list" \
    LR=0.25 \
    INIT_GATE_CHECKPOINT="$init_gate" \
    DYNAMIC_OPD_EXPERT_ROLLOUT="$dynamic_opd_rollouts"
}

run_layer28() {
  local gpu_list="${GPU_LIST_LAYER28:-${GPU_LIST:-0,1}}"
  local strategy="${LAYER28_STRATEGY:-layer-band}"
  local dynamic_opd_rollouts
  dynamic_opd_rollouts="$(augmented_expert_rollouts_csv)"

  local safe_strategy="${strategy//[^A-Za-z0-9_]/_}"
  local init_gate="$INIT_GATE_DIR/init_${safe_strategy}_28layer_tmc033_r0.json"
  build_init_gate configs/gated_grpo_4expert_r1scaled_layer28.yaml "$strategy" "$init_gate"

  echo "=== 4-Expert R1-Scaled 28-Layer Experiment ==="
  echo "GPU: $gpu_list  STRATEGY: $strategy  LR: ${LAYER28_LR:-0.10}  Iters: ${NUM_ITERS:-10}"

  run_strategy \
    STRATEGY="$strategy" \
    CONFIG=configs/gated_grpo_4expert_r1scaled_layer28.yaml \
    MODE="$MODE" \
    RUN_NAME="${RUN_NAME_LAYER28:-expD_r1scaled_layer28_20260518}" \
    RUN_DIR="$RUN_ROOT/${RUN_NAME_LAYER28:-expD_r1scaled_layer28_20260518}" \
    GPU_LIST="$gpu_list" \
    ROLLOUT_GPUS="$gpu_list" \
    LR="${LAYER28_LR:-0.10}" \
    INIT_GATE_CHECKPOINT="$init_gate" \
    DYNAMIC_OPD_EXPERT_ROLLOUT="$dynamic_opd_rollouts"
}

run_layer28_hier() {
  LAYER28_STRATEGY="${LAYER28_STRATEGY:-layer-band-parameter}" \
  LAYER28_LR="${LAYER28_LR:-0.25}" \
  RUN_NAME_LAYER28="${RUN_NAME_LAYER28:-expD_r1scaled_layer28_hier_20260518}" \
  run_layer28
}

case "$PHASE" in
  3band)
    run_3band
    ;;
  layer28)
    run_layer28
    ;;
  layer28_hier)
    run_layer28_hier
    ;;
  both)
    echo "[info] Running 3-band first, then 28-layer sequentially."
    run_3band
    run_layer28
    ;;
  *)
    echo "[error] unknown PHASE=$PHASE; use 3band, layer28, layer28_hier, or both" >&2
    exit 2
    ;;
esac
