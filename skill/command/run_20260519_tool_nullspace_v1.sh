#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

CONFIG="${CONFIG:-configs/gated_grpo_4expert_r1math_layer28.yaml}"
MODE="${MODE:-$ROOT/modes/opvec4_r1math_scaled_20260519/mode_manifest.json}"
CALIB_DIR="${CALIB_DIR:-$ROOT/data/calibration/20260519_tool_nullspace_v1}"
CALIBRATION="${CALIBRATION:-$CALIB_DIR/tool32_memory32_code40_toolnullspace_seed20260519.prompts.jsonl}"
INIT_GATE_CHECKPOINT="${INIT_GATE_CHECKPOINT:-$CALIB_DIR/init_gates/init_layer_band_parameter_28layer_tmc033_r005.json}"
TOOL_NULLSPACE_REPLAY_ROLLOUT="${TOOL_NULLSPACE_REPLAY_ROLLOUT:-$CALIB_DIR/toolnullspace_tool_replay_rollouts_seed20260519.jsonl}"
EXTRA_TOOL_CODE_EXPERT="$CALIB_DIR/toolnullspace_extra_expert_rollouts_seed20260519.jsonl"

EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
CODE_AUG_DIR="${CODE_AUG_DIR:-$ROOT/data/calibration/20260516_code_opd_aug}"

TOOL_EXPERT="$EXPERT_DIR/tool_expert_paper96_s2_seed20260514.jsonl"
MEMORY_EXPERT="$EXPERT_DIR/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT_OLD="$EXPERT_DIR/code_expert_paper96_s2_seed20260514.jsonl"
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

dynamic_opd_rollouts_csv() {
  local paths=(
    "$TOOL_EXPERT"
    "$MEMORY_EXPERT"
    "$CODE_EXPERT_OLD"
    "$CODE_EXPERT_REASONFLUX"
    "$CODE_EXPERT_DEEPSEEK"
    "$CODE_EXPERT_MEMORY"
    "$EXTRA_TOOL_CODE_EXPERT"
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

require_file "$CONFIG"
require_file "$MODE"
require_file "$CALIBRATION"
require_file "$INIT_GATE_CHECKPOINT"
require_file "$TOOL_NULLSPACE_REPLAY_ROLLOUT"
require_file "$EXTRA_TOOL_CODE_EXPERT"

export CONFIG
export MODE
export RUN_NAME="${RUN_NAME:-expM1_toolnull_r1init005_layer28_opdret_nullspace_20260519}"
export RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
export STRATEGY=layer-band-parameter
export INIT_VALUE=0.3333333333333333
export INIT_GATE_CHECKPOINT
export CALIBRATION
export GPU_LIST="${GPU_LIST:-0,1}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-$GPU_LIST}"
export NUM_ITERS="${NUM_ITERS:-20}"
export START_ITERATION="${START_ITERATION:-1}"
export NUM_PROMPTS="${NUM_PROMPTS:-104}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
export STORE_TOKEN_LOGPROBS=0
export OPTIMIZER=sgd
export SGD_MOMENTUM=0.2
export PERSIST_OPTIMIZER_STATE=1
export LR="${LR:-0.25}"
export PRIOR_LOSS_WEIGHT=0.0
export MAX_COEFF_DELTA=1.0
export UPDATE_EPOCHS=1
export UPDATE_BATCH_SIZE=4
export BATCH_LOSS_REDUCTION=mean
export OPTIMIZER_STEP_SCOPE=epoch
export LOSS_GRANULARITY=sequence
export FRONTIER_ORDER=task-interleaved
export FRONTIER_TOOL_QUOTA=32
export FRONTIER_MEMORY_QUOTA=32
export FRONTIER_CODE_QUOTA=40
export USE_RETENTION=1
export RETENTION_OBJECTIVE=nll
export RETENTION_LOSS_WEIGHT=1.0
export RETENTION_POSITIVE_REWARD_THRESHOLD=1.0
export RETENTION_TASK_BALANCED_LOSS_SCALE=1
export RETENTION_DYNAMIC_SCALE=0
export PPO_LOSS_WEIGHT=0.0
export OPD_LOSS_WEIGHT=1.0
export OPD_PAIRWISE_LOSS_WEIGHT=0.0
export OPD_POSITIVE_REWARD_THRESHOLD=1.0
export OPD_LENGTH_NORMALIZE_LOGPROB=1
export RETENTION_LENGTH_NORMALIZE_LOGPROB=1
export OPD_TASK_BALANCED_LOSS_SCALE=1
export OPD_DYNAMIC_SCALE=0
export LENGTH_NORMALIZE_POLICY_LOGPROB=1
export LENGTH_NORMALIZE_LOGPROB=0
export TASK_NORMALIZE_ADVANTAGES=0
export ADVANTAGE_NORMALIZATION=centered
export USE_FRONTIER_WEIGHT=0
export DYNAMIC_OPD_EXPERT_ROLLOUT="${DYNAMIC_OPD_EXPERT_ROLLOUT:-$(dynamic_opd_rollouts_csv)}"
export DYNAMIC_OPD_TASKS=tool,memory,code
export DYNAMIC_OPD_KEY=prompt_id
export DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0
export DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0
export DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1
export DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2
export DYNAMIC_OPD_PER_TASK=32
export DYNAMIC_OPD_QUOTA=tool=32,memory=32,code=40
export DYNAMIC_OPD_REQUIRE_ALL_TASKS=1
export TASK_WEIGHT_TOOL=1.0
export TASK_WEIGHT_MEMORY=1.0
export TASK_WEIGHT_CODE=1.0
export TOOL_NULLSPACE_GATE_GRADIENTS=1
export TOOL_NULLSPACE_REPLAY_ROLLOUT
export TOOL_NULLSPACE_ROWS=16
export TOOL_NULLSPACE_MIN_ROWS=16
export TOOL_NULLSPACE_RANK=0
export TOOL_NULLSPACE_EPS=1e-6
export TOOL_NULLSPACE_POSITIVE_REWARD_THRESHOLD=1.0
export MAX_NEW_TOKENS=1024
export TOOL_MAX_NEW_TOKENS=512
export CODE_MAX_NEW_TOKENS=4096
export MEMORY_UPDATE_MAX_NEW_TOKENS=2048
export MEMORY_FINAL_MAX_NEW_TOKENS=2048
export MAX_PROMPT_TOKENS=8192
export MAX_MODEL_LEN=12288
export MAX_LOGPROB_TOKENS=12288
export ROLLOUT_BATCH_SIZE=32
export ROLLOUT_SHARDS=auto
export TENSOR_PARALLEL_SIZE=1
export GPU_MEMORY_UTILIZATION=0.82
export TEMPERATURE=0.7
export TOP_P=0.95
export SEED_VALUE="${SEED_VALUE:-20260519}"
export PROGRESS_EVERY=10

echo "[tool-nullspace-v1] run=$RUN_NAME gpu=$GPU_LIST prompts=$NUM_PROMPTS iters=$NUM_ITERS"
echo "[tool-nullspace-v1] calibration=$CALIBRATION"
echo "[tool-nullspace-v1] init_gate=$INIT_GATE_CHECKPOINT"
echo "[tool-nullspace-v1] tool_nullspace_replay=$TOOL_NULLSPACE_REPLAY_ROLLOUT"

bash skill/command/run_qbank_c033333_gate_strategy.sh
