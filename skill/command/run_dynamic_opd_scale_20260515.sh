#!/usr/bin/env bash
set -euo pipefail

# Dynamic OPD-scale matrix.
# Main runs use OPD as an early repair signal, balance OPD/retention by task,
# and scale auxiliary losses from explicit GRPO-relative targets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
RUN_TAG="${RUN_TAG:-20260515_dynopdscale_i10}"
MONITOR_PORT="${MONITOR_PORT:-8783}"
DRY_RUN="${DRY_RUN:-0}"

CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
EXPERT_SAMPLES_PER_PROMPT="${EXPERT_SAMPLES_PER_PROMPT:-2}"

TOOL_EXPERT_ROLLOUT="${TOOL_EXPERT_ROLLOUT:-$EXPERT_DIR/tool_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl}"
MEMORY_EXPERT_ROLLOUT="${MEMORY_EXPERT_ROLLOUT:-$EXPERT_DIR/memory_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl}"
CODE_EXPERT_ROLLOUT="${CODE_EXPERT_ROLLOUT:-$EXPERT_DIR/code_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl}"
DYNAMIC_OPD_EXPERT_ROLLOUT="$TOOL_EXPERT_ROLLOUT,$MEMORY_EXPERT_ROLLOUT,$CODE_EXPERT_ROLLOUT"

NUM_ITERS="${NUM_ITERS:-10}"
NUM_PROMPTS="${NUM_PROMPTS:-96}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "[error] required file missing or empty: $path" >&2
    exit 2
  fi
}

require_file "$CALIBRATION"
require_file "$TOOL_EXPERT_ROLLOUT"
require_file "$MEMORY_EXPERT_ROLLOUT"
require_file "$CODE_EXPERT_ROLLOUT"
require_file "$ROOT/modes/opvec4/mode_manifest.json"

COMMON_ENV=(
  PY="$PY"
  ROOT="$ROOT"
  CALIBRATION="$CALIBRATION"
  NUM_ITERS="$NUM_ITERS"
  NUM_PROMPTS="$NUM_PROMPTS"
  SAMPLES_PER_PROMPT="$SAMPLES_PER_PROMPT"
  INIT_VALUE="0.3333333333333333"
  UPDATE_EPOCHS="1"
  UPDATE_BATCH_SIZE="4"
  BATCH_LOSS_REDUCTION="mean"
  OPTIMIZER_STEP_SCOPE="epoch"
  LOSS_GRANULARITY="sequence"
  STORE_TOKEN_LOGPROBS="0"
  TASK_NORMALIZE_ADVANTAGES="0"
  ADVANTAGE_NORMALIZATION="centered"
  USE_FRONTIER_WEIGHT="0"
  FRONTIER_ORDER="task-interleaved"
  FRONTIER_SHUFFLE_SEED="20260515"
  FRONTIER_TOOL_QUOTA="32"
  FRONTIER_MEMORY_QUOTA="32"
  FRONTIER_CODE_QUOTA="32"
  MAX_FRONTIER_ROWS_PER_TASK="32"
  LENGTH_NORMALIZE_POLICY_LOGPROB="1"
  LENGTH_NORMALIZE_LOGPROB="0"
  OPD_LENGTH_NORMALIZE_LOGPROB="0"
  RETENTION_LENGTH_NORMALIZE_LOGPROB="1"
  RETENTION_DYNAMIC_SCALE="1"
  RETENTION_TASK_BALANCED_LOSS_SCALE="1"
  RETENTION_SCALE_TARGET="0.5"
  RETENTION_SCALE_MIN="0.05"
  RETENTION_SCALE_MAX="100.0"
  OPTIMIZER="sgd"
  SGD_MOMENTUM="0.5"
  PERSIST_OPTIMIZER_STATE="1"
  PPO_LOSS_WEIGHT="6.0"
  MIN_GRAD_NORM_FOR_STEP="0.0"
  OPD_LOSS_WEIGHT="1.0"
  OPD_PAIRWISE_LOSS_WEIGHT="0.0"
  OPD_PAIRWISE_MARGIN="0.0"
  OPD_POSITIVE_REWARD_THRESHOLD="1.0"
  MAX_OPD_PAIRWISE_PAIRS_PER_ROW="0"
  OPD_DYNAMIC_SCALE="1"
  OPD_TASK_BALANCED_LOSS_SCALE="1"
  OPD_SCALE_MIN="0.05"
  OPD_SCALE_MAX="100.0"
  OPD_SCALE_RATE_HIGH="0.20"
  OPD_SCALE_RATE_MID="0.10"
  OPD_SCALE_RATE_LOW="0.03"
  OPD_SCALE_TARGET_HIGH="5.0"
  OPD_SCALE_TARGET_MID="3.0"
  OPD_SCALE_TARGET_LOW="1.0"
  OPD_SCALE_TARGET_TAIL="0.33"
  DYNAMIC_OPD_EXPERT_ROLLOUT="$DYNAMIC_OPD_EXPERT_ROLLOUT"
  DYNAMIC_OPD_TASKS="tool,memory,code"
  DYNAMIC_OPD_CURRENT_MAX_SUCCESS="0"
  DYNAMIC_OPD_POSITIVE_THRESHOLD="1.0"
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW="1"
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW="2"
  DYNAMIC_OPD_PER_TASK="32"
  USE_RETENTION="1"
  RETENTION_OBJECTIVE="nll"
  RETENTION_LOSS_WEIGHT="1.0"
  RETENTION_POSITIVE_REWARD_THRESHOLD="1.0"
  MAX_RETENTION_ROWS_PER_TASK="8"
  MAX_RETENTION_ROWS="24"
  MAX_NEW_TOKENS="1024"
  TOOL_MAX_NEW_TOKENS="512"
  CODE_MAX_NEW_TOKENS="4096"
  MEMORY_UPDATE_MAX_NEW_TOKENS="2048"
  MEMORY_FINAL_MAX_NEW_TOKENS="2048"
  MAX_PROMPT_TOKENS="8192"
  MAX_MODEL_LEN="12288"
  MAX_LOGPROB_TOKENS="12288"
  ROLLOUT_SHARDS="auto"
  ROLLOUT_BATCH_SIZE="32"
  TENSOR_PARALLEL_SIZE="1"
  GPU_MEMORY_UTILIZATION="0.82"
  POST_BAKE_SLEEP_SECONDS="10"
  SEED_VALUE="20260515"
)

run_command() {
  local session="$1"
  local label="$2"
  local gpus="$3"
  local strategy="$4"
  local run_name="$5"
  local lr="$6"
  local prior="$7"
  local max_delta="$8"
  local extra_env="$9"

  local run_dir="$ROOT/runs/gated_grpo/$run_name"
  mkdir -p "$run_dir"

  local full_cmd="cd '$REPO_ROOT' && env ${COMMON_ENV[*]} $extra_env STRATEGY='$strategy' GPU_LIST='$gpus' ROLLOUT_GPUS='$gpus' RUN_NAME='$run_name' RUN_DIR='$run_dir' LR='$lr' PRIOR_LOSS_WEIGHT='$prior' MAX_COEFF_DELTA='$max_delta' DRY_RUN='$DRY_RUN' bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee '$run_dir/run.log'"

  echo "[launch] $label session=$session gpus=$gpus run_dir=$run_dir"
  if is_truthy "$DRY_RUN"; then
    echo "[dry-run][$label] $full_cmd"
    zsh -lc "$full_cmd"
    return
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session exists: $session"
    return
  fi
  tmux new-session -d -s "$session" "$full_cmd"
}

A_RUN="dynscale_A_gc_${RUN_TAG}"
B_RUN="dynscale_B_gp_${RUN_TAG}"
C_RUN="dynscale_C_gp_fixedopd_${RUN_TAG}"
D_RUN="dynscale_D_gc_noopd_${RUN_TAG}"

run_command "opvec_dynscale_A_gc_${RUN_TAG}" "A global-coefficient dynamic OPD scale" "0,1" "global-coefficient" "$A_RUN" "0.06" "0.0" "0.45" ""
run_command "opvec_dynscale_B_gp_${RUN_TAG}" "B global-parameter dynamic OPD scale" "2,3" "global-parameter" "$B_RUN" "0.035" "0.0" "0.45" ""
run_command "opvec_dynscale_C_gp_fixedopd_${RUN_TAG}" "C global-parameter fixed OPD reference" "4,5" "global-parameter" "$C_RUN" "0.04" "0.0" "0.45" "OPD_DYNAMIC_SCALE=0 OPD_TASK_BALANCED_LOSS_SCALE=0 OPD_LOSS_WEIGHT=0.12 OPD_PAIRWISE_LOSS_WEIGHT=0.06 MAX_OPD_PAIRWISE_PAIRS_PER_ROW=2"
run_command "opvec_dynscale_D_gc_noopd_${RUN_TAG}" "D global-coefficient no OPD control" "6,7" "global-coefficient" "$D_RUN" "0.06" "0.0" "0.45" "OPD_DYNAMIC_SCALE=0 OPD_TASK_BALANCED_LOSS_SCALE=0 OPD_LOSS_WEIGHT=0.0 OPD_PAIRWISE_LOSS_WEIGHT=0.0 DYNAMIC_OPD_EXPERT_ROLLOUT="

if ! is_truthy "$DRY_RUN"; then
  MONITOR_SESSION="opvec_monitor_dynscale_${RUN_TAG}"
  if tmux has-session -t "$MONITOR_SESSION" 2>/dev/null; then
    echo "[monitor][skip] tmux session exists: $MONITOR_SESSION"
  else
    tmux new-session -d -s "$MONITOR_SESSION" \
      "cd '$REPO_ROOT' && '$PY' scripts/monitor/opvec_run_monitor.py --host 0.0.0.0 --port '$MONITOR_PORT' \
        --run-dir A_gc_dynscale='$ROOT/runs/gated_grpo/$A_RUN' \
        --run-dir B_gp_dynscale='$ROOT/runs/gated_grpo/$B_RUN' \
        --run-dir C_gp_fixedopd='$ROOT/runs/gated_grpo/$C_RUN' \
        --run-dir D_gc_noopd='$ROOT/runs/gated_grpo/$D_RUN' \
        --quiet"
    echo "[monitor] session=$MONITOR_SESSION url=http://127.0.0.1:$MONITOR_PORT"
  fi
fi
