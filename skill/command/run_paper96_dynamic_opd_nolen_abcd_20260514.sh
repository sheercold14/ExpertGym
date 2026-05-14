#!/usr/bin/env bash
set -euo pipefail

# Paper96 dynamic-OPD no-length-normalization ABCD matrix.
# All four runs use same-prompt dynamic OPD from current all-fail prompts.
# A/C disable preservation+prior to test whether OPD recovers yesterday's strong
# gate movement; B/D keep NLL all-success preservation+prior as protected variants.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
RUN_TAG="${RUN_TAG:-20260514_dynopd_nolen_abcd_i8}"
MONITOR_PORT="${MONITOR_PORT:-8771}"
DRY_RUN="${DRY_RUN:-0}"

CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
EXPERT_SAMPLES_PER_PROMPT="${EXPERT_SAMPLES_PER_PROMPT:-2}"

TOOL_EXPERT_ROLLOUT="${TOOL_EXPERT_ROLLOUT:-$EXPERT_DIR/tool_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl}"
MEMORY_EXPERT_ROLLOUT="${MEMORY_EXPERT_ROLLOUT:-$EXPERT_DIR/memory_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl}"
CODE_EXPERT_ROLLOUT="${CODE_EXPERT_ROLLOUT:-$EXPERT_DIR/code_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl}"
DYNAMIC_OPD_EXPERT_ROLLOUT="$TOOL_EXPERT_ROLLOUT,$MEMORY_EXPERT_ROLLOUT,$CODE_EXPERT_ROLLOUT"

NUM_ITERS="${NUM_ITERS:-8}"
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
  FRONTIER_SHUFFLE_SEED="20260514"
  FRONTIER_TOOL_QUOTA="32"
  FRONTIER_MEMORY_QUOTA="32"
  FRONTIER_CODE_QUOTA="32"
  MAX_FRONTIER_ROWS_PER_TASK="32"
  LENGTH_NORMALIZE_POLICY_LOGPROB="1"
  LENGTH_NORMALIZE_LOGPROB="0"
  OPTIMIZER="sgd"
  SGD_MOMENTUM="0.8"
  PERSIST_OPTIMIZER_STATE="1"
  LR="0.04"
  PPO_LOSS_WEIGHT="6.0"
  MIN_GRAD_NORM_FOR_STEP="0.0"
  OPD_LOSS_WEIGHT="0.12"
  OPD_PAIRWISE_LOSS_WEIGHT="0.06"
  OPD_PAIRWISE_MARGIN="0.0"
  OPD_POSITIVE_REWARD_THRESHOLD="1.0"
  MAX_OPD_PAIRWISE_PAIRS_PER_ROW="2"
  DYNAMIC_OPD_EXPERT_ROLLOUT="$DYNAMIC_OPD_EXPERT_ROLLOUT"
  DYNAMIC_OPD_TASKS="tool,memory,code"
  DYNAMIC_OPD_CURRENT_MAX_SUCCESS="0"
  DYNAMIC_OPD_POSITIVE_THRESHOLD="1.0"
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW="1"
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW="2"
  DYNAMIC_OPD_PER_TASK="32"
  MAX_NEW_TOKENS="1024"
  TOOL_MAX_NEW_TOKENS="768"
  MEMORY_UPDATE_MAX_NEW_TOKENS="1536"
  MEMORY_FINAL_MAX_NEW_TOKENS="768"
  CODE_MAX_NEW_TOKENS="2048"
  MAX_PROMPT_TOKENS="8192"
  MAX_MODEL_LEN="16384"
  MAX_LOGPROB_TOKENS="12288"
  ROLLOUT_SHARDS="auto"
  ROLLOUT_BATCH_SIZE="32"
  TENSOR_PARALLEL_SIZE="1"
  GPU_MEMORY_UTILIZATION="0.82"
  POST_BAKE_SLEEP_SECONDS="10"
  TEMPERATURE="0.7"
  TOP_P="0.95"
  GRADIENT_CHECKPOINTING="1"
  MAX_MEMORY_PER_GPU="70GiB"
  CPU_MAX_MEMORY="180GiB"
  PROGRESS_EVERY="10"
)

run_command() {
  local session="$1"
  local label="$2"
  local gpus="$3"
  local strategy="$4"
  local run_name="$5"
  local use_retention="$6"
  local prior="$7"
  local max_delta="$8"
  local run_dir="$ROOT/runs/gated_grpo/$run_name"
  mkdir -p "$run_dir"

  local retention_env=()
  if is_truthy "$use_retention"; then
    retention_env=(
      USE_RETENTION="1"
      RETENTION_OBJECTIVE="nll"
      RETENTION_LOSS_WEIGHT="0.05"
      RETENTION_POSITIVE_REWARD_THRESHOLD="1.0"
      MAX_RETENTION_ROWS_PER_TASK="8"
      MAX_RETENTION_ROWS="24"
    )
  else
    retention_env=(
      USE_RETENTION="0"
      RETENTION_OBJECTIVE="nll"
      RETENTION_LOSS_WEIGHT=""
      RETENTION_POSITIVE_REWARD_THRESHOLD="1.0"
      MAX_RETENTION_ROWS_PER_TASK=""
      MAX_RETENTION_ROWS=""
    )
  fi

  local full_cmd="cd '$REPO_ROOT' && env ${COMMON_ENV[*]} ${retention_env[*]} STRATEGY='$strategy' GPU_LIST='$gpus' ROLLOUT_GPUS='$gpus' RUN_NAME='$run_name' RUN_DIR='$run_dir' PRIOR_LOSS_WEIGHT='$prior' MAX_COEFF_DELTA='$max_delta' DRY_RUN='$DRY_RUN' bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee '$run_dir/run.log'"

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

A_RUN="paper96_A_gc_dynopd_nolen_noret_i${NUM_ITERS}_${RUN_TAG}"
B_RUN="paper96_B_gc_dynopd_nolen_ret_i${NUM_ITERS}_${RUN_TAG}"
C_RUN="paper96_C_gp_dynopd_nolen_noret_i${NUM_ITERS}_${RUN_TAG}"
D_RUN="paper96_D_gp_dynopd_nolen_ret_i${NUM_ITERS}_${RUN_TAG}"

run_command "paper96_A_gc_dynopd_nolen_${RUN_TAG}" "A global-coefficient dynamic-OPD no-retention" "0,1" "global-coefficient" "$A_RUN" "0" "0.0" "1.0"
run_command "paper96_B_gc_dynopd_nolen_${RUN_TAG}" "B global-coefficient dynamic-OPD retention" "2,3" "global-coefficient" "$B_RUN" "1" "0.005" "0.40"
run_command "paper96_C_gp_dynopd_nolen_${RUN_TAG}" "C global-parameter dynamic-OPD no-retention" "4,5" "global-parameter" "$C_RUN" "0" "0.0" "1.0"
run_command "paper96_D_gp_dynopd_nolen_${RUN_TAG}" "D global-parameter dynamic-OPD retention" "6,7" "global-parameter" "$D_RUN" "1" "0.005" "0.40"

if ! is_truthy "$DRY_RUN"; then
  MONITOR_SESSION="opvec_monitor_paper96_dynopd_nolen_${RUN_TAG}"
  if tmux has-session -t "$MONITOR_SESSION" 2>/dev/null; then
    echo "[monitor][skip] tmux session exists: $MONITOR_SESSION"
  else
    tmux new-session -d -s "$MONITOR_SESSION" \
      "cd '$REPO_ROOT' && '$PY' scripts/monitor/opvec_run_monitor.py --host 0.0.0.0 --port '$MONITOR_PORT' \
        --run-dir A_gc_dynopd_noret='$ROOT/runs/gated_grpo/$A_RUN' \
        --run-dir B_gc_dynopd_ret='$ROOT/runs/gated_grpo/$B_RUN' \
        --run-dir C_gp_dynopd_noret='$ROOT/runs/gated_grpo/$C_RUN' \
        --run-dir D_gp_dynopd_ret='$ROOT/runs/gated_grpo/$D_RUN' \
        --quiet"
    echo "[monitor] session=$MONITOR_SESSION url=http://127.0.0.1:$MONITOR_PORT"
  fi
fi

cat <<EOF
[paper96-dynopd-nolen] RUN_TAG=$RUN_TAG
[paper96-dynopd-nolen] monitor=http://127.0.0.1:$MONITOR_PORT
[paper96-dynopd-nolen] A=$ROOT/runs/gated_grpo/$A_RUN
[paper96-dynopd-nolen] B=$ROOT/runs/gated_grpo/$B_RUN
[paper96-dynopd-nolen] C=$ROOT/runs/gated_grpo/$C_RUN
[paper96-dynopd-nolen] D=$ROOT/runs/gated_grpo/$D_RUN
EOF
