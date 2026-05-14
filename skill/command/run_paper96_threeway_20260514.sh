#!/usr/bin/env bash
set -euo pipefail

# Three-way paper-run matrix for OP-VEC gated GRPO.
# A: global-coefficient + OPD + retention
# B: global-coefficient + retention, no OPD
# C: global-parameter + OPD + retention

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MONITOR_PORT="${MONITOR_PORT:-8768}"

CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
OPD_FIXED="${OPD_FIXED:-$ROOT/data/calibration/high_info_v1_seed20260511.distill_balanced21_paperfix_rewardtrain_len_seed20260514.jsonl}"

NUM_ITERS="${NUM_ITERS:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-96}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"

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
  FRONTIER_TOOL_QUOTA="32"
  FRONTIER_MEMORY_QUOTA="32"
  FRONTIER_CODE_QUOTA="32"
  MAX_FRONTIER_ROWS_PER_TASK="32"
  USE_RETENTION="1"
  RETENTION_LOSS_WEIGHT="0.03"
  MAX_RETENTION_ROWS_PER_TASK="8"
  MAX_RETENTION_ROWS="24"
  LENGTH_NORMALIZE_POLICY_LOGPROB="1"
  LENGTH_NORMALIZE_LOGPROB="1"
  OPTIMIZER="sgd"
  SGD_MOMENTUM="0.8"
  LR="0.04"
  PPO_LOSS_WEIGHT="6.0"
  PRIOR_LOSS_WEIGHT="0.005"
  MAX_COEFF_DELTA="0.40"
  MIN_GRAD_NORM_FOR_STEP="0.0"
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

launch_run() {
  local label="$1"
  local session="$2"
  local gpus="$3"
  local strategy="$4"
  local run_name="$5"
  local opd_mode="$6"
  local run_dir="$ROOT/runs/gated_grpo/$run_name"
  mkdir -p "$run_dir"

  local opd_env=()
  if [[ "$opd_mode" == "on" ]]; then
    opd_env=(
      OPD_DISTILL_ROLLOUT="$OPD_FIXED"
      OPD_LOSS_WEIGHT="0.12"
      OPD_PAIRWISE_LOSS_WEIGHT="0.06"
      OPD_PAIRWISE_MARGIN="0.0"
      OPD_POSITIVE_REWARD_THRESHOLD="1.0"
      MAX_OPD_DISTILL_ROWS="21"
      MAX_OPD_PAIRWISE_PAIRS_PER_ROW="2"
    )
  else
    opd_env=(
      OPD_LOSS_WEIGHT="0.0"
      OPD_PAIRWISE_LOSS_WEIGHT="0.0"
    )
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session exists: $session"
    return
  fi

  echo "[launch] $label session=$session gpus=$gpus run_dir=$run_dir"
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && env ${COMMON_ENV[*]} ${opd_env[*]} STRATEGY='$strategy' GPU_LIST='$gpus' ROLLOUT_GPUS='$gpus' RUN_NAME='$run_name' RUN_DIR='$run_dir' bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee '$run_dir/run.log'"
}

A_RUN="paper96_A_gc_opd_i${NUM_ITERS}_${RUN_TAG}"
B_RUN="paper96_B_gc_noopd_i${NUM_ITERS}_${RUN_TAG}"
C_RUN="paper96_C_gp_opd_i${NUM_ITERS}_${RUN_TAG}"

launch_run "A global-coefficient OPD retention" "paper96_A_gc_opd_${RUN_TAG}" "0,1" "global-coefficient" "$A_RUN" "on"
launch_run "B global-coefficient no-OPD retention" "paper96_B_gc_noopd_${RUN_TAG}" "4,5" "global-coefficient" "$B_RUN" "off"
launch_run "C global-parameter OPD retention" "paper96_C_gp_opd_${RUN_TAG}" "6,7" "global-parameter" "$C_RUN" "on"

if ! tmux has-session -t "opvec_monitor_paper96_${RUN_TAG}" 2>/dev/null; then
  tmux new-session -d -s "opvec_monitor_paper96_${RUN_TAG}" \
    "cd '$REPO_ROOT' && '$PY' scripts/monitor/opvec_run_monitor.py --host 0.0.0.0 --port '$MONITOR_PORT' --run-dir A_gc_opd='$ROOT/runs/gated_grpo/$A_RUN' --run-dir B_gc_noopd='$ROOT/runs/gated_grpo/$B_RUN' --run-dir C_gp_opd='$ROOT/runs/gated_grpo/$C_RUN' --quiet"
fi

cat <<EOF
[paper96] RUN_TAG=$RUN_TAG
[paper96] monitor=http://127.0.0.1:$MONITOR_PORT
[paper96] A=$ROOT/runs/gated_grpo/$A_RUN
[paper96] B=$ROOT/runs/gated_grpo/$B_RUN
[paper96] C=$ROOT/runs/gated_grpo/$C_RUN
EOF
