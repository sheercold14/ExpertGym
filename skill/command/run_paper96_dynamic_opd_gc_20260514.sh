#!/usr/bin/env bash
set -euo pipefail

# Paper96-D: global-coefficient + dynamic OPD.
# For each iteration, policy all-failure prompts are matched against offline
# same-prompt expert trajectories and only those matched rows receive OPD loss.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
RUN_TAG="${RUN_TAG:-20260514_paper96_dynopd_i8}"
BASE_RUN_TAG="${BASE_RUN_TAG:-20260514_paper96_i8}"
MONITOR_PORT="${MONITOR_PORT:-8769}"
SESSION="${SESSION:-paper96_D_gc_dynopd_${RUN_TAG}}"

CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
TOOL_EXPERT_MODEL="${TOOL_EXPERT_MODEL:-/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold}"
MEMORY_EXPERT_MODEL="${MEMORY_EXPERT_MODEL:-/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B}"
CODE_EXPERT_MODEL="${CODE_EXPERT_MODEL:-/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B}"

NUM_ITERS="${NUM_ITERS:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-96}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
EXPERT_SAMPLES_PER_PROMPT="${EXPERT_SAMPLES_PER_PROMPT:-2}"

TOOL_EXPERT_ROLLOUT="$EXPERT_DIR/tool_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl"
MEMORY_EXPERT_ROLLOUT="$EXPERT_DIR/memory_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl"
CODE_EXPERT_ROLLOUT="$EXPERT_DIR/code_expert_paper96_s${EXPERT_SAMPLES_PER_PROMPT}_seed20260514.jsonl"
DYNAMIC_OPD_EXPERT_ROLLOUT="$TOOL_EXPERT_ROLLOUT,$MEMORY_EXPERT_ROLLOUT,$CODE_EXPERT_ROLLOUT"

RUN_NAME="paper96_D_gc_dynamic_opd_i${NUM_ITERS}_${RUN_TAG}"
RUN_DIR="$ROOT/runs/gated_grpo/$RUN_NAME"

is_complete_jsonl() {
  local path="$1"
  local min_rows="$2"
  [[ -s "$path" ]] && [[ "$(wc -l < "$path")" -ge "$min_rows" ]]
}

collect_expert_rollout() {
  local task="$1"
  local gpu="$2"
  local model="$3"
  local output="$4"
  local seed="$5"

  if is_complete_jsonl "$output" 32; then
    echo "[expert-rollout][skip] task=$task output=$output"
    return
  fi

  mkdir -p "$(dirname "$output")"
  echo "[expert-rollout] task=$task gpu=$gpu model=$model output=$output"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train/opvec_collect_vllm_rollouts.py \
    --config configs/gated_grpo.yaml \
    --mode-manifest "$ROOT/modes/opvec4/mode_manifest.json" \
    --seed-manifest "$CALIBRATION" \
    --policy-model "$model" \
    --policy-id "${task}_expert_paper96_offline" \
    --output "$output" \
    --run-id "${task}-expert-paper96-offline" \
    --num-prompts "$NUM_PROMPTS" \
    --samples-per-prompt "$EXPERT_SAMPLES_PER_PROMPT" \
    --tasks "$task" \
    --use-manifest-order \
    --no-gate-values \
    --max-new-tokens 1024 \
    --tool-max-new-tokens 768 \
    --memory-update-max-new-tokens 1536 \
    --memory-final-max-new-tokens 768 \
    --code-max-new-tokens 2048 \
    --max-prompt-tokens 8192 \
    --max-model-len 16384 \
    --vllm-batch-size 32 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.82 \
    --dtype bfloat16 \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed "$seed" \
    --stream-output \
    --progress-every 10
}

run_worker() {
  mkdir -p "$RUN_DIR" "$EXPERT_DIR"
  echo "[paper96-D] run_dir=$RUN_DIR"
  echo "[paper96-D] calibration=$CALIBRATION"
  echo "[paper96-D] expert_dir=$EXPERT_DIR"

  collect_expert_rollout tool 2 "$TOOL_EXPERT_MODEL" "$TOOL_EXPERT_ROLLOUT" 2026051401 &
  tool_pid=$!
  collect_expert_rollout memory 3 "$MEMORY_EXPERT_MODEL" "$MEMORY_EXPERT_ROLLOUT" 2026051402 &
  memory_pid=$!
  wait "$tool_pid"
  wait "$memory_pid"
  collect_expert_rollout code 2 "$CODE_EXPERT_MODEL" "$CODE_EXPERT_ROLLOUT" 2026051403

  echo "[paper96-D] dynamic_opd_expert_rollout=$DYNAMIC_OPD_EXPERT_ROLLOUT"
  env \
    ROOT="$ROOT" \
    CALIBRATION="$CALIBRATION" \
    NUM_ITERS="$NUM_ITERS" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    SAMPLES_PER_PROMPT="$SAMPLES_PER_PROMPT" \
    INIT_VALUE="0.3333333333333333" \
    STRATEGY="global-coefficient" \
    GPU_LIST="2,3" \
    ROLLOUT_GPUS="2,3" \
    RUN_NAME="$RUN_NAME" \
    RUN_DIR="$RUN_DIR" \
    UPDATE_EPOCHS="1" \
    UPDATE_BATCH_SIZE="4" \
    BATCH_LOSS_REDUCTION="mean" \
    OPTIMIZER_STEP_SCOPE="epoch" \
    LOSS_GRANULARITY="sequence" \
    STORE_TOKEN_LOGPROBS="0" \
    TASK_NORMALIZE_ADVANTAGES="0" \
    ADVANTAGE_NORMALIZATION="centered" \
    USE_FRONTIER_WEIGHT="0" \
    FRONTIER_ORDER="task-interleaved" \
    FRONTIER_TOOL_QUOTA="32" \
    FRONTIER_MEMORY_QUOTA="32" \
    FRONTIER_CODE_QUOTA="32" \
    MAX_FRONTIER_ROWS_PER_TASK="32" \
    USE_RETENTION="1" \
    RETENTION_LOSS_WEIGHT="0.03" \
    MAX_RETENTION_ROWS_PER_TASK="8" \
    MAX_RETENTION_ROWS="24" \
    LENGTH_NORMALIZE_POLICY_LOGPROB="1" \
    LENGTH_NORMALIZE_LOGPROB="1" \
    OPTIMIZER="sgd" \
    SGD_MOMENTUM="0.8" \
    LR="0.04" \
    PPO_LOSS_WEIGHT="6.0" \
    PRIOR_LOSS_WEIGHT="0.005" \
    MAX_COEFF_DELTA="0.40" \
    MIN_GRAD_NORM_FOR_STEP="0.0" \
    OPD_LOSS_WEIGHT="0.12" \
    OPD_PAIRWISE_LOSS_WEIGHT="0.06" \
    OPD_PAIRWISE_MARGIN="0.0" \
    OPD_POSITIVE_REWARD_THRESHOLD="1.0" \
    MAX_OPD_PAIRWISE_PAIRS_PER_ROW="2" \
    DYNAMIC_OPD_EXPERT_ROLLOUT="$DYNAMIC_OPD_EXPERT_ROLLOUT" \
    DYNAMIC_OPD_TASKS="tool,memory,code" \
    DYNAMIC_OPD_CURRENT_MAX_SUCCESS="0" \
    DYNAMIC_OPD_POSITIVE_THRESHOLD="1.0" \
    DYNAMIC_OPD_MAX_POSITIVES_PER_ROW="1" \
    DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW="2" \
    DYNAMIC_OPD_PER_TASK="32" \
    MAX_NEW_TOKENS="1024" \
    TOOL_MAX_NEW_TOKENS="768" \
    MEMORY_UPDATE_MAX_NEW_TOKENS="1536" \
    MEMORY_FINAL_MAX_NEW_TOKENS="768" \
    CODE_MAX_NEW_TOKENS="2048" \
    MAX_PROMPT_TOKENS="8192" \
    MAX_MODEL_LEN="16384" \
    MAX_LOGPROB_TOKENS="12288" \
    ROLLOUT_SHARDS="auto" \
    ROLLOUT_BATCH_SIZE="32" \
    TENSOR_PARALLEL_SIZE="1" \
    GPU_MEMORY_UTILIZATION="0.82" \
    POST_BAKE_SLEEP_SECONDS="10" \
    TEMPERATURE="0.7" \
    TOP_P="0.95" \
    GRADIENT_CHECKPOINTING="1" \
    MAX_MEMORY_PER_GPU="70GiB" \
    CPU_MAX_MEMORY="180GiB" \
    PROGRESS_EVERY="10" \
    bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee "$RUN_DIR/run.log"
}

start_monitor() {
  local monitor_session="opvec_monitor_paper96_dynamic_${RUN_TAG}"
  if tmux has-session -t "$monitor_session" 2>/dev/null; then
    echo "[monitor][skip] tmux session exists: $monitor_session"
    return
  fi

  tmux new-session -d -s "$monitor_session" \
    "cd '$REPO_ROOT' && '$PY' scripts/monitor/opvec_run_monitor.py --host 0.0.0.0 --port '$MONITOR_PORT' \
      --run-dir A_gc_opd='$ROOT/runs/gated_grpo/paper96_A_gc_opd_i${NUM_ITERS}_${BASE_RUN_TAG}' \
      --run-dir B_gc_noopd='$ROOT/runs/gated_grpo/paper96_B_gc_noopd_i${NUM_ITERS}_${BASE_RUN_TAG}' \
      --run-dir C_gp_opd='$ROOT/runs/gated_grpo/paper96_C_gp_opd_i${NUM_ITERS}_${BASE_RUN_TAG}' \
      --run-dir D_gc_dynamic_opd='$RUN_DIR' \
      --quiet"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[skip] tmux session exists: $SESSION"
else
  tmux new-session -d -s "$SESSION" \
    "cd '$REPO_ROOT' && RUN_TAG='$RUN_TAG' BASE_RUN_TAG='$BASE_RUN_TAG' MONITOR_PORT='$MONITOR_PORT' bash skill/command/run_paper96_dynamic_opd_gc_20260514.sh --worker"
  echo "[launch] session=$SESSION gpus=2,3 run_dir=$RUN_DIR"
fi

start_monitor

cat <<EOF
[paper96-D] RUN_TAG=$RUN_TAG
[paper96-D] monitor=http://127.0.0.1:$MONITOR_PORT
[paper96-D] run_dir=$RUN_DIR
[paper96-D] tmux attach -t $SESSION
EOF
