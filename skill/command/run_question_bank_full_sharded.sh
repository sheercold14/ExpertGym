#!/usr/bin/env bash
set -euo pipefail

# Sharded vLLM rollout pipeline for the HotpotQA question-bank version.
# It assumes PHASE=prepare and PHASE=build-model have already been run via
# run_question_bank_ta_avgvec_c033333.sh.
#
# Phases:
#   PHASE=rollout-all     run tool, memory, and code rollout shards
#   PHASE=rollout-tool    run only tool rollout shards
#   PHASE=rollout-memory  run only memory rollout shards
#   PHASE=rollout-code    run only code rollout shards
#   PHASE=bank            aggregate all full shard rollouts into question_bank.jsonl
#   PHASE=sample          sample calib100 + guard from question_bank.jsonl
#   PHASE=post            run bank + sample
#
# Set DRY_RUN=1 to print commands without executing them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PHASE="${PHASE:-rollout-all}"
DRY_RUN="${DRY_RUN:-0}"

QB_VERSION="${QB_VERSION:-ta_avgvec_c033333_hotpotqa_v1}"
QB="${QB:-$ROOT/data/question_bank/$QB_VERSION}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/question_bank/$QB_VERSION}"
ROLL_OUT_DIR="$QB/rollouts/full"
CALIB_DIR="$QB/calibration"
LOG_DIR="$RUN_ROOT/logs/full_sharded"
COMMANDS_FILE="$QB/commands.sh"

TA_NAME="${TA_NAME:-ta_avgvec_c033333}"
POLICY="${POLICY:-/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/$TA_NAME/model}"
SEED_MANIFEST="${SEED_MANIFEST:-$QB/source_reward_hybrid_tool_code_hotpotqa_memory_traj_seed20260511.jsonl}"
QUESTION_BANK="${QUESTION_BANK:-$QB/question_bank.jsonl}"
CALIB_PREFIX="${CALIB_PREFIX:-$CALIB_DIR/calib100_seed20260511}"

ROLLOUT_SEED="${ROLLOUT_SEED:-20260511}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MEMORY_UPDATE_MAX_NEW_TOKENS="${MEMORY_UPDATE_MAX_NEW_TOKENS:-1024}"
MEMORY_FINAL_MAX_NEW_TOKENS="${MEMORY_FINAL_MAX_NEW_TOKENS:-1024}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
TOOL_TOTAL="${TOOL_TOTAL:-500}"
MEMORY_TOTAL="${MEMORY_TOTAL:-500}"
CODE_TOTAL="${CODE_TOTAL:-500}"
TOOL_SHARD_SIZE="${TOOL_SHARD_SIZE:-100}"
MEMORY_SHARD_SIZE="${MEMORY_SHARD_SIZE:-100}"
CODE_SHARD_SIZE="${CODE_SHARD_SIZE:-100}"
TOOL_MAX_NEW_TOKENS="${TOOL_MAX_NEW_TOKENS:-1024}"
MEMORY_MAX_NEW_TOKENS="${MEMORY_MAX_NEW_TOKENS:-2048}"
CODE_MAX_NEW_TOKENS="${CODE_MAX_NEW_TOKENS:-1024}"

TOOL_QUOTA="${TOOL_QUOTA:-34}"
MEMORY_QUOTA="${MEMORY_QUOTA:-33}"
CODE_QUOTA="${CODE_QUOTA:-33}"
GUARD_PER_TASK="${GUARD_PER_TASK:-3}"

setup_dirs() {
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$ROLL_OUT_DIR" "$CALIB_DIR" "$LOG_DIR"
    if [[ ! -f "$COMMANDS_FILE" ]]; then
      {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        echo "# Commands recorded by $(basename "$0")"
        echo
      } > "$COMMANDS_FILE"
      chmod +x "$COMMANDS_FILE"
    fi
  fi
}

print_cmd() {
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
}

record_cmd() {
  if [[ "$DRY_RUN" != "1" ]]; then
    {
      echo
      echo "# phase=$PHASE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf '%q ' "$@"
      printf '\n'
    } >> "$COMMANDS_FILE"
  fi
}

run_cmd() {
  print_cmd "$@"
  record_cmd "$@"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

rollout_one_shard() {
  local task="$1"
  local offset="$2"
  local count="$3"
  local gpu="$4"
  local max_new_tokens="$5"
  local output="$ROLL_OUT_DIR/${task}_${TA_NAME}_n${SAMPLES_PER_PROMPT}_offset$(printf '%03d' "$offset")_len$(printf '%03d' "$count").jsonl"
  local log="$LOG_DIR/${task}_offset$(printf '%03d' "$offset")_len$(printf '%03d' "$count").log"
  if [[ -s "$output" ]]; then
    local existing_rows
    existing_rows="$(wc -l < "$output")"
    if (( existing_rows >= count )); then
      echo "[skip] shard complete: $output rows=$existing_rows"
      return 0
    fi
    echo "[rerun] incomplete shard: $output rows=$existing_rows expected=$count"
  fi
  local cmd=(
    env CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" scripts/train/opvec_collect_vllm_rollouts.py
    --config configs/gated_grpo.yaml
    --mode-manifest "$ROOT/modes/opvec4/mode_manifest.json"
    --policy-model "$POLICY"
    --policy-id "$TA_NAME"
    --no-gate-values
    --seed-manifest "$SEED_MANIFEST"
    --output "$output"
    --run-id "qb-${QB_VERSION}-${TA_NAME}-${task}-n${SAMPLES_PER_PROMPT}-offset${offset}"
    --tasks "$task"
    --num-prompts "$count"
    --prompt-offset "$offset"
    --use-manifest-order
    --samples-per-prompt "$SAMPLES_PER_PROMPT"
    --max-new-tokens "$max_new_tokens"
    --memory-update-max-new-tokens "$MEMORY_UPDATE_MAX_NEW_TOKENS"
    --memory-final-max-new-tokens "$MEMORY_FINAL_MAX_NEW_TOKENS"
    --max-prompt-tokens "$MAX_PROMPT_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --vllm-batch-size "$VLLM_BATCH_SIZE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --seed "$ROLLOUT_SEED"
    --stream-output
    --progress-every 25
    --behavior-span-reward-weight 0.0
  )
  print_cmd "${cmd[@]}"
  record_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}" > "$log" 2>&1
  fi
}

run_task_shards() {
  local task="$1"
  local total="$2"
  local shard_size="$3"
  local max_new_tokens="$4"
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "GPU_LIST is empty" >&2
    exit 2
  fi
  local -a pids=()
  local index=0
  for ((offset=0; offset<total; offset+=shard_size)); do
    local count="$shard_size"
    if (( offset + count > total )); then
      count=$((total - offset))
    fi
    local gpu="${gpus[$((index % ${#gpus[@]}))]}"
    if [[ "$DRY_RUN" == "1" ]]; then
      rollout_one_shard "$task" "$offset" "$count" "$gpu" "$max_new_tokens"
    else
      rollout_one_shard "$task" "$offset" "$count" "$gpu" "$max_new_tokens" &
      pids+=("$!")
    fi
    index=$((index + 1))
    if [[ "$DRY_RUN" != "1" && "${#pids[@]}" -ge "${#gpus[@]}" ]]; then
      wait_batch pids
      pids=()
    fi
  done
  if [[ "$DRY_RUN" != "1" && "${#pids[@]}" -gt 0 ]]; then
    wait_batch pids
  fi
}

wait_batch() {
  local -n batch_pids="$1"
  local failed=0
  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "At least one rollout shard failed. Check logs in $LOG_DIR" >&2
    exit 1
  fi
}

build_question_bank() {
  local -a cmd=("$PY" scripts/data/build_question_bank_from_rollouts.py)
  local found=0
  for path in "$ROLL_OUT_DIR"/*_"$TA_NAME"_n"$SAMPLES_PER_PROMPT"_offset*_len*.jsonl; do
    if [[ -s "$path" ]]; then
      cmd+=(--rollouts "$path")
      found=1
    fi
  done
  if [[ "$found" == "0" ]]; then
    echo "No full shard rollouts found in $ROLL_OUT_DIR" >&2
    exit 1
  fi
  cmd+=(
    --seed-manifest "$SEED_MANIFEST"
    --output "$QUESTION_BANK"
    --summary "$QB/question_bank.summary.json"
  )
  run_cmd "${cmd[@]}"
}

sample_calibration() {
  run_cmd "$PY" scripts/data/sample_question_bank.py \
    --question-bank "$QUESTION_BANK" \
    --output-prefix "$CALIB_PREFIX" \
    --quota "tool=$TOOL_QUOTA" \
    --quota "memory=$MEMORY_QUOTA" \
    --quota "code=$CODE_QUOTA" \
    --guard-per-task "$GUARD_PER_TASK" \
    --seed "$ROLLOUT_SEED"
}

setup_dirs

case "$PHASE" in
  rollout-tool)
    run_task_shards tool "$TOOL_TOTAL" "$TOOL_SHARD_SIZE" "$TOOL_MAX_NEW_TOKENS"
    ;;
  rollout-memory)
    run_task_shards memory "$MEMORY_TOTAL" "$MEMORY_SHARD_SIZE" "$MEMORY_MAX_NEW_TOKENS"
    ;;
  rollout-code)
    run_task_shards code "$CODE_TOTAL" "$CODE_SHARD_SIZE" "$CODE_MAX_NEW_TOKENS"
    ;;
  rollout-all)
    run_task_shards tool "$TOOL_TOTAL" "$TOOL_SHARD_SIZE" "$TOOL_MAX_NEW_TOKENS"
    run_task_shards memory "$MEMORY_TOTAL" "$MEMORY_SHARD_SIZE" "$MEMORY_MAX_NEW_TOKENS"
    run_task_shards code "$CODE_TOTAL" "$CODE_SHARD_SIZE" "$CODE_MAX_NEW_TOKENS"
    ;;
  bank)
    build_question_bank
    ;;
  sample)
    sample_calibration
    ;;
  post)
    build_question_bank
    sample_calibration
    ;;
  *)
    echo "Unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac

echo "[done] phase=$PHASE qb=$QB"
