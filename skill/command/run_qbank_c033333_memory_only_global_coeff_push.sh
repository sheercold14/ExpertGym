#!/usr/bin/env bash
set -euo pipefail

# Isolated memory-only gate push from c=1/3 with exactly three direct global
# coefficients: tool, memory, code. No common+residual parameterization and no
# train-coefficient lock are used.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
QB="${QB:-$ROOT/data/question_bank/ta_avgvec_c033333_hotpotqa_v1}"
CALIBRATION="${CALIBRATION:-$QB/calibration/calib100_seed20260511.prompts.jsonl}"

STRATEGY="${STRATEGY:-global-coefficient}"
INIT_VALUE="${INIT_VALUE:-0.3333333333333333}"
RUN_NAME="${RUN_NAME:-qbank_c033333_global_coeff_memory_only_push_n33_lr008_gpu075_i1_seed20260513}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
INIT_GATE="${INIT_GATE:-$RUN_DIR/init_${STRATEGY}_c033333.json}"

GPU_LIST="${GPU_LIST:-4,5,6,7}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$GPU_LIST}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

NUM_ITERS="${NUM_ITERS:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-33}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
SEED_VALUE="${SEED_VALUE:-20260513}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MEMORY_UPDATE_MAX_NEW_TOKENS="${MEMORY_UPDATE_MAX_NEW_TOKENS:-2048}"
MEMORY_FINAL_MAX_NEW_TOKENS="${MEMORY_FINAL_MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-1024}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-24}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
ROLLOUT_SHARDS="${ROLLOUT_SHARDS:-auto}"
ROLLOUT_SHARD_STAGGER_SECONDS="${ROLLOUT_SHARD_STAGGER_SECONDS:-8}"
POST_BAKE_SLEEP_SECONDS="${POST_BAKE_SLEEP_SECONDS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

LR="${LR:-0.08}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.0}"
MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.6}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
UPDATE_BATCH_SIZE="${UPDATE_BATCH_SIZE:-4}"
BATCH_LOSS_REDUCTION="${BATCH_LOSS_REDUCTION:-mean}"
LOSS_GRANULARITY="${LOSS_GRANULARITY:-sequence}"
PPO_LOSS_WEIGHT="${PPO_LOSS_WEIGHT:-1.0}"
ADVANTAGE_NORMALIZATION="${ADVANTAGE_NORMALIZATION:-centered}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-0}"
USE_FRONTIER_WEIGHT="${USE_FRONTIER_WEIGHT:-0}"
FRONTIER_MEMORY_QUOTA="${FRONTIER_MEMORY_QUOTA:-100}"
MAX_FRONTIER_ROWS_PER_TASK="${MAX_FRONTIER_ROWS_PER_TASK:-100}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-1}"

DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-180GiB}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
DRY_RUN="${DRY_RUN:-0}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$RUN_DIR"
LOG_PATH="${LOG_PATH:-$RUN_DIR/train.log}"
if ! is_truthy "$DRY_RUN"; then
  exec > >(tee -a "$LOG_PATH") 2>&1
fi

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

GRADIENT_ARGS=()
if is_truthy "$GRADIENT_CHECKPOINTING"; then
  GRADIENT_ARGS+=(--gradient-checkpointing)
fi

OBJECTIVE_ARGS=(--advantage-normalization "$ADVANTAGE_NORMALIZATION")
if is_truthy "$LENGTH_NORMALIZE_POLICY_LOGPROB"; then
  OBJECTIVE_ARGS+=(--length-normalize-policy-logprob)
fi
if is_truthy "$TASK_NORMALIZE_ADVANTAGES"; then
  OBJECTIVE_ARGS+=(--task-normalize-advantages)
fi
if is_truthy "$USE_FRONTIER_WEIGHT"; then
  OBJECTIVE_ARGS+=(--use-frontier-weight)
fi

DRY_ARGS=()
if is_truthy "$DRY_RUN"; then
  DRY_ARGS+=(--dry-run)
fi

echo "[run] memory-only direct coefficient push from init=$INIT_VALUE"
echo "[run] run_dir=$RUN_DIR"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES rollout_gpus=$ROLLOUT_GPUS"
echo "[run] strategy=$STRATEGY num_iters=$NUM_ITERS num_prompts=$NUM_PROMPTS samples_per_prompt=$SAMPLES_PER_PROMPT"
echo "[run] lr=$LR prior=$PRIOR_LOSS_WEIGHT max_delta=$MAX_COEFF_DELTA"
echo "[run] loss_granularity=$LOSS_GRANULARITY update_batch_size=$UPDATE_BATCH_SIZE advantage=$ADVANTAGE_NORMALIZATION task_norm=$TASK_NORMALIZE_ADVANTAGES frontier_weight=$USE_FRONTIER_WEIGHT"

"$PY" scripts/modes/build_constant_gate_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --gate-parameterization "$STRATEGY" \
  --value "$INIT_VALUE" \
  --output "$INIT_GATE" >/dev/null

"$PY" scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --seed-manifest "$CALIBRATION" \
  --tasks memory \
  --run-dir "$RUN_DIR" \
  --run-id "$RUN_NAME" \
  --num-iters "$NUM_ITERS" \
  --num-prompts "$NUM_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --use-manifest-order \
  --gate-parameterization "$STRATEGY" \
  --init-gate-checkpoint "$INIT_GATE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --memory-update-max-new-tokens "$MEMORY_UPDATE_MAX_NEW_TOKENS" \
  --memory-final-max-new-tokens "$MEMORY_FINAL_MAX_NEW_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --max-logprob-tokens "$MAX_LOGPROB_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --vllm-batch-size "$ROLLOUT_BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --rollout-shards "$ROLLOUT_SHARDS" \
  --rollout-gpus "$ROLLOUT_GPUS" \
  --rollout-shard-stagger-seconds "$ROLLOUT_SHARD_STAGGER_SECONDS" \
  --post-bake-sleep-seconds "$POST_BAKE_SLEEP_SECONDS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "$SEED_VALUE" \
  --update-epochs "$UPDATE_EPOCHS" \
  --update-batch-size "$UPDATE_BATCH_SIZE" \
  --batch-loss-reduction "$BATCH_LOSS_REDUCTION" \
  --loss-granularity "$LOSS_GRANULARITY" \
  --frontier-order as-is \
  --ppo-loss-weight "$PPO_LOSS_WEIGHT" \
  --best-response-loss-weight 0.0 \
  --pairwise-loss-weight 0.0 \
  --pairwise-margin 0.0 \
  --max-pairwise-pairs-per-row 0 \
  --min-grad-norm-for-step 0.0 \
  --lr "$LR" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --max-coefficient-delta-from-init "$MAX_COEFF_DELTA" \
  --behavior-span-reward-weight 0.0 \
  --frontier-task-quota "memory=$FRONTIER_MEMORY_QUOTA" \
  --max-frontier-rows-per-task "$MAX_FRONTIER_ROWS_PER_TASK" \
  --task-weight memory=1.0 \
  --device "$DEVICE" \
  --device-map "$DEVICE_MAP" \
  --torch-dtype "$TORCH_DTYPE" \
  "${MAX_MEMORY_ARGS[@]}" \
  "${GRADIENT_ARGS[@]}" \
  "${OBJECTIVE_ARGS[@]}" \
  "${DRY_ARGS[@]}" \
  --progress-every "$PROGRESS_EVERY"

echo "[done] train log: $LOG_PATH"
