#!/usr/bin/env bash
set -euo pipefail

# Qwen AdaMerging wrapper. The implementation lives in the existing local
# ExpertGym worktree where the Qwen AdaMerging adapter was developed.

SOURCE_ROOT="${SOURCE_ROOT:-/mnt/cache/wuruixiao/users/lsc/era-2026/ExpertGym/worktrees/worktree}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
GPU="${GPU:-0}"
METHOD="${METHOD:-task_wise_adamerging}"
RUN_NAME="${RUN_NAME:-qwen_${METHOD}_20260518}"
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging}"
DATA_DIR="${DATA_DIR:-/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1_merge}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_BATCHES_PER_EPOCH="${MAX_BATCHES_PER_EPOCH:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LR="${LR:-0.001}"
INIT_LAMBDA="${INIT_LAMBDA:-0.3}"
TIES_KEEP_RATIO="${TIES_KEEP_RATIO:-0.2}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "$PY" Qwen/model_merging.py
  --method "$METHOD"
  --base_model /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct
  --expert_models
    /mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold
    /mnt/cache/wuruixiao/models/RL-MemoryAgent-7B
    /mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B
  --data_dir "$DATA_DIR"
  --output_path "$OUTPUT_PATH"
  --run_name "$RUN_NAME"
  --num_epochs "$NUM_EPOCHS"
  --learning_rate "$LR"
  --batch_size "$BATCH_SIZE"
  --max_batches_per_epoch "$MAX_BATCHES_PER_EPOCH"
  --max_length "$MAX_LENGTH"
  --samples_per_task 15
  --weight_coeffs_init_value "$INIT_LAMBDA"
  --ties_keep_ratio "$TIES_KEEP_RATIO"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q' "$SOURCE_ROOT" "$GPU"
  printf ' %q' "${cmd[@]}"
  printf '\n'
else
  cd "$SOURCE_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
fi
