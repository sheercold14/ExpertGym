#!/usr/bin/env bash
set -euo pipefail

# c1: B-style paper96 training from init=1.0 with per-layer layer-band gates.
# This uses configs/gated_grpo_layer28.yaml so layer-band means one band per
# transformer layer instead of the default early/mid/late bands.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_NAME="${RUN_NAME:-c1_init1_layerband_grpo_opd_ret_20260517}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
GPU_LIST="${GPU_LIST:-2,3}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$GPU_LIST}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
CONFIG="${CONFIG:-configs/gated_grpo_layer28.yaml}"
MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
INIT_GATE_CHECKPOINT="${INIT_GATE_CHECKPOINT:-$ROOT/data/init_gates/c1_20260517/init_layer_band_layer28_init1.json}"

if [[ "$DRY_RUN" != "1" && -d "$RUN_DIR" ]]; then
  mapfile -t EXISTING_RUN_FILES < <(
    find "$RUN_DIR" -mindepth 1 -maxdepth 1 \
      ! -name train.log \
      ! -name monitor_8795.log \
      -print 2>/dev/null
  )
  if (( ${#EXISTING_RUN_FILES[@]} > 0 )); then
    if [[ "$OVERWRITE" != "1" ]]; then
      echo "[error] RUN_DIR already exists and is not empty: $RUN_DIR" >&2
      echo "[error] set OVERWRITE=1 only if you intentionally want to remove it" >&2
      exit 2
    fi
    for path in "${EXISTING_RUN_FILES[@]}"; do
      rm -rf "$path"
    done
  fi
fi

mkdir -p "$RUN_DIR" "$(dirname "$INIT_GATE_CHECKPOINT")"

CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
TOOL_EXPERT="$EXPERT_DIR/tool_expert_paper96_s2_seed20260514.jsonl"
MEMORY_EXPERT="$EXPERT_DIR/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT="$EXPERT_DIR/code_expert_paper96_s2_seed20260514.jsonl"
EXPERT_ROLLOUTS="${EXPERT_ROLLOUTS:-$TOOL_EXPERT,$MEMORY_EXPERT,$CODE_EXPERT}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

for path in "$CONFIG" "$MODE" "$CALIBRATION" "$TOOL_EXPERT" "$MEMORY_EXPERT" "$CODE_EXPERT"; do
  require_file "$path"
done

"$PY" scripts/modes/build_constant_gate_checkpoint.py \
  --config "$CONFIG" \
  --mode-manifest "$MODE" \
  --gate-parameterization layer-band \
  --value 1.0 \
  --output "$INIT_GATE_CHECKPOINT" >/dev/null

env \
  DRY_RUN="$DRY_RUN" \
  GPU_LIST="$GPU_LIST" \
  ROLLOUT_GPUS="$ROLLOUT_GPUS" \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  CONFIG="$CONFIG" \
  MODE="$MODE" \
  STRATEGY=layer-band \
  INIT_VALUE=1.0 \
  INIT_GATE_CHECKPOINT="$INIT_GATE_CHECKPOINT" \
  CALIBRATION="$CALIBRATION" \
  NUM_ITERS="${NUM_ITERS:-20}" \
  NUM_PROMPTS=96 \
  SAMPLES_PER_PROMPT=4 \
  STORE_TOKEN_LOGPROBS=0 \
  OPTIMIZER=sgd \
  SGD_MOMENTUM=0.2 \
  PERSIST_OPTIMIZER_STATE=1 \
  LR="${LR:-0.1876}" \
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
  RETENTION_LOSS_WEIGHT=0.5 \
  RETENTION_POSITIVE_REWARD_THRESHOLD=1.0 \
  RETENTION_TASK_BALANCED_LOSS_SCALE=1 \
  RETENTION_SCALE_TARGET=0.5 \
  OPD_LOSS_WEIGHT=1.0 \
  OPD_PAIRWISE_LOSS_WEIGHT=0.0 \
  OPD_POSITIVE_REWARD_THRESHOLD=1.0 \
  OPD_LENGTH_NORMALIZE_LOGPROB=1 \
  RETENTION_LENGTH_NORMALIZE_LOGPROB=1 \
  OPD_TASK_BALANCED_LOSS_SCALE=1 \
  LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
  LENGTH_NORMALIZE_LOGPROB=0 \
  TASK_NORMALIZE_ADVANTAGES=0 \
  ADVANTAGE_NORMALIZATION=centered \
  USE_FRONTIER_WEIGHT=0 \
  PPO_LOSS_WEIGHT=1.0 \
  BEST_RESPONSE_LOSS_WEIGHT=0.0 \
  PAIRWISE_LOSS_WEIGHT=0.0 \
  DYNAMIC_OPD_EXPERT_ROLLOUT="$EXPERT_ROLLOUTS" \
  DYNAMIC_OPD_TASKS=tool,memory,code \
  DYNAMIC_OPD_KEY=prompt_id \
  DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0 \
  DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0 \
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1 \
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2 \
  DYNAMIC_OPD_PER_TASK=32 \
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
  SEED_VALUE=20260517 \
  PROGRESS_EVERY=10 \
  bash skill/command/run_qbank_c033333_gate_strategy.sh
