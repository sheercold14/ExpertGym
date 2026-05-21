#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
CONFIG="${CONFIG:-configs/gated_grpo_layer28.yaml}"
MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"

EXP_ID="${EXP_ID:?Set EXP_ID, e.g. r1_e0_anchor}"
CALIB="${CALIB:?Set CALIB to TRC trajectory JSONL}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/trc/$EXP_ID}"
GPU_LIST="${GPU_LIST:-0,1}"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_MEMORY_ENTRIES="${MAX_MEMORY_ENTRIES:-0=70GiB 1=70GiB}"

MAX_MEMORY_ARGS=()
if [[ -n "$MAX_MEMORY_ENTRIES" ]]; then
  for item in $MAX_MEMORY_ENTRIES; do
    MAX_MEMORY_ARGS+=(--max-memory "$item")
  done
fi

append_repeated_args() {
  local flag="$1"
  local values="$2"
  local -n target="$3"
  if [[ -n "$values" ]]; then
    for item in $values; do
      target+=("$flag" "$item")
    done
  fi
}

TASK_ARGS=()
append_repeated_args "--task-hidden-layers" "${TASK_HIDDEN_LAYERS:-}" TASK_ARGS
append_repeated_args "--task-topk-tokens" "${TASK_TOPK_TOKENS:-}" TASK_ARGS
append_repeated_args "--task-residual-weight-power" "${TASK_RESIDUAL_WEIGHT_POWER:-}" TASK_ARGS
append_repeated_args "--task-directional-projection-floor" "${TASK_DIRECTIONAL_PROJECTION_FLOOR:-}" TASK_ARGS
append_repeated_args "--task-directional-projection-weight" "${TASK_DIRECTIONAL_PROJECTION_WEIGHT:-}" TASK_ARGS
append_repeated_args "--task-response-span-mode" "${TASK_RESPONSE_SPAN_MODE:-}" TASK_ARGS
append_repeated_args "--task-loss-multiplier" "${TASK_LOSS_MULTIPLIER:-}" TASK_ARGS
append_repeated_args "--task-residual-target-coefficients" "${TASK_RESIDUAL_TARGET_COEFFICIENTS:-}" TASK_ARGS
append_repeated_args "--task-response-nll-weight" "${TASK_RESPONSE_NLL_WEIGHT:-}" TASK_ARGS
append_repeated_args "--trajectory-turn-loss-task" "${TRAJECTORY_TURN_LOSS_TASKS:-}" TASK_ARGS
append_repeated_args "--contrastive-negative-task" "${CONTRASTIVE_NEGATIVE_TASKS:-}" TASK_ARGS

cd "$REPO"
mkdir -p "$RUN_DIR"

{
  echo "EXP_ID=$EXP_ID"
  echo "GPU_LIST=$GPU_LIST"
  echo "CALIB=$CALIB"
  echo "RUN_DIR=$RUN_DIR"
  echo "CONFIG=$CONFIG"
  echo "MODE=$MODE"
  echo "EPOCHS=${EPOCHS:-12}"
  echo "TRAIN_TASKS=${TRAIN_TASKS:-}"
  echo "TRAINABLE_EXPERTS=${TRAINABLE_EXPERTS:-}"
  echo "MAX_ROWS_PER_TASK=${MAX_ROWS_PER_TASK:-0}"
  echo "LR=${LR:-0.02}"
  echo "BETA_BASE=${BETA_BASE:-0.05}"
  echo "PROMPT_DRIFT_TOKENS=${PROMPT_DRIFT_TOKENS:-256}"
  echo "RESPONSE_RESIDUAL_WEIGHT=${RESPONSE_RESIDUAL_WEIGHT:-1.0}"
  echo "PROMPT_RESIDUAL_WEIGHT=${PROMPT_RESIDUAL_WEIGHT:-0.0}"
  echo "PROMPT_RESIDUAL_TOKENS=${PROMPT_RESIDUAL_TOKENS:-256}"
  echo "RESPONSE_NLL_WEIGHT=${RESPONSE_NLL_WEIGHT:-0.0}"
  echo "TASK_RESPONSE_NLL_WEIGHT=${TASK_RESPONSE_NLL_WEIGHT:-}"
  echo "GAMMA_GATE=${GAMMA_GATE:-0.005}"
  echo "COEFFICIENT_FLOOR=${COEFFICIENT_FLOOR:-0.95}"
  echo "COEFFICIENT_FLOOR_WEIGHT=${COEFFICIENT_FLOOR_WEIGHT:-0.1}"
  echo "TASK_EXPERT_COEFFICIENT_FLOOR=${TASK_EXPERT_COEFFICIENT_FLOOR:-0.0}"
  echo "TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=${TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT:-0.0}"
  echo "TASK_HIDDEN_LAYERS=${TASK_HIDDEN_LAYERS:-}"
  echo "TASK_TOPK_TOKENS=${TASK_TOPK_TOKENS:-}"
  echo "TASK_RESIDUAL_WEIGHT_POWER=${TASK_RESIDUAL_WEIGHT_POWER:-}"
  echo "TASK_DIRECTIONAL_PROJECTION_FLOOR=${TASK_DIRECTIONAL_PROJECTION_FLOOR:-}"
  echo "TASK_DIRECTIONAL_PROJECTION_WEIGHT=${TASK_DIRECTIONAL_PROJECTION_WEIGHT:-}"
  echo "TASK_RESPONSE_SPAN_MODE=${TASK_RESPONSE_SPAN_MODE:-}"
  echo "TASK_LOSS_MULTIPLIER=${TASK_LOSS_MULTIPLIER:-}"
  echo "RESIDUAL_TARGET_SOURCE=${RESIDUAL_TARGET_SOURCE:-row-expert}"
  echo "RESIDUAL_TARGET_COEFFICIENTS=${RESIDUAL_TARGET_COEFFICIENTS:-}"
  echo "TASK_RESIDUAL_TARGET_COEFFICIENTS=${TASK_RESIDUAL_TARGET_COEFFICIENTS:-}"
  echo "RESIDUAL_TARGET_GATE_CHECKPOINT=${RESIDUAL_TARGET_GATE_CHECKPOINT:-}"
  echo "TRAJECTORY_TURN_LOSS_TASKS=${TRAJECTORY_TURN_LOSS_TASKS:-}"
  echo "CONTRASTIVE_NEGATIVE_LOSS_WEIGHT=${CONTRASTIVE_NEGATIVE_LOSS_WEIGHT:-0.0}"
  echo "CONTRASTIVE_NEGATIVE_MARGIN=${CONTRASTIVE_NEGATIVE_MARGIN:-0.05}"
  echo "CONTRASTIVE_NEGATIVE_RESPONSE_KEY=${CONTRASTIVE_NEGATIVE_RESPONSE_KEY:-negative_response}"
  echo "CONTRASTIVE_NEGATIVE_TASKS=${CONTRASTIVE_NEGATIVE_TASKS:-}"
  echo "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}"
  echo "PROGRESS_EVERY_ROWS=${PROGRESS_EVERY_ROWS:-0}"
  echo "LOG_GATE_GRADIENTS=${LOG_GATE_GRADIENTS:-0}"
} > "$RUN_DIR/run.env"

EXTRA_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi
if [[ "${LOG_GATE_GRADIENTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-gate-gradients)
fi

"$PY" scripts/trc/train_trc_layer_gates.py \
  --config "$CONFIG" \
  --mode-manifest "$MODE" \
  --calibration "$CALIB" \
  --output-dir "$RUN_DIR" \
  --gate-parameterization layer-band-coefficient \
  --init-value "${INIT_VALUE:-1.0}" \
  --epochs "${EPOCHS:-12}" \
  --train-tasks "${TRAIN_TASKS:-}" \
  --trainable-experts "${TRAINABLE_EXPERTS:-}" \
  --max-rows-per-task "${MAX_ROWS_PER_TASK:-0}" \
  --shuffle \
  --seed "${SEED:-20260519}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --lr "${LR:-0.02}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-96}" \
  --progress-every-rows "${PROGRESS_EVERY_ROWS:-0}" \
  --hidden-layers "${HIDDEN_LAYERS:-8,16,24,28}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-1536}" \
  --max-response-tokens "${MAX_RESPONSE_TOKENS:-512}" \
  --topk-tokens "${TOPK_TOKENS:-128}" \
  --prompt-drift-tokens "${PROMPT_DRIFT_TOKENS:-256}" \
  --response-residual-weight "${RESPONSE_RESIDUAL_WEIGHT:-1.0}" \
  --prompt-residual-weight "${PROMPT_RESIDUAL_WEIGHT:-0.0}" \
  --prompt-residual-tokens "${PROMPT_RESIDUAL_TOKENS:-256}" \
  --response-nll-weight "${RESPONSE_NLL_WEIGHT:-0.0}" \
  --residual-weight-power "${RESIDUAL_WEIGHT_POWER:-0.5}" \
  --residual-objective "${RESIDUAL_OBJECTIVE:-directional}" \
  --residual-target-source "${RESIDUAL_TARGET_SOURCE:-row-expert}" \
  --residual-target-coefficients "${RESIDUAL_TARGET_COEFFICIENTS:-}" \
  --residual-target-gate-checkpoint "${RESIDUAL_TARGET_GATE_CHECKPOINT:-}" \
  --directional-projection-floor "${DIRECTIONAL_PROJECTION_FLOOR:-0.8}" \
  --directional-projection-weight "${DIRECTIONAL_PROJECTION_WEIGHT:-0.1}" \
  --coefficient-floor "${COEFFICIENT_FLOOR:-0.95}" \
  --coefficient-floor-weight "${COEFFICIENT_FLOOR_WEIGHT:-0.1}" \
  --task-expert-coefficient-floor "${TASK_EXPERT_COEFFICIENT_FLOOR:-0.0}" \
  --task-expert-coefficient-floor-weight "${TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT:-0.0}" \
  --response-span-mode "${RESPONSE_SPAN_MODE:-auto}" \
  --task-balanced-loss \
  --contrastive-negative-loss-weight "${CONTRASTIVE_NEGATIVE_LOSS_WEIGHT:-0.0}" \
  --contrastive-negative-margin "${CONTRASTIVE_NEGATIVE_MARGIN:-0.05}" \
  --contrastive-negative-response-key "${CONTRASTIVE_NEGATIVE_RESPONSE_KEY:-negative_response}" \
  --beta-base "${BETA_BASE:-0.05}" \
  --gamma-gate "${GAMMA_GATE:-0.005}" \
  --device "${DEVICE:-cuda}" \
  --device-map "$DEVICE_MAP" \
  "${MAX_MEMORY_ARGS[@]}" \
  "${TASK_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

SELECT_ARGS=(
  --run-dir "$RUN_DIR"
  --output "$RUN_DIR/selected.gates.json"
  --selection-mode "${SELECT_MODE:-loss-plateau}"
  --plateau-relative-improvement "${SELECT_PLATEAU_RELATIVE_IMPROVEMENT:-0.01}"
  --plateau-patience "${SELECT_PLATEAU_PATIENCE:-2}"
  --plateau-min-epoch "${SELECT_PLATEAU_MIN_EPOCH:-4}"
  --gate-penalty "${SELECT_GATE_PENALTY:-0.0}"
  --residual-weight "${SELECT_RESIDUAL_WEIGHT:-0.0}"
)

if [[ "${SELECT_GATE_PENALTY:-0.0}" != "0" && "${SELECT_GATE_PENALTY:-0.0}" != "0.0" ]]; then
  SELECT_ARGS+=(
    --min-memory-gate "${SELECT_MIN_MEMORY_GATE:-0.82}"
    --max-memory-gate "${SELECT_MAX_MEMORY_GATE:-1.20}"
    --min-tool-gate "${SELECT_MIN_TOOL_GATE:-1.05}"
    --max-tool-gate "${SELECT_MAX_TOOL_GATE:-1.20}"
    --min-code-gate "${SELECT_MIN_CODE_GATE:-1.08}"
    --max-code-gate "${SELECT_MAX_CODE_GATE:-1.18}"
  )
fi

"$PY" scripts/trc/select_trc_gate_checkpoint.py "${SELECT_ARGS[@]}"

BAKED_DIR="${BAKED_DIR:-$ROOT/checkpoints/$EXP_ID-selected}"
"$PY" scripts/eval/opvec_bake_checkpoint.py \
  --config "$CONFIG" \
  --mode-manifest "$MODE" \
  --gate-checkpoint "$RUN_DIR/selected.gates.json" \
  --output "$BAKED_DIR"

echo "[done] EXP_ID=$EXP_ID"
echo "[done] RUN_DIR=$RUN_DIR"
echo "[done] BAKED_DIR=$BAKED_DIR"
