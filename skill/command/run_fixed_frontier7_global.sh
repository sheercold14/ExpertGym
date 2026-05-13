#!/usr/bin/env bash
set -euo pipefail

# Train global OP-VEC gates on the fixed balanced frontier calibration set.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/frontier_balanced_7each_seed20260511.jsonl}"
RUN_NAME="${RUN_NAME:-fixed_frontier7_global_e3_20260511}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
BAKE_OUTPUT="${BAKE_OUTPUT:-$ROOT/checkpoints/$RUN_NAME}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

MAX_STEPS="${MAX_STEPS:-3}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-8192}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-55GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-160GiB}"
LR="${LR:-0.005}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.15}"
EARLY_STOP_GRAD_NORM="${EARLY_STOP_GRAD_NORM:-0.02}"
EARLY_STOP_GATE_DELTA="${EARLY_STOP_GATE_DELTA:-0.001}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-0}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-0}"
ADVANTAGE_FIELD="${ADVANTAGE_FIELD:-}"
ADVANTAGE_FIELD_FRONTIER_WEIGHT="${ADVANTAGE_FIELD_FRONTIER_WEIGHT:-1}"
TRAIN_COEFFICIENTS="${TRAIN_COEFFICIENTS:-}"
TOOL_MIN_MARGIN_OVER_MEMORY="${TOOL_MIN_MARGIN_OVER_MEMORY:-0}"
TOOL_MIN_MARGIN_OVER_CODE="${TOOL_MIN_MARGIN_OVER_CODE:-0}"
INIT_ARGS=()
if [[ -n "${INIT_GATE_CHECKPOINT:-}" ]]; then
  INIT_ARGS+=(--init-gate-checkpoint "$INIT_GATE_CHECKPOINT")
fi

OBJECTIVE_ARGS=()
if [[ "$TASK_NORMALIZE_ADVANTAGES" == "1" || "$TASK_NORMALIZE_ADVANTAGES" == "true" || "$TASK_NORMALIZE_ADVANTAGES" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--task-normalize-advantages)
fi
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--length-normalize-policy-logprob)
fi
if [[ -n "$ADVANTAGE_FIELD" ]]; then
  OBJECTIVE_ARGS+=(--advantage-field "$ADVANTAGE_FIELD")
  if [[ "$ADVANTAGE_FIELD_FRONTIER_WEIGHT" == "0" || "$ADVANTAGE_FIELD_FRONTIER_WEIGHT" == "false" || "$ADVANTAGE_FIELD_FRONTIER_WEIGHT" == "no" ]]; then
    OBJECTIVE_ARGS+=(--no-advantage-field-frontier-weight)
  fi
fi
if [[ -n "$TRAIN_COEFFICIENTS" ]]; then
  OBJECTIVE_ARGS+=(--train-coefficient "$TRAIN_COEFFICIENTS")
fi
if [[ "$TOOL_MIN_MARGIN_OVER_MEMORY" != "0" && "$TOOL_MIN_MARGIN_OVER_MEMORY" != "0.0" ]]; then
  OBJECTIVE_ARGS+=(--tool-min-margin-over-memory "$TOOL_MIN_MARGIN_OVER_MEMORY")
fi
if [[ "$TOOL_MIN_MARGIN_OVER_CODE" != "0" && "$TOOL_MIN_MARGIN_OVER_CODE" != "0.0" ]]; then
  OBJECTIVE_ARGS+=(--tool-min-margin-over-code "$TOOL_MIN_MARGIN_OVER_CODE")
fi

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

mkdir -p "$RUN_DIR" "$ROOT/checkpoints"

echo "[run] calibration: $CALIBRATION"
echo "[run] run dir: $RUN_DIR"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] max memory args: ${MAX_MEMORY_ARGS[*]}"
echo "[run] objective args: ${OBJECTIVE_ARGS[*]:-none}"

"$PY" scripts/train/opvec_update_gates_from_rollouts.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --rollouts "$CALIBRATION" \
  --output "$RUN_DIR/gate_updates.jsonl" \
  --max-steps "$MAX_STEPS" \
  --max-logprob-tokens "$MAX_LOGPROB_TOKENS" \
  --fill-missing-old-logprob \
  --lr "$LR" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --ppo-loss-weight 1.0 \
  --best-response-loss-weight 0.0 \
  --pairwise-loss-weight 0.0 \
  --pairwise-margin 0.0 \
  --max-pairwise-pairs-per-row 0 \
  --min-grad-norm-for-step 0.0 \
  --early-stop-grad-norm "$EARLY_STOP_GRAD_NORM" \
  --early-stop-gate-delta "$EARLY_STOP_GATE_DELTA" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --device cuda \
  --torch-dtype bfloat16 \
  --gate-parameterization global \
  --device-map auto \
  "${MAX_MEMORY_ARGS[@]}" \
  --gradient-checkpointing \
  --task-weight tool=1.0 \
  --task-weight memory=1.0 \
  --task-weight code=1.0 \
  --frontier-task-quota tool=7 \
  --frontier-task-quota memory=7 \
  --frontier-task-quota code=7 \
  --max-coefficient-delta-from-init "$MAX_COEFF_DELTA" \
  "${OBJECTIVE_ARGS[@]}" \
  "${INIT_ARGS[@]}"

"$PY" scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --gate-checkpoint "$RUN_DIR/gate_updates.gates.json" \
  --output "$BAKE_OUTPUT"

echo "[done] gates: $RUN_DIR/gate_updates.gates.json"
echo "[done] baked checkpoint: $BAKE_OUTPUT"
