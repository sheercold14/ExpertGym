#!/usr/bin/env bash
set -euo pipefail

# Train fine-grained OP-VEC coefficients on the fixed balanced frontier
# calibration set.  This uses the global-parameter manager: three global expert
# coefficients plus one residual coefficient per mergeable parameter/expert.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash skill/command/run_fixed_frontier7_global_parameter.sh

Useful environment overrides:
  INIT_GATE_CHECKPOINT=/path/to/global/gate_updates.gates.json
  RUN_NAME=fixed_frontier7_global_parameter_custom
  GPU_LIST=0,1,2,3,4,5
  MAX_STEPS=3
  MAX_LOGPROB_TOKENS=8192
  LR=0.003
  PRIOR_LOSS_WEIGHT=0.05
  MAX_COEFF_DELTA=0.05
  TASK_NORMALIZE_ADVANTAGES=1
  LENGTH_NORMALIZE_POLICY_LOGPROB=1
EOF
  exit 0
fi

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
RUN_NAME="${RUN_NAME:-fixed_frontier7_global_parameter_e3_from_e3b_20260511}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
BAKE_OUTPUT="${BAKE_OUTPUT:-$ROOT/checkpoints/$RUN_NAME}"

DEFAULT_INIT_GATE="$ROOT/runs/gated_grpo/fixed_frontier7_global_continue_e3b_20260511/gate_updates.gates.json"
INIT_GATE_CHECKPOINT="${INIT_GATE_CHECKPOINT:-$DEFAULT_INIT_GATE}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

MAX_STEPS="${MAX_STEPS:-3}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-8192}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-55GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-160GiB}"
LR="${LR:-0.003}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.05}"
MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.05}"
EARLY_STOP_GRAD_NORM="${EARLY_STOP_GRAD_NORM:-0.02}"
EARLY_STOP_GATE_DELTA="${EARLY_STOP_GATE_DELTA:-0.0005}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-0}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-0}"

INIT_ARGS=()
if [[ -n "$INIT_GATE_CHECKPOINT" ]]; then
  if [[ ! -f "$INIT_GATE_CHECKPOINT" ]]; then
    echo "[error] INIT_GATE_CHECKPOINT not found: $INIT_GATE_CHECKPOINT" >&2
    exit 1
  fi
  INIT_ARGS+=(--init-gate-checkpoint "$INIT_GATE_CHECKPOINT")
fi

OBJECTIVE_ARGS=()
if [[ "$TASK_NORMALIZE_ADVANTAGES" == "1" || "$TASK_NORMALIZE_ADVANTAGES" == "true" || "$TASK_NORMALIZE_ADVANTAGES" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--task-normalize-advantages)
fi
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--length-normalize-policy-logprob)
fi

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

mkdir -p "$RUN_DIR" "$ROOT/checkpoints"

echo "[run] calibration: $CALIBRATION"
echo "[run] init gate: ${INIT_GATE_CHECKPOINT:-none}"
echo "[run] run dir: $RUN_DIR"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] max memory args: ${MAX_MEMORY_ARGS[*]}"
echo "[run] global-parameter regularization: lr=$LR prior=$PRIOR_LOSS_WEIGHT max_delta=$MAX_COEFF_DELTA"
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
  --gate-parameterization global-parameter \
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
