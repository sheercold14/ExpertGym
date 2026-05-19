#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
CONFIG="${CONFIG:-configs/gated_grpo_layer28.yaml}"
MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
CALIB="${CALIB:-$ROOT/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/trc/trc_layer_init1_20260519}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_MEMORY_ENTRIES="${MAX_MEMORY_ENTRIES:-0=70GiB 1=70GiB}"

MAX_MEMORY_ARGS=()
if [[ -n "$MAX_MEMORY_ENTRIES" ]]; then
  for item in $MAX_MEMORY_ENTRIES; do
    MAX_MEMORY_ARGS+=(--max-memory "$item")
  done
fi

cd "$REPO"

"$PY" scripts/trc/train_trc_layer_gates.py \
  --config "$CONFIG" \
  --mode-manifest "$MODE" \
  --calibration "$CALIB" \
  --output-dir "$RUN_DIR" \
  --gate-parameterization layer-band-coefficient \
  --init-value "${INIT_VALUE:-1.0}" \
  --epochs "${EPOCHS:-3}" \
  --shuffle \
  --seed "${SEED:-20260519}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --lr "${LR:-0.01}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-8}" \
  --hidden-layers "${HIDDEN_LAYERS:-8,16,24,28}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-1536}" \
  --max-response-tokens "${MAX_RESPONSE_TOKENS:-512}" \
  --topk-tokens "${TOPK_TOKENS:-128}" \
  --prompt-drift-tokens "${PROMPT_DRIFT_TOKENS:-256}" \
  --beta-base "${BETA_BASE:-0.02}" \
  --gamma-gate "${GAMMA_GATE:-0.001}" \
  --device "${DEVICE:-cuda}" \
  --device-map "$DEVICE_MAP" \
  "${MAX_MEMORY_ARGS[@]}"
