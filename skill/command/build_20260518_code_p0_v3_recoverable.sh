#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
BANK_DIR="${BANK_DIR:-$ROOT/data/calibration/code_p0_v3_20260518}"
INPUT="${INPUT:-$BANK_DIR/train_code64.prompts.jsonl}"
OUTPUT="${OUTPUT:-$BANK_DIR/train_recoverable_code.prompts.jsonl}"
REASONFLUX_ROLLOUT="${REASONFLUX_ROLLOUT:-$BANK_DIR/expert_rollouts/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl}"
DEEPSEEK_ROLLOUT="${DEEPSEEK_ROLLOUT:-$BANK_DIR/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_code_p0_v3_train64_s8_seed20260518.merged.jsonl}"
POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-1.0}"
SEED="${SEED:-20260518}"
TAG="${TAG:-code_p0_v3_recoverable_20260518}"

export PYTHONDONTWRITEBYTECODE=1

"$PY" scripts/data/build_recoverable_code_calibration.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --expert-rollout "$REASONFLUX_ROLLOUT" \
  --expert-rollout "$DEEPSEEK_ROLLOUT" \
  --positive-threshold "$POSITIVE_THRESHOLD" \
  --tool-count 0 \
  --memory-count 0 \
  --code-count -1 \
  --seed "$SEED" \
  --tag "$TAG"
