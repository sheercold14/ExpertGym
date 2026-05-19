#!/usr/bin/env bash
set -euo pipefail

# Build the SOTA-oriented recovery calibration bank.
#
# Outputs:
#   train128  = tool32 + memory48 + code48
#   monitor64 = tool16 + memory24 + code24
#   guard64   = tool16 + memory24 + code24
#
# This script only combines existing leakage-controlled manifests.  It does not
# start training, run expert rollout, or overwrite older calibration banks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/calibration/sota_recovery_calib_v3_20260518}"
SEED="${SEED:-20260518}"

"$PY" scripts/data/build_sota_recovery_calibration_v3.py \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  "$@"

echo "[done] train:   $OUTPUT_DIR/train128.prompts.jsonl"
echo "[done] monitor: $OUTPUT_DIR/monitor64.prompts.jsonl"
echo "[done] guard:   $OUTPUT_DIR/guard64.prompts.jsonl"
echo "[done] summary: $OUTPUT_DIR/summary.json"
