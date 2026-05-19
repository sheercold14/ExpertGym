#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
OUT="${OUT:-/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1}"

cd "$REPO"

"$PY" scripts/trc/build_trc_calibration_v1.py \
  --output-dir "$OUT" \
  --per-task "${PER_TASK:-32}" \
  --positive-threshold "${POSITIVE_THRESHOLD:-1.0}"
