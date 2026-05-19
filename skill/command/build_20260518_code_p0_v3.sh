#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518}"
CODECONTESTS="${CODECONTESTS:-/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json}"
SEED="${SEED:-20260518}"
TAG="${TAG:-code_p0_v3_20260518}"
MIN_TESTS="${MIN_TESTS:-10}"
REWARD_TESTS="${REWARD_TESTS:-6}"
GUARD_TESTS="${GUARD_TESTS:-4}"

export PYTHONDONTWRITEBYTECODE=1

"$PY" scripts/data/build_code_p0_calibration_bank.py \
  --codecontests "$CODECONTESTS" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --tag "$TAG" \
  --min-tests "$MIN_TESTS" \
  --reward-tests "$REWARD_TESTS" \
  --guard-tests "$GUARD_TESTS"
