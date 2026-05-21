#!/usr/bin/env bash
set -euo pipefail

# Quick CURE regression on code hurt subsets:
#   - LiveBenchCodeHurtRcrfVsTa16
#   - LiveCodeBenchCodeHurtRcrfVsTa16
#
# Required:
#   MODEL=/path/to/hf_model_or_checkpoint
#
# Optional:
#   GPU=0                         # physical GPU id passed to CURE --gpu_groups
#   GPU_GROUPS="[[1],[2]]"         # optional raw CURE gpu_groups override
#   DATASETS="LiveBenchCodeHurtRcrfVsTa16 LiveCodeBenchCodeHurtRcrfVsTa16"
#   K_CODE=4 K_CASE=4 TEMP=1.0 MAX_TEST=8
#   MAX_MODEL_LEN=32768 MAX_GENERATION_TOKEN=10000 NUM_CHUNKS=16

if [[ -z "${MODEL:-}" ]]; then
  echo "ERROR: set MODEL=/path/to/model" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CURE_EVAL_DIR="${CURE_EVAL_DIR:-/mnt/cache/wuruixiao/users/lsc/CURE/evaluation}"
GPU="${GPU:-0}"
GPU_GROUPS="${GPU_GROUPS:-[[$GPU]]}"
DATASETS="${DATASETS:-LiveBenchCodeHurtRcrfVsTa16 LiveCodeBenchCodeHurtRcrfVsTa16}"

K_CODE="${K_CODE:-4}"
K_CASE="${K_CASE:-4}"
TEMP="${TEMP:-1.0}"
MAX_TEST="${MAX_TEST:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_GENERATION_TOKEN="${MAX_GENERATION_TOKEN:-10000}"
NUM_CHUNKS="${NUM_CHUNKS:-16}"
SINGLE_EVAL="${SINGLE_EVAL:-False}"
IS_FINAL_EVAL="${IS_FINAL_EVAL:-False}"
EXE_VERBOSE="${EXE_VERBOSE:-True}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"

cd "$CURE_EVAL_DIR"

for DATASET in $DATASETS; do
  echo "[code-hurt-eval] dataset=$DATASET model=$MODEL gpu=$GPU"
  "$PYTHON_BIN" eval.py \
    --use_api False \
    --pretrained_model "$MODEL" \
    --single_eval "$SINGLE_EVAL" \
    --dataset "$DATASET" \
    --k_code "$K_CODE" \
    --k_case "$K_CASE" \
    --scale_tuple_list "[($K_CODE, $K_CASE)]" \
    --temp "$TEMP" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_generation_token "$MAX_GENERATION_TOKEN" \
    --max_test "$MAX_TEST" \
    --num_chunks "$NUM_CHUNKS" \
    --gpu_groups "$GPU_GROUPS" \
    --is_final_eval "$IS_FINAL_EVAL" \
    --exe_verbose "$EXE_VERBOSE" \
    --trust_remote_code "$TRUST_REMOTE_CODE"
done
