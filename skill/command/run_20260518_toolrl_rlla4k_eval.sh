#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy/eval/toolrl_rlla4k_20260518}"
RUN_ID="${RUN_ID:-toolrl-rlla4k-eval}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a baked HF checkpoint}"
GPU_LIST="${GPU_LIST:-0}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
CONFIG="${CONFIG:-configs/gated_grpo.yaml}"
MANIFEST="${MANIFEST:-/tmp/shared-storage/OnPolicy/data/evaluation/toolrl_rlla4k_test_20260518/toolrl_rlla4k_test_all80.prompts.jsonl}"

SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
SEED="${SEED:-20260518}"

RUN_DIR="$ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"

"$PY" scripts/train/opvec_collect_vllm_rollouts.py \
  --config "$CONFIG" \
  --policy-model "$MODEL_PATH" \
  --policy-id "$RUN_ID" \
  --seed-manifest "$MANIFEST" \
  --output "$RUN_DIR/rollouts.jsonl" \
  --run-id "$RUN_ID" \
  --num-prompts 80 \
  --use-manifest-order \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --tool-max-new-tokens "$MAX_NEW_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --vllm-batch-size "$VLLM_BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --seed "$SEED" \
  --greedy \
  --stream-output \
  --progress-every 10 \
  --no-gate-values

"$PY" scripts/eval/summarize_rollouts.py \
  --rollouts "$RUN_DIR/rollouts.jsonl" \
  --output "$RUN_DIR/summary.json"

echo "$RUN_DIR/summary.json"
