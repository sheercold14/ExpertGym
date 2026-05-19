#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy/eval/sota_monitor64_20260518}"
RUN_ID="${RUN_ID:-sota-monitor64-eval}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a baked HF checkpoint}"
GPU_LIST="${GPU_LIST:-0}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
CONFIG="${CONFIG:-configs/gated_grpo.yaml}"
MANIFEST="${MANIFEST:-/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/monitor64.prompts.jsonl}"

NUM_PROMPTS="${NUM_PROMPTS:-64}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOOL_MAX_NEW_TOKENS="${TOOL_MAX_NEW_TOKENS:-512}"
CODE_MAX_NEW_TOKENS="${CODE_MAX_NEW_TOKENS:-4096}"
MEMORY_UPDATE_MAX_NEW_TOKENS="${MEMORY_UPDATE_MAX_NEW_TOKENS:-2048}"
MEMORY_FINAL_MAX_NEW_TOKENS="${MEMORY_FINAL_MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
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
  --num-prompts "$NUM_PROMPTS" \
  --use-manifest-order \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --tool-max-new-tokens "$TOOL_MAX_NEW_TOKENS" \
  --code-max-new-tokens "$CODE_MAX_NEW_TOKENS" \
  --memory-update-max-new-tokens "$MEMORY_UPDATE_MAX_NEW_TOKENS" \
  --memory-final-max-new-tokens "$MEMORY_FINAL_MAX_NEW_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --vllm-batch-size "$VLLM_BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --temperature 0.7 \
  --top-p 0.95 \
  --seed "$SEED" \
  --stream-output \
  --progress-every 8 \
  --no-gate-values

"$PY" scripts/eval/summarize_rollouts.py \
  --rollouts "$RUN_DIR/rollouts.jsonl" \
  --output "$RUN_DIR/summary.json"

echo "$RUN_DIR/summary.json"
