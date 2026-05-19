#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
GPU="${GPU:-6}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy/runs/state_distribution/p0_evaltarget96_k8_20260518}"
MODE="${MODE:-/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json}"
SEED_MANIFEST="${SEED_MANIFEST:-/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-96}"
SEED_VALUE="${SEED_VALUE:-20260518}"

run_collect() {
  local name="$1"
  local policy_model="$2"
  local gate_checkpoint="$3"
  local out_dir="$ROOT/$name"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train/opvec_collect_vllm_rollouts.py \
    --config configs/gated_grpo.yaml \
    --mode-manifest "$MODE" \
    --policy-model "$policy_model" \
    --output "$out_dir/rollouts.jsonl" \
    --run-id "p0-state-k8-${name}-20260518" \
    --num-prompts "$NUM_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-new-tokens 1024 \
    --tool-max-new-tokens 512 \
    --code-max-new-tokens 4096 \
    --memory-update-max-new-tokens 2048 \
    --memory-final-max-new-tokens 2048 \
    --max-prompt-tokens 8192 \
    --max-model-len 12288 \
    --vllm-batch-size 32 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.82 \
    --dtype bfloat16 \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed "$SEED_VALUE" \
    --stream-output \
    --progress-every 10 \
    --seed-manifest "$SEED_MANIFEST" \
    --use-manifest-order \
    --gate-checkpoint "$gate_checkpoint" \
    --behavior-span-reward-weight 0.0
}

run_collect \
  c033 \
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_001/baked_policy \
  /tmp/shared-storage/OnPolicy/data/init_gates/eg72_p1_evaltarget_20260518/init_gc_c033333.json

run_collect \
  init1 \
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_001/baked_policy \
  /tmp/shared-storage/OnPolicy/data/init_gates/eg72_p1_evaltarget_20260518/init_gc_init1.json

