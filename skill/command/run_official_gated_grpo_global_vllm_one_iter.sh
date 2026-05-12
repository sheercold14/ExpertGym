#!/usr/bin/env bash
set -euo pipefail

# Official-aligned one-iteration benchmark for the bake+vLLM rollout path.

cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"

export MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
export SEED="${SEED:-$ROOT/data/source_reward/routed1_correct_official_seed20260510.jsonl}"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-160GiB}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

RUN_NAME="${RUN_NAME:-official_global_vllm_48x4_i1_seed20260510}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"

NUM_ITERS="${NUM_ITERS:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-48}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-0}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-0}"
ADVANTAGE_FIELD="${ADVANTAGE_FIELD:-}"
ADVANTAGE_FIELD_FRONTIER_WEIGHT="${ADVANTAGE_FIELD_FRONTIER_WEIGHT:-1}"
TRAIN_COEFFICIENTS="${TRAIN_COEFFICIENTS:-}"
TOOL_MIN_MARGIN_OVER_MEMORY="${TOOL_MIN_MARGIN_OVER_MEMORY:-0}"
TOOL_MIN_MARGIN_OVER_CODE="${TOOL_MIN_MARGIN_OVER_CODE:-0}"

LR="${LR:-0.005}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.15}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED_VALUE="${SEED_VALUE:-20260510}"

mkdir -p "$ROOT/runs/gated_grpo"

echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] seed manifest: $SEED"
echo "[run] mode manifest: $MODE"
echo "[run] run dir: $RUN_DIR"
echo "[run] vLLM tp=$TENSOR_PARALLEL_SIZE batch=$ROLLOUT_BATCH_SIZE max_model_len=$MAX_MODEL_LEN"
echo "[run] HF update max memory args: ${MAX_MEMORY_ARGS[*]}"
echo "[run] HF update gradient checkpointing: $GRADIENT_CHECKPOINTING"

GRADIENT_CHECKPOINTING_ARGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" || "$GRADIENT_CHECKPOINTING" == "yes" ]]; then
  GRADIENT_CHECKPOINTING_ARGS+=(--gradient-checkpointing)
fi
OBJECTIVE_ARGS=()
if [[ "$TASK_NORMALIZE_ADVANTAGES" == "1" || "$TASK_NORMALIZE_ADVANTAGES" == "true" || "$TASK_NORMALIZE_ADVANTAGES" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--task-normalize-advantages)
fi
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--length-normalize-policy-logprob)
fi
if [[ -n "$ADVANTAGE_FIELD" ]]; then
  OBJECTIVE_ARGS+=(--advantage-field "$ADVANTAGE_FIELD")
  if [[ "$ADVANTAGE_FIELD_FRONTIER_WEIGHT" == "0" || "$ADVANTAGE_FIELD_FRONTIER_WEIGHT" == "false" || "$ADVANTAGE_FIELD_FRONTIER_WEIGHT" == "no" ]]; then
    OBJECTIVE_ARGS+=(--no-advantage-field-frontier-weight)
  fi
fi
if [[ -n "$TRAIN_COEFFICIENTS" ]]; then
  OBJECTIVE_ARGS+=(--train-coefficient "$TRAIN_COEFFICIENTS")
fi
if [[ "$TOOL_MIN_MARGIN_OVER_MEMORY" != "0" && "$TOOL_MIN_MARGIN_OVER_MEMORY" != "0.0" ]]; then
  OBJECTIVE_ARGS+=(--tool-min-margin-over-memory "$TOOL_MIN_MARGIN_OVER_MEMORY")
fi
if [[ "$TOOL_MIN_MARGIN_OVER_CODE" != "0" && "$TOOL_MIN_MARGIN_OVER_CODE" != "0.0" ]]; then
  OBJECTIVE_ARGS+=(--tool-min-margin-over-code "$TOOL_MIN_MARGIN_OVER_CODE")
fi

"$PY" scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --seed-manifest "$SEED" \
  --run-dir "$RUN_DIR" \
  --run-id "$RUN_NAME" \
  --num-iters "$NUM_ITERS" \
  --num-prompts "$NUM_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --gate-parameterization global \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --max-logprob-tokens "$MAX_LOGPROB_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --vllm-batch-size "$ROLLOUT_BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "$SEED_VALUE" \
  --lr "$LR" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --max-coefficient-delta-from-init "$MAX_COEFF_DELTA" \
  --behavior-span-reward-weight 0.0 \
  --frontier-task-quota tool=16 \
  --frontier-task-quota memory=16 \
  --frontier-task-quota code=16 \
  --task-weight tool=1.0 \
  --task-weight memory=1.0 \
  --task-weight code=1.0 \
  --device-map auto \
  "${MAX_MEMORY_ARGS[@]}" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  "${OBJECTIVE_ARGS[@]}" \
  --progress-every 4

echo "[done] run dir: $RUN_DIR"
