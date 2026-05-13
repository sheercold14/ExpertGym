#!/usr/bin/env bash
set -euo pipefail

# High-information fixed-calibration global OP-VEC training.
#
# The loop intentionally separates on-policy GRPO and offline expert recovery:
#   stage A: current-policy rollout on fixed prompts, PPO/GRPO update only;
#   stage B: fixed expert-recovery rows, PPO disabled, best-response/pairwise only;
#   guard: fixed heldout prompts summarized before the next iteration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$ROOT/data/source_reward/routed1_correct_official_seed20260510.jsonl}"
BUNDLE_PREFIX="${BUNDLE_PREFIX:-$ROOT/data/calibration/high_info_v1_seed20260511}"
PROMPTS="${PROMPTS:-$BUNDLE_PREFIX.prompts.jsonl}"
DISTILL="${DISTILL:-$BUNDLE_PREFIX.distill.jsonl}"
GUARD="${GUARD:-$BUNDLE_PREFIX.guard.jsonl}"

FRONTIER_ROLLOUT="${FRONTIER_ROLLOUT:-$ROOT/data/calibration/frontier_balanced_7each_seed20260511.jsonl}"
DISTILL_ROLLOUT_A="${DISTILL_ROLLOUT_A:-$ROOT/data/calibration/fixed_balanced13_codecont_tool_memorysplit_diverse_expertrecovery_20260511.jsonl}"
DISTILL_ROLLOUT_B="${DISTILL_ROLLOUT_B:-$ROOT/data/calibration/toolstrict_memoryevidence_independent_posgrad8each_20260511.jsonl}"
DISTILL_ROLLOUT_C="${DISTILL_ROLLOUT_C:-$ROOT/data/calibration/tool24_memcode8_strict_train32_memagent_reward_seed20260511.jsonl}"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-160GiB}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

RUN_NAME="${RUN_NAME:-high_info_v1_global_twostage_i3_seed20260511}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
BAKE_OUTPUT="${BAKE_OUTPUT:-$ROOT/checkpoints/$RUN_NAME}"

NUM_ITERS="${NUM_ITERS:-3}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
GUARD_SAMPLES_PER_PROMPT="${GUARD_SAMPLES_PER_PROMPT:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED_VALUE="${SEED_VALUE:-20260511}"

STAGE_A_STEPS="${STAGE_A_STEPS:-1}"
RUN_STAGE_B="${RUN_STAGE_B:-0}"
STAGE_B_STEPS="${STAGE_B_STEPS:-1}"
LR_STAGE_A="${LR_STAGE_A:-0.003}"
LR_STAGE_B="${LR_STAGE_B:-0.001}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
MAX_COEFF_DELTA_STAGE="${MAX_COEFF_DELTA_STAGE:-0.04}"
MAX_COEFF_DELTA_STAGE_A="${MAX_COEFF_DELTA_STAGE_A:-$MAX_COEFF_DELTA_STAGE}"
MAX_COEFF_DELTA_STAGE_B="${MAX_COEFF_DELTA_STAGE_B:-0.01}"
DISTILL_BEST_RESPONSE_LOSS_WEIGHT="${DISTILL_BEST_RESPONSE_LOSS_WEIGHT:-0.005}"
DISTILL_PAIRWISE_LOSS_WEIGHT="${DISTILL_PAIRWISE_LOSS_WEIGHT:-0.0}"
DISTILL_PAIRWISE_MARGIN="${DISTILL_PAIRWISE_MARGIN:-0.0}"
MAX_PAIRWISE_PAIRS_PER_ROW="${MAX_PAIRWISE_PAIRS_PER_ROW:-4}"
FRONTIER_TOOL_QUOTA="${FRONTIER_TOOL_QUOTA:-16}"
FRONTIER_MEMORY_QUOTA="${FRONTIER_MEMORY_QUOTA:-16}"
FRONTIER_CODE_QUOTA="${FRONTIER_CODE_QUOTA:-16}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-1}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-1}"
LENGTH_NORMALIZE_DISTILL_LOGPROB="${LENGTH_NORMALIZE_DISTILL_LOGPROB:-1}"
ENFORCE_GUARD="${ENFORCE_GUARD:-1}"
MIN_TOOL_PARSEABLE_RATE="${MIN_TOOL_PARSEABLE_RATE:-0.25}"
MAX_TOOL_ZERO_CALL_RATE="${MAX_TOOL_ZERO_CALL_RATE:-0.75}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  echo "[cmd] $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$ROOT/data/calibration" "$ROOT/runs/gated_grpo" "$ROOT/checkpoints" "$RUN_DIR"
fi

if [[ ! -f "$MODE" ]]; then
  run_cmd "$PY" scripts/modes/build_opvec4_modes.py --config configs/gated_grpo.yaml
fi

if [[ ! -f "$PROMPTS" || ! -f "$DISTILL" || ! -f "$GUARD" ]]; then
  run_cmd "$PY" scripts/data/build_high_info_calibration.py \
    --source-manifest "$SOURCE_MANIFEST" \
    --frontier-rollout "$FRONTIER_ROLLOUT" \
    --distill-rollout "$DISTILL_ROLLOUT_A" \
    --distill-rollout "$DISTILL_ROLLOUT_B" \
    --distill-rollout "$DISTILL_ROLLOUT_C" \
    --output-prefix "$BUNDLE_PREFIX" \
    --strict
fi

PROMPT_COUNT="$(wc -l < "$PROMPTS" | tr -d ' ')"
GUARD_COUNT="$(wc -l < "$GUARD" | tr -d ' ')"

OBJECTIVE_A_ARGS=()
if [[ "$TASK_NORMALIZE_ADVANTAGES" == "1" || "$TASK_NORMALIZE_ADVANTAGES" == "true" || "$TASK_NORMALIZE_ADVANTAGES" == "yes" ]]; then
  OBJECTIVE_A_ARGS+=(--task-normalize-advantages)
fi
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  OBJECTIVE_A_ARGS+=(--length-normalize-policy-logprob)
fi

OBJECTIVE_B_ARGS=()
if [[ "$LENGTH_NORMALIZE_DISTILL_LOGPROB" == "1" || "$LENGTH_NORMALIZE_DISTILL_LOGPROB" == "true" || "$LENGTH_NORMALIZE_DISTILL_LOGPROB" == "yes" ]]; then
  OBJECTIVE_B_ARGS+=(--length-normalize-logprob)
fi

GUARD_ARGS=()
if [[ "$ENFORCE_GUARD" == "1" || "$ENFORCE_GUARD" == "true" || "$ENFORCE_GUARD" == "yes" ]]; then
  GUARD_ARGS+=(--fail-on-guard)
fi

echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] run_dir=$RUN_DIR"
echo "[run] prompts=$PROMPTS count=$PROMPT_COUNT"
echo "[run] distill=$DISTILL"
echo "[run] guard=$GUARD count=$GUARD_COUNT"
echo "[run] max memory args: ${MAX_MEMORY_ARGS[*]}"
echo "[run] run_stage_b=$RUN_STAGE_B"
echo "[run] stage A max coefficient delta=$MAX_COEFF_DELTA_STAGE_A"
echo "[run] stage B max coefficient delta=$MAX_COEFF_DELTA_STAGE_B"

GATE_CHECKPOINT="${INIT_GATE_CHECKPOINT:-}"
for iteration in $(seq 1 "$NUM_ITERS"); do
  ITER_DIR="$RUN_DIR/iter_$(printf '%03d' "$iteration")"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$ITER_DIR"
  fi
  INPUT_BAKE="$ITER_DIR/baked_policy_input"
  ROLLOUTS="$ITER_DIR/rollouts.jsonl"
  STAGE_A_UPDATES="$ITER_DIR/stage_a_gate_updates.jsonl"
  STAGE_B_UPDATES="$ITER_DIR/stage_b_gate_updates.jsonl"
  CANDIDATE_BAKE="$ITER_DIR/baked_policy_candidate"
  GUARD_ROLLOUTS="$ITER_DIR/guard_rollouts.jsonl"
  GUARD_SUMMARY="$ITER_DIR/guard_rollouts.summary.json"

  BAKE_INPUT_ARGS=()
  COLLECT_GATE_ARGS=()
  UPDATE_A_INIT_ARGS=()
  if [[ -n "$GATE_CHECKPOINT" ]]; then
    BAKE_INPUT_ARGS+=(--gate-checkpoint "$GATE_CHECKPOINT")
    COLLECT_GATE_ARGS+=(--gate-checkpoint "$GATE_CHECKPOINT")
    UPDATE_A_INIT_ARGS+=(--init-gate-checkpoint "$GATE_CHECKPOINT")
  fi

  run_cmd "$PY" scripts/eval/opvec_bake_checkpoint.py \
    --config configs/gated_grpo.yaml \
    --mode-manifest "$MODE" \
    --output "$INPUT_BAKE" \
    "${BAKE_INPUT_ARGS[@]}"

  run_cmd "$PY" scripts/train/opvec_collect_vllm_rollouts.py \
    --config configs/gated_grpo.yaml \
    --mode-manifest "$MODE" \
    --policy-model "$INPUT_BAKE" \
    --seed-manifest "$PROMPTS" \
    --output "$ROLLOUTS" \
    --run-id "$RUN_NAME-iter$(printf '%03d' "$iteration")" \
    --num-prompts "$PROMPT_COUNT" \
    --use-manifest-order \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --vllm-batch-size "$VLLM_BATCH_SIZE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$TORCH_DTYPE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$((SEED_VALUE + iteration - 1))" \
    --stream-output \
    --progress-every 4 \
    "${COLLECT_GATE_ARGS[@]}" \
    --behavior-span-reward-weight 0.0

  run_cmd "$PY" scripts/train/opvec_update_gates_from_rollouts.py \
    --config configs/gated_grpo.yaml \
    --mode-manifest "$MODE" \
    --rollouts "$ROLLOUTS" \
    --output "$STAGE_A_UPDATES" \
    --max-steps "$STAGE_A_STEPS" \
    --max-logprob-tokens "$MAX_LOGPROB_TOKENS" \
    --fill-missing-old-logprob \
    --lr "$LR_STAGE_A" \
    --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
    --ppo-loss-weight 1.0 \
    --best-response-loss-weight 0.0 \
    --pairwise-loss-weight 0.0 \
    --pairwise-margin 0.0 \
    --max-pairwise-pairs-per-row 0 \
    --min-grad-norm-for-step 0.0 \
    --device cuda \
    --torch-dtype "$TORCH_DTYPE" \
    --gate-parameterization global \
    --device-map auto \
    "${MAX_MEMORY_ARGS[@]}" \
    --gradient-checkpointing \
    --frontier-task-quota "tool=$FRONTIER_TOOL_QUOTA" \
    --frontier-task-quota "memory=$FRONTIER_MEMORY_QUOTA" \
    --frontier-task-quota "code=$FRONTIER_CODE_QUOTA" \
    --task-weight tool=1.0 \
    --task-weight memory=1.0 \
    --task-weight code=1.0 \
    --max-coefficient-delta-from-init "$MAX_COEFF_DELTA_STAGE_A" \
    "${OBJECTIVE_A_ARGS[@]}" \
    "${UPDATE_A_INIT_ARGS[@]}"

  STAGE_A_GATE="$ITER_DIR/stage_a_gate_updates.gates.json"

  if [[ "$RUN_STAGE_B" == "1" || "$RUN_STAGE_B" == "true" || "$RUN_STAGE_B" == "yes" ]]; then
    run_cmd "$PY" scripts/train/opvec_update_gates_from_rollouts.py \
      --config configs/gated_grpo.yaml \
      --mode-manifest "$MODE" \
      --rollouts "$DISTILL" \
      --output "$STAGE_B_UPDATES" \
      --max-steps "$STAGE_B_STEPS" \
      --max-logprob-tokens "$MAX_LOGPROB_TOKENS" \
      --lr "$LR_STAGE_B" \
      --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
      --ppo-loss-weight 0.0 \
      --best-response-loss-weight "$DISTILL_BEST_RESPONSE_LOSS_WEIGHT" \
      --pairwise-loss-weight "$DISTILL_PAIRWISE_LOSS_WEIGHT" \
      --pairwise-margin "$DISTILL_PAIRWISE_MARGIN" \
      --max-pairwise-pairs-per-row "$MAX_PAIRWISE_PAIRS_PER_ROW" \
      --min-grad-norm-for-step 0.0 \
      --device cuda \
      --torch-dtype "$TORCH_DTYPE" \
      --gate-parameterization global \
      --device-map auto \
      "${MAX_MEMORY_ARGS[@]}" \
      --gradient-checkpointing \
      --task-weight tool=1.0 \
      --task-weight memory=1.0 \
      --task-weight code=1.0 \
      --max-coefficient-delta-from-init "$MAX_COEFF_DELTA_STAGE_B" \
      --init-gate-checkpoint "$STAGE_A_GATE" \
      "${OBJECTIVE_B_ARGS[@]}"
    GATE_CHECKPOINT="$ITER_DIR/stage_b_gate_updates.gates.json"
  else
    echo "[iter $iteration] skipping Stage B distill; set RUN_STAGE_B=1 to enable low-weight expert recovery"
    GATE_CHECKPOINT="$STAGE_A_GATE"
  fi

  run_cmd "$PY" scripts/eval/opvec_bake_checkpoint.py \
    --config configs/gated_grpo.yaml \
    --mode-manifest "$MODE" \
    --gate-checkpoint "$GATE_CHECKPOINT" \
    --output "$CANDIDATE_BAKE"

  run_cmd "$PY" scripts/train/opvec_collect_vllm_rollouts.py \
    --config configs/gated_grpo.yaml \
    --mode-manifest "$MODE" \
    --policy-model "$CANDIDATE_BAKE" \
    --seed-manifest "$GUARD" \
    --output "$GUARD_ROLLOUTS" \
    --run-id "$RUN_NAME-guard$(printf '%03d' "$iteration")" \
    --num-prompts "$GUARD_COUNT" \
    --use-manifest-order \
    --samples-per-prompt "$GUARD_SAMPLES_PER_PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --vllm-batch-size "$VLLM_BATCH_SIZE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$TORCH_DTYPE" \
    --greedy \
    --seed "$((SEED_VALUE + 1000 + iteration))" \
    --stream-output \
    --progress-every 2 \
    --gate-checkpoint "$GATE_CHECKPOINT" \
    --behavior-span-reward-weight 0.0

  run_cmd "$PY" scripts/eval/summarize_rollouts.py \
    --rollouts "$GUARD_ROLLOUTS" \
    --output "$GUARD_SUMMARY" \
    --min-tool-parseable-rate "$MIN_TOOL_PARSEABLE_RATE" \
    --max-tool-zero-call-rate "$MAX_TOOL_ZERO_CALL_RATE" \
    "${GUARD_ARGS[@]}"

  echo "[iter $iteration] candidate gate: $GATE_CHECKPOINT"
  echo "[iter $iteration] guard summary: $GUARD_SUMMARY"
done

run_cmd "$PY" scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --gate-checkpoint "$GATE_CHECKPOINT" \
  --output "$BAKE_OUTPUT"

echo "[done] final gates: $GATE_CHECKPOINT"
echo "[done] baked checkpoint: $BAKE_OUTPUT"
