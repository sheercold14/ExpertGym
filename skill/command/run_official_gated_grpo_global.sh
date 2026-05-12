#!/usr/bin/env bash
set -euo pipefail

# Official-aligned OP-VEC Gated-GRPO training launcher.
#
# Defaults are chosen for a real first run, not a smoke test:
# - official routed correct-pool manifest
# - full OP-VEC gated modules, no --max-gated-modules cap
# - global gate parameterization
# - official task rewards only; no behavior-span shaping
# - true old_logprob collection; no --skip-logprob

cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"

export MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
export SEED="${SEED:-$ROOT/data/source_reward/routed1_correct_official_seed20260510.jsonl}"

GPU_LIST="${GPU_LIST:-0,1,2,5}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-160GiB}"

RUN_NAME="${RUN_NAME:-official_global_48x4_i2_seed20260510}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
BAKE_OUTPUT="${BAKE_OUTPUT:-$ROOT/checkpoints/gated_grpo_$RUN_NAME}"

NUM_ITERS="${NUM_ITERS:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-48}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"

# Memory trajectories are multi-turn, so prompt/logprob windows must be larger
# than the original final-answer-only smoke defaults.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-8192}"

LR="${LR:-0.005}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.15}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-0}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-0}"

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED_VALUE="${SEED_VALUE:-20260510}"

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

mkdir -p "$ROOT/data/source_reward" "$ROOT/runs/gated_grpo" "$ROOT/checkpoints"

if [[ ! -f "$SEED" ]]; then
  echo "[prepare] building official routed seed manifest: $SEED"
  "$PY" scripts/data/build_routed_correct_seed_manifest.py \
    --input-root /mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1 \
    --output "$SEED"
fi

if [[ ! -f "$MODE" ]]; then
  echo "[prepare] mode manifest missing; building: $MODE"
  "$PY" scripts/modes/build_opvec4_modes.py \
    --config configs/gated_grpo.yaml
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

if [[ -n "${INIT_GATE_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--init-gate-checkpoint "$INIT_GATE_CHECKPOINT")
fi
if [[ "$TASK_NORMALIZE_ADVANTAGES" == "1" || "$TASK_NORMALIZE_ADVANTAGES" == "true" || "$TASK_NORMALIZE_ADVANTAGES" == "yes" ]]; then
  EXTRA_ARGS+=(--task-normalize-advantages)
fi
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  EXTRA_ARGS+=(--length-normalize-policy-logprob)
fi

echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] max memory args: ${MAX_MEMORY_ARGS[*]}"
echo "[run] seed manifest: $SEED"
echo "[run] mode manifest: $MODE"
echo "[run] run dir: $RUN_DIR"
echo "[run] bake output: $BAKE_OUTPUT"

"$PY" scripts/train/opvec_gated_grpo_loop.py \
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
  --progress-every 4 \
  --bake-final \
  --bake-output "$BAKE_OUTPUT" \
  "${EXTRA_ARGS[@]}"

echo "[done] run dir: $RUN_DIR"
echo "[done] baked checkpoint: $BAKE_OUTPUT"
