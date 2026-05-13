#!/usr/bin/env bash
set -euo pipefail

# Collect one small token-aware HF rollout file, then update the same rollouts
# with legacy sequence loss and VeRL-style token loss for a clean A/B smoke.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"

MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
SEED="${SEED:-$ROOT/data/source_reward/routed1_correct_official_seed20260510.jsonl}"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUN_NAME="${RUN_NAME:-sequence_vs_token_smoke10_seed20260510}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
ROLL_DIR="$RUN_DIR/rollout"
SEQ_DIR="$RUN_DIR/sequence_update"
TOK_DIR="$RUN_DIR/token_update"
ROLLOUTS="$ROLL_DIR/rollouts.jsonl"

NUM_PROMPTS="${NUM_PROMPTS:-10}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
UPDATE_BATCH_SIZE="${UPDATE_BATCH_SIZE:-4}"
BATCH_LOSS_REDUCTION="${BATCH_LOSS_REDUCTION:-mean}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-4096}"

LR="${LR:-0.005}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.15}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED_VALUE="${SEED_VALUE:-20260510}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-160GiB}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_COLLECT="${FORCE_COLLECT:-0}"
USE_MANIFEST_ORDER="${USE_MANIFEST_ORDER:-0}"

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

mkdir -p "$ROOT/data/source_reward" "$ROOT/runs/gated_grpo" "$ROLL_DIR" "$SEQ_DIR" "$TOK_DIR"

if [[ ! -f "$SEED" ]]; then
  "$PY" scripts/data/build_routed_correct_seed_manifest.py \
    --input-root /mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1 \
    --output "$SEED"
fi

if [[ ! -f "$MODE" ]]; then
  "$PY" scripts/modes/build_opvec4_modes.py \
    --config configs/gated_grpo.yaml
fi

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  echo "[cmd] $(quote_cmd "$@")"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

COLLECT_CMD=(
  "$PY" scripts/train/opvec_collect_hf_rollouts.py
  --config configs/gated_grpo.yaml
  --mode-manifest "$MODE"
  --seed-manifest "$SEED"
  --output "$ROLLOUTS"
  --run-id "$RUN_NAME-rollout"
  --num-prompts "$NUM_PROMPTS"
  --samples-per-prompt "$SAMPLES_PER_PROMPT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-prompt-tokens "$MAX_PROMPT_TOKENS"
  --max-logprob-tokens "$MAX_LOGPROB_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --seed "$SEED_VALUE"
  --device cuda
  --torch-dtype bfloat16
  --gate-parameterization global
  --device-map auto
  "${MAX_MEMORY_ARGS[@]}"
  --behavior-span-reward-weight 0.0
  --stream-output
  --store-token-logprobs
  --progress-every 1
)

if [[ "$USE_MANIFEST_ORDER" == "1" || "$USE_MANIFEST_ORDER" == "true" || "$USE_MANIFEST_ORDER" == "yes" ]]; then
  COLLECT_CMD+=(--use-manifest-order)
fi

if [[ -n "${INIT_GATE_CHECKPOINT:-}" ]]; then
  COLLECT_CMD+=(--gate-checkpoint "$INIT_GATE_CHECKPOINT")
fi
if [[ -n "${MAX_GATED_MODULES:-}" ]]; then
  COLLECT_CMD+=(--max-gated-modules "$MAX_GATED_MODULES")
fi

if [[ "$FORCE_COLLECT" == "1" || ! -s "$ROLLOUTS" ]]; then
  run_cmd "${COLLECT_CMD[@]}"
else
  echo "[reuse] rollout exists: $ROLLOUTS"
fi

UPDATE_COMMON=(
  "$PY" scripts/train/opvec_update_gates_from_rollouts.py
  --config configs/gated_grpo.yaml
  --mode-manifest "$MODE"
  --rollouts "$ROLLOUTS"
  --max-steps "$UPDATE_EPOCHS"
  --max-logprob-tokens "$MAX_LOGPROB_TOKENS"
  --lr "$LR"
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT"
  --ppo-loss-weight 1.0
  --best-response-loss-weight 0.0
  --pairwise-loss-weight 0.0
  --device cuda
  --torch-dtype bfloat16
  --gate-parameterization global
  --device-map auto
  "${MAX_MEMORY_ARGS[@]}"
  --update-batch-size "$UPDATE_BATCH_SIZE"
  --batch-loss-reduction "$BATCH_LOSS_REDUCTION"
  --max-coefficient-delta-from-init "$MAX_COEFF_DELTA"
  --frontier-task-quota tool=4
  --frontier-task-quota memory=3
  --frontier-task-quota code=3
)

if [[ -n "${INIT_GATE_CHECKPOINT:-}" ]]; then
  UPDATE_COMMON+=(--init-gate-checkpoint "$INIT_GATE_CHECKPOINT")
fi
if [[ -n "${MAX_GATED_MODULES:-}" ]]; then
  UPDATE_COMMON+=(--max-gated-modules "$MAX_GATED_MODULES")
fi

run_cmd "${UPDATE_COMMON[@]}" \
  --loss-granularity sequence \
  --output "$SEQ_DIR/gate_updates.jsonl"

run_cmd "${UPDATE_COMMON[@]}" \
  --loss-granularity token \
  --output "$TOK_DIR/gate_updates.jsonl"

echo "[done] rollout: $ROLLOUTS"
echo "[done] sequence summary: $SEQ_DIR/gate_updates.summary.json"
echo "[done] token summary: $TOK_DIR/gate_updates.summary.json"
