#!/usr/bin/env bash
set -euo pipefail

# Train one OP-VEC gate strategy from the 1/3 task-arithmetic point on the
# HotpotQA-v1 question-bank calibration data.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  STRATEGY=global bash skill/command/run_qbank_c033333_gate_strategy.sh
  STRATEGY=layer-band bash skill/command/run_qbank_c033333_gate_strategy.sh
  STRATEGY=parameter bash skill/command/run_qbank_c033333_gate_strategy.sh

Key overrides:
  RUN_NAME=qbank_c033333_<strategy>_i2_seed20260511
  GPU_LIST=0,1,2,3,4
  NUM_ITERS=2
  NUM_PROMPTS=100
  SAMPLES_PER_PROMPT=4
  MAX_NEW_TOKENS=1024
  LR=0.003
  MAX_COEFF_DELTA=0.05
  DRY_RUN=1
EOF
  exit 0
fi

cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
QB="${QB:-$ROOT/data/question_bank/ta_avgvec_c033333_hotpotqa_v1}"
CALIBRATION="${CALIBRATION:-$QB/calibration/calib100_seed20260511.prompts.jsonl}"
INIT_VALUE="${INIT_VALUE:-0.3333333333333333}"
STRATEGY="${STRATEGY:-global}"
SAFE_STRATEGY="${STRATEGY//[^A-Za-z0-9_]/_}"
RUN_NAME="${RUN_NAME:-qbank_c033333_${SAFE_STRATEGY}_i2_seed20260511}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
INIT_GATE="$QB/init_gates/init_${SAFE_STRATEGY}_c033333.json"

GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

NUM_ITERS="${NUM_ITERS:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED_VALUE="${SEED_VALUE:-20260511}"

case "$STRATEGY" in
  global)
    LR="${LR:-0.003}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.05}"
    ;;
  layer-band)
    LR="${LR:-0.002}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.03}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.05}"
    ;;
  parameter)
    LR="${LR:-0.001}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.05}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.03}"
    ;;
  global-parameter)
    LR="${LR:-0.0015}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.05}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.03}"
    ;;
  *)
    echo "[error] unknown STRATEGY=$STRATEGY" >&2
    exit 2
    ;;
esac

UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-1}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-180GiB}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

OBJECTIVE_ARGS=()
if [[ "$TASK_NORMALIZE_ADVANTAGES" == "1" || "$TASK_NORMALIZE_ADVANTAGES" == "true" || "$TASK_NORMALIZE_ADVANTAGES" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--task-normalize-advantages)
fi
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--length-normalize-policy-logprob)
fi
GRADIENT_ARGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" || "$GRADIENT_CHECKPOINTING" == "yes" ]]; then
  GRADIENT_ARGS+=(--gradient-checkpointing)
fi
DRY_ARGS=()
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  DRY_ARGS+=(--dry-run)
fi

mkdir -p "$RUN_DIR" "$(dirname "$INIT_GATE")"

echo "[run] strategy=$STRATEGY init=$INIT_VALUE"
echo "[run] calibration=$CALIBRATION"
echo "[run] run_dir=$RUN_DIR"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] lr=$LR prior=$PRIOR_LOSS_WEIGHT max_delta=$MAX_COEFF_DELTA"

"$PY" scripts/modes/build_constant_gate_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --gate-parameterization "$STRATEGY" \
  --value "$INIT_VALUE" \
  --output "$INIT_GATE" >/dev/null

"$PY" scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --seed-manifest "$CALIBRATION" \
  --run-dir "$RUN_DIR" \
  --run-id "$RUN_NAME" \
  --num-iters "$NUM_ITERS" \
  --num-prompts "$NUM_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --use-manifest-order \
  --gate-parameterization "$STRATEGY" \
  --init-gate-checkpoint "$INIT_GATE" \
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
  --update-epochs "$UPDATE_EPOCHS" \
  --lr "$LR" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --max-coefficient-delta-from-init "$MAX_COEFF_DELTA" \
  --behavior-span-reward-weight 0.0 \
  --frontier-task-quota tool=32 \
  --frontier-task-quota memory=32 \
  --frontier-task-quota code=32 \
  --task-weight tool=1.0 \
  --task-weight memory=1.0 \
  --task-weight code=1.0 \
  --device-map auto \
  "${MAX_MEMORY_ARGS[@]}" \
  "${GRADIENT_ARGS[@]}" \
  "${OBJECTIVE_ARGS[@]}" \
  "${DRY_ARGS[@]}" \
  --progress-every 10

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "yes" ]]; then
  "$PY" scripts/eval/summarize_gate_strategy_run.py \
    --run-dir "$RUN_DIR" \
    --strategy "$STRATEGY" \
    --init-value "$INIT_VALUE" \
    --output "$RUN_DIR/strategy_summary.json" >/dev/null
  echo "[done] summary: $RUN_DIR/strategy_summary.json"
fi
