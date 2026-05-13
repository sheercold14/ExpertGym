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

Common overrides:
  RUN_NAME=...                         default qbank_c033333_<strategy>_i2_seed20260511
  RUN_DIR=...                          default $ROOT/runs/gated_grpo/$RUN_NAME
  CALIBRATION=...                      default $QB/calibration/calib100_seed20260511.prompts.jsonl

Gate strategy:
  STRATEGY=global|layer-band|parameter|global-parameter
  INIT_VALUE=0.3333333333333333        initial task-vector coefficient
  MAX_GATED_MODULES=                   empty means all modules; use 1 only for smoke tests

Data / loop:
  NUM_ITERS=20
  NUM_PROMPTS=100
  SAMPLES_PER_PROMPT=4
  TASKS=                               optional comma list: tool,memory,code
  MEMORY_KIND=                         optional memory filter
  PROMPT_ID=                           optional single prompt id; incompatible with multi-shard rollout

Rollout / vLLM:
  GPU_LIST=0,1,2,3
  ROLLOUT_GPUS=$GPU_LIST
  ROLLOUT_SHARDS=auto                  auto uses one single-GPU vLLM worker per rollout GPU
  ROLLOUT_SHARD_STAGGER_SECONDS=0
  ROLLOUT_BATCH_SIZE=32
  TENSOR_PARALLEL_SIZE=1
  GPU_MEMORY_UTILIZATION=0.82
  MAX_NEW_TOKENS=1024
  TOOL_MAX_NEW_TOKENS=512
  CODE_MAX_NEW_TOKENS=4096
  MEMORY_UPDATE_MAX_NEW_TOKENS=2048
  MEMORY_FINAL_MAX_NEW_TOKENS=2048
  MAX_PROMPT_TOKENS=8192
  MAX_MODEL_LEN=12288
  MAX_LOGPROB_TOKENS=$MAX_MODEL_LEN
  TEMPERATURE=0.7
  TOP_P=0.95
  GREEDY=0|1
  SEED_VALUE=20260512

Update / objective:
  UPDATE_EPOCHS=1
  UPDATE_BATCH_SIZE=4
  BATCH_LOSS_REDUCTION=mean|sum
  LOSS_GRANULARITY=token|sequence
  STORE_TOKEN_LOGPROBS=0|1|auto        recommended 0; avoids vLLM-old/HF-current mismatch
  TASK_NORMALIZE_ADVANTAGES=0|1          default 0; keep per-prompt GRPO normalization, do not rescale across tasks
  ADVANTAGE_NORMALIZATION=centered|zscore default centered; centered subtracts row mean without dividing by row std
  USE_FRONTIER_WEIGHT=0|1                default 0; keep frontier filtering but do not scale advantages by frontier weight
  LENGTH_NORMALIZE_POLICY_LOGPROB=0|1
  LENGTH_NORMALIZE_LOGPROB=0|1
  LR=                                  default depends on STRATEGY
  PRIOR_LOSS_WEIGHT=                   default depends on STRATEGY
  PPO_LOSS_WEIGHT=1.0
  BEST_RESPONSE_LOSS_WEIGHT=0.0
  PAIRWISE_LOSS_WEIGHT=0.0
  PAIRWISE_MARGIN=0.0
  MAX_PAIRWISE_PAIRS_PER_ROW=0
  MIN_GRAD_NORM_FOR_STEP=0.0
  MAX_COEFF_DELTA=                     default depends on STRATEGY

Frontier / task balance:
  FRONTIER_ORDER=as-is|shuffle|task-interleaved
  FRONTIER_SHUFFLE_SEED=               empty means seed + iteration - 1
  FRONTIER_TOOL_QUOTA=32
  FRONTIER_MEMORY_QUOTA=32
  FRONTIER_CODE_QUOTA=32
  MAX_FRONTIER_ROWS_PER_TASK=
  USE_RETENTION=0|1                   default 0; all-success rows become KL retention rows when enabled
  RETENTION_LOSS_WEIGHT=              recommended 0.05 when USE_RETENTION=1
  MAX_RETENTION_ROWS=                 recommended 64 when USE_RETENTION=1
  TASK_WEIGHT_TOOL=1.0
  TASK_WEIGHT_MEMORY=1.0
  TASK_WEIGHT_CODE=1.0
  ADVANTAGE_FIELD=                     e.g. reward_delta_vs_baseline
  ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT=0|1

Constraints / extras:
  TRAIN_COEFFICIENTS=                  e.g. global.tool,global.memory
  TOOL_MIN_MARGIN_OVER_MEMORY=0.0
  TOOL_MIN_MARGIN_OVER_CODE=0.0
  POSITIVE_REWARD_THRESHOLD=
  BEHAVIOR_SPAN_REWARD_WEIGHT=0.0
  RECOMPUTE_FRONTIER=0|1

System:
  PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
  ROOT=/tmp/shared-storage/OnPolicy
  DEVICE=cuda
  DEVICE_MAP=auto
  TORCH_DTYPE=bfloat16
  GRADIENT_CHECKPOINTING=0|1
  MAX_MEMORY_PER_GPU=70GiB
  CPU_MAX_MEMORY=180GiB
  PROGRESS_EVERY=10
  DRY_RUN=0|1

Recommended safe start:
  DRY_RUN=1 STRATEGY=global STORE_TOKEN_LOGPROBS=0 FRONTIER_ORDER=task-interleaved \
    bash skill/command/run_qbank_c033333_gate_strategy.sh
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

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

GPU_LIST="${GPU_LIST:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

NUM_ITERS="${NUM_ITERS:-20}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
TASKS="${TASKS:-}"
MEMORY_KIND="${MEMORY_KIND:-}"
PROMPT_ID="${PROMPT_ID:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOOL_MAX_NEW_TOKENS="${TOOL_MAX_NEW_TOKENS:-512}"
CODE_MAX_NEW_TOKENS="${CODE_MAX_NEW_TOKENS:-4096}"
MEMORY_UPDATE_MAX_NEW_TOKENS="${MEMORY_UPDATE_MAX_NEW_TOKENS:-2048}"
MEMORY_FINAL_MAX_NEW_TOKENS="${MEMORY_FINAL_MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-$MAX_MODEL_LEN}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
ROLLOUT_SHARDS="${ROLLOUT_SHARDS:-auto}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$GPU_LIST}"
ROLLOUT_SHARD_STAGGER_SECONDS="${ROLLOUT_SHARD_STAGGER_SECONDS:-0}"
POST_BAKE_SLEEP_SECONDS="${POST_BAKE_SLEEP_SECONDS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
GREEDY="${GREEDY:-0}"
SEED_VALUE="${SEED_VALUE:-20260512}"

case "$STRATEGY" in
  global)
    LR="${LR:-0.03}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.01}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.2}"
    ;;
  layer-band)
    LR="${LR:-0.02}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.1}"
    ;;
  parameter)
    LR="${LR:-0.01}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.1}"
    ;;
  global-parameter)
    LR="${LR:-0.015}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.1}"
    ;;
  *)
    echo "[error] unknown STRATEGY=$STRATEGY" >&2
    exit 2
    ;;
esac

UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
UPDATE_BATCH_SIZE="${UPDATE_BATCH_SIZE:-4}"
BATCH_LOSS_REDUCTION="${BATCH_LOSS_REDUCTION:-mean}"
LOSS_GRANULARITY="${LOSS_GRANULARITY:-token}"
FRONTIER_ORDER="${FRONTIER_ORDER:-task-interleaved}"
FRONTIER_SHUFFLE_SEED="${FRONTIER_SHUFFLE_SEED:-}"
STORE_TOKEN_LOGPROBS="${STORE_TOKEN_LOGPROBS:-0}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-0}"
ADVANTAGE_NORMALIZATION="${ADVANTAGE_NORMALIZATION:-centered}"
USE_FRONTIER_WEIGHT="${USE_FRONTIER_WEIGHT:-0}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-1}"
LENGTH_NORMALIZE_LOGPROB="${LENGTH_NORMALIZE_LOGPROB:-0}"
PPO_LOSS_WEIGHT="${PPO_LOSS_WEIGHT:-1.0}"
BEST_RESPONSE_LOSS_WEIGHT="${BEST_RESPONSE_LOSS_WEIGHT:-0.0}"
PAIRWISE_LOSS_WEIGHT="${PAIRWISE_LOSS_WEIGHT:-0.0}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.0}"
MAX_PAIRWISE_PAIRS_PER_ROW="${MAX_PAIRWISE_PAIRS_PER_ROW:-0}"
MIN_GRAD_NORM_FOR_STEP="${MIN_GRAD_NORM_FOR_STEP:-0.0}"
BEHAVIOR_SPAN_REWARD_WEIGHT="${BEHAVIOR_SPAN_REWARD_WEIGHT:-0.0}"
FRONTIER_TOOL_QUOTA="${FRONTIER_TOOL_QUOTA:-32}"
FRONTIER_MEMORY_QUOTA="${FRONTIER_MEMORY_QUOTA:-32}"
FRONTIER_CODE_QUOTA="${FRONTIER_CODE_QUOTA:-32}"
MAX_FRONTIER_ROWS_PER_TASK="${MAX_FRONTIER_ROWS_PER_TASK:-}"
USE_RETENTION="${USE_RETENTION:-0}"
RETENTION_LOSS_WEIGHT="${RETENTION_LOSS_WEIGHT:-}"
MAX_RETENTION_ROWS="${MAX_RETENTION_ROWS:-}"
TASK_WEIGHT_TOOL="${TASK_WEIGHT_TOOL:-1.0}"
TASK_WEIGHT_MEMORY="${TASK_WEIGHT_MEMORY:-1.0}"
TASK_WEIGHT_CODE="${TASK_WEIGHT_CODE:-1.0}"
ADVANTAGE_FIELD="${ADVANTAGE_FIELD:-}"
ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT="${ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT:-0}"
TRAIN_COEFFICIENTS="${TRAIN_COEFFICIENTS:-}"
TOOL_MIN_MARGIN_OVER_MEMORY="${TOOL_MIN_MARGIN_OVER_MEMORY:-0.0}"
TOOL_MIN_MARGIN_OVER_CODE="${TOOL_MIN_MARGIN_OVER_CODE:-0.0}"
POSITIVE_REWARD_THRESHOLD="${POSITIVE_REWARD_THRESHOLD:-}"
RECOMPUTE_FRONTIER="${RECOMPUTE_FRONTIER:-0}"
MAX_GATED_MODULES="${MAX_GATED_MODULES:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}"
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-180GiB}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
DRY_RUN="${DRY_RUN:-0}"

case "$BATCH_LOSS_REDUCTION" in
  mean|sum) ;;
  *) echo "[error] BATCH_LOSS_REDUCTION must be mean or sum, got $BATCH_LOSS_REDUCTION" >&2; exit 2 ;;
esac
case "$LOSS_GRANULARITY" in
  token|sequence) ;;
  *) echo "[error] LOSS_GRANULARITY must be token or sequence, got $LOSS_GRANULARITY" >&2; exit 2 ;;
esac
case "$FRONTIER_ORDER" in
  as-is|shuffle|task-interleaved) ;;
  *) echo "[error] FRONTIER_ORDER must be as-is, shuffle, or task-interleaved, got $FRONTIER_ORDER" >&2; exit 2 ;;
esac
case "$STORE_TOKEN_LOGPROBS" in
  0|1|true|false|yes|no|auto) ;;
  *) echo "[error] STORE_TOKEN_LOGPROBS must be 0, 1, true, false, yes, no, or auto; got $STORE_TOKEN_LOGPROBS" >&2; exit 2 ;;
esac
case "$ADVANTAGE_NORMALIZATION" in
  centered|zscore) ;;
  *) echo "[error] ADVANTAGE_NORMALIZATION must be centered or zscore, got $ADVANTAGE_NORMALIZATION" >&2; exit 2 ;;
esac

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
MAX_MEMORY_ARGS=()
for gpu_index in "${!VISIBLE_GPU_LIST[@]}"; do
  MAX_MEMORY_ARGS+=(--max-memory "${gpu_index}=${MAX_MEMORY_PER_GPU}")
done
MAX_MEMORY_ARGS+=(--max-memory "cpu=${CPU_MAX_MEMORY}")

FILTER_ARGS=()
if [[ -n "$TASKS" ]]; then
  FILTER_ARGS+=(--tasks "$TASKS")
fi
if [[ -n "$MEMORY_KIND" ]]; then
  FILTER_ARGS+=(--memory-kind "$MEMORY_KIND")
fi
if [[ -n "$PROMPT_ID" ]]; then
  FILTER_ARGS+=(--prompt-id "$PROMPT_ID")
fi
if is_truthy "$GREEDY"; then
  FILTER_ARGS+=(--greedy)
fi

OBJECTIVE_ARGS=()
if [[ "$TASK_NORMALIZE_ADVANTAGES" == "1" || "$TASK_NORMALIZE_ADVANTAGES" == "true" || "$TASK_NORMALIZE_ADVANTAGES" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--task-normalize-advantages)
fi
OBJECTIVE_ARGS+=(--advantage-normalization "$ADVANTAGE_NORMALIZATION")
if is_truthy "$USE_FRONTIER_WEIGHT"; then
  OBJECTIVE_ARGS+=(--use-frontier-weight)
fi
if is_truthy "$LENGTH_NORMALIZE_LOGPROB"; then
  OBJECTIVE_ARGS+=(--length-normalize-logprob)
fi
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--length-normalize-policy-logprob)
fi
if [[ "$STORE_TOKEN_LOGPROBS" == "auto" ]]; then
  if [[ "$LOSS_GRANULARITY" == "token" ]]; then
    OBJECTIVE_ARGS+=(--store-token-logprobs)
  fi
elif [[ "$STORE_TOKEN_LOGPROBS" == "1" || "$STORE_TOKEN_LOGPROBS" == "true" || "$STORE_TOKEN_LOGPROBS" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--store-token-logprobs)
fi
GRADIENT_ARGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" || "$GRADIENT_CHECKPOINTING" == "yes" ]]; then
  GRADIENT_ARGS+=(--gradient-checkpointing)
fi
FRONTIER_ORDER_ARGS=(--frontier-order "$FRONTIER_ORDER")
if [[ -n "$FRONTIER_SHUFFLE_SEED" ]]; then
  FRONTIER_ORDER_ARGS+=(--frontier-shuffle-seed "$FRONTIER_SHUFFLE_SEED")
fi

UPDATE_EXTRA_ARGS=(
  --ppo-loss-weight "$PPO_LOSS_WEIGHT"
  --best-response-loss-weight "$BEST_RESPONSE_LOSS_WEIGHT"
  --pairwise-loss-weight "$PAIRWISE_LOSS_WEIGHT"
  --pairwise-margin "$PAIRWISE_MARGIN"
  --max-pairwise-pairs-per-row "$MAX_PAIRWISE_PAIRS_PER_ROW"
  --min-grad-norm-for-step "$MIN_GRAD_NORM_FOR_STEP"
)
if [[ -n "$MAX_GATED_MODULES" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-gated-modules "$MAX_GATED_MODULES")
fi
if [[ -n "$MAX_FRONTIER_ROWS_PER_TASK" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-frontier-rows-per-task "$MAX_FRONTIER_ROWS_PER_TASK")
fi
if is_truthy "$USE_RETENTION"; then
  UPDATE_EXTRA_ARGS+=(--use-retention)
fi
if [[ -n "$RETENTION_LOSS_WEIGHT" ]]; then
  UPDATE_EXTRA_ARGS+=(--retention-loss-weight "$RETENTION_LOSS_WEIGHT")
fi
if [[ -n "$MAX_RETENTION_ROWS" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-retention-rows "$MAX_RETENTION_ROWS")
fi
if [[ -n "$ADVANTAGE_FIELD" ]]; then
  UPDATE_EXTRA_ARGS+=(--advantage-field "$ADVANTAGE_FIELD")
  if is_truthy "$ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT"; then
    UPDATE_EXTRA_ARGS+=(--advantage-field-frontier-weight)
  fi
  if ! is_truthy "$ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT"; then
    UPDATE_EXTRA_ARGS+=(--no-advantage-field-frontier-weight)
  fi
fi
if [[ -n "$TRAIN_COEFFICIENTS" ]]; then
  UPDATE_EXTRA_ARGS+=(--train-coefficient "$TRAIN_COEFFICIENTS")
fi
if [[ "$TOOL_MIN_MARGIN_OVER_MEMORY" != "0" && "$TOOL_MIN_MARGIN_OVER_MEMORY" != "0.0" ]]; then
  UPDATE_EXTRA_ARGS+=(--tool-min-margin-over-memory "$TOOL_MIN_MARGIN_OVER_MEMORY")
fi
if [[ "$TOOL_MIN_MARGIN_OVER_CODE" != "0" && "$TOOL_MIN_MARGIN_OVER_CODE" != "0.0" ]]; then
  UPDATE_EXTRA_ARGS+=(--tool-min-margin-over-code "$TOOL_MIN_MARGIN_OVER_CODE")
fi
if [[ -n "$POSITIVE_REWARD_THRESHOLD" ]]; then
  UPDATE_EXTRA_ARGS+=(--positive-reward-threshold "$POSITIVE_REWARD_THRESHOLD")
fi
if is_truthy "$RECOMPUTE_FRONTIER"; then
  UPDATE_EXTRA_ARGS+=(--recompute-frontier)
fi

FRONTIER_QUOTA_ARGS=()
if [[ -n "$FRONTIER_TOOL_QUOTA" ]]; then
  FRONTIER_QUOTA_ARGS+=(--frontier-task-quota "tool=$FRONTIER_TOOL_QUOTA")
fi
if [[ -n "$FRONTIER_MEMORY_QUOTA" ]]; then
  FRONTIER_QUOTA_ARGS+=(--frontier-task-quota "memory=$FRONTIER_MEMORY_QUOTA")
fi
if [[ -n "$FRONTIER_CODE_QUOTA" ]]; then
  FRONTIER_QUOTA_ARGS+=(--frontier-task-quota "code=$FRONTIER_CODE_QUOTA")
fi

TASK_WEIGHT_ARGS=(
  --task-weight "tool=$TASK_WEIGHT_TOOL"
  --task-weight "memory=$TASK_WEIGHT_MEMORY"
  --task-weight "code=$TASK_WEIGHT_CODE"
)

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
echo "[run] loss_granularity=$LOSS_GRANULARITY update_batch_size=$UPDATE_BATCH_SIZE store_token_logprobs=$STORE_TOKEN_LOGPROBS"
echo "[run] task_normalize_advantages=$TASK_NORMALIZE_ADVANTAGES advantage_normalization=$ADVANTAGE_NORMALIZATION use_frontier_weight=$USE_FRONTIER_WEIGHT advantage_field_frontier_weight=$ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT"
echo "[run] frontier_order=$FRONTIER_ORDER frontier_shuffle_seed=${FRONTIER_SHUFFLE_SEED:-auto}"
echo "[run] task_weights=tool:$TASK_WEIGHT_TOOL,memory:$TASK_WEIGHT_MEMORY,code:$TASK_WEIGHT_CODE quotas=tool:$FRONTIER_TOOL_QUOTA,memory:$FRONTIER_MEMORY_QUOTA,code:$FRONTIER_CODE_QUOTA"
echo "[run] retention=$USE_RETENTION retention_loss_weight=${RETENTION_LOSS_WEIGHT:-none} max_retention_rows=${MAX_RETENTION_ROWS:-none}"
echo "[run] tokens=default:$MAX_NEW_TOKENS,tool:$TOOL_MAX_NEW_TOKENS,code:$CODE_MAX_NEW_TOKENS,memory_update:$MEMORY_UPDATE_MAX_NEW_TOKENS,memory_final:$MEMORY_FINAL_MAX_NEW_TOKENS"
echo "[run] rollout_shards=$ROLLOUT_SHARDS rollout_gpus=$ROLLOUT_GPUS vllm_batch_size=$ROLLOUT_BATCH_SIZE"

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
  "${FILTER_ARGS[@]}" \
  --run-dir "$RUN_DIR" \
  --run-id "$RUN_NAME" \
  --num-iters "$NUM_ITERS" \
  --num-prompts "$NUM_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --use-manifest-order \
  --gate-parameterization "$STRATEGY" \
  --init-gate-checkpoint "$INIT_GATE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --tool-max-new-tokens "$TOOL_MAX_NEW_TOKENS" \
  --code-max-new-tokens "$CODE_MAX_NEW_TOKENS" \
  --memory-update-max-new-tokens "$MEMORY_UPDATE_MAX_NEW_TOKENS" \
  --memory-final-max-new-tokens "$MEMORY_FINAL_MAX_NEW_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --max-logprob-tokens "$MAX_LOGPROB_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --vllm-batch-size "$ROLLOUT_BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --rollout-shards "$ROLLOUT_SHARDS" \
  --rollout-gpus "$ROLLOUT_GPUS" \
  --rollout-shard-stagger-seconds "$ROLLOUT_SHARD_STAGGER_SECONDS" \
  --post-bake-sleep-seconds "$POST_BAKE_SLEEP_SECONDS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "$SEED_VALUE" \
  --update-epochs "$UPDATE_EPOCHS" \
  --update-batch-size "$UPDATE_BATCH_SIZE" \
  --batch-loss-reduction "$BATCH_LOSS_REDUCTION" \
  --loss-granularity "$LOSS_GRANULARITY" \
  "${FRONTIER_ORDER_ARGS[@]}" \
  "${UPDATE_EXTRA_ARGS[@]}" \
  --lr "$LR" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --max-coefficient-delta-from-init "$MAX_COEFF_DELTA" \
  --behavior-span-reward-weight "$BEHAVIOR_SPAN_REWARD_WEIGHT" \
  "${FRONTIER_QUOTA_ARGS[@]}" \
  "${TASK_WEIGHT_ARGS[@]}" \
  --device "$DEVICE" \
  --device-map "$DEVICE_MAP" \
  --torch-dtype "$TORCH_DTYPE" \
  "${MAX_MEMORY_ARGS[@]}" \
  "${GRADIENT_ARGS[@]}" \
  "${OBJECTIVE_ARGS[@]}" \
  "${DRY_ARGS[@]}" \
  --progress-every "$PROGRESS_EVERY"

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "yes" ]]; then
  "$PY" scripts/eval/summarize_gate_strategy_run.py \
    --run-dir "$RUN_DIR" \
    --strategy "$STRATEGY" \
    --init-value "$INIT_VALUE" \
    --output "$RUN_DIR/strategy_summary.json" >/dev/null
  echo "[done] summary: $RUN_DIR/strategy_summary.json"
fi
