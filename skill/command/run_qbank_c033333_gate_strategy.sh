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
  CONFIG=configs/gated_grpo.yaml

Gate strategy:
  STRATEGY=global|global-coefficient|layer-band|layer-band-coefficient|layer-band-parameter|parameter|global-parameter
  INIT_VALUE=0.3333333333333333        initial task-vector coefficient
  INIT_GATE_CHECKPOINT=                optional explicit gate JSON; when set, skip constant init creation
  MAX_GATED_MODULES=                   empty means all modules; use 1 only for smoke tests

Data / loop:
  NUM_ITERS=20
  START_ITERATION=1
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
  OPTIMIZER_STEP_SCOPE=batch|epoch       batch steps every UPDATE_BATCH_SIZE rows; epoch accumulates all rows then steps once
  LOSS_GRANULARITY=token|sequence
  STORE_TOKEN_LOGPROBS=0|1|auto        recommended 0; avoids vLLM-old/HF-current mismatch
  TASK_NORMALIZE_ADVANTAGES=0|1          default 0; keep per-prompt GRPO normalization, do not rescale across tasks
  ADVANTAGE_NORMALIZATION=centered|zscore default centered; centered subtracts row mean without dividing by row std
  USE_FRONTIER_WEIGHT=0|1                default 0; keep frontier filtering but do not scale advantages by frontier weight
  LENGTH_NORMALIZE_POLICY_LOGPROB=0|1
  LENGTH_NORMALIZE_LOGPROB=0|1
  OPD_LENGTH_NORMALIZE_LOGPROB=inherit|0|1
  RETENTION_LENGTH_NORMALIZE_LOGPROB=inherit|0|1
  RETENTION_DYNAMIC_SCALE=0|1           auto-scale retention NLL per task to a fixed GRPO-relative target
  RETENTION_TASK_BALANCED_LOSS_SCALE=0|1 average retention by task instead of raw row count
  RETENTION_SCALE_TARGET=0.5             target retention/GRPO ratio when retention rows exist
  PCGRAD_GATE_GRADIENTS=0|1              enable optional task PCGrad for gate gradients; requires OPTIMIZER_STEP_SCOPE=epoch
  PCGRAD_EPS=1e-12
  PCGRAD_TASKS=                          optional comma list: tool,memory,code
  TOOL_NULLSPACE_GATE_GRADIENTS=0|1      project total gate gradient away from Tool behavior-span gradients
  TOOL_NULLSPACE_REPLAY_ROLLOUT=         optional comma-separated Tool positive rollout JSONL paths
  TOOL_NULLSPACE_ROWS=16
  TOOL_NULLSPACE_MIN_ROWS=1
  TOOL_NULLSPACE_RANK=0                  0 uses all numerical ranks
  TOOL_NULLSPACE_EPS=1e-6
  OPD_DYNAMIC_SCALE=0|1                  auto-scale OPD per task from current OPD loss magnitudes
  OPD_TASK_BALANCED_LOSS_SCALE=0|1       average OPD by task instead of raw row count
  OPD_SCALE_TARGET_HIGH/MID/LOW/TAIL     target OPD/GRPO ratios by recoverable all-fail rate
  LR=                                  default depends on STRATEGY
  OPTIMIZER=adamw|sgd
  SGD_MOMENTUM=0.0
  SGD_NESTEROV=0|1
  PERSIST_OPTIMIZER_STATE=0|1
  OPTIMIZER_STATE_CHECKPOINT=
  PRIOR_LOSS_WEIGHT=                   default depends on STRATEGY
  PPO_LOSS_WEIGHT=1.0
  BEST_RESPONSE_LOSS_WEIGHT=0.0
  PAIRWISE_LOSS_WEIGHT=0.0
  PAIRWISE_MARGIN=0.0
  MAX_PAIRWISE_PAIRS_PER_ROW=0
  MIN_GRAD_NORM_FOR_STEP=0.0
  MAX_COEFF_DELTA=                     default depends on STRATEGY
  MAX_COEFF_DELTA_BY_EXPERT=           optional comma list, e.g. reasoning=0.002
  COEFF_BOUND_BY_EXPERT=               optional comma list, e.g. reasoning=0.0:0.003

Frontier / task balance:
  FRONTIER_ORDER=as-is|shuffle|task-interleaved
  FRONTIER_SHUFFLE_SEED=               empty means seed + iteration - 1
  FRONTIER_SAMPLE_BEFORE_LIMIT=0|1     randomly sample before applying frontier quotas
  IGNORE_CONFIG_FRONTIER_TASK_QUOTA=1  default 1; unset CLI quota means all frontier rows
  FRONTIER_ROWS_PER_TASK=              optional shorthand, e.g. 4 sets all three task quotas
  FRONTIER_TOOL_QUOTA=                 empty means no tool cap
  FRONTIER_MEMORY_QUOTA=               empty means no memory cap
  FRONTIER_CODE_QUOTA=                 empty means no code cap
  MAX_FRONTIER_ROWS_PER_TASK=
  USE_RETENTION=0|1                   default 0; all-success rows become KL retention rows when enabled
  RETENTION_OBJECTIVE=kl|nll           kl is legacy; nll preserves all-success rows with non-zero NLL gradient
  RETENTION_LOSS_WEIGHT=              recommended 0.05 when USE_RETENTION=1
  RETENTION_POSITIVE_REWARD_THRESHOLD=1.0
  MAX_RETENTION_ROWS=                 recommended 64 when USE_RETENTION=1
  MAX_RETENTION_ROWS_PER_TASK=        optional per-task cap before MAX_RETENTION_ROWS
  RETENTION_SAMPLE_BEFORE_LIMIT=0|1   randomly sample before applying retention caps
  RETENTION_SHUFFLE_SEED=             empty means use frontier shuffle seed
  OPD_DISTILL_ROLLOUT=                optional comma-separated OPD distill JSONL paths
  OPD_LOSS_WEIGHT=0.0                 sequence expert-positive likelihood loss for OPD rows
  OPD_PAIRWISE_LOSS_WEIGHT=0.0        pairwise expert-positive vs current-negative loss for OPD rows
  OPD_PAIRWISE_MARGIN=0.0
  OPD_POSITIVE_REWARD_THRESHOLD=      empty means best reward_train in each OPD row
  MAX_OPD_DISTILL_ROWS=
  MAX_OPD_PAIRWISE_PAIRS_PER_ROW=0
  DYNAMIC_OPD_EXPERT_ROLLOUT=        optional comma-separated expert rollout JSONL paths; each iter selects current all-fail prompts
  DYNAMIC_OPD_TASKS=tool,memory,code
  DYNAMIC_OPD_PER_TASK=32
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2
  DYNAMIC_OPD_REQUIRE_ALL_TASKS=0|1 skip dynamic OPD for an update unless every dynamic OPD task has rows
  USE_OPD_ALL_SUCCESS=0|1             add auxiliary OPD loss on all-success rows
  OPD_ALL_SUCCESS_LOSS_WEIGHT=0.0
  OPD_ALL_SUCCESS_POSITIVE_REWARD_THRESHOLD=1.0
  MAX_OPD_ALL_SUCCESS_ROWS=
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
CONFIG="${CONFIG:-configs/gated_grpo.yaml}"
QB="${QB:-$ROOT/data/question_bank/ta_avgvec_c033333_hotpotqa_v1}"
CALIBRATION="${CALIBRATION:-$QB/calibration/calib100_seed20260511.prompts.jsonl}"
INIT_VALUE="${INIT_VALUE:-0.3333333333333333}"
STRATEGY="${STRATEGY:-global}"
SAFE_STRATEGY="${STRATEGY//[^A-Za-z0-9_]/_}"
RUN_NAME="${RUN_NAME:-qbank_c033333_${SAFE_STRATEGY}_i2_seed20260511}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
INIT_GATE="$QB/init_gates/init_${SAFE_STRATEGY}_c033333.json"
INIT_GATE_CHECKPOINT="${INIT_GATE_CHECKPOINT:-}"
if [[ -n "$INIT_GATE_CHECKPOINT" ]]; then
  INIT_GATE="$INIT_GATE_CHECKPOINT"
fi

GPU_LIST="${GPU_LIST:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

NUM_ITERS="${NUM_ITERS:-20}"
START_ITERATION="${START_ITERATION:-1}"
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
  global-coefficient)
    LR="${LR:-0.03}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.01}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.2}"
    ;;
  layer-band)
    LR="${LR:-0.02}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.1}"
    ;;
  layer-band-coefficient)
    LR="${LR:-0.02}"
    PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.02}"
    MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.1}"
    ;;
  layer-band-parameter)
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
OPTIMIZER_STEP_SCOPE="${OPTIMIZER_STEP_SCOPE:-batch}"
LOSS_GRANULARITY="${LOSS_GRANULARITY:-token}"
FRONTIER_ORDER="${FRONTIER_ORDER:-task-interleaved}"
FRONTIER_SHUFFLE_SEED="${FRONTIER_SHUFFLE_SEED:-}"
FRONTIER_SAMPLE_BEFORE_LIMIT="${FRONTIER_SAMPLE_BEFORE_LIMIT:-0}"
IGNORE_CONFIG_FRONTIER_TASK_QUOTA="${IGNORE_CONFIG_FRONTIER_TASK_QUOTA:-1}"
STORE_TOKEN_LOGPROBS="${STORE_TOKEN_LOGPROBS:-0}"
TASK_NORMALIZE_ADVANTAGES="${TASK_NORMALIZE_ADVANTAGES:-0}"
ADVANTAGE_NORMALIZATION="${ADVANTAGE_NORMALIZATION:-centered}"
USE_FRONTIER_WEIGHT="${USE_FRONTIER_WEIGHT:-0}"
LENGTH_NORMALIZE_POLICY_LOGPROB="${LENGTH_NORMALIZE_POLICY_LOGPROB:-1}"
LENGTH_NORMALIZE_LOGPROB="${LENGTH_NORMALIZE_LOGPROB:-0}"
OPD_LENGTH_NORMALIZE_LOGPROB="${OPD_LENGTH_NORMALIZE_LOGPROB:-inherit}"
RETENTION_LENGTH_NORMALIZE_LOGPROB="${RETENTION_LENGTH_NORMALIZE_LOGPROB:-inherit}"
RETENTION_DYNAMIC_SCALE="${RETENTION_DYNAMIC_SCALE:-0}"
RETENTION_TASK_BALANCED_LOSS_SCALE="${RETENTION_TASK_BALANCED_LOSS_SCALE:-0}"
RETENTION_SCALE_TARGET="${RETENTION_SCALE_TARGET:-0.5}"
RETENTION_SCALE_MIN="${RETENTION_SCALE_MIN:-0.05}"
RETENTION_SCALE_MAX="${RETENTION_SCALE_MAX:-100.0}"
RETENTION_SCALE_EPS="${RETENTION_SCALE_EPS:-1e-6}"
PCGRAD_GATE_GRADIENTS="${PCGRAD_GATE_GRADIENTS:-0}"
PCGRAD_EPS="${PCGRAD_EPS:-1e-12}"
PCGRAD_TASKS="${PCGRAD_TASKS:-}"
TOOL_NULLSPACE_GATE_GRADIENTS="${TOOL_NULLSPACE_GATE_GRADIENTS:-0}"
TOOL_NULLSPACE_REPLAY_ROLLOUT="${TOOL_NULLSPACE_REPLAY_ROLLOUT:-}"
TOOL_NULLSPACE_ROWS="${TOOL_NULLSPACE_ROWS:-16}"
TOOL_NULLSPACE_MIN_ROWS="${TOOL_NULLSPACE_MIN_ROWS:-1}"
TOOL_NULLSPACE_RANK="${TOOL_NULLSPACE_RANK:-0}"
TOOL_NULLSPACE_EPS="${TOOL_NULLSPACE_EPS:-1e-6}"
TOOL_NULLSPACE_POSITIVE_REWARD_THRESHOLD="${TOOL_NULLSPACE_POSITIVE_REWARD_THRESHOLD:-1.0}"
OPD_DYNAMIC_SCALE="${OPD_DYNAMIC_SCALE:-0}"
OPD_TASK_BALANCED_LOSS_SCALE="${OPD_TASK_BALANCED_LOSS_SCALE:-0}"
OPD_SCALE_MIN="${OPD_SCALE_MIN:-0.05}"
OPD_SCALE_MAX="${OPD_SCALE_MAX:-100.0}"
OPD_SCALE_RATE_HIGH="${OPD_SCALE_RATE_HIGH:-0.20}"
OPD_SCALE_RATE_MID="${OPD_SCALE_RATE_MID:-0.10}"
OPD_SCALE_RATE_LOW="${OPD_SCALE_RATE_LOW:-0.03}"
OPD_SCALE_TARGET_HIGH="${OPD_SCALE_TARGET_HIGH:-5.0}"
OPD_SCALE_TARGET_MID="${OPD_SCALE_TARGET_MID:-3.0}"
OPD_SCALE_TARGET_LOW="${OPD_SCALE_TARGET_LOW:-1.0}"
OPD_SCALE_TARGET_TAIL="${OPD_SCALE_TARGET_TAIL:-0.33}"
PPO_LOSS_WEIGHT="${PPO_LOSS_WEIGHT:-1.0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SGD_MOMENTUM="${SGD_MOMENTUM:-0.0}"
SGD_NESTEROV="${SGD_NESTEROV:-0}"
PERSIST_OPTIMIZER_STATE="${PERSIST_OPTIMIZER_STATE:-0}"
OPTIMIZER_STATE_CHECKPOINT="${OPTIMIZER_STATE_CHECKPOINT:-}"
BEST_RESPONSE_LOSS_WEIGHT="${BEST_RESPONSE_LOSS_WEIGHT:-0.0}"
PAIRWISE_LOSS_WEIGHT="${PAIRWISE_LOSS_WEIGHT:-0.0}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.0}"
MAX_PAIRWISE_PAIRS_PER_ROW="${MAX_PAIRWISE_PAIRS_PER_ROW:-0}"
MIN_GRAD_NORM_FOR_STEP="${MIN_GRAD_NORM_FOR_STEP:-0.0}"
MAX_COEFF_DELTA_BY_EXPERT="${MAX_COEFF_DELTA_BY_EXPERT:-}"
COEFF_BOUND_BY_EXPERT="${COEFF_BOUND_BY_EXPERT:-}"
BEHAVIOR_SPAN_REWARD_WEIGHT="${BEHAVIOR_SPAN_REWARD_WEIGHT:-0.0}"
FRONTIER_ROWS_PER_TASK="${FRONTIER_ROWS_PER_TASK:-}"
FRONTIER_TOOL_QUOTA="${FRONTIER_TOOL_QUOTA:-$FRONTIER_ROWS_PER_TASK}"
FRONTIER_MEMORY_QUOTA="${FRONTIER_MEMORY_QUOTA:-$FRONTIER_ROWS_PER_TASK}"
FRONTIER_CODE_QUOTA="${FRONTIER_CODE_QUOTA:-$FRONTIER_ROWS_PER_TASK}"
MAX_FRONTIER_ROWS_PER_TASK="${MAX_FRONTIER_ROWS_PER_TASK:-}"
USE_RETENTION="${USE_RETENTION:-0}"
RETENTION_OBJECTIVE="${RETENTION_OBJECTIVE:-kl}"
RETENTION_LOSS_WEIGHT="${RETENTION_LOSS_WEIGHT:-}"
RETENTION_POSITIVE_REWARD_THRESHOLD="${RETENTION_POSITIVE_REWARD_THRESHOLD:-1.0}"
MAX_RETENTION_ROWS="${MAX_RETENTION_ROWS:-}"
MAX_RETENTION_ROWS_PER_TASK="${MAX_RETENTION_ROWS_PER_TASK:-}"
RETENTION_SAMPLE_BEFORE_LIMIT="${RETENTION_SAMPLE_BEFORE_LIMIT:-0}"
RETENTION_SHUFFLE_SEED="${RETENTION_SHUFFLE_SEED:-}"
OPD_DISTILL_ROLLOUT="${OPD_DISTILL_ROLLOUT:-}"
OPD_LOSS_WEIGHT="${OPD_LOSS_WEIGHT:-0.0}"
OPD_PAIRWISE_LOSS_WEIGHT="${OPD_PAIRWISE_LOSS_WEIGHT:-0.0}"
OPD_PAIRWISE_MARGIN="${OPD_PAIRWISE_MARGIN:-0.0}"
OPD_POSITIVE_REWARD_THRESHOLD="${OPD_POSITIVE_REWARD_THRESHOLD:-}"
MAX_OPD_DISTILL_ROWS="${MAX_OPD_DISTILL_ROWS:-}"
MAX_OPD_PAIRWISE_PAIRS_PER_ROW="${MAX_OPD_PAIRWISE_PAIRS_PER_ROW:-0}"
DYNAMIC_OPD_EXPERT_ROLLOUT="${DYNAMIC_OPD_EXPERT_ROLLOUT:-}"
DYNAMIC_OPD_TASKS="${DYNAMIC_OPD_TASKS:-tool,memory,code}"
DYNAMIC_OPD_KEY="${DYNAMIC_OPD_KEY:-prompt_id}"
DYNAMIC_OPD_CURRENT_MAX_SUCCESS="${DYNAMIC_OPD_CURRENT_MAX_SUCCESS:-0}"
DYNAMIC_OPD_POSITIVE_THRESHOLD="${DYNAMIC_OPD_POSITIVE_THRESHOLD:-1.0}"
DYNAMIC_OPD_MAX_POSITIVES_PER_ROW="${DYNAMIC_OPD_MAX_POSITIVES_PER_ROW:-1}"
DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW="${DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW:-2}"
DYNAMIC_OPD_PER_TASK="${DYNAMIC_OPD_PER_TASK:-32}"
DYNAMIC_OPD_QUOTA="${DYNAMIC_OPD_QUOTA:-}"
DYNAMIC_OPD_REQUIRE_ALL_TASKS="${DYNAMIC_OPD_REQUIRE_ALL_TASKS:-0}"
USE_OPD_ALL_SUCCESS="${USE_OPD_ALL_SUCCESS:-0}"
OPD_ALL_SUCCESS_LOSS_WEIGHT="${OPD_ALL_SUCCESS_LOSS_WEIGHT:-0.0}"
OPD_ALL_SUCCESS_POSITIVE_REWARD_THRESHOLD="${OPD_ALL_SUCCESS_POSITIVE_REWARD_THRESHOLD:-1.0}"
MAX_OPD_ALL_SUCCESS_ROWS="${MAX_OPD_ALL_SUCCESS_ROWS:-}"
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
case "$OPTIMIZER_STEP_SCOPE" in
  batch|epoch) ;;
  *) echo "[error] OPTIMIZER_STEP_SCOPE must be batch or epoch, got $OPTIMIZER_STEP_SCOPE" >&2; exit 2 ;;
esac
case "$LOSS_GRANULARITY" in
  token|sequence) ;;
  *) echo "[error] LOSS_GRANULARITY must be token or sequence, got $LOSS_GRANULARITY" >&2; exit 2 ;;
esac
case "$OPTIMIZER" in
  adamw|sgd) ;;
  *) echo "[error] OPTIMIZER must be adamw or sgd, got $OPTIMIZER" >&2; exit 2 ;;
esac
case "$FRONTIER_ORDER" in
  as-is|shuffle|task-interleaved) ;;
  *) echo "[error] FRONTIER_ORDER must be as-is, shuffle, or task-interleaved, got $FRONTIER_ORDER" >&2; exit 2 ;;
esac
case "$RETENTION_OBJECTIVE" in
  kl|nll) ;;
  *) echo "[error] RETENTION_OBJECTIVE must be kl or nll, got $RETENTION_OBJECTIVE" >&2; exit 2 ;;
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
case "${OPD_LENGTH_NORMALIZE_LOGPROB,,}" in
  inherit|"") ;;
  1|true|yes|y|on) OBJECTIVE_ARGS+=(--opd-length-normalize-logprob) ;;
  0|false|no|n|off) OBJECTIVE_ARGS+=(--no-opd-length-normalize-logprob) ;;
  *) echo "[error] OPD_LENGTH_NORMALIZE_LOGPROB must be inherit, 0, or 1; got $OPD_LENGTH_NORMALIZE_LOGPROB" >&2; exit 2 ;;
esac
case "${RETENTION_LENGTH_NORMALIZE_LOGPROB,,}" in
  inherit|"") ;;
  1|true|yes|y|on) OBJECTIVE_ARGS+=(--retention-length-normalize-logprob) ;;
  0|false|no|n|off) OBJECTIVE_ARGS+=(--no-retention-length-normalize-logprob) ;;
  *) echo "[error] RETENTION_LENGTH_NORMALIZE_LOGPROB must be inherit, 0, or 1; got $RETENTION_LENGTH_NORMALIZE_LOGPROB" >&2; exit 2 ;;
esac
if is_truthy "$RETENTION_DYNAMIC_SCALE"; then
  OBJECTIVE_ARGS+=(--retention-dynamic-scale)
fi
if is_truthy "$RETENTION_TASK_BALANCED_LOSS_SCALE"; then
  OBJECTIVE_ARGS+=(--retention-task-balanced-loss-scale)
fi
OBJECTIVE_ARGS+=(
  --retention-scale-target "$RETENTION_SCALE_TARGET"
  --retention-scale-min "$RETENTION_SCALE_MIN"
  --retention-scale-max "$RETENTION_SCALE_MAX"
  --retention-scale-eps "$RETENTION_SCALE_EPS"
)
if is_truthy "$OPD_DYNAMIC_SCALE"; then
  OBJECTIVE_ARGS+=(--opd-dynamic-scale)
fi
if is_truthy "$OPD_TASK_BALANCED_LOSS_SCALE"; then
  OBJECTIVE_ARGS+=(--opd-task-balanced-loss-scale)
fi
OBJECTIVE_ARGS+=(
  --opd-scale-min "$OPD_SCALE_MIN"
  --opd-scale-max "$OPD_SCALE_MAX"
  --opd-scale-rate-high "$OPD_SCALE_RATE_HIGH"
  --opd-scale-rate-mid "$OPD_SCALE_RATE_MID"
  --opd-scale-rate-low "$OPD_SCALE_RATE_LOW"
  --opd-scale-target-high "$OPD_SCALE_TARGET_HIGH"
  --opd-scale-target-mid "$OPD_SCALE_TARGET_MID"
  --opd-scale-target-low "$OPD_SCALE_TARGET_LOW"
  --opd-scale-target-tail "$OPD_SCALE_TARGET_TAIL"
)
if [[ "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "1" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "true" || "$LENGTH_NORMALIZE_POLICY_LOGPROB" == "yes" ]]; then
  OBJECTIVE_ARGS+=(--length-normalize-policy-logprob)
fi
PCGRAD_ARGS=()
if is_truthy "$PCGRAD_GATE_GRADIENTS"; then
  if [[ "$OPTIMIZER_STEP_SCOPE" != "epoch" ]]; then
    echo "[error] PCGRAD_GATE_GRADIENTS=1 requires OPTIMIZER_STEP_SCOPE=epoch" >&2
    exit 2
  fi
  PCGRAD_ARGS+=(--pcgrad-gate-gradients --pcgrad-eps "$PCGRAD_EPS")
  if [[ -n "$PCGRAD_TASKS" ]]; then
    IFS=',' read -r -a PCGRAD_TASK_LIST <<< "$PCGRAD_TASKS"
    for pcgrad_task in "${PCGRAD_TASK_LIST[@]}"; do
      if [[ -n "$pcgrad_task" ]]; then
        PCGRAD_ARGS+=(--pcgrad-task "$pcgrad_task")
      fi
    done
  fi
fi
TOOL_NULLSPACE_ARGS=()
if is_truthy "$TOOL_NULLSPACE_GATE_GRADIENTS"; then
  TOOL_NULLSPACE_ARGS+=(
    --tool-nullspace-gate-gradients
    --tool-nullspace-rows "$TOOL_NULLSPACE_ROWS"
    --tool-nullspace-min-rows "$TOOL_NULLSPACE_MIN_ROWS"
    --tool-nullspace-rank "$TOOL_NULLSPACE_RANK"
    --tool-nullspace-eps "$TOOL_NULLSPACE_EPS"
    --tool-nullspace-positive-reward-threshold "$TOOL_NULLSPACE_POSITIVE_REWARD_THRESHOLD"
  )
  if [[ -n "$TOOL_NULLSPACE_REPLAY_ROLLOUT" ]]; then
    IFS=',' read -r -a TOOL_NULLSPACE_REPLAY_LIST <<< "$TOOL_NULLSPACE_REPLAY_ROLLOUT"
    for replay_path in "${TOOL_NULLSPACE_REPLAY_LIST[@]}"; do
      if [[ -n "$replay_path" ]]; then
        TOOL_NULLSPACE_ARGS+=(--tool-nullspace-replay-rollout "$replay_path")
      fi
    done
  fi
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
if is_truthy "$FRONTIER_SAMPLE_BEFORE_LIMIT"; then
  FRONTIER_ORDER_ARGS+=(--sample-frontier-before-limit)
fi
if is_truthy "$IGNORE_CONFIG_FRONTIER_TASK_QUOTA"; then
  FRONTIER_ORDER_ARGS+=(--ignore-config-frontier-task-quota)
fi
if is_truthy "$RETENTION_SAMPLE_BEFORE_LIMIT"; then
  FRONTIER_ORDER_ARGS+=(--sample-retention-before-limit)
fi
if [[ -n "$RETENTION_SHUFFLE_SEED" ]]; then
  FRONTIER_ORDER_ARGS+=(--retention-shuffle-seed "$RETENTION_SHUFFLE_SEED")
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
UPDATE_EXTRA_ARGS+=(--retention-objective "$RETENTION_OBJECTIVE")
if [[ -n "$RETENTION_POSITIVE_REWARD_THRESHOLD" ]]; then
  UPDATE_EXTRA_ARGS+=(--retention-positive-reward-threshold "$RETENTION_POSITIVE_REWARD_THRESHOLD")
fi
if [[ -n "$RETENTION_LOSS_WEIGHT" ]]; then
  UPDATE_EXTRA_ARGS+=(--retention-loss-weight "$RETENTION_LOSS_WEIGHT")
fi
if [[ -n "$MAX_RETENTION_ROWS" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-retention-rows "$MAX_RETENTION_ROWS")
fi
if [[ -n "$MAX_RETENTION_ROWS_PER_TASK" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-retention-rows-per-task "$MAX_RETENTION_ROWS_PER_TASK")
fi
if [[ -n "$OPD_DISTILL_ROLLOUT" ]]; then
  IFS=',' read -r -a OPD_ROLLOUT_LIST <<< "$OPD_DISTILL_ROLLOUT"
  for opd_path in "${OPD_ROLLOUT_LIST[@]}"; do
    if [[ -n "$opd_path" ]]; then
      UPDATE_EXTRA_ARGS+=(--opd-distill-rollout "$opd_path")
    fi
  done
fi
if [[ "$OPD_LOSS_WEIGHT" != "0" && "$OPD_LOSS_WEIGHT" != "0.0" ]]; then
  UPDATE_EXTRA_ARGS+=(--opd-loss-weight "$OPD_LOSS_WEIGHT")
fi
if [[ "$OPD_PAIRWISE_LOSS_WEIGHT" != "0" && "$OPD_PAIRWISE_LOSS_WEIGHT" != "0.0" ]]; then
  UPDATE_EXTRA_ARGS+=(--opd-pairwise-loss-weight "$OPD_PAIRWISE_LOSS_WEIGHT")
fi
if [[ "$OPD_PAIRWISE_MARGIN" != "0" && "$OPD_PAIRWISE_MARGIN" != "0.0" ]]; then
  UPDATE_EXTRA_ARGS+=(--opd-pairwise-margin "$OPD_PAIRWISE_MARGIN")
fi
if [[ -n "$MAX_OPD_DISTILL_ROWS" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-opd-distill-rows "$MAX_OPD_DISTILL_ROWS")
fi
if [[ "$MAX_OPD_PAIRWISE_PAIRS_PER_ROW" != "0" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-opd-pairwise-pairs-per-row "$MAX_OPD_PAIRWISE_PAIRS_PER_ROW")
fi
if [[ -n "$OPD_POSITIVE_REWARD_THRESHOLD" ]]; then
  UPDATE_EXTRA_ARGS+=(--opd-positive-reward-threshold "$OPD_POSITIVE_REWARD_THRESHOLD")
fi
DYNAMIC_OPD_ARGS=()
if [[ -n "$DYNAMIC_OPD_EXPERT_ROLLOUT" ]]; then
  IFS=',' read -r -a DYNAMIC_OPD_ROLLOUT_LIST <<< "$DYNAMIC_OPD_EXPERT_ROLLOUT"
  for opd_path in "${DYNAMIC_OPD_ROLLOUT_LIST[@]}"; do
    if [[ -n "$opd_path" ]]; then
      DYNAMIC_OPD_ARGS+=(--dynamic-opd-expert-rollout "$opd_path")
    fi
  done
  DYNAMIC_OPD_ARGS+=(
    --dynamic-opd-tasks "$DYNAMIC_OPD_TASKS"
    --dynamic-opd-key "$DYNAMIC_OPD_KEY"
    --dynamic-opd-current-max-success "$DYNAMIC_OPD_CURRENT_MAX_SUCCESS"
    --dynamic-opd-positive-threshold "$DYNAMIC_OPD_POSITIVE_THRESHOLD"
    --dynamic-opd-max-positives-per-row "$DYNAMIC_OPD_MAX_POSITIVES_PER_ROW"
    --dynamic-opd-max-negatives-per-row "$DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW"
    --dynamic-opd-per-task "$DYNAMIC_OPD_PER_TASK"
  )
  if [[ -n "$DYNAMIC_OPD_QUOTA" ]]; then
    IFS=',' read -r -a DYNAMIC_OPD_QUOTA_LIST <<< "$DYNAMIC_OPD_QUOTA"
    for quota_value in "${DYNAMIC_OPD_QUOTA_LIST[@]}"; do
      if [[ -n "$quota_value" ]]; then
        DYNAMIC_OPD_ARGS+=(--dynamic-opd-quota "$quota_value")
      fi
    done
  fi
  if is_truthy "$DYNAMIC_OPD_REQUIRE_ALL_TASKS"; then
    DYNAMIC_OPD_ARGS+=(--dynamic-opd-require-all-tasks)
  fi
fi
if is_truthy "$USE_OPD_ALL_SUCCESS"; then
  UPDATE_EXTRA_ARGS+=(--use-opd-all-success)
fi
if [[ "$OPD_ALL_SUCCESS_LOSS_WEIGHT" != "0" && "$OPD_ALL_SUCCESS_LOSS_WEIGHT" != "0.0" ]]; then
  UPDATE_EXTRA_ARGS+=(--opd-all-success-loss-weight "$OPD_ALL_SUCCESS_LOSS_WEIGHT")
fi
if [[ -n "$MAX_OPD_ALL_SUCCESS_ROWS" ]]; then
  UPDATE_EXTRA_ARGS+=(--max-opd-all-success-rows "$MAX_OPD_ALL_SUCCESS_ROWS")
fi
if [[ -n "$OPD_ALL_SUCCESS_POSITIVE_REWARD_THRESHOLD" ]]; then
  UPDATE_EXTRA_ARGS+=(--opd-all-success-positive-reward-threshold "$OPD_ALL_SUCCESS_POSITIVE_REWARD_THRESHOLD")
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
if [[ -n "$MAX_COEFF_DELTA_BY_EXPERT" ]]; then
  IFS=',' read -r -a MAX_COEFF_DELTA_BY_EXPERT_LIST <<< "$MAX_COEFF_DELTA_BY_EXPERT"
  for expert_delta in "${MAX_COEFF_DELTA_BY_EXPERT_LIST[@]}"; do
    if [[ -n "$expert_delta" ]]; then
      UPDATE_EXTRA_ARGS+=(--max-coefficient-delta-from-init-by-expert "$expert_delta")
    fi
  done
fi
if [[ -n "$COEFF_BOUND_BY_EXPERT" ]]; then
  IFS=',' read -r -a COEFF_BOUND_BY_EXPERT_LIST <<< "$COEFF_BOUND_BY_EXPERT"
  for expert_bound in "${COEFF_BOUND_BY_EXPERT_LIST[@]}"; do
    if [[ -n "$expert_bound" ]]; then
      UPDATE_EXTRA_ARGS+=(--coefficient-bound-by-expert "$expert_bound")
    fi
  done
fi
if is_truthy "$RECOMPUTE_FRONTIER"; then
  UPDATE_EXTRA_ARGS+=(--recompute-frontier)
fi
if is_truthy "$SGD_NESTEROV"; then
  UPDATE_EXTRA_ARGS+=(--sgd-nesterov)
fi
if is_truthy "$PERSIST_OPTIMIZER_STATE"; then
  UPDATE_EXTRA_ARGS+=(--persist-optimizer-state)
fi
if [[ -n "$OPTIMIZER_STATE_CHECKPOINT" ]]; then
  UPDATE_EXTRA_ARGS+=(--optimizer-state-checkpoint "$OPTIMIZER_STATE_CHECKPOINT")
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
echo "[run] config=$CONFIG"
echo "[run] calibration=$CALIBRATION"
echo "[run] run_dir=$RUN_DIR"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] optimizer=$OPTIMIZER sgd_momentum=$SGD_MOMENTUM persist_optimizer_state=$PERSIST_OPTIMIZER_STATE"
echo "[run] lr=$LR prior=$PRIOR_LOSS_WEIGHT max_delta=$MAX_COEFF_DELTA"
echo "[run] max_delta_by_expert=${MAX_COEFF_DELTA_BY_EXPERT:-none}"
echo "[run] coeff_bound_by_expert=${COEFF_BOUND_BY_EXPERT:-none}"
echo "[run] loss_granularity=$LOSS_GRANULARITY update_batch_size=$UPDATE_BATCH_SIZE optimizer_step_scope=$OPTIMIZER_STEP_SCOPE store_token_logprobs=$STORE_TOKEN_LOGPROBS"
echo "[run] task_normalize_advantages=$TASK_NORMALIZE_ADVANTAGES advantage_normalization=$ADVANTAGE_NORMALIZATION use_frontier_weight=$USE_FRONTIER_WEIGHT advantage_field_frontier_weight=$ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT"
echo "[run] length_norm policy=$LENGTH_NORMALIZE_POLICY_LOGPROB legacy=$LENGTH_NORMALIZE_LOGPROB opd=$OPD_LENGTH_NORMALIZE_LOGPROB retention=$RETENTION_LENGTH_NORMALIZE_LOGPROB"
echo "[run] retention_dynamic_scale=$RETENTION_DYNAMIC_SCALE retention_task_balanced=$RETENTION_TASK_BALANCED_LOSS_SCALE retention_target=$RETENTION_SCALE_TARGET"
echo "[run] pcgrad_gate_gradients=$PCGRAD_GATE_GRADIENTS pcgrad_eps=$PCGRAD_EPS pcgrad_tasks=${PCGRAD_TASKS:-all-observed}"
echo "[run] tool_nullspace_gate_gradients=$TOOL_NULLSPACE_GATE_GRADIENTS rows=$TOOL_NULLSPACE_ROWS min_rows=$TOOL_NULLSPACE_MIN_ROWS rank=$TOOL_NULLSPACE_RANK replay=${TOOL_NULLSPACE_REPLAY_ROLLOUT:-none}"
echo "[run] opd_dynamic_scale=$OPD_DYNAMIC_SCALE opd_task_balanced=$OPD_TASK_BALANCED_LOSS_SCALE scale_targets=$OPD_SCALE_TARGET_HIGH/$OPD_SCALE_TARGET_MID/$OPD_SCALE_TARGET_LOW/$OPD_SCALE_TARGET_TAIL"
echo "[run] frontier_order=$FRONTIER_ORDER frontier_shuffle_seed=${FRONTIER_SHUFFLE_SEED:-auto} frontier_sample_before_limit=$FRONTIER_SAMPLE_BEFORE_LIMIT ignore_config_frontier_task_quota=$IGNORE_CONFIG_FRONTIER_TASK_QUOTA"
echo "[run] task_weights=tool:$TASK_WEIGHT_TOOL,memory:$TASK_WEIGHT_MEMORY,code:$TASK_WEIGHT_CODE quotas=tool:${FRONTIER_TOOL_QUOTA:-all},memory:${FRONTIER_MEMORY_QUOTA:-all},code:${FRONTIER_CODE_QUOTA:-all}"
echo "[run] retention=$USE_RETENTION retention_objective=$RETENTION_OBJECTIVE retention_loss_weight=${RETENTION_LOSS_WEIGHT:-none} retention_positive_threshold=${RETENTION_POSITIVE_REWARD_THRESHOLD:-none} max_retention_rows=${MAX_RETENTION_ROWS:-none} max_retention_rows_per_task=${MAX_RETENTION_ROWS_PER_TASK:-none} retention_sample_before_limit=$RETENTION_SAMPLE_BEFORE_LIMIT retention_shuffle_seed=${RETENTION_SHUFFLE_SEED:-frontier}"
echo "[run] opd_rollout=${OPD_DISTILL_ROLLOUT:-none} opd_loss=$OPD_LOSS_WEIGHT opd_pairwise=$OPD_PAIRWISE_LOSS_WEIGHT max_opd_rows=${MAX_OPD_DISTILL_ROWS:-none}"
echo "[run] dynamic_opd_rollout=${DYNAMIC_OPD_EXPERT_ROLLOUT:-none} dynamic_opd_tasks=$DYNAMIC_OPD_TASKS dynamic_opd_per_task=$DYNAMIC_OPD_PER_TASK require_all_tasks=$DYNAMIC_OPD_REQUIRE_ALL_TASKS"
echo "[run] opd_all_success=$USE_OPD_ALL_SUCCESS opd_all_success_loss=$OPD_ALL_SUCCESS_LOSS_WEIGHT max_opd_all_success_rows=${MAX_OPD_ALL_SUCCESS_ROWS:-none}"
echo "[run] tokens=default:$MAX_NEW_TOKENS,tool:$TOOL_MAX_NEW_TOKENS,code:$CODE_MAX_NEW_TOKENS,memory_update:$MEMORY_UPDATE_MAX_NEW_TOKENS,memory_final:$MEMORY_FINAL_MAX_NEW_TOKENS"
echo "[run] rollout_shards=$ROLLOUT_SHARDS rollout_gpus=$ROLLOUT_GPUS vllm_batch_size=$ROLLOUT_BATCH_SIZE"

if [[ -z "$INIT_GATE_CHECKPOINT" ]]; then
  "$PY" scripts/modes/build_constant_gate_checkpoint.py \
    --config "$CONFIG" \
    --mode-manifest "$MODE" \
    --gate-parameterization "$STRATEGY" \
    --value "$INIT_VALUE" \
    --output "$INIT_GATE" >/dev/null
else
  echo "[run] using explicit init gate checkpoint: $INIT_GATE"
fi

"$PY" scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  --config "$CONFIG" \
  --mode-manifest "$MODE" \
  --seed-manifest "$CALIBRATION" \
  "${FILTER_ARGS[@]}" \
  --run-dir "$RUN_DIR" \
  --run-id "$RUN_NAME" \
  --num-iters "$NUM_ITERS" \
  --start-iteration "$START_ITERATION" \
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
  --optimizer-step-scope "$OPTIMIZER_STEP_SCOPE" \
  --loss-granularity "$LOSS_GRANULARITY" \
  "${FRONTIER_ORDER_ARGS[@]}" \
  "${UPDATE_EXTRA_ARGS[@]}" \
  --lr "$LR" \
  --optimizer "$OPTIMIZER" \
  --sgd-momentum "$SGD_MOMENTUM" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --max-coefficient-delta-from-init "$MAX_COEFF_DELTA" \
  --behavior-span-reward-weight "$BEHAVIOR_SPAN_REWARD_WEIGHT" \
  "${PCGRAD_ARGS[@]}" \
  "${TOOL_NULLSPACE_ARGS[@]}" \
  "${DYNAMIC_OPD_ARGS[@]}" \
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
