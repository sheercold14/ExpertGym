#!/usr/bin/env bash
set -euo pipefail

# Build a clean question-bank pipeline using an untuned average task-vector
# baseline: theta = base + (1/3) * sum_i(expert_i - base).
#
# Phases:
#   PHASE=prepare      create raw Tool/Code + HotpotQA Memory hybrid manifests
#   PHASE=build-model  build the TA average-vector checkpoint
#   PHASE=rollout      vLLM rollout with official RewardRouter rewards
#   PHASE=bank         aggregate rollout rewards into a question bank
#   PHASE=sample       sample calibration and guard manifests
#   PHASE=all          run all phases in order
#
# Set DRY_RUN=1 to print commands without executing them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
FORCE_ROLLOUT="${FORCE_ROLLOUT:-0}"

QB_VERSION="${QB_VERSION:-ta_avgvec_c033333_hotpotqa_v1}"
QB="${QB:-$ROOT/data/question_bank/$QB_VERSION}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/question_bank/$QB_VERSION}"
ROLL_OUT_DIR="$QB/rollouts"
CALIB_DIR="$QB/calibration"
LOG_DIR="$RUN_ROOT/logs"
COMMANDS_FILE="$QB/commands.sh"
RUN_MANIFEST="$QB/run_manifest.json"

TA_SCALE="${TA_SCALE:-0.3333333333333333}"
TA_NAME="${TA_NAME:-ta_avgvec_c033333}"
TA_BUILDER="${TA_BUILDER:-/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/scripts/build_task_arithmetic_merge.py}"
BASE_MODEL="${BASE_MODEL:-/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct}"
TOOL_MODEL="${TOOL_MODEL:-/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold}"
MEMORY_MODEL="${MEMORY_MODEL:-/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B}"
CODE_MODEL="${CODE_MODEL:-/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B}"
POLICY="${POLICY:-/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/$TA_NAME/model}"

RAW_MANIFEST="${RAW_MANIFEST:-$QB/source_reward_raw_tool${TOOL_LIMIT:-500}_code${CODE_LIMIT:-500}_seed${DATA_SEED:-20260511}.jsonl}"
MEMORY_MANIFEST="${MEMORY_MANIFEST:-$QB/hotpotqa_train_memory${MEMORY_LIMIT:-500}_chunk${MEMORY_CHUNK_TOKENS:-5000}_seed${DATA_SEED:-20260511}.jsonl}"
SEED_MANIFEST="${SEED_MANIFEST:-$QB/source_reward_hybrid_tool_code_hotpotqa_memory_traj_seed${DATA_SEED:-20260511}.jsonl}"
ROLLOUTS="${ROLLOUTS:-$ROLL_OUT_DIR/baseline_${TA_NAME}_s${ROLLOUT_SEED:-20260511}_n${SAMPLES_PER_PROMPT:-8}.jsonl}"
QUESTION_BANK="${QUESTION_BANK:-$QB/question_bank.jsonl}"
CALIB_PREFIX="${CALIB_PREFIX:-$CALIB_DIR/calib100_seed${DATA_SEED:-20260511}}"

DATA_SEED="${DATA_SEED:-20260511}"
TOOL_LIMIT="${TOOL_LIMIT:-500}"
CODE_LIMIT="${CODE_LIMIT:-500}"
HOTPOTQA_TRAIN="${HOTPOTQA_TRAIN:-/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/hotpotqa/hotpotqa_train_32k.parquet}"
MEMORY_LIMIT="${MEMORY_LIMIT:-500}"
MEMORY_CHUNK_TOKENS="${MEMORY_CHUNK_TOKENS:-5000}"
MEMORY_MAX_CHUNKS="${MEMORY_MAX_CHUNKS:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
TASKS="${TASKS:-}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
ROLLOUT_SEED="${ROLLOUT_SEED:-20260511}"

GPU_LIST="${GPU_LIST:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MEMORY_UPDATE_MAX_NEW_TOKENS="${MEMORY_UPDATE_MAX_NEW_TOKENS:-1024}"
MEMORY_FINAL_MAX_NEW_TOKENS="${MEMORY_FINAL_MAX_NEW_TOKENS:-1024}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
DTYPE="${DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

TOOL_QUOTA="${TOOL_QUOTA:-34}"
MEMORY_QUOTA="${MEMORY_QUOTA:-33}"
CODE_QUOTA="${CODE_QUOTA:-33}"
GUARD_PER_TASK="${GUARD_PER_TASK:-3}"

setup_dirs() {
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$QB" "$RUN_ROOT" "$ROLL_OUT_DIR" "$CALIB_DIR" "$LOG_DIR" "$(dirname "$POLICY")"
    if [[ ! -f "$COMMANDS_FILE" || "$FORCE_REBUILD" == "1" ]]; then
      {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        echo "# Commands recorded by $(basename "$0")"
        echo "# Created at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
      } > "$COMMANDS_FILE"
      chmod +x "$COMMANDS_FILE"
    fi
  fi
}

print_cmd() {
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
}

record_cmd() {
  if [[ "$DRY_RUN" != "1" ]]; then
    {
      echo
      echo "# phase=$PHASE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "$COMMANDS_FILE"
    printf '%q ' "$@" >> "$COMMANDS_FILE"
    printf '\n' >> "$COMMANDS_FILE"
  fi
}

run_cmd() {
  print_cmd "$@"
  record_cmd "$@"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_logged() {
  local log_file="$1"
  shift
  print_cmd "$@"
  record_cmd "$@"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@" 2>&1 | tee "$log_file"
  fi
}

write_run_manifest() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  export QB_VERSION QB RUN_ROOT TA_SCALE TA_NAME TA_BUILDER BASE_MODEL TOOL_MODEL MEMORY_MODEL CODE_MODEL POLICY
  export RAW_MANIFEST MEMORY_MANIFEST SEED_MANIFEST ROLLOUTS QUESTION_BANK CALIB_PREFIX DATA_SEED
  export HOTPOTQA_TRAIN TOOL_LIMIT CODE_LIMIT MEMORY_LIMIT MEMORY_CHUNK_TOKENS MEMORY_MAX_CHUNKS NUM_PROMPTS PROMPT_OFFSET TASKS SAMPLES_PER_PROMPT ROLLOUT_SEED
  export MAX_NEW_TOKENS MEMORY_UPDATE_MAX_NEW_TOKENS MEMORY_FINAL_MAX_NEW_TOKENS MAX_PROMPT_TOKENS MAX_MODEL_LEN TEMPERATURE TOP_P DTYPE
  "$PY" - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def git_value(args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None

relevant_paths = [
    "configs/gated_grpo.yaml",
    "scripts/data/build_hotpotqa_memory_seed_manifest.py",
    "scripts/data/build_question_bank_from_rollouts.py",
    "scripts/data/build_routed_correct_seed_manifest.py",
    "scripts/data/build_source_reward_seed_manifest.py",
    "scripts/data/merge_seed_manifests.py",
    "scripts/data/sample_question_bank.py",
    "scripts/train/opvec_collect_vllm_rollouts.py",
    "skill/command/run_question_bank_ta_avgvec_c033333.sh",
]

payload = {
    "format": "opvec_question_bank_pipeline_manifest_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "worktree": str(Path.cwd()),
    "git_commit": git_value(["rev-parse", "HEAD"]),
    "git_status_relevant": git_value(["status", "--short", "--", *relevant_paths]),
    "qb_version": os.environ["QB_VERSION"],
    "question_bank_root": os.environ["QB"],
    "run_root": os.environ["RUN_ROOT"],
    "baseline": {
        "name": os.environ["TA_NAME"],
        "method": "task_arithmetic_average_task_vector",
        "formula": "theta = theta_base + (1/3) * sum_i(theta_expert_i - theta_base)",
        "scaling_coefficient": float(os.environ["TA_SCALE"]),
        "builder": os.environ["TA_BUILDER"],
        "policy_model": os.environ["POLICY"],
        "base_model": os.environ["BASE_MODEL"],
        "expert_models": {
            "tool": os.environ["TOOL_MODEL"],
            "memory": os.environ["MEMORY_MODEL"],
            "code": os.environ["CODE_MODEL"],
        },
    },
    "data": {
        "raw_manifest": os.environ["RAW_MANIFEST"],
        "memory_manifest": os.environ["MEMORY_MANIFEST"],
        "seed_manifest": os.environ["SEED_MANIFEST"],
        "memory_source": "hotpotqa_train_32k_parquet",
        "hotpotqa_train": os.environ["HOTPOTQA_TRAIN"],
        "tool_limit": int(os.environ["TOOL_LIMIT"]),
        "code_limit": int(os.environ["CODE_LIMIT"]),
        "memory_limit": int(os.environ["MEMORY_LIMIT"]),
        "memory_chunk_tokens": int(os.environ["MEMORY_CHUNK_TOKENS"]),
        "memory_max_chunks": int(os.environ["MEMORY_MAX_CHUNKS"]),
        "seed": int(os.environ["DATA_SEED"]),
    },
    "rollout": {
        "output": os.environ["ROLLOUTS"],
        "num_prompts": int(os.environ["NUM_PROMPTS"]),
        "prompt_offset": int(os.environ["PROMPT_OFFSET"]),
        "tasks": os.environ["TASKS"] or None,
        "samples_per_prompt": int(os.environ["SAMPLES_PER_PROMPT"]),
        "seed": int(os.environ["ROLLOUT_SEED"]),
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
        "memory_update_max_new_tokens": int(os.environ["MEMORY_UPDATE_MAX_NEW_TOKENS"]),
        "memory_final_max_new_tokens": int(os.environ["MEMORY_FINAL_MAX_NEW_TOKENS"]),
        "max_prompt_tokens": int(os.environ["MAX_PROMPT_TOKENS"]),
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
        "temperature": float(os.environ["TEMPERATURE"]),
        "top_p": float(os.environ["TOP_P"]),
        "dtype": os.environ["DTYPE"],
        "reward": "RewardRouter official task reward; behavior_span_reward_weight=0.0",
    },
    "outputs": {
        "question_bank": os.environ["QUESTION_BANK"],
        "calibration_prefix": os.environ["CALIB_PREFIX"],
    },
}
path = Path(os.environ["QB"]) / "run_manifest.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
}

prepare_manifests() {
  if [[ "$FORCE_REBUILD" == "1" || ! -s "$RAW_MANIFEST" ]]; then
    run_cmd "$PY" scripts/data/build_source_reward_seed_manifest.py \
      --output "$RAW_MANIFEST" \
      --tool-limit "$TOOL_LIMIT" \
      --memory-final-limit 0 \
      --code-limit "$CODE_LIMIT" \
      --seed "$DATA_SEED"
  else
    echo "[skip] raw manifest exists: $RAW_MANIFEST"
  fi

  if [[ "$FORCE_REBUILD" == "1" || ! -s "$MEMORY_MANIFEST" ]]; then
    run_cmd "$PY" scripts/data/build_hotpotqa_memory_seed_manifest.py \
      --input "$HOTPOTQA_TRAIN" \
      --output "$MEMORY_MANIFEST" \
      --limit "$MEMORY_LIMIT" \
      --seed "$DATA_SEED" \
      --chunk-tokenizer "$BASE_MODEL" \
      --chunk-size-tokens "$MEMORY_CHUNK_TOKENS" \
      --max-chunks "$MEMORY_MAX_CHUNKS"
  else
    echo "[skip] HotpotQA memory manifest exists: $MEMORY_MANIFEST"
  fi

  if [[ "$FORCE_REBUILD" == "1" || ! -s "$SEED_MANIFEST" ]]; then
    run_cmd "$PY" scripts/data/merge_seed_manifests.py \
      --input "$RAW_MANIFEST::tool,code" \
      --input "$MEMORY_MANIFEST::memory" \
      --output "$SEED_MANIFEST"
  else
    echo "[skip] hybrid seed manifest exists: $SEED_MANIFEST"
  fi
}

build_model() {
  if [[ "$FORCE_REBUILD" == "1" || ! -f "$POLICY/model.safetensors.index.json" ]]; then
    run_logged "$LOG_DIR/build_${TA_NAME}.log" \
      "$PY" "$TA_BUILDER" \
      --base-model "$BASE_MODEL" \
      --expert-models "$TOOL_MODEL" "$MEMORY_MODEL" "$CODE_MODEL" \
      --output-dir "$POLICY" \
      --scaling-coefficient "$TA_SCALE" \
      --overwrite
  else
    echo "[skip] TA average-vector checkpoint exists: $POLICY"
  fi
}

rollout_baseline() {
  if [[ "$FORCE_ROLLOUT" != "1" && -s "$ROLLOUTS" ]]; then
    echo "[skip] rollout exists: $ROLLOUTS"
    return
  fi
  local cmd=(
    "$PY" scripts/train/opvec_collect_vllm_rollouts.py
    --config configs/gated_grpo.yaml \
    --mode-manifest "$ROOT/modes/opvec4/mode_manifest.json" \
    --policy-model "$POLICY" \
    --policy-id "$TA_NAME" \
    --no-gate-values \
    --seed-manifest "$SEED_MANIFEST" \
    --output "$ROLLOUTS" \
    --run-id "qb-${QB_VERSION}-${TA_NAME}-n${SAMPLES_PER_PROMPT}" \
    --num-prompts "$NUM_PROMPTS" \
    --prompt-offset "$PROMPT_OFFSET" \
    --use-manifest-order \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --memory-update-max-new-tokens "$MEMORY_UPDATE_MAX_NEW_TOKENS" \
    --memory-final-max-new-tokens "$MEMORY_FINAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --vllm-batch-size "$VLLM_BATCH_SIZE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$ROLLOUT_SEED" \
    --stream-output \
    --progress-every 10 \
    --behavior-span-reward-weight 0.0
  )
  if [[ -n "$TASKS" ]]; then
    cmd+=(--tasks "$TASKS")
  fi
  run_cmd "${cmd[@]}"
}

build_question_bank() {
  run_cmd "$PY" scripts/data/build_question_bank_from_rollouts.py \
    --rollouts "$ROLLOUTS" \
    --seed-manifest "$SEED_MANIFEST" \
    --output "$QUESTION_BANK" \
    --summary "$QB/question_bank.summary.json"
}

sample_calibration() {
  run_cmd "$PY" scripts/data/sample_question_bank.py \
    --question-bank "$QUESTION_BANK" \
    --output-prefix "$CALIB_PREFIX" \
    --quota "tool=$TOOL_QUOTA" \
    --quota "memory=$MEMORY_QUOTA" \
    --quota "code=$CODE_QUOTA" \
    --guard-per-task "$GUARD_PER_TASK" \
    --seed "$DATA_SEED"
}

setup_dirs
write_run_manifest

case "$PHASE" in
  prepare)
    prepare_manifests
    ;;
  build-model)
    build_model
    ;;
  rollout)
    rollout_baseline
    ;;
  bank)
    build_question_bank
    ;;
  sample)
    sample_calibration
    ;;
  all)
    prepare_manifests
    build_model
    rollout_baseline
    build_question_bank
    sample_calibration
    ;;
  post)
    build_question_bank
    sample_calibration
    ;;
  *)
    echo "Unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac

echo "[done] phase=$PHASE qb=$QB"
