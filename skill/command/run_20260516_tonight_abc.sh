#!/usr/bin/env bash
set -euo pipefail

# Controlled A/B/C runs for 2026-05-16.
# PHASE=train_a|train_b|train_c|build_reasoning_modes|build_c_init

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PHASE="${PHASE:-train_a}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/gated_grpo}"
CALIBRATION="${CALIBRATION:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
BASE_MODE="${BASE_MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
REASONING_MODE_DIR="${REASONING_MODE_DIR:-$ROOT/modes/opvec4_reasoning_20260516}"
REASONING_MODE="${REASONING_MODE:-$REASONING_MODE_DIR/mode_manifest.json}"
EXPERT_DIR="${EXPERT_DIR:-$ROOT/data/calibration/paper96_expert_rollouts_seed20260514}"
CODE_AUG_DIR="${CODE_AUG_DIR:-$ROOT/data/calibration/20260516_code_opd_aug}"

TOOL_EXPERT="$EXPERT_DIR/tool_expert_paper96_s2_seed20260514.jsonl"
MEMORY_EXPERT="$EXPERT_DIR/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT_OLD="$EXPERT_DIR/code_expert_paper96_s2_seed20260514.jsonl"
CODE_EXPERT_REASONFLUX="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl"
CODE_EXPERT_REASONFLUX2="$CODE_AUG_DIR/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl"
CODE_EXPERT_DEEPSEEK="$CODE_AUG_DIR/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl"
CODE_EXPERT_MEMORY="$CODE_AUG_DIR/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl"
CODE_EXPERT_EXTRA_ROLLOUTS="${CODE_EXPERT_EXTRA_ROLLOUTS:-}"

COMMON_ENV=(
  STRATEGY=global-parameter
  INIT_VALUE=0.3333333333333333
  CALIBRATION="$CALIBRATION"
  NUM_ITERS=20
  NUM_PROMPTS=96
  SAMPLES_PER_PROMPT=4
  STORE_TOKEN_LOGPROBS=0
  OPTIMIZER=sgd
  SGD_MOMENTUM=0.2
  PERSIST_OPTIMIZER_STATE=1
  LR="${LR:-0.1876}"
  PRIOR_LOSS_WEIGHT=0.0
  MAX_COEFF_DELTA=1.0
  UPDATE_EPOCHS=1
  UPDATE_BATCH_SIZE=4
  BATCH_LOSS_REDUCTION=mean
  OPTIMIZER_STEP_SCOPE=epoch
  LOSS_GRANULARITY=sequence
  FRONTIER_ORDER=task-interleaved
  FRONTIER_TOOL_QUOTA=32
  FRONTIER_MEMORY_QUOTA=32
  FRONTIER_CODE_QUOTA=32
  USE_RETENTION=1
  RETENTION_OBJECTIVE=nll
  RETENTION_POSITIVE_REWARD_THRESHOLD=1.0
  RETENTION_TASK_BALANCED_LOSS_SCALE=1
  RETENTION_SCALE_TARGET=0.5
  OPD_LOSS_WEIGHT=1.0
  OPD_POSITIVE_REWARD_THRESHOLD=1.0
  OPD_LENGTH_NORMALIZE_LOGPROB=1
  RETENTION_LENGTH_NORMALIZE_LOGPROB=1
  OPD_TASK_BALANCED_LOSS_SCALE=1
  LENGTH_NORMALIZE_POLICY_LOGPROB=1
  LENGTH_NORMALIZE_LOGPROB=0
  TASK_NORMALIZE_ADVANTAGES=0
  ADVANTAGE_NORMALIZATION=centered
  USE_FRONTIER_WEIGHT=0
  DYNAMIC_OPD_TASKS=tool,memory,code
  DYNAMIC_OPD_KEY=prompt_id
  DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0
  DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0
  DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1
  DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2
  DYNAMIC_OPD_PER_TASK=32
  MAX_NEW_TOKENS=1024
  TOOL_MAX_NEW_TOKENS=512
  CODE_MAX_NEW_TOKENS=4096
  MEMORY_UPDATE_MAX_NEW_TOKENS=2048
  MEMORY_FINAL_MAX_NEW_TOKENS=2048
  MAX_PROMPT_TOKENS=8192
  MAX_MODEL_LEN=12288
  MAX_LOGPROB_TOKENS=12288
  ROLLOUT_BATCH_SIZE=32
  ROLLOUT_SHARDS=auto
  TENSOR_PARALLEL_SIZE=1
  GPU_MEMORY_UTILIZATION=0.82
  TEMPERATURE=0.7
  TOP_P=0.95
  SEED_VALUE=20260516
  PROGRESS_EVERY=10
  DRY_RUN="${DRY_RUN:-0}"
)

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

build_reasoning_modes() {
  mkdir -p "$REASONING_MODE_DIR"
  "$PY" scripts/modes/build_opvec4_modes.py \
    --config configs/gated_grpo_reasoning.yaml \
    --output-dir "$REASONING_MODE_DIR"
}

build_c_init() {
  require_file "$REASONING_MODE"
  local output="${C_INIT_GATE:-$CODE_AUG_DIR/init_global_parameter_tmc033333_reasoning0.json}"
  mkdir -p "$(dirname "$output")"
  "$PY" - "$REASONING_MODE" "$output" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from opvec.modeling.manifest import manifest_param_names

mode = Path(sys.argv[1])
output = Path(sys.argv[2])
experts = ("tool", "memory", "code", "reasoning")
initial = {"tool": 1.0 / 3.0, "memory": 1.0 / 3.0, "code": 1.0 / 3.0, "reasoning": 0.0}
gates = {f"__global__::{expert}": value for expert, value in initial.items()}
for param_name in manifest_param_names(mode):
    for expert, value in initial.items():
        gates[f"{param_name}::{expert}"] = value
payload = {
    "format": "opvec_reasoning_zero_init_gate_checkpoint_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mode_manifest": str(mode.resolve()),
    "gate_parameterization": "global-parameter",
    "experts": list(experts),
    "initial_global_coefficients": initial,
    "num_gate_values": len(gates),
    "gates": gates,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(output), "num_gate_values": len(gates), "experts": list(experts)}, ensure_ascii=False))
PY
}

augmented_expert_rollouts_csv() {
  local paths=(
    "$TOOL_EXPERT"
    "$MEMORY_EXPERT"
    "$CODE_EXPERT_OLD"
    "$CODE_EXPERT_REASONFLUX"
    "$CODE_EXPERT_DEEPSEEK"
    "$CODE_EXPERT_MEMORY"
  )
  if [[ -n "$CODE_EXPERT_EXTRA_ROLLOUTS" ]]; then
    local extra
    local old_ifs="$IFS"
    IFS=','
    for extra in $CODE_EXPERT_EXTRA_ROLLOUTS; do
      if [[ -n "$extra" ]]; then
        paths+=("$extra")
      fi
    done
    IFS="$old_ifs"
  elif [[ -f "$CODE_EXPERT_REASONFLUX2" && -f "${CODE_EXPERT_REASONFLUX2%.jsonl}.coverage.json" ]]; then
    paths+=("$CODE_EXPERT_REASONFLUX2")
  fi
  local path
  for path in "${paths[@]}"; do
    require_file "$path"
  done
  local IFS=,
  echo "${paths[*]}"
}

run_strategy() {
  env "${COMMON_ENV[@]}" "$@" bash skill/command/run_qbank_c033333_gate_strategy.sh
}

case "$PHASE" in
  build_reasoning_modes)
    build_reasoning_modes
    ;;
  build_c_init)
    build_c_init
    ;;
  train_a)
    require_file "$TOOL_EXPERT"; require_file "$MEMORY_EXPERT"; require_file "$CODE_EXPERT_OLD"
    run_strategy \
      CONFIG=configs/gated_grpo.yaml \
      MODE="$BASE_MODE" \
      RUN_NAME=expA_gp_grpo_opd_from_init_20260516 \
      RUN_DIR="$RUN_ROOT/expA_gp_grpo_opd_from_init_20260516" \
      GPU_LIST="${GPU_LIST:-0,1}" \
      ROLLOUT_GPUS="${ROLLOUT_GPUS:-${GPU_LIST:-0,1}}" \
      PPO_LOSS_WEIGHT=1.0 \
      DYNAMIC_OPD_EXPERT_ROLLOUT="$TOOL_EXPERT,$MEMORY_EXPERT,$CODE_EXPERT_OLD"
    ;;
  train_b)
    dynamic_opd_rollouts="$(augmented_expert_rollouts_csv)"
    run_strategy \
      CONFIG=configs/gated_grpo.yaml \
      MODE="$BASE_MODE" \
      RUN_NAME=expB_gp_code_opd_aug_20260516 \
      RUN_DIR="$RUN_ROOT/expB_gp_code_opd_aug_20260516" \
      GPU_LIST="${GPU_LIST:-2,3}" \
      ROLLOUT_GPUS="${ROLLOUT_GPUS:-${GPU_LIST:-2,3}}" \
      PPO_LOSS_WEIGHT=0.0 \
      DYNAMIC_OPD_EXPERT_ROLLOUT="$dynamic_opd_rollouts"
    ;;
  train_c)
    dynamic_opd_rollouts="$(augmented_expert_rollouts_csv)"
    if [[ ! -f "$REASONING_MODE" ]]; then
      build_reasoning_modes
    fi
    build_c_init
    run_strategy \
      CONFIG=configs/gated_grpo_reasoning.yaml \
      MODE="$REASONING_MODE" \
      RUN_NAME=expC_gp_reasoning_code_opd_aug_20260516 \
      RUN_DIR="$RUN_ROOT/expC_gp_reasoning_code_opd_aug_20260516" \
      INIT_GATE_CHECKPOINT="${C_INIT_GATE:-$CODE_AUG_DIR/init_global_parameter_tmc033333_reasoning0.json}" \
      GPU_LIST="${GPU_LIST:-4,5}" \
      ROLLOUT_GPUS="${ROLLOUT_GPUS:-${GPU_LIST:-4,5}}" \
      PPO_LOSS_WEIGHT=0.0 \
      DYNAMIC_OPD_EXPERT_ROLLOUT="$dynamic_opd_rollouts"
    ;;
  *)
    echo "[error] unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac
