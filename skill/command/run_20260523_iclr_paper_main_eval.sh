#!/usr/bin/env bash
set -euo pipefail

# Paper-main full Eval6 queue for the ICLR ExpertGym draft.
#
# Default behavior is DRY_RUN=1: print reproducible commands without launching
# the long external harnesses.  Set DRY_RUN=0 only after checking GPU allocation.

usage() {
  cat <<'EOF'
Usage:
  DRY_RUN=1 bash skill/command/run_20260523_iclr_paper_main_eval.sh

Phases:
  PHASE=list         print selected candidates and paths
  PHASE=tool_memory run Tool BFCL + Memory HotpotQA full harness
  PHASE=code        run Code/CURE full harness
  PHASE=all         run tool_memory then code, sequentially

Environment:
  CANDIDATES=bcrc_v18_alias_v9,no_behavior_v1_code_only,hard_behavior_v8
  DRY_RUN=1|0
  ROOT=/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523
  RUN_ID=iclr_main_eval6_20260523
  TOOL_GPU=0
  TOOL_PORT=8160
  MEMORY_GPU_IDS=0
  CODE_GPU_GROUPS="[[0,1]]"

Rationale:
  The RCF-BC mechanism rows are not comparable to Eval6 baseline rows until
  these exact candidates are evaluated through skill/command/run_full_eval_suite.sh.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

PHASE="${PHASE:-tool_memory}"
DRY_RUN="${DRY_RUN:-1}"
CANDIDATES="${CANDIDATES:-bcrc_v18_alias_v9,no_behavior_v1_code_only,hard_behavior_v8}"

ROOT="${ROOT:-/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523}"
RUN_ID="${RUN_ID:-iclr_main_eval6_20260523}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-expertgym-iclr-main-eval6}"

TOOL_GPU="${TOOL_GPU:-0}"
TOOL_PORT="${TOOL_PORT:-8160}"
TOOL_CATEGORIES="${TOOL_CATEGORIES:-parallel,parallel_multiple,live_parallel,live_parallel_multiple}"

MEMORY_GPU_IDS="${MEMORY_GPU_IDS:-0}"
MEMORY_TP="${MEMORY_TP:-1}"
MEMORY_DATASETS="${MEMORY_DATASETS:-eval_50 eval_100 eval_qa_1_32768 eval_qa_1_65536}"

CODE_GPU_GROUPS="${CODE_GPU_GROUPS:-[[0,1]]}"
CODE_MAX_TEST="${CODE_MAX_TEST:-8}"
CODE_MAX_GENERATION_TOKEN="${CODE_MAX_GENERATION_TOKEN:-10000}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/shared-storage/OnPolicy/checkpoints}"
GATE_ROOT="${GATE_ROOT:-/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates}"

candidate_model_path() {
  case "$1" in
    bcrc_v18_alias_v9)
      # v18 and v9 have numerically identical gate coefficients, but only the
      # v9 baked checkpoint currently exists.  The paper queue uses this alias
      # explicitly instead of silently falling back.
      echo "$CHECKPOINT_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9"
      ;;
    no_behavior_v1_code_only)
      echo "$CHECKPOINT_ROOT/rcrf_code_contrast_v1"
      ;;
    hard_behavior_v8)
      echo "$CHECKPOINT_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_v8"
      ;;
    strict_cleanup_v19)
      echo "$CHECKPOINT_ROOT/rcrf_archetype_consistency_v19"
      ;;
    scalar_code_half_v14)
      echo "$CHECKPOINT_ROOT/rcrf_v9_code_half_v14"
      ;;
    scalar_code_zero_v15)
      echo "$CHECKPOINT_ROOT/rcrf_v9_code_zero_v15"
      ;;
    *)
      echo "[error] unknown candidate: $1" >&2
      return 2
      ;;
  esac
}

candidate_gate_path() {
  case "$1" in
    bcrc_v18_alias_v9)
      echo "$GATE_ROOT/residual_capability_field_behavior_constraints_v18/gates.json"
      ;;
    no_behavior_v1_code_only)
      echo "$GATE_ROOT/rcrf_code_contrast_v1/gates.json"
      ;;
    hard_behavior_v8)
      echo "$GATE_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_v8/gates.json"
      ;;
    strict_cleanup_v19)
      echo "$GATE_ROOT/rcrf_archetype_consistency_v19/gates.json"
      ;;
    scalar_code_half_v14)
      echo "$GATE_ROOT/rcrf_v9_code_half_v14/gates.json"
      ;;
    scalar_code_zero_v15)
      echo "$GATE_ROOT/rcrf_v9_code_zero_v15/gates.json"
      ;;
    *)
      echo "[error] unknown candidate: $1" >&2
      return 2
      ;;
  esac
}

candidate_model_name() {
  case "$1" in
    bcrc_v18_alias_v9) echo "bcrc-v18-alias-v9" ;;
    no_behavior_v1_code_only) echo "no-behavior-v1-code-only" ;;
    hard_behavior_v8) echo "hard-behavior-v8" ;;
    strict_cleanup_v19) echo "strict-cleanup-v19" ;;
    scalar_code_half_v14) echo "scalar-code-half-v14" ;;
    scalar_code_zero_v15) echo "scalar-code-zero-v15" ;;
    *) echo "[error] unknown candidate: $1" >&2; return 2 ;;
  esac
}

candidate_list() {
  IFS=',' read -ra items <<< "$CANDIDATES"
  for item in "${items[@]}"; do
    item="${item// /}"
    [[ -n "$item" ]] && echo "$item"
  done
}

run_env_cmd() {
  local -a env_args=()
  while [[ "$#" -gt 0 && "$1" == *=* ]]; do
    env_args+=("$1")
    shift
  done
  echo "+ env ${env_args[*]} $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    env "${env_args[@]}" "$@"
  fi
}

validate_candidate() {
  local candidate="$1"
  local model gate
  model="$(candidate_model_path "$candidate")"
  gate="$(candidate_gate_path "$candidate")"
  [[ -f "$model/config.json" ]] || { echo "[error] missing model config: $model/config.json" >&2; return 2; }
  [[ -f "$model/tokenizer_config.json" ]] || { echo "[error] missing tokenizer config: $model/tokenizer_config.json" >&2; return 2; }
  [[ -f "$gate" ]] || { echo "[error] missing gate checkpoint: $gate" >&2; return 2; }
}

list_candidates() {
  while read -r candidate; do
    local model gate model_name
    model="$(candidate_model_path "$candidate")"
    gate="$(candidate_gate_path "$candidate")"
    model_name="$(candidate_model_name "$candidate")"
    validate_candidate "$candidate"
    printf '%s\n  model: %s\n  gate:  %s\n  name:  %s\n' "$candidate" "$model" "$gate" "$model_name"
  done < <(candidate_list)
}

run_tool_memory() {
  while read -r candidate; do
    local model model_name summary_dir
    validate_candidate "$candidate"
    model="$(candidate_model_path "$candidate")"
    model_name="$(candidate_model_name "$candidate")"
    summary_dir="$ROOT/$candidate/$RUN_ID/tool_memory"
    run_env_cmd \
      RUN_TOOL=1 \
      RUN_MEMORY=1 \
      RUN_CODE=0 \
      TOOL_GPU="$TOOL_GPU" \
      TOOL_PORT="$TOOL_PORT" \
      TOOL_CATEGORIES="$TOOL_CATEGORIES" \
      MEMORY_GPU_IDS="$MEMORY_GPU_IDS" \
      MEMORY_TP="$MEMORY_TP" \
      MEMORY_DATASETS="$MEMORY_DATASETS" \
      RUN_ID="$RUN_ID" \
      EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      ROOT="$ROOT" \
      SUMMARY_DIR="$summary_dir" \
      bash skill/command/run_full_eval_suite.sh "$model" "$model_name"
  done < <(candidate_list)
}

run_code() {
  while read -r candidate; do
    local model model_name summary_dir
    validate_candidate "$candidate"
    model="$(candidate_model_path "$candidate")"
    model_name="$(candidate_model_name "$candidate")"
    summary_dir="$ROOT/$candidate/$RUN_ID/code"
    run_env_cmd \
      RUN_TOOL=0 \
      RUN_MEMORY=0 \
      RUN_CODE=1 \
      CODE_GPU_GROUPS="$CODE_GPU_GROUPS" \
      CODE_MAX_TEST="$CODE_MAX_TEST" \
      CODE_MAX_GENERATION_TOKEN="$CODE_MAX_GENERATION_TOKEN" \
      RUN_ID="$RUN_ID" \
      EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      ROOT="$ROOT" \
      SUMMARY_DIR="$summary_dir" \
      bash skill/command/run_full_eval_suite.sh "$model" "$model_name"
  done < <(candidate_list)
}

case "$PHASE" in
  list)
    list_candidates
    ;;
  tool_memory)
    run_tool_memory
    ;;
  code)
    run_code
    ;;
  all)
    run_tool_memory
    run_code
    ;;
  *)
    echo "[error] unknown PHASE=$PHASE" >&2
    usage >&2
    exit 2
    ;;
esac

echo "[done] PHASE=$PHASE CANDIDATES=$CANDIDATES DRY_RUN=$DRY_RUN"
