#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]]; then
  cat <<'EOF'
Usage:
  bash skill/command/run_20260518_p1_candidate_eval6.sh <candidate>

Candidates:
  main_global_i7   main_global iter007 gate, evaluated via iter008/baked_policy
  main_global_i3   main_global iter003 gate, evaluated via iter004/baked_policy
  main_gc_i7       main_gc iter007 gate, evaluated via iter008/baked_policy
  main_gc_i5       main_gc iter005 gate, evaluated via iter006/baked_policy
  opd_gc_i4        OPD-only iter004 gate, evaluated via iter005/baked_policy
  init1_gc_i3      init1 upper-init iter003 gate, evaluated via iter004/baked_policy

Examples:
  RUN_TOOL=1 RUN_MEMORY=1 RUN_CODE=1 TOOL_GPU=0 MEMORY_GPU_IDS=1 CODE_GPU_GROUPS="[[2,3]]" \
    bash skill/command/run_20260518_p1_candidate_eval6.sh main_global_i3

Scheduling rule:
  BFCL Tool uses shared config/.env. Do not run two RUN_TOOL=1 jobs concurrently.
EOF
  exit 0
fi

CANDIDATE="$1"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
SUMMARY_ROOT="${SUMMARY_ROOT:-/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-expertgym-p1-evaltarget-20260518}"

case "$CANDIDATE" in
  main_global_i7)
    MODEL_PATH="/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_008/baked_policy"
    MODEL_NAME="expertgym-p1-main-global-i7"
    RUN_ID="${RUN_ID:-expertgym_p1_main_global_i7_eval6_20260518}"
    ;;
  main_global_i3)
    MODEL_PATH="/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_004/baked_policy"
    MODEL_NAME="expertgym-p1-main-global-i3"
    RUN_ID="${RUN_ID:-expertgym_p1_main_global_i3_eval6_20260518}"
    ;;
  main_gc_i5)
    MODEL_PATH="/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_006/baked_policy"
    MODEL_NAME="expertgym-p1-main-gc-i5"
    RUN_ID="${RUN_ID:-expertgym_p1_main_gc_i5_eval6_20260518}"
    ;;
  main_gc_i7)
    MODEL_PATH="/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_008/baked_policy"
    MODEL_NAME="expertgym-p1-main-gc-i7"
    RUN_ID="${RUN_ID:-expertgym_p1_main_gc_i7_eval6_20260518}"
    ;;
  opd_gc_i4)
    MODEL_PATH="/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_opd_gc_c033_evaltarget_fast_20260518/iter_005/baked_policy"
    MODEL_NAME="expertgym-p1-opd-gc-i4"
    RUN_ID="${RUN_ID:-expertgym_p1_opd_gc_i4_eval6_20260518}"
    ;;
  init1_gc_i3)
    MODEL_PATH="/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_004/baked_policy"
    MODEL_NAME="expertgym-p1-init1-gc-i3"
    RUN_ID="${RUN_ID:-expertgym_p1_init1_gc_i3_eval6_20260518}"
    ;;
  *)
    echo "[error] unknown candidate: $CANDIDATE" >&2
    exit 2
    ;;
esac

SUMMARY_DIR="${SUMMARY_DIR:-$SUMMARY_ROOT/$CANDIDATE/$RUN_ID}"

echo "[candidate] $CANDIDATE"
echo "[model] $MODEL_PATH"
echo "[model_name] $MODEL_NAME"
echo "[run_id] $RUN_ID"
echo "[summary_dir] $SUMMARY_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
RUN_ID="$RUN_ID" \\
EXPERIMENT_NAME="$EXPERIMENT_NAME" \\
SUMMARY_DIR="$SUMMARY_DIR" \\
bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME"
EOF
  exit 0
fi

export RUN_ID
export EXPERIMENT_NAME
export SUMMARY_DIR

bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME"
