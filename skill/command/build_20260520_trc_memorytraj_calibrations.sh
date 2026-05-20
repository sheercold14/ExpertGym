#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
OUT="${OUT:-$ROOT/data/calibration/20260520_trc_round3_memorytraj}"

TOOL_PAPER96="$ROOT/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl"
TOOL_L4="$ROOT/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_official_answer_expert_rollouts_seed20260519.jsonl"
MEMORY_PAPER96="$ROOT/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_L5="$ROOT/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_expert_success_rollouts_seed20260519.jsonl"
RF_EVAL_19="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260519.jsonl"
RF_EVAL_18="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260518.jsonl"
RF_EVAL_17="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260517.jsonl"
RF_OLD="$ROOT/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl"

cd "$REPO"
mkdir -p "$OUT"

build_one() {
  local name="$1"
  local max_update_turns="$2"
  local turn_policy="$3"
  "$PY" scripts/trc/build_trc_calibration_v1.py \
    --output-dir "$OUT/$name" \
    --per-task 32 \
    --tool-rollout "$TOOL_L4" \
    --tool-rollout "$TOOL_PAPER96" \
    --memory-rollout "$MEMORY_PAPER96" \
    --code-rollout "$CODE_L5" \
    --code-rollout "$RF_EVAL_19" \
    --code-rollout "$RF_EVAL_18" \
    --code-rollout "$RF_EVAL_17" \
    --code-rollout "$RF_OLD" \
    --include-sample-expert reasonflux \
    --exclude-sample-expert deepseek \
    --memory-response-source trajectory-turns \
    --memory-trajectory-max-update-turns "$max_update_turns" \
    --memory-trajectory-turn-policy "$turn_policy"
}

build_one "mtr_full_toolaug_code_rf" 0 uniform
build_one "mtr_uniform4_toolaug_code_rf" 4 uniform
build_one "mtr_late3_toolaug_code_rf" 3 late

echo "[done] TRC memory-trajectory calibration root: $OUT"
