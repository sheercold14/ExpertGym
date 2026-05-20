#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
OUT="${OUT:-$ROOT/data/calibration/20260519_trc_round1}"

TOOL_PAPER96="$ROOT/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl"
MEMORY_PAPER96="$ROOT/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl"
CODE_L5="$ROOT/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_expert_success_rollouts_seed20260519.jsonl"
TOOL_L4="$ROOT/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_official_answer_expert_rollouts_seed20260519.jsonl"

RF_EVAL_19="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260519.jsonl"
RF_EVAL_18="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260518.jsonl"
RF_EVAL_17="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260517.jsonl"
RF_OLD="$ROOT/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl"

DS_EVAL_19="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260519.jsonl"
DS_EVAL_18="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260518.jsonl"
DS_EVAL_17="$ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260517.jsonl"
DS_OLD="$ROOT/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl"

cd "$REPO"
mkdir -p "$OUT"

"$PY" scripts/trc/build_trc_calibration_v1.py \
  --output-dir "$OUT/e1_code_eval_rf" \
  --per-task 32 \
  --tool-rollout "$TOOL_PAPER96" \
  --memory-rollout "$MEMORY_PAPER96" \
  --code-rollout "$CODE_L5" \
  --code-rollout "$RF_EVAL_19" \
  --code-rollout "$RF_EVAL_18" \
  --code-rollout "$RF_EVAL_17" \
  --code-rollout "$RF_OLD" \
  --include-sample-expert reasonflux \
  --exclude-sample-expert deepseek

"$PY" scripts/trc/build_trc_calibration_v1.py \
  --output-dir "$OUT/e2_code_eval_multiteacher" \
  --per-task 32 \
  --tool-rollout "$TOOL_PAPER96" \
  --memory-rollout "$MEMORY_PAPER96" \
  --code-rollout "$CODE_L5" \
  --code-rollout "$DS_EVAL_19" \
  --code-rollout "$DS_EVAL_18" \
  --code-rollout "$DS_EVAL_17" \
  --code-rollout "$RF_EVAL_19" \
  --code-rollout "$RF_EVAL_18" \
  --code-rollout "$RF_EVAL_17" \
  --code-rollout "$RF_OLD" \
  --code-rollout "$DS_OLD"

"$PY" scripts/trc/build_trc_calibration_v1.py \
  --output-dir "$OUT/e3_tool_code_eval_multiteacher" \
  --per-task 32 \
  --tool-rollout "$TOOL_L4" \
  --tool-rollout "$TOOL_PAPER96" \
  --memory-rollout "$MEMORY_PAPER96" \
  --code-rollout "$CODE_L5" \
  --code-rollout "$DS_EVAL_19" \
  --code-rollout "$DS_EVAL_18" \
  --code-rollout "$DS_EVAL_17" \
  --code-rollout "$RF_EVAL_19" \
  --code-rollout "$RF_EVAL_18" \
  --code-rollout "$RF_EVAL_17" \
  --code-rollout "$RF_OLD" \
  --code-rollout "$DS_OLD"

echo "[done] TRC round1 calibration root: $OUT"
