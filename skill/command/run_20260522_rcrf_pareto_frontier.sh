#!/usr/bin/env bash
set -euo pipefail

# RCRF Pareto Frontier reproduction harness.
#
# Phases:
#   PHASE=manifest        build Memory full-trajectory behavior manifest
#   PHASE=probe_memory    run Memory full-trajectory signed-utility probe
#   PHASE=atlas           build residual conflict atlas from current evidence
#   PHASE=role_route      build v12 role-routed gate from the atlas
#   PHASE=generate        generate selected gate checkpoints
#   PHASE=bake            bake selected gate checkpoints to HF checkpoints
#   PHASE=quick_eval      run Tool+Memory quick eval for selected baked checkpoints
#   PHASE=code_hurt_eval  run CURE hurt16 regression for selected baked checkpoints
#   PHASE=diagnose        rebuild atlas + operating-point delta comparison
#   PHASE=dashboard       build static residual diagnostic dashboard
#   PHASE=clusters        summarize residual conflict archetypes
#   PHASE=paper_table     build paper-facing evidence table from existing artifacts
#   PHASE=effect_table    build counterfactual residual effect table from existing artifacts
#   PHASE=ledger          build row-level residual attribution ledger
#   PHASE=validation_plan build grouped validation cards from the attribution ledger
#   PHASE=validation_interventions build minimal gate interventions for top validation cards
#   PHASE=paper_main      generate + bake + quick_eval + code_hurt_eval + diagnose + dashboard + clusters + paper_table + effect_table + ledger + validation_plan + validation_interventions
#   PHASE=all             generate + bake + quick_eval
#
# Use DRY_RUN=1 to print commands without executing them.

PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
CURE_PY="${CURE_PY:-/mnt/cache/wuruixiao/miniconda3/envs/CURE/bin/python}"
CONFIG="${CONFIG:-configs/gated_grpo.yaml}"
MODE_MANIFEST="${MODE_MANIFEST:-/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json}"
CANDIDATES="${CANDIDATES:-v8,v9,v10,v11}"

ROOT="${ROOT:-/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521}"
GATE_ROOT="${GATE_ROOT:-$ROOT/contrast_gates}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/shared-storage/OnPolicy/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite}"

BASE_GATES="${BASE_GATES:-/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json}"
TOOL_MEMORY_SIGNATURE_SUMMARY="${TOOL_MEMORY_SIGNATURE_SUMMARY:-$ROOT/probes/tool_memory_positive_signature_s32_20260521/signed_utility_summary.json}"
MEMORY_FULLTRAJ_DIR="${MEMORY_FULLTRAJ_DIR:-$ROOT/behavior_span_manifests/memory_fulltraj_20260521}"
MEMORY_FULLTRAJ_SUMMARY="${MEMORY_FULLTRAJ_SUMMARY:-$ROOT/probes/memory_fulltraj_positive_s32_20260521/signed_utility_summary.json}"
ATLAS_OUTPUT_DIR="${ATLAS_OUTPUT_DIR:-$ROOT/analysis/rcrf_conflict_atlas_20260522}"
MEMORY_SOURCE="${MEMORY_SOURCE:-/tmp/shared-storage/ExpertGym/eval/loss_qp_equal_weight/hotpotqa/hotpotqa_inference_results.jsonl}"
MEMORY_SUMMARY_JSON="${MEMORY_SUMMARY_JSON:-/tmp/shared-storage/ExpertGym/eval/loss_qp_equal_weight/hotpotqa/evaluation_summary.json}"

CONTRAST_LIVEBENCH_PROMPT="${CONTRAST_LIVEBENCH_PROMPT:-$ROOT/contrast/livebench_prompt_alllayers_s16_20260521/contrast_module_summary.jsonl}"
CONTRAST_LIVEBENCH_REASONING="${CONTRAST_LIVEBENCH_REASONING:-$ROOT/contrast/livebench_reasoning_alllayers_s16_20260521/contrast_module_summary.jsonl}"
CONTRAST_LIVECODEBENCH_CODE="${CONTRAST_LIVECODEBENCH_CODE:-$ROOT/contrast/livecodebench_code_alllayers_s16_20260521/contrast_module_summary.jsonl}"
CONTRAST_LIVECODEBENCH_PROMPT="${CONTRAST_LIVECODEBENCH_PROMPT:-$ROOT/contrast/livecodebench_prompt_alllayers_s16_20260521/contrast_module_summary.jsonl}"

TOOL_GPU="${TOOL_GPU:-0}"
TOOL_PORT="${TOOL_PORT:-8151}"
MEMORY_GPU_IDS="${MEMORY_GPU_IDS:-$TOOL_GPU}"
MEMORY_DATASETS="${MEMORY_DATASETS:-eval_50}"
CODE_GPU="${CODE_GPU:-2}"
CODE_DATASETS="${CODE_DATASETS:-LiveBenchCodeHurtRcrfVsTa16 LiveCodeBenchCodeHurtRcrfVsTa16}"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
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

candidate_list() {
  IFS=',' read -ra items <<< "$CANDIDATES"
  for item in "${items[@]}"; do
    item="${item// /}"
    [[ -n "$item" ]] && echo "$item"
  done
}

gate_dir_for() {
  case "$1" in
    v8) echo "$GATE_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_v8" ;;
    v9) echo "$GATE_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9" ;;
    v10) echo "$GATE_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10" ;;
    v11) echo "$GATE_ROOT/rcrf_code_spanaware_tmpos_s32_tasktyped_v11" ;;
    v12) echo "$GATE_ROOT/rcrf_role_routed_v12" ;;
    v13) echo "$GATE_ROOT/rcrf_role_routed_positive_only_v13" ;;
    v14_code_half) echo "$GATE_ROOT/rcrf_v9_code_half_v14" ;;
    v15_code_zero) echo "$GATE_ROOT/rcrf_v9_code_zero_v15" ;;
    v16_source_suppress) echo "$GATE_ROOT/rcrf_source_conflict_suppress_v16" ;;
    v17_source_route) echo "$GATE_ROOT/rcrf_source_conflict_route_v17" ;;
    v18_rcf_bc) echo "$GATE_ROOT/residual_capability_field_behavior_constraints_v18" ;;
    v19_archetype_consistency) echo "$GATE_ROOT/rcrf_archetype_consistency_v19" ;;
    v20_code_noise_half) echo "$GATE_ROOT/rcrf_code_noise_weak_half_v20" ;;
    v21_code_noise_zero) echo "$GATE_ROOT/rcrf_code_noise_weak_zero_v21" ;;
    v22_code_negative_noise_half) echo "$GATE_ROOT/rcrf_code_negative_noise_half_v22" ;;
    v23_code_weak_half) echo "$GATE_ROOT/rcrf_code_weak_half_v23" ;;
    *)
      echo "[error] unknown candidate: $1" >&2
      return 2
      ;;
  esac
}

checkpoint_for() {
  case "$1" in
    v8) echo "$CHECKPOINT_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_v8" ;;
    v9) echo "$CHECKPOINT_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9" ;;
    v10) echo "$CHECKPOINT_ROOT/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10" ;;
    v11) echo "$CHECKPOINT_ROOT/rcrf_code_spanaware_tmpos_s32_tasktyped_v11" ;;
    v12) echo "$CHECKPOINT_ROOT/rcrf_role_routed_v12" ;;
    v13) echo "$CHECKPOINT_ROOT/rcrf_role_routed_positive_only_v13" ;;
    v14_code_half) echo "$CHECKPOINT_ROOT/rcrf_v9_code_half_v14" ;;
    v15_code_zero) echo "$CHECKPOINT_ROOT/rcrf_v9_code_zero_v15" ;;
    v16_source_suppress) echo "$CHECKPOINT_ROOT/rcrf_source_conflict_suppress_v16" ;;
    v17_source_route) echo "$CHECKPOINT_ROOT/rcrf_source_conflict_route_v17" ;;
    v18_rcf_bc) echo "$CHECKPOINT_ROOT/residual_capability_field_behavior_constraints_v18" ;;
    v19_archetype_consistency) echo "$CHECKPOINT_ROOT/rcrf_archetype_consistency_v19" ;;
    v20_code_noise_half) echo "$CHECKPOINT_ROOT/rcrf_code_noise_weak_half_v20" ;;
    v21_code_noise_zero) echo "$CHECKPOINT_ROOT/rcrf_code_noise_weak_zero_v21" ;;
    v22_code_negative_noise_half) echo "$CHECKPOINT_ROOT/rcrf_code_negative_noise_half_v22" ;;
    v23_code_weak_half) echo "$CHECKPOINT_ROOT/rcrf_code_weak_half_v23" ;;
    *)
      echo "[error] unknown candidate: $1" >&2
      return 2
      ;;
  esac
}

common_gate_args=(
  --base-gates "$BASE_GATES"
  --contrast-summary "$CONTRAST_LIVEBENCH_PROMPT"
  --contrast-summary "$CONTRAST_LIVEBENCH_REASONING"
  --contrast-summary "$CONTRAST_LIVECODEBENCH_CODE"
  --contrast-summary "$CONTRAST_LIVECODEBENCH_PROMPT"
  --normalization per-file
  --scale-quantile 0.9
  --max-delta 0.05
  --min-abs-score 0.1
  --aggregation conservative
  --conflict-penalty 0.35
  --min-coeff 0.55
  --max-coeff 1.12
  --preserve-summary "$TOOL_MEMORY_SIGNATURE_SUMMARY"
  --preserve-summary "$MEMORY_FULLTRAJ_SUMMARY"
  --preserve-task tool
  --preserve-task memory
  --preserve-min-normalized-utility 0.4
  --preserve-min-positive-fraction 0.5
  --preserve-negative-scale 0.0
  --harm-veto-summary "$TOOL_MEMORY_SIGNATURE_SUMMARY"
  --harm-veto-summary "$MEMORY_FULLTRAJ_SUMMARY"
  --harm-veto-task tool
  --harm-veto-task memory
  --harm-veto-min-normalized-harm 0.4
)

generate_one() {
  local name="$1"
  local output_dir
  output_dir="$(gate_dir_for "$name")"
  case "$name" in
    v8)
      run_cmd "$PY" scripts/attention_pauh/build_contrast_aware_residual_gates.py \
        "${common_gate_args[@]}" \
        --output-dir "$output_dir" \
        --harm-veto-positive-scale 0.0
      ;;
    v9)
      run_cmd "$PY" scripts/attention_pauh/build_contrast_aware_residual_gates.py \
        "${common_gate_args[@]}" \
        --output-dir "$output_dir" \
        --harm-veto-positive-scale 0.5
      ;;
    v10)
      run_cmd "$PY" scripts/attention_pauh/build_contrast_aware_residual_gates.py \
        "${common_gate_args[@]}" \
        --output-dir "$output_dir" \
        --harm-veto-positive-scale-mode evidence-ratio
      ;;
    v11)
      run_cmd "$PY" scripts/attention_pauh/build_contrast_aware_residual_gates.py \
        "${common_gate_args[@]}" \
        --output-dir "$output_dir" \
        --harm-veto-positive-scale 0.0 \
        --harm-veto-task-positive-scale tool=0.0 \
        --harm-veto-task-positive-scale memory=0.5
      ;;
    v12)
      run_cmd "$PY" scripts/analysis/build_rcrf_role_routed_gates.py \
        --output-dir "$output_dir"
      ;;
    v13)
      run_cmd "$PY" scripts/analysis/build_rcrf_role_routed_gates.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_role_routed_positive_only_v13 \
        --code-negative-action hold \
        --protected-harm-action hold
      ;;
    v14_code_half)
      run_cmd "$PY" scripts/analysis/build_expert_scaled_gate_ablation.py \
        --base-gate "$(gate_dir_for v9)/gates.json" \
        --output-dir "$output_dir" \
        --variant-name rcrf_v9_code_half_v14 \
        --expert code \
        --scale 0.5
      ;;
    v15_code_zero)
      run_cmd "$PY" scripts/analysis/build_expert_scaled_gate_ablation.py \
        --base-gate "$(gate_dir_for v9)/gates.json" \
        --output-dir "$output_dir" \
        --variant-name rcrf_v9_code_zero_v15 \
        --expert code \
        --set-value 0.0
      ;;
    v16_source_suppress)
      run_cmd "$PY" scripts/analysis/build_rcrf_role_routed_gates.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_source_conflict_suppress_v16 \
        --code-negative-action hold \
        --protected-harm-action hold \
        --source-conflict-action suppress-dominant \
        --source-conflict-min-strength 1.0 \
        --source-conflict-dominance-ratio 1.25 \
        --source-conflict-protected-support-action hold
      ;;
    v17_source_route)
      run_cmd "$PY" scripts/analysis/build_rcrf_role_routed_gates.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_source_conflict_route_v17 \
        --code-negative-action hold \
        --protected-harm-action hold \
        --source-conflict-action route-dominant \
        --source-conflict-min-strength 1.0 \
        --source-conflict-dominance-ratio 1.25 \
        --source-conflict-protected-support-action hold
      ;;
    v18_rcf_bc)
      run_cmd "$PY" scripts/attention_pauh/build_contrast_aware_residual_gates.py \
        "${common_gate_args[@]}" \
        --output-dir "$output_dir" \
        --harm-veto-positive-scale 0.5
      ;;
    v19_archetype_consistency)
      run_cmd "$PY" scripts/analysis/build_rcrf_archetype_policy_gates.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_archetype_consistency_v19
      ;;
    v20_code_noise_half)
      run_cmd "$PY" scripts/analysis/build_rcrf_archetype_scaled_gate_ablation.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_code_noise_weak_half_v20 \
        --expert code \
        --archetype code_negative_noise \
        --archetype weak_or_uninformative \
        --scale-coefficient 0.5
      ;;
    v21_code_noise_zero)
      run_cmd "$PY" scripts/analysis/build_rcrf_archetype_scaled_gate_ablation.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_code_noise_weak_zero_v21 \
        --expert code \
        --archetype code_negative_noise \
        --archetype weak_or_uninformative \
        --set-coefficient 0.0
      ;;
    v22_code_negative_noise_half)
      run_cmd "$PY" scripts/analysis/build_rcrf_archetype_scaled_gate_ablation.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_code_negative_noise_half_v22 \
        --expert code \
        --archetype code_negative_noise \
        --scale-coefficient 0.5
      ;;
    v23_code_weak_half)
      run_cmd "$PY" scripts/analysis/build_rcrf_archetype_scaled_gate_ablation.py \
        --output-dir "$output_dir" \
        --variant-name rcrf_code_weak_half_v23 \
        --expert code \
        --archetype weak_or_uninformative \
        --scale-coefficient 0.5
      ;;
  esac
}

run_manifest() {
  run_cmd "$PY" scripts/analysis/build_behavior_span_manifest.py \
    --source memory "$MEMORY_SOURCE" \
    --summary-json "$MEMORY_SUMMARY_JSON" \
    --output-dir "$MEMORY_FULLTRAJ_DIR" \
    --max-positive-per-task 32 \
    --max-negative-per-task 32 \
    --memory-response-mode full-trajectory
}

run_probe_memory() {
  run_cmd "$PY" scripts/attention_pauh/probe_signed_utility.py \
    --mode-manifest "$MODE_MANIFEST" \
    --trajectory-jsonl "$MEMORY_FULLTRAJ_DIR/behavior_positive.jsonl" \
    --output-dir "$(dirname "$MEMORY_FULLTRAJ_SUMMARY")" \
    --tasks memory \
    --experts tool,memory,code \
    --scope all-linear \
    --span response \
    --samples-per-task 32 \
    --max-seq-length 8192 \
    --response-tail-tokens 0 \
    --write-row-details
}

run_atlas() {
  run_cmd "$PY" scripts/analysis/build_rcrf_conflict_atlas.py \
    --output-dir "$ATLAS_OUTPUT_DIR"
}

run_role_route() {
  run_cmd "$PY" scripts/analysis/build_rcrf_role_routed_gates.py \
    --output-dir "$(gate_dir_for v12)"
}

run_generate() {
  while read -r name; do
    generate_one "$name"
  done < <(candidate_list)
}

run_bake() {
  while read -r name; do
    local gate output
    gate="$(gate_dir_for "$name")/gates.json"
    output="$(checkpoint_for "$name")"
    run_cmd "$PY" scripts/eval/opvec_bake_checkpoint.py \
      --config "$CONFIG" \
      --mode-manifest "$MODE_MANIFEST" \
      --gate-checkpoint "$gate" \
      --output "$output"
  done < <(candidate_list)
}

run_quick_eval() {
  while read -r name; do
    local model summary_dir
    model="$(checkpoint_for "$name")"
    summary_dir="$EVAL_ROOT/$(basename "$model")/quick_tool_memory"
    run_env_cmd \
      RUN_TOOL=1 \
      RUN_MEMORY=1 \
      RUN_CODE=0 \
      TOOL_GPU="$TOOL_GPU" \
      TOOL_PORT="$TOOL_PORT" \
      MEMORY_GPU_IDS="$MEMORY_GPU_IDS" \
      MEMORY_DATASETS="$MEMORY_DATASETS" \
      RUN_ID="quick_tool_memory" \
      EXPERIMENT_NAME="rcrf" \
      ROOT="/tmp/shared-storage/ExpertGym/rcrf" \
      SUMMARY_DIR="$summary_dir" \
      bash skill/command/run_full_eval_suite.sh "$model" "$(basename "$model")"
  done < <(candidate_list)
}

run_code_hurt_eval() {
  while read -r name; do
    local model
    model="$(checkpoint_for "$name")"
    run_env_cmd \
      MODEL="$model" \
      GPU="$CODE_GPU" \
      PYTHON_BIN="$CURE_PY" \
      DATASETS="$CODE_DATASETS" \
      bash scripts/eval/run_cure_code_hurt_eval.sh
  done < <(candidate_list)
}

run_diagnose() {
  run_atlas
  run_cmd "$PY" scripts/analysis/compare_rcrf_operating_points.py
}

run_dashboard() {
  run_cmd "$PY" scripts/analysis/build_rcrf_diagnostic_dashboard.py
}

run_clusters() {
  run_cmd "$PY" scripts/analysis/build_rcrf_conflict_clusters.py
}

run_paper_table() {
  run_cmd "$PY" scripts/analysis/build_rcrf_paper_evidence_table.py
}

run_effect_table() {
  run_cmd "$PY" scripts/analysis/build_rcrf_counterfactual_effect_table.py
}

run_ledger() {
  run_cmd "$PY" scripts/analysis/build_rcrf_attribution_ledger.py
}

run_validation_plan() {
  run_cmd "$PY" scripts/analysis/build_rcrf_validation_plan.py
}

run_validation_interventions() {
  run_cmd "$PY" scripts/analysis/build_rcrf_validation_interventions.py
}

case "$PHASE" in
  manifest)
    run_manifest
    ;;
  probe_memory)
    run_probe_memory
    ;;
  atlas)
    run_atlas
    ;;
  role_route)
    run_role_route
    ;;
  generate)
    run_generate
    ;;
  bake)
    run_bake
    ;;
  quick_eval)
    run_quick_eval
    ;;
  code_hurt_eval)
    run_code_hurt_eval
    ;;
  diagnose)
    run_diagnose
    ;;
  dashboard)
    run_dashboard
    ;;
  clusters)
    run_clusters
    ;;
  paper_table)
    run_paper_table
    ;;
  effect_table)
    run_effect_table
    ;;
  ledger)
    run_ledger
    ;;
  validation_plan)
    run_validation_plan
    ;;
  validation_interventions)
    run_validation_interventions
    ;;
  paper_main)
    run_generate
    run_bake
    run_quick_eval
    run_code_hurt_eval
    run_diagnose
    run_dashboard
    run_clusters
    run_paper_table
    run_effect_table
    run_ledger
    run_validation_plan
    run_validation_interventions
    ;;
  all)
    run_generate
    run_bake
    run_quick_eval
    ;;
  *)
    echo "[error] unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac

echo "[done] PHASE=$PHASE CANDIDATES=$CANDIDATES DRY_RUN=$DRY_RUN"
