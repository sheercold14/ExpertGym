#!/usr/bin/env bash
set -euo pipefail

# Structured Capability Gates v1.
#
# Phases:
#   PHASE=generate    build mechanism-constrained gate candidates
#   PHASE=bake        bake all candidate gates to HF checkpoints
#   PHASE=quick_eval  run Tool+Memory quick evaluation for baked candidates
#   PHASE=all         generate + bake + quick_eval

ROOT="${ROOT:-/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521}"
PHASE="${PHASE:-all}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
CONFIG="${CONFIG:-configs/gated_grpo.yaml}"
MODE_MANIFEST="${MODE_MANIFEST:-/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json}"
PROFILES="${PROFILES:-balanced,code_mid_push,code_safe}"
CANDIDATES="${CANDIDATES:-}"
CODE_MID_LAYERS="${CODE_MID_LAYERS:-8-20}"
CODE_CONFLICT_LAYERS="${CODE_CONFLICT_LAYERS:-24,27}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/shared-storage/OnPolicy/checkpoints/structured_capability_gates_v1_20260521}"
EVAL_ROOT="${EVAL_ROOT:-/tmp/shared-storage/ExpertGym/structured_capability_gates/eval/scg_v1_20260521}"
GPU_LIST="${GPU_LIST:-1}"
TOOL_GPU="${TOOL_GPU:-${GPU_LIST%%,*}}"
MEMORY_GPU_IDS="${MEMORY_GPU_IDS:-${GPU_LIST%%,*}}"
MEMORY_DATASETS="${MEMORY_DATASETS:-eval_50 eval_100}"
RUN_CODE="${RUN_CODE:-0}"

mkdir -p "$ROOT" "$CHECKPOINT_ROOT" "$EVAL_ROOT"

run_generate() {
  "$PY" scripts/attention_pauh/build_structured_capability_gates.py \
    --mode-manifest "$MODE_MANIFEST" \
    --output-dir "$ROOT" \
    --profiles "$PROFILES" \
    --code-mid-layers "$CODE_MID_LAYERS" \
    --code-conflict-layers "$CODE_CONFLICT_LAYERS"
}

candidate_names() {
  "$PY" - "$ROOT/candidate_manifest.json" "$CANDIDATES" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
allow_raw = sys.argv[2].strip()
allow = {item.strip() for item in allow_raw.split(",") if item.strip()}
for item in payload["candidates"]:
    if not allow or item["name"] in allow:
        print(item["name"])
PY
}

run_bake() {
  while read -r name; do
    [[ -n "$name" ]] || continue
    gate="$ROOT/$name/gates.json"
    output="$CHECKPOINT_ROOT/$name"
    echo "[bake] $name -> $output"
    "$PY" scripts/eval/opvec_bake_checkpoint.py \
      --config "$CONFIG" \
      --mode-manifest "$MODE_MANIFEST" \
      --gate-checkpoint "$gate" \
      --output "$output"
  done < <(candidate_names)
}

run_quick_eval() {
  while read -r name; do
    [[ -n "$name" ]] || continue
    model="$CHECKPOINT_ROOT/$name"
    summary_dir="$EVAL_ROOT/$name/quick_tool_memory"
    echo "[quick_eval] $name -> $summary_dir"
    RUN_TOOL=1 \
    RUN_MEMORY=1 \
    RUN_CODE="$RUN_CODE" \
    TOOL_GPU="$TOOL_GPU" \
    MEMORY_GPU_IDS="$MEMORY_GPU_IDS" \
    MEMORY_DATASETS="$MEMORY_DATASETS" \
    RUN_ID="scg_v1_${name}_quick_20260521" \
    EXPERIMENT_NAME="scg-v1-${name}" \
    SUMMARY_DIR="$summary_dir" \
    bash skill/command/run_full_eval_suite.sh "$model" "scg-v1-${name}"
  done < <(candidate_names)
}

case "$PHASE" in
  generate)
    run_generate
    ;;
  bake)
    run_bake
    ;;
  quick_eval)
    run_quick_eval
    ;;
  all)
    run_generate
    run_bake
    run_quick_eval
    ;;
  *)
    echo "Unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac

echo "[done] PHASE=$PHASE ROOT=$ROOT"
