#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/tmp/shared-storage/ExpertGym/baselines/qwen7b}"
MODE="${MODE:-/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
SEED="${SEED:-20260518}"
SCALING="${SCALING:-0.3333333333333333}"
TIES_KEEP_RATIO="${TIES_KEEP_RATIO:-0.2}"
DARE_DROP_RATE="${DARE_DROP_RATE:-0.8}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

METHODS="${METHODS:-task_arithmetic ties dare_ta dare_ties}"
OUT_ROOT="$ROOT/static_merges"
mkdir -p "$OUT_ROOT"

run_one() {
  local method="$1"
  local out="$OUT_ROOT/${method}_c${SCALING//./p}_k${TIES_KEEP_RATIO//./p}_d${DARE_DROP_RATE//./p}_seed${SEED}"
  local cmd=(
    "$PY" scripts/baselines/build_static_merge_baseline.py
    --mode-manifest "$MODE"
    --output-dir "$out"
    --method "$method"
    --scaling-coefficient "$SCALING"
    --ties-keep-ratio "$TIES_KEEP_RATIO"
    --dare-drop-rate "$DARE_DROP_RATE"
    --seed "$SEED"
  )
  if [[ "$FORCE" == "1" ]]; then
    cmd+=(--force)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}" 2>&1 | tee "$OUT_ROOT/${method}.build.log"
  fi
}

for method in $METHODS; do
  run_one "$method"
done
