#!/usr/bin/env bash
set -euo pipefail

# Evaluate all baked checkpoints from a collapse boundary sweep.
#
# Usage:
#   GPU=0 TASKS=tool,memory bash scripts/analysis/run_collapse_sweep_eval.sh \
#     /tmp/shared-storage/ExpertGym/analysis/collapse_sweep/tool_sweep
#
# Environment:
#   GPU=0              GPU index for eval
#   TASKS=tool,memory  Which tasks to eval (tool, memory, code; comma-separated)
#   SKIP_EXISTING=1    Skip if results already exist
#   DRY_RUN=0          Print commands without executing

SWEEP_DIR="${1:?Usage: bash $0 <sweep_dir>}"
GPU="${GPU:-0}"
TASKS="${TASKS:-tool,memory}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/skill/command/run_full_eval_suite.sh"

if [[ ! -f "$SWEEP_DIR/sweep_manifest.jsonl" ]]; then
  echo "[error] No sweep_manifest.jsonl in $SWEEP_DIR" >&2
  exit 1
fi

# Parse tasks
RUN_TOOL=0; RUN_MEMORY=0; RUN_CODE=0
IFS=',' read -ra TASK_LIST <<< "$TASKS"
for t in "${TASK_LIST[@]}"; do
  case "$t" in
    tool)   RUN_TOOL=1 ;;
    memory) RUN_MEMORY=1 ;;
    code)   RUN_CODE=1 ;;
    *) echo "[warn] Unknown task: $t" ;;
  esac
done

echo "=== Collapse Boundary Sweep Eval ==="
echo "Sweep dir:  $SWEEP_DIR"
echo "GPU:        $GPU"
echo "Tasks:      $TASKS"
echo "Skip existing: $SKIP_EXISTING"
echo

RESULTS_FILE="$SWEEP_DIR/eval_results.jsonl"
TOTAL=0
DONE=0

while IFS= read -r line; do
  tag=$(echo "$line" | "$PY" -c "import sys,json; print(json.loads(sys.stdin.read())['tag'])")
  checkpoint=$(echo "$line" | "$PY" -c "import sys,json; print(json.loads(sys.stdin.read())['checkpoint_dir'])")
  baked=$(echo "$line" | "$PY" -c "import sys,json; print(json.loads(sys.stdin.read())['baked'])")

  TOTAL=$((TOTAL + 1))

  if [[ "$baked" != "True" ]]; then
    echo "[$tag] not baked, skipping"
    continue
  fi

  if [[ ! -d "$checkpoint" ]]; then
    echo "[$tag] checkpoint missing: $checkpoint"
    continue
  fi

  eval_dir="$SWEEP_DIR/$tag/eval"

  if [[ "$SKIP_EXISTING" == "1" && -f "$eval_dir/done" ]]; then
    echo "[$tag] eval already done, skipping"
    DONE=$((DONE + 1))
    continue
  fi

  echo "[$tag] evaluating on GPU $GPU ..."

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] RUN_TOOL=$RUN_TOOL RUN_MEMORY=$RUN_MEMORY RUN_CODE=$RUN_CODE bash $EVAL_SCRIPT $checkpoint $tag"
    continue
  fi

  mkdir -p "$eval_dir"

  # Run eval
  env \
    RUN_TOOL="$RUN_TOOL" \
    RUN_MEMORY="$RUN_MEMORY" \
    RUN_CODE="$RUN_CODE" \
    TOOL_GPU="$GPU" \
    MEMORY_GPU_IDS="$GPU" \
    CODE_GPU_GROUPS="[[$GPU]]" \
    EXPERIMENT_NAME="collapse-sweep-$(basename "$SWEEP_DIR")" \
    bash "$EVAL_SCRIPT" "$checkpoint" "$tag" \
    > "$eval_dir/eval.log" 2>&1 || {
      echo "  [WARN] eval returned non-zero, check $eval_dir/eval.log"
    }

  # Mark done
  touch "$eval_dir/done"
  DONE=$((DONE + 1))
  echo "  done ($DONE/$TOTAL)"

done < "$SWEEP_DIR/sweep_manifest.jsonl"

echo
echo "=== Eval complete: $DONE/$TOTAL ==="
echo "Run the collector to build the results table:"
echo "  python scripts/analysis/collect_collapse_sweep_results.py $SWEEP_DIR"
