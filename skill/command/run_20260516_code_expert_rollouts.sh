#!/usr/bin/env bash
set -euo pipefail

# Generate isolated code-task expert rollouts for the 2026-05-16 OPD augmentation.
# Each output is a standalone JSONL under OUT_DIR and can be passed to
# DYNAMIC_OPD_EXPERT_ROLLOUT together with the unchanged tool/memory expert files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

POLICY="${POLICY:-code}"
OUT_DIR="${OUT_DIR:-$ROOT/data/calibration/20260516_code_opd_aug}"
SEED_MANIFEST="${SEED_MANIFEST:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
SEED_VALUE="${SEED_VALUE:-20260516}"
GPU_LIST="${GPU_LIST:-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CODE_MAX_NEW_TOKENS="${CODE_MAX_NEW_TOKENS:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
PROGRESS_EVERY="${PROGRESS_EVERY:-4}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_DIR"

model_for_policy() {
  case "$1" in
    code) echo "/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B" ;;
    deepseek) echo "/mnt/cache/wuruixiao/models/DeepSeek-R1-Distill-Qwen-7B" ;;
    memory) echo "/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B" ;;
    *) echo "[error] POLICY must be code, deepseek, or memory; got $1" >&2; return 2 ;;
  esac
}

name_for_policy() {
  case "$1" in
    code) echo "reasonflux_coder7b" ;;
    deepseek) echo "deepseek_r1_distill_qwen7b" ;;
    memory) echo "rl_memoryagent7b" ;;
  esac
}

run_one() {
  local policy="$1"
  local model
  local safe_name
  model="$(model_for_policy "$policy")"
  safe_name="$(name_for_policy "$policy")"
  local output="$OUT_DIR/code_expert_${safe_name}_s${SAMPLES_PER_PROMPT}_seed${SEED_VALUE}.jsonl"
  local run_id="code-expert-${safe_name}-paper96-s${SAMPLES_PER_PROMPT}-seed${SEED_VALUE}"
  echo "[run] policy=$policy model=$model output=$output gpu=$GPU_LIST"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PY" scripts/train/opvec_collect_vllm_rollouts.py \
    --config configs/gated_grpo.yaml \
    --seed-manifest "$SEED_MANIFEST" \
    --policy-model "$model" \
    --policy-id "$run_id" \
    --run-id "$run_id" \
    --output "$output" \
    --tasks code \
    --num-prompts 32 \
    --use-manifest-order \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --code-max-new-tokens "$CODE_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --vllm-batch-size "$VLLM_BATCH_SIZE" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$SEED_VALUE" \
    --no-gate-values \
    --stream-output \
    --progress-every "$PROGRESS_EVERY"
  "$PY" - "$output" <<'PY'
import collections
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
hist = collections.Counter()
covered = 0
for row in rows:
    positives = sum(float(sample.get("reward_train", sample.get("reward", 0.0))) >= 1.0 for sample in row.get("samples", []))
    hist[positives] += 1
    covered += positives > 0
summary = {
    "format": "code_expert_rollout_coverage_v1",
    "rollout": str(path),
    "rows": len(rows),
    "covered_prompts": covered,
    "coverage": covered / len(rows) if rows else 0.0,
    "positive_count_histogram": dict(sorted(hist.items())),
}
summary_path = path.with_suffix(".coverage.json")
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY
}

if [[ "$POLICY" == "all" ]]; then
  for item in code deepseek memory; do
    run_one "$item"
  done
else
  run_one "$POLICY"
fi
