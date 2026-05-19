#!/usr/bin/env bash
set -euo pipefail

# Generate isolated Code expert rollouts for Code P0 v3 train prompts.
# These files are used only as same-prompt OPD positives / coverage audits.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

POLICY="${POLICY:-reasonflux}"
SEED_MANIFEST="${SEED_MANIFEST:-$ROOT/data/calibration/code_p0_v3_20260518/train_code64.prompts.jsonl}"
OUT_DIR="${OUT_DIR:-$ROOT/data/calibration/code_p0_v3_20260518/expert_rollouts}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
SEED_VALUE="${SEED_VALUE:-20260518}"
GPU_LIST="${GPU_LIST:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
CODE_MAX_NEW_TOKENS="${CODE_MAX_NEW_TOKENS:-10000}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
PROGRESS_EVERY="${PROGRESS_EVERY:-4}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_DIR"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] missing required file: $1" >&2
    exit 2
  fi
}

model_for_policy() {
  case "$1" in
    reasonflux) echo "/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B" ;;
    deepseek) echo "/mnt/cache/wuruixiao/models/DeepSeek-R1-Distill-Qwen-7B" ;;
    *) echo "[error] POLICY must be reasonflux, deepseek, or all; got $1" >&2; return 2 ;;
  esac
}

safe_name_for_policy() {
  case "$1" in
    reasonflux) echo "reasonflux_coder7b" ;;
    deepseek) echo "deepseek_r1_distill_qwen7b" ;;
  esac
}

run_one() {
  local policy="$1"
  local model safe_name output run_id
  model="$(model_for_policy "$policy")"
  safe_name="$(safe_name_for_policy "$policy")"
  require_file "$SEED_MANIFEST"
  require_file "$model/config.json"

  local shard_suffix=""
  if [[ "$PROMPT_OFFSET" != "0" || "$NUM_PROMPTS" != "64" ]]; then
    shard_suffix="_offset${PROMPT_OFFSET}_n${NUM_PROMPTS}"
  fi
  output="$OUT_DIR/code_expert_${safe_name}_code_p0_v3_train64_s${SAMPLES_PER_PROMPT}${shard_suffix}_seed${SEED_VALUE}.jsonl"
  run_id="code-expert-${safe_name}-code-p0-v3-train64-s${SAMPLES_PER_PROMPT}${shard_suffix}-seed${SEED_VALUE}"
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
    --num-prompts "$NUM_PROMPTS" \
    --prompt-offset "$PROMPT_OFFSET" \
    --use-manifest-order \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --max-new-tokens "$CODE_MAX_NEW_TOKENS" \
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
    positives = 0
    for sample in row.get("samples", []):
        reward = sample.get("reward_train", sample.get("task_reward", sample.get("reward", 0.0)))
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0
        positives += int(reward >= 1.0 or bool(sample.get("success")))
    hist[positives] += 1
    covered += int(positives > 0)
summary = {
    "format": "code_p0_v3_expert_rollout_coverage_v1",
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
  for item in reasonflux deepseek; do
    run_one "$item"
  done
else
  run_one "$POLICY"
fi
