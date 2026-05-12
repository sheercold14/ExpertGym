#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash third_party/verl/examples/opvec_gated_grpo/run_smoke10_native_bridge.sh

Key overrides:
  PY=/mnt/cache/wuruixiao/miniconda3/envs/easyrl/bin/python
  GPU_LIST=0,1,2
  LIMIT=10
  SAMPLES_PER_PROMPT=2
  RUN_NAME=verl_opvec_smoke10_c033333_global_i1_seed20260511
EOF
  exit 0
fi

VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPVEC_REPO_ROOT="${OPVEC_REPO_ROOT:-$(cd "$VERL_ROOT/../.." && pwd)}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/easyrl/bin/python}"

export OPVEC_REPO_ROOT
export PYTHONPATH="$VERL_ROOT:$OPVEC_REPO_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
QB="${QB:-$ROOT/data/question_bank/ta_avgvec_c033333_hotpotqa_v1}"
SOURCE_CALIBRATION="${SOURCE_CALIBRATION:-$QB/calibration/calib100_seed20260511.prompts.jsonl}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-$ROOT/data/verl_opvec/smoke10_c033333}"
SMOKE_CALIBRATION="${SMOKE_CALIBRATION:-$SMOKE_DATA_DIR/calib10_seed20260511.prompts.jsonl}"
SMOKE_PARQUET="${SMOKE_PARQUET:-$SMOKE_DATA_DIR/calib10_seed20260511.parquet}"

RUN_NAME="${RUN_NAME:-verl_opvec_smoke10_c033333_global_i1_seed20260511}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/verl_opvec/$RUN_NAME}"
GPU_LIST="${GPU_LIST:-0,1,2}"

"$PY" "$VERL_ROOT/verl/experimental/opvec/prepare_data.py" \
  --input "$SOURCE_CALIBRATION" \
  --output "$SMOKE_CALIBRATION" \
  --parquet "$SMOKE_PARQUET" \
  --limit "${LIMIT:-10}"

cd "$OPVEC_REPO_ROOT"

START_SECONDS="$(date +%s)"
CALIBRATION="$SMOKE_CALIBRATION" \
RUN_DIR="$RUN_DIR" \
RUN_NAME="$RUN_NAME" \
GPU_LIST="$GPU_LIST" \
STRATEGY="${STRATEGY:-global}" \
NUM_ITERS="${NUM_ITERS:-1}" \
NUM_PROMPTS="${NUM_PROMPTS:-10}" \
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-2}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}" \
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}" \
MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-4096}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}" \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}" \
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-70GiB}" \
CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-180GiB}" \
PY="$PY" \
bash skill/command/run_qbank_c033333_gate_strategy.sh
END_SECONDS="$(date +%s)"

"$PY" - <<PY
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
summary_path = run_dir / "verl_smoke_summary.json"
manifest_path = run_dir / "gated_grpo_bake_vllm_loop_manifest.json"
strategy_path = run_dir / "strategy_summary.json"
payload = {
    "format": "opvec_verl_native_bridge_smoke_summary_v1",
    "run_name": "$RUN_NAME",
    "run_dir": str(run_dir),
    "smoke_calibration": "$SMOKE_CALIBRATION",
    "smoke_parquet": "$SMOKE_PARQUET",
    "gpu_list": "$GPU_LIST",
    "elapsed_seconds_wall": int("$END_SECONDS") - int("$START_SECONDS"),
}
if manifest_path.exists():
    payload["loop_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
if strategy_path.exists():
    payload["strategy_summary"] = json.loads(strategy_path.read_text(encoding="utf-8"))
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({"summary": str(summary_path), "elapsed_seconds_wall": payload["elapsed_seconds_wall"]}, ensure_ascii=False, indent=2))
PY
