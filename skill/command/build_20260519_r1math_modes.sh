#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"

CONFIG="${CONFIG:-configs/gated_grpo_4expert_r1math_layer28.yaml}"
RAW_DIR="${RAW_DIR:-$ROOT/modes/opvec4_r1math_raw_20260519}"
SCALED_DIR="${SCALED_DIR:-$ROOT/modes/opvec4_r1math_scaled_20260519}"
MATH_BASE="${MATH_BASE:-/mnt/cache/wuruixiao/models/Qwen2.5-Math-7B}"

if [[ ! -f "$MATH_BASE/model.safetensors.index.json" ]]; then
  echo "[error] missing Qwen2.5-Math-7B safetensors index: $MATH_BASE/model.safetensors.index.json" >&2
  echo "[hint] run: HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir $MATH_BASE" >&2
  exit 2
fi

echo "[build] raw correct-R1 modes -> $RAW_DIR"
"$PY" scripts/modes/build_opvec4_modes.py \
  --config "$CONFIG" \
  --output-dir "$RAW_DIR"

echo "[build] scaled correct-R1 modes -> $SCALED_DIR"
"$PY" scripts/modes/build_scaled_r1_modes.py \
  --source-manifest "$RAW_DIR/mode_manifest.json" \
  --output-dir "$SCALED_DIR"

echo "[check] manifest summary"
"$PY" - <<'PY'
import json
from pathlib import Path

for path in [
    Path("/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519/mode_manifest.json"),
    Path("/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json"),
]:
    data = json.loads(path.read_text())
    print(path)
    print("  format:", data.get("format"))
    print("  base:", data.get("base_model"))
    print("  delta_bases:", data.get("delta_bases"))
    print("  experts:", data.get("expert_names"))
    print("  entries:", len(data.get("basis_entries", [])))
    if data.get("reasoning_scale_factor") is not None:
        print("  reasoning_scale_factor:", data.get("reasoning_scale_factor"))
PY

echo "[done] $SCALED_DIR/mode_manifest.json"
