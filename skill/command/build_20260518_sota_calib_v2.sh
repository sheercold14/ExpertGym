#!/usr/bin/env bash
set -euo pipefail

# Build a disjoint SOTA-oriented calibration bank:
#   train128  = tool32 + memory48 + code48
#   monitor64 = tool16 + memory24 + code24
#   guard64   = tool16 + memory24 + code24
#
# This script only creates data manifests.  It does not start training and it
# does not overwrite paper96/eval_targeted96 artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/calibration/sota_calib_v2_20260518}"
COMPONENT_DIR="$OUTPUT_DIR/components"
SEED="${SEED:-20260518}"

PAPER96="${PAPER96:-$ROOT/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl}"
CASE_CANDIDATES="${CASE_CANDIDATES:-$ROOT/analysis/eval_case_browser/bfcl_live_calibration_candidates.jsonl}"
CODECONTESTS="${CODECONTESTS:-/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json}"
HOTPOTQA_TRAIN="${HOTPOTQA_TRAIN:-/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/hotpotqa/hotpotqa_train_32k.parquet}"
CHUNK_TOKENIZER="${CHUNK_TOKENIZER:-/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct}"

mkdir -p "$COMPONENT_DIR"

echo "[build] output_dir=$OUTPUT_DIR"
echo "[build] seed=$SEED"
echo "[build] paper96=$PAPER96"
echo "[build] case_candidates=$CASE_CANDIDATES"
echo "[build] codecontests=$CODECONTESTS"
echo "[build] hotpotqa_train=$HOTPOTQA_TRAIN"

"$PY" scripts/data/build_eval_targeted_calibration.py \
  --case-candidates "$CASE_CANDIDATES" \
  --paper96 "$PAPER96" \
  --codecontests "$CODECONTESTS" \
  --output-dir "$COMPONENT_DIR/tool_code_pool" \
  --tool-source-count 32 \
  --tool-synthetic-count 32 \
  --memory-count 0 \
  --code-count 96 \
  --code-source-count 32 \
  --code-targeted-count 64 \
  --seed "$SEED"

"$PY" scripts/data/build_hotpotqa_memory_seed_manifest.py \
  --input "$HOTPOTQA_TRAIN" \
  --output "$COMPONENT_DIR/memory_pool.prompts.jsonl" \
  --limit 96 \
  --seed "$SEED" \
  --split sota_calib_v2_pool \
  --chunk-tokenizer "$CHUNK_TOKENIZER"

"$PY" scripts/data/partition_calibration_bank.py \
  --input "$COMPONENT_DIR/tool_code_pool/eval_targeted96.prompts.jsonl" \
  --input "$COMPONENT_DIR/memory_pool.prompts.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --split-spec "train128:tool=32,memory=48,code=48" \
  --split-spec "monitor64:tool=16,memory=24,code=24" \
  --split-spec "guard64:tool=16,memory=24,code=24" \
  --seed "$SEED" \
  --tag "sota_calib_v2_20260518"

echo "[done] train:   $OUTPUT_DIR/train128.prompts.jsonl"
echo "[done] monitor: $OUTPUT_DIR/monitor64.prompts.jsonl"
echo "[done] guard:   $OUTPUT_DIR/guard64.prompts.jsonl"
echo "[done] summary: $OUTPUT_DIR/summary.json"
