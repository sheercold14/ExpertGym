#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
DOWNLOAD_SESSION="${DOWNLOAD_SESSION:-download_qwen25_math7b_20260519}"
LOG_DIR="$ROOT/runs/gated_grpo"
LOG="$LOG_DIR/orchestrate_20260519_r1math_L1_L2.log"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

gpu_pair_free() {
  local first="$1"
  local second="$2"
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$first,$second")
  [[ "${used[0]}" -lt 5000 && "${used[1]}" -lt 5000 ]]
}

wait_for_download() {
  log "waiting for Math-7B download session: $DOWNLOAD_SESSION"
  while tmux has-session -t "$DOWNLOAD_SESSION" 2>/dev/null; do
    sleep 60
  done
  log "download session ended"
  test -f /mnt/cache/wuruixiao/models/Qwen2.5-Math-7B/model-00001-of-00004.safetensors
  test -f /mnt/cache/wuruixiao/models/Qwen2.5-Math-7B/model-00004-of-00004.safetensors
  test -f /mnt/cache/wuruixiao/models/Qwen2.5-Math-7B/model.safetensors.index.json
}

build_modes() {
  if [[ -f "$ROOT/modes/opvec4_r1math_scaled_20260519/mode_manifest.json" ]]; then
    log "correct-R1 scaled mode manifest already exists"
    return
  fi
  log "building correct-R1 modes"
  bash skill/command/build_20260519_r1math_modes.sh 2>&1 | tee -a "$LOG"
}

start_l1() {
  if tmux has-session -t train_L1_r1math_20260519 2>/dev/null; then
    log "L1 tmux already running"
    return
  fi
  if [[ -d "$ROOT/runs/gated_grpo/expL1_r1math_layer28_hier_20it_20260519/iter_001" ]]; then
    log "L1 run directory already has iter_001; not relaunching"
    return
  fi
  log "starting L1 on GPU 0,1"
  tmux new-session -d -s train_L1_r1math_20260519 \
    "cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && PHASE=L1 GPU_LIST=0,1 NUM_ITERS=20 bash skill/command/run_20260519_r1math_L_experiments.sh 2>&1 | tee $ROOT/runs/gated_grpo/expL1_r1math_layer28_hier_20it_20260519.train.log"
}

start_l2_when_free() {
  if tmux has-session -t train_L2_r1math_20260519 2>/dev/null; then
    log "L2 tmux already running"
    return
  fi
  if [[ -d "$ROOT/runs/gated_grpo/expL2_r1math_layer28_hier_freezeR1_20it_20260519/iter_001" ]]; then
    log "L2 run directory already has iter_001; not relaunching"
    return
  fi
  log "waiting for old diagnostic experiments to release a pair for L2"
  while true; do
    if ! tmux has-session -t train_r1_layer28_hier_20260518 2>/dev/null && gpu_pair_free 2 3; then
      L2_GPUS=2,3
      break
    fi
    if ! tmux has-session -t train_r1_3band_continue10_20260518 2>/dev/null && gpu_pair_free 4 5; then
      L2_GPUS=4,5
      break
    fi
    log "L2 still waiting"
    sleep 300
  done
  log "starting L2 on GPU $L2_GPUS"
  tmux new-session -d -s train_L2_r1math_20260519 \
    "cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && PHASE=L2 GPU_LIST=$L2_GPUS NUM_ITERS=20 bash skill/command/run_20260519_r1math_L_experiments.sh 2>&1 | tee $ROOT/runs/gated_grpo/expL2_r1math_layer28_hier_freezeR1_20it_20260519.train.log"
}

main() {
  wait_for_download
  build_modes
  start_l1
  start_l2_when_free
}

main "$@"
