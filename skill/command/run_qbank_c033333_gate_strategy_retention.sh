#!/usr/bin/env bash
set -euo pipefail

# Retention-control launcher for comparing against run_qbank_c033333_gate_strategy.sh.
# It keeps the base launcher defaults unchanged and only enables KL retention for
# all-success rows unless the caller overrides the variables below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STRATEGY_FOR_NAME="${STRATEGY:-global}"
SAFE_STRATEGY="${STRATEGY_FOR_NAME//[^A-Za-z0-9_]/_}"

export USE_RETENTION="${USE_RETENTION:-1}"
export RETENTION_LOSS_WEIGHT="${RETENTION_LOSS_WEIGHT:-0.05}"
export MAX_RETENTION_ROWS="${MAX_RETENTION_ROWS:-64}"
export RUN_NAME="${RUN_NAME:-qbank_c033333_${SAFE_STRATEGY}_retention_i2_seed20260511}"

exec bash "$SCRIPT_DIR/run_qbank_c033333_gate_strategy.sh" "$@"
