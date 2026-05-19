#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
BANK="${BANK:-$ROOT/data/calibration/code_p0_v3_20260518}"

export GPU_LIST="${GPU_LIST:-2,3}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-$GPU_LIST}"
export CONFIG="${CONFIG:-configs/gated_grpo_reasoning_layer28.yaml}"
export MODE="${MODE:-$ROOT/modes/opvec4_reasoning_20260516/mode_manifest.json}"
export CALIBRATION="${CALIBRATION:-$BANK/train_code64.prompts.jsonl}"
export RUN_NAME="${RUN_NAME:-r1_codep0_layer28_z001_codeonly_sanity_20260518}"
export RUN_DIR="${RUN_DIR:-$ROOT/runs/gated_grpo/$RUN_NAME}"
export STRATEGY="${STRATEGY:-layer-band}"
export INIT_GATE_CHECKPOINT="${INIT_GATE_CHECKPOINT:-$ROOT/data/init_gates/r1_scaled_20260518/all1_r1_z001.layer-band.json}"

export TASKS="${TASKS:-code}"
export NUM_ITERS="${NUM_ITERS:-4}"
export NUM_PROMPTS="${NUM_PROMPTS:-64}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
export CODE_MAX_NEW_TOKENS="${CODE_MAX_NEW_TOKENS:-10000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
export MAX_LOGPROB_TOKENS="${MAX_LOGPROB_TOKENS:-24576}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
export ROLLOUT_SHARDS="${ROLLOUT_SHARDS:-auto}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"

export PPO_LOSS_WEIGHT="${PPO_LOSS_WEIGHT:-1.0}"
export OPD_LOSS_WEIGHT="${OPD_LOSS_WEIGHT:-1.0}"
export USE_RETENTION="${USE_RETENTION:-1}"
export RETENTION_OBJECTIVE="${RETENTION_OBJECTIVE:-nll}"
export RETENTION_LOSS_WEIGHT="${RETENTION_LOSS_WEIGHT:-0.05}"
export ADVANTAGE_NORMALIZATION="${ADVANTAGE_NORMALIZATION:-zscore}"
export STORE_TOKEN_LOGPROBS="${STORE_TOKEN_LOGPROBS:-0}"
export OPTIMIZER="${OPTIMIZER:-sgd}"
export SGD_MOMENTUM="${SGD_MOMENTUM:-0.2}"
export OPTIMIZER_STEP_SCOPE="${OPTIMIZER_STEP_SCOPE:-epoch}"
export LOSS_GRANULARITY="${LOSS_GRANULARITY:-sequence}"
export LR="${LR:-0.05}"
export PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-0.0}"
export MAX_COEFF_DELTA="${MAX_COEFF_DELTA:-0.15}"
export MAX_COEFF_DELTA_BY_EXPERT="${MAX_COEFF_DELTA_BY_EXPERT:-reasoning=0.002}"

export DYNAMIC_OPD_EXPERT_ROLLOUT="${DYNAMIC_OPD_EXPERT_ROLLOUT:-$BANK/expert_rollouts/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl,$BANK/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_code_p0_v3_train64_s8_seed20260518.merged.jsonl}"
export DYNAMIC_OPD_TASKS="${DYNAMIC_OPD_TASKS:-code}"
export DYNAMIC_OPD_PER_TASK="${DYNAMIC_OPD_PER_TASK:-64}"
export DYNAMIC_OPD_MAX_POSITIVES_PER_ROW="${DYNAMIC_OPD_MAX_POSITIVES_PER_ROW:-1}"
export DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW="${DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW:-2}"

export TASK_WEIGHT_CODE="${TASK_WEIGHT_CODE:-1.0}"
export TASK_WEIGHT_TOOL="${TASK_WEIGHT_TOOL:-1.0}"
export TASK_WEIGHT_MEMORY="${TASK_WEIGHT_MEMORY:-1.0}"

bash skill/command/run_qbank_c033333_gate_strategy.sh
