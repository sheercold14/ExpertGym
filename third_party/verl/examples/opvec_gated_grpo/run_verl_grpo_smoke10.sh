#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash third_party/verl/examples/opvec_gated_grpo/run_verl_grpo_smoke10.sh

Purpose:
  Run the OP-VEC gated-GRPO path through verl.trainer.main_ppo.

Key overrides:
  PY=/mnt/cache/wuruixiao/miniconda3/envs/easyrl/bin/python
  GPU_LIST=0,1
  LIMIT=10
  TASKS=tool,code
  STRATEGY=global
  INIT_VALUE=0.3333333333333333
  SAMPLES_PER_PROMPT=2
  MAX_RESPONSE_LENGTH=1024
  ACTOR_LR=0.01
EOF
  exit 0
fi

VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPVEC_REPO_ROOT="${OPVEC_REPO_ROOT:-$(cd "$VERL_ROOT/../.." && pwd)}"
PY="${PY:-/mnt/cache/wuruixiao/miniconda3/envs/easyrl/bin/python}"

ROOT="${ROOT:-/tmp/shared-storage/OnPolicy}"
CONFIG="${CONFIG:-$OPVEC_REPO_ROOT/configs/gated_grpo.yaml}"
MODE="${MODE:-$ROOT/modes/opvec4/mode_manifest.json}"
MODEL_PATH="${MODEL_PATH:-/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct}"
SOURCE_CALIBRATION="${SOURCE_CALIBRATION:-$ROOT/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.prompts.jsonl}"
HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"

STRATEGY="${STRATEGY:-global}"
INIT_VALUE="${INIT_VALUE:-0.3333333333333333}"
LIMIT="${LIMIT:-10}"
OFFSET="${OFFSET:-0}"
TASKS="${TASKS:-}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
MAX_GATED_MODULES="${MAX_GATED_MODULES:-0}"
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-$VERL_ROOT/examples/opvec_gated_grpo/opvec_agent_loops.yaml}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

GPU_LIST="${GPU_LIST:-0,1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-$("$PY" - "$GPU_LIST" <<'PY'
import sys
print(len([item for item in sys.argv[1].split(",") if item.strip()]))
PY
)}"

RUN_NAME="${RUN_NAME:-verl_grpo_smoke${LIMIT}_c033333_${STRATEGY}_i1_seed20260511}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/verl_opvec/$RUN_NAME}"
DATA_DIR="${DATA_DIR:-$ROOT/data/verl_opvec/$RUN_NAME}"
CALIBRATION_JSONL="${CALIBRATION_JSONL:-$DATA_DIR/calib${LIMIT}_seed20260511.prompts.jsonl}"
CALIBRATION_PARQUET="${CALIBRATION_PARQUET:-$DATA_DIR/calib${LIMIT}_seed20260511.parquet}"
INIT_GATE="${INIT_GATE:-$DATA_DIR/init_${STRATEGY}_c033333.gates.json}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$LIMIT}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$LIMIT}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
ACTOR_LR="${ACTOR_LR:-0.01}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.50}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-$NGPUS_PER_NODE}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

export OPVEC_REPO_ROOT
export ONPOLICY_STORAGE_ROOT="${ONPOLICY_STORAGE_ROOT:-$ROOT}"
export PYTHONPATH="$VERL_ROOT:$OPVEC_REPO_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME
export HF_DATASETS_CACHE
export HF_MODULES_CACHE

export OPVEC_ENABLE_VERL_PATCH=1
export OPVEC_CONFIG="$CONFIG"
export OPVEC_MODE_MANIFEST="$MODE"
export OPVEC_GATE_PARAMETERIZATION="$STRATEGY"
export OPVEC_GATE_CHECKPOINT="$INIT_GATE"
export OPVEC_INIT_GATE_CHECKPOINT="$INIT_GATE"
export OPVEC_MAX_GATED_MODULES="$MAX_GATED_MODULES"
export OPVEC_FREEZE_BASE="${OPVEC_FREEZE_BASE:-1}"
export OPVEC_METRICS_JSONL="${OPVEC_METRICS_JSONL:-$RUN_DIR/metrics.jsonl}"
export OPVEC_MEMAGENT_UPDATE_MAX_TOKENS="${OPVEC_MEMAGENT_UPDATE_MAX_TOKENS:-${MEMORY_UPDATE_MAX_NEW_TOKENS:-1024}}"
export OPVEC_MEMAGENT_FINAL_MAX_TOKENS="${OPVEC_MEMAGENT_FINAL_MAX_TOKENS:-${MEMORY_FINAL_MAX_NEW_TOKENS:-1024}}"

mkdir -p "$DATA_DIR" "$RUN_DIR" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

if [[ ! -s "$MODE" ]]; then
  "$PY" "$OPVEC_REPO_ROOT/scripts/modes/build_opvec4_modes.py" \
    --config "$CONFIG"
fi

"$PY" "$OPVEC_REPO_ROOT/scripts/modes/build_constant_gate_checkpoint.py" \
  --config "$CONFIG" \
  --mode-manifest "$MODE" \
  --gate-parameterization "$STRATEGY" \
  --value "$INIT_VALUE" \
  --output "$INIT_GATE" >/dev/null

PREPARE_ARGS=(
  --input "$SOURCE_CALIBRATION"
  --output "$CALIBRATION_JSONL"
  --parquet "$CALIBRATION_PARQUET"
  --limit "$LIMIT"
  --offset "$OFFSET"
)
if [[ -n "$TASKS" ]]; then
  PREPARE_ARGS+=(--tasks "$TASKS")
fi
"$PY" "$VERL_ROOT/verl/experimental/opvec/prepare_data.py" "${PREPARE_ARGS[@]}"

"$PY" - "$RUN_DIR/launch_manifest.json" <<PY
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "format": "opvec_verl_grpo_launch_manifest_v1",
    "verl_root": "$VERL_ROOT",
    "opvec_repo_root": "$OPVEC_REPO_ROOT",
    "config": "$CONFIG",
    "mode_manifest": "$MODE",
    "model_path": "$MODEL_PATH",
    "source_calibration": "$SOURCE_CALIBRATION",
    "calibration_jsonl": "$CALIBRATION_JSONL",
    "calibration_parquet": "$CALIBRATION_PARQUET",
    "init_gate": "$INIT_GATE",
    "run_dir": "$RUN_DIR",
    "run_name": "$RUN_NAME",
    "gpu_list": "$GPU_LIST",
    "ngpus_per_node": int("$NGPUS_PER_NODE"),
    "strategy": "$STRATEGY",
    "init_value": float("$INIT_VALUE"),
    "limit": int("$LIMIT"),
    "tasks": "$TASKS",
    "samples_per_prompt": int("$SAMPLES_PER_PROMPT"),
    "agent_loop_num_workers": int("$AGENT_LOOP_NUM_WORKERS"),
    "max_prompt_length": int("$MAX_PROMPT_LENGTH"),
    "max_response_length": int("$MAX_RESPONSE_LENGTH"),
    "max_model_len": int("$MAX_MODEL_LEN"),
    "max_gated_modules": int("$MAX_GATED_MODULES"),
    "agent_loop_config_path": "$AGENT_LOOP_CONFIG_PATH",
    "attn_implementation": "$ATTN_IMPLEMENTATION",
    "opvec_memagent_update_max_tokens": int(os.environ["OPVEC_MEMAGENT_UPDATE_MAX_TOKENS"]),
    "opvec_memagent_final_max_tokens": int(os.environ["OPVEC_MEMAGENT_FINAL_MAX_TOKENS"]),
    "actor_lr": float("$ACTOR_LR"),
    "env": {
        key: os.environ.get(key)
        for key in [
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "HF_MODULES_CACHE",
            "OPVEC_ENABLE_VERL_PATCH",
            "OPVEC_CONFIG",
            "OPVEC_MODE_MANIFEST",
            "OPVEC_GATE_PARAMETERIZATION",
            "OPVEC_GATE_CHECKPOINT",
            "OPVEC_MAX_GATED_MODULES",
            "OPVEC_METRICS_JSONL",
            "OPVEC_MEMAGENT_UPDATE_MAX_TOKENS",
            "OPVEC_MEMAGENT_FINAL_MAX_TOKENS",
        ]
    },
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"launch_manifest": str(path)}, ensure_ascii=False, indent=2))
PY

cd "$VERL_ROOT"

"$PY" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  data.train_files="$CALIBRATION_PARQUET" \
  data.val_files="$CALIBRATION_PARQUET" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=False \
  data.truncation=left \
  data.shuffle=False \
  data.dataloader_num_workers=0 \
  data.trust_remote_code=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPLEMENTATION" \
  actor_rollout_ref.model.external_lib=verl.experimental.opvec.external_lib \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bf16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n="$SAMPLES_PER_PROMPT" \
  actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$TOP_P" \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_NUM_WORKERS" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  reward.custom_reward_function.path=pkg://verl.experimental.opvec.reward_fn \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  reward.num_workers=4 \
  trainer.balance_batch=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=onpolicy_merge \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  +ray_kwargs.ray_init.num_gpus="$NGPUS_PER_NODE" \
  "$@"
