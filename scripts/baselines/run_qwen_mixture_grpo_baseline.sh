#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/baselines/run_qwen_mixture_grpo_baseline.sh

Purpose:
  Run a clean Mixture/full-parameter GRPO baseline with VeRL. This baseline
  starts from a static merged checkpoint and trains all actor parameters with
  executable rewards. It does not install OP-VEC gates and does not use OPD or
  retention losses.

Key overrides:
  PY=/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python
  GPU_LIST=0,1,2,3,4,5,6,7
  MODEL_PATH=/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/task_arithmetic_c0p3333333333333333_k0p2_d0p8_seed20260518
  SOURCE_CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl
  LIMIT=24
  SAMPLES_PER_PROMPT=2
  TOTAL_TRAINING_STEPS=1
  DRY_RUN=1
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/third_party/verl}"
PY="${PY:-/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python}"

ROOT="${ROOT:-/tmp/shared-storage/ExpertGym/baselines/mixture_grpo}"
ONPOLICY_ROOT="${ONPOLICY_ROOT:-/tmp/shared-storage/OnPolicy}"
MODEL_PATH="${MODEL_PATH:-/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/task_arithmetic_c0p3333333333333333_k0p2_d0p8_seed20260518}"
SOURCE_CALIBRATION="${SOURCE_CALIBRATION:-$ONPOLICY_ROOT/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-$("$PY" - "$GPU_LIST" <<'PY'
import sys
print(len([item for item in sys.argv[1].split(",") if item.strip()]))
PY
)}"

RUN_NAME="${RUN_NAME:-mixture_grpo_ta13_evaltarget_l${LIMIT:-24}_n${SAMPLES_PER_PROMPT:-2}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/$RUN_NAME}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data}"
CALIBRATION_JSONL="${CALIBRATION_JSONL:-$DATA_DIR/calibration.prompts.jsonl}"
CALIBRATION_PARQUET="${CALIBRATION_PARQUET:-$DATA_DIR/calibration.parquet}"

LIMIT="${LIMIT:-24}"
OFFSET="${OFFSET:-0}"
TASKS="${TASKS:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$LIMIT}"
if [[ -z "${PPO_MINI_BATCH_SIZE:-}" ]]; then
  if (( TRAIN_BATCH_SIZE < 8 )); then
    PPO_MINI_BATCH_SIZE="$TRAIN_BATCH_SIZE"
  else
    PPO_MINI_BATCH_SIZE=8
  fi
fi
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:--1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-$NGPUS_PER_NODE}"
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-$VERL_ROOT/examples/opvec_gated_grpo/opvec_agent_loops.yaml}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
DRY_RUN="${DRY_RUN:-0}"

HF_HOME="${HF_HOME:-$ONPOLICY_ROOT/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"

export OPVEC_REPO_ROOT="$REPO_ROOT"
export ONPOLICY_STORAGE_ROOT="$ONPOLICY_ROOT"
export PYTHONPATH="$VERL_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME
export HF_DATASETS_CACHE
export HF_MODULES_CACHE
unset OPVEC_ENABLE_VERL_PATCH
unset OPVEC_GATE_CHECKPOINT
unset OPVEC_INIT_GATE_CHECKPOINT

mkdir -p "$DATA_DIR" "$RUN_DIR" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

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

payload = {
    "format": "expertgym_mixture_grpo_launch_manifest_v1",
    "repo_root": "$REPO_ROOT",
    "verl_root": "$VERL_ROOT",
    "model_path": "$MODEL_PATH",
    "source_calibration": "$SOURCE_CALIBRATION",
    "calibration_jsonl": "$CALIBRATION_JSONL",
    "calibration_parquet": "$CALIBRATION_PARQUET",
    "run_name": "$RUN_NAME",
    "run_dir": "$RUN_DIR",
    "gpu_list": "$GPU_LIST",
    "ngpus_per_node": int("$NGPUS_PER_NODE"),
    "limit": int("$LIMIT"),
    "tasks": "$TASKS",
    "samples_per_prompt": int("$SAMPLES_PER_PROMPT"),
    "max_prompt_length": int("$MAX_PROMPT_LENGTH"),
    "max_response_length": int("$MAX_RESPONSE_LENGTH"),
    "actor_lr": float("$ACTOR_LR"),
    "kl_loss_coef": float("$KL_LOSS_COEF"),
    "total_training_steps": int("$TOTAL_TRAINING_STEPS"),
    "notes": [
        "full actor parameters are trainable",
        "no OP-VEC gate external_lib",
        "no OPD loss",
        "no retention loss",
        "reward uses verl.experimental.opvec.reward_fn -> OP-VEC RewardRouter",
    ],
    "env": {
        key: os.environ.get(key)
        for key in [
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "HF_MODULES_CACHE",
            "OPVEC_REPO_ROOT",
            "ONPOLICY_STORAGE_ROOT",
        ]
    },
}
path = Path(sys.argv[1])
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"launch_manifest": str(path)}, ensure_ascii=False, indent=2))
PY

CMD=(
  "$PY" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=True
  algorithm.use_kl_in_reward=False
  data.train_files="$CALIBRATION_PARQUET"
  data.val_files="$CALIBRATION_PARQUET"
  data.prompt_key=prompt
  data.reward_fn_key=data_source
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=False
  data.truncation=left
  data.shuffle=False
  data.dataloader_num_workers=0
  data.trust_remote_code=True
  actor_rollout_ref.model.path="$MODEL_PATH"
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPLEMENTATION"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.optim.weight_decay=0.0
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bf16
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL"
  actor_rollout_ref.rollout.n="$SAMPLES_PER_PROMPT"
  actor_rollout_ref.rollout.temperature="$TEMPERATURE"
  actor_rollout_ref.rollout.top_p="$TOP_P"
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH"
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH"
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_NUM_WORKERS"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.enable_prefix_caching=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  reward.custom_reward_function.path=pkg://verl.experimental.opvec.reward_fn
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name=naive
  reward.num_workers=4
  trainer.balance_batch=True
  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name=expertgym_baselines
  trainer.experiment_name="$RUN_NAME"
  trainer.n_gpus_per_node="$NGPUS_PER_NODE"
  trainer.nnodes=1
  trainer.val_before_train=False
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
  trainer.default_local_dir="$RUN_DIR/checkpoints"
  trainer.rollout_data_dir="$RUN_DIR/rollouts"
  +ray_kwargs.ray_init.num_gpus="$NGPUS_PER_NODE"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$VERL_ROOT"
"${CMD[@]}" "$@"
