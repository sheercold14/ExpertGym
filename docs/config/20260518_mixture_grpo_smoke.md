# 2026-05-18 Mixture / Full-Parameter GRPO Smoke

Purpose: validate the clean Mixture Training baseline path after WUDI /
ExpertMerging were excluded from the current rerun batch.

This is a smoke run only, not a reportable baseline score.

## Run

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_20260518
tmux: baseline_mixture_grpo_smoke_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_20260518.launch.log
```

## Command

```bash
GPU_LIST=6,7 \
NGPUS_PER_NODE=2 \
RUN_NAME=mixture_grpo_ta13_evaltarget_l8_n1_step1_20260518 \
LIMIT=8 \
SAMPLES_PER_PROMPT=1 \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=4 \
TOTAL_TRAINING_STEPS=1 \
TOTAL_EPOCHS=1 \
MAX_RESPONSE_LENGTH=512 \
MAX_MODEL_LEN=6144 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 \
ROLLOUT_MAX_NUM_SEQS=32 \
PPO_MAX_TOKEN_LEN_PER_GPU=4096 \
AGENT_LOOP_NUM_WORKERS=2 \
ROLLOUT_GPU_MEM_UTIL=0.35 \
ACTOR_LR=1e-6 \
KL_LOSS_COEF=0.001 \
SAVE_FREQ=1 \
TEST_FREQ=-1 \
bash scripts/baselines/run_qwen_mixture_grpo_baseline.sh
```

## Baseline Definition

- Initial model:
  `/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/task_arithmetic_c0p3333333333333333_k0p2_d0p8_seed20260518`
- Training: VeRL GRPO, full actor parameters trainable.
- Reward: `verl.experimental.opvec.reward_fn` -> OP-VEC `RewardRouter`.
- No OP-VEC gate external lib.
- No OPD loss.
- No retention loss.
- No WUDI / ExpertMerging rerun.

## Success Criteria

1. VeRL can load the TA-1/3 merged checkpoint.
2. vLLM rollout can produce samples for the OP-VEC prompt parquet.
3. RewardRouter routes Tool / Memory / Code rewards without error.
4. Full actor backward and optimizer step finish.
5. A checkpoint is saved under the run directory.

If this smoke passes, schedule a larger 24-48 prompt baseline run after the P1
gate experiments free enough GPUs.

## 2026-05-18 04:24 CST Result

Status: failed before rollout; do not report as a valid baseline run.

Root cause:

- GPU 6/7 looked free at launch time, but `eg72_opd_gc_c033_evaltarget_fast_20260518`
  immediately advanced into `iter_008` and reacquired GPU 6/7 for rollout.
- The smoke then competed with the P1 OPD-only run for the same cards.
- VeRL/FSDP loaded, but vLLM failed during KV cache initialization:

```text
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`
```

Verified before failure:

- TA-1/3 checkpoint loaded into VeRL/FSDP.
- Dataset conversion to parquet worked.
- Config overrides were applied: `max_response_length=512`,
  `tensor_parallel_size=1`, `max_num_batched_tokens=4096`,
  `max_num_seqs=32`, `temperature=0.7`, `top_p=0.95`.

Next action:

- Do not retry while any `train_eg72_*` tmux session can still reclaim those
  GPUs.
- Retry only after a two-card or eight-card block is truly exclusive, or after
  stopping the corresponding P1 run intentionally.

## 2026-05-18 04:49 CST One-GPU Smoke

Status: failed/ended before rollout; do not report as a valid baseline run.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_20260518
tmux: baseline_mixture_grpo_gpu4_smoke_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_20260518.launch.log
```

Command deltas:

```bash
GPU_LIST=4
LIMIT=8
SAMPLES_PER_PROMPT=1
TOTAL_TRAINING_STEPS=1
MAX_RESPONSE_LENGTH=512
MAX_MODEL_LEN=4608
ROLLOUT_GPU_MEM_UTIL=0.35
```

Observed:

- Dataset conversion and launch manifest succeeded.
- Actor rollout overrides were applied correctly:
  `tensor_model_parallel_size=1`, `gpu_memory_utilization=0.35`,
  `response_length=512`, `max_model_len=4608`.
- The visible `tensor_model_parallel_size=2` in the log belongs to disabled
  `reward.reward_model.rollout` default config, not the active actor rollout.
- VeRL loaded the 7.62B model and built FSDP objects on one GPU, but the process
  ended before rollout/checkpoint creation. No checkpoint exists under the run
  directory.

Interpretation:

- One-GPU full-parameter GRPO is not a reliable baseline route for this model
  and current VeRL config.
- Next retry should use an exclusive two-card or larger block and keep the run
  as smoke until rollout, reward, backward, and checkpoint save all complete.

## 2026-05-18 05:25 CST Two-GPU Smoke With Actor Param Offload

Status: failed during mixed-task rollout postprocess; not a valid checkpoint.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_paramoffload_20260518
tmux: baseline_mixture_grpo_gpu45_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_paramoffload_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_paramoffload_20260518.launch.log
```

Command deltas:

```bash
GPU_LIST=4,5
LIMIT=8
SAMPLES_PER_PROMPT=1
TOTAL_TRAINING_STEPS=1
MAX_RESPONSE_LENGTH=512
MAX_MODEL_LEN=6144
ROLLOUT_GPU_MEM_UTIL=0.70
ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096
ROLLOUT_MAX_NUM_SEQS=16
PPO_MAX_TOKEN_LEN_PER_GPU=4096
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

Verified:

- Two-card FSDP initialized.
- vLLM server initialized on GPU 4/5, which fixes the previous KV-cache
  initialization failure.

Failure:

```text
KeyError: 'acc'
```

Root cause:

- VeRL agent-loop postprocess assumed all rows in a mixed-task batch had the
  same `reward_extra_info` keys. In mixed Tool / Memory / Code batches, some
  rows can have task-specific keys such as `acc` while others do not.

Minimal compatibility patch applied:

```text
third_party/verl/verl/experimental/agent_loop/agent_loop.py
```

Patch behavior:

- Build the union of `reward_extra_info` keys across rows.
- Fill missing keys with `None`.
- Do not change scalar reward values, OP-VEC RewardRouter, native OP-VEC
  training, WUDI, or ExpertMerging.

Next retry should reuse the same two-card offload recipe with a new run name.

## 2026-05-18 05:31 CST Two-GPU Retry After Extra-Info Patch

Status: failed after rollout collation; not a valid checkpoint.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_extrainfofix_20260518
tmux: baseline_mixture_grpo_gpu45_fix_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_extrainfofix_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_extrainfofix_20260518.launch.log
```

Verified:

- Extra-info union patch fixed the previous `KeyError: acc`.
- vLLM rollout and RewardRouter progressed into trainer-side batch assembly.

Failure:

```text
AssertionError: 25 % 2 != 0
```

Root cause:

- `trainer.balance_batch=True` requires the final sequence list length to be
  divisible by DP size.
- Mixed single-turn tasks plus MemAgent trajectory expansion can produce an
  odd number of training sequences even when the prompt batch has an even size.

Next retry:

```bash
trainer.balance_batch=False
```

This disables only sequence-length repartitioning for the smoke. It does not
change RewardRouter semantics or the baseline objective.

## 2026-05-18 05:36 CST Two-GPU Retry Without Balance Batch

Status: failed after rollout at old-logprob dispatch; not a valid checkpoint.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_nobalance_20260518
tmux: baseline_mixture_grpo_gpu45_nobalance_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_nobalance_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu45_nobalance_20260518.launch.log
```

Command deltas from the previous retry:

```bash
trainer.balance_batch=False
```

Verified:

- vLLM rollout can start under the two-card offload recipe.
- Mixed Tool / Memory / Code `reward_extra_info` collation no longer fails.
- The failure occurs after rollout, when computing old logprobs for the actor.

Failure:

```text
AssertionError: expecting td with length divisible by chunks, but got 25 and 2
```

Root cause:

- MemAgent multi-turn expansion changes the number of training sequences after
  rollout. In this smoke, 8 prompt rows became 25 training rows.
- `trainer.balance_batch=False` avoids length-balancing repartition, but VeRL's
  worker-group dispatch still chunks the old-logprob batch by data-parallel
  size. With two workers, 25 rows cannot be split evenly.

Next action:

- Use `n_gpus_per_node=1` for the next smoke so the expanded trajectory count
  is always divisible by DP size.
- Keep actor parameter offload enabled; the earlier one-card attempt without
  actor param offload did not produce a checkpoint.
- If the one-card route is too slow or memory-bound, the next auditable option
  is a 5-card smoke because the observed expanded count 25 is divisible by 5.
  Padding/dropping trajectories inside VeRL would be a code-path change and
  should not be mixed into the baseline before a separate review.

## 2026-05-18 05:41 CST One-GPU DP Retry With Actor Param Offload

Status: failed during vLLM initialization; not a valid checkpoint.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu6_1dp_paramoffload_20260518
tmux: baseline_mixture_grpo_gpu6_1dp_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu6_1dp_paramoffload_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu6_1dp_paramoffload_20260518.launch.log
```

Command deltas:

```bash
GPU_LIST=6
actor_rollout_ref.actor.fsdp_config.param_offload=True
trainer.balance_batch=False
ROLLOUT_GPU_MEM_UTIL=0.70
MAX_MODEL_LEN=6144
```

Verified:

- One-card DP removes the previous expanded-trajectory divisibility issue.
- Actor/ref FSDP initialized with world size 1.

Failure:

```text
ValueError: Free memory on device (49.14/79.33 GiB) on startup is less than desired GPU memory utilization (0.7, 55.53 GiB).
```

Next retry:

- Keep one-card DP and actor param offload.
- Lower `ROLLOUT_GPU_MEM_UTIL` to 0.45.
- Lower `MAX_MODEL_LEN` to 4608 and keep the smoke small.

## 2026-05-18 05:44 CST One-GPU Low-Memory Retry

Status: failed at old-logprob microbatching; not a valid checkpoint.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu7_1dp_lowmem_20260518
tmux: baseline_mixture_grpo_gpu7_1dp_lowmem_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu7_1dp_lowmem_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu7_1dp_lowmem_20260518.launch.log
```

Command deltas:

```bash
GPU_LIST=7
actor_rollout_ref.actor.fsdp_config.param_offload=True
trainer.balance_batch=False
ROLLOUT_GPU_MEM_UTIL=0.45
MAX_MODEL_LEN=4608
ROLLOUT_MAX_NUM_BATCHED_TOKENS=2048
ROLLOUT_MAX_NUM_SEQS=8
PPO_MAX_TOKEN_LEN_PER_GPU=3072
```

Verified:

- vLLM started successfully under the lower memory setting.
- Rollout progressed into old-logprob computation.
- The previous vLLM memory threshold failure is fixed.

Failure:

```text
AssertionError: max_token_len must be greater than the sequence length. Got max_token_len=3072 and max_seq_len=tensor(4608)
```

Next retry:

- Keep the same one-card low-memory recipe.
- Increase `PPO_MAX_TOKEN_LEN_PER_GPU` and logprob max token length to 5120.

## 2026-05-18 05:49 CST One-GPU Low-Memory Retry With Token Limit 5120

Status: failed at old-logprob forward input preparation; not a valid checkpoint.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu6_1dp_lowmem_tok5120_20260518
tmux: baseline_mixture_grpo_gpu6_1dp_tok5120_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu6_1dp_lowmem_tok5120_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu6_1dp_lowmem_tok5120_20260518.launch.log
```

Command deltas:

```bash
PPO_MAX_TOKEN_LEN_PER_GPU=5120
```

Verified:

- vLLM startup works.
- The previous `max_token_len < max_seq_len` failure is fixed.

Failure:

```text
KeyError: key "temperature" not found in TensorDict
```

Root cause:

- When MemAgent expands trajectories, `ray_trainer.py` replaces the original
  trainer batch with `gen_batch_output`.
- That generated `DataProto` did not carry `meta_info["temperature"]`, while
  FSDP old-logprob forward expects a temperature field.

Minimal compatibility patch applied:

```text
third_party/verl/verl/experimental/agent_loop/agent_loop.py
```

Patch behavior:

- Agent-loop `_postprocess()` now includes
  `meta_info["temperature"] = self.rollout_config.temperature`.
- This does not change scalar reward, rollout text, OP-VEC gate training, OPD,
  retention, or baseline loss definition; it only preserves a VeRL metadata
  field required for old-logprob recomputation after trajectory expansion.

Next retry:

- Reuse the same one-card low-memory + token-limit-5120 command.

## 2026-05-18 05:53 CST One-GPU Retry With Temperature Metadata Patch

Status: failed at actor update mini-batch split; not a valid checkpoint.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu7_1dp_temperaturefix_20260518
tmux: baseline_mixture_grpo_gpu7_tempfix_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu7_1dp_temperaturefix_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu7_1dp_temperaturefix_20260518.launch.log
```

Verified:

- vLLM startup works.
- MemAgent expanded trajectory batch reaches old-logprob.
- Old-logprob succeeds after preserving `meta_info["temperature"]`.
- The run reaches actor update, which is the farthest successful stage so far.

Failure:

```text
AssertionError: 25 % 4 != 0
```

Root cause:

- The 8 prompt smoke expands to 25 training rows.
- Actor update requires the expanded TensorDict length to be divisible by
  `PPO_MINI_BATCH_SIZE`.

Next retry:

- Keep the same one-card low-memory command.
- Set `PPO_MINI_BATCH_SIZE=1` for smoke robustness under variable multi-turn
  expansion.

## 2026-05-18 05:56 CST One-GPU Retry With Mini-Batch Size 1

Status: succeeded; valid smoke checkpoint produced.

Run:

```text
run_name: mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_1dp_minib1_20260518
tmux: baseline_mixture_grpo_gpu4_minib1_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_1dp_minib1_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_1dp_minib1_20260518.launch.log
checkpoint: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_1dp_minib1_20260518/checkpoints/global_step_1/actor
rollout: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_1dp_minib1_20260518/rollouts/1.jsonl
```

Command deltas:

```bash
PPO_MINI_BATCH_SIZE=1
```

Observed metrics:

```text
train/expanded_agent_loop_rows: 25
critic/score/mean: 0.3640
actor/grad_norm: 19.3209
actor/lr: 1e-06
timing_s/gen: 27.46
timing_s/old_log_prob: 12.59
timing_s/ref: 53.15
timing_s/update_actor: 77.40
timing_s/save_checkpoint: 34.89
timing_s/step: 206.90
```

Interpretation:

- This proves the clean full-parameter GRPO baseline path is executable:
  TA-1/3 checkpoint -> VeRL vLLM rollout -> OP-VEC RewardRouter -> old logprob
  -> actor update -> checkpoint save.
- This is still a smoke checkpoint, not a reportable baseline score, because it
  used only 8 prompts and 1 training step.
- The production recipe should keep:
  - one-card DP or a DP size that divides expanded trajectory rows;
  - actor parameter offload;
  - low vLLM utilization 0.45;
  - `PPO_MINI_BATCH_SIZE=1` unless trajectory padding/chunking is audited.

Next action:

- Run the same recipe on the full 96-prompt calibration set for a reportable
  Mixture/full-parameter GRPO baseline candidate.
