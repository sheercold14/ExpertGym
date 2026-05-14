# OP-VEC All-Success OPD Experiments 2026-05-13

## Goal

在现有 Gated-GRPO frontier loss 上，额外加入 all-success rows 的 OPD best-response loss，观察它是否能给 task-vector scalar 提供更强持续推动，并判断下一步优化方向。

## Code Path

- Main updater: `scripts/train/opvec_update_gates_from_rollouts.py`
- Shell entry: `skill/command/run_qbank_c033333_gate_strategy_opd.sh`
- Mechanism: normal frontier GRPO loss remains enabled; rows skipped as `all_success` are additionally optimized with sequence best-response loss `-log p(success_response)` when `USE_OPD_ALL_SUCCESS=1`.
- Defaults are unchanged unless OPD env vars are explicitly set.

## Running Experiments

### A. Mild OPD

- Run name: `qbank_c033333_global_opd_as_mild_i6_20260513`
- GPUs: `4,5`
- Strategy: `global`
- Init coefficient: `0.3333333333333333`
- Iterations: `6`
- Prompts/samples: `100 x 4`
- Objective: `PPO_LOSS_WEIGHT=2.0`, `OPD_ALL_SUCCESS_LOSS_WEIGHT=0.05`
- Optimizer: `LR=0.012`, `PRIOR_LOSS_WEIGHT=0.003`, `MAX_COEFF_DELTA=0.5`
- Update: `UPDATE_EPOCHS=2`, `UPDATE_BATCH_SIZE=2`, `BATCH_LOSS_REDUCTION=mean`
- Advantage: `reward_train`, `ADVANTAGE_NORMALIZATION=centered`, `TASK_NORMALIZE_ADVANTAGES=0`, `USE_FRONTIER_WEIGHT=0`

### B. Strong OPD

- Run name: `qbank_c033333_global_opd_as_strong_i6_20260513`
- GPUs: `6,7`
- Strategy: `global`
- Init coefficient: `0.3333333333333333`
- Iterations: `6`
- Prompts/samples: `100 x 4`
- Objective: `PPO_LOSS_WEIGHT=3.0`, `OPD_ALL_SUCCESS_LOSS_WEIGHT=0.12`
- Optimizer: `LR=0.024`, `PRIOR_LOSS_WEIGHT=0.002`, `MAX_COEFF_DELTA=0.6`
- Update: `UPDATE_EPOCHS=2`, `UPDATE_BATCH_SIZE=2`, `BATCH_LOSS_REDUCTION=mean`
- Advantage: `reward_train`, `ADVANTAGE_NORMALIZATION=centered`, `TASK_NORMALIZE_ADVANTAGES=0`, `USE_FRONTIER_WEIGHT=0`

## Monitoring Checklist

- Per iteration rollout reward and success rate by task.
- `frontier_task_counts` and `opd_all_success_task_counts`.
- `epoch_summaries`: `grad_norm_max`, `gate_delta_max`, optimizer steps.
- Final/iter gate movement from init, especially effective `tool/memory/code` scalar means.
- Failure mode: zero gate grad, no all-success OPD rows, reward collapse, or scalar hitting projection bound too early.

## Observations

### Iteration 1 Rollout Baseline

Both experiments launched successfully on GPU `4-7`; no overlap with the existing GPU `0-3` retention experiments.

Mild OPD, iter 001 rollout:

- Frontier rows from shard summaries: `34 + 36 = 70`.
- Task reward/success from merged rollout:
  - Tool: raw mean `-0.0313`, success `0.4632`, exact `0.3309`, kept frontier rows `23`.
  - Memory: mean `0.3182`, success `0.3182`, kept frontier rows `21`.
  - Code: mean `0.4413`, success `0.5303`, kept frontier rows `26`.

Strong OPD, iter 001 rollout:

- Frontier rows from shard summaries: `38 + 34 = 72`.
- Task reward/success from merged rollout:
  - Tool: raw mean `0.2097`, success `0.4853`, exact `0.3529`, kept frontier rows `26`.
  - Memory: mean `0.3258`, success `0.3258`, kept frontier rows `23`.
  - Code: mean `0.3674`, success `0.4621`, kept frontier rows `23`.

Both runs entered iter 001 gate update. The token-level update used HF logprob plus `UPDATE_EPOCHS=2`; after more than 25 minutes no `gate_updates.summary.json` had been written. I terminated both runs to avoid spending the full night on one update point.

Decision: switch the next runs to sequence-level loss and `UPDATE_EPOCHS=1`, keeping all-success OPD enabled. This sacrifices token-level PPO fidelity for faster evidence on whether OPD can move the global scalar and improve reward.

## Fast Follow-Up Runs

### C. Sequence Mild OPD

- Run name: `qbank_c033333_global_opd_as_seq_mild_i4_20260513`
- GPUs: `4,5`
- Iterations: `4`
- Prompts/samples: `60 x 4`
- Objective: `LOSS_GRANULARITY=sequence`, `PPO_LOSS_WEIGHT=2.0`, `OPD_ALL_SUCCESS_LOSS_WEIGHT=0.05`
- Optimizer: `LR=0.020`, `PRIOR_LOSS_WEIGHT=0.002`, `MAX_COEFF_DELTA=0.5`
- Update: `UPDATE_EPOCHS=1`, `UPDATE_BATCH_SIZE=2`

### D. Sequence Strong OPD

- Run name: `qbank_c033333_global_opd_as_seq_strong_i4_20260513`
- GPUs: `6,7`
- Iterations: `4`
- Prompts/samples: `60 x 4`
- Objective: `LOSS_GRANULARITY=sequence`, `PPO_LOSS_WEIGHT=3.0`, `OPD_ALL_SUCCESS_LOSS_WEIGHT=0.15`
- Optimizer: `LR=0.035`, `PRIOR_LOSS_WEIGHT=0.0015`, `MAX_COEFF_DELTA=0.6`
- Update: `UPDATE_EPOCHS=1`, `UPDATE_BATCH_SIZE=2`
