# OP-VEC Gate Optimizer Dynamics - 2026-05-14

## Objective

Test two optimizer variants for differentiable Gate-GRPO in task-vector coefficient space:

- `epoch-scope + SGD/momentum`, to preserve gradient magnitude across coefficients.
- `epoch-scope + persistent AdamW`, to avoid fresh-Adam sign-step behavior across outer rollout/update iterations.

Success criteria:

- Per-iteration wall time under 25 minutes.
- Overall `reward_train` trends upward over 10-15 iterations.
- Direct task-vector coefficients do not collapse; ideally they increase with expert-specific differentiation.
- Stop and revise if reward fails to rise or Code/Memory collapse while Tool saturates.

## Shared Setup

```bash
STRATEGY=global-coefficient
INIT=tool/memory/code=0.333333
NUM_PROMPTS=48
SAMPLES_PER_PROMPT=4
NUM_ITERS=15
UPDATE_EPOCHS=1
UPDATE_BATCH_SIZE=4
BATCH_LOSS_REDUCTION=mean
OPTIMIZER_STEP_SCOPE=epoch
LOSS_GRANULARITY=sequence
TASK_NORMALIZE_ADVANTAGES=0
ADVANTAGE_NORMALIZATION=zscore
USE_FRONTIER_WEIGHT=0
STORE_TOKEN_LOGPROBS=0
LENGTH_NORMALIZE_POLICY_LOGPROB=0
USE_RETENTION=0
PPO_LOSS_WEIGHT=6.0
PRIOR_LOSS_WEIGHT=0.0
MAX_COEFF_DELTA=1.0
```

The first 48 calibration prompts are balanced: `tool=16`, `memory=16`, `code=16`.

## Runs

| run | tmux | GPUs | optimizer | LR | state |
|---|---|---:|---|---:|---|
| `qbank_c033333_global_coeff_sgd_m08_epoch_n48_i15_20260514_005450` | `opvec_sgd_m08_20260514_005450` | `0,1,2,3` | SGD momentum `0.8` | `0.05` | persisted |
| `qbank_c033333_global_coeff_adam_persist_epoch_n48_i15_20260514_005450` | `opvec_adam_persist_20260514_005450` | `4,5,6,7` | AdamW | `0.025` | persisted |

## Monitoring Notes

Pending first iteration results.
