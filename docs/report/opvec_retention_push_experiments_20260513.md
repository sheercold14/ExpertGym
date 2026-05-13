# OP-VEC Retention Push Experiments 2026-05-13

## Objective

用 GPU 0-3 跑两组 2-GPU 对照实验，验证在从 `1/3` task-vector 初始点重新训练时，增大 LR / PPO loss 后能否更快推动 global task-vector 系数，同时通过 KL retention 保持已经全对的 Tool 行为。

## Runs

| run_id | GPU | path | key difference |
|---|---:|---|---|
| `retention_push_centered_i8` | `0,1` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_retention_push_centered_i8_20260513_175332` | `ADVANTAGE_NORMALIZATION=centered` |
| `retention_push_zscore_i8` | `2,3` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_retention_push_zscore_i8_20260513_175332` | `ADVANTAGE_NORMALIZATION=zscore` |

## Shared Parameters

```bash
STRATEGY=global
INIT_VALUE=0.3333333333333333
NUM_ITERS=8
NUM_PROMPTS=100
SAMPLES_PER_PROMPT=4
LR=0.006
PPO_LOSS_WEIGHT=2.0
PRIOR_LOSS_WEIGHT=0.003
MAX_COEFF_DELTA=0.5
UPDATE_EPOCHS=2
UPDATE_BATCH_SIZE=2
BATCH_LOSS_REDUCTION=mean
LOSS_GRANULARITY=token
TASK_NORMALIZE_ADVANTAGES=0
USE_FRONTIER_WEIGHT=0
STORE_TOKEN_LOGPROBS=0
LENGTH_NORMALIZE_POLICY_LOGPROB=1
FRONTIER_ORDER=task-interleaved
TASK_WEIGHT_TOOL=1.0
TASK_WEIGHT_MEMORY=1.0
TASK_WEIGHT_CODE=1.0
USE_RETENTION=1
RETENTION_LOSS_WEIGHT=0.05
MAX_RETENTION_ROWS=64
```

## Monitoring Criteria

- `effective coefficient`: `common + *_residual`，观察 Tool / Memory / Code 是否从 `1/3` 明显移动。
- `reward`: 观察 Tool 是否保持，Memory / Code 是否有持续推进。
- `frontier_task_counts`: 判断有效信号是否足够，特别是 Tool all-success 后是否进入 retention。
- `retention_rows` / `retention_loss`: 判断保护项是否真的生效。
- `clip_frac` / `approx_kl`: 判断更新是否过猛。
- `grad_norm`: 判断梯度链路是否有效。

## Live Notes

- 17:53:32 started both runs via tmux:
  - `opvec_push_centered_20260513_175332`
  - `opvec_push_zscore_20260513_175332`
- Initial `nvidia-smi` showed GPU 0-3 free before launch.

## Epoch Metrics

Pending. Metrics will be appended after each completed iteration summary is available.
