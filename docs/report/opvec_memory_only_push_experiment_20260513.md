# OP-VEC Memory-Only Push Experiment 2026-05-13

## Goal

验证从 `1/3` task-vector 初始点出发时，如果只使用 calib100 中的 Memory prompt，并且只训练 `global.memory` 系数，当前 reward/GRPO 梯度是否足以把 Memory scalar 推到更高区间，以及 reward 是否随之上涨。

核心问题不是做最终训练，而是做 reachability diagnosis：

- 如果 memory-only 都推不动，说明 reward/梯度链路本身有问题。
- 如果 memory-only 能推很动，但 mixed-task 训练推不动，问题更可能来自混合任务采样、loss 权重、frontier 组成、参数耦合或更新预算。

## Code / Script Change

新增独立脚本：

```bash
skill/command/run_qbank_c033333_memory_only_push.sh
```

这个脚本只位于 `skill/command`，没有修改主训练代码。它和主 qbank wrapper 的关键区别：

- init gate 写入本次 `RUN_DIR/init_global_c033333.json`，不覆盖共享的 `data/question_bank/.../init_gates/init_global_c033333.json`。
- 使用 `--tasks memory`，只采样 calib100 中的 Memory prompt。
- 使用 `--train-coefficient global.memory`，只允许 Memory effective coefficient 更新；Tool/Code 被 projection 固定在初始 `1/3`。
- 默认 `NUM_PROMPTS=33`，因为 calib100 中实际 Memory prompt 数是 33，而不是 100。
- vLLM 默认使用 GPU `4,5,6,7`，`GPU_MEMORY_UTILIZATION=0.75`，`ROLLOUT_SHARD_STAGGER_SECONDS=8`，避免多 shard 同时初始化时的显存 profiling 波动。
- 默认 `MAX_LOGPROB_TOKENS=1024`，避免 Memory 长输出下 HF update 过慢。

## Final Effective Run

Run directory:

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_memory_only_push_n33_lr008_gpu075_stagger_i4_seed20260513
```

Important files:

```text
iter_001/rollouts.jsonl
iter_001/gate_updates_1024.summary.json
iter_001/gate_updates_1024.gates.json
eval_after_iter001_1024/rollouts.jsonl
eval_after_iter001_1024/rollouts.summary.json
```

Main parameters:

```bash
STRATEGY=global
INIT_VALUE=0.3333333333333333
TASKS=memory
NUM_PROMPTS=33
SAMPLES_PER_PROMPT=4
LR=0.08
PRIOR_LOSS_WEIGHT=0.0
MAX_COEFF_DELTA=0.6
UPDATE_EPOCHS=1
UPDATE_BATCH_SIZE=4
BATCH_LOSS_REDUCTION=mean
LOSS_GRANULARITY=sequence
PPO_LOSS_WEIGHT=1.0
ADVANTAGE_NORMALIZATION=centered
TASK_NORMALIZE_ADVANTAGES=0
USE_FRONTIER_WEIGHT=0
TRAIN_COEFFICIENTS=global.memory
LENGTH_NORMALIZE_POLICY_LOGPROB=1
MAX_LOGPROB_TOKENS=1024
```

## Setup Adjustments During Run

Several operational adjustments were made before the final evidence point:

1. Initial `NUM_PROMPTS=100` was wrong for this filtered experiment. After `--tasks memory`, calib100 only has 33 Memory rows, so sharding became `25/8/0/0`. This was stopped and changed to `NUM_PROMPTS=33`, giving `9/8/8/8`.
2. `GPU_MEMORY_UTILIZATION=0.82` caused vLLM startup failure on one shard because transient free memory was below the requested threshold. It was reduced to `0.75`.
3. Even at `0.75`, simultaneous vLLM startup could trigger memory profiling assertions when another shard released memory during initialization. `ROLLOUT_SHARD_STAGGER_SECONDS=8` fixed this.
4. A normal `MAX_LOGPROB_TOKENS=4096` update was too slow for rapid diagnosis. The complete rollout was kept, and the same `iter_001/rollouts.jsonl` was updated again with `MAX_LOGPROB_TOKENS=1024`.

These changes are operational, not reward/algorithm changes.

## Iteration 1 Rollout at 1/3

Initial rollout gate:

```json
{
  "common": 0.3333333333333333,
  "tool_residual": 0.0,
  "memory_residual": 0.0,
  "code_residual": 0.0
}
```

Memory rollout statistics:

| metric | value |
|---|---:|
| rows | 33 |
| samples | 132 |
| mean reward_train | 0.3561 |
| success rate | 0.3561 |
| success samples | 47 / 132 |
| zero-reward samples | 85 / 132 |
| frontier rows | 20 |
| non-frontier rows | 13 |

Shard frontier rows:

| shard | rows | frontier rows |
|---:|---:|---:|
| 0 | 9 | 5 |
| 1 | 8 | 3 |
| 2 | 8 | 5 |
| 3 | 8 | 7 |

Interpretation: Memory-only on-policy rollout has enough mixed-success rows. The GRPO signal is not sparse in this isolated setting: `20/33` prompts enter frontier.

## Gate Update Result

The successful update used the same rollout but `MAX_LOGPROB_TOKENS=1024`.

Summary:

| metric | value |
|---|---:|
| kept_frontier_rows | 20 |
| filled_missing_old_logprobs | 80 |
| optimizer_steps | 5 |
| skipped_optimizer_steps | 0 |
| grad_norm_max | 0.01943 |
| gate_delta_max | 0.26815 |
| clip_frac_mean | 0.8 |

Final gates:

```json
{
  "common": 0.46740755438804626,
  "tool_residual": -0.13407419621944427,
  "memory_residual": 0.26814839243888855,
  "code_residual": -0.13407419621944427
}
```

Effective coefficients:

| task | before | after | delta |
|---|---:|---:|---:|
| tool | 0.3333 | 0.3333 | +0.0000 |
| memory | 0.3333 | 0.7356 | +0.4022 |
| code | 0.3333 | 0.3333 | +0.0000 |

Because only `global.memory` is trainable, Tool/Code remain anchored at `1/3`. The global gate representation still changes `common/residual`, but the effective Tool/Code coefficients are preserved by projection.

## Post-Update Evaluation

After applying the updated gate, I baked the policy and reran Memory-only rollout on the same 33 Memory prompts with a new sampling seed.

Post-update gate:

```text
memory effective coefficient = 0.7356
tool/code effective coefficient = 0.3333
```

Reward comparison:

| condition | memory coeff | rows | samples | mean reward_train | success | success samples | zero samples | frontier rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before update | 0.3333 | 33 | 132 | 0.3561 | 0.3561 | 47 | 85 | 20 |
| after update | 0.7356 | 33 | 132 | 0.8561 | 0.8561 | 113 | 19 | 5 |

The post-update rollout used seed `20260514`; the pre-update rollout used seed `20260513`, so this is not a paired deterministic comparison. Still, the change is large enough to support the reachability conclusion.

## Conclusion

Memory-only reward and GRPO gradient are sufficient to move the scalar strongly:

- One update moved Memory effective coefficient from `0.3333` to `0.7356`.
- Memory reward increased from `0.3561` to `0.8561` in the post-update rollout.
- Frontier rows dropped from `20` to `5` after improvement, which is expected because many prompts became all-success and no longer contribute GRPO advantage.

This strongly suggests the current blocker is not “Memory reward has no gradient signal” or “global scalar cannot move”. The blocker is more likely in the mixed-task training dynamics:

- mixed task batches dilute or oppose Memory-only gradient;
- Tool/Code/Memory frontier composition changes the effective update direction;
- global `common + residual` coupling forces task competition unless coefficients are constrained or task-wise phases are used;
- long Memory/Code logprob recomputation makes full updates slow, reducing iteration velocity;
- all-success/all-failure rows disappear from GRPO, so once a task improves, its continued signal becomes sparse unless retention/OPD/replay is designed carefully.

## Recommended Next Step

Use this as evidence for a two-stage schedule:

1. short per-task reachability phase, especially Memory-only and Code-only, to verify each task can independently push its scalar into the useful interval;
2. mixed-task phase with task-balanced frontier quotas and smaller update cost (`MAX_LOGPROB_TOKENS=1024` or a more efficient sequence-level proxy), while monitoring whether mixed updates undo task-specific gains.

Do not interpret the Memory-only scalar `0.7356` as a hand-crafted checkpoint prior. It is an outcome of reward-driven on-policy optimization in the isolated task setting.

## Direct Three-Coefficient Follow-Up

After the first run, we added a separate `global-coefficient` gate parameterization to avoid the `common + residual` internal representation. This parameterization has exactly three trainable parameters:

```json
{
  "tool": 0.3333333333333333,
  "memory": 0.3333333333333333,
  "code": 0.3333333333333333
}
```

Run:

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_coeff_memory_only_push_n33_lr008_gpu075_i1_seed20260513
```

Script:

```text
skill/command/run_qbank_c033333_memory_only_global_coeff_push.sh
```

Main settings:

| setting | value |
|---|---:|
| gate parameterization | global-coefficient |
| trainable coefficients | tool, memory, code |
| train-coefficient lock | none |
| tasks | memory |
| prompts | 33 |
| samples per prompt | 4 |
| lr | 0.08 |
| prior_loss_weight | 0.0 |
| max_coefficient_delta_from_init | 0.6 |
| update_batch_size | 4 |
| loss_granularity | sequence |
| max_logprob_tokens | 1024 |
| advantage_normalization | centered |
| task_normalize_advantages | false |

Update summary:

| metric | value |
|---|---:|
| kept_frontier_rows | 23 |
| filled_missing_old_logprobs | 92 |
| optimizer_steps | 6 |
| skipped_optimizer_steps | 0 |
| grad_norm_max | 0.01089 |
| gate_delta_max | 0.31714 |
| clip_frac_mean | 0.82609 |
| approx_kl_mean | 40.08725 |

Final direct coefficients:

| task | before | after | delta |
|---|---:|---:|---:|
| tool | 0.3333 | 0.1808 | -0.1526 |
| memory | 0.3333 | 0.0162 | -0.3171 |
| code | 0.3333 | 0.4407 | +0.1074 |

The update trajectory moved away from the Memory vector immediately:

| optimizer step | tool | memory | code |
|---:|---:|---:|---:|
| 1 | 0.2533 | 0.2533 | 0.4133 |
| 2 | 0.1857 | 0.1769 | 0.4041 |
| 3 | 0.1763 | 0.1103 | 0.4306 |
| 4 | 0.1835 | 0.0826 | 0.4334 |
| 5 | 0.1782 | 0.0521 | 0.4409 |
| 6 | 0.1808 | 0.0162 | 0.4407 |

Reward comparison:

| condition | rows | samples | mean reward_train | success samples | zero samples | frontier rows | all-zero rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| train rollout before update, seed 20260513 | 33 | 132 | 0.3106 | 41 | 91 | 23 | 9 |
| post-update eval, seed 20260514 series | 33 | 132 | 0.2273 | 30 | 102 | 15 | 16 |

The post-update eval seed was changed to `20260514` series only to avoid evaluating on the exact same stochastic samples used for the update. The prompt set stayed fixed. A same-seed rerun can reduce sampling noise for paired debugging, but it would no longer be an independent post-update sample.

Interpretation:

- Direct three-parameter training confirms the coefficients can all move without a `common + residual` bottleneck.
- However, in Memory-only PPO with all three direct coefficients free, the gradient pushed probability mass toward the Code vector and strongly suppressed the Memory vector.
- This is opposite to the locked `global.memory` run, where Memory moved to `0.7356` and reward rose sharply.
- Therefore the important issue is not just parameter freedom. The current on-policy likelihood gradient is assigning credit through the full task-vector basis, and Memory-correct samples in this rollout appear more aligned with the Code direction than with the Memory vector direction.

Immediate implication:

For reachability, keep task-specific coefficient control when diagnosing a task. For mixed training, do not assume unlocking all three global coefficients will let each task discover its own vector. We need either task-conditioned updates, per-task phases, or an auxiliary alignment/check that verifies the reward-positive samples actually produce gradients in the intended task-vector direction.

## Direct Three-Coefficient, Old-Style Conservative Settings

A follow-up run tested the same direct `global-coefficient` parameterization, but with optimizer/objective settings matched to the earlier stable `global_i20` run as closely as possible. The only intentional differences were:

- task filter: Memory prompts only;
- gate form: direct `tool/memory/code`, no `common + residual`;
- GPU placement: isolated on physical GPUs 4-7.

Run:

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_coeff_memory_only_like_old_i2_20260513_2350
```

Settings:

| setting | value |
|---|---:|
| gate parameterization | global-coefficient |
| tasks | memory |
| num iters | 2 |
| prompts | 33 |
| samples per prompt | 4 |
| lr | 0.003 |
| prior_loss_weight | 0.01 |
| max_coefficient_delta_from_init | 0.2 |
| loss_granularity | token |
| batch_loss_reduction | mean |
| update_batch_size | 4 |
| advantage_normalization | zscore |
| task_normalize_advantages | true |
| use_frontier_weight | true |
| length_normalize_policy_logprob | true |
| max_logprob_tokens | 12288 |

Note on `max_logprob_tokens`: this matches the old run's default path (`MAX_MODEL_LEN=12288`) but is a compute cap, not an expected output length. In observed rollouts, Memory outputs are typically around 1.5k-2.1k max response length. The 12k token-level update is therefore too slow for search; each update took about 20 minutes.

Results:

| iter | memory reward | success samples | frontier rows | tool coeff | memory coeff | code coeff | approx_kl | clip_frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 rollout/update | 0.3561 | 47 / 132 | 23 | 0.3190 | 0.3381 | 0.3341 | 0.000280 | 0.00270 |
| 2 rollout/update | 0.3485 | 46 / 132 | 22 | 0.3165 | 0.3420 | 0.3294 | -0.000062 | 0.00222 |

Interpretation:

- Under the old conservative update regime, direct three-coefficient training no longer collapses Memory. Memory moves upward monotonically: `0.3333 -> 0.3381 -> 0.3420`.
- The movement is very small, similar in character to the original `global_i20` run, where Memory also rose gradually rather than jumping.
- Reward does not improve over these two Memory-only iterations: `0.3561 -> 0.3485` on the second on-policy rollout. This is within sampling noise and does not support a short-horizon reward gain.
- The bad direct-coefficient run above was primarily caused by aggressive sequence-level updating (`lr=0.08`, `approx_kl≈40`, `clip_frac≈0.826`), not by direct coefficients being inherently unusable.
- For practical search, use `max_logprob_tokens=4096` unless a specific audit requires 12288. It should cover normal Memory trajectories while avoiding 20-minute updates.
