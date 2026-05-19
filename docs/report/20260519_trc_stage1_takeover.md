# 20260519 TRC Stage-1 Takeover Report

## Goal

第一阶段目标：用 expert 成功轨迹训练一个能力尽可能强的 gate checkpoint，作为后续 GRPO/OPD reward refinement 的初始化。该阶段不再只追求 calibration loss 最低，而是优先保证 Tool/Memory/Code 三种能力不互相破坏。

## Code State

已提交：

```text
5b92bb3 Add TRC stage1 balanced residual options
3b68ed4 Add directional TRC stage1 objective
```

核心脚本：

```text
scripts/trc/train_trc_layer_gates.py
skill/command/run_20260519_trc_layer_init1_v3_directional.sh
docs/config/20260519_trc_stage1_harness.md
```

默认旧 TRC v1 MSE 路径仍可复现；v2/v3 都通过显式 CLI 开关启用。

## Why V2 Was Stopped

v2 使用 target-normalized MSE + span-aware alignment，但仍优化：

```text
r_merge ≈ r_expert
```

这个目标会把非目标 expert residual 当作误差。静态 gate 要合并多能力，不能在 tool 样本上要求 code/memory residual 归零。v2 前两轮动态：

| run | epoch | tool | memory | code |
|---|---:|---:|---:|---:|
| v2 main | 2 | 0.8378 | 0.7671 | 1.2249 |
| v2 anchor | 2 | 0.8315 | 0.7669 | 1.2248 |

判断：方向错误，已早停。

## V3 Objective

v3 改为 directional TRC：

```text
L_dir = 1 - cos(r_merge, r_expert)
```

含义：merged residual 需要包含目标 expert 的能力方向，但不惩罚其他 expert 的额外正交分量。另加轻量 projection floor 和 coefficient floor，避免目标方向投影太低或某 expert coefficient 快速坍缩。

默认 v3 设置：

```text
init=1.0
residual_objective=directional
response_span_mode=auto
task_balanced_loss=on
accumulation_steps=96
directional_projection_floor=0.8
directional_projection_weight=0.1
coefficient_floor=0.9
coefficient_floor_weight=0.05
```

## V3 Training Results

### v3 directional

Run:

```text
/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_20260519
```

| epoch | residual loss | tool | memory | code |
|---:|---:|---:|---:|---:|
| 1 | 0.5914 | 1.0279 | 0.9700 | 1.0300 |
| 4 | 0.5315 | 1.1034 | 0.8798 | 1.1202 |
| 8 | 0.4423 | 1.1799 | 0.7583 | 1.2416 |

### v3 directional anchor

Run:

```text
/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_anchor_20260519
```

Differences: `lr=0.02`, `beta_base=0.05`, `gamma_gate=0.005`, `coefficient_floor=0.95`, `coefficient_floor_weight=0.1`.

| epoch | residual loss | tool | memory | code |
|---:|---:|---:|---:|---:|
| 1 | 0.5914 | 1.0171 | 0.9800 | 1.0200 |
| 4 | 0.5540 | 1.0606 | 0.9199 | 1.0801 |
| 8 | 0.4958 | 1.1068 | 0.8391 | 1.1609 |

判断：`v3_anchor_i4` 是均衡候选，`v3_anchor_i8` 是更强推候选，`v3_dir_i8` 是激进候选。

## Baked Candidates

```text
/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i4_20260519
/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i8_20260519
/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_dir_i8_20260519
```

三者均已成功 bake，`num_delta_entries=588`。

## Full Eval Status

正在评测：

```text
trc_stage1_v3_anchor_i4_20260519
trc_stage1_v3_anchor_i8_20260519
trc_stage1_v3_dir_i8_20260519
```

Tool/BFCL 已完成：

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple |
|---|---:|---:|---:|---:|
| v3_anchor_i4 | 0.885 | 0.855 | 0.750 | 0.625 |
| v3_anchor_i8 | 0.890 | 0.855 | 0.750 | 0.625 |
| v3_dir_i8 | 0.905 | 0.850 | 0.8125 | 0.625 |

Memory 已完成，Code/CURE 正在跑 LiveCodeBench 或等待补齐；正式表同步在：

```text
docs/evaluation/20260519_stage1_candidates_eval.md
```

当前已完成结果：

| model | Tool mean | Tool live mean | Memory EM | Memory F1 | Code LiveBench Acc | Code LiveBench BoN |
|---|---:|---:|---:|---:|---:|---:|
| v3_anchor_i4 | 0.7788 | 0.6875 | 0.6348 | 0.7603 | 0.3633 | 0.4297 |
| v3_anchor_i8 | 0.7800 | 0.6875 | 0.6406 | 0.7594 | 0.3770 | 0.4609 |
| v3_dir_i8 | 0.7981 | 0.7188 | 0.6445 | 0.7663 | pending | pending |

等待项：三个模型的 Code CURE 完整结果，尤其是 `dir_i8` 的 LiveBench / LiveCodeBench Acc / TP / BoN。

补充评测：`trc_stage1_v3_dir_i8_20260519` 已作为激进 directional 候选进入 full eval。当前 Tool mean=0.7981、live mean=0.7188，高于两个 anchor 候选；Memory mean F1=0.7663、EM=0.6445，也没有坍缩且略高于 anchor。关键风险转为 Code 是否能维持。

## Current Interpretation

- v3 解决了 v2 的 Tool 快速坍缩问题。
- `dir_i8` 目前在 Tool 和 Memory 的正式评测上均领先，是当前第一阶段最强候选，但必须等待 Code CURE 完成后才能定主模型。
- `anchor_i8` 比 `anchor_i4` 更推 Code/Tool，且 Memory 官方 F1 与 i4 基本持平；若 `dir_i8` Code 明显掉，`anchor_i8` 是保守备选。
- 如果 `dir_i8` Code 不显著低于 anchor，优先选 `dir_i8` 作为 stage-1 main checkpoint；否则保留 `dir_i8` 与 `anchor_i8` 双候选进入下一阶段。
