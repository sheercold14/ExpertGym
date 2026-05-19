# 2026-05-18 SOTA Recovery Calibration v3 Report

## 结论

为了冲正式 SOTA，calibration 不能再只是“均衡采样 96/128 条 prompt”。当前最关键的问题是训练动力：如果训练集中大量 Code hard rows 没有 expert positive，dynamic OPD 会跳过它们，GRPO 又只看到稀疏 frontier，最终 gate 只能在局部 proxy 上震荡。

v3 的策略是：

```text
训练集：verified-recoverable，保证 OPD / frontier / retention 都有信号
监控集：harder audit，检查是否真的泛化到 Code/Tool/Memory 能力
```

## 当前构建结果

路径：

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_recovery_calib_v3_20260518/
```

| split | rows | tool | memory | code |
|---|---:|---:|---:|---:|
| train128 | 128 | 32 | 48 | 48 |
| monitor64 | 64 | 16 | 24 | 24 |
| guard64 | 64 | 16 | 24 | 24 |

train128 来源：

| source bucket | rows | 含义 |
|---|---:|---|
| `sota_v2_recoverable` | 80 | Tool 32 + Memory 48 |
| `code_p0_v3_recoverable` | 36 | Code P0 中专家 verified positive 的 Code rows |
| `sota_v2_recoverable_code` | 12 | sota_v2 中 recoverable Code anchor |

monitor/guard 来源：

| source bucket | rows/split | 含义 |
|---|---:|---|
| `sota_v2` | 40 | Tool/Memory audit |
| `code_p0_v3_audit` | 20 | harder Code P0 audit rows |
| `sota_v2_eval_targeted` | 4 | Code anchor audit |

## 为什么这版更适合 SOTA

`sota_calib_v2` 的正式评测失败说明：把 hard Code probes 放进 train 不会自动提升 Code，反而会稀释 OPD。v2 的 Code expert union coverage 只有 `21/48`，其中 targeted CodeContests 行 coverage 约 `7/32`。这类题应该做 monitor/guard，而不是作为 OPD 主训练的核心。

`code_p0_v3` 的 recoverable split 有 `36/64` 条 verified positive。v3 把这 36 条作为 Code 主动力，并补 12 条 v2 recoverable anchor，使 Code train 达到 48 条，同时仍保留 Tool/Memory 的训练量。

## 快速收敛设置

推荐优先跑：

```bash
PHASE=train_hier GPU_LIST=4,5 \
  bash skill/command/run_20260518_sota_recovery_v3.sh
```

默认训练配置：

```text
init = 1.0 strong prior
gate = layer-band-parameter
loss = GRPO + dynamic OPD + NLL retention
frontier = 每任务最多 4 条
retention = 每任务最多 8 条
dynamic OPD = 每任务最多 32 条
rollout = auto shard，多卡并行
```

这个设置的目的不是把所有 prompt 都反传，而是让每轮进入 update 的样本都高信息密度：

```text
all-fail + expert-positive -> OPD 推能力
frontier -> GRPO 修方向
all-success -> NLL retention 防退化
```

## 风险与 stop rule

需要重点监控：

| 风险 | 监控信号 | 处理 |
|---|---|---|
| Code train 上涨但 monitor 不涨 | monitor64 code 持平/下降 | 不送 Eval6，回到 Code reward/test 设计 |
| Tool 被牺牲 | ToolRL all80 低于 TA-0.75 明显 | 停止或提高 retention/tool frontier |
| hierarchical 过拟合 | layer residual 大而 global 无规律 | 降到 global-parameter 对照 |
| OPD 后期耗尽 | all-fail + expert-positive 低于 3/task | 切到 GRPO+retention 或停止 |

## 与论文主线的关系

v3 支撑的论文故事是：

> task vectors provide structured priors; executable feedback learns their composition.

这里的 calibration 不是普通训练集，而是一个可执行反馈 probe bank：

- recoverable rows 证明当前组合缺某种能力，但 expert task vector 能补；
- frontier rows 给 GRPO 连续方向；
- stable rows 约束不退化；
- monitor/guard rows 防止只在 train proxy 上优化。

如果 v3 + hierarchical gate 在 monitor/official eval 上有提升，它可以成为主实验；如果只有 train proxy 提升，则结论是 calibration/reward 仍未对齐正式 eval，不应 claim SOTA。
