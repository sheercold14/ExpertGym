# 20260519 Stage-1 Training Control

## Goal

第一阶段目标是产出一个能力最强、可 bake、可复现实评的 merged checkpoint，作为后续 on-policy reward refinement 的初始化。当前不把 calibration proxy 作为最终目标；proxy 只用于筛选候选，最终判断以 Tool / Memory / Code 正式评测为准。

## Decision Priority

| priority | signal | use |
|---:|---|---|
| P0 | 正式 Eval6/Full suite Tool, Memory, Code | 决定第一阶段最强 checkpoint |
| P1 | calibration rollout reward by task | 判断是否值得继续训练或早停 |
| P1 | gate coefficient trajectory | 判断能力是否被推到合理区间，检查 collapse |
| P2 | training loss / gradient norm | 诊断动力不足、过冲、任务冲突 |

## Active Candidate Families

### TRC directional stage-1

TRC 使用 expert 成功轨迹做 dense residual alignment。v1/v2 的 MSE 目标会把非目标 expert residual 当成错误，导致多能力合并时互相压制；当前高价值版本是 v3 directional：

```text
L_dir = 1 - cos(r_merge, r_expert)
```

v3 只要求 merged residual 包含目标 expert 方向，不惩罚额外正交能力，因此比 MSE 更适合作为静态 gate 初始化。

当前正式评测候选：

| candidate | path | status |
|---|---|---|
| `trc_stage1_v3_anchor_i4_20260519` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i4_20260519` | full eval running |
| `trc_stage1_v3_anchor_i8_20260519` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i8_20260519` | full eval running |

### M-series nullspace on-policy stage-1

M1/M2 是正在运行的 on-policy OPD + retention + tool nullspace 对照：

| run | init | R1 | parameterization | high-level read |
|---|---|---:|---|---|
| `M1_null_r033_r1_005` | tool/memory/code 1/3 | 0.05 | 28-layer band | code gate 上涨，memory/tool 基本不动 |
| `M2_null_init05_r1_0` | tool/memory/code 0.5 | 0 | 28-layer band | proxy 高，但 gate 变化弱，OPD 经常因缺 tool all-fail 被跳过 |

M-series 是否进入正式评测，取决于后续 15-20 iter 是否出现同时满足三点的 checkpoint：overall proxy 不低、Tool 不崩、Memory 或 Code 至少一项较起点有可解释提升。

## Early Stop Rules

训练不应机械跑满。出现以下情况可以早停或降优先级：

1. Tool reward 连续 3 轮明显低于同初始化 baseline，且 tool gate 同步下降。
2. Memory gate 不再增长，同时 Memory reward 低于候选起点。
3. Code gate 单边增长但 Code reward 不涨，说明只是 loss/scale 偏置。
4. dynamic OPD 因 `require-all-tasks` 连续多轮跳过，说明当前 all-fail 可恢复集合已经不再支持 OPD 主导训练。

## Reporting Standard

每个进入候选池的 checkpoint 必须记录：

- checkpoint / gate path；
- train config 文档或启动命令；
- proxy reward by task；
- gate mean by expert；
- Tool / Memory / Code 正式评测；
- 是否作为第一阶段主候选。

主记录位置：

```text
docs/report/20260519_trc_stage1_takeover.md
docs/evaluation/20260519_stage1_candidates_eval.md
```

