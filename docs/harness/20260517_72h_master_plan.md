# 2026-05-17 ExpertGym 72h Master Plan

## 目标

在 72 小时内把 ExpertGym 从“很多探索实验”收束成一条可投稿论文主线：

1. 证明 agent task-vector composition 不能只靠 geometry，需要 executable feedback。
2. 证明 calibration prompts 是 probes，不是普通训练集：frontier / recoverable / stable / unsolved 提供不同 credit。
3. 证明 same-prompt Recovery-OPD 是 verifier-grounded recovery，不是 generic imitation。
4. 在 Tool / Memory / Code 的正式 eval6 上给出非退化的主实验结果和清晰诊断。

## 当前事实

已完成的正式 eval6 说明：

| 模型 | Tool | Tool Live | Memory F1 | Code Acc | Code BoN | 价值 |
|---|---:|---:|---:|---:|---:|---|
| best-ever TAME | 0.7954 | 0.7083 | 0.7720 | 0.3597 | 0.4408 | 当前最强参考，不是 ExpertGym 主方法 |
| TA-0.75 | 0.7850 | 0.6875 | 0.7588 | 0.3585 | 0.4222 | 强静态 baseline |
| OP-VEC GP OPD best iter9 | 0.7835 | pending | 0.7649 | 0.3487 | 0.4144 | 历史诊断参考；不作为主线起点或复现实验基础 |
| ExpG init1 GRPO+OPD+Ret final | 0.7942 | 0.7083 | 0.7548 | 0.3382 | 0.4252 | ExpertGym 训练能提升 Tool/BoN，但 Code Acc 不够 |
| ExpF init1 GRPO+Ret final | 0.7788 | 0.6875 | 0.7612 | 0.3460 | 0.3998 | init1 + GRPO/Ret 更稳，Code Acc 当前最高的 ExpertGym run |
| ABC-A 1/3 GRPO+OPD | 0.7823 | 0.6771 | 0.7346 | 0.3431 | 0.3919 | 1/3 起点下最均衡的正式 run |
| OPD-B step012 i15 | 0.7835 | pending | 0.7171 | 0.3360 | 0.3636 | OPD-only 能推动 proxy，但 formal Code 不强 |

高价值结论：

- OPD-only + retention 是有效的早期 recovery 阶段，proxy 能从约 0.42 推到 0.66-0.70。
- Full GRPO+OPD+retention 的正式 eval 没有压倒 TA-0.75，但能产生可解释的 Tool / Memory / Code trade-off。
- init=1 的静态 prior 很强，说明 task-vector prior scale 是核心变量；论文不能把 1/3 当唯一合理起点而不解释。
- Code 的主要短板不是只有“缺 expert positive”，而是 formal CURE 与 calibration/reward 的差异、generated-test selection、stdin/stdout 和 hidden tests。
- Tool formal live 短板集中在 BFCL live/canonical/default/parallel alignment，不是简单 tool gate 不够。
- 当前 overall 最强参考是 best-ever TAME，其次是 TA-0.75；ExpertGym 目前不能 claim 全面 SOTA，只能 claim 在可执行反馈下做 non-regressive refinement、给出可解释诊断，并在局部分项上逼近或超过静态 prior。
- `opvec-gp-opd-best-iter9` 只能作为历史诊断参考。主实验不能建立在某个偶然好 checkpoint 上，必须从原则化 prior 出发：`TA-1/3` 作为 symmetric prior，`init1/scale-calibrated` 作为 strong geometric prior。

## 论文 claim 风险与处理

| 风险 | 审稿人质疑 | 处理 |
|---|---|---|
| OPD 看起来像 imitation | “你只是 distill expert” | 做 same-prompt recoverable vs offline imitation；报告 OPD 触发条件和 expert-success verifier guard |
| GRPO 弱且慢 | “为什么不用 distillation/sweep” | 先做 state distribution，说明 frontier 稀疏，GRPO 只负责 frontier direction |
| init1 比 1/3 强 | “1/3 claim 不成立” | 把 1/3 定义为 symmetric prior，把 init1/scale-calibrated prior 作为 stronger geometric prior；方法可从任意 prior 接 executable feedback |
| formal eval 不显著超过 TA | “没有性能收益” | 主 claim 调整为 non-regressive executable coefficient learning + diagnostic insight；只在有数值支撑处 claim improvement |
| module/layer gates 不稳 | “过拟合小 calibration” | capacity ladder 作为 P2，必须配 held-out guard；不作为主 claim |
| coefficient learning 不新 | “AdaMerging/Expert Merging 已经学系数” | 区分 unlabeled hidden/logit/entropy alignment 与 executable state-conditioned recovery；补 AdaMerging/Expert Merging baseline 或明确待补 |
| 方法只是 sweep | “global 3 个系数为什么不用 grid search” | 做或引用 static scale/sweep defense；强调 on-policy state routing 和 non-regression |

## 72 小时阶段划分

### T0-T8：P0 信号诊断与当前 run 收尾

目标：填论文 E3 state distribution，决定后续用 1/3 还是 scale-calibrated prior 做主线。

必须完成：

- `P0-state-c033-k8`: 1/3 prior，96 或 150 balanced prompts，K=8，统计 frontier/recoverable/stable/unsolved。
- `P0-state-init1-k8`: init=1 prior，同样 prompt/K，判断强 prior 是否改变 credit state。
- `TA-1/3 formal eval6`: 补严格 equal-prior checkpoint 的正式评测；TA-0.75 不能替代 E1/E2 里的 symmetric prior 行。
- 当前 b1/c1 run 继续观察，但只作为诊断，不自动晋升主实验。

晋级规则：

- 如果 1/3 的 all-fail/recoverable 占比高、frontier 低，论文中明确“1/3 exposes credit collapse”；主性能实验可以加入 scale-calibrated prior。
- 如果 init1 formal/proxy 明显更好，作为 “strong geometric prior + executable refinement” 路线。

### T8-T28：P1 主实验闭环

目标：在 5 小时内跑出可送 eval 的主方法候选。

P1 启动前置条件：

- 完成 `paper96` 在 `TA-1/3` 和 `init1` 下的 state distribution。
- 基于 `docs/harness/calibration_design.md` 生成或选择 `train96 / monitor48 / guard48`，不能只把 `paper96` 当作主 bank。
- 每个训练 run 只在 `train96` 上反传；`monitor48` 用于早停/选 checkpoint；`guard48` 用于最终 non-regression。

并行 3 组，每组 2 GPU：

| Run | 目的 | 起点 | 参数空间 | Loss | 预算 |
|---|---|---|---|---|---|
| `EG-main-gc-c033-fast` | 1/3 global 3 主方法 | 1/3 | global-coefficient | routed GRPO+OPD+Ret | 12 iter / <5h |
| `EG-main-gp-c033-fast` | common+residual / global-parameter 对照 | 1/3 | global-parameter | routed GRPO+OPD+Ret | 12 iter / <5h |
| `EG-main-gc-init1-fast` | strong prior refinement | init1 | global-coefficient | routed GRPO+OPD+Ret | 12 iter / <5h |

快速设置：

```bash
FRONTIER_ROWS_PER_TASK=4
FRONTIER_SAMPLE_BEFORE_LIMIT=1
MAX_RETENTION_ROWS_PER_TASK=8
MAX_RETENTION_ROWS=24
RETENTION_SAMPLE_BEFORE_LIMIT=1
UPDATE_BATCH_SIZE=8
GRADIENT_CHECKPOINTING=1
```

晋级规则：

- proxy overall 比起首轮提升 `>=0.08`。
- 三任务中任一任务 formal proxy 不出现连续 3 轮崩溃。
- gate 有可解释移动，不是全靠 clip/retention 抵消。
- 选每组 best proxy checkpoint，送小评测或完整 eval6。

### T28-T44：P1/P2 防御实验

目标：证明不是普通 loss mixing / imitation。

优先级：

1. `OPD-only + Ret`: 复现 early recovery，作为 “Recovery credit exists”。
2. `GRPO-only + Ret`: 证明 frontier direction 单独不足/慢，但有方向性。
3. `offline OPD / imitation`: 如果工程可控，用 static expert positives 不看 current all-fail，证明 generic imitation 更容易造成 regression。
4. `unrouted weighted sum`: 若工程量过大，可先用 “no state filter OPD + frontier + retention all mixed” 近似，不把它作为最终强 claim。
5. `static sweep defense`: global 3/4 系数空间必须至少有 best static scale 或小网格，否则主方法会被看作手动搜索。
6. `TA-1/3 formal eval`: 必补 E1/E2 equal-prior reference，避免只和 TA-0.75 比较造成 baseline 选择偏差。

### T44-T60：正式 eval6 与论文主表填数

最多送评 4 个模型：

| 类型 | 数量 | 选择规则 |
|---|---:|---|
| strongest ExpertGym | 1 | proxy overall + non-regression 最好 |
| best 1/3 ExpertGym | 1 | 支撑论文默认起点 |
| OPD-only / GRPO-only ablation | 1-2 | 支撑 E4/E5 |
| capacity/c1 | 0-1 | 只有 proxy 明显强才送 |

评测结果统一写：

```text
docs/evaluation/YYYYMMDD_expertgym_72h_eval6.md
```

### T60-T72：论文改写与图表

目标：把实验结果回填论文而不是继续盲跑。

必须产出：

- `main.tex` 表格填关键数值或附 `TBD but available in report`。
- 修正 GRPO 表述：ratio 是 old/new policy trajectory probability ratio under coefficient-induced policies，不是 coefficient ratio。
- 增补 AdaMerging / Expert Merging 的区别与 baseline 风险。
- state distribution 图/表。
- main comparison 表。
- routing / recovery ablation 表。
- coefficient trajectory 图。
- 失败分析段落：Code/Tool calibration mismatch。

## GPU 调度原则

8 卡划分：

| GPU | 默认用途 |
|---|---|
| 0,1 | 主训练 A |
| 2,3 | 主训练 B |
| 4,5 | 主训练 C 或 state rollout |
| 6 | eval worker / Tool/Memory |
| 7 | eval worker / Code 或备用 |

规则：

- 训练每组默认 2 卡。
- 不同时启动超过 3 个训练 run，保留至少 1-2 卡用于 eval 或救火。
- 任何 run 超过 5 小时未达到晋级信号，停止并记录为失败诊断。
- 完整 eval6 只给候选，不给每个 ablation。

## 停止规则

训练中止条件：

- `overall proxy` 连续 4 轮下降且无单任务改善。
- 任一任务 reward 降到 baseline 以下超过 0.08 并持续 3 轮。
- `dynamic OPD rows` 三任务合计低于 3 且 frontier 也低于 6，继续训练无信号。
- `gate delta` 接近 0 且 grad norm 低，说明信号耗尽。
- 单轮耗时 > 35min，改用 fast frontier/retention sampling。

晋级到正式评测条件：

- proxy overall 为当前 batch top-2。
- worst-task drop 不超过 预设 `0.03-0.05`。
- gate trajectory 可解释，且不是只提升一个任务、牺牲两个任务。
