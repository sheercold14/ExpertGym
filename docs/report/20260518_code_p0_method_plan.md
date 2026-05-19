# 2026-05-18 Code P0 Method Plan

## 结论

当前主线必须转成 **Code-first**：只有当训练阶段 Code reward 增益能够稳定传递到 heldout monitor，并进一步提升正式 CURE/LiveBench/LiveCodeBench 指标时，才把它作为论文方法。否则它只是 proxy overfit 或 task-vector 系数漂移。

最重要的判断标准：

```text
train code reward 上升
  -> monitor/guard code reward 上升
  -> formal CURE sample/code acc 或 BoN acc 上升
```

三段缺一不可。当前 recoverable101 / SOTA-v2 实验能推动 gate 中的 code 系数，但正式 Code 结果尚未证明同比例提升，因此下一阶段不再把 Code 当 overall reward 的附属项，而是单独建立可验证训练信号。

## 当前证据

### 正式评测进度

`recoverable101 GC iter3` 已完成 Tool / Memory 部分，Code 仍在 CURE unit-test 执行阶段。

- baked policy: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/sota_v2_recoverable101_gc_init1_grpo_opd_ret_20260518/iter_004/baked_policy`
- Code eval log: `/tmp/shared-storage/ExpertGym/eval/p0_sota_v2_20260518/gc_i3/code/run.log`
- LiveBench code partial: sample/code acc `0.3730`，BoN `(4,4)` acc `0.4375`

这不是足够强的 Code 证据。需要等 LiveCodeBench 完整结果后再判断，但现有 partial 已说明：只让 code gate 上升，不等价于 CURE 能力一定上升。

### 现有 Code calibration 审计

`sota_calib_v2_20260518/train128.prompts.jsonl`：

| task | rows |
|---|---:|
| tool | 32 |
| memory | 48 |
| code | 48 |

Code row 中：

- `32` 条在 `reference.metadata` 内直接带 `test_input/test_output`；
- `16` 条 paper96 anchor 只带 `question_id`，reward adapter 会回查 `CodeContests_train`；
- 每条可解析到约 `10` 个 source tests，训练实际默认最多执行 `OPVEC_CODE_REWARD_MAX_TESTS=8`。

`sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl`：

| task | rows |
|---|---:|
| tool | 32 |
| memory | 48 |
| code | 21 |

Code recoverable positive 来源：

- ReasonFlux-Coder rollout positive: `39` 个 sample；
- DeepSeek-R1-Distill-Qwen-7B rollout positive: `35` 个 sample；
- 最终只有 `21/48` 个 Code prompt 保留为 recoverable。

关键问题不是“完全没有 tests”，而是：

1. **Code recoverable 覆盖不足**：只有 21 条，且 OPD positive 稀疏。
2. **测试集合过弱或过窄**：训练只执行 source tests 的前若干条，可能不能代表 hidden-like boundary。
3. **训练只覆盖 code generation，不覆盖 CURE/ReasonFlux 的 case/test selection 分支**：正式 BoN `(4,4)` 依赖生成测试/选择器能力。
4. **train prompt 与 formal eval 能力轴未闭环**：训练 code reward 上升时，必须看 monitor/guard 和 CURE subset 是否同步上升。

## 方法定义

### Code Bank 三类样本

Code calibration 不再用单一 pass-rate prompt bank，而拆成三类 credit operator：

| 类别 | 当前模型表现 | expert 表现 | 训练信号 | 目标 |
|---|---|---|---|---|
| generation | current samples 全错或几乎全错 | expert 至少一个 verified pass | OPD + GRPO | 提高 pass@1 / any-pass |
| frontier | current K samples 有对有错，reward 有方差 | 不强制 | GRPO z-score | 利用相对优势提升 code reward |
| partial-edge | 能过样例/部分 tests，但挂 hidden-like tests | expert 或 reference 可过 | OPD + guard tests | 修复边界、格式、复杂用例 |

`selection` 暂时作为第二阶段：如果 formal BoN 低但 any-pass 高，则需要训练 unit-test/case generator 或 selector；第一阶段先把 code generation 能力拉起来。

### Reward 设计

训练 code reward 必须执行代码，而不是自然语言相似度或 public example fallback。

推荐每条 Code row 显式存：

```text
reference.metadata.reward_test_input
reference.metadata.reward_test_output
reference.metadata.guard_test_input
reference.metadata.guard_test_output
reference.metadata.code_tags
reference.metadata.source_task_id
```

训练反传只用 `reward_tests`；monitor/guard 只读不反传。这样能防止“前 8 个 source tests 过拟合”，同时保留可审计来源。

当前 `CodeRewardAdapter` 已支持从 `reference.metadata.test_input/test_output` 或 `question_id` 回查 CodeContests tests。下一版应把 train/guard test split 显式化，避免隐式读取和不可控 test ordering。

### Advantage 与样本数

对 Code，优先对齐 CURE：

```text
raw reward = source/hidden-like tests pass rate
advantage = per-prompt z-score
no variance row = 不进入 GRPO frontier
samples_per_prompt = 8 起步，关键实验用 16
code_max_new_tokens >= 10000
```

OPD 只用于 `current all-fail + expert verified pass` 的 same-prompt positive。不能把 expert 的所有 code 都蒸馏给模型，否则会变成普通 imitation，而不是 on-policy repair。

### 成功判据

每个 Code-first 实验必须记录三层指标：

| 层级 | 指标 | 用途 |
|---|---|---|
| train bank | code mean reward / success / frontier rows / all-fail rows / OPD rows | 判断训练信号是否有效 |
| monitor/guard | heldout code reward、pass@1 proxy、any-pass proxy | 判断是否泛化到同分布未反传题 |
| formal CURE | LiveBench sample acc、LCB sample acc、BoN acc | 判断是否能写进论文主结果 |

只有同时满足：

```text
monitor code reward 不下降
formal CURE code 指标上升
tool/memory 不发生明显 collapse
```

才晋级为主方法 checkpoint。

## 下一步实验

### P0-1：构建 Code-first v3 Bank

状态：已完成第一版构造器和数据输出。

代码入口：

```text
scripts/data/build_code_p0_calibration_bank.py
skill/command/build_20260518_code_p0_v3.sh
docs/config/20260518_code_p0_v3.md
```

当前输出：

```text
/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/
  train_code64.prompts.jsonl
  monitor_code32.prompts.jsonl
  guard_code32.prompts.jsonl
  code_p0_blueprints.jsonl
  summary.json
  README.md
```

校验：

```text
train rows 64 unique_task_ids 64 bad 0
monitor rows 32 unique_task_ids 32 bad 0
guard rows 32 unique_task_ids 32 bad 0
total unique 128
```

关键设计：

- 只用 CodeContests train，不读正式 CURE/LiveBench/LiveCodeBench prompt/tests/output。
- 每题显式拆 `reward_test_input/output` 和 `guard_test_input/output`。
- 当前 `CodeRewardAdapter` 读取 `reference.metadata.test_input/test_output`，构造器已将其设置为 reward slice，因此不改 reward 主路径。
- split 按 CodeContests `task_id` 完全 disjoint。

Expert rollout 覆盖已完成：

| expert | rows | covered prompts | coverage | mean reward |
|---|---:|---:|---:|---:|
| ReasonFlux-Coder-7B | 64 | 29 | 0.4531 | 0.4089 |
| DeepSeek-R1-Distill-Qwen-7B | 64 | 31 | 0.4844 | 0.4306 |

用两个专家合并后，`train_recoverable_code.prompts.jsonl` 选出 36/64 条可恢复 Code prompt：

| role | rows |
|---|---:|
| generation | 7 |
| frontier | 15 |
| partial_edge | 11 |
| stable | 3 |

这比旧 `sota_calib_v2_recoverable_code` 的 21 条 Code recoverable 更适合作为 Code P0 的 OPD 初始信号；但仍必须用 monitor/guard 和 formal CURE 验证，不能只看 train reward。

目录建议：

```text
/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/
  train_code64.prompts.jsonl
  monitor_code32.prompts.jsonl
  guard_code32.prompts.jsonl
  expert_rollouts/
  state_audit/
  summary.json
```

数据来源：

- CodeContests train，不使用 LiveBench/LiveCodeBench prompt/tests/output；
- 保留部分 paper96 anchor，但不超过 25%；
- 重点采样 `stdin_stdout / format_sensitive / math / greedy / array / string / simulation / graph`；
- 每题拆出 reward tests 与 guard tests，source task id 去重。

目标配比：

| split | generation recoverable | frontier | partial-edge | stable | rows |
|---|---:|---:|---:|---:|---:|
| train | 24 | 20 | 12 | 8 | 64 |
| monitor | 10 | 10 | 8 | 4 | 32 |
| guard | 10 | 10 | 8 | 4 | 32 |

### P0-2：Code-only Sanity

目的：先证明 gate 在 Code bank 上真的能学习，不被 Tool/Memory 信号稀释。

设置：

```text
task filter = code
init = 1/3 或 init1 各一条短跑
samples_per_prompt = 8
advantage = zscore
loss = GRPO + OPD + retention
max_new_tokens = 10000
train = train_code64
monitor = monitor_code32
```

通过标准：

- train code reward 连续上升；
- monitor code reward 同步上升；
- all-fail 数下降；
- gate code 系数变化方向和 reward 变化一致。

### P0-3：Joint Non-collapse

在 Code-only 有效后，再放回 Tool/Memory：

```text
train = tool32 + memory32/48 + code64
code samples_per_prompt = 8/16
tool/memory samples_per_prompt = 4
code frontier quota 独立控制
retention: tool/memory stable rows 固定保留
```

目标不是 maximization of overall proxy，而是：

```text
code formal eval 上升，同时 tool/memory 不低于 TA-0.75 或 best-run non-regression 阈值。
```

### P0-4：Formal CURE 快速闭环

每个候选只先跑 Code quick eval：

- LiveBench code；
- LiveCodeBench subset；
- BoN `(4,4)`；
- 与 TA-0.75、best-ever、GC iter3 做 case-level diff。

若 quick eval 显示 Code 不涨，不进入全量 eval6。

## 论文方法叙述

这条线可以形成清晰 claim：

```text
ExpertGym does not merely tune task-vector coefficients on a small calibration set.
It constructs executable, reward-aware probes whose states determine the credit operator:
frontier -> GRPO, recoverable -> OPD, stable -> retention, unsolved -> acquisition.
For Code, executable unit-test probes are necessary for the learned gate movement to transfer to CURE.
```

审稿人可接受的证据链：

1. 随机/旧 96 calibration：Code proxy 或 gate 变化不能稳定转化到 CURE。
2. Code P0 v3 bank：train/monitor reward 同步上升。
3. 正式 CURE：sample acc / BoN acc 至少一个核心指标上升。
4. Case study：提升集中在 `stdin/stdout`、hidden-like、format-sensitive、partial-edge 等先前失败类型。

## 暂不做

- 不用正式 CURE prompt/tests 直接训练；
- 不把 overall reward 作为唯一选 checkpoint 标准；
- 不继续在 Code 未闭环前大规模调 Tool/Memory；
- 不把 selection/unit-test branch 和 generation branch 混成一个不可解释的 reward；
- 不把 init1 高 reward 当论文主方法，只作为 upper-initialization ablation。
