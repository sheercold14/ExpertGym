# Calibration Design for ExpertGym

## 核心原则

Calibration 不是普通训练集，也不是为了把 proxy reward 调高而堆 prompt。它应该是一个 executable probe bank：每条样本都必须说明自己给哪一种 credit。

中心目标：

```text
让 gate 在 frozen task-vector 空间里看到可泛化的 reward 梯度，而不是记住 96 条 prompt。
```

因此 calibration bank 必须同时满足：

1. reward 与正式 eval 的 verifier 语义一致；
2. OPD target 是 same-prompt verified positive，不是泛化 imitation；
3. 样本覆盖 frontier / recoverable / stable / unsolved 四类状态；
4. 有独立 heldout calibration monitor，用来早停和选 checkpoint；
5. 正式 eval6 只用于最终验证，不参与调参。

## 四类样本角色

| state | 模型当前表现 | expert 表现 | 训练信号 | 作用 |
|---|---|---|---|---|
| frontier | K 个 rollout 有对有错或 reward 有方差 | 不要求 | GRPO | 提供方向梯度 |
| recoverable | current all-fail | same-prompt expert verified positive | Recovery-OPD | 推 gate 朝可恢复能力移动 |
| stable | current all / mostly success | 不要求 | NLL retention | 防止已经会的能力被破坏 |
| unsolved | current all-fail | expert 也无 verified positive | 不更新 / 数据获取 | 标记当前 expert set 上限 |

推荐每任务 32 条的初始目标比例：

| task | frontier | recoverable | stable | unsolved | 说明 |
|---|---:|---:|---:|---:|---|
| tool | 8-10 | 10-12 | 6-8 | 2-4 | live/parallel/schema 类 frontier 和 recoverable 都重要 |
| memory | 6-8 | 10-12 | 8-10 | 2-4 | memory 需要保留已会轨迹，避免只推 final answer |
| code | 8-10 | 10-12 | 4-6 | 4-6 | code 必须保留 generation 与 test-selection 两类 failure |

这个比例不是硬编码数据分布，而是 state-selection 目标。实际构建时应先对候选池 rollout，再按状态采样。

## Reward-Aware 设计

### Tool

正式短板：BFCL live / canonicalization / default / parallel alignment。

Calibration 设计：

- 保留一半原始 ToolRL/RLLA anchor，避免偏离专家训练分布。
- 加入 fresh BFCL-style synthetic，但不能复制 eval prompt/schema/entity。
- 每条样本必须能被当前 Tool reward adapter 验证 reference call。
- reward 分解记录：
  - parse / JSON validity；
  - function name；
  - call count / parallel alignment；
  - argument exactness；
  - enum exactness；
  - default-value discipline；
  - canonicalization。

OPD target：

- 只选 expert 输出中官方 adapter 判定 positive 的 `<tool_call>`。
- 同 prompt current all-fail 时才施加 OPD。
- stable tool 样本只做 retention，不做 distillation。

防过拟合：

- heldout tool monitor 必须使用不同 function names、不同 schema slots、不同实体语言。
- 不能用同一个 synthetic template 的变量替换同时出现在 train 和 monitor。

### Memory

正式短板：不能只看 final answer，要关注 MemAgent trajectory。

Calibration 设计：

- 使用 HotpotQA train/dev disjoint question/article。
- 每条样本保留 update turns + final turn。
- reward 仍以官方 final verifier 为主，但训练日志必须记录 trajectory 长度、update turns 数、final answer success。
- 如果可接入官方 MemAgent reward，应优先用其 trajectory reward；否则至少保证 OPD loss 覆盖 update turns + final turn。

OPD target：

- 只选 memory expert 在同 prompt 上 final verified positive 的完整轨迹。
- OPD NLL 覆盖 update turns + final turn，不只 final answer。
- current all-fail 且 expert success 时进入 recoverable。

防过拟合：

- train / monitor 按 question id 和 Wikipedia article id 双重去重。
- heldout monitor 中保留长轨迹样本，防止模型只学短 HotpotQA。

### Code

正式短板：CURE/LiveCodeBench 更像 stdin/stdout + hidden tests；当前 calibration 经常只过 public examples，训练 reward 不涨。

Calibration 设计：

- 不直接使用 formal eval prompt/tests。
- 从 train-side CodeContests / fresh generated tasks 构造 CURE-style prompt。
- 每条样本至少有多组 unit tests，区分 public-like 与 hidden-like tests。
- reward 分解记录：
  - parse code block；
  - syntax；
  - public tests pass rate；
  - hidden-like tests pass rate；
  - full pass；
  - selection success：是否在多 sample 中选到正确代码。

OPD target：

- 不是自然语言解题过程，而是 verified code solution。
- expert positive 必须通过该 prompt 的 selected tests。
- 对 code prompt 可用 BoN expert rollout 扩大 positive pool，但 positive 文件必须单独存放并标记 expert/source/seed。

防过拟合：

- train / monitor 题目来源 disjoint。
- generated tasks 必须换数据生成种子、题面实体和 hidden tests。
- checkpoint 选择不能只看 calibration train reward，必须看 monitor any-pass / pass@1 / BoN selection。

## 两层 Calibration Bank

推荐组织：

```text
/tmp/shared-storage/OnPolicy/data/calibration/expertgym_72h/
  train96_v1/
    prompts.jsonl
    expert_rollouts/
    state_audit/
    summary.json
  monitor96_v1/
    prompts.jsonl
    expert_rollouts/
    state_audit/
    summary.json
```

使用方式：

| bank | 用途 | 是否反传 |
|---|---|---|
| train96 | 产生 GRPO / OPD / retention 梯度 | yes |
| monitor96 | 每 N iter rollout，选 checkpoint / 早停 / 检查过拟合 | no |
| formal eval6 | 最终报告 | no，且不能频繁调参 |

如果 train reward 上涨但 monitor 不涨，判定为 calibration overfit，不晋级正式 eval。

## 72h 可执行版本

优先级：

1. 用 P0 state distribution 审计 `paper96` 在 `TA-1/3` 和 `init1` 下的 state 结构。
2. 对已有 `eval_targeted96_cure_aligned_20260517` 做同样 state audit。
3. 组合出 `train96_balanced_v1`：
   - 先保留 paper96 中 state 有效的 frontier/recoverable/stable；
   - Tool 补 BFCL-style recoverable/frontier；
   - Code 补 CURE-style verified-code recoverable/frontier；
   - Memory 保持 HotpotQA 轨迹多样性。
4. 构建 `monitor48_v1` 与 `guard48_v1`，同分布但 prompt/schema/problem disjoint。
5. P1 训练只看 train96 反传，monitor/guard 只做 selection 和 non-regression。

推荐目录：

```text
/tmp/shared-storage/OnPolicy/data/calibration/calib_bank_v1_20260518/
  train96.prompts.jsonl
  monitor48.prompts.jsonl
  guard48.prompts.jsonl
  expert_rollouts/
  state_tables/
  summary.json
```

### train96_balanced_v1 目标配比

每条样本必须带元信息：

```text
task, state, failure_tags, source_family, leakage_safe, opd_source, heldout_group
```

| task | frontier | recoverable | stable | unsolved | 样本类型 |
|---|---:|---:|---:|---:|---|
| tool | 12 | 11 | 7 | 2 | ToolRL anchor 8；BFCL non-live parallel 6；BFCL live-style canonical/default/enum 10；parallel_multiple/mixed namespace 8 |
| memory | 8 | 8 | 14 | 2 | HotpotQA train final-answer 16；long-context final 8；memory-update retention 8 |
| code | 10 | 14 | 5 | 3 | paper96 code anchor 8；CURE-style generation 12；selection/BoN failure 8；partial-edge 4 |

全局目标：frontier 约 31%，recoverable 约 34%，stable 约 27%，unsolved 不超过 8%。unsolved 只进入 acquisition queue，不进入 loss。

### monitor / guard 切分

heldout 不能随机切分，必须按 family 切：

| task | family split |
|---|---|
| tool | schema namespace / entity / language / enum family disjoint |
| memory | HotpotQA question id / article id / answer entity disjoint |
| code | task id / algorithm tag / test generator / solution template disjoint |

`monitor48` 用于 early stop 和 checkpoint selection，不进梯度。`guard48` 只用于最终 non-regression 检查，禁止调参回看。

## 论文可讲的 insight

如果实验成立，论文故事不是“我们调了更好的 calibration”，而是：

- calibration prompts are probes；
- 不同 state 对应不同 credit operator；
- reward-aware probe selection 比随机 small calibration 更能暴露 executable composition signal；
- heldout monitor 说明 gate 学到的是 task-vector composition，而不是记住 calibration prompt。

## 不应做的事

- 不把 formal eval 题直接放入 calibration。
- 不用 expert 所有轨迹做 NLL；只用 current-failure + expert-success 的 same-prompt recovery。
- 不用 train calibration reward 选择最终 checkpoint。
- 不把 code public example pass 当作 full positive。
- 不在 Tool synthetic 中复用 BFCL eval schema 或实体。
- 不把 `paper96` 直接当作下一版主 bank；它只保留为 baseline/anchor 和 state-audit 对照。
