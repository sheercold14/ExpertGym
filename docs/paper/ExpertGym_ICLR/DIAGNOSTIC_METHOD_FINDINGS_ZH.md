# 8765 诊断方法与论文结论提炼

更新时间：2026-05-23

## 1. 定位

`http://127.0.0.1:8765` 不是普通训练曲线前端，而是 ExpertGym 当前论文的机制诊断台。它回答的问题不是：

```text
tool / memory / code 三个 expert 的全局系数应该是多少？
```

而是：

```text
每个 expert residual 在具体任务行为 span 上到底表达了什么，它是能力、保护行为、噪声，还是跨任务伤害？
```

因此，8765 给论文提供的是机制证据和方法设计依据，不应直接被写成“某个 checkpoint 最高分”的调参故事。

## 2. 诊断单位

ExpertGym 的 task vector 不能视为 task-pure direction。可解释单位应从 expert-level scalar 下沉到：

```text
r = (expert, layer, module)
```

当前主表对应：

```text
3 experts * 28 layers * 7 modules = 588 residual entries
```

对每个 residual entry，诊断台计算：

```text
u_r = Delta W_r h
expression(r) = mean ||Delta W_r h||^2
signed_effect(r) = mean[-<grad, Delta W_r h>]
```

`expression` 判断该 residual 是否在当前行为 span 上真的被激活；`signed_effect` 判断该激活方向一阶上是降低还是提高 teacher-forced behavior loss。这个定义能区分三类全局系数看不到的情况：

| 类型 | 含义 | 合并含义 |
| --- | --- | --- |
| 大但不表达 | 参数范数大，但当前 span 上 `DeltaW h` 小 | 不应仅凭 norm 保留或放大 |
| 表达且有益 | `expression` 高，`signed_effect` 正 | 可以作为能力或行为支持证据 |
| 表达但有害 | `expression` 高，`signed_effect` 负 | 需要抑制或 soft veto |

## 3. 行为 span

8765 的关键不是把整条回答压成一个 reward，而是先定义任务行为在哪里发生：

| 任务 | 诊断 span | 论文含义 |
| --- | --- | --- |
| Tool | tool-call / function-call span | 保护结构化调用格式和调用选择 |
| Memory | update turns + final answer trajectory | 保护长轨迹记忆更新和最终回答 |
| Code | prompt constraints + reasoning + final code block | 诊断代码理解、推理和可执行输出 |

这解释了此前训练中常见的不稳定现象：同一个 residual 可能对 Code final block 有益，但对 prompt constraint parsing、Memory trajectory 或 Tool call format 有害。

## 4. Code 的 outcome contrast

Code 不能只通过模仿 expert positive trajectory 来定义能力。更干净的信号是同 prompt 的成功/失败对比：

```text
code_utility(r) = signed_effect_r(pass trajectory) - signed_effect_r(fail trajectory)
```

这个定义问的是：哪个 residual entry 把同一道题从失败轨迹推向可通过测试的轨迹。它比“靠近 expert 风格”更接近评测能力，也解释了为什么单纯拉高 code expert 系数经常无效。

## 5. 当前最可靠的机制发现

| 发现 | 证据 | 方法含义 |
| --- | --- | --- |
| task vector 不是任务纯净方向 | 同一 expert delta 内有 utility、harm、format behavior 和 noise | 不能只做 expert-level scalar sweep |
| clean Code repair residual 稀疏 | `code_repair_only` 约 `60/588` | Code 需要 residual-level selection |
| 真正共享正向 residual 很少 | `shared_positive` 约 `17/588` | synergy 需要被识别，不能假设全局存在 |
| Code source/span 冲突常见 | `code_source_conflict_with_behavior` 约 `167/588` | Code 不能被视为一个平滑方向 |
| Memory-Code 是主冲突 | code-memory conflict 明显高于 memory-tool | Memory 需要 residual-level behavior protection |
| MLP 承载更多冲突 | MLP 中 conflict / harm 更集中，attention 更多弱证据 | 层/模块角色比 expert 名字更重要 |

最强的 Code 警告来自 source/span 分析：LiveBench prompt evidence 与 LiveCodeBench prompt evidence 在 residual key 上接近强负相关，Pearson 约 `-0.995`，sign conflict rate 约 `63.95%`。这说明 Code 能力不是一个可以平滑增大的单轴方向。

## 6. 从诊断到算法

当前最稳的论文方法应收束为：

```text
Behavior-Constrained Residual Composition
```

中文可以表述为：

```text
行为约束下的残差组合
```

最小规则如下：

| 证据类型 | 操作 |
| --- | --- |
| capability contrast 为正，且不伤 Tool/Memory | 提高该 residual |
| capability contrast 为正，但伤 protected behavior | soft shrink / soft veto |
| residual 支持 Tool/Memory behavior | 不低于行为保护底线 |
| 多任务符号冲突或 source/span 冲突强 | 保守处理，不做硬推 |
| evidence 弱或缺失 | 不动 |

这个规则的价值在于：它不是验证集 sweep 出来的系数，而是由 `DeltaW h` 的 span-conditioned utility / harm 直接导出。

## 7. 论文 claim 边界

当前 8765 诊断已经能支撑：

```text
Agent task vectors exhibit residual-level, span-conditioned utility and harm.
BCRC operationalizes this diagnostic structure with a simple behavior-constrained residual composition rule.
```

当前不应强行支撑：

```text
BCRC is SOTA across Tool, Memory, and Code.
```

原因是完整 Eval6 仍需要主方法和 ablation 共同确认；当前已经完成的 full rows 也显示 BCRC 更像一个可解释的 behavior-constrained operating point，而不是无条件最高分 checkpoint。

## 8. 实验使用原则

后续实验应按下面的判据推进，而不是继续盲目调 gate：

1. 如果 Code 不涨，先查 `pass/fail contrast` 和 `LB/LCB source-span conflict`，不要只看 code coefficient。
2. 如果 Memory 下降，查 Memory full trajectory span 上哪些 residual 被 Code 证据压掉。
3. 如果 Tool live 波动，优先看 tool-call span 的 parseability / exact call / zero-call，而不是只看 BFCL live 小样本均值。
4. 如果 hard routing 保护了 Tool/Memory 但 Code 下降，说明低置信 continuous residual drift 可能承载 Code 能力。
5. 如果 scalar shrink 能换来 Memory 提升但 Code 下降，它只能作为 negative control，不能作为主方法。

## 9. 二元置零诊断

三专家联合图容易把互补、冗余和伤害混在一起。当前新增的 8765 分析页采用二元视角：

| 视角 | 保留 | 置零 | 用途 |
| --- | --- | --- | --- |
| `TM(code=0)` | Tool + Memory | Code | 验证 Code residual 是否是纯干扰 |
| `TC(memory=0)` | Tool + Code | Memory | 观察 Memory 是否是 Code 冲突主因，以及删除 Memory 的能力代价 |
| `MC(tool=0)` | Memory + Code | Tool | 观察 Tool 是否主要承担格式行为保护 |

初步结论：

1. `code=0` 会释放 Memory，但显著破坏 Code，因此它是 negative control，不是方法。
2. `memory=0` 会删除大量 Memory support residual；它可能减少一部分 Code 冲突，但会失去 Memory 主能力通道。
3. `tool=0` 删除的能量相对小，但删除的是 tool-call span 的格式行为锚点，解释了 Tool live 的脆弱性。
4. 二元冲突图比三专家联合图更适合作为论文诊断图，因为它能清楚回答“谁和谁冲突、在哪个任务 span 冲突、置零第三者会删掉什么”。

产物：

```text
docs/report/RCRF/20260523_pairwise_zero_diagnostics.md
docs/report/RCRF/20260523_iclr_pairwise_zero_figure.md
docs/paper/ExpertGym_ICLR/figures/pairwise_zero_diagnostics.pdf
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523
```

论文使用边界：

- 可以说：置零一个 expert 会同时删除能力 residual、行为保护 residual 和冲突 residual，因此 expert-level zeroing 只能作为 negative control。
- 可以说：二元冲突具有 task-span 条件性，缺失 behavior probe 应显式标为 `N/A`，不能当成 0 冲突。
- 不应说：虚拟 `memory=0` 或 `tool=0` gate 已证明正式评测提升；它们目前是机制预测，需要单独 bake/eval 才能成为 benchmark 结论。

## 10. 一句话总结

8765 的核心结论是：

```text
Agent task-vector merging 的有效结构不是 expert-level coefficient，而是 residual entry 与 behavior span 交互产生的 utility / harm / conflict。
```

因此，论文主线应从“寻找最优全局系数”转向“诊断 residual-level 行为证据，并用简单保守规则做可审查的能力组合”。
