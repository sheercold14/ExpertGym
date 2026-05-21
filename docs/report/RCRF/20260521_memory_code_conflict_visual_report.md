# 2026-05-21 Memory-Code 冲突机制可视化报告

## 0. 结论先行

当前最清晰的机制发现是：**Code 能力不是一个平滑、单一的 task vector 方向；同一个 residual key 在不同 code 子分布、不同 span 上会出现强烈方向翻转。Memory 与 Code 的冲突也不是简单的 expert-level 冲突，而是 residual key 级别、source/span 条件化的冲突。**

因此，继续调 `gate` 阈值或强行保护某个 expert 不是主线。更合理的下一步是：把每个 residual key 的 **Code outcome contrast** 和 **Memory/Tool 行为 utility** 放在同一张证据表里，做保守的 residual routing：

- Code 正向、Memory/Tool 中性或正向：保留或增强。
- Code 负向、Memory/Tool 中性：抑制。
- Code 负向、Memory/Tool 正向：冲突位置，回到 base 或做任务条件化，不应简单强压或强推。
- Code 正向、Memory/Tool 负向：冲突位置，优先保守处理。

这条线有希望做，因为它不依赖 sweep，也不依赖 calibration 上端到端把非平滑 gate 空间学出来；它从模型响应机制出发，识别 residual 的 utility / harm / conflict，再做最简单的结构化合并。

## 1. 分析产物

分析目录：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521`

核心表格：

- `source_conflict_pairs.csv`：不同 code source/span 的 residual 方向冲突。
- `code_hurt_metrics.csv`：v1-v4 在 16 条 code hurt 子集上的 pass-any / candidate pass rate / test-point rate。
- `gate_memory_summary.csv`：不同 gate 版本对 memory expert residual 的改写统计。
- `memory_delta_rows.csv`：逐 residual key 的 memory gate delta。
- `preserve_rows.csv`：v4 中被 Memory utility floor 保护的 key。
- `analysis_summary.json`：汇总统计。

核心图：

![](/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521/source_pearson_heatmap.svg)

![](/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521/source_conflict_rate_heatmap.svg)

![](/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521/code_hurt_pass_any.svg)

![](/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521/code_hurt_test_point_rate.svg)

![](/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521/memory_delta_protection_counts.svg)

![](/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521/memory_delta_by_layer.svg)

## 2. Code source/span 的核心冲突

不同 Code 来源之间的方向冲突非常强，尤其是 prompt residual：

| left | right | Pearson | conflict / overlap | conflict rate |
|---|---:|---:|---:|---:|
| LB_prompt | LCB_prompt | -0.9948 | 376 / 588 | 63.95% |
| LB_prompt | LCB_code | -0.0145 | 332 / 588 | 56.46% |
| LB_reasoning | LCB_prompt | -0.0543 | 332 / 588 | 56.46% |
| LB_code | LCB_prompt | -0.0144 | 310 / 588 | 52.72% |
| LB_code | LCB_code | -0.0382 | 300 / 588 | 51.02% |
| LCB_code | LCB_prompt | 0.0199 | 246 / 588 | 41.84% |

这说明两件事：

1. **Code 不是一个统一方向。** LiveBench 和 LiveCodeBench 对同一 residual key 的需求经常相反。
2. **prompt span 不是无关项。** Code prompt residual 里包含任务理解、约束解析、格式控制等能力，但它的方向高度依赖数据分布；盲目对齐 prompt 会把另一类 code 任务拉坏。

这解释了为什么过去“只推 code gate”或者“只学 expert positive residual”容易失败：我们并没有在优化一个全局光滑的 code scalar，而是在许多 residual key 上同时面对分布条件化的正负证据。

## 3. v1-v4 对 Code hurt 子集的证据

| variant | 方法摘要 | LB pass-any | LB test-point | LCB pass-any | LCB test-point | 结论 |
|---|---|---:|---:|---:|---:|---|
| v1_code_only | 只用 final-code contrast | 0.2500 | 0.3105 | 0.7500 | 0.3214 | 能修一部分 LCB，但 LB 弱，span 信息不足。 |
| v2_spanaware | prompt/reasoning/code span-aware conservative routing | 0.4375 | 0.3398 | 0.8125 | 0.5189 | 当前最干净的正结果：Code hurt 修复明显，Tool 基本不掉，Memory 小幅下降。 |
| v3_memory_hard_floor | v2 + memory expert negative delta 全部保护 | 0.4375 | 0.4121 | 0.6250 | 0.3424 | 证伪 expert-level hard preserve：LB 有帮助，但 LCB 大幅损伤。 |
| v4_memory_utility_floor | v2 + 小规模 Memory utility floor | 0.3125 | 0.3730 | 0.6250 | 0.3697 | 小规模 utility signature 太粗或太噪，不能替代 source/span 级证据。 |

最重要的是 v2 和 v3/v4 的对比：

- v2 说明 **source/span-aware 的保守 residual routing 有实际能力恢复信号**。
- v3 说明 **“只要是 memory expert 的负向 delta 就保护”过粗**，会压掉 LCB 需要抑制的 memory residual。
- v4 说明 **用一个小 Memory signature 做 floor 也不够**；它保护了一些 memory utility key，但这些 key 与 code 子分布仍可能冲突。

## 4. Memory gate 改写统计

| variant | memory changed | positive | negative | mean abs delta | max abs delta | protected / floored | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| v1_code_only | 196 | 82 | 114 | 0.01059 | 0.02926 | 0 | Code-only 会广泛改写 memory residual。 |
| v2_spanaware | 195 | 66 | 129 | 0.00748 | 0.03629 | 0 | 更保守，但仍对 memory residual 有大量负向抑制。 |
| v3_memory_hard_floor | 25 | 25 | 0 | 0.00212 | 0.03125 | 130 | 过强保护 memory expert，导致 LCB 修复明显变差。 |
| v4_memory_utility_floor | 173 | 64 | 109 | 0.00685 | 0.03559 | 21 | 小规模 Memory utility floor 没有解决冲突。 |

这张表的含义不是“Memory delta 越少越好”，而是：

**Memory expert residual 里既有 Memory 行为需要的方向，也有对某些 Code 子分布有害的方向。**

所以 expert-level 的保护粒度不对。真正要判断的是：某个具体 residual key 在某类 behavior span 上是否提供 utility，在另一个任务/source 上是否产生 harm。

## 5. 对 Tool / Memory / Code 的统一解释

### Tool

Tool 的核心脆弱点是格式和 tool-call behavior span。之前 Tool 在 BFCL live 上波动大，说明单看全局 reward 或全局 gate 不能稳定保护 tool-call 输出结构。对于 Tool，合理证据应该来自：

- tool-call span 的 NLL / behavior utility；
- BFCL non-live 与 live 子类的分布分解；
- 历史模型能够做对、当前合并模型容易丢失的 tool-call 样本。

Tool 不应该简单靠“tool expert gate 高”解释；它需要识别哪些 residual key 维护 tool-call 行为，哪些 key 是冗余或干扰。

### Memory

Memory 的关键能力不是 final answer，而是 retrieval/update/final 的轨迹。Memory F1 受完整行为链影响，所以只保护 final span 或只看少量 boxed answer 不够。Memory utility 应该来自：

- update turns + final turn 的行为 span；
- 正确轨迹相对失败轨迹的 hidden/residual difference；
- 对 HotpotQA/RAM/MemAgent 风格评测一致的 trajectory signal。

当前 v2 的 Memory 小幅下降提示：Code 修复中存在少量 Memory utility key 被误抑制。

### Code

Code 当前最明确的问题是：能力方向不是单纯 code expert residual。LiveBench / LiveCodeBench 中，prompt 理解、约束解析、边界条件、稳定输出和最终代码都可能贡献正确率。只对齐 final code block 会漏掉关键能力；直接加 prompt 又会因 source 冲突而伤害另一分布。

因此 Code 的更合理闭环是：

- 对 hurt subset 找 pass / fail 轨迹；
- 分离 prompt / reasoning / final code span；
- 统计每个 residual key 对不同 source/span 的正负贡献；
- 只采用跨 source 一致正向的 key；冲突 key 保守处理。

## 6. 现在不该继续做什么

不建议继续做以下事情作为主线：

- 继续围绕 v3/v4 的阈值调参。
- 只看 gate 系数是否推高。
- 用 expert-level 保护替代 residual-level 证据。
- 用单一 code calibration reward 期待 GRPO 自动找到全局最优组合。
- 把 Memory / Tool / Code 都压成一个统一 scalar 方向。

原因是现有证据已经表明：冲突发生在 residual key 和 source/span 条件上，而不是简单的任务级系数空间。

## 7. 建议的下一步方法

我建议把方法收束为一个更简单、可解释、可论文化的版本：

### 7.1 数据证据

对每个任务只保留少量高价值样本：

- Code：hurt subset + train-only code 样本，覆盖 LiveBench / LiveCodeBench 的 prompt、reasoning、final code span。
- Memory：完整轨迹样本，覆盖 update/final behavior span。
- Tool：BFCL non-live/live tool-call span，优先选历史模型可做对、当前合并容易掉的样本。

### 7.2 Residual 证据表

对每个 expert residual key 统计：

| 字段 | 含义 |
|---|---|
| code_utility | Code pass 轨迹相对 fail 轨迹是否需要该 residual。 |
| memory_utility | Memory 正确轨迹行为 span 是否需要该 residual。 |
| tool_utility | Tool-call 正确行为 span 是否需要该 residual。 |
| conflict_count | 该 key 在不同任务/source/span 上符号冲突次数。 |
| agreement_count | 该 key 在不同任务/source/span 上符号一致次数。 |
| confidence | 证据强度，来自样本数量、效应量、跨 source 一致性。 |

### 7.3 最小 routing rule

只使用四类规则：

1. 多任务一致正向：保留或轻微增强。
2. 多任务一致负向：抑制。
3. 单任务正向、其他任务中性：保留。
4. 任务间冲突：回到 base / 不动 / 降低幅度，不做强决策。

这比 sweep 更像方法，因为它用的是模型行为证据，而不是验证集上搜索 scalar。

## 8. 当前最有价值结论

如果要写成论文里的 insight，可以这样表达：

> Task vectors are not task-pure directions. Their utility is localized to residual keys and conditioned on the behavior span and evaluation source. Naively increasing or preserving an expert vector can amplify both useful capability and distribution-specific harm. A more reliable merge should route residuals by outcome-aware utility and cross-task conflict evidence.

中文版本：

> task vector 不是任务纯净方向。一个 expert residual 里同时包含能力、冗余和对其他任务有害的分布特异成分。能力表达发生在具体 residual key 和具体行为 span 上；因此合并时不应该只学任务级 gate，而应该根据 outcome-aware utility 和跨任务冲突证据选择 residual。

## 9. 立刻可执行的研究路线

短期不继续调阈值，改为做诊断闭环：

1. 在 Memory 完整轨迹上重新构建 behavior-span utility，不再用小规模 final/signature 近似。
2. 在 Tool tool-call span 上构建 behavior utility，分 live / non-live。
3. 把 Code v2 已有效的 source/span contrast 与 Memory/Tool utility 放在同一 residual evidence table。
4. 用固定、简单的 conservative routing rule 生成一个新 gate。
5. 先评 Tool + Memory，过门槛后再评 Code，避免 Code 长评测阻塞迭代。

这条路线比继续做 OPD/GRPO gate 学习更稳定，因为它绕开了非平滑 reward 和小 calibration 上的梯度不可靠问题，同时仍然服务于论文核心：从能力轨迹中识别 task vector 的 utility/harm/conflict。

