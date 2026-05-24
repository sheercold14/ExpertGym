# Pairwise-Zero Diagnostics for 8765

生成时间：2026-05-23 15:45:35

## 1. 为什么做两两分析

三专家联合分析容易把互补、冗余和伤害混在一起。这里每次只保留两个 expert，把第三个 expert 的 196 个 residual coefficient 置为 0，形成三个可审查的二元视角：

| 视角 | 保留 | 置零 | 目的 |
| --- | --- | --- | --- |
| TM(code=0) | Tool + Memory | Code | 看 Code residual 到底是能力还是干扰源 |
| TC(memory=0) | Tool + Code | Memory | 看 Memory residual 是否是 Code 冲突主因，以及 Memory 能力代价 |
| MC(tool=0) | Memory + Code | Tool | 看 Tool 是否主要是格式行为保护，而不是能力主干 |

注意：这些 gate 是诊断视角，不是最终方法。它们的作用是解释 residual 结构，而不是直接声称评测性能。

## 2. 二元 residual 冲突

| pair | task | active modules | opposite-sign rate | both-positive rate | expression dominance |
| --- | --- | ---: | ---: | ---: | --- |
| `tool_memory__code_zero` | tool | 196 | 0.500 | 0.474 | tool 0.04 / memory 0.00 |
| `tool_memory__code_zero` | memory | 196 | 0.648 | 0.316 | tool 0.00 / memory 0.28 |
| `tool_memory__code_zero` | code | 156 | 0.455 | 0.167 | tool 0.01 / memory 0.47 |
| `tool_code__memory_zero` | tool | N/A | N/A | N/A | unavailable: one expert lacks this task-span signal |
| `tool_code__memory_zero` | memory | N/A | N/A | N/A | unavailable: one expert lacks this task-span signal |
| `tool_code__memory_zero` | code | 136 | 0.529 | 0.294 | tool 0.01 / code 0.26 |
| `memory_code__tool_zero` | tool | N/A | N/A | N/A | unavailable: one expert lacks this task-span signal |
| `memory_code__tool_zero` | memory | N/A | N/A | N/A | unavailable: one expert lacks this task-span signal |
| `memory_code__tool_zero` | code | 150 | 0.607 | 0.187 | memory 0.27 / code 0.08 |

读法：`opposite-sign rate` 高，说明两个 expert 在同一任务 span 上经常推向相反方向；`both-positive rate` 高，说明它们更像协同。

## 3. 置零一个 expert 会删掉什么

| view | zero expert | removed rows | own expression | tool support | memory support | code positive strength | code negative strength | key removed roles |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tool_memory__code_zero` | code | 196 | 3.721 | 0 | 0 | 381.13 | 348.22 | code_repair_only:43, uninformative:45 |
| `tool_code__memory_zero` | memory | 196 | 2.966 | 20 | 84 | 285.90 | 368.61 | code_repair_only:4, code_repair_vs_protected_harm:14, code_negative_but_protected_support:21, code_source_conflict_with_behavior:104, uninformative:2 |
| `memory_code__tool_zero` | tool | 196 | 0.1313 | 126 | 0 | 97.46 | 113.17 | code_repair_only:13, shared_positive:17, code_repair_vs_protected_harm:2, code_negative_but_protected_support:37, code_source_conflict_with_behavior:63, uninformative:31 |

## 4. 现在最有价值的结论

1. **Code 不是纯干扰项。** `code=0` 会删掉 43 个 `code_repair_only` row 和 0 个 `shared_positive` row；已有 v15 评测也显示 Memory 上升但 Code 明显下降。所以不能把 Code expert 整体压掉。

2. **Memory-Code 是最需要解释的二元关系。** 在 `memory_code__tool_zero` 视角下，Code task 的 opposite-sign rate 为 0.607；Memory task 不能直接比较，因为当前 atlas 没有 Code expert 在 Memory span 上的 signed-effect。这仍然说明 Memory/Code 冲突需要 residual key + span 级诊断，不能简化成“memory 系数太高”。

3. **Memory 不能粗暴置零。** `memory=0` 会删掉 84 个 Memory support row，同时移除大量 Code 正/负混合证据。它可能让部分 Code 冲突减少，但代价是 Memory 轨迹能力失去主通道。

4. **Tool 更像格式行为保护。** `tool=0` 删除的 own-task expression 均值是 0.1313，通常比 Memory 小，但它删掉的是 tool-call span 上的格式行为锚点；这解释了 Tool live 的波动为什么不能只靠整体 reward 分析。

5. **论文主图应从三专家联合图改成二元图。** 二元图能清楚展示：哪个 pair 冲突、在哪个 task 上冲突、置零第三方会删掉哪些 role。这比三专家合在一起更符合第一性诊断。


## 5. 与已有评测证据的连接

已有 `code=0` 的真实 quick / code-hurt 评测可以作为 TM(code=0) 的外部验证：

| candidate | Tool quick | Memory eval50 F1 | Code hurt acc mean | Code hurt BoN mean |
| --- | ---: | ---: | ---: | ---: |
| reference `v9` | 0.7931 | 0.7575 | 0.2344 | 0.4375 |
| code zero `v15` | 0.7800 | 0.7841 | 0.1250 | 0.1562 |

这条证据说明：删除 Code residual 的确能释放一部分 Memory，但 Code 能力显著下降。因此 `code=0` 是 negative control，不是方法。

## 6. 可视化产物

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523/figures/pairwise_conflict_rate_heatmap.png`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523/figures/zero_expert_role_risk_heatmap.png`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523/figures/pairwise_expression_dominance.png`

## 7. 虚拟 gate 产物

- `tool_memory__code_zero`: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523/virtual_gates/tool_memory__code_zero/gates.json`
- `tool_code__memory_zero`: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523/virtual_gates/tool_code__memory_zero/gates.json`
- `memory_code__tool_zero`: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523/virtual_gates/memory_code__tool_zero/gates.json`

这些 gate 只用于诊断。如果后续要 bake / eval，应单独记录为 ablation，不要和主方法 checkpoint 混在一起。

## 8. 下一步建议

1. 论文图优先画二元冲突热图，而不是三专家总热图；读者更容易理解哪个 pair 在哪个 task 上冲突。
2. 将 `code=0` 放在 paper 里作为 scalar negative control：它释放 Memory，但摧毁 Code，证明全局置零不是方法。
3. 对 `memory=0` 和 `tool=0` 暂时只作为机制预测，不立刻跑完整评测；先用小规模 quick eval 验证预测方向。
4. 方法主线继续保持简单：二元诊断定位冲突，最终算法只做 residual-level soft constraint，而不是三专家黑盒联合调参。
