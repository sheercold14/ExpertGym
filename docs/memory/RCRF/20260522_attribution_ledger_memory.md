# 2026-05-22 RCF-BC Attribution Ledger 记忆

## 背景

用户提出疑问：`code` expert 系数是否整体太大，是否应该调小或置零。

已有机械消融已经回答了全局版本：

| 模型 | code mean | Tool quick mean | Memory F1 | LB hurt acc / BoN | LCB hurt acc / BoN |
|---|---:|---:|---:|---:|---:|
| `v18_rcf_bc / v9` | 0.9007 | 0.7931 | 0.7575 | 0.1406 / 0.2500 | 0.3281 / 0.6250 |
| `v14_code_half` | 0.4504 | 0.7944 | 0.7774 | 0.2031 / 0.2500 | 0.2500 / 0.1875 |
| `v15_code_zero` | 0.0000 | 0.7800 | 0.7841 | 0.0781 / 0.1250 | 0.1719 / 0.1875 |

结论：全局压低 `code` 会释放 Memory，但会破坏 Code，甚至影响 Tool live。问题不是“code 系数太大”这么简单，而是 code residual 内部混合了必要能力、冗余和跨任务干扰。

## 新增 ledger

脚本：

```text
scripts/analysis/build_rcrf_attribution_ledger.py
```

复现入口：

```bash
PHASE=ledger bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_attribution_ledger_20260522/
docs/report/RCRF/20260522_rcrf_attribution_ledger.md
```

它把三层证据合并到 588 行 residual ledger：

1. `rcrf_conflict_clusters_20260522/conflict_cluster_rows.jsonl` 的 row-level 机制簇。
2. `v18_rcf_bc` 当前系数和相对 base 的 delta。
3. `rcrf_counterfactual_effects_20260522/counterfactual_effect_summary.json` 的 group-level mechanical ablation 效果。

## 关键统计

| decision | rows | changed | 含义 |
|---|---:|---:|---|
| `audit_before_prune` | 60 | 12 | 低置信或负向 code 行不能直接剪，需要先 audit；剪 group 会伤 Tool/Memory |
| `keep_capability_delta` | 77 | 56 | 干净 code repair 行，可以作为能力 delta |
| `keep_continuous_field` | 279 | 83 | mixed source conflict 行，不能 hard route，需要连续场 |
| `soft_constrained_capability` | 28 | 21 | code repair 但有 behavior harm，需要软约束 |
| `protect_behavior_support` | 66 | 4 | 支持 Tool/Memory 行，不能因为 code negative 就删 |

counterfactual group：

| group | rows | changed | 观察 |
|---|---:|---:|---|
| `v22` | 15 | 3 | `code_negative_noise` 单独减半会伤 Tool live 和 Memory |
| `v23` | 45 | 9 | `weak_or_uninformative` 单独减半也会伤 Tool live 和 Memory，但 LiveCodeBench 单采样可能升 |

## 方法启发

这一步把问题从 task scalar 转为 row-level attribution：

> 一个 residual row 的标签只是 proxy hypothesis，必须结合 counterfactual effect 和 behavior-support evidence 决定处理方式。

因此后续框架应保持：

1. **continuous residual capability field**：混合证据行不能二值剪掉。
2. **behavior-support audit**：看起来 code 负向或 weak 的行，如果承载 Tool/Memory behavior，不能轻易压低。
3. **counterfactual validation**：每个高层标签都要经过 bake 后 Tool/Memory/Code 指标验证，避免把相关性当因果。

论文中可把全局 code half/zero 和 row-level ledger 放在一起，作为“task-vector 不是 task-pure direction”的核心证据。

## 2026-05-22 追加：可执行动作层

ledger 已追加三列：

| 字段 | 含义 |
|---|---|
| `routing_action` | 当前 row 在框架里的使用方式，例如保留能力 delta、行为约束、连续场、禁止未验证剪枝 |
| `validation_priority` | 下一轮验证优先级：`high / medium / low` |
| `next_validation` | 可读的下一步验证要求 |

当前动作分布：

| routing_action | rows | 作用 |
|---|---:|---|
| `retain_capability_delta` | 77 | 干净 code 能力行 |
| `retain_continuous_field` | 279 | mixed evidence 行，不能 hard route |
| `retain_with_behavior_constraint` | 28 | code 能力与 Tool/Memory 行为冲突，需要软约束 |
| `protect_behavior_anchor` | 66 | Tool/Memory behavior anchor |
| `do_not_prune_without_counterfactual` | 60 | 低置信/负向 code 行，剪之前必须做反事实验证 |
| `behavior_guard` | 31 | 行为负约束或 veto |
| `keep_small_until_validated` | 18 | 低置信小 delta |
| `hold_base` | 29 | 暂时不动 |

验证优先级已收窄：

| priority | rows | changed |
|---|---:|---:|
| `high` | 172 | 91 |
| `medium` | 387 | 114 |
| `low` | 29 | 0 |

这让框架从“解释已有模型”推进到“指导下一轮实验”：高优先级行是下一步最值得做 isolated shrink/restore、behavior anchor audit 或 source/span counterfactual 的对象。

## 2026-05-22 追加：Validation Planner

新增脚本：

```text
scripts/analysis/build_rcrf_validation_plan.py
```

复现：

```bash
PHASE=validation_plan bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
docs/report/RCRF/20260522_rcrf_validation_plan.md
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_validation_plan_20260522/
```

planner 读取 ledger 后按以下 key 聚合成验证卡：

```text
routing_action / validation_priority / archetype / expert / layer_band / module_family
```

每张卡包含：

- hypothesis：机制假设；
- success criterion：什么评测变化能证明机制成立；
- failure read：如果不成立应该如何解释；
- representative keys：优先干预的 residual row。

生成结果：

| item | value |
|---|---:|
| validation cards | 48 |
| P0 cards shown in report | 12 |

P0 排序当前集中在：

1. early-layer `code_source_conflict` 的 code residual；
2. Memory residual 上的 `code_repair_with_behavior_harm` Pareto boundary；
3. `code_negative_with_behavior_support` 的 behavior anchor；
4. `code_negative_noise` 的 behavior guard。

这一步把 RCF-BC 的长期闭环固定为：

```text
residual evidence -> ledger decision -> routing action -> validation card -> counterfactual experiment -> write back to ledger/effect table
```

后续如果要继续推进，应该优先从 P0 card 生成最小 isolated gate intervention，而不是继续全局调 expert 系数。

## 2026-05-22 追加：Validation Interventions

新增脚本：

```text
scripts/analysis/build_rcrf_validation_interventions.py
```

复现：

```bash
PHASE=validation_interventions bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
docs/report/RCRF/20260522_rcrf_validation_interventions.md
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/validation_card_interventions_20260522/
```

默认从 validation plan 取前 6 张 P0 card，生成可 bake 的 OP-VEC `gates.json`。每个候选：

- 仍包含 588 个 gate key；
- 只改一张 card 的 representative residual rows；
- 不跑 bake，不跑 eval，不改变主方法 `v18_rcf_bc`。

auto operation 规则：

| routing action | operation | 意义 |
|---|---|---|
| `retain_continuous_field` | `drop-delta` | 测试这组连续 delta 是否必要 |
| `retain_capability_delta` | `drop-delta` | 测试 clean ability delta 是否因果 |
| `retain_with_behavior_constraint` | `half-delta` | 测试 Pareto boundary 的连续缩放 |
| `protect_behavior_anchor` | `shrink-coeff` | 测试行为 anchor 被压低是否伤 Tool/Memory |
| `do_not_prune_without_counterfactual` | `shrink-coeff` | 测试低置信 row 是否真的可剪 |
| `behavior_guard` | `drop-delta` | 测试当前 guard delta 是否必要 |

当前已生成 6 个候选，覆盖：

1. `code_source_conflict / code / early_00_09 / attention`
2. `code_source_conflict / code / early_00_09 / mlp`
3. `code_repair_with_behavior_harm / memory / middle_10_19 / mlp`
4. `code_source_conflict / memory / early_00_09 / attention`
5. `code_repair_with_behavior_harm / memory / late_20_27 / attention`
6. `code_source_conflict / memory / early_00_09 / mlp`

下一步如果要验证框架，应 bake 这些候选，先跑 Tool/Memory quick，再决定是否跑 Code hurt。通过后把结果写回 counterfactual effect table 和 ledger。
