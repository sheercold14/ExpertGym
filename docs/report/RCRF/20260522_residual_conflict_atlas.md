# RCRF Residual Conflict Atlas：从候选 gate 到可泛化能力归因

## 1. 目的

当前目标不是继续调一个单点 gate，而是回答：

> 对三类 RL expert task vector，哪些 residual 是能力表达，哪些 residual 是模式冲突，哪些 residual 是冗余噪声？

因此新增 `Residual Conflict Atlas`，把已有机制证据对齐到同一个最小单位：

```text
(param_name, expert)
```

当前三专家 OP-VEC 共有：

```text
28 layers x 7 linear modules x 3 experts = 588 residual entries
```

每一行 residual 同时记录：

- Code pass/fail contrast：同 prompt 成功轨迹相对失败轨迹的 residual direction。
- Tool behavior utility/harm：tool-call span 是否需要保护。
- Memory behavior utility/harm：full trajectory span 是否需要保护。
- v8/v9/v10/v11 gate delta：不同 routing 策略实际怎么动它。

这个 atlas 是 RCRF 的核心诊断层：它把“模型能力是否冲突”从整体 reward 现象拆成 residual-level 可审查证据。

## 2. 复现命令

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/build_rcrf_conflict_atlas.py \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522
```

也可以走统一 harness：

```bash
PHASE=atlas bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

- rows: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_rows.jsonl`
- csv: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_rows.csv`
- summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_summary.md`

## 3. 角色定义

| role | 含义 | 处理原则 |
|---|---|---|
| `code_repair_only` | Code pass/fail 明确正向，Tool/Memory 无保护冲突 | 可安全提高 |
| `shared_positive` | Code repair 与 Tool/Memory support 同向 | 协同 residual，优先保留/提高 |
| `code_repair_vs_protected_harm` | Code 正向，但伤 Tool/Memory behavior span | Pareto 冲突，需要 soft/hard routing |
| `code_repair_shared_and_harm` | 同时有 support 和 harm，说明 residual 多义 | 不应二值化，适合 task-typed routing |
| `code_negative_but_protected_support` | Code 负向，但 Tool/Memory 需要它 | 不能因为 Code 压低而破坏 protected behavior |
| `code_source_conflict_with_behavior` | Code 不同 span/source 符号冲突，且有行为证据 | 必须回到 span 级诊断，不能 scalar sweep |
| `protected_support_only` | 无 Code 信号，但 Tool/Memory 需要 | preservation floor |
| `protected_harm_only` | 无 Code 信号，但伤 Tool/Memory | pruning / veto 候选 |
| `uninformative` | 当前证据不足 | 不动 |

## 4. 关键统计

总行数：`588`

| role | count |
|---|---:|
| `code_source_conflict_with_behavior` | 167 |
| `code_source_conflict` | 112 |
| `uninformative` | 78 |
| `code_repair_only` | 60 |
| `code_negative_but_protected_support` | 58 |
| `code_negative_noise` | 56 |
| `shared_positive` | 17 |
| `code_repair_vs_protected_harm` | 16 |
| `code_repair_shared_and_harm` | 12 |
| `protected_support_only` | 7 |
| `protected_harm_only` | 4 |
| `protected_mixed_behavior` | 1 |

最重要的结论：

1. 只有 `60/588` 是相对干净的 Code repair residual。
2. 真正 `shared_positive` 只有 `17/588`，说明三任务协同 residual 很稀缺。
3. `167/588` 是 Code source conflict 且带 Tool/Memory behavior 证据，说明 Code 不是单一方向；prompt / reasoning / code span 之间会相互冲突。
4. `58/588` 是 Code 负向但 protected behavior 正向，说明“为了 Code 压低某些 residual”会直接伤 Tool/Memory。

这解释了为什么 scalar gate / GRPO gate 容易失败：全局系数无法区分这些 residual role。

## 5. Expert 分布

| expert | 高价值观察 |
|---|---|
| `code` | `code_repair_only=43`，但 `code_source_conflict=93`，说明 code expert 自身也不是纯 Code 能力包。 |
| `memory` | `code_source_conflict_with_behavior=104`，是冲突主体；Memory residual 同时承载 trajectory behavior 和部分 Code span 信号。 |
| `tool` | `shared_positive=17`，`code_negative_but_protected_support=37`；Tool residual 里存在少量协同能力，但更多是需要保护的 behavior span。 |

## 6. Module / Layer 分布

Module family：

| module_family | 观察 |
|---|---|
| `mlp` | `code_source_conflict_with_behavior=102`，比 attention 的 65 更高，是主要冲突承载位置。 |
| `attention` | `uninformative=77`、`code_negative_noise=48` 较多，更像包含大量弱信号或非目标 span 信号。 |

Layer band：

| layer band | 观察 |
|---|---|
| early `0-9` | `code_source_conflict=60`，早层 span/source 冲突强。 |
| middle `10-19` | `code_source_conflict_with_behavior=71`，中层是 Memory/Tool behavior 与 Code 方向冲突的核心区域。 |
| late `20-27` | `code_repair_only=29`，晚层更适合做 Code repair，但也有 `code_negative_but_protected_support=24`。 |

## 7. v8-v11 为什么形成 Pareto Frontier

关键 role 上的平均 delta：

| role | count | v8 hard | v9 soft | v10 ratio | v11 task-typed |
|---|---:|---:|---:|---:|---:|
| `code_repair_only` | 60 | 0.003280 | 0.006710 | 0.004860 | 0.005998 |
| `code_repair_vs_protected_harm` | 16 | 0.001787 | 0.006699 | 0.004591 | 0.004487 |
| `code_repair_shared_and_harm` | 12 | 0.000776 | 0.006164 | 0.003638 | 0.003794 |
| `shared_positive` | 17 | 0.000455 | 0.001981 | 0.000948 | 0.001267 |
| `code_negative_but_protected_support` | 58 | 0.000641 | 0.000641 | 0.000641 | 0.000641 |

解释：

- v8 hard 几乎不推动 conflict positive residual，因此 Tool/Memory 强，但 Code repair 弱。
- v9 soft 对 `code_repair_only` 和 `code_repair_vs_protected_harm` 都有较大推动，所以 Code hurt subset 最好，但 Memory 稍弱。
- v10 ratio 把很多 conflict residual 压得过低，是一个负结果。
- v11 task-typed 介于 v8 和 v9 之间：Tool/Memory 更稳，Code 较弱。

这说明 RCRF 不是在找一个 magic coefficient，而是在暴露一个可解释的 Pareto frontier。

## 8. 方法收束

RCRF 的原则应收敛为：

1. 先做 residual role attribution，不直接训练 gate。
2. `code_repair_only` 与 `shared_positive` 是主要可提高集合。
3. `code_repair_vs_protected_harm` 是 Pareto knob，必须按 protected task 类型处理。
4. Tool call-span harm 更接近 hard constraint。
5. Memory full-trajectory harm 更接近 soft constraint，因为它和 Code reasoning 共享更多 residual。
6. 对 `code_source_conflict_with_behavior`，不应该继续全局 sweep；应该增加 span/source 诊断，确认到底是 prompt、reasoning 还是 final code 在冲突。

## 9. 论文级 Claim

当前 atlas 支持一个比“gate 调参”更稳的 claim：

> RL expert task vectors are not task-pure. Their residual utility is outcome-aware and span-conditioned. A residual conflict atlas can reveal whether each residual is a capability carrier, behavior-preservation constraint, synergy point, or conflict point, enabling principled Pareto routing for expert composition.

中文：

> RL expert 的 task vector 不是纯任务能力包。每个 residual 的作用依赖任务、输出 span 和成功/失败结果。RCRF 通过 residual-level conflict atlas，将模式组合问题从黑箱系数搜索转化为可审查的能力归因和 Pareto routing 问题。
