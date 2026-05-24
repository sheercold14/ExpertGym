# 2026-05-22 Main Method: Residual Capability Field with Behavior Constraints

## Name

```text
RCF-BC = Residual Capability Field with Behavior Constraints
```

中文：

```text
带行为约束的残差能力场
```

## Why This Method

当前实验已经排除了两个更简单但错误的方向：

1. **Task scalar shrinkage**：把 code expert 减半或置零能提高 Memory，但会摧毁 Code。
2. **Hard role routing**：把 atlas 离散成少数 role/action 能保持 Tool/Memory，但恢复不了 Code。

因此主方法应是：

```text
continuous capability evidence field
+ behavior constraints
+ residual-level audit
```

而不是：

```text
single task coefficient
or
hard atlas role rule
```

## Algorithm

### Step 1: Code Capability Field

对同一 Code prompt 的 pass/fail trajectories 做 residual utility contrast：

```text
score(param, expert) =
  normalized utility(pass trajectory)
  - normalized utility(fail trajectory)
```

多个 source/span 使用 conservative aggregation：

- LiveBench prompt span
- LiveBench reasoning span
- LiveCodeBench code span
- LiveCodeBench prompt span

这一步得到连续的 Code capability field，而不是离散标签。

### Step 2: Behavior Utility Floor

用 Tool / Memory behavior-positive trajectories 找出不能降低的 residual：

```text
if residual supports Tool call behavior or Memory full trajectory:
    negative Code delta is floored at base coefficient
```

当前配置：

```text
--preserve-task tool
--preserve-task memory
--preserve-negative-scale 0.0
```

### Step 3: Behavior Harm Veto

用 Tool / Memory behavior evidence 找出提高后可能破坏行为的 residual：

```text
if residual harms Tool/Memory behavior:
    positive Code delta *= 0.5
```

当前主方法使用 soft constraint：

```text
--harm-veto-task tool
--harm-veto-task memory
--harm-veto-positive-scale 0.5
```

解释：

- Tool/Memory 不作为 reward scalar 混进 Code。
- 它们作为 behavior constraint 约束 Code capability field。
- 这保留了 v9 的 Code 能力，同时不让 Tool/Memory 崩。

### Step 4: Expert-Mean Recenter

默认保持每个 expert 的 mean coefficient 接近 base：

```text
--preserve-expert-mean default on
```

这防止方法退化成“整体加某个 expert”，使 residual-level steering 更可解释。

## Reproduction

RCF-BC 已作为一等候选加入：

```bash
PHASE=generate CANDIDATES=v18_rcf_bc \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/residual_capability_field_behavior_constraints_v18/gates.json
```

完整闭环文档：

```text
docs/harness/20260522_rcf_bc_reproducible_loop.md
```

完整主线命令：

```bash
PHASE=paper_main CANDIDATES=v18_rcf_bc TOOL_GPU=0 MEMORY_GPU_IDS=0 CODE_GPU=2 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

静态诊断面板：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_diagnostic_dashboard_20260522/index.html
```

冲突簇报告：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_clusters_20260522/conflict_cluster_report.md
```

数值等价检查：

```text
v18_rcf_bc == v9_soft
num_keys = 588
max_abs_diff = 0.0
different_count = 0
```

因此 v18 是语义化主方法名，v9 是同一方法的早期实验编号。

## Gate Structure

RCF-BC / v18:

```text
num_gates = 588
changed = 205
code mean = 0.900703
memory mean = 0.987386
tool mean = 1.004171
```

Decision reasons:

| reason | count |
|---|---:|
| preserve_utility_floor | 200 |
| below_min_abs_score | 204 |
| pass_fail_negative | 92 |
| behavior_harm_veto | 78 |
| pass_fail_positive | 14 |

This is the central empirical signature:

- many small positive and negative deltas;
- no global expert shrinkage;
- behavior evidence constrains the continuous Code field.

## Conflict Archetypes

Residual rows can be grouped into mechanism-facing archetypes:

| archetype | rows | changed | interpretation |
|---|---:|---:|---|
| `clean_code_repair` | 77 | 56 | sparse but distributed capability residuals |
| `code_repair_with_behavior_harm` | 28 | 21 | Pareto boundary between Code repair and Tool/Memory behavior |
| `code_source_conflict` | 279 | 83 | main evidence for continuous field over hard routing |
| `code_negative_with_behavior_support` | 58 | 2 | do not suppress behavior-supporting residuals blindly |
| `code_negative_noise` | 56 | 16 | cleanest suppression target |
| `behavior_only` | 12 | 5 | pure behavior constraint evidence |
| `weak_or_uninformative` | 78 | 22 | audit/background rows |

## Archetype Consistency Ablation

`v19_archetype_consistency` is a deterministic projection from `v18_rcf_bc`:

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/gates.json
```

It resets only deltas that contradict cluster semantics:

- weak/uninformative drift;
- behavior-only drift;
- positive delta on Code-negative noise;
- negative delta on behavior-supporting residuals;
- negative delta on Code repair rows.

Summary:

| metric | v18 source | v19 projected |
|---|---:|---:|
| changed rows | 205 | 173 |
| positive rows | 106 | 91 |
| negative rows | 99 | 82 |
| mean abs delta | 0.002639 | 0.002365 |

This is a mechanism ablation for the paper: if v19 preserves performance, archetype consistency improves interpretability; if it hurts Code, the result shows that low-confidence continuous drift can still carry useful capability signal.

Evaluation confirms the second case:

| model | Tool quick | Memory eval_50 F1 | LiveBench hurt acc / BoN | LiveCodeBench hurt acc / BoN |
|---|---:|---:|---:|---:|
| `v18_rcf_bc` | 0.880 / 0.855 / 0.8125 / 0.625 | 0.7575 | 0.2500 / 0.6250 | 0.6250 / 0.6555 |
| `v19_archetype_consistency` | 0.880 / 0.865 / 0.8125 / 0.625 | 0.7793 | 0.1250 / 0.1875 | 0.3281 / 0.4375 |

This makes `v18_rcf_bc` the better main method despite being less semantically clean: Code repair depends on a continuous low-confidence residual field. `v19` is useful as a negative ablation showing why RCF-BC should not over-project residuals into hard archetype rules.

## Current Evaluation

Because `v18_rcf_bc` is numerically identical to v9, it inherits v9's evaluation:

| model | Tool quick | Memory eval_50 F1 | LiveBench hurt acc / BoN | LiveCodeBench hurt acc / BoN |
|---|---:|---:|---:|---:|
| RCF-BC / v18 / v9 | 0.880 / 0.855 / 0.8125 / 0.625 | 0.7575 | 0.2500 / 0.6250 | 0.6250 / 0.6555 |

## Paper Claim

The clean claim:

> RL expert task vectors are mixtures of capability and behavior-changing residuals. RCF-BC estimates a continuous residual capability field from outcome contrast, then constrains it with behavior utility/harm evidence from other tasks. This exposes and controls mode conflict without reducing merging to global scalar search.

中文：

> RL expert 的 task vector 不是纯能力包，而是能力残差和行为改变残差的混合。RCF-BC 用成功/失败轨迹构建连续残差能力场，再用其他任务的行为 utility/harm 作为约束，从而在 residual 级别处理模式冲突，而不是依赖全局系数搜索。
