# 2026-05-22 RCRF Operating Point Delta Diagnosis

## Purpose

v9 是当前最强的连续证据 operating point；v13/v16/v17 是更可解释的离散 role/source routing，但 Code 明显弱。这个诊断回答：

> 离散 routing 相比 v9 到底丢掉了哪些 residual delta？

新增脚本：

```bash
PYTHONDONTWRITEBYTECODE=1 $PY scripts/analysis/compare_rcrf_operating_points.py
```

输出目录：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522
```

## Main Results

| candidate | changed | positive | negative | mean abs delta |
|---|---:|---:|---:|---:|
| v9 continuous overlay | 205 | 106 | 99 | 0.002639 |
| v13 positive-only role | 73 | 73 | 0 | 0.002073 |
| v16 source suppress | 112 | 73 | 39 | 0.005213 |
| v17 source route | 146 | 107 | 39 | 0.006812 |

v9 的关键特征不是“改得更大”，而是“改得更密、更连续、更均衡”：三类 expert 都有正负 delta，且总体 mean delta 接近 0。

## What v13/v16/v17 Lose From v9

Reference = v9。

| candidate | v9 changed | lost | sign mismatch | big gap |
|---|---:|---:|---:|---:|
| v13 | 205 | 145 | 1 | 3 |
| v16 | 205 | 124 | 1 | 42 |
| v17 | 205 | 110 | 11 | 52 |

丢失最多的 role：

| role | v9 changed | v13 lost | v16 lost | v17 lost |
|---|---:|---:|---:|---:|
| code_source_conflict | 50 | 50 | 38 | 30 |
| code_source_conflict_with_behavior | 33 | 33 | 24 | 18 |
| uninformative | 22 | 22 | 22 | 22 |
| code_negative_noise | 16 | 16 | 16 | 16 |
| code_repair_vs_protected_harm | 13 | 10 | 10 | 10 |
| code_repair_shared_and_harm | 8 | 7 | 7 | 7 |

## Key Diagnosis

### 1. v9 的 Code 能力不只来自 clean repair rows

v13/v16/v17 都覆盖了 `code_repair_only`：

```text
code_repair_only: v9 changed 49, v13/v16/v17 changed 60
```

但它们 Code 仍然弱。这说明 clean positive repair rows 不是充分条件。

### 2. v9 大量使用“低置信连续 delta”

v9 改了很多 atlas 里被离散规则判成 ambiguous 的 row：

- `uninformative`: 22 rows
- `code_negative_noise`: 16 rows
- `protected_harm_only`: 3 rows
- `protected_support_only`: 2 rows

这些 row 单独看不适合 hard routing，但作为连续 evidence field 的一部分可能提供小的 steering signal。

### 3. v16/v17 的 source-conflict rule 太粗

v16/v17 能恢复部分 source-conflict 操作，但仍丢掉很多 v9 delta：

- v16 仍丢 `38 + 24 = 62` 个 source-conflict delta。
- v17 仍丢 `30 + 18 = 48` 个 source-conflict delta，并引入 11 个 sign mismatch。

这解释了评测现象：

- Tool / Memory 稳：离散 rule 保守，behavior-safe。
- Code 不足：Code 需要更连续、更细粒度的 residual steering。

### 4. v9 的可泛化原则

当前最合理的抽象不是：

```text
atlas role -> hard action
```

而是：

```text
continuous Code pass/fail evidence field
+ behavior utility/harm constraints
+ audit table解释哪些 residual 被约束
```

这比 hard role routing 更接近第一性原理：能力是分布式残差场，不是少数离散标签。

## Method Implication

下一版主方法应该命名和描述为：

```text
Residual Capability Field with Behavior Constraints
```

核心步骤：

1. 用 same-prompt pass/fail contrast 构建连续 Code capability field。
2. 用 Tool tool-call span 构建 hard behavior constraint。
3. 用 Memory full trajectory span 构建 soft behavior constraint。
4. 用 conservative aggregation + bounded delta 生成 gate。
5. 用 atlas 只做可审查解释和 ablation，不把 atlas role 作为唯一决策器。

## Files

- per-row comparison: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/operating_point_rows.jsonl`
- role aggregation: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/delta_by_role.csv`
- source-pattern loss: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/reference_lost_by_source_pattern.csv`
- markdown summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/operating_point_comparison.md`
