# 2026-05-22 Counterfactual Effect Memory

## Why This Matters

RCF-BC 之前的 row label 是基于 trajectory contrast / behavior span 的 proxy evidence。v20-v23 证明：proxy label 不能直接当成可剪规则。必须把 row intervention 的真实评测变化也纳入归因框架。

## New Artifact

脚本：

```text
scripts/analysis/build_rcrf_counterfactual_effect_table.py
```

报告：

```text
docs/report/RCRF/20260522_counterfactual_residual_effects.md
```

它读取：

- `rcrf_paper_evidence_table.csv`
- v14/v15 的 `expert_scale_summary.json`
- v20-v23 的 `archetype_scaled_summary.json`

输出：

- 每个 intervention 相对 `v18_rcf_bc` 的指标 delta。
- 每 10 个直接改动 rows 的归一化影响。
- `v20 - (v22 + v23)` 的非加性项。

## Key Numbers

| candidate | direct rows | dTool live | dMemory F1 | dLB BoN | dLCB BoN |
|---|---:|---:|---:|---:|---:|
| `v20` | 60 | 0.0000 | +0.0197 | -0.1250 | -0.1250 |
| `v22` | 15 | -0.0625 | -0.0075 | +0.0625 | -0.1875 |
| `v23` | 45 | -0.0625 | -0.0134 | -0.0625 | -0.0625 |

Non-additivity:

```text
v20 - (v22 + v23)
dTool live = +0.1250
dMemory F1 = +0.0407
dLB BoN = -0.1250
dLCB BoN = +0.1250
```

## Interpretation

1. v20 能提升 Memory，不代表 `code_negative_noise` 或 `weak_or_uninformative` 任一组是安全噪声。
2. v22/v23 单独剪都伤 Tool live 和 Memory，说明这些 rows 是 capability/behavior coupling points。
3. v20 的效果存在强非加性，说明 residual rows 的组合不是独立线性加和。
4. 论文方法应强调 continuous residual field + behavior constraints + counterfactual audit，而不是 hard pruning。

## Next Method Implication

下一版方法应当：

- 保持 continuous residual field。
- 对低证据/负证据 row 增加 behavior-support audit。
- 用 counterfactual effect table 作为方法消融的一部分，证明 row-level attribution 的必要性和局限。
- 避免写成“找到噪声 row 然后删除”的简单剪枝故事。

