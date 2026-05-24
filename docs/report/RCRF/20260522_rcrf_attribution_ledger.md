# 2026-05-22 RCF-BC Attribution Ledger

## 目的

这是 RCF-BC 框架的逐 residual row 审计表。它把 residual 机制证据、当前系数和 counterfactual group effect 合并到同一张表里，让 row label 保持为可验证假设，而不是未经验证的因果结论。

## 决策汇总

| decision | rows | changed | mean code+ | mean code- | mean behavior harm |
|---|---:|---:|---:|---:|---:|
| `audit_before_prune` | 60 | 12 | 0.0000 | 0.2156 | 0.0000 |
| `behavior_constraint` | 31 | 11 | 0.0000 | 0.8737 | 1.7229 |
| `hold` | 29 | 0 | 0.0000 | 0.1300 | 0.0796 |
| `keep_capability_delta` | 77 | 56 | 0.9086 | 0.0000 | 0.0614 |
| `keep_continuous_field` | 279 | 83 | 2.3278 | 2.5762 | 0.9572 |
| `low_confidence_keep_small` | 18 | 18 | 0.0000 | 0.1288 | 0.0781 |
| `protect_behavior_support` | 66 | 4 | 0.0000 | 0.9865 | 1.0172 |
| `soft_constrained_capability` | 28 | 21 | 1.6100 | 0.0000 | 1.7423 |

## 可执行动作

| action | rows | changed | mean code+ | mean code- | mean behavior harm |
|---|---:|---:|---:|---:|---:|
| `behavior_guard` | 31 | 11 | 0.0000 | 0.8737 | 1.7229 |
| `do_not_prune_without_counterfactual` | 60 | 12 | 0.0000 | 0.2156 | 0.0000 |
| `hold_base` | 29 | 0 | 0.0000 | 0.1300 | 0.0796 |
| `keep_small_until_validated` | 18 | 18 | 0.0000 | 0.1288 | 0.0781 |
| `protect_behavior_anchor` | 66 | 4 | 0.0000 | 0.9865 | 1.0172 |
| `retain_capability_delta` | 77 | 56 | 0.9086 | 0.0000 | 0.0614 |
| `retain_continuous_field` | 279 | 83 | 2.3278 | 2.5762 | 0.9572 |
| `retain_with_behavior_constraint` | 28 | 21 | 1.6100 | 0.0000 | 1.7423 |

## 验证优先级

| priority | rows | changed | mean code+ | mean code- | mean behavior harm |
|---|---:|---:|---:|---:|---:|
| `high` | 172 | 91 | 1.4290 | 1.5853 | 0.9301 |
| `low` | 29 | 0 | 0.0000 | 0.1300 | 0.0796 |
| `medium` | 387 | 114 | 1.3403 | 1.4304 | 0.7301 |

## 反事实分组

| group | rows | changed | mean code+ | mean code- | mean behavior harm |
|---|---:|---:|---:|---:|---:|
| `v22` | 15 | 3 | 0.0000 | 0.8623 | 0.0000 |
| `v23` | 45 | 9 | 0.0000 | 0.0000 | 0.0000 |

## 高风险行

| key | archetype | decision | group | dTool live | dMemory | dLB BoN | dLCB BoN |
|---|---|---|---|---:|---:|---:|---:|
| `model.layers.3.mlp.gate_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.4.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.5.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.5.self_attn.q_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.5.self_attn.v_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.6.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.6.self_attn.q_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.6.self_attn.v_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.7.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.7.self_attn.q_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.8.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.8.self_attn.q_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.8.self_attn.v_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.9.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.9.self_attn.q_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.10.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.10.self_attn.q_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.10.self_attn.v_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.11.self_attn.k_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |
| `model.layers.11.self_attn.q_proj.weight::code` | `weak_or_uninformative` | `audit_before_prune` | `v23` | -0.0625 | -0.0134 | -0.0625 | -0.0625 |

## 框架结论

- ledger 将 proxy row label 与 counterfactual effect 分开，避免把相关性标签当作因果结论。
- 当 code_negative_noise 或 weak_or_uninformative group shrink 会伤 Tool/Memory 时，这些 row 被标为 audit_before_prune，而不是直接剪掉。
- 当前证据支持 continuous residual field + behavior-support audit，而不是 hard pruning 或全局 task scalar suppression。

## 产物

- CSV: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_attribution_ledger_20260522/rcrf_attribution_ledger.csv`
- JSONL: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_attribution_ledger_20260522/rcrf_attribution_ledger.jsonl`
- Summary JSON: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_attribution_ledger_20260522/rcrf_attribution_ledger_summary.json`
- Cluster rows: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_clusters_20260522/conflict_cluster_rows.jsonl`
- Counterfactual effects: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_counterfactual_effects_20260522/counterfactual_effect_summary.json`
