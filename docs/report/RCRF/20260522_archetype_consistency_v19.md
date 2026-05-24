# 2026-05-22 RCF-BC v19: Archetype Consistency Projection

## Purpose

`v19_archetype_consistency` 是 `v18_rcf_bc` 的机制消融，不是新一轮调参。

它回答一个具体问题：

> 如果 residual row 已经被归入机制簇，那么 gate delta 是否应该满足最基本的簇语义一致性？

例如：

- `code_negative_noise` 不应被抬高；
- `code_negative_with_behavior_support` 不应被压低；
- `weak_or_uninformative` 不应因为 expert-mean recenter 产生漂移；
- `behavior_only` 不应被 Code field 直接推动；
- `code_source_conflict` 仍保留连续小幅 delta，不回到 hard routing。

## Reproduction

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

PHASE=generate CANDIDATES=v19_archetype_consistency \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

Output:

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/gates.json
```

Summary:

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/archetype_policy_summary.md
```

## Rule

Start from `v18_rcf_bc` and project inconsistent deltas to base:

| archetype | rule |
|---|---|
| `clean_code_repair` | keep positive delta; reset negative delta |
| `code_repair_with_behavior_harm` | keep soft-constrained positive delta; reset negative delta |
| `code_source_conflict` | keep continuous delta |
| `code_negative_noise` | keep negative suppression; reset positive delta |
| `code_negative_with_behavior_support` | reset any delta |
| `behavior_only` | reset drift |
| `weak_or_uninformative` | reset drift |

No expert mean recenter is applied after projection, because recentering is exactly the source of several no-evidence drifts.

## Delta Summary

| metric | v18 source | v19 projected |
|---|---:|---:|
| changed rows | 205 | 173 |
| positive rows | 106 | 91 |
| negative rows | 99 | 82 |
| mean abs delta | 0.002639 | 0.002365 |
| max abs delta | 0.029167 | 0.028125 |

Reset rows:

| reason | count |
|---|---:|
| `reset_weak_or_uninformative_drift` | 22 |
| `reset_behavior_only_drift` | 5 |
| `reset_negative_delta_on_behavior_support` | 2 |
| `reset_negative_noise_positive_delta` | 1 |
| `reset_clean_repair_negative_delta` | 1 |
| `reset_repair_harm_negative_delta` | 1 |

## Interpretation

v19 is a stricter mechanism baseline:

- It keeps v18's central insight: continuous `code_source_conflict` deltas are allowed.
- It removes drift that cannot be justified by residual evidence.
- It protects behavior-supporting residuals from being suppressed by Code-negative evidence.

This should be evaluated as:

```text
v18_rcf_bc vs v19_archetype_consistency
```

not as:

```text
threshold sweep
```

Expected diagnostic value:

- If v19 keeps Tool/Memory and only mildly changes Code, archetype consistency is useful.
- If v19 hurts Code substantially, some apparently weak/uninformative recenter drift was actually contributing to Code; this becomes evidence that capability field needs low-confidence continuous components.
- If v19 improves Tool/Memory without hurting Code, it is a cleaner paper-facing method than v18.

## Evaluation

The v19 checkpoint was baked and evaluated on 2026-05-22:

```text
/tmp/shared-storage/OnPolicy/checkpoints/rcrf_archetype_consistency_v19
```

### Tool + Memory Quick

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Memory eval_50 F1 |
|---|---:|---:|---:|---:|---:|
| `v18_rcf_bc` | 0.880 | 0.855 | 0.8125 | 0.625 | 0.7575 |
| `v19_archetype_consistency` | 0.880 | 0.865 | 0.8125 | 0.625 | 0.7793 |

Raw files:

```text
/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_archetype_consistency_v19/quick_tool_memory/logs/tool_bfcl.log
/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/rcrf-memory/rcrf_archetype_consistency_v19/quick_tool_memory/eval_50/evaluation_summary.json
```

### Code Hurt Subsets

| model | LiveBench hurt acc / BoN | LiveCodeBench hurt acc / BoN |
|---|---:|---:|
| `v18_rcf_bc` | 0.2500 / 0.6250 | 0.6250 / 0.6555 |
| `v19_archetype_consistency` | 0.1250 / 0.1875 | 0.3281 / 0.4375 |

Raw files:

```text
/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_archetype_consistency_v19-LiveBenchCodeHurtRcrfVsTa16.txt
/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_archetype_consistency_v19-LiveCodeBenchCodeHurtRcrfVsTa16.txt
```

## Conclusion

v19 is **not** the main method.

It improves behavior-preservation metrics:

- Tool stays at the v18 level and parallel_multiple improves from `0.855` to `0.865`.
- Memory eval_50 F1 improves from `0.7575` to `0.7793`.

But it substantially damages Code hurt recovery:

- LiveBench hurt BoN drops from `0.6250` to `0.1875`.
- LiveCodeBench hurt acc drops from `0.6250` to `0.3281`.

The negative result is important:

> Some low-confidence or apparently uninformative continuous deltas are necessary for Code repair.

Therefore the paper-facing method should keep the v18 principle:

```text
continuous residual capability field
+ behavior constraints
```

and should not collapse into:

```text
strict archetype projection
```

v19 should be reported as an ablation showing that excessive semantic cleanup improves behavior preservation but removes distributed Code capability signal.

## Files

- Script: `scripts/analysis/build_rcrf_archetype_policy_gates.py`
- Gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/gates.json`
- Decision rows: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/decision_rows.jsonl`
- Summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/archetype_policy_summary.md`
