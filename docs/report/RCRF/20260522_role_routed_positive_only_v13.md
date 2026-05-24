# RCRF v13：Positive-Only Role Routing 负结果

## 1. 目的

v12 的结论是：

- Tool / Memory 能保住；
- Code 明显弱于 v9；
- 最可疑原因是 v12 把 `code_negative_noise` 全部 suppress，负向 Code contrast 过粗。

v13 因此只做一个隔离实验：

```text
保留 positive role routing
关闭 code_negative_noise suppression
关闭 protected_harm_only suppression
```

它验证的问题是：

> 如果不压任何 negative residual，只提高 atlas 认为正向的 residual，Code 是否能恢复，同时 Tool/Memory 是否仍稳定？

## 2. 复现命令

生成 gate：

```bash
PHASE=generate CANDIDATES=v13 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

bake：

```bash
PHASE=bake CANDIDATES=v13 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

Tool / Memory quick：

```bash
PHASE=quick_eval CANDIDATES=v13 TOOL_GPU=0 TOOL_PORT=8153 MEMORY_GPU_IDS=1 MEMORY_DATASETS=eval_50 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

Code hurt16：

```bash
PHASE=code_hurt_eval CANDIDATES=v13 CODE_GPU=2 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

## 3. Gate 规则

v13 仍使用 `scripts/analysis/build_rcrf_role_routed_gates.py`，但打开：

```bash
--variant-name rcrf_role_routed_positive_only_v13
--code-negative-action hold
--protected-harm-action hold
```

规则：

| role | v13 操作 |
|---|---|
| `code_repair_only` | raise |
| `shared_positive` | raise |
| `code_repair_vs_protected_harm` | Tool harm 不 raise；Memory harm soft raise |
| `code_repair_shared_and_harm` | 同上 |
| `code_negative_noise` | hold |
| `protected_harm_only` | hold |
| `code_negative_but_protected_support` | hold |
| `code_source_conflict*` | hold |

## 4. Gate 统计

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_positive_only_v13/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_role_routed_positive_only_v13`
- summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_positive_only_v13/role_routing_summary.md`

总体：

| metric | value |
|---|---:|
| changed | `73/588` |
| positive delta | 73 |
| negative delta | 0 |
| mean abs delta | 0.002073 |

按 role：

| role | changed | + | - |
|---|---:|---:|---:|
| `code_repair_only` | 60 | 60 | 0 |
| `shared_positive` | 9 | 9 | 0 |
| `code_repair_vs_protected_harm` | 3 | 3 | 0 |
| `code_repair_shared_and_harm` | 1 | 1 | 0 |
| `code_negative_noise` | 0 | 0 | 0 |
| `protected_harm_only` | 0 | 0 | 0 |
| `code_source_conflict*` | 0 | 0 | 0 |

按 expert：

| expert | changed | mean_delta |
|---|---:|---:|
| code | 43 | 0.003986 |
| memory | 7 | 0.000581 |
| tool | 23 | 0.001650 |

结构均值：

| expert | mean coefficient |
|---|---:|
| code | 0.9047 |
| memory | 0.9880 |
| tool | 1.0058 |

## 5. 评测结果

Tool / Memory：

| metric | v13 |
|---|---:|
| Tool parallel | 0.8800 |
| Tool parallel_multiple | 0.8550 |
| Tool live_parallel | 0.8125 |
| Tool live_parallel_multiple | 0.6250 |
| Memory eval_50 F1 | 0.7564 |

Code hurt16：

| dataset | BoN acc | BoN accum |
|---|---:|---:|
| LiveBenchCodeHurtRcrfVsTa16 | 0.1250 | 0.2422 |
| LiveCodeBenchCodeHurtRcrfVsTa16 | 0.3125 | 0.5294 |

对比：

| candidate | Tool quick | Memory eval_50 F1 | LiveBench hurt BoN | LiveCodeBench hurt BoN |
|---|---|---:|---|---|
| v9 soft | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v12 role-routed | `0.880 / 0.865 / 0.8125 / 0.625` | 0.7627 | `0.2500 / 0.4453` | `0.4375 / 0.5294` |
| v13 positive-only | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7564 | `0.1250 / 0.2422` | `0.3125 / 0.5294` |

## 6. 结论

v13 是清晰负结果：

1. Positive-only routing 可以保住 Tool / Memory。
2. Positive-only routing 不能恢复 Code，甚至 LiveBench hurt 明显弱于 v12。
3. 因此 v12 Code 差不只是因为 `code_negative_noise` suppression。
4. 更关键的问题是：当前 role-routing 把全部 `code_source_conflict*` hold 住，而 v9 的 Code gain 很可能来自其中一部分 source-conflict residual。

这给出下一步的第一性修正：

> Code source conflict 不是无信息噪声。它表示不同 Code span/source 需要不同 residual。RCRF 下一步必须从 role-level routing 进入 source-conditioned conflict routing。

## 7. 下一步

不要继续调 `max_delta`。应做 v14：

1. 保留 Tool / Memory behavior protection。
2. 保留 `code_repair_only` / `shared_positive` raise。
3. 对 `code_source_conflict*` 不再全部 hold。
4. 根据 source/span 类型拆分：
   - LiveBench prompt；
   - LiveBench reasoning；
   - LiveCodeBench code；
   - LiveCodeBench prompt。
5. 只有当某个 source family 与目标评测一致，并且正向强于负向时，才 soft raise。

换句话说，v13 证明 RCRF 的下一层归因单位应该从：

```text
(param_name, expert, role)
```

升级为：

```text
(param_name, expert, role, source/span)
```
