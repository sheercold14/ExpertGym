# RCRF v12：基于 Residual Role 的可解释 Gate Routing

## 1. 目的

v8-v11 证明 RCRF 可以暴露 Pareto frontier，但它们仍然直接从 contrast / harm 规则生成 gate。v12 的目标是把流程再拆清楚：

```text
机制证据 -> residual conflict atlas -> residual role -> gate routing
```

这让方法更接近“能力归因框架”，而不是一组 gate 生成技巧。

## 2. 复现命令

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/build_rcrf_role_routed_gates.py \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12
```

或：

```bash
PHASE=role_route bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/gates.json`
- role summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/role_routing_summary.md`
- structure summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/gate_structure_summary.md`

## 3. Routing 原则

v12 使用固定 role 规则，不根据评测结果调参：

| atlas role | 操作 |
|---|---|
| `code_repair_only` | 提高 |
| `shared_positive` | 提高 |
| `code_repair_vs_protected_harm` | 如果 harm 来自 Tool，不提高；如果只来自 Memory，软提高 |
| `code_repair_shared_and_harm` | 同上，但更保守 |
| `code_negative_noise` | 压低 |
| `protected_harm_only` | 压低 |
| `code_negative_but_protected_support` | 保持，不因为 Code 负向而伤 Tool/Memory |
| `code_source_conflict*` | 保持，等待更细 span/source 证据 |
| `protected_support_only` | 保持 |
| `uninformative` | 保持 |

步幅不手设到某个结果，而是用 atlas 中 role strength 的 `0.9 quantile` 做归一：

```text
delta = max_delta * min(strength / q90_strength, 1)
```

默认：

```text
max_delta = 0.05
tool_harm_positive_scale = 0.0
memory_harm_positive_scale = 0.5
mixed_harm_positive_scale = 0.25
```

这对应当前机制假设：

- Tool call behavior 更像 hard constraint；
- Memory full trajectory 与 Code reasoning 有共享，因此是 soft constraint；
- source conflict 不应被 scalar gate 猜测处理。

## 4. v12 Gate 统计

总计：

| metric | value |
|---|---:|
| changed | `133/588` |
| positive delta | 73 |
| negative delta | 60 |
| mean abs delta | 0.004696 |

按 role：

| role | changed | + | - | mean_abs_delta |
|---|---:|---:|---:|---:|
| `code_repair_only` | 60 | 60 | 0 | 0.017877 |
| `shared_positive` | 9 | 9 | 0 | 0.006064 |
| `code_repair_vs_protected_harm` | 3 | 3 | 0 | 0.002131 |
| `code_repair_shared_and_harm` | 1 | 1 | 0 | 0.000741 |
| `code_negative_noise` | 56 | 0 | 56 | 0.026091 |
| `protected_harm_only` | 4 | 0 | 4 | 0.020405 |
| `code_negative_but_protected_support` | 0 | 0 | 0 | 0 |
| `code_source_conflict*` | 0 | 0 | 0 | 0 |

按 expert：

| expert | changed | + | - | mean_delta | mean_abs_delta |
|---|---:|---:|---:|---:|---:|
| code | 58 | 43 | 15 | 0.001875 | 0.006097 |
| memory | 39 | 7 | 32 | -0.004241 | 0.005403 |
| tool | 36 | 23 | 13 | 0.000713 | 0.002588 |

按 layer band：

| layer band | changed | + | - | mean_delta |
|---|---:|---:|---:|---:|
| early `0-9` | 46 | 24 | 22 | -0.001258 |
| middle `10-19` | 41 | 16 | 25 | -0.002009 |
| late `20-27` | 46 | 33 | 13 | 0.002154 |

结构摘要：

| expert | mean coefficient | min | max |
|---|---:|---:|---:|
| code | 0.9026 | 0.6309 | 1.1200 |
| memory | 0.9831 | 0.6770 | 1.1200 |
| tool | 1.0049 | 0.6651 | 1.1200 |

重要解释：

- v12 不是简单压低 Code。它压低的是 `code_negative_noise`，同时提高 late-layer Code repair residual。
- v12 也不是简单保 Tool/Memory。它允许 `shared_positive` 和无 Tool harm 的 Code repair 提高。
- `code_negative_but_protected_support` 全部保持，避免为了 Code 修复破坏 Tool/Memory behavior。

## 5. 和 v8-v11 的关系

| candidate | 核心差异 |
|---|---|
| v8 | hard harm veto，保护强但 Code repair 被压 |
| v9 | fixed soft harm scale，Code repair 最强但 Memory 稍弱 |
| v11 | task-typed harm scale，Tool/Memory 更稳但 Code 弱于 v9 |
| v12 | 先做 residual role attribution，再按 role routing；source conflict 不动，noise 显式压低 |

v12 的价值不在于还没有验证的最终指标，而在于它把 RCRF 方法从“公式组合”推进到“先归因、再路由”的清晰流程。

## 6. 下一步验证

已完成最小闭环评测。

### 6.1 Tool / Memory Quick Eval

命令：

```bash
CANDIDATES=v12 PHASE=quick_eval TOOL_GPU=0 TOOL_PORT=8152 MEMORY_GPU_IDS=1 MEMORY_DATASETS=eval_50 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- summary dir: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_role_routed_v12/quick_tool_memory`
- memory summary: `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/rcrf-memory/rcrf_role_routed_v12/quick_tool_memory/eval_50/evaluation_summary.json`

结果：

| metric | v12 |
|---|---:|
| Tool parallel | 0.8800 |
| Tool parallel_multiple | 0.8650 |
| Tool live_parallel | 0.8125 |
| Tool live_parallel_multiple | 0.6250 |
| Memory eval_50 F1 | 0.7627 |

对比：

| candidate | Tool parallel | parallel_multiple | live_parallel | live_parallel_multiple | Memory eval_50 F1 |
|---|---:|---:|---:|---:|---:|
| v9 soft | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7575 |
| v11 task-typed | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7701 |
| v12 role-routed | 0.8800 | 0.8650 | 0.8125 | 0.6250 | 0.7627 |

结论：

- v12 没有破坏 Tool。
- Memory 介于 v9 和 v11 之间。
- 因此 role-routing 至少能保持 protected behavior，是有效的 behavior-preserving operating point。

### 6.2 Code Hurt16 Eval

命令：

```bash
CANDIDATES=v12 PHASE=code_hurt_eval CODE_GPU=2 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- `/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_role_routed_v12-LiveBenchCodeHurtRcrfVsTa16.txt`
- `/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_role_routed_v12-LiveCodeBenchCodeHurtRcrfVsTa16.txt`

结果：

| dataset | BoN acc | BoN accum |
|---|---:|---:|
| LiveBenchCodeHurtRcrfVsTa16 | 0.2500 | 0.4453 |
| LiveCodeBenchCodeHurtRcrfVsTa16 | 0.4375 | 0.5294 |

对比：

| candidate | LiveBench BoN acc | LiveBench BoN accum | LiveCodeBench BoN acc | LiveCodeBench BoN accum |
|---|---:|---:|---:|---:|
| v9 soft | 0.2500 | 0.6250 | 0.6250 | 0.6555 |
| v11 task-typed | 0.2500 | 0.5469 | 0.3750 | 0.5126 |
| v12 role-routed | 0.2500 | 0.4453 | 0.4375 | 0.5294 |

结论：

- v12 的 Code 明显弱于 v9。
- v12 略优于 v11 的 LiveCodeBench acc，但 accum 仍低，不能作为 balanced best point。
- 机制解释：v12 压低了全部 `code_negative_noise`，这在小 hurt subset 上看似合理，但对 Code 泛化过强；负向 Code contrast 不能直接当 pruning 信号。

## 7. 修正后的下一步

v12 给出明确负结果后，下一版不应继续调 `max_delta`，而应改 routing 逻辑：

1. 保留 positive routing：`code_repair_only` / `shared_positive` 可提高。
2. 不再默认 suppress `code_negative_noise`；负向 contrast 必须按 source/span 细分。
3. `code_source_conflict*` 不能全 hold，也不能全动；应拆成 prompt / reasoning / final-code 哪个 source 在冲突。
4. Tool harm 仍保持 hard 或 near-hard；这个结论被 v12 再次支持。
5. Memory harm 继续 soft；v12 Memory 证明 soft behavior protection 可行。

因此 v12 的论文价值是一个清晰负结果：

> Positive residual role routing preserves Tool/Memory, but naive suppression of Code-negative residuals hurts Code repair. This shows that negative Code contrast is less reliable than positive pass/fail repair evidence and must be span-conditioned.
