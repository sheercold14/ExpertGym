# 2026-05-22 Residual-Level Code Scale Ablation

## Motivation

全局实验已经证明：

- 把全部 `code` expert 系数减半会释放 Memory，但 LiveCodeBench 和 BoN 明显掉。
- 把全部 `code` expert 系数置零会进一步提高 Memory，但 Code 基本崩。

因此问题不是“code task vector 整体太大”，而是：

> code residual 中同时存在能力 row、噪声 row、以及和 Tool/Memory 行为冲突的 row。合理单位应是 residual row，而不是 task-level scalar。

## New Diagnostic

新增通用脚本：

```text
scripts/analysis/build_rcrf_archetype_scaled_gate_ablation.py
```

它从已有 RCF-BC gate 和 conflict-cluster rows 出发，只按 `archetype + expert` 过滤 residual row，然后机械修改对应 coefficient。该脚本不重新打分、不训练、不改 reward。

## Variants

本轮只验证最小假设：

> 如果“code 系数太大”是局部噪声造成的，那么只压低 `code_negative_noise` 和 `weak_or_uninformative` 中的 code rows，应该比全局压低 code 更安全。

| variant | source | selected rows | operation | hypothesis |
|---|---|---:|---|---|
| `v20_code_noise_half` | `v18_rcf_bc` | 60 / 588 | selected code coefficients `* 0.5` | 局部压低噪声，应尽量保留 Code |
| `v21_code_noise_zero` | `v18_rcf_bc` | 60 / 588 | selected code coefficients `= 0` | 强负例，测试局部置零是否仍比全局置零安全 |
| `v22_code_negative_noise_half` | `v18_rcf_bc` | 15 / 588 | only `code_negative_noise` code coefficients `* 0.5` | 检验真负向 code evidence 是否是安全剪枝目标 |
| `v23_code_weak_half` | `v18_rcf_bc` | 45 / 588 | only `weak_or_uninformative` code coefficients `* 0.5` | 检验 LiveBench 掉分是否来自 weak rows 被误剪 |

选择条件：

```text
expert = code
archetype in {code_negative_noise, weak_or_uninformative}
```

未触碰：

- `clean_code_repair`
- `code_repair_with_behavior_harm`
- `code_source_conflict`
- 所有 `tool` / `memory` rows

## Commands

```bash
PHASE=generate CANDIDATES=v20_code_noise_half,v21_code_noise_zero \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh

PHASE=bake CANDIDATES=v20_code_noise_half,v21_code_noise_zero \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh

PHASE=generate CANDIDATES=v22_code_negative_noise_half,v23_code_weak_half \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh

PHASE=bake CANDIDATES=v22_code_negative_noise_half,v23_code_weak_half \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

后续快评：

```bash
PHASE=quick_eval CANDIDATES=v20_code_noise_half,v21_code_noise_zero \
  TOOL_GPU=0 TOOL_PORT=8160 MEMORY_GPU_IDS=1 MEMORY_DATASETS=eval_50 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

如 Tool/Memory 不崩，再跑 Code hurt subset：

```bash
PHASE=code_hurt_eval CANDIDATES=v20_code_noise_half,v21_code_noise_zero CODE_GPU=2 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

## Generated Gate Summary

| variant | selected rows | mean before | mean after | mean delta |
|---|---:|---:|---:|---:|
| `v20_code_noise_half` | 60 | 0.809282 | 0.404641 | -0.404641 |
| `v21_code_noise_zero` | 60 | 0.809282 | 0.000000 | -0.809282 |
| `v22_code_negative_noise_half` | 15 | 0.845879 | 0.422939 | -0.422939 |
| `v23_code_weak_half` | 45 | 0.797083 | 0.398541 | -0.398541 |

按 archetype：

| variant | archetype | rows | mean before | mean after |
|---|---|---:|---:|---:|
| `v20` | `code_negative_noise` | 15 | 0.845879 | 0.422939 |
| `v20` | `weak_or_uninformative` | 45 | 0.797083 | 0.398541 |
| `v21` | `code_negative_noise` | 15 | 0.845879 | 0.000000 |
| `v21` | `weak_or_uninformative` | 45 | 0.797083 | 0.000000 |
| `v22` | `code_negative_noise` | 15 | 0.845879 | 0.422939 |
| `v23` | `weak_or_uninformative` | 45 | 0.797083 | 0.398541 |

## Evaluation Results

| model | changed code rows | Tool mean | Memory F1 | LB hurt code_acc / BoN | LCB hurt code_acc / BoN |
|---|---:|---:|---:|---:|---:|
| `v18_rcf_bc / v9` | 0 | 0.7931 | 0.7575 | 0.1406 / 0.2500 | 0.3281 / 0.6250 |
| `v14_global_code_half` | 196 | 0.7944 | 0.7774 | 0.2031 / 0.2500 | 0.2500 / 0.1875 |
| `v15_global_code_zero` | 196 | 0.7800 | 0.7841 | 0.0781 / 0.1250 | 0.1719 / 0.1875 |
| `v20_local_noise_half` | 60 | 0.7944 | 0.7772 | 0.0781 / 0.1250 | 0.3125 / 0.5000 |
| `v21_local_noise_zero` | 60 | 0.7931 | 0.7664 | 0.1250 / 0.3750 | 0.2500 / 0.2500 |
| `v22_code_negative_noise_half` | 15 | 0.7788 | 0.7500 | 0.1406 / 0.3125 | 0.3125 / 0.4375 |
| `v23_code_weak_half` | 45 | 0.7788 | 0.7441 | 0.1406 / 0.1875 | 0.3594 / 0.5625 |

## Interpretation

这个实验不是为了找到最终 best checkpoint，而是验证 RCF-BC 的核心粒度。评测后结论更具体：

1. `v20` 只动 60 行就把 Memory F1 从 `0.7575` 提到 `0.7772`，几乎复现全局 `code_half` 的 Memory gain。这说明 Memory 受损可以被局部化，不需要整体压低 code expert。
2. `v20` 的 LiveCodeBench 比全局 `code_half` 明显好，尤其 BoN `0.5000 > 0.1875`，说明 residual-level shrink 比 task-level shrink 更保留一部分 Code 能力。
3. 但 `v20` 的 LiveBench hurt 掉到 `0.0781 / 0.1250`，说明 `weak_or_uninformative` 并不等价于无用；里面有当前 contrast evidence 没捕获到的 LiveBench 能力 row。
4. `v21` 置零没有带来更强 Memory，且 Code 仍受伤，说明强 pruning 不是正确方向。
5. `v22/v23` 拆开后都让 Tool live_parallel 从 `0.8125` 掉到 `0.7500`，Memory 也低于 v18。这说明这两类 rows 单独看都不是安全剪枝目标。
6. `v23` 的 LiveCodeBench 反而最高，`0.3594 / 0.5625`，证明 `weak_or_uninformative` 包含重要 Code 能力，只是当前 evidence 没有把它归因出来。
7. `v20` 的 Memory gain 更像两类 row 同时缩放后的非线性交互，而不是某个 archetype 本身可剪。

因此，当前最有价值的发现是：

> RCF-BC 的 row-level attribution 已经能定位能力/行为冲突区域，但“负向/弱证据”不能直接等价为噪声。下一步应把 behavior support 作为独立约束加入这些 rows，并增强 Code evidence source，而不是继续加大 pruning。

## Artifacts

| variant | gate | summary |
|---|---|---|
| `v20` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_noise_weak_half_v20/gates.json` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_noise_weak_half_v20/archetype_scaled_summary.md` |
| `v21` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_noise_weak_zero_v21/gates.json` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_noise_weak_zero_v21/archetype_scaled_summary.md` |
