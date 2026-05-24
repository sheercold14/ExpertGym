# 2026-05-22 Code Shrink Residual Ablation Memory

## Question

怀疑：Code 能力不稳是否因为 `code` expert coefficient 太大？

## Global Result

全局压低 code 已经验证：

| model | change | Memory F1 | LB hurt code_acc / BoN | LCB hurt code_acc / BoN |
|---|---|---:|---:|---:|
| `v18_rcf_bc / v9` | none | 0.7575 | 0.1406 / 0.2500 | 0.3281 / 0.6250 |
| `v14` | all code coeff `*0.5` | 0.7774 | 0.2031 / 0.2500 | 0.2500 / 0.1875 |
| `v15` | all code coeff `=0` | 0.7841 | 0.0781 / 0.1250 | 0.1719 / 0.1875 |

结论：全局 code shrink 能释放 Memory，但会伤 Code。Code residual 不是纯噪声。

## Residual-Level Test

新增脚本：

```text
scripts/analysis/build_rcrf_archetype_scaled_gate_ablation.py
```

新增候选：

| model | direct rewrite | selected rows |
|---|---|---:|
| `v20_code_noise_half` | `expert=code` 且 `archetype in {code_negative_noise, weak_or_uninformative}` 的 coefficient `*0.5` | 60 |
| `v21_code_noise_zero` | 同一批 rows coefficient `=0` | 60 |

结果：

| model | Tool mean | Memory F1 | LB hurt code_acc / BoN | LCB hurt code_acc / BoN |
|---|---:|---:|---:|---:|
| `v20` | 0.7944 | 0.7772 | 0.0781 / 0.1250 | 0.3125 / 0.5000 |
| `v21` | 0.7931 | 0.7664 | 0.1250 / 0.3750 | 0.2500 / 0.2500 |
| `v22` | 0.7788 | 0.7500 | 0.1406 / 0.3125 | 0.3125 / 0.4375 |
| `v23` | 0.7788 | 0.7441 | 0.1406 / 0.1875 | 0.3594 / 0.5625 |

## Insight

1. Memory gain 可以局部化：v20 只直接改 60 行，就几乎复现 v14 的 Memory gain。
2. 局部化比全局 shrink 更保留 LiveCodeBench，尤其 v20 的 LCB BoN 0.5000 远高于 v14 的 0.1875。
3. 但 LiveBench hurt 明显掉，说明 `weak_or_uninformative` 不是无用 row；它包含当前证据没有覆盖的 Code 能力。
4. `v21` 置零没有进一步改善 Memory，且仍伤 Code，说明强 pruning 不是方向。
5. v22 只剪 `code_negative_noise` 的 15 行，Tool live_parallel 和 Memory 都掉，但 Code 没明显掉。这类 row 更像行为支撑 row，而不是 code 噪声。
6. v23 只剪 `weak_or_uninformative` 的 45 行，LiveCodeBench 反而强，但 Tool/Memory 掉。这说明弱证据 row 包含真实 Code 能力，也包含行为冲突。
7. v20 的 Memory gain 不是单个 archetype 安全可剪，而是两类 row 同时缩放后的交互效应；论文方法不能写成 hard pruning。

## Next Principle

不要继续用全局 task scalar，也不要把 low-evidence row 直接视为噪声。下一步应该增强 Code evidence：

- 区分 LiveBench-style prompt/reasoning 能力和 LiveCodeBench-style code generation 能力。
- 对 `weak_or_uninformative` 做二次诊断，而不是直接 prune。
- 保留 continuous residual field，用行为证据做约束，不做硬 routing。
- 对所有“负向/弱证据”row 加入 Tool/Memory behavior-support 检查，避免把行为支撑行误判为噪声。
