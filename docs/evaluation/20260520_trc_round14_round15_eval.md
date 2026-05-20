# 20260520 TRC Round14 / Round15 Evaluation

## Scope

本页跟踪 Code BoN-to-Acc 方向的实验。核心问题不是继续单纯推高 code gate，而是验证：

1. R5A/R11B 的高 BoN 是否稳定；
2. R5A 配置是否可复现；
3. 对同一 prompt 加入 positive/negative contrast 后，能否把 BoN 转成 single-sample Acc；
4. Tool/Memory 是否仍能通过 quick gate。

## Training

| ID | run id | calibration | objective delta | selected epoch | code gate | memory gate | tool gate | status |
|---|---|---|---|---:|---:|---:|---:|---|
| R14B | `trc_r14b_mixed_train24_contrast8_e8_20260520` | Tool32/Memory32 + Code train24 + formal contrast8 | R5A-like directional loss + Code negative contrast weight 1.5 | 8 | 1.1598 | 0.9974 | 1.1506 | baked; Tool/Memory quick eval launched |
| R15A | `trc_r15a_r5a_repro_e12_20260520` | exact R5A v2 late3/codeproj bank | strict R5A reproduction | pending | pending | pending | pending | running |
| R15B | `trc_r15b_r5a_train24_contrast8_e12_20260520` | R5A Tool/Memory + Code train24 + formal contrast8 | R5A + Code negative contrast weight 1.5 | 12 | 1.2383 | 0.9983 | 1.2152 | baked; Tool/Memory quick eval launched |

## Quick Gate

| ID | checkpoint | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R14B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r14b_mixed_train24_contrast8_e8_20260520-selected` | 0.7944 | 0.8125 | 0.6250 | 0.8800 | 0.8600 | 0.7578 | 0.7608 | 0.7504 | 0.7753 | 0.7447 | strict reject by Memory; Code result is diagnostic |
| R15A | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | wait for bake |
| R15B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r15b_r5a_train24_contrast8_e12_20260520-selected` | 0.7788 | 0.7500 | 0.6250 | 0.8850 | 0.8550 | pending | pending | pending | pending | pending | Tool below strict gate; Memory still running |

## ToolRL all80

ToolRL `rlla_4k/test` all80 是 Tool 源分布 sanity check，不按 BFCL 子类平均，主要看 `success_rate / mean_reward / parseable / zero_call`。

| ID | run id | success rate | mean reward | exact rate | parseable | zero-call | status |
|---|---|---:|---:|---:|---:|---:|---|
| TA 0.75 ref | `ta_c075_global_toolrl_all80_20260518` | 0.6125 | 0.8310 | 0.5250 | 0.9000 | 0.1000 | historical reference |
| best-ever ref | `bestever_tame_cg_r1calib_toolrl_all80_20260518` | 0.6250 | 0.8381 | 0.5375 | 0.9000 | 0.1000 | historical reference |
| R14B | `trc_r14b_toolrl_all80_20260520` | 0.6375 | 0.8363 | 0.5375 | 0.9000 | 0.1000 | done |
| R15A | `trc_r15a_r5a_repro_toolrl_all80_20260520` | 0.6375 | 0.8363 | 0.5375 | 0.9000 | 0.1000 | done |
| R15B | `trc_r15b_contrast_toolrl_all80_20260520` | 0.6375 | 0.8352 | 0.5375 | 0.9000 | 0.1000 | done |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R5A repeat | `code_repeat_20260520_1350` | 0.3906 | 0.4688 | pending | pending | pending | pending | LiveBench done; LiveCodeBench running |
| R11B repeat | `code_repeat_20260520_1350` | pending | pending | pending | pending | pending | pending | running; stability check |
| R14B | `code_20260520_1408_r14b` | pending | pending | pending | pending | pending | pending | running on GPU0 |
| R15A | pending | pending | pending | pending | pending | pending | pending | run after bake/quick gate |
| R15B | pending | pending | pending | pending | pending | pending | pending | run after bake/quick gate |

## Current Notes

- R14B reached epoch 8 cleanly in about 13.3 min. Gate movement is healthy: code and tool rise to about 1.16/1.15, memory stays near 1.0.
- R14B Tool quick gate passed with mean about 0.7944, but Memory mean F1 is `0.7578`, just below the strict `0.76` gate. Its Code run is therefore kept as a contrast diagnostic, not a promoted candidate.
- BFCL 子类需要谨慎解释：`parallel`/`parallel_multiple` 各 200 题，1 题约 0.005；`live_parallel` 只有 16 题，1 题约 0.0625。因此 live 子类提升可能很显著，但也更容易受单题波动影响。ToolRL all80 已加入作为源 ToolRL 能力 sanity check。
- ToolRL all80 显示 R14B/R15A/R15B 都达到 `51/80 = 0.6375`，高于 TA0.75 历史参考 `0.6125`，也略高于 best-ever 参考 `0.6250`。这说明当前 Tool 源分布能力保持良好，BFCL 子类差异更像 heldout 分布/小样本问题。
- R15B BFCL mean is `0.7788`; it loses mainly on `live_parallel=0.7500` while ToolRL all80 remains strong. This suggests the Code contrast variant did not break source ToolRL, but it is not a promoted three-task candidate unless Memory/Tool thresholds are relaxed.
- R15A epoch 1-6 matches R5A trajectory closely, so strict R5A reproduction is currently valid.
- R15B has active Code contrast rows, but contrast loss is small. Treat it as a BoN-to-Acc diagnostic, not as proof that current contrast strength is enough.
- R5A/R11B repeats are necessary because both have high BoN. R5A repeat LiveBench Acc improved from the previous 0.3672 to 0.3906 while BoN changed from 0.4844 to 0.4688, so Code metrics have non-trivial stochastic variance; final decision should use full LiveBench+LiveCodeBench mean.
