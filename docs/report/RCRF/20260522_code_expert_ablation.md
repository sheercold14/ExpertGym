# 2026-05-22 Code Expert Coefficient Ablation

## Question

当前 v9 是 RCRF 最均衡点，但 Code / Memory 之间仍有冲突。一个直接怀疑是：

> Code expert 系数整体太大，压制了 Memory / Tool；如果把 Code 系数调小或置零，整体能力是否更好？

为了避免引入新规则，本实验只从 v9 gate 做机械消融，不重新计算 residual evidence，不改 reward / evaluation / bake 逻辑。

## Artifacts

| variant | gate | checkpoint | code expert mean |
|---|---|---|---:|
| v9 baseline | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/gates.json` | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9` | 0.9007 |
| v14 code_half | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_v9_code_half_v14/gates.json` | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_half_v14` | 0.4504 |
| v15 code_zero | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_v9_code_zero_v15/gates.json` | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_zero_v15` | 0.0000 |

## Commands

```bash
PHASE=generate CANDIDATES=v14_code_half,v15_code_zero \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh

PHASE=bake CANDIDATES=v14_code_half,v15_code_zero \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh

PHASE=quick_eval CANDIDATES=v14_code_half,v15_code_zero \
  TOOL_GPU=0 TOOL_PORT=8154 MEMORY_GPU_IDS=1 MEMORY_DATASETS=eval_50 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh

PHASE=code_hurt_eval CANDIDATES=v14_code_half,v15_code_zero CODE_GPU=2 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

## Results

| model | Tool parallel | Tool parallel_multi | Tool live_parallel | Tool live_parallel_multi | Memory eval_50 F1 | LiveBench hurt code_acc / BoN acc | LiveCodeBench hurt code_acc / BoN acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| v9 baseline | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7575 | 0.1406 / 0.2500 | 0.3281 / 0.6250 |
| v14 code_half | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7774 | 0.2031 / 0.2500 | 0.2500 / 0.1875 |
| v15 code_zero | 0.8800 | 0.8650 | 0.7500 | 0.6250 | 0.7841 | 0.0781 / 0.1250 | 0.1719 / 0.1875 |

## Interpretation

1. Global Code residual does suppress Memory: reducing code expert coefficients monotonically improves Memory eval_50 F1 from `0.7575 -> 0.7774 -> 0.7841`.
2. Tool is mostly insensitive to Code removal on non-live parallel categories, but `live_parallel` drops when Code is zeroed.
3. Code ability is highly sensitive to global Code residual scale. Halving Code slightly improves single-sample LiveBench hurt but sharply hurts LiveCodeBench and BoN; zeroing Code makes both Code subsets much weaker.
4. Therefore the issue is not simply “Code coefficient too large”. Code residual contains both necessary Code ability and cross-task interference. The next method should not globally shrink Code. It should identify which Code residual rows are behavior-harmful and which rows are Code-critical.

## Research Implication

This is useful negative evidence for the paper:

> Scalar task-vector shrinkage can improve one task by removing interference, but it destroys the same expert's recoverable capability. The right unit is residual-level routing, not task-level scalar suppression.

The ablation supports RCRF's core framing: the code expert is not a pure harmful direction; it is a mixture of capability and conflict.
