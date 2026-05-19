# 2026-05-17 P0 Static Baselines Eval6

## Scope

本表记录 P0 静态 task-vector baseline 的正式评测进展，用于支撑论文地基：`TA-1/3`、`TA-0.75`、`init1`。原始产物仍保留在 harness 输出目录；本文件只记录可用于论文/报告的数值。

## Summary

| model | type | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ta-c033333-global-20260517 | TA-1/3 | 0.7848 | 0.6771 | 0.5195 | 0.6465 | 0.3409 | 0.4688 | 0.4173 | done |
| ta-c075-global-20260517 | TA-0.75 | 0.7850 | 0.6875 | 0.6289 | 0.7587 | 0.3494 | 0.4734 | 0.4173 | done |
| ta-init1-global-20260517 | init1 | 0.7631 | 0.6562 | 0.6387 | 0.7583 | 0.3394 | 0.4506 | 0.4115 | done |

## Tool / BFCL

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | note |
|---|---:|---:|---:|---:|---:|---|
| ta-c033333-global-20260517 | 0.9150 | 0.8700 | 0.6875 | 0.6667 | 0.7848 | done |
| ta-c075-global-20260517 | 0.9050 | 0.8600 | 0.7500 | 0.6250 | 0.7850 | first run invalid due 404/model-not-found; rerun `20260517_p0_ta075_tool_rerun01` is valid |
| ta-init1-global-20260517 | 0.8800 | 0.8600 | 0.6875 | 0.6250 | 0.7631 | done |

## Memory / HotpotQA

| model | eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 |
|---|---:|---:|---:|---:|---:|---:|
| ta-c033333-global-20260517 | 0.6876 | 0.6191 | 0.6281 | 0.6511 | 0.5195 | 0.6465 |
| ta-c075-global-20260517 | 0.7530 | 0.7551 | 0.7608 | 0.7659 | 0.6289 | 0.7587 |
| ta-init1-global-20260517 | 0.7659 | 0.7609 | 0.7381 | 0.7685 | 0.6387 | 0.7583 |

## Code / CURE

| model | run id | GPU group | status |
|---|---|---|---|
| ta-c033333-global-20260517 | `20260517_p0_ta13_code_rerun01` | `[[0,7]]` | ACC 0.3409 / TP 0.4688 / BoN 0.4173 |
| ta-c075-global-20260517 | `20260517_p0_ta075_code` | `[[1,2]]` | ACC 0.3494 / TP 0.4734 / BoN 0.4173 |
| ta-init1-global-20260517 | `20260517_p0_init1_code` | `[[3,5]]` | ACC 0.3394 / TP 0.4506 / BoN 0.4115 |

## Raw Outputs

| item | path |
|---|---|
| TA-1/3 Tool/Memory | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c033333-global-20260517/20260517_p0_ta13_eval6` |
| TA-1/3 Code rerun | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c033333-global-20260517/20260517_p0_ta13_code_rerun01` |
| TA-0.75 Tool rerun | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c075-global-20260517/20260517_p0_ta075_tool_rerun01` |
| TA-0.75 Memory | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c075-global-20260517/20260517_p0_ta075_tool_memory` |
| TA-0.75 Code | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c075-global-20260517/20260517_p0_ta075_code` |
| init1 Tool/Memory | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-init1-global-20260517/20260517_p0_init1_tool_memory` |
| init1 Code | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-init1-global-20260517/20260517_p0_init1_code` |

## Notes

- `ta-c075` 首轮 Tool 全 0 是无效评测：日志中出现 OpenAI 404/model-not-found，随后生成结果不可解析；已单独重跑。
- Code/CURE 会在 generation 和 ground-truth unit-test 阶段之间释放 GPU；这些 GPU 视为保留资源，不能被长任务抢占，否则下一数据集会 OOM。
