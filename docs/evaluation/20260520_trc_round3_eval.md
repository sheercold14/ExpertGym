# 20260520 TRC Round3 Evaluation

## 评测口径

- 先跑 Tool/BFCL 与 Memory/HotpotQA。
- Tool mean `>= 0.79` 且 Memory mean F1 `>= 0.76` 才进入 Code/CURE。
- Round3 的核心变量是 memory trajectory loss 与 coefficient-level retention，不用 gate 安全区间选择 checkpoint。

## Round3 Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 F1 | eval_100 F1 | qa_32768 F1 | qa_65536 F1 | Code/CURE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R3D | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3d_globalfloor50_u4_20260520-selected` | uniform4 + global coefficient floor=1.0,w=50 | 0.7944 | 0.8125 | 0.6250 | 0.8800 | 0.8600 | 0.7636 | 0.7691 | 0.7704 | 0.7738 | 0.7409 | LiveBench 0.3867 / LiveCodeBench 0.2710 |
| R3F | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3f_taskfloor50_u4_20260520-selected` | uniform4 + task-aware expert floor=1.0,w=50 | 0.7775 | 0.7500 | 0.6250 | 0.8800 | 0.8550 | pending | pending | pending | pending | pending | not prioritized |
| R3I | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3i_u4_codefull_taskfloor50_20260520-selected` | uniform4 + code full-response span + task-aware floor=50 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| R3J | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3j_late3_taskfloor50_20260520-selected` | late3 memory + task-aware floor=50 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7673 | 0.7738 | 0.7605 | 0.7697 | 0.7652 | LiveBench 0.3496 / LiveCodeBench 0.2779 |
| R3K | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3k_u4_codefull_globalfloor50_20260520-selected` | uniform4 + code full-response span + global floor=50 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7715 | 0.7750 | 0.7436 | 0.7821 | 0.7852 | LiveBench 0.3652 / LiveCodeBench 0.2794 |
| R3L | deleted | uniform4 + code-block span + code topk=384 + global floor=50 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7494 | 0.7577 | 0.7334 | 0.7654 | 0.7410 | rejected |
| R3M | deleted | uniform4 + code full-response span + tool multiplier=2.0 + global floor=50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | skipped | skipped | skipped | skipped | skipped | rejected |
| R3N | deleted | late3 + code-block topk384 + global floor=50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | skipped | skipped | skipped | skipped | skipped | rejected |
| R3O | deleted | late3 + code-block topk384 + task-aware floor=50 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7585 | 0.7590 | 0.7519 | 0.7634 | 0.7596 | rejected: Memory below 0.76 |

## Code / CURE

截至 2026-05-20 03:25 CST，R3D 的 CURE Code 已完整完成；R3J 的 LiveBench 已落盘，LiveCodeBench 仍在运行。

| ID | run id | LiveBench Acc | LiveBench TP | LiveBench BoN(4,4) Acc | LiveBench BoN TP | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN(4,4) Acc | mean Acc | mean TP | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R3D | `code_20260520_0216_rerun` | 0.3867 | 0.4750 | 0.4766 | 0.5544 | 0.2710 | 0.3818 | 0.3542 | 0.3289 | 0.4284 | 0.4154 | done |
| R3J | `code_20260520_0225` | 0.3496 | 0.4613 | 0.4375 | 0.5583 | 0.2779 | 0.3938 | 0.3562 | 0.3137 | 0.4275 | 0.3968 | done |
| R3K | `code_20260520_0330` | 0.3652 | pending | 0.4531 | pending | 0.2794 | pending | 0.3542 | 0.3223 | pending | 0.4037 | done |

## R3O Memory Status

R3O Tool 已完成，但 Memory/HotpotQA mean F1 为 `0.7585`，低于进入 Code 的 `0.76` 门槛；baked checkpoint 已删除，run artifacts 和 eval logs 保留。

| ID | eval_50 F1 | eval_100 F1 | qa_32768 F1 | qa_65536 F1 | mean F1 | status |
|---|---:|---:|---:|---:|---:|---|
| R3O | 0.7590 | 0.7519 | 0.7634 | 0.7596 | 0.7585 | rejected; baked deleted |

## Training Gate Summary

| ID | selected epoch | tool gate | memory gate | code gate | decision |
|---|---:|---:|---:|---:|---|
| R3D | 8 | 1.1520 | 0.9996 | 1.1599 | promoted to Code |
| R3F | 8 | 1.1520 | 1.0019 | 1.1599 | Tool below threshold; wait Memory only for diagnosis |
| R3I | 8 | 1.1567 | 1.0018 | 1.1599 | Tool+Memory running |
| R3J | 8 | 1.1523 | 1.0017 | 1.1599 | Tool+Memory running |
| R3K | 8 | 1.1567 | 0.9997 | 1.1599 | Tool+Memory running |
| R3L | 8 | 1.1523 | 0.9995 | 1.1599 | Tool+Memory running |
| R3M | 8 | 1.1577 | 0.9998 | 1.1596 | Tool collapsed; checkpoint deleted |
| R3N | 8 | 1.1527 | 0.9994 | 1.1599 | Tool collapsed; checkpoint deleted |
| R3O | 8 | 1.1528 | 1.0018 | 1.1599 | Tool pass but Memory below threshold; baked checkpoint deleted |

## Raw Outputs

| item | path |
|---|---|
| R3D Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3d_globalfloor50_u4_20260520-selected/toolmem_20260520_0140_fix` |
| R3D Memory artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/trc-r3d-toolmem-memory/trc_r3d_globalfloor50_u4_20260520-selected/toolmem_20260520_0140_fix` |
| R3D Code root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3d_globalfloor50_u4_20260520-selected/code_20260520_0216_rerun` |
| R3F Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3f_taskfloor50_u4_20260520-selected/toolmem_20260520_0140_fix` |
| R3I Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3i_u4_codefull_taskfloor50_20260520-selected/toolmem_20260520_0200` |
| R3J Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3j_late3_taskfloor50_20260520-selected/toolmem_20260520_0200` |
| R3J Code root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3j_late3_taskfloor50_20260520-selected/code_20260520_0225` |
| R3J CURE raw root | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/trc-r3j-code-code/trc_r3j_late3_taskfloor50_20260520-selected/code_20260520_0225` |
| R3K Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3k_u4_codefull_globalfloor50_20260520-selected/toolmem_20260520_0216` |
| R3K Tool/Memory root rebake | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3k_u4_codefull_globalfloor50_20260520-selected/toolmem_20260520_0240_rebake` |
| R3K Code root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3k_u4_codefull_globalfloor50_20260520-selected/code_20260520_0330` |
| R3L Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3l_u4_codeblock384_globalfloor50_20260520-selected/toolmem_20260520_0232` |
| R3M Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3m_u4_codefull_toolprotect_globalfloor50_20260520-selected/toolmem_20260520_0232` |
| R3N Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3n_late3_codeblock384_globalfloor50_20260520-selected/toolmem_20260520_0306` |
| R3O Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r3o_late3_codeblock384_taskfloor50_20260520-selected/toolmem_20260520_0306` |
| R3O Memory artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/trc-r3o-toolmem-memory/trc_r3o_late3_codeblock384_taskfloor50_20260520-selected/toolmem_20260520_0306` |

## 当前判断

R3D 是 Round3 第一个满足 Tool/Memory 双阈值的模型。相比 R2D/R2E，Tool 略低但仍过线，Memory 从 `0.7403/0.7530` 提升到 `0.7636`，说明 memory trajectory + coefficient retention 的组合有效。

`code_20260520_0200` 在 `481/512` 时因后续训练占用同组 GPU 触发 CURE vLLM OOM，属于资源调度失败，不计入模型结果；已用空闲 GPU 4/7 以 `code_20260520_0216_rerun` 重跑。

R3K 第一次 Tool eval 失败是因为 checkpoint bake 缺少第 4 个 shard；已删除不完整目录并从 `selected.gates.json` 重新 bake，确认 `15G / 4 shards` 后以 `toolmem_20260520_0240_rebake` 重跑。

R3D Code/CURE rerun completed:

- LiveBench code acc `0.3867`, BoN `(4,4)` acc `0.4766`。
- LiveCodeBench code acc `0.2710`, BoN `(4,4)` acc `0.3542`。
- Simple mean code acc `0.3289`, BoN acc `0.4154`。

R3J Code/CURE completed:

- LiveBench code acc `0.3496`, BoN `(4,4)` acc `0.4375`。
- LiveCodeBench code acc `0.2779`, BoN `(4,4)` acc `0.3562`。
- Simple mean code acc `0.3137`, BoN acc `0.3968`。
- R3J has better Memory than R3D (`0.7673` vs `0.7636`) but worse Code (`0.3137` vs `0.3289`), so late3 alone is not enough for Code.

R3K Code/CURE completed:

- LiveBench code acc `0.3652`, BoN `(4,4)` acc `0.4531`。
- LiveCodeBench code acc `0.2794`, BoN `(4,4)` acc `0.3542`。
- Simple mean code acc `0.3223`, BoN acc `0.4037`。
- R3K has the best Round3 Memory (`0.7715`) but still does not beat R3D on Code, suggesting full-response code span improves Memory compatibility more than Code correctness.

R3O Tool passed but Memory mean F1 only `0.7585`, below the `0.76` promotion threshold. Its baked checkpoint has been deleted; run artifacts and eval logs are retained.
