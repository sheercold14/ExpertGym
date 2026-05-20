# 20260519 TRC Round1 Evaluation

## 评测口径

- Checkpoint 选择：只按 TRC training loss 收敛/最低选择，不使用 gate 安全区间；gate 只作为诊断信息。
- 快评测：先跑 BFCL Tool 与 HotpotQA Memory；Tool mean acc >= 0.79 且 Memory mean F1 >= 0.76 的模型优先进入 Code/CURE。
- Code：E1/E3 已启动 CURE 评测；LiveBench 已完成，LiveCodeBench 仍在运行。
- Round2：R2D/R2E 正在做 Tool+Memory 快评测；Tool 已完成，Memory 仍在运行。

## Round1 配置索引

| ID | checkpoint | calibration | selected epoch | 备注 |
|---|---|---:|---:|---|
| E0 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e0_anchor_20260519-selected` | anchor TRC96 | 12 | 原始 anchor 对照 |
| E1 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e1_code_rf_20260519-selected` | code eval-aligned ReasonFlux | 12 | code trajectory 使用 RF-only 版本 |
| E2 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e2_code_multi_20260519-selected` | code eval-aligned multi-teacher | 12 | code trajectory 多教师 |
| E3 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e3_tool_code_multi_20260519-selected` | tool+code eval-aligned multi-teacher | 12 | 同时增强 Tool/Code calibration |

## Tool / Memory 快评测

| ID | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 F1 | eval_100 F1 | qa_32768 F1 | qa_65536 F1 | Code/CURE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7619 | 0.7654 | 0.7385 | 0.7907 | 0.7531 | skipped |
| E1 | 0.7969 | 0.8125 | 0.6250 | 0.9000 | 0.8500 | 0.7568 | 0.7360 | 0.7653 | 0.7948 | 0.7311 | LiveBench done; LiveCodeBench running |
| E2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7655 | 0.7604 | 0.7489 | 0.7962 | 0.7565 | skipped |
| E3 | 0.7956 | 0.8125 | 0.6250 | 0.9000 | 0.8450 | 0.7634 | 0.7674 | 0.7534 | 0.7750 | 0.7576 | LiveBench done; LiveCodeBench running |

## Code / CURE 当前结果

截至 2026-05-20 00:48 CST，E1/E3 的 CURE Code 评测均已完成。

| ID | run id | LiveBench Acc | LiveBench TP | LiveBench BoN(4,4) Acc | LiveBench BoN TP | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1 | `trc_r1_e1_code_20260519` | 0.3633 | 0.4608 | 0.4453 | 0.5690 | 0.2789 | 0.3795 | 0.3796 | done |
| E3 | `trc_r1_e3_code_20260519` | 0.3496 | 0.4422 | 0.4531 | 0.5632 | 0.2681 | 0.3726 | 0.3699 | done |

## Round2 Tool / Memory 快评测

截至 2026-05-20 00:48 CST，R2D/R2E 的 Tool/BFCL 与 Memory/HotpotQA 均已完成。

| ID | checkpoint | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 F1 | eval_100 F1 | qa_32768 F1 | qa_65536 F1 | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R2D | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r2d_cleanrf_wide_codeheavy_20260519-selected` | 0.8085 | 0.8125 | 0.6667 | 0.8950 | 0.8600 | 0.7403 | 0.7325 | 0.7275 | 0.7682 | 0.7328 | below memory threshold; skip Code for now |
| R2E | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r2e_cleanrf_wide_codelayers_20260520-selected` | 0.8085 | 0.8125 | 0.6667 | 0.8950 | 0.8600 | 0.7530 | 0.7336 | 0.7588 | 0.7598 | 0.7601 | below memory threshold; skip Code for now |

## 当前判断

E3 仍是 Round1 当前最干净候选：Tool 过线，Memory mean F1 过线，且 calibration 同时包含 Tool/Code eval-aligned trajectory。E1 的 Tool 过线但 Memory mean F1 略低于 0.76；E1 在已完成的 LiveBench Acc 上略高于 E3，但 E3 的 LiveBench BoN(4,4) Acc 略高，二者都应等 LiveCodeBench 完成后再做 Code 结论。

E0/E2 的 Memory 不差，但 Tool 输出格式全崩，主要错误是 BFCL AST parse failure，因此不适合作为主模型继续投入 Code 评测。

R2D/R2E 的 Tool mean 高于 E1/E3，且 live mean 也更高，但二者 Memory 都没有过 `0.76` 阈值。R2E 比 R2D 更均衡，Memory mean F1 从 `0.7403` 提升到 `0.7530`，但仍不够。这支持当前诊断：Round2 虽然能保住 Tool，但 memory calibration 仍使用 final-answer-only proxy，不能覆盖 MemAgent 的 update / evidence aggregation span。因此 R2D/R2E 暂不送 Code，后续优先看 Round3 memory-trajectory。

## Checkpoint Cleanup

2026-05-20 00:23 已删除无效 baked checkpoint，仅保留 run metrics / selected gates / eval logs 以便审计。

删除：

- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e0_anchor_20260519-selected`: Tool mean 0。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e2_code_multi_20260519-selected`: Tool mean 0。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1x_e1_e24_code_rf_20260519-selected`: 延长训练后 memory gate 明显下压，未进入正式 Code。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1x_e3_e24_tool_code_multi_20260519-selected`: 延长训练后 memory gate 明显下压，未进入正式 Code。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r2a_cleanrf_wide_dir_20260519-selected`: Tool mean 0。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r2b_cleanrf_wide_resp_20260519-selected`: Tool mean 0。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r2d_cleanrf_wide_codeheavy_20260519-selected`: Tool 强但 Memory mean F1 0.7403，未进入 Code。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r2e_cleanrf_wide_codelayers_20260520-selected`: Tool 强但 Memory mean F1 0.7530，未进入 Code。

保留：

- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e1_code_rf_20260519-selected`: Tool 强，正在 Code/CURE。
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r1_e3_tool_code_multi_20260519-selected`: Tool/Memory 过线，正在 Code/CURE。

## Raw Outputs

| item | path |
|---|---|
| E1 Code log | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r1_e1_code_rf_20260519_selected/trc_r1_e1_code_20260519/logs/code_cure.log` |
| E1 CURE root | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/trc-round1-code-code/trc_r1_e1_code_rf_20260519_selected/trc_r1_e1_code_20260519` |
| E3 Code log | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r1_e3_tool_code_multi_20260519_selected/trc_r1_e3_code_20260519/logs/code_cure.log` |
| E3 CURE root | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/trc-round1-code-code/trc_r1_e3_tool_code_multi_20260519_selected/trc_r1_e3_code_20260519` |
| R2D Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r2d_cleanrf_wide_codeheavy_20260519_selected/trc_r2d_tm_20260520` |
| R2D Memory artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/trc-round2-tm-memory/trc_r2d_cleanrf_wide_codeheavy_20260519_selected/trc_r2d_tm_20260520` |
| R2E Tool/Memory root | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_r2e_cleanrf_wide_codelayers_20260520_selected/trc_r2e_tm_20260520` |
| R2E Memory artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/trc-round2-tm-memory/trc_r2e_cleanrf_wide_codelayers_20260520_selected/trc_r2e_tm_20260520` |
