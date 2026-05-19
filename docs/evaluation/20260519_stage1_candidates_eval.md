# 2026-05-19 Stage-1 Candidate Full Evaluation

## Scope

本表记录第一阶段 TRC directional 候选的 full-suite 正式评测。候选来自 `20260519_trc_stage1_takeover.md`：

- `trc_stage1_v3_anchor_i4_20260519`: 更保守，Memory gate 保留更高。
- `trc_stage1_v3_anchor_i8_20260519`: 更激进，Code/Tool gate 推得更高。
- `trc_stage1_v3_dir_i8_20260519`: 非 anchor 的激进 directional 候选，用于检查更强 TRC 推动是否换来 Code/Tool 增益。

## Evaluation Objects

| model | checkpoint | train setting | status |
|---|---|---|---|
| `trc_stage1_v3_anchor_i4_20260519` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i4_20260519` | TRC directional anchor, epoch 4 | Code LiveCodeBench running |
| `trc_stage1_v3_anchor_i8_20260519` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i8_20260519` | TRC directional anchor, epoch 8 | Code LiveCodeBench running |
| `trc_stage1_v3_dir_i8_20260519` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_dir_i8_20260519` | TRC directional, epoch 8 | Code CURE running |

## Summary

| model | type | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| trc_stage1_v3_anchor_i4_20260519 | TRC directional anchor, e4 | 0.7788 | 0.6875 | 0.6348 | 0.7603 | pending | pending | pending | Code LiveCodeBench running |
| trc_stage1_v3_anchor_i8_20260519 | TRC directional anchor, e8 | 0.7800 | 0.6875 | 0.6406 | 0.7594 | pending | pending | pending | Code LiveCodeBench running |
| trc_stage1_v3_dir_i8_20260519 | TRC directional, e8 | 0.7981 | 0.7188 | 0.6445 | 0.7663 | pending | pending | pending | Code CURE running |

## Tool / BFCL

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | live mean | status |
|---|---:|---:|---:|---:|---:|---:|---|
| trc_stage1_v3_anchor_i4_20260519 | 0.8850 | 0.8550 | 0.7500 | 0.6250 | 0.7788 | 0.6875 | done |
| trc_stage1_v3_anchor_i8_20260519 | 0.8900 | 0.8550 | 0.7500 | 0.6250 | 0.7800 | 0.6875 | done |
| trc_stage1_v3_dir_i8_20260519 | 0.9050 | 0.8500 | 0.8125 | 0.6250 | 0.7981 | 0.7188 | done |

## Memory / HotpotQA

| model | eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 | status |
|---|---:|---:|---:|---:|---:|---:|---|
| trc_stage1_v3_anchor_i4_20260519 | 0.7504 | 0.7502 | 0.7743 | 0.7666 | 0.6348 | 0.7603 | done |
| trc_stage1_v3_anchor_i8_20260519 | 0.7478 | 0.7365 | 0.7834 | 0.7699 | 0.6406 | 0.7594 | done |
| trc_stage1_v3_dir_i8_20260519 | 0.7686 | 0.7456 | 0.7807 | 0.7701 | 0.6445 | 0.7663 | done |

## Code / CURE

| model | run id | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| trc_stage1_v3_anchor_i4_20260519 | stage1_20260519 | 0.3633 | 0.4665 | 0.4297 | pending | pending | pending | pending | pending | pending | LiveCodeBench running |
| trc_stage1_v3_anchor_i8_20260519 | stage1_20260519_rerun | 0.3770 | 0.4804 | 0.4609 | pending | pending | pending | pending | pending | pending | LiveCodeBench running |
| trc_stage1_v3_dir_i8_20260519 | stage1_20260519_dir_i8 | pending | pending | pending | pending | pending | pending | pending | pending | pending | Code CURE running |

## Raw Outputs

| item | path |
|---|---|
| i4 summary dir | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_anchor_i4_20260519/stage1_20260519` |
| i8 summary dir | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_anchor_i8_20260519/stage1_20260519_rerun` |
| i4 CURE raw root | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/trc_stage1_v3_anchor_i4_20260519/stage1_20260519` |
| i8 CURE raw root | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/trc_stage1_v3_anchor_i8_20260519/stage1_20260519_rerun` |
| dir i8 summary dir | `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_dir_i8_20260519/stage1_20260519_dir_i8` |
| eval wrapper | `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/skill/command/run_full_eval_suite.sh` |

## Interim Read

在 Tool/Memory 已完成的范围内，`dir_i8` 同时取得最高 Tool mean、Tool live mean、Memory EM 和 Memory F1。它目前只缺 Code CURE；如果 Code 不显著低于 anchor，它应优先作为第一阶段主候选。`anchor_i8` 在已完成 Code LiveBench 上优于 `anchor_i4`。
