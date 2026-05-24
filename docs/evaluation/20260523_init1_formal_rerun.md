# 2026-05-23 init1 Formal Rerun

## Scope

本文件记录 `init1` baseline 的正式重测结果。目标模型为 `/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517`，对外模型名为 `ta-init1-global-20260523-rerun`。

旧 init1 对照来自 `docs/evaluation/20260517_p0_static_baselines_eval6.md`。

## Commands

Tool + Memory 有效命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
mkdir -p /tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523
export MODEL_PATH=/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517
export MODEL_NAME=ta-init1-global-20260523-rerun
export RUN_ID=20260523_init1_formal_rerun
export ROOT=/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523
export EXPERIMENT_NAME=expertgym-init1-formal-rerun-20260523
RUN_TOOL=1 RUN_MEMORY=1 RUN_CODE=0 \
  TOOL_GPU=0 TOOL_PORT=8170 \
  MEMORY_GPU_IDS=0 MEMORY_TP=1 \
  MEMORY_DATASETS="eval_50 eval_100 eval_qa_1_32768 eval_qa_1_65536" \
  SUMMARY_DIR=$ROOT/$MODEL_NAME/$RUN_ID/tool_memory \
  bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME" \
  2>&1 | tee $ROOT/${RUN_ID}_tool_memory.log
```

Code 首次使用 `CODE_GPU_GROUPS="[[2,3]]"` 时和另一条 code 评测冲突，物理 GPU2 显存满导致 OOM；该次只完成 LiveBench，不作为正式 Code 结果。有效 Code 重跑使用 GPU0 单卡：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export MODEL_PATH=/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517
export MODEL_NAME=ta-init1-global-20260523-rerun
export RUN_ID=20260523_init1_formal_rerun_code_gpu0
export ROOT=/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523
export EXPERIMENT_NAME=expertgym-init1-formal-rerun-20260523-code
RUN_TOOL=0 RUN_MEMORY=0 RUN_CODE=1 \
  CODE_GPU_GROUPS="[[0]]" CODE_MAX_TEST=8 CODE_MAX_GENERATION_TOKEN=10000 \
  SUMMARY_DIR=$ROOT/$MODEL_NAME/$RUN_ID/code \
  bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME" \
  2>&1 | tee $ROOT/${RUN_ID}.log
```

## Summary

| model | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ta-init1-global-20260523-rerun | 0.7631 | 0.65625 | 0.6465 | 0.7677 | 0.3291 | 0.4456 | 0.4017 | done |

## Tool / BFCL

| parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | live mean |
|---:|---:|---:|---:|---:|---:|
| 0.8800 | 0.8650 | 0.6875 | 0.6250 | 0.7631 | 0.65625 |

## Memory / HotpotQA

| dataset | EM | F1 |
|---|---:|---:|
| eval_50 | 0.6016 | 0.7494 |
| eval_100 | 0.6172 | 0.7559 |
| eval_qa_1_32768 | 0.6641 | 0.7639 |
| eval_qa_1_65536 | 0.7031 | 0.8016 |
| mean | 0.6465 | 0.7677 |

## Code / CURE

| dataset | Acc | TP / unit-test acc | Est. test acc | BoN(4,4) Acc | BoN(4,4) TP | result |
|---|---:|---:|---:|---:|---:|---|
| LiveBench | 0.3672 | 0.4728 | 0.4724 | 0.4453 | 0.5788 | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/eval/cure_feedback/expertgym-init1-formal-rerun-20260523-code/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun_code_gpu0/results/LiveBench.txt` |
| LiveCodeBench | 0.2911 | 0.4185 | 0.5091 | 0.3581 | 0.5257 | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/eval/cure_feedback/expertgym-init1-formal-rerun-20260523-code/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun_code_gpu0/results/LiveCodeBench.txt` |
| mean | 0.3291 | 0.4456 | 0.4908 | 0.4017 | 0.5523 | - |

## Diff vs 2026-05-17 init1

| metric | 2026-05-17 init1 | 2026-05-23 rerun | delta |
|---|---:|---:|---:|
| Tool mean | 0.7631 | 0.7631 | +0.0000 |
| Tool live mean | 0.6562 | 0.65625 | +0.00005 |
| Memory EM | 0.6387 | 0.6465 | +0.0078 |
| Memory F1 | 0.7583 | 0.7677 | +0.0094 |
| Code Acc | 0.3394 | 0.3291 | -0.0103 |
| Code TP | 0.4506 | 0.4456 | -0.0050 |
| Code BoN(4,4) Acc | 0.4115 | 0.4017 | -0.0098 |

Memory 相比旧记录小幅上升；Code 三项均低于旧记录。Tool 总体 mean 与旧记录持平，当前 BFCL 明细中 `parallel_multiple` 为 0.8650，较旧值 0.8600 高 0.0050。

## Artifacts

| item | path |
|---|---|
| output root | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523` |
| Tool + Memory log | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/20260523_init1_formal_rerun_tool_memory.log` |
| Tool + Memory summary dir | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun/tool_memory` |
| Tool BFCL log | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun/tool_memory/logs/tool_bfcl.log` |
| Memory summary | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/expertgym-init1-formal-rerun-20260523-memory/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun/summary.json` |
| Code valid log | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/20260523_init1_formal_rerun_code_gpu0.log` |
| Code summary dir | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun_code_gpu0/code` |
| Code feedback dir | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/eval/cure_feedback/expertgym-init1-formal-rerun-20260523-code/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun_code_gpu0` |
| Code summary | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/eval/cure_feedback/expertgym-init1-formal-rerun-20260523-code/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun_code_gpu0/summary.json` |
| Code report | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/eval/cure_feedback/expertgym-init1-formal-rerun-20260523-code/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun_code_gpu0/report.md` |
| invalid first Code log | `/tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/20260523_init1_formal_rerun_code.log` |

## Notes

- 有效 Code 结果来自 `RUN_ID=20260523_init1_formal_rerun_code_gpu0`，`gpu_groups="[[0]]"`，日志结尾包含 `[done] summary_dir: /tmp/shared-storage/ExpertGym/init1_formal_rerun_20260523/ta-init1-global-20260523-rerun/20260523_init1_formal_rerun_code_gpu0/code`。
- 首次 Code `[[2,3]]` 因 GPU2 与其他 code 评测冲突导致 OOM，未纳入正式结果。
