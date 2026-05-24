# 2026-05-23 cg-tool-extra020 Reproduction Rerun

## Scope

本文件记录 `cg-tool-extra020` 的重新烘焙与正式重测结果。该实验是不加 R1、非 PAUH 的三专家静态合并锚点。

合并公式：

```text
theta = base
      + 0.50 * tool_delta
      + 0.75 * memory_delta
      + 0.75 * code_delta
      + 0.20 * safe_mask(tool_delta, code_delta) * tool_delta
```

其中 `safe_mask(tool_delta, code_delta)` 只在 Tool residual 与 Code residual 元素级不冲突的位置打开；`lm_head`、`norm`、`embed_tokens`、`bias` 不合并。重烘焙 summary 显示 `merged_tensors=196`，`copied_excluded_tensors=143`，`gate_open_fraction=0.9743655818`，与旧 checkpoint 一致。

## Reproduced Checkpoint

| item | path |
|---|---|
| rebaked model | `/tmp/shared-storage/ExpertGym/reproduce/cg_tool_extra020_20260523_rebake1/baked_policy` |
| rebake log | `/tmp/shared-storage/ExpertGym/reproduce/cg_tool_extra020_20260523_rebake1/rebake.log` |
| rebake summary | `/tmp/shared-storage/ExpertGym/reproduce/cg_tool_extra020_20260523_rebake1/baked_policy/conflict_gated_residual_summary.json` |
| original canonical model | `/tmp/shared-storage/AgentMerging_plan/experiments/conflict_gated_residual/ITER_019_cg_tool_extra020/model` |
| source build script | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/pro_report_20260430_125810/04_configs_and_scripts/build_conflict_gated_residual_merge.py` |

Source models:

| role | model |
|---|---|
| base | `/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct` |
| tool | `/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold` |
| memory | `/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B` |
| code | `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B` |

## Commands

Rebake:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export PYTHONPATH=/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/AgentMerging_plan
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  /mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/pro_report_20260430_125810/04_configs_and_scripts/build_conflict_gated_residual_merge.py \
  --base-model /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct \
  --tool-model /mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold \
  --memory-model /mnt/cache/wuruixiao/models/RL-MemoryAgent-7B \
  --code-model /mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B \
  --output-dir /tmp/shared-storage/ExpertGym/reproduce/cg_tool_extra020_20260523_rebake1/baked_policy \
  --run-name cg-tool-extra020-20260523-rebake \
  --alpha-tool-base 0.5 \
  --alpha-memory 0.75 \
  --alpha-code 0.75 \
  --alpha-tool-extra 0.2 \
  --protected-eps 0.0 \
  --tool-extra-protect code
```

Tool + Memory:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export MODEL_PATH=/tmp/shared-storage/ExpertGym/reproduce/cg_tool_extra020_20260523_rebake1/baked_policy
export MODEL_NAME=cg-tool-extra020-20260523-rebake
export RUN_ID=20260523_cg_tool_extra020_rerun
export ROOT=/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523
export EXPERIMENT_NAME=expertgym-cg-tool-extra020-rerun-20260523
RUN_TOOL=1 RUN_MEMORY=1 RUN_CODE=0 \
  TOOL_GPU=0 TOOL_PORT=8191 \
  MEMORY_GPU_IDS=0 MEMORY_TP=1 \
  MEMORY_DATASETS="eval_50 eval_100 eval_qa_1_32768 eval_qa_1_65536" \
  SUMMARY_DIR=$ROOT/$MODEL_NAME/$RUN_ID/tool_memory \
  bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME"
```

Code 有效重跑使用两个单卡 worker。首次 `CODE_GPU_GROUPS="[[2,3]]"` 因 CURE/vLLM tensor-parallel 初始化 OOM，未纳入结果。

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export MODEL_PATH=/tmp/shared-storage/ExpertGym/reproduce/cg_tool_extra020_20260523_rebake1/baked_policy
export MODEL_NAME=cg-tool-extra020-20260523-rebake
export RUN_ID=20260523_cg_tool_extra020_rerun_code_single2x
export ROOT=/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523
export EXPERIMENT_NAME=expertgym-cg-tool-extra020-rerun-20260523
RUN_TOOL=0 RUN_MEMORY=0 RUN_CODE=1 \
  CODE_GPU_GROUPS="[[2],[3]]" \
  CODE_MAX_TEST=8 CODE_MAX_GENERATION_TOKEN=10000 \
  SUMMARY_DIR=$ROOT/$MODEL_NAME/$RUN_ID/code \
  bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME"
```

## Summary

| model | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cg-tool-extra020-20260523-rebake | 0.7825 | 0.6875 | 0.6328 | 0.7618 | 0.3546 | 0.4831 | 0.4007 | done |

## Tool / BFCL

| parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | live mean |
|---:|---:|---:|---:|---:|---:|
| 0.9000 | 0.8550 | 0.7500 | 0.6250 | 0.7825 | 0.6875 |

## Memory / HotpotQA

| dataset | EM | F1 |
|---|---:|---:|
| eval_50 | 0.6094 | 0.7562 |
| eval_100 | 0.6172 | 0.7553 |
| eval_qa_1_32768 | 0.6406 | 0.7657 |
| eval_qa_1_65536 | 0.6641 | 0.7701 |
| mean | 0.6328 | 0.7618 |

## Code / CURE

| dataset | Acc | TP / unit-test acc | Est. test acc | BoN(4,4) Acc | BoN(4,4) TP | result |
|---|---:|---:|---:|---:|---:|---|
| LiveBench | 0.3984 | 0.5110 | 0.4111 | 0.4453 | 0.5906 | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/eval/cure_feedback/expertgym-cg-tool-extra020-rerun-20260523-code/cg-tool-extra020-20260523-rebake/20260523_cg_tool_extra020_rerun_code_single2x/results/LiveBench.txt` |
| LiveCodeBench | 0.3107 | 0.4551 | 0.4512 | 0.3562 | 0.5161 | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/eval/cure_feedback/expertgym-cg-tool-extra020-rerun-20260523-code/cg-tool-extra020-20260523-rebake/20260523_cg_tool_extra020_rerun_code_single2x/results/LiveCodeBench.txt` |
| mean | 0.3546 | 0.4831 | 0.4312 | 0.4007 | 0.5534 | - |

## Diff vs Old Record

旧记录来自 `/tmp/shared-storage/TAME/experiments/tame-3expert-opvec96-20260517/analysis_summary.md`。

| metric | old cg-tool-extra020 | 2026-05-23 rerun | delta |
|---|---:|---:|---:|
| Tool mean | 0.7942 | 0.7825 | -0.0117 |
| Tool live_parallel | 0.7500 | 0.7500 | +0.0000 |
| Tool live_parallel_multiple | 0.6667 | 0.6250 | -0.0417 |
| Memory F1 mean | 0.7610 | 0.7618 | +0.0008 |
| Code Acc mean | 0.3497 | 0.3546 | +0.0049 |
| Code BoN mean | 0.4037 | 0.4007 | -0.0030 |

结论：重烘焙可复现原配方，Memory 与 Code 基本落在旧记录波动范围内；Tool 总分低于旧记录，主要由 `live_parallel_multiple` 从约 0.667 降到 0.625 导致。

## Artifacts

| item | path |
|---|---|
| output root | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523` |
| Tool + Memory log | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/20260523_cg_tool_extra020_rerun_tool_memory.log` |
| Tool BFCL log | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/cg-tool-extra020-20260523-rebake/20260523_cg_tool_extra020_rerun/tool_memory/logs/tool_bfcl.log` |
| Memory summary | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/expertgym-cg-tool-extra020-rerun-20260523-memory/cg-tool-extra020-20260523-rebake/20260523_cg_tool_extra020_rerun/summary.json` |
| Memory report | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/expertgym-cg-tool-extra020-rerun-20260523-memory/cg-tool-extra020-20260523-rebake/20260523_cg_tool_extra020_rerun/report.md` |
| Code valid log | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/20260523_cg_tool_extra020_rerun_code_single2x.log` |
| Code summary | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/eval/cure_feedback/expertgym-cg-tool-extra020-rerun-20260523-code/cg-tool-extra020-20260523-rebake/20260523_cg_tool_extra020_rerun_code_single2x/summary.json` |
| Code report | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/eval/cure_feedback/expertgym-cg-tool-extra020-rerun-20260523-code/cg-tool-extra020-20260523-rebake/20260523_cg_tool_extra020_rerun_code_single2x/report.md` |
| invalid first Code log | `/tmp/shared-storage/ExpertGym/cg_tool_extra020_rerun_20260523/20260523_cg_tool_extra020_rerun_code_code.log` |

