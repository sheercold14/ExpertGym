# 2026-05-23 DeepSeek Anchor + Agent Full Task Vector

## Scope

本文件记录一次诊断性实验：以 `DeepSeek-R1-Distill-Qwen-7B` 作为 anchor，直接全量加入三个 Agent expert 的 OP-VEC task vector。

合并公式：

```text
theta = DeepSeek-R1-Distill-Qwen-7B
      + 1.0 * (ToolRL - Qwen2.5-Instruct)
      + 1.0 * (RL-MemoryAgent - Qwen2.5-Instruct)
      + 1.0 * (ReasonFlux-Coder - Qwen2.5-Instruct)
```

这里的“全量”指 OP-VEC manifest 中可合并的 196 个权重张量、588 个 expert delta entry 均使用系数 1.0；`lm_head`、`norm`、`embed_tokens`、`bias` 等 sidecar 直接从 DeepSeek anchor 复制。

## Reproduced Checkpoint

| item | path |
|---|---|
| baked model | `/tmp/shared-storage/ExpertGym/reproduce/deepseek_anchor_agentfull_20260523/baked_policy` |
| bake log | `/tmp/shared-storage/ExpertGym/reproduce/deepseek_anchor_agentfull_20260523/bake.log` |
| bake summary | `/tmp/shared-storage/ExpertGym/reproduce/deepseek_anchor_agentfull_20260523/baked_policy/anchor_task_vector_bake_summary.json` |
| bake script | `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/eval/bake_anchor_task_vector_checkpoint.py` |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` |

Source models:

| role | model |
|---|---|
| anchor | `/mnt/cache/wuruixiao/models/DeepSeek-R1-Distill-Qwen-7B` |
| manifest base | `/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct` |
| tool expert | `/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold` |
| memory expert | `/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B` |
| code expert | `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B` |

Bake summary:

| field | value |
|---|---:|
| num merged tensors | 196 |
| num delta entries | 588 |
| tool weighted delta block L2 sum | 14.9920 |
| memory weighted delta block L2 sum | 64.1261 |
| code weighted delta block L2 sum | 7.4675 |
| mean anchor tensor L2 norm | 126.5705 |
| mean updated tensor L2 norm | 126.5743 |

## Commands

Bake:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/eval/bake_anchor_task_vector_checkpoint.py \
  --anchor-model /mnt/cache/wuruixiao/models/DeepSeek-R1-Distill-Qwen-7B \
  --output /tmp/shared-storage/ExpertGym/reproduce/deepseek_anchor_agentfull_20260523/baked_policy \
  --expert-weights tool=1.0,memory=1.0,code=1.0
```

Tool + Memory:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
ROOT=/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523
MODEL_PATH=/tmp/shared-storage/ExpertGym/reproduce/deepseek_anchor_agentfull_20260523/baked_policy
MODEL_NAME=deepseek-anchor-agentfull-20260523
RUN_ID=20260523_deepseek_anchor_agentfull
EXPERIMENT_NAME=expertgym-deepseek-anchor-agentfull-20260523
RUN_TOOL=1 RUN_MEMORY=1 RUN_CODE=0 \
  TOOL_GPU=0 TOOL_PORT=8192 \
  MEMORY_GPU_IDS=0 MEMORY_TP=1 \
  MEMORY_DATASETS="eval_50 eval_100 eval_qa_1_32768 eval_qa_1_65536" \
  SUMMARY_DIR=$ROOT/$MODEL_NAME/$RUN_ID/tool_memory \
  bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME"
```

Code:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
ROOT=/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523
MODEL_PATH=/tmp/shared-storage/ExpertGym/reproduce/deepseek_anchor_agentfull_20260523/baked_policy
MODEL_NAME=deepseek-anchor-agentfull-20260523
RUN_ID=20260523_deepseek_anchor_agentfull_code
EXPERIMENT_NAME=expertgym-deepseek-anchor-agentfull-20260523
RUN_TOOL=0 RUN_MEMORY=0 RUN_CODE=1 \
  CODE_GPU_GROUPS="[[2],[3]]" \
  CODE_MAX_TEST=8 CODE_MAX_GENERATION_TOKEN=10000 \
  SUMMARY_DIR=$ROOT/$MODEL_NAME/$RUN_ID/code \
  bash skill/command/run_full_eval_suite.sh "$MODEL_PATH" "$MODEL_NAME"
```

## Summary

| model | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek-anchor-agentfull-20260523 | 0.3760 | 0.2396 | 0.0293 | 0.0534 | 0.4299 | 0.4658 | 0.5348 | done |

结论：全量加入三个 Agent task vector 后，Code 明显增强，但 Tool 和 Memory 几乎失效。该模型不能作为三任务 merged model 使用，但它证明 DeepSeek/R1 anchor 对 Code 有强增益，同时也暴露跨 anchor 直接相加会破坏 Tool/Memory 行为协议。

## Tool / BFCL

| parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | live mean | non-live mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5600 | 0.4650 | 0.3125 | 0.1667 | 0.3760 | 0.2396 | 0.5125 |

Tool 主要失败形式包括 `ast_decoder:decoder_failed` 和 `cannot_find_match`，说明问题不是单纯 tool knowledge 不足，而是 tool-call 格式、schema matching 和 live multi-call 行为被 DeepSeek/R1 anchor 的生成风格严重扰动。

## Memory / HotpotQA

| dataset | EM | F1 | subEM |
|---|---:|---:|---:|
| eval_50 | 0.0234 | 0.0477 | 0.3984 |
| eval_100 | 0.0547 | 0.0909 | 0.2500 |
| eval_qa_1_32768 | 0.0234 | 0.0447 | 0.2812 |
| eval_qa_1_65536 | 0.0156 | 0.0302 | 0.2500 |
| mean | 0.0293 | 0.0534 | 0.2950 |

Memory 的 F1 极低。结合 subEM 仍非零，模型并非完全没有检索/推理迹象，而是最终回答协议、boxed/final answer 约束和 MemAgent-style 输出行为没有被保住。

## Code / CURE

| dataset | Acc | TP / unit-test acc | Est. test acc | BoN(4,4) Acc | BoN(4,4) TP | avg code length |
|---|---:|---:|---:|---:|---:|---:|
| LiveBench | 0.3876 | 0.4258 | 0.6256 | 0.4862 | 0.5328 | 6467.84 |
| LiveCodeBench | 0.4722 | 0.5058 | 0.6262 | 0.5833 | 0.6297 | 6318.09 |
| mean | 0.4299 | 0.4658 | 0.6259 | 0.5348 | 0.5813 | 6392.96 |

Code 是该实验唯一正向结果。它显著高于 init1 和 cg-tool-extra020，尤其 BoN 提升很大，说明 DeepSeek/R1 anchor 的 reasoning/code search 能力能提高候选解上限。但单样本 Acc 与 BoN 仍有较大差距，后续如果沿这条线做 Code，需要解决 selection/stability，而不是只继续放大 task vector。

## Comparison

| model | Tool mean | Tool live mean | Memory F1 | Code Acc | Code BoN(4,4) Acc |
|---|---:|---:|---:|---:|---:|
| init1 rerun | 0.7631 | 0.6563 | 0.7677 | 0.3291 | 0.4017 |
| cg-tool-extra020 rerun | 0.7825 | 0.6875 | 0.7618 | 0.3546 | 0.4007 |
| DeepSeek anchor + agent full | 0.3760 | 0.2396 | 0.0534 | 0.4299 | 0.5348 |

Delta against init1:

| metric | delta |
|---|---:|
| Tool mean | -0.3871 |
| Tool live mean | -0.4167 |
| Memory F1 | -0.7143 |
| Code Acc | +0.1008 |
| Code BoN(4,4) Acc | +0.1331 |

Delta against cg-tool-extra020:

| metric | delta |
|---|---:|
| Tool mean | -0.4065 |
| Tool live mean | -0.4479 |
| Memory F1 | -0.7084 |
| Code Acc | +0.0753 |
| Code BoN(4,4) Acc | +0.1341 |

## Interpretation

1. 直接把 Qwen-Instruct expert residual 加到 DeepSeek/R1 anchor 上不是稳健的三任务合并方法。Code 得益于 anchor 本身的 reasoning/code prior，但 Tool/Memory 的行为协议被破坏。
2. Memory delta 的 block-norm 最大，但 Memory F1 接近 0，说明 residual 能量大不等于能力可迁移。跨 anchor 时，输出格式、trajectory protocol 和 behavior span 比参数范数更关键。
3. Code 的 BoN 大幅提高，说明这条线可以作为 Code 上限诊断：DeepSeek/R1 anchor 能生成更多可通过候选，但需要额外机制提高单次采样稳定性。
4. 对论文主线更有价值的结论不是“DeepSeek 全量相加有效”，而是“异质 reasoning anchor 可以提升 Code search space，但必须配套 Tool/Memory behavior preservation 或 anchor-aware residual filtering”。

## Artifacts

| item | path |
|---|---|
| eval root | `/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523` |
| Tool + Memory summary dir | `/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523/deepseek-anchor-agentfull-20260523/20260523_deepseek_anchor_agentfull/tool_memory` |
| Tool BFCL log | `/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523/deepseek-anchor-agentfull-20260523/20260523_deepseek_anchor_agentfull/tool_memory/logs/tool_bfcl.log` |
| Tool score dir | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/deepseek-anchor-agentfull-20260523-tool-20260523-deepseek-anchor-agentfull-20260523_deepseek_anchor_agentfull/deepseek-anchor-agentfull-20260523-tool-20260523-deepseek-anchor-agentfull` |
| Memory summary | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/expertgym-deepseek-anchor-agentfull-20260523-memory/deepseek-anchor-agentfull-20260523/20260523_deepseek_anchor_agentfull/summary.json` |
| Code summary dir | `/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523/deepseek-anchor-agentfull-20260523/20260523_deepseek_anchor_agentfull_code/code` |
| Code summary | `/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523/eval/cure_feedback/expertgym-deepseek-anchor-agentfull-20260523-code/deepseek-anchor-agentfull-20260523/20260523_deepseek_anchor_agentfull_code/summary.json` |
| Code report | `/tmp/shared-storage/ExpertGym/deepseek_anchor_agentfull_eval_20260523/eval/cure_feedback/expertgym-deepseek-anchor-agentfull-20260523-code/deepseek-anchor-agentfull-20260523/20260523_deepseek_anchor_agentfull_code/report.md` |

