# 2026-05-18 P0 SOTA Calibration V2 Audit

## 结论

`sota_calib_v2_20260518` 已构建并完成第一轮 expert coverage 审计。Tool / Memory 可进入动态 OPD 主训练；Code 暂不建议直接用当前 `train128` 全量进入主训练，因为新 targeted CodeContests 行过硬，两个 code expert 合并后只有 `21/48` 有 verified positive。

P0 决策：

- Tool：保留当前设计，正式评测增加 ToolRL `rlla_4k/test` all80 overall correct，不再只追 BFCL live 子类均值。
- Memory：保留当前 HotpotQA trajectory 设计，coverage 足够支撑 OPD。
- Code：先补 `samples_per_prompt=8` 的 ReasonFlux / DeepSeek expert rollout；若 union coverage 仍低于约 `0.65`，就把 targeted code 行降级为 monitor/guard，主训练只用 expert-recoverable code 行。

当前主线优先级：

1. 方法设计目标是把 task-vector composition 推到 SOTA，而不是在 BFCL Live 这种高波动子集上做单点最优。
2. Tool 的稳定代理指标改为 ToolRL test all80 overall correct：80 个 test prompt 直接统计做对比例，不做 live / non-live / AST / multi-turn 子类平均。
3. Memory / Code 的 P0 是 calibration 设计：训练题必须能通过官方 reward 或 expert trajectory 产生可学习正信号；hard probes 用于 monitor/guard，而不是强塞进 OPD 主训练稀释梯度。
4. 候选 checkpoint 必须同时过三类检查：训练 proxy 不崩、ToolRL all80 不明显掉、monitor64 三任务 reward 没有只牺牲某一项。

## 数据路径

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/
  train128.prompts.jsonl
  monitor64.prompts.jsonl
  guard64.prompts.jsonl
  summary.json
  expert_rollouts/
```

Split 计数：

| split | tool | memory | code | total |
|---|---:|---:|---:|---:|
| train128 | 32 | 48 | 48 | 128 |
| monitor64 | 16 | 24 | 24 | 64 |
| guard64 | 16 | 24 | 24 | 64 |

## Expert Coverage

第一轮 `samples_per_prompt=4`：

| task / expert | rows | covered | coverage | zero-positive |
|---|---:|---:|---:|---:|
| Tool / ToolRL-Qwen2.5-7B | 32 | 27 | 0.8438 | 5 |
| Memory / RL-MemoryAgent-7B | 48 | 37 | 0.7708 | 11 |
| Code / ReasonFlux-Coder-7B | 48 | 17 | 0.3542 | 31 |
| Code / DeepSeek-R1-Distill-Qwen-7B | 48 | 15 | 0.3125 | 33 |
| Code union | 48 | 21 | 0.4375 | 27 |

Code coverage breakdown：

| code role | covered | total | coverage |
|---|---:|---:|---:|
| source_code_anchor_from_paper96 | 14 | 16 | 0.8750 |
| on_policy_eval_style_code_probe | 7 | 32 | 0.2188 |

Interpretation：

- `paper96` code anchors 是可恢复的，适合作为 OPD train rows。
- 新 targeted CodeContests 行更像 hard monitor；如果强行进入 OPD 主训练，会导致 all-fail 行没有 expert positive，训练动力变弱。
- 这解释了此前“Code reward/gate 不涨”的主要机制之一：不是 gate 学不到，而是当前 code calibration 的可恢复 positive 密度太低。

## 当前动作

已启动 Code expert augmentation：

```bash
POLICY=code     SAMPLES_PER_PROMPT=8 GPU_LIST=0 bash skill/command/run_20260518_sota_v2_expert_rollouts.sh
POLICY=deepseek SAMPLES_PER_PROMPT=8 GPU_LIST=1 bash skill/command/run_20260518_sota_v2_expert_rollouts.sh
```

输出：

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/
  code_expert_reasonflux_coder7b_sota_v2_train128_s8_seed20260518.jsonl
  code_expert_deepseek_r1_distill_qwen7b_sota_v2_train128_s8_seed20260518.jsonl
```

## Recoverable-Code Train Split

基于 `samples_per_prompt=4` 的两个 code expert，已构建 recoverable-code 训练 split：

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/
  train_recoverable101.prompts.jsonl
  train_recoverable101.prompts.summary.json
```

计数：

| task | rows |
|---|---:|
| Tool | 32 |
| Memory | 48 |
| Code | 21 |
| Total | 101 |

Code role：

| role | rows |
|---|---:|
| source_code_anchor_from_paper96 | 14 |
| on_policy_eval_style_code_probe | 7 |

构建命令：

```bash
python scripts/data/build_recoverable_code_calibration.py \
  --input /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/train128.prompts.jsonl \
  --output /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl \
  --expert-rollout /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/code_expert_reasonflux_coder7b_sota_v2_train128_s4_seed20260518.jsonl \
  --expert-rollout /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_sota_v2_train128_s4_seed20260518.jsonl \
  --tool-count 32 \
  --memory-count 48 \
  --code-count -1 \
  --seed 20260518
```

这个 split 是当前主训练更合理的版本：它不丢掉 Tool/Memory 的训练信号，但避免 27 条 no-positive hard Code prompt 稀释 OPD。

## S8 Augmentation Result

`samples_per_prompt=8` 没有提升 Code prompt coverage，只是增加了已覆盖 prompt 的 positive sample 数：

| expert | rows | covered | coverage |
|---|---:|---:|---:|
| ReasonFlux s8 | 48 | 17 | 0.3542 |
| DeepSeek s8 | 48 | 15 | 0.3125 |

结论：当前 targeted Code prompts 的主要问题不是采样数太少，而是题目/测试分布对这两个 7B expert 太硬。后续应构造更分层的 Code pool，而不是继续简单加 samples。

## First Iteration Signal

已启动两个 recoverable101 主实验：

| run | gate space | iter | OPD selected | frontier selected | first gate movement |
|---|---|---:|---|---|---|
| `sota_v2_recoverable101_gc_init1_grpo_opd_ret_20260518` | global-coefficient | 1 | code=9, memory=2, tool=1 | code=4, memory=3, tool=4 | code=1.0205, memory=0.9584, tool=0.9531 |
| `sota_v2_recoverable101_gp_init1_grpo_opd_ret_20260518` | global-parameter | 1 | code=10, memory=1, tool=1 | code=4, memory=4, tool=4 | mean code=1.0205, memory=0.9783, tool=0.9562 |

Interpretation：

- Code 终于成为主要 OPD 入口，说明 recoverable split 修复了“Code 没正样本导致推不动”的核心问题。
- 从 `init=1` 出发时，第一步不是所有 expert 一起涨，而是 Code 上升、Tool 下降、Memory 小幅下降；这符合当前 data state：Code all-fail recoverable 比例最高。
- 这一步不能直接说明 eval 会涨，但它证明 calibration 信号方向更合理，值得继续跑到 8 iter 后做 monitor / ToolRL / Eval6 候选筛选。

## Iteration 1-4 Gate Dynamics

`recoverable101` 两个主实验都在 iter4 后被 guard 停止：Code coefficient 持续上升，但 Tool coefficient 快速下滑到约 `0.80`，继续训练很可能换来 Code proxy 增长但 Tool eval 损伤。

GC / global-coefficient：

| iter | OPD selected | frontier selected | retention | tool | memory | code |
|---:|---|---|---:|---:|---:|---:|
| 1 | code=9, memory=2, tool=1 | code=4, memory=3, tool=4 | 20 | 0.9531 | 0.9584 | 1.0205 |
| 2 | code=12, memory=3, tool=1 | code=4, memory=2, tool=4 | 17 | 0.8986 | 0.9648 | 1.0481 |
| 3 | code=12, memory=4, tool=2 | code=4, memory=2, tool=4 | 19 | 0.8553 | 0.9620 | 1.0749 |
| 4 | code=9, memory=1, tool=1 | code=4, memory=4, tool=4 | 20 | 0.7973 | 0.9498 | 1.1010 |

GP / global-parameter mean：

| iter | OPD selected | frontier selected | retention | tool | memory | code |
|---:|---|---|---:|---:|---:|---:|
| 1 | code=10, memory=1, tool=1 | code=4, memory=4, tool=4 | 18 | 0.9562 | 0.9783 | 1.0205 |
| 2 | code=8, memory=1, tool=1 | code=4, memory=3, tool=2 | 19 | 0.9009 | 0.9556 | 1.0472 |
| 3 | code=9, memory=2, tool=1 | code=4, memory=3, tool=4 | 21 | 0.8398 | 0.9657 | 1.0725 |
| 4 | code=12, memory=4, tool=3 | code=4, memory=1, tool=4 | 18 | 0.7968 | 0.9589 | 1.1004 |

Candidate policy mapping：

| logical checkpoint | baked policy path |
|---|---|
| iter3 after update | `iter_004/baked_policy` |
| iter4 after update | `iter_005/baked_policy` |

Fast evals launched:

| candidate | ToolRL all80 success | ToolRL mean reward | monitor64 tool | monitor64 memory | monitor64 code |
|---|---:|---:|---:|---:|---:|
| GC iter3 | 0.6000 | 0.8177 | 0.8840 / 0.6875 | 0.7812 / 0.7812 | 0.2578 / 0.1458 |
| GC iter4 | 0.5750 | 0.8181 | 0.8989 / 0.7188 | 0.7604 / 0.7604 | 0.2128 / 0.1458 |
| GP iter3 | 0.5875 | 0.8288 | 0.9044 / 0.7344 | 0.7396 / 0.7396 | 0.2331 / 0.1250 |
| GP iter4 | 0.5875 | 0.8093 | 0.9004 / 0.7344 | 0.7604 / 0.7604 | 0.1953 / 0.1250 |

Selection rule：若 iter4 的 ToolRL all80 明显低于 iter3，优先送 iter3 进正式 Eval6；若 iter4 Tool 不掉且 monitor64 Code 明显更高，再考虑 iter4。

ToolRL interpretation：

- `GC iter3` 暂时是 Tool success 最好候选。
- `GP iter3` 的 mean reward 最高，但 success 低于 `GC iter3`，说明有更多 partial-correct / parseable 但未完全做对的样本。
- `iter4` 没有在 ToolRL all80 上带来明确收益，是否保留取决于 monitor64 中 Code/Memory 是否显著补偿。

Reference ToolRL all80：

| model | success | mean reward | exact | parseable | zero-call |
|---|---:|---:|---:|---:|---:|
| best-ever TAME cg+r1calib | 0.6250 | 0.8381 | 0.5375 | 0.9000 | 0.1000 |
| TA-0.75 global | 0.6125 | 0.8310 | 0.5250 | 0.9000 | 0.1000 |
| recoverable101 GC iter3 | 0.6000 | 0.8177 | 0.5250 | 0.8750 | 0.1250 |
| recoverable101 GC iter4 | 0.5750 | 0.8181 | 0.5250 | 0.8750 | 0.1250 |
| recoverable101 GP iter3 | 0.5875 | 0.8288 | 0.5250 | 0.8875 | 0.1125 |
| recoverable101 GP iter4 | 0.5875 | 0.8093 | 0.5375 | 0.8625 | 0.1375 |

ToolRL conclusion：recoverable101 训练没有超过 TA / best-ever 的 ToolRL all80，最好候选 `GC iter3` 距 TA 差 `0.0125`、距 best-ever 差 `0.0250`。因此这条线如果进入正式评测，必须由 Memory/Code 的提升来证明 tradeoff 合理；不能只凭 train proxy 宣称 Tool 更强。

Monitor64 conclusion：

- `GC iter3` 是最均衡候选：ToolRL all80 最好，monitor Memory/Code 也最高。
- `GP iter3/iter4` 的 monitor Tool 更高，但 ToolRL all80 不如 `GC iter3`，Code monitor 也更低。
- `iter4` 没有把 Code 转化成更高 monitor reward，说明继续推高 Code coefficient 并不自动等于 Code 能力提升。

正式评测已启动：

| component | path |
|---|---|
| Tool BFCL | `/tmp/shared-storage/ExpertGym/eval/p0_sota_v2_20260518/gc_i3/tool` |
| Memory HotpotQA | `/tmp/shared-storage/ExpertGym/eval/p0_sota_v2_20260518/gc_i3/memory` |
| Code CURE | `/tmp/shared-storage/ExpertGym/eval/p0_sota_v2_20260518/gc_i3/code` |

First formal result:

| metric | value |
|---|---:|
| BFCL parallel | 0.8900 |
| BFCL parallel_multiple | 0.8700 |
| BFCL live_parallel | 0.7500 |
| BFCL live_parallel_multiple | 0.6250 |
| BFCL 4-category mean | 0.7838 |

Tool formal conclusion：GC iter3 的 Tool 没有超过 best-ever (`0.7954`)；它与 TA-0.75 (`0.7850`) 基本持平。后续是否保留该候选，取决于 Memory / Code 正式评测是否带来补偿。

Memory formal partial result:

| HotpotQA dataset | EM | Sub EM | F1 |
|---|---:|---:|---:|
| eval_50 | 0.6094 | 0.7656 | 0.7497 |
| eval_100 | 0.6016 | 0.7656 | 0.7368 |
| eval_qa_1_32768 | 0.6875 | 0.8438 | 0.7905 |
| eval_qa_1_65536 | failed | failed | failed |

Memory note：`eval_qa_1_65536` 在单卡 TP=1 和双卡 TP=2 两种设置下都出现 vLLM 子进程消失、GPU `[Not Found]` 残留 context、无结果文件的问题。该项目前记录为 harness failure，不作为模型能力结论。

Code formal result:

| CURE subset | sample acc | TP / accumulate acc | BoN(4,4) |
|---|---:|---:|---:|
| LiveBench | 0.3730 | 0.4816 | 0.4375 |
| LiveCodeBench | 0.3004 | 0.4294 | 0.3601 |
| mean | 0.3367 | 0.4555 | 0.3988 |

Code conclusion：GC iter3 没有超过 TA-1/3 Code mean Acc (`0.3409`) 和 BoN (`0.4173`)，也低于 best-ever / R1-inject 的 Code 结果。它的 training/monitor proxy 与正式 CURE 没有形成可用闭环，因此当前 recoverable101 只能作为诊断，不应作为论文主方法结果。下一步应转向 Code P0 v3 bank 与 scaled DeepSeek-R1 prior，而不是继续依赖 overall proxy 或单纯推高 code coefficient。

Code raw summary：`/tmp/shared-storage/OnPolicy/eval/cure_feedback/expertgym-p0-sota-v2-20260518-code/expertgym-p0-sota-v2-gc-i3/gc_i3_code_20260518/summary.json`

## Main Training Gate

只有满足下面条件才启动 `sota_v2_gc/gp` 主训练：

| condition | threshold |
|---|---:|
| Tool expert coverage | >= 0.75 |
| Memory expert coverage | >= 0.70 |
| Code union expert coverage | >= 0.65 preferred |
| Code source-anchor coverage | >= 0.80 |

如果 Code union coverage 仍不足：

1. 构建 `sota_calib_v2_recoverable_code`：
   - Tool 32 不变；
   - Memory 48 不变；
   - Code train 改为 recoverable source anchors + recoverable targeted rows；
   - 原 hard targeted rows 放入 monitor/guard。
2. 主训练用 recoverable Code，monitor 用 hard Code 检查泛化。
3. 论文叙事：calibration 不是泄露 eval，而是通过可执行 reward 选择“可恢复能力探针”；hard probes 用于验证不过拟合。

## ToolRL Test

Tool 的正式辅助评测使用：

```bash
MODEL_PATH=/path/to/baked_policy \
RUN_ID=my-model-toolrl-all80 \
GPU_LIST=0 \
bash skill/command/run_20260518_toolrl_rlla4k_eval.sh
```

主指标为：

```text
task_stats.tool.success_rate
```

这是 ToolRL all80 做对比例，不按子类平均。

## Monitor64 Test

三任务快速 proxy 使用：

```bash
MODEL_PATH=/path/to/baked_policy \
RUN_ID=my-model-monitor64 \
GPU_LIST=0 \
bash skill/command/run_20260518_sota_monitor64_eval.sh
```

主读数：

```text
task_stats.tool.mean_reward / success_rate
task_stats.memory.mean_reward / success_rate
task_stats.code.mean_reward / success_rate
```

Monitor64 只用于筛 checkpoint 和看训练方向；正式论文表仍以 Eval6 / ToolRL all80 / Code official suite / Memory official suite 为准。
