# 2026-05-17 H 实验配置：global-coefficient + GRPO + OPD

## 目的

在上一轮 B 的 paper96 calibration 设置下，把可学习 gate 从 `global-parameter` 换成 `global-coefficient`，只学习三个直接 task-vector 系数：

```text
tool, memory, code
```

目标是检验：在更低维的系数空间里，同时打开 `GRPO + dynamic OPD + retention`，能否比 common+residual 的 global-parameter 更稳定地找到整体 reward 更高的组合。

## 核心设置

| 项 | 值 |
|---|---|
| 实验名 | `H` |
| launcher | `skill/command/run_20260517_expH_global_coeff_grpo_opd_ret.sh` |
| run dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expH_gc_c033333_grpo_opd_ret_20260517` |
| gate strategy | `global-coefficient` |
| 可学习参数 | `3` 个直接系数：tool / memory / code |
| init checkpoint | `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/init_gates/init_global_coefficient_c033333.json` |
| init | `tool=memory=code=1/3` |
| prompts | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| prompts count | `96 = tool 32 + memory 32 + code 32` |
| samples per prompt | `4` |
| iterations | `20` |
| optimizer | SGD, momentum `0.2`, persisted state |
| lr | `0.1876` |
| optimizer step | epoch-scope |
| loss granularity | sequence |
| GRPO | enabled, `PPO_LOSS_WEIGHT=1.0` |
| OPD | enabled, `OPD_LOSS_WEIGHT=1.0` |
| retention | enabled, NLL, task-balanced row scale, target `0.5` |
| prior | `0.0` |
| max coefficient delta from init | `1.0` |
| rollout | 2 single-GPU vLLM shards by `ROLLOUT_SHARDS=auto` |
| intended GPU | preferred `2,3`; fallback `6,7` if G final update releases first |

## OPD expert pool

沿用 B 的 code-aug expert pool：

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl
```

## 启动命令

确认 GPU 2/3 没有训练或评测进程后：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expH_gc_20260517 \
  'GPU_LIST=2,3 bash skill/command/run_20260517_expH_global_coeff_grpo_opd_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expH_gc_c033333_grpo_opd_ret_20260517/train.log'
```

当前实际启动由 watcher 管理，避免抢占 F/G 的 final update：

```text
tmux: watch_launch_expH_gc_20260517
log: /tmp/shared-storage/OnPolicy/runs/gated_grpo/expH_gc_c033333_grpo_opd_ret_20260517/launch_wait.log
rule: F iter20 完成且 2/3 空闲时用 2/3；否则 G iter20 完成且 6/7 空闲时用 6/7；使用 .launch_started 防止重复启动。
```

## 监控重点

| 指标 | 判断 |
|---|---|
| overall proxy reward | 选 best checkpoint 的主依据 |
| tool / memory / code proxy reward | 判断三任务是否出现单项崩溃 |
| 三个 global coefficients | 判断直接系数是否比 global-parameter 更可解释 |
| `opd_distill_from_allfail.summary.json` | 确认 OPD 是否仍有 all-fail positive 信号 |
| `gate_updates.summary.json` grad / delta | 判断三参数下是否过冲 |

Code 正式评测会滞后补齐；当前优先看训练 proxy 和 Tool/Memory 正式 eval。

## 运行观察

更新时间：`2026-05-17 12:30 CST`。

当前状态：H 已完成 `iter_003` update 与 `iter_004` rollout，正在 `iter_004` update；尚未发现 traceback / NaN / OOM。

| iter | gate tool | gate memory | gate code | overall sample reward | tool reward | memory reward | code reward | all-fail tool/memory/code | all-success tool/memory/code | OPD rows tool/memory/code | retention rows | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|
| 1 | 0.3318 | 0.3447 | 0.3354 | 0.3595 | 0.3984 | 0.3203 | 0.3598 | 16 / 12 / 14 | 0 / 3 / 0 | 11 / 10 / 9 | 3 | 首轮能推动 gate，memory 上升最快但未过冲 |
| 2 | 0.3307 | 0.3561 | 0.3382 | 0.4461 | 0.4915 | 0.4297 | 0.4170 | 11 / 8 / 12 | 2 / 3 / 1 | 7 / 7 / 7 | 6 | 三任务 proxy 都涨，OPD rows 保持均衡，all-fail 明显下降 |
| 3 | 0.3304 | 0.3684 | 0.3414 | 0.4213 | 0.5208 | 0.3438 | 0.3994 | 10 / 8 / 11 | 3 / 1 / 1 | 5 / 5 / 6 | 5 | update 继续推高 memory/code，tool 基本不动；本轮 proxy 中 memory/code 回落 |
| 4 rollout | 0.3304 | 0.3684 | 0.3414 | 0.5434 | 0.7852 | 0.3828 | 0.4623 | 10 / 8 / 12 | 10 / 1 / 4 | 5 / 6 / 7 | pending | proxy 明显上升，主要来自 Tool；Code 同步回升，Memory 仍弱于 iter2 |

阶段判断：

- H 当前训练动态仍正常：`overall sample reward 0.3595 -> 0.4461 -> 0.4213 -> 0.5434`，iter4 创当前最高 proxy。
- gate 方向持续偏 memory/code：`memory 0.3333 -> 0.3684`，`code 0.3333 -> 0.3414`，`tool 0.3333 -> 0.3304`；但 iter4 Tool reward 大幅上升，说明 tool 能力没有随系数小幅下降而崩。
- OPD 信号从 `30 -> 21 -> 16 -> 18` rows，iter4 的 `tool/memory/code=5/6/7` 仍基本均衡。下一轮重点看 iter4 update 后是否继续推 memory/code，以及 Tool all-success 增多后 retention 是否稳定。
