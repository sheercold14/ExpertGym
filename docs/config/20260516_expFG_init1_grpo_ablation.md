# 2026-05-16 F/G 实验配置：init=1.0 的 GRPO 消融

## 目标

旧 F/G 实验废弃，不复用其 run dir。本组从 task-vector 系数 `1.0` 初始化出发，检验强 task-vector 起点下，GRPO 是否能维持/修复整体能力，以及 OPD 是否仍提供额外收益。

| 实验 | run dir | 变量 |
|---|---|---|
| F | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_init1_grpo_ret_20260516` | `GRPO + retention`，关闭 OPD |
| G | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_init1_grpo_opd_ret_20260516` | `GRPO + dynamic OPD + retention` |

## 公共设置

| 项 | 值 |
|---|---|
| gate strategy | `global-parameter` |
| init checkpoint | `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/init_gates/init_global_parameter_c1.json` |
| init effective coefficient | `tool=1.0, memory=1.0, code=1.0` for every mergeable module |
| prompts | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| prompts count | `96 = tool 32 + memory 32 + code 32` |
| samples per prompt | `4` |
| iterations | `20` |
| optimizer | SGD, momentum `0.2`, persisted state |
| lr | `0.1876` |
| loss granularity | `sequence` |
| update | epoch-scope step, `UPDATE_BATCH_SIZE=4` only slices accumulation |
| GRPO | enabled, `PPO_LOSS_WEIGHT=1.0` |
| frontier quotas | `tool=32, memory=32, code=32` |
| retention | enabled, NLL, task-balanced row scale |
| prior | `0.0` |
| max coefficient delta | `1.0` |

## F

启动脚本：

```text
skill/command/run_20260516_expF_init1_grpo_ret.sh
```

关键差异：

```text
OPD_LOSS_WEIGHT=0.0
DYNAMIC_OPD_EXPERT_ROLLOUT=
```

## G

启动脚本：

```text
skill/command/run_20260516_expG_init1_grpo_opd_ret.sh
```

OPD expert pool 与之前 B 一致：

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl
```

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expF_init1_20260516 \
  'GPU_LIST=2,3 bash skill/command/run_20260516_expF_init1_grpo_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_init1_grpo_ret_20260516/train.log'

tmux new -d -s train_expG_init1_20260516 \
  'GPU_LIST=6,7 bash skill/command/run_20260516_expG_init1_grpo_opd_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_init1_grpo_opd_ret_20260516/train.log'
```

## 监控重点

- iter1 proxy 是否已经崩：尤其 tool 格式与 memory final answer。
- gate 是否从 `1.0` 被快速拉回，还是在强 task-vector 区域稳定。
- F/G 的差异：如果 G 明显更稳，说明 init=1 下 OPD 仍有修复作用；如果 F 更稳，说明 OPD 可能在强起点下引入多任务冲突。
