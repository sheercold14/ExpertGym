# 2026-05-16 G 实验配置：parameter gate + old OPD + GRPO + retention

## 目标

G 在 B 的训练数据与基础超参上，测试更高自由度的 gate 参数化：不使用 `global-parameter = global + residual`，而是直接学习每个 mergeable module 上每个 expert 的系数，即 `parameter` 策略。损失使用 `OPD + GRPO + retention`，并且不使用 code augmentation expert rollouts，只用旧三专家 rollout。

## 核心设置

| 项 | 值 |
|---|---|
| run | `expG_param_oldopd_grpo_ret_20260516` |
| run dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_param_oldopd_grpo_ret_20260516` |
| launcher | `skill/command/run_20260516_expG_parameter_oldopd_grpo_ret.sh` |
| gate strategy | `parameter` |
| trainable coefficients | `196 modules × 3 experts = 588` direct coefficients |
| init | all coefficients `1/3` |
| prompts | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| samples per prompt | `4` |
| iterations | `20` |
| GRPO | enabled, `PPO_LOSS_WEIGHT=1.0` |
| frontier quota | `tool=32, memory=32, code=32` |
| OPD | dynamic all-fail current rows + old expert positives |
| OPD expert pool | old tool / old memory / old code only |
| code augmentation | disabled |
| retention | enabled, all-success NLL |
| optimizer | SGD, momentum `0.2`, epoch-scope step |
| lr | `0.1876`, kept from B-style setting for controlled comparison |
| prior | `0.0` |
| max coefficient delta | `1.0` |
| GPU | `6,7` |

## OPD Expert Pool

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
```

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expG_20260516 \
  'GPU_LIST=6,7 bash skill/command/run_20260516_expG_parameter_oldopd_grpo_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_param_oldopd_grpo_ret_20260516/train.log'
```

Dry run：

```bash
DRY_RUN=1 GPU_LIST=6,7 bash skill/command/run_20260516_expG_parameter_oldopd_grpo_ret.sh
```

## 对照点

- 对 B：G 不用 code augmentation，并将 loss 改成 `OPD + GRPO + retention`。
- 对 D/E：G 不用 `global-parameter`，改为 direct `parameter`。
- 对 F：F 只做 code augmentation code OPD；G 做三任务旧 OPD + 三任务 GRPO。
