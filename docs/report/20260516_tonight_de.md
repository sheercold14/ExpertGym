# 2026-05-16 晚间 D/E 实验报告

## 实验设置

配置文档：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260516_tonight_de.md`

| 实验 | run dir | 变量 |
|---|---|---|
| D | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expD_gp_code_aug_memory_grpo_20260516` | B 配置 + code OPD augmentation + memory-only GRPO |
| E | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expE_gp_code_aug_all_grpo_20260516` | B 配置 + code OPD augmentation + tool/memory/code GRPO |

公共训练数据：`/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl`

OPD expert pool：旧三专家 rollout + `/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/` 下四个 code augmentation rollout。

## 启动记录

更新时间：`2026-05-16 16:36 CST`。

| 实验 | tmux | GPU | 状态 |
|---|---|---|---|
| D | `train_expD_20260516` | `0,1` | 已启动 |
| E | `train_expE_20260516` | `4,5` | 已启动 |

启动命令：

```bash
tmux new -d -s train_expD_20260516 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=0,1 PHASE=train_d bash skill/command/run_20260516_tonight_de.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expD_gp_code_aug_memory_grpo_20260516/train.log'

tmux new -d -s train_expE_20260516 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=4,5 PHASE=train_e bash skill/command/run_20260516_tonight_de.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expE_gp_code_aug_all_grpo_20260516/train.log'
```

## 对照关系

| 对照 | 差异 |
|---|---|
| B vs D | D 在 B 的基础上加入 memory-only GRPO |
| A vs E | E 在 A 的基础上加入 code OPD augmentation |
| D vs E | 判断 GRPO 信号只给 memory 还是三任务全开更合理 |

## 待填结果

| run | best proxy iter | best overall | tool | memory | code | best global tool | best global memory | best global code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D | pending | pending | pending | pending | pending | pending | pending | pending |
| E | pending | pending | pending | pending | pending | pending | pending | pending |

## 正式评测

训练完成后选择各自 proxy overall 最高的 `iter_*/baked_policy` 送入 eval6。评测结果统一记录到：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/
```
