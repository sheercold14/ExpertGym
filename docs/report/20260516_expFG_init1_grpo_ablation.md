# 2026-05-16 F/G 实验报告：init=1.0 的 GRPO 消融

配置文档：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260516_expFG_init1_grpo_ablation.md`

## 实验设置

| 实验 | loss | run dir | GPU |
|---|---|---|---|
| F | `GRPO + retention` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_init1_grpo_ret_20260516` | `2,3` |
| G | `GRPO + OPD + retention` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_init1_grpo_opd_ret_20260516` | `6,7` |

两组均使用 `global-parameter` gate，从 `init_global_parameter_c1.json` 出发。旧 F/G 实验不再作为当前 F/G。

## 启动记录

更新时间：`2026-05-16 CST`。

| 项 | 状态 |
|---|---|
| init checkpoint | 已生成并验证，effective coefficients 全为 `1.0` |
| F dry-run | 已通过：`PPO_LOSS_WEIGHT=1.0`，`OPD_LOSS_WEIGHT=0.0`，`dynamic_opd_rollout=none` |
| G dry-run | 已通过：`PPO_LOSS_WEIGHT=1.0`，`OPD_LOSS_WEIGHT=1.0`，使用 B 的 augmented OPD expert pool |
| F training | 已启动，tmux `train_expF_init1_20260516` |
| G training | 已启动，tmux `train_expG_init1_20260516` |
| frontend | 已启动，tmux `opvec_monitor_fg_init1_20260516`，端口 `8790` |

前端：

```text
http://10.119.31.17:8790
```

启动命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expF_init1_20260516 \
  'GPU_LIST=2,3 bash skill/command/run_20260516_expF_init1_grpo_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_init1_grpo_ret_20260516/train.log'

tmux new -d -s train_expG_init1_20260516 \
  'GPU_LIST=6,7 bash skill/command/run_20260516_expG_init1_grpo_opd_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_init1_grpo_opd_ret_20260516/train.log'
```

## 待填训练结果

| run | best proxy iter | best overall | tool | memory | code | best gate tool | best gate memory | best gate code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | pending | pending | pending | pending | pending | pending | pending | pending |
| G | pending | pending | pending | pending | pending | pending | pending | pending |

## 初期判断标准

- 若 iter1/2 overall 明显低于 `0.4` 且 all-fail 暴涨，说明 `init=1.0` 过强。
- 若 F 能稳定，GRPO 在强起点可以单独维持能力。
- 若 G 高于 F，OPD 在强起点仍有额外修复价值。
- 若 G 低于 F，OPD 可能在强起点下引入专家轨迹冲突。
