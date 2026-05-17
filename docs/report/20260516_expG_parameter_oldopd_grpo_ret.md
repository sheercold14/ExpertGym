# 2026-05-16 G 实验报告：parameter gate + old OPD + GRPO + retention

配置文档：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260516_expG_parameter_oldopd_grpo_ret.md`

## 实验设置

| 项 | 值 |
|---|---|
| run | `expG_param_oldopd_grpo_ret_20260516` |
| run dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_param_oldopd_grpo_ret_20260516` |
| gate | `parameter`，588 个 direct module-level coefficients |
| loss | `GRPO + dynamic OPD + retention` |
| OPD source | old tool / old memory / old code expert rollouts only |
| code augmentation | disabled |
| GPU | `6,7` |

## 启动记录

更新时间：`2026-05-16 21:20 CST`。

| tmux | 状态 |
|---|---|
| `train_expG_20260516` | 已启动，首轮 vLLM rollout 中 |
| `opvec_monitor_g_20260516` | 已启动，端口 `8791` |

前端：

```text
http://10.119.31.17:8791
```

启动命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expG_20260516 \
  'GPU_LIST=6,7 bash skill/command/run_20260516_expG_parameter_oldopd_grpo_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expG_param_oldopd_grpo_ret_20260516/train.log'
```

## 待填训练结果

| run | best proxy iter | best overall | tool | memory | code | best mean gate tool | best mean gate memory | best mean gate code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G | pending | pending | pending | pending | pending | pending | pending | pending |

## 待检查项

- `gate_parameterization` 已确认：`parameter`。
- `parameter_coefficients` 已确认：首轮 bake 输出 `num_delta_entries=588`。
- `dynamic OPD expert_rollouts` 已确认：只有旧 tool / memory / code 三专家，无 code augmentation 路径。
- `frontier_task_counts` 已确认：`tool=32, memory=32, code=32`。
- 相比 B/F，code reward 是否更稳定提升；相比 D/E，588 direct coefficients 是否更容易过拟合或冲突。

## 首轮状态

`iter_001` bake 完成，用时 `52.4s`。当前启动 2 个 vLLM rollout shard：

| shard | GPU | prompt offset | prompts |
|---|---:|---:|---:|
| 00 | 6 | 0 | 48 |
| 01 | 7 | 48 | 48 |

rollout 使用 baked policy，但 gate checkpoint 仍记录为 `init_parameter_c033333.json`，符合首轮从 `1/3` 初始化出发的设计。
