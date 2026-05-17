# 2026-05-16 F 实验报告：code augmentation only OPD

配置文档：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260516_expF_code_aug_opd_only.md`

## 实验设置

| 项 | 值 |
|---|---|
| run | `expF_gp_code_aug_only_code_opd_20260516` |
| run dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_gp_code_aug_only_code_opd_20260516` |
| 对照 | 新一轮 B：`expB_gp_code_opd_aug_20260516` |
| 变量 | dynamic OPD 只使用四个 code augmentation rollout，且只对 code task 生效 |
| GRPO | disabled, `PPO_LOSS_WEIGHT=0.0` |
| retention | enabled, all-success NLL |
| gate | `global-parameter`, init `1/3` |
| GPU | `2,3` |

## 启动记录

更新时间：`2026-05-16 21:04 CST`。

| tmux | 状态 |
|---|---|
| `train_expF_20260516` | 已启动 |

启动命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expF_20260516 \
  'GPU_LIST=2,3 bash skill/command/run_20260516_expF_code_aug_opd_only.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_gp_code_aug_only_code_opd_20260516/train.log'
```

## 待填训练结果

## 初始校验

记录时间：`2026-05-16 21:14 CST`。

| 项 | 值 | 说明 |
|---|---:|---|
| iter1 rollout rows | `96` | 三任务各 32 prompt，4 samples/prompt |
| iter1 rollout elapsed | `362.2s` | 两卡 vLLM shard |
| dynamic OPD tasks | `code` | 符合 F 设置 |
| dynamic OPD selected rows | `7` | 全部为 code |
| dynamic OPD selected_task_counts | `code=7` | 无 tool/memory OPD |
| skipped task_filtered | `64` | tool/memory prompt 被按任务过滤 |
| skipped no_expert_positive | `5` | code all-fail 中仍有 5 条无 augmentation positive |
| update | running | iter1 HF update 正在执行 |

| run | best proxy iter | best overall | tool | memory | code | best global tool | best global memory | best global code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | pending | pending | pending | pending | pending | pending | pending | pending |

## 待检查项

- `opd_distill_task_counts` 是否只包含 `code`。
- `frontier_task_counts` 是否为空或全 0。
- code gate 是否相较 B 更直接上涨。
- code reward 是否真正持续提升，而不是只在早期波动。
- tool/memory 是否因为缺少 OPD/GRPO 只靠 retention 而退化。
