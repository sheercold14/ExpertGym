# 2026-05-14 Balanced GRPO + Dynamic OPD + NLL Retention

## 目标

用 96 条 paper96 calibration prompts，从 `1/3` task-vector checkpoint 出发，比较四种 gate 参数化能否在保持 Tool/Memory/Code 三种 proxy reward 的同时，把 task-vector coefficient 推到更强能力区间。

## 共享设置

| 参数 | 值 |
|---|---:|
| prompts | 96 |
| samples per prompt | 4 |
| iterations | 15 |
| GRPO weight | 3.0 |
| dynamic OPD NLL weight | 3.0 |
| OPD pairwise weight | 0.0 |
| retention objective | NLL preservation |
| retention weight | 0.5 |
| policy logprob length norm | on |
| OPD/retention logprob length norm | on |
| optimizer | SGD |
| momentum | 0.2 |
| optimizer step scope | epoch |
| dynamic OPD quota | 8 rows per task |
| retention cap | 8 rows per task, 24 total |

## 实验矩阵

| Run | GPU | Strategy | LR | Prior | Max Delta | 目的 |
|---|---|---|---:|---:|---:|---|
| A | 0,1 | `global-coefficient` | 0.03 | 0.0 | 0.55 | 直接学习三个 expert 系数 |
| B | 2,3 | `global` | 0.03 | 0.0 | 0.55 | common + residual，测试共同上移与任务差异 |
| C | 4,5 | `layer-band` | 0.02 | 0.005 | 0.40 | 分层组控制，测试层段组合是否更稳 |
| D | 6,7 | `global-parameter` | 0.015 | 0.005 | 0.40 | global strength + 参数 residual，近似细粒度 588 |

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
RUN_TAG=20260514_balanced_w3_ret05_i15_v1 \
MONITOR_PORT=8782 \
  bash skill/command/run_balanced_grpo_opd_retention_20260514.sh
```

## 运行目录

| Run | Directory |
|---|---|
| A | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/bal15_A_gc_w3_ret05_m02_20260514_balanced_w3_ret05_i15_v1` |
| B | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/bal15_B_global_w3_ret05_m02_20260514_balanced_w3_ret05_i15_v1` |
| C | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/bal15_C_layerband_w3_ret05_m02_20260514_balanced_w3_ret05_i15_v1` |
| D | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/bal15_D_gp_w3_ret05_m02_20260514_balanced_w3_ret05_i15_v1` |

## 监控

- Frontend: `http://127.0.0.1:8782`
- Supervisor status: `docs/report/opvec_bal15_balanced_w3_ret05_i15_v1_supervisor.md`
- Supervisor JSON: `docs/report/opvec_bal15_balanced_w3_ret05_i15_v1_supervisor.json`
- Supervisor session: `opvec_supervisor_bal15_20260514_balanced_w3_ret05_i15_v1`

Supervisor 使用 `reward_train` 口径监控，避免 Tool raw reward `[-3,4]` 扭曲整体曲线。若第 5-8 轮出现明显崩盘，它会停止对应 tmux session 并写入 `stopped_by_supervisor.json`。

## Iter1 Proxy Reward

| Run | Overall | Tool | Memory | Code |
|---|---:|---:|---:|---:|
| A | 0.3818 | 0.4129 | 0.3438 | 0.3887 |
| B | 0.3871 | 0.3985 | 0.3594 | 0.4033 |
| C | 0.4459 | 0.5182 | 0.3984 | 0.4209 |
| D | 0.4002 | 0.4173 | 0.3438 | 0.4395 |

## 自动评测

所有 run 完成或被 supervisor 停止后，脚本会选择每个 run 中 proxy balanced score 最高的已 rollout `baked_policy`，生成：

`/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_bal15_balanced_w3_ret05_i15_v1_addons.py`

并在 tmux session `eval_bal15_balanced_w3_ret05_i15_v1` 中启动 eval6。
