# ExpertGym Harness

本目录是 ExpertGym 论文收束期的实验总控入口。目标不是把所有可能实验都跑完，而是在 72 小时内做出能支撑论文主 claim 的核心证据链。

## 核心论文句子

```text
Task vectors provide structured priors; executable feedback learns their composition.
```

实验必须服务这个句子。每个实验都要回答以下至少一个问题：

| 问题 | 对应 claim | 实验优先级 |
|---|---|---:|
| task vector 本身是不是有可组合的先验能力？ | E1 prior table | P0 |
| 为什么纯 GRPO 慢/弱？ | executable credit collapse / state distribution | P0 |
| executable feedback 是否比静态几何或纯 imitation 更有用？ | E2/E5 main defense | P1 |
| state-conditioned routing 是否比 loss mixing 更合理？ | E4 routing | P1 |
| global / layer / module 系数空间是否提供额外收益？ | E7 capacity ladder | P2 |
| Code/Tool 的 formal eval 短板能否被 calibration 反映？ | E6 diagnostic | P2 |
| 混合 agent 能力是否有组合泛化？ | E8 optional | P3 |

## 文件索引

| 文件 | 用途 |
|---|---|
| [20260517_72h_master_plan.md](20260517_72h_master_plan.md) | 72 小时科研执行计划、GPU 调度、晋级规则 |
| [experiment_matrix.md](experiment_matrix.md) | 主实验矩阵、每个实验和论文表格的对应关系 |
| [runbook_5h_fast_iteration.md](runbook_5h_fast_iteration.md) | 单实验 5 小时内的训练设置、命令模板、停止规则 |
| [calibration_design.md](calibration_design.md) | reward-aware calibration bank 设计、OPD target 选择、heldout 防过拟合规则 |
| [reviewer_risks_and_baselines.md](reviewer_risks_and_baselines.md) | 审稿风险、必须 baseline、防止被质疑为 imitation/sweep/loss mixing |
| [subagent_tasks.md](subagent_tasks.md) | subagent 分工与交付物 |

## 目录约定

正式实验文档：

```text
docs/config/YYYYMMDD_<run_family>.md
```

论文实验报告：

```text
docs/report/expertgym_72h/
```

正式评测主表：

```text
docs/evaluation/
```

长期记忆：

```text
docs/memory/expertgym_72h/
```

训练产物：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/<run_name>/
```

所有正式启动的 run 必须满足：

- 有唯一 `RUN_NAME` 和 `RUN_DIR`。
- 有 `docs/config` 配置记录。
- `train.log`、`gated_grpo_bake_vllm_loop_manifest.json`、`strategy_summary.json` 可回溯。
- 送评模型必须写入 `docs/evaluation`，并链接原始 `summary.json`。
- 不在仓库内写临时 rollout、checkpoint、缓存或无归属脚本。
