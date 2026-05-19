# 2026-05-18 SOTA Recovery Calibration v3

## 目标

以正式 SOTA 为核心重新组织 calibration data：训练集优先保证可学习信号密度，monitor/guard 保留 harder probes，避免 train proxy 上涨但 formal eval 不涨。

核心原则：

```text
train = verified-recoverable / high-gradient rows
monitor/guard = harder audit rows
```

## 构建命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

bash skill/command/build_20260518_sota_recovery_calib_v3.sh
```

输出：

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_recovery_calib_v3_20260518/
  train128.prompts.jsonl
  monitor64.prompts.jsonl
  guard64.prompts.jsonl
  summary.json
  README.md
  init_gates/
```

## Split 设计

| split | tool | memory | code | 目的 |
|---|---:|---:|---:|---|
| train128 | 32 | 48 | 48 | 高密度 OPD/GRPO/retention 训练 |
| monitor64 | 16 | 24 | 24 | 快速 early-stop / checkpoint selection |
| guard64 | 16 | 24 | 24 | 防止 train/monitor 过拟合 |

训练集来源：

| task | rows | 来源 |
|---|---:|---|
| Tool | 32 | `sota_calib_v2_recoverable_code_20260518` 中 Tool anchor |
| Memory | 48 | `sota_calib_v2_recoverable_code_20260518` 中 HotpotQA memory anchor |
| Code | 36 | `code_p0_v3_20260518/train_recoverable_code` |
| Code | 12 | `sota_calib_v2_recoverable_code_20260518` 中 recoverable code anchor |

monitor/guard 的 Code 主要来自 Code P0 hard audit：

```text
20 Code P0 audit + 4 sota_v2 eval-target anchor
```

## 训练入口

```bash
# 推荐主实验：hierarchical layer-band
PHASE=train_hier GPU_LIST=4,5 \
  bash skill/command/run_20260518_sota_recovery_v3.sh

# 对照：global coefficient
PHASE=train_gc GPU_LIST=0,1 \
  bash skill/command/run_20260518_sota_recovery_v3.sh

# 对照：global parameter
PHASE=train_gp GPU_LIST=2,3 \
  bash skill/command/run_20260518_sota_recovery_v3.sh
```

默认关键设置：

```text
INIT_VALUE=1.0
NUM_ITERS=12
NUM_PROMPTS=128
SAMPLES_PER_PROMPT=4
FRONTIER_ROWS_PER_TASK=4
MAX_RETENTION_ROWS_PER_TASK=8
OPD_DYNAMIC_SCALE=1
OPD_TASK_BALANCED_LOSS_SCALE=1
RETENTION_OBJECTIVE=nll
RETENTION_SCALE_TARGET=0.5
TASK_NORMALIZE_ADVANTAGES=0
ROLLOUT_SHARDS=auto
```

v3 训练脚本显式接入 6 组 expert rollouts：

```text
ToolRL expert on sota_v2 train128
RL-MemoryAgent expert on sota_v2 train128
ReasonFlux expert on sota_v2 train128
DeepSeek-R1-Distill expert on sota_v2 train128
ReasonFlux expert on Code P0 train64
DeepSeek-R1-Distill expert on Code P0 train64 merged shards
```

## Init Gate 隔离

`run_20260518_sota_recovery_v3.sh` 设置：

```text
QB=/tmp/shared-storage/OnPolicy/data/calibration/sota_recovery_calib_v3_20260518
```

因此 init gate 写入：

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_recovery_calib_v3_20260518/init_gates/
```

不会覆盖公共 question-bank 的 c033333 init gate。公共 c033333 layer-band-parameter init 已恢复为 `0.3333333333333333`。

## 选择规则

训练中不以 train reward 单独选模型。候选 checkpoint 必须满足：

1. train overall reward 上升；
2. monitor64 Code 不下降，最好同步上升；
3. ToolRL all80 不明显低于 TA-0.75；
4. Memory monitor 不牺牲；
5. gate movement 可解释，不能只把某个 task vector 推到极端。

若 train code 上涨但 monitor/guard code 不涨，说明仍是 calibration proxy mismatch，不进入正式 Eval6。
