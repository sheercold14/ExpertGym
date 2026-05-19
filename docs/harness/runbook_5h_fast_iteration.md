# 5h Fast Iteration Runbook

## 设计原则

每个训练实验必须能在 5 小时内给出可判断结果。完整 full-GRPO 每轮约 30 分钟，不适合作为快速搜索；fast setting 通过随机控制 frontier / retention 行数来保留论文信号，同时控制 update 时间。

## 推荐 fast override

```bash
NUM_ITERS=12
NUM_PROMPTS=96
SAMPLES_PER_PROMPT=4

UPDATE_BATCH_SIZE=8
BATCH_LOSS_REDUCTION=mean
OPTIMIZER_STEP_SCOPE=epoch
LOSS_GRANULARITY=sequence

FRONTIER_ROWS_PER_TASK=4
FRONTIER_SAMPLE_BEFORE_LIMIT=1
FRONTIER_ORDER=task-interleaved

MAX_RETENTION_ROWS_PER_TASK=8
MAX_RETENTION_ROWS=24
RETENTION_SAMPLE_BEFORE_LIMIT=1

STORE_TOKEN_LOGPROBS=0
TASK_NORMALIZE_ADVANTAGES=0
ADVANTAGE_NORMALIZATION=centered
USE_FRONTIER_WEIGHT=0

OPD_TASK_BALANCED_LOSS_SCALE=1
RETENTION_TASK_BALANCED_LOSS_SCALE=1
OPD_LENGTH_NORMALIZE_LOGPROB=1
RETENTION_LENGTH_NORMALIZE_LOGPROB=1

OPTIMIZER=sgd
SGD_MOMENTUM=0.2
PERSIST_OPTIMIZER_STATE=1
PRIOR_LOSS_WEIGHT=0.0
MAX_COEFF_DELTA=1.0
```

解释：

- `FRONTIER_ROWS_PER_TASK=4`：GRPO 仍有三任务 frontier direction，但不让 update 被 30-40 条长序列拖垮。
- `MAX_RETENTION_ROWS_PER_TASK=8`：stable rows 保持 boundary credit，但不主导 update。
- 随机采样开启：避免固定选文件前几条造成偏置。
- 不设置这些上限时 wrapper 默认 full frontier / full retention。

## global-coefficient 主方法模板

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

RUN_NAME=eg72_main_gc_c033_fast_YYYYMMDD \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_fast_YYYYMMDD \
GPU_LIST=0,1 \
STRATEGY=global-coefficient \
INIT_VALUE=0.3333333333333333 \
PPO_LOSS_WEIGHT=1.0 \
OPD_LOSS_WEIGHT=1.0 \
USE_RETENTION=1 \
RETENTION_OBJECTIVE=nll \
RETENTION_LOSS_WEIGHT=0.5 \
DYNAMIC_OPD_EXPERT_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl \
NUM_ITERS=12 \
NUM_PROMPTS=96 \
SAMPLES_PER_PROMPT=4 \
UPDATE_BATCH_SIZE=8 \
FRONTIER_ROWS_PER_TASK=4 \
FRONTIER_SAMPLE_BEFORE_LIMIT=1 \
MAX_RETENTION_ROWS_PER_TASK=8 \
MAX_RETENTION_ROWS=24 \
RETENTION_SAMPLE_BEFORE_LIMIT=1 \
OVERWRITE=1 \
bash skill/command/run_qbank_c033333_gate_strategy.sh
```

## OPD-only recovery模板

```bash
PPO_LOSS_WEIGHT=0.0
OPD_LOSS_WEIGHT=1.0
USE_RETENTION=1
RETENTION_OBJECTIVE=nll
RETENTION_LOSS_WEIGHT=0.5
```

用途：证明 recoverable same-prompt expert success 能推动系数，是 early-stage recovery。

## GRPO-only frontier模板

```bash
PPO_LOSS_WEIGHT=1.0
OPD_LOSS_WEIGHT=0.0
USE_RETENTION=1
RETENTION_OBJECTIVE=nll
RETENTION_LOSS_WEIGHT=0.5
```

用途：证明 frontier direction 存在，但如果 frontier 稀疏，单独 GRPO 进展慢。

## 启动规范

每个 run 先写 config 文档：

```text
docs/config/YYYYMMDD_<run_name>.md
```

再 dry-run：

```bash
DRY_RUN=1 <env...> bash skill/command/run_qbank_c033333_gate_strategy.sh
```

再 tmux：

```bash
tmux new -d -s train_<run_name> \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && <env...> bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/<run_name>/train.log'
```

监控：

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/monitor/opvec_run_monitor.py \
  --run-dir <label>=/tmp/shared-storage/OnPolicy/runs/gated_grpo/<run_name> \
  --init-value <0.3333333333333333 or 1.0> \
  --host 127.0.0.1 \
  --port <free_port>
```

## 每轮检查清单

每完成一个 iter 检查：

```text
iter_XXX/rollouts.summary.json
iter_XXX/opd_distill_from_allfail.summary.json
iter_XXX/gate_updates.summary.json
iter_XXX/gate_updates.gates.json
```

必须记录：

- overall / task proxy reward。
- kept frontier rows by task。
- OPD rows by task。
- retention rows。
- gate coefficient movement。
- grad norm / gate delta。
- iteration timing: bake / collect / update。

## 5 小时内决策

第 3-4 轮：

- 如果 reward 不动且 frontier/OPD 都低，停止，换数据/起点。
- 如果只有一个任务涨，观察 retention/worst-task drop。
- 如果 code/tool formal proxy 与 expected mismatch，转 P2 diagnostic，不继续主线。

第 8-12 轮：

- 选 best proxy checkpoint。
- 如果相对首轮 overall +0.08 以上且 worst-task 不崩，送 eval。
- 如果 proxy 高但正式短板已知严重，只送一个代表，不要浪费 eval 资源。

