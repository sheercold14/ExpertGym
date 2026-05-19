# 2026-05-18 P0 SOTA Calibration V2 Config

## 目标

P0 主线从“补表”切到“把核心方法推到 SOTA 合理区间”：

- Tool：BFCL live 波动大，不再只追 live；加入 ToolRL `rlla_4k/test` all80 overall correct，按做对题目比例统计。
- Memory：用 HotpotQA train 的 trajectory-style prompts 训练，不再只靠 final answer proxy。
- Code：用 CodeContests train 构造 CURE-like executable probes，显式覆盖 generation / selection / partial-edge 能力。

## 数据产物

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/
  train128.prompts.jsonl
  monitor64.prompts.jsonl
  guard64.prompts.jsonl
  summary.json
  components/
    tool_code_pool/
    memory_pool.prompts.jsonl
```

计数：

| split | tool | memory | code | total |
|---|---:|---:|---:|---:|
| train128 | 32 | 48 | 48 | 128 |
| monitor64 | 16 | 24 | 24 | 64 |
| guard64 | 16 | 24 | 24 | 64 |

分层：

- Tool train：16 paper96 ToolRL/RLLA source anchors + 16 fresh BFCL-style synthetic probes。
- Code train：16 paper96 Code frontier anchors + 32 CodeContests-train CURE-style targeted probes。
- Memory train：48 HotpotQA train trajectory prompts。

## 构建命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

bash skill/command/build_20260518_sota_calib_v2.sh
```

验证已完成：

```text
train128:  tool=32, memory=48, code=48
monitor64: tool=16, memory=24, code=24
guard64:   tool=16, memory=24, code=24
```

## Expert Rollout 命令

训练前需要给 `train128` 生成同 prompt expert trajectories，用于 dynamic OPD。

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

# 可分卡并行跑；也可以 POLICY=all 顺序跑。
POLICY=tool     GPU_LIST=0 bash skill/command/run_20260518_sota_v2_expert_rollouts.sh
POLICY=memory   GPU_LIST=1 bash skill/command/run_20260518_sota_v2_expert_rollouts.sh
POLICY=code     GPU_LIST=2 bash skill/command/run_20260518_sota_v2_expert_rollouts.sh
POLICY=deepseek GPU_LIST=3 bash skill/command/run_20260518_sota_v2_expert_rollouts.sh
```

输出：

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/
```

每个 rollout 会额外写 `.coverage.json`，用于检查 expert 对 train128 的 same-prompt positive 覆盖率。若某任务 coverage 低，先补 expert samples 或换 prompt，不直接进入主训练。

## 主实验

推荐先跑两个强 prior 版本。默认 `INIT_VALUE=1.0`，因为当前目标是先把 Memory/Code 能力表达出来，再由 executable feedback 和 retention 控制退化。若要做更保守对照，显式设 `INIT_VALUE=0.75`。

### Recoverable-Code 当前主线

第一轮 expert coverage 显示 full `train128` 的 Code targeted rows 过硬，因此当前实际启动主线使用 recoverable-code train split：

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl
```

计数：

| task | rows |
|---|---:|
| Tool | 32 |
| Memory | 48 |
| Code | 21 |

启动前建议先创建日志目录，避免 `tee` 早于 wrapper 内部 `mkdir`：

```bash
mkdir -p /tmp/shared-storage/OnPolicy/runs/gated_grpo/sota_v2_recoverable101_gc_init1_grpo_opd_ret_20260518
mkdir -p /tmp/shared-storage/OnPolicy/runs/gated_grpo/sota_v2_recoverable101_gp_init1_grpo_opd_ret_20260518
```

已启动的两个 8-iter runs：

```bash
CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl \
NUM_PROMPTS=101 \
NUM_ITERS=8 \
RUN_NAME=sota_v2_recoverable101_gc_init1_grpo_opd_ret_20260518 \
PHASE=train_gc \
GPU_LIST=2,3 \
bash skill/command/run_20260518_p0_sota_v2.sh

CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl \
NUM_PROMPTS=101 \
NUM_ITERS=8 \
RUN_NAME=sota_v2_recoverable101_gp_init1_grpo_opd_ret_20260518 \
PHASE=train_gp \
GPU_LIST=4,5 \
bash skill/command/run_20260518_p0_sota_v2.sh
```

### GC：global coefficient，3 个 expert 系数

```bash
PHASE=train_gc \
GPU_LIST=0,1 \
bash skill/command/run_20260518_p0_sota_v2.sh
```

### GP：global-parameter/common+residual 风格

```bash
PHASE=train_gp \
GPU_LIST=2,3 \
bash skill/command/run_20260518_p0_sota_v2.sh
```

关键超参：

| item | value | reason |
|---|---:|---|
| prompts | 128 | 覆盖 Tool/Memory/Code，同时仍可 5h 内迭代 |
| samples/prompt | 4 | 保持 GRPO frontier 信号 |
| frontier rows/task | 4 random | 控制 GRPO update 成本，避免长任务主导 |
| retention rows/task | 8 random | 保留 all-success 行为 |
| OPD | dynamic all-fail only | 只在当前 policy 失败、expert 成功时施加 |
| OPD/retention length norm | on | 降低 Memory/Code 长轨迹长度主导 |
| task-balanced OPD/retention | on | 三任务按任务平均，不按行数或 token 数压制 |
| OPD/retention dynamic scale | on | 减少手调 loss weight，按当前 loss 量级自适应 |
| optimizer | SGD momentum 0.2 | 延续当前最稳 gate 更新路径 |
| step scope | epoch | 每轮只更新一次 gate，降低小 batch 抖动 |

## ToolRL Test 评测

任意 baked checkpoint：

```bash
MODEL_PATH=/path/to/baked_policy \
RUN_ID=my-model-toolrl-all80 \
GPU_LIST=0 \
bash skill/command/run_20260518_toolrl_rlla4k_eval.sh
```

主读：

```text
task_stats.tool.success_rate
```

这是 `ToolRL rlla_4k/test` all80 做对比例，不按子类平均。

## Stop / Promote 规则

每 2-3 iter 读：

- train reward 是否上升；
- monitor64 是否同向上升；
- ToolRL all80 是否保持；
- gate 是否只靠单任务过冲。

送正式 Eval6 的最低条件：

- Memory monitor 有明显正趋势；
- Code monitor 至少 generation 或 selection 一项改善；
- ToolRL all80 不显著低于 TA/static reference；
- BFCL live 只作为诊断，不因单个 live 子类波动否决模型。
