# 2026-05-15 OPD-only + Retention + PCGuard ABCD 实验

## 超参数置顶

| 项 | A | B | C | D |
|---|---|---|---|---|
| 目的 | OPD-only 推 `global-coefficient` | OPD-only 推 `global-parameter` | A + PCGuard | B + PCGuard |
| strategy | `global-coefficient` | `global-parameter` | `global-coefficient` | `global-parameter` |
| 可学习 gate | 4 个全局 task coefficients | 588 个 common+residual 参数 | 同 A | 同 B |
| GPU | `0,1` | `2,3` | `4,5` | `6,7` |
| prompts | `paper96`，三任务各 32 | 同 A | 同 A | 同 A |
| samples/prompt | `4` | `4` | `4` | `4` |
| iterations | `15` | `15` | `15` | `15` |
| init | `1/3` task-vector | 同 A | 同 A | 同 A |
| update | `epoch-scope`，每轮一个 optimizer step | 同 A | 同 A | 同 A |
| loss | `OPD NLL + all-success retention NLL`，不跑 GRPO frontier loss | 同 A | 同 A | 同 A |
| OPD source | 每轮从当前 policy `all_fail` prompt 动态选 expert-positive 轨迹 | 同 A | 同 A | 同 A |
| OPD scale | `OPD_LOSS_WEIGHT=1.0`，`OPD_DYNAMIC_SCALE=0`，`OPD_TASK_BALANCED_LOSS_SCALE=1` | 同 A | 同 A | 同 A |
| retention | `RETENTION_LOSS_WEIGHT=0.5`，`RETENTION_OBJECTIVE=nll`，`RETENTION_TASK_BALANCED_LOSS_SCALE=1` | 同 A | 同 A | 同 A |
| length norm | `OPD_LENGTH_NORMALIZE_LOGPROB=1`，`RETENTION_LENGTH_NORMALIZE_LOGPROB=1` | 同 A | 同 A | 同 A |
| optimizer | `SGD lr=0.02 momentum=0.2` | `SGD lr=0.015 momentum=0.2` | 同 A | 同 B |
| clip / delta | `MAX_COEFF_DELTA=1.0`，不做紧 trust-region 限制 | 同 A | 同 A | 同 A |
| PCGuard | off | off | on, symmetric PCGrad | on, symmetric PCGrad |

监控规则：不因为 gate 偏移本身停止；只有出现明显崩溃才停，包括 overall reward 连续大幅下滑且 tool/memory/code 某一项接近不可用、gate 爆到边界并伴随 reward 崩、update loss/grad 出现 NaN/Inf、或者 rollout/update 反复失败。

## 数据与轨迹

| 类型 | 路径 |
|---|---|
| calibration prompts | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| tool expert rollout | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl` |
| memory expert rollout | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl` |
| code expert rollout | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl` |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` |

动态 OPD 逻辑：每轮先用当前 gate rollout 96 prompts；对同一 prompt 的 4 个 samples，如果当前 policy 全错，则尝试从对应 expert rollout 中找 `reward_train >= 1.0` 的 positive trajectory；每个任务最多选 `32` 条 OPD rows。若某任务没有全错样本或 expert 没有 positive，则该任务本轮不产生 OPD loss，只由 retention / 其他 loss 约束。

## Run ID

| 实验 | tmux | run dir |
|---|---|---|
| A | `opdA_gc_nopc_20260515` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdA_gc_nopc_20260515` |
| B | `opdB_gp_nopc_20260515` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdB_gp_nopc_20260515` |
| C | `opdC_gc_pcguard_20260515` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdC_gc_pcguard_20260515` |
| D | `opdD_gp_pcguard_20260515` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdD_gp_pcguard_20260515` |

前端监控：计划挂载四个 run dir，按 run_id 区分曲线；重点看 `overall_reward_mean`、三任务 reward、`grad_norm`、`gate_delta`、三任务 gate effective mean、dynamic OPD selected rows、PCGrad conflict/cosine。

## 启动命令模板

共同环境：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
DYNAMIC_OPD_EXPERT_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl

NUM_ITERS=15
NUM_PROMPTS=96
SAMPLES_PER_PROMPT=4
INIT_VALUE=0.3333333333333333
OPTIMIZER=sgd
SGD_MOMENTUM=0.2
PERSIST_OPTIMIZER_STATE=1
PPO_LOSS_WEIGHT=0.0
BEST_RESPONSE_LOSS_WEIGHT=0.0
PAIRWISE_LOSS_WEIGHT=0.0
PRIOR_LOSS_WEIGHT=0.0
USE_RETENTION=1
RETENTION_OBJECTIVE=nll
RETENTION_LOSS_WEIGHT=0.5
RETENTION_POSITIVE_REWARD_THRESHOLD=1.0
RETENTION_TASK_BALANCED_LOSS_SCALE=1
RETENTION_DYNAMIC_SCALE=0
OPD_LOSS_WEIGHT=1.0
OPD_PAIRWISE_LOSS_WEIGHT=0.0
OPD_POSITIVE_REWARD_THRESHOLD=1.0
OPD_TASK_BALANCED_LOSS_SCALE=1
OPD_DYNAMIC_SCALE=0
OPD_LENGTH_NORMALIZE_LOGPROB=1
RETENTION_LENGTH_NORMALIZE_LOGPROB=1
LENGTH_NORMALIZE_POLICY_LOGPROB=1
LENGTH_NORMALIZE_LOGPROB=0
UPDATE_EPOCHS=1
UPDATE_BATCH_SIZE=4
BATCH_LOSS_REDUCTION=mean
OPTIMIZER_STEP_SCOPE=epoch
LOSS_GRANULARITY=sequence
STORE_TOKEN_LOGPROBS=0
TASK_NORMALIZE_ADVANTAGES=0
ADVANTAGE_NORMALIZATION=centered
USE_FRONTIER_WEIGHT=0
FRONTIER_TOOL_QUOTA=0
FRONTIER_MEMORY_QUOTA=0
FRONTIER_CODE_QUOTA=0
FRONTIER_ORDER=task-interleaved
DYNAMIC_OPD_TASKS=tool,memory,code
DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0
DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0
DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1
DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2
DYNAMIC_OPD_PER_TASK=32
MAX_COEFF_DELTA=1.0
ROLLOUT_SHARDS=auto
ROLLOUT_BATCH_SIZE=32
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.82
MAX_NEW_TOKENS=1024
TOOL_MAX_NEW_TOKENS=512
MEMORY_UPDATE_MAX_NEW_TOKENS=2048
MEMORY_FINAL_MAX_NEW_TOKENS=2048
CODE_MAX_NEW_TOKENS=4096
MAX_PROMPT_TOKENS=8192
MAX_MODEL_LEN=12288
MAX_LOGPROB_TOKENS=12288
GRADIENT_CHECKPOINTING=1
MAX_MEMORY_PER_GPU=70GiB
CPU_MAX_MEMORY=180GiB
SEED_VALUE=20260515
```

A/B 不加 PCGuard；C/D 加：

```bash
PCGRAD_GATE_GRADIENTS=1
PCGRAD_EPS=1e-12
PCGRAD_TASKS=tool,memory,code
```

## 观察记录

### 14:35 首轮 update

| 实验 | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---:|---:|---:|---:|---|---|
| A | `7/8/1` | `4` | `0.0710` | `0.00141` | `0.33315 / 0.33474 / 0.33339` | 未崩；推动偏弱 |
| B | `5/5/3` | `3` | `0.0473` | `0.00072` | `0.33329 / 0.33403 / 0.33341` | 未崩；推动偏弱 |
| C | `11/9/5` | `5` | `0.0559` | `0.00077` | `0.33298 / 0.33406 / 0.33411` | 未崩；PCGuard 生效 |
| D | `10/8/4` | `5` | `0.0660` | `0.00072` | `0.33306 / 0.33398 / 0.33400` | 未崩；PCGuard 生效 |

首轮结论：

- 四个 run 都完成 iter1 update，均无 NaN/Inf、无脚本崩溃、无异常 gate 爆炸。
- `OPD_LENGTH_NORMALIZE_LOGPROB=1 + OPD_LOSS_WEIGHT=1.0` 下，首轮 gate step 明显弱于 2026-05-14 best run：best 首轮 `gate_delta≈0.0143`，本次首轮只有 `0.0007-0.0014`。
- PCGuard run 的 summary 记录 `pcgrad.enabled=true`，`conflict_count_max=3`，说明三任务梯度确实存在冲突并发生投影；但投影后总步幅没有放大。
- 目前按用户规则不停止：gate 偏移正常，且尚未出现 reward 崩溃；继续观察 iter2/iter3 reward 是否改善。

### 14:45 重启：按目标步幅反推 LR

用户要求让 `lr * grad_norm ~= 0.01-0.015`。低 LR 版 iter2 的最新梯度量级如下：

| 原 run | latest iter | grad_norm | old LR | old gate_delta | 目标 `0.012/grad_norm` |
|---|---|---:|---:|---:|---:|
| A | iter2 | `0.073798` | `0.02` | `0.00160` | `0.1626` |
| B | iter2 | `0.063950` | `0.015` | `0.00111` | `0.1876` |
| C | iter2 | `0.053518` | `0.02` | `0.00098` | `0.2242` |
| D | iter2 | `0.080944` | `0.015` | `0.00120` | `0.1483` |

重启策略：

- 停止低 LR 版 A/B/C/D 和旧 monitor；保留已有 run 目录用于追溯。
- 新 run 仅改 `LR`，其他 OPD-only / retention / PCGuard 设置保持不变。
- 目标是首轮 `gate_delta` 进入 `0.01-0.015`；如果首轮超过 `0.03` 且 reward 同步下降，再判定为可能过冲。

| 新实验 | run dir | LR |
|---|---|---:|
| A-step012 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdA_gc_nopc_step012_20260515` | `0.1626` |
| B-step012 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdB_gp_nopc_step012_20260515` | `0.1876` |
| C-step012 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdC_gc_pcguard_step012_20260515` | `0.2242` |
| D-step012 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdD_gp_pcguard_step012_20260515` | `0.1483` |

首轮结果：

| 新实验 | OPD rows tool/memory/code | retention rows | grad_norm | `LR*grad_norm` | gate_delta | gate mean tool/memory/code | 判定 |
|---|---:|---:|---:|---:|---:|---|---|
| A-step012 | `11/9/2` | `2` | `0.0812` | `0.0132` | `0.0126` | `0.32957 / 0.34590 / 0.33476` | 达到目标 |
| B-step012 | `9/5/3` | `6` | `0.0734` | `0.0138` | `0.0137` | `0.33060 / 0.34676 / 0.33405` | 达到目标 |
| C-step012 | `8/8/4` | `5` | `0.0619` | `0.0139` | `0.0088` | `0.32617 / 0.34132 / 0.34212` | PCGuard 后实际步幅略低 |
| D-step012 | `7/4/5` | `2` | `0.0697` | `0.0103` | `0.0072` | `0.32894 / 0.34027 / 0.33944` | PCGuard 后实际步幅略低 |

结论：

- A/B 的实际 gate step 已进入 `0.01-0.015` 目标区间。
- C/D 的 `LR*grad_norm` 也进入目标区间，但 PCGuard 投影后真实 gate_delta 只有 `0.007-0.009`；这说明 PCGrad 不是简单放大/缩小总梯度，而是在冲突方向上削掉了一部分有效位移。
- 当前仍未出现 NaN/Inf 或 gate 爆炸；继续跑，看 iter2 reward 是否因更大步幅开始上升或出现过冲。

### 15:50 监控：step012 已出现正反馈

| 实验 | 已完成 | overall reward | 任务 reward 最新 | gate 最新 tool/memory/code | 判定 |
|---|---|---|---|---|---|
| A-step012 | iter5 | `0.364 -> 0.416 -> 0.472 -> 0.521 -> 0.568` | `tool=0.8669, memory=0.3984, code=0.4375` | `0.32636 / 0.39293 / 0.33902` | 最强，连续上升 |
| B-step012 | iter4 | `0.442 -> 0.432 -> 0.434 -> 0.490` | `tool=0.7494, memory=0.3438, code=0.3770` | `0.32233 / 0.38918 / 0.33778` | 回升，继续观察 |
| C-step012 | iter4 | `0.397 -> 0.368 -> 0.413 -> 0.442` | `tool=0.5396, memory=0.3438, code=0.4434` | `0.32449 / 0.37442 / 0.36353` | 从低点恢复 |
| D-step012 | iter4 | `0.398 -> 0.388 -> 0.379 -> 0.424` | `tool=0.5406, memory=0.3516, code=0.3789` | `0.32299 / 0.36380 / 0.35374` | 从低点恢复 |

当前没有明显崩溃：无 NaN/Inf/Traceback，gate 未爆到边界。A 的增长最稳定，但 tool all-success 数量上升很快，后续需观察 retention 是否开始主导、是否牺牲 memory/code。

### 16:00 监控：A/B 继续领先，PCGuard 组仍偏保守

当前这组 `step012` 是 OPD-only + retention 配置：

- `PPO_LOSS_WEIGHT=0` 且 `FRONTIER_*_QUOTA=0`，所以没有 frontier/GRPO loss。
- 全错 prompt：如果 expert rollout 有正样本，进入 dynamic OPD loss。
- 全对 prompt：进入 all-success NLL retention。
- 部分成功 prompt：本轮不直接参与 loss，只进入 rollout/reward 统计，并影响后续是否变成全错或全对。

| 实验 | 已完成 | overall reward 近况 | 最新任务 reward | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---|---|---|---:|---:|---:|---:|---|---|
| A-step012 | iter6 | `0.521 -> 0.568 -> 0.552` | `tool=0.9299, memory=0.3438, code=0.3828` | `1/6/2` | `20` | `0.0460` | `0.0096` | `0.32565 / 0.40257 / 0.34003` | 未崩；tool 过快变全对，retention 占比升高 |
| B-step012 | iter5 | `0.490 -> 0.569` | `tool=0.9133, memory=0.3750, code=0.4180` | `2/5/2` | `24` | `0.0438` | `0.0115` | `0.32257 / 0.40030 / 0.33887` | 未崩；与 A 接近，仍在 update 后观察 iter6 |
| C-step012 | iter5 | `0.442 -> 0.495` | `tool=0.7487, memory=0.3594, code=0.3779` | `5/5/3` | `13` | `0.0447` | `0.0094` | `0.32765 / 0.38149 / 0.37295` | 未崩；PCGuard 组恢复但慢于 A/B |
| D-step012 | iter5 | `0.424 -> 0.434` | `tool=0.5663, memory=0.3125, code=0.4219` | `8/10/1` | `4` | `0.0428` | `0.0062` | `0.32219 / 0.36756 / 0.35980` | 未崩；最弱，继续观察是否自然恢复 |

补充判断：

- A/B 的 no-PCGuard 版本当前更符合“先用 OPD 推动 gate，再靠 retention 保住已会能力”的目标；A 的 tool reward 已很高，但 memory/code 开始回落，需要继续看 iter7/iter8 是否被 retention 拉住。
- C/D 的 PCGuard 没有崩，但投影后实际有效步幅更小；如果最终 reward 落后，说明当前三任务冲突里简单 symmetric PCGrad 可能削掉了有用的共同上升方向。
- 目前所有 run 仍活跃：A/C/D 在 rollout，B 在 iter6 update。暂不停止。

### 16:20 监控：A 达到当前最高点，未触发停止条件

| 实验 | 已完成 | latest overall | latest task reward | all-fail tool/memory/code | all-success tool/memory/code | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A-step012 | iter8 | `0.5943` | `tool=0.9597, memory=0.4609, code=0.3623` | `7/6/13` | `19/5/1` | `2/4/2` | `25` | `0.0515` | `0.0106` | `0.32563 / 0.42420 / 0.34274` | 当前最好；memory 恢复，code 偏低 |
| B-step012 | iter7 | `0.5729` | `tool=0.9561, memory=0.3906, code=0.3721` | `7/8/12` | `18/2/0` | `1/6/2` | `20` | `0.0598` | `0.0138` | `0.32492 / 0.42631 / 0.34135` | iter8 rollout 已完成，等待 update |
| C-step012 | iter6 | `0.5275` | `tool=0.8676, memory=0.3281, code=0.3867` | `7/10/14` | `13/1/3` | `1/8/4` | `17` | `0.0353` | `0.0074` | `0.32317 / 0.38541 / 0.38034` | PCGuard 组继续上升但慢 |
| D-step012 | iter6 | `0.4851` | `tool=0.6917, memory=0.3516, code=0.4121` | `12/8/11` | `7/1/2` | `6/6/2` | `10` | `0.0538` | `0.0086` | `0.32233 / 0.37582 / 0.36321` | 仍未崩，落后于 A/B |

当前停止条件判断：

- 没有 NaN/Inf/Traceback。
- 没有连续多轮整体 reward 崩溃；A/B 仍保持高位，C/D 仍在恢复。
- gate step 仍在 `0.006-0.014` 的可控范围，没有爆到边界。
- 主要风险不是崩溃，而是 OPD-only 阶段逐渐把 tool 推到饱和，同时 code 信号不足；这需要最终评测验证是否 proxy 偏置。

### 16:40 监控：A iter9 继续上涨，当前最佳

| 实验 | 已完成 | latest overall | latest task reward | all-fail tool/memory/code | all-success tool/memory/code | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A-step012 | iter9 | `0.6278` | `tool=0.9605, memory=0.4766, code=0.4463` | `7/8/12` | `20/5/2` | `1/6/3` | `27` | `0.0372` | `0.0081` | `0.32555 / 0.43228 / 0.34391` | 当前最佳；code 回升，未崩 |
| B-step012 | iter8 | `0.6090` | `tool=0.9577, memory=0.4453, code=0.4238` | `4/7/12` | `20/4/2` | `0/4/2` | `26` | `0.0423` | `0.0107` | `0.32431 / 0.43675 / 0.34292` | 第二；iter9 rollout 已完成，等待 update |
| C-step012 | iter7 | `0.5511` | `tool=0.8867, memory=0.3750, code=0.3916` | `7/10/11` | `16/2/2` | `2/7/1` | `20` | `0.0385` | `0.0091` | `0.32601 / 0.38799 / 0.38939` | PCGuard 组仍在上升 |
| D-step012 | iter7 | `0.5134` | `tool=0.7844, memory=0.3438, code=0.4121` | `9/12/14` | `11/1/3` | `3/10/3` | `15` | `0.0622` | `0.0091` | `0.32441 / 0.38455 / 0.36870` | PCGuard 组上升但最慢 |

判断：

- A/B 的无 PCGuard 版本当前明显优于 C/D。
- A 不仅 overall 上涨，code 也从 iter8 的 `0.3623` 回升到 `0.4463`，暂时不能判定为 tool 单向吞噬。
- A/B 的 tool 已接近饱和，全对样本增加后 retention rows 达到 `26-27`；后续 watch point 是 retention 是否让 OPD 可恢复样本越来越少，导致 gate 停止推进。
- 当前仍不满足停止条件。

### 16:50 监控：B iter9 刷新最高 reward

| 实验 | 已完成 | latest overall | latest task reward | all-fail tool/memory/code | all-success tool/memory/code | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A-step012 | iter9 | `0.6278` | `tool=0.9605, memory=0.4766, code=0.4463` | `7/8/12` | `20/5/2` | `1/6/3` | `27` | `0.0372` | `0.0081` | `0.32555 / 0.43228 / 0.34391` | 第二；iter10 rollout 接近完成 |
| B-step012 | iter9 | `0.6308` | `tool=0.9427, memory=0.5469, code=0.4027` | `5/6/13` | `19/7/2` | `0/4/3` | `28` | `0.0458` | `0.0107` | `0.32340 / 0.44719 / 0.34497` | 当前最佳；memory 明显更强 |
| C-step012 | iter7 | `0.5511` | `tool=0.8867, memory=0.3750, code=0.3916` | `7/10/11` | `16/2/2` | `2/7/1` | `20` | `0.0385` | `0.0091` | `0.32601 / 0.38799 / 0.38939` | iter8 rollout 完成，等待 update |
| D-step012 | iter7 | `0.5134` | `tool=0.7844, memory=0.3438, code=0.4121` | `9/12/14` | `11/1/3` | `3/10/3` | `15` | `0.0622` | `0.0091` | `0.32441 / 0.38455 / 0.36870` | iter8 rollout 完成，等待 update |

判断：

- 当前最值得后续评测的 checkpoint 是 `B-step012 iter9`，其次 `A-step012 iter9`。
- B 的 memory 达到 `0.5469`，说明这组参数没有只把 tool 推满；但 code 仍低于 A iter9。
- PCGuard 组尚未完成最新 update，不提前下结论；但截至 iter7 已明显落后。

### 17:00 监控：PCGuard 组未崩，D 开始追赶

| 实验 | 已完成 | latest overall | latest task reward | all-fail tool/memory/code | all-success tool/memory/code | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A-step012 | iter9 | `0.6278` | `tool=0.9605, memory=0.4766, code=0.4463` | `7/8/12` | `20/5/2` | `1/6/3` | `27` | `0.0372` | `0.0081` | `0.32555 / 0.43228 / 0.34391` | iter10 rollout 完成，等待 update |
| B-step012 | iter9 | `0.6308` | `tool=0.9427, memory=0.5469, code=0.4027` | `5/6/13` | `19/7/2` | `0/4/3` | `28` | `0.0458` | `0.0107` | `0.32340 / 0.44719 / 0.34497` | 当前最佳，iter10 rollout 中 |
| C-step012 | iter8 | `0.5511` | `tool=0.9052, memory=0.3750, code=0.3730` | `5/8/15` | `15/4/2` | `1/6/3` | `21` | `0.0317` | `0.0079` | `0.32320 / 0.38990 / 0.39732` | 平台期；code gate 推高但 reward 未同步涨 |
| D-step012 | iter8 | `0.5756` | `tool=0.8529, memory=0.4609, code=0.4131` | `6/7/12` | `15/5/3` | `1/4/3` | `23` | `0.0362` | `0.0060` | `0.32399 / 0.38781 / 0.37414` | 明显追赶；仍低于 A/B |

判断：

- PCGuard 并非失败，D 的 iter8 有明显增益；但截至目前没有超过无 PCGuard 的 A/B。
- C 的 code gate 被推到 `0.397`，但 code reward 没上来，说明 PCGuard 投影后的方向可能更“保守/分散”，不一定带来 proxy reward 最优。
- A/B 仍是主要候选；继续看 iter10 是否出现回落。

### 17:10 监控：A iter10 刷新最高且三任务更均衡

| 实验 | 已完成 | latest overall | latest task reward | all-fail tool/memory/code | all-success tool/memory/code | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A-step012 | iter10 | `0.6563` | `tool=0.9661, memory=0.5234, code=0.4795` | `3/5/10` | `19/3/5` | `0/3/3` | `27` | `0.0363` | `0.0074` | `0.32457 / 0.43963 / 0.34514` | 当前最佳；三任务同步改善 |
| B-step012 | iter9 | `0.6308` | `tool=0.9427, memory=0.5469, code=0.4027` | `5/6/13` | `19/7/2` | `0/4/3` | `28` | `0.0458` | `0.0107` | `0.32340 / 0.44719 / 0.34497` | 第二；iter10 rollout 完成，等待 update |
| C-step012 | iter8 | `0.5511` | `tool=0.9052, memory=0.3750, code=0.3730` | `5/8/15` | `15/4/2` | `1/6/3` | `21` | `0.0317` | `0.0079` | `0.32320 / 0.38990 / 0.39732` | PCGuard 组平台期 |
| D-step012 | iter8 | `0.5756` | `tool=0.8529, memory=0.4609, code=0.4131` | `6/7/12` | `15/5/3` | `1/4/3` | `23` | `0.0362` | `0.0060` | `0.32399 / 0.38781 / 0.37414` | PCGuard 组中较好 |

判断：

- A 的 tool 已基本饱和，但本轮 OPD 不再依赖 tool 样本，主要由 memory/code 继续提供可恢复信号；这是理想的 OPD-only 第一阶段动态。
- A 的 `gate_delta` 下降到 `0.0074`，说明随着全错样本减少，推动自然变小，没有过冲。
- 当前最值得保留/评测的 checkpoint 更新为 `A-step012 iter10`。

### 17:25 监控：A/B 高点后回落，best checkpoint 保留

| 实验 | 已完成 | latest overall | latest task reward | all-fail tool/memory/code | all-success tool/memory/code | OPD rows tool/memory/code | retention rows | grad_norm | gate_delta | gate mean tool/memory/code | 判定 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A-step012 | iter11 | `0.5991` | `tool=0.9516, memory=0.3828, code=0.4629` | `6/10/9` | `21/4/4` | `0/8/1` | `29` | `0.0424` | `0.0082` | `0.32367 / 0.44785 / 0.34657` | 从 iter10 回落；未全面崩 |
| B-step012 | iter10 | `0.6048` | `tool=0.9655, memory=0.4609, code=0.3881` | `5/7/12` | `21/5/1` | `0/5/1` | `27` | `0.0282` | `0.0068` | `0.32247 / 0.45380 / 0.34789` | 从 iter9 回落；等待 iter11 |
| C-step012 | iter9 | `0.5410` | `tool=0.8876, memory=0.3828, code=0.3525` | `7/9/12` | `16/3/2` | `2/6/3` | `21` | `0.0411` | `0.0076` | `0.32898 / 0.39332 / 0.40487` | PCGuard 组开始回落 |
| D-step012 | iter9 | `0.5688` | `tool=0.8906, memory=0.3672, code=0.4486` | `7/10/14` | `14/2/3` | `1/8/2` | `19` | `0.0419` | `0.0053` | `0.32203 / 0.39139 / 0.37861` | 小回落；code 较好 |

判断：

- 当前 best 仍是 `A-step012 iter10`，其次 `B-step012 iter9`。
- A/B 的回落主要来自 memory/code 的随机波动和 OPD 可恢复样本减少，不是 NaN/梯度爆炸。
- 不停止：best checkpoint 已保留，继续跑完整 15 iter 可以观察后期是否恢复或进入 retention 主导平台。

### 17:35 监控：B 连续回落，A 等待 iter12

最近五轮 proxy reward：

| 实验 | 最近轨迹 | 当前 best |
|---|---|---|
| A-step012 | `0.5731 -> 0.5943 -> 0.6278 -> 0.6563 -> 0.5991` | iter10 `0.6563` |
| B-step012 | `0.5729 -> 0.6090 -> 0.6308 -> 0.6048 -> 0.5984` | iter9 `0.6308` |
| C-step012 | `0.4953 -> 0.5275 -> 0.5511 -> 0.5511 -> 0.5410` | iter7/8 `0.5511` |
| D-step012 | `0.4336 -> 0.4851 -> 0.5134 -> 0.5756 -> 0.5688` | iter8 `0.5756` |

判断：

- B 已连续两轮从高点回落，主要由 code 从 `0.4027 -> 0.3881 -> 0.3311` 下滑导致；这是后期过推或 retention/OPD 样本稀疏后的不稳定信号。
- A 只有一轮回落，且 code 仍有 `0.4629`；需要等 iter12 判断是否恢复。
- 当前仍不停止，因为没有崩溃，且完整 15 iter 曲线对判断后期训练动态有价值。

### 17:45 监控：B 恢复，C 出现 PCGuard 组高点

当前各实验 best：

| 实验 | best iter | best overall | best task reward | 当前阶段判断 |
|---|---|---:|---|---|
| A-step012 | iter10 | `0.6563` | `tool=0.9661, memory=0.5234, code=0.4795` | 全局最佳；iter11/12 后期回落 |
| B-step012 | iter9 | `0.6308` | `tool=0.9427, memory=0.5469, code=0.4027` | iter12 恢复到 `0.6252`，仍略低于 best |
| C-step012 | iter10 | `0.6048` | `tool=0.8916, memory=0.4453, code=0.4775` | PCGuard 组新高，code 最强之一 |
| D-step012 | iter8 | `0.5756` | `tool=0.8529, memory=0.4609, code=0.4131` | PCGuard/global-parameter 中等 |

判断：

- A/B 无 PCGuard 仍整体更强；C 说明 PCGuard 不是无效，只是推进慢，且可能更偏向 code。
- 后期继续训练的收益不稳定，推荐最终评测至少包含 `A iter10`、`B iter9`、`C iter10`。

### 18:05 监控：B iter13 刷新全局最高

| 实验 | best iter | best overall | best task reward | 备注 |
|---|---|---:|---|---|
| B-step012 | iter13 | `0.6587` | `tool=0.9675, memory=0.6094, code=0.3992` | 当前全局最高；memory 最强 |
| A-step012 | iter10 | `0.6563` | `tool=0.9661, memory=0.5234, code=0.4795` | overall 第二；code 更强 |
| C-step012 | iter10 | `0.6048` | `tool=0.8916, memory=0.4453, code=0.4775` | PCGuard 组最佳；code 接近 A |
| D-step012 | iter8 | `0.5756` | `tool=0.8529, memory=0.4609, code=0.4131` | PCGuard/global-parameter 较弱 |

判断：

- B 在 iter10/11 回落后，iter12/13 恢复并超过 A，说明后期波动不是简单单调崩溃。
- A 和 B 的取舍很清楚：B 更偏 memory，A 更均衡且 code 更高；最终完整评测需要两者都送。
- C 可作为 PCGuard 对照：overall 较低，但 code 能力不差。

### 18:25 监控：A 完成 15/15，并在最终轮刷新最高

| 实验 | 当前状态 | best iter | best overall | best task reward | 备注 |
|---|---|---|---:|---|---|
| A-step012 | 完成 15/15 | iter15 | `0.6623` | `tool=0.9645, memory=0.6094, code=0.4131` | 当前全局最高；最终轮恢复 |
| B-step012 | iter15 rollout 中 | iter13 | `0.6587` | `tool=0.9675, memory=0.6094, code=0.3992` | 接近 A，等待最终轮 |
| C-step012 | iter14 等待/进行中 | iter10 | `0.6048` | `tool=0.8916, memory=0.4453, code=0.4775` | PCGuard global-coefficient，code 较强 |
| D-step012 | iter14 rollout 中 | iter12 | `0.5844` | `tool=0.9359, memory=0.3906, code=0.4268` | PCGuard global-parameter |

判断：

- A 的曲线证明“iter10 后回落”不是崩溃；最终 iter15 恢复并刷新最高。
- 目前最值得优先完整评测的是 `A-step012 iter15` 和 `B-step012 iter13/最终轮`。
- 仍需等 B/C/D 完成后做最终排序。

### 18:45 监控：B 完成 15/15，并刷新全局最高

| 实验 | 当前状态 | best iter | best overall | best task reward | 备注 |
|---|---|---|---:|---|---|
| B-step012 | 完成 15/15 | iter15 | `0.6823` | `tool=0.9635, memory=0.6562, code=0.4271` | 当前全局最高；memory 明显最强 |
| A-step012 | 完成 15/15 | iter15 | `0.6623` | `tool=0.9645, memory=0.6094, code=0.4131` | 第二；global-coefficient 完整跑通 |
| C-step012 | iter15 等待/进行中 | iter14 | `0.6080` | `tool=0.9618, memory=0.3672, code=0.4951` | PCGuard global-coefficient；code 最强 |
| D-step012 | iter15 rollout 中 | iter12 | `0.5844` | `tool=0.9359, memory=0.3906, code=0.4268` | PCGuard global-parameter |

判断：

- 目前最应该送完整评测的是 `B-step012 iter15` 和 `A-step012 iter15`。
- C 虽然 overall 低，但 code proxy 最高，可作为 PCGuard 的任务偏向分析点。
- B/A 都在最终轮达到各自最佳，说明这组 OPD-only + retention 配置没有出现不可恢复崩溃。

### 18:55 最终训练结果与评测启动

四个实验均完成 15/15，没有 NaN/Inf/Traceback，没有 gate 爆炸。

| 排名 | 实验 | best iter | proxy overall | tool | memory | code | checkpoint |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | B: global-parameter, no PCGuard | iter15 | `0.6823` | `0.9635` | `0.6562` | `0.4271` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdB_gp_nopc_step012_20260515/iter_015/baked_policy` |
| 2 | A: global-coefficient, no PCGuard | iter15 | `0.6623` | `0.9645` | `0.6094` | `0.4131` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdA_gc_nopc_step012_20260515/iter_015/baked_policy` |
| 3 | C: global-coefficient, PCGuard | iter14 | `0.6080` | `0.9618` | `0.3672` | `0.4951` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdC_gc_pcguard_step012_20260515/iter_014/baked_policy` |
| 4 | D: global-parameter, PCGuard | iter15 | `0.5859` | `0.9520` | `0.4375` | `0.3682` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdD_gp_pcguard_step012_20260515/iter_015/baked_policy` |

训练结论：

- 无 PCGuard 的 A/B 明显优于 PCGuard 的 C/D。
- B 的 memory proxy 最强，A 更均衡；二者都值得完整评测。
- C 的 overall 较低但 code proxy 最高，可作为“PCGuard 保护/偏向 code 方向”的分析样本。
- OPD-only + retention 第一阶段可以把 proxy reward 从约 `0.36-0.44` 推到 `0.66-0.68`，训练信号是有效的。

已按 eval6 roadmap 启动完整评测：

| model name | checkpoint | GPU | port |
|---|---|---:|---:|
| `expertgym-opdB-gp-nopc-step012-i15` | B iter15 | 2 | 8092 |
| `expertgym-opdA-gc-nopc-step012-i15` | A iter15 | 3 | 8093 |
| `expertgym-opdC-gc-pcguard-step012-i14` | C iter14 | 4 | 8094 |
| `expertgym-opdD-gp-pcguard-step012-i15` | D iter15 | 5 | 8095 |

评测 tmux：

```bash
tmux attach -t eval_expertgym_opd_step012_20260515
```

评测 runner：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_expertgym_opd_step012_20260515_addons.py
```

### 20:20 完整评测结果已回收

四个 checkpoint 的 Tool、Memory、Code 完整 eval6 评测均已完成，并已 append 到：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/aggregate_report_zh.md
```

| 实验 | Tool mean | Memory mean F1 | Code avg Acc | Code avg BoN | 结论 |
|---|---:|---:|---:|---:|---|
| B: global-parameter, no PCGuard, iter15 | `0.7835` | `0.7171` | `0.3360` | `0.3636` | Memory 最好，但 Code 最弱 |
| A: global-coefficient, no PCGuard, iter15 | `0.7835` | `0.6924` | `0.3570` | `0.4105` | Code 最好，整体更均衡 |
| C: global-coefficient, PCGuard, iter14 | `0.7927` | `0.6504` | `0.3511` | `0.4017` | Tool/Code 尚可，但 Memory 明显掉 |
| D: global-parameter, PCGuard, iter15 | `0.7915` | `0.6661` | `0.3350` | `0.3880` | 三项均未超过 A/B |

评测结论：

- A/B 的 proxy 排名能部分迁移到真实评测，但迁移不完全：B 的 proxy memory 最强，真实 Memory 也更高；A 的 proxy code 较好，真实 Code 也最高。
- PCGuard 没有带来整体收益：C/D 的 Tool 略高，但 Memory 损失较大，综合不如无 PCGuard。
- 与 5 月 14 日最佳 `opvec_best_gp_opd_reproduction_20260514` 相比，B 的 Memory F1 `0.7171` 仍低于最佳 `0.7649`，说明这批 OPD-only + retention 虽然能推 gate，但尚未推到最佳 memory 能力区间，或后期信号已经变稀疏。

### 20:20 A/B 继续训练 5 轮

按用户要求，从 A/B 的 iter15 gate 与 optimizer state 继续训练 5 个 outer iteration，保留原 run 不覆盖：

| 实验 | 新 run_dir | 起点 | 目标轮次 | GPU |
|---|---|---|---|---|
| A continue | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdA_gc_nopc_step012_continue5_20260515` | A iter15 | iter16-20 | `0,1` |
| B continue | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdB_gp_nopc_step012_continue5_20260515` | B iter15 | iter16-20 | `6,7` |

继续训练配置保持与 A/B 原实验一致：OPD-only + retention NLL，dynamic OPD 只从当前 policy all-fail 且 expert positive 的样本构造，OPD/retention 均启用 length-normalized logprob 与 task-balanced loss scale，SGD momentum `0.2`，epoch-scope optimizer step，保留 optimizer state。

继续训练监控前端：

```text
http://127.0.0.1:8786
```

### 20:36 C/D 加入 GRPO + OPD + retention 继续训练

为验证 PCGuard 组后期是否能利用 partial-success 的 GRPO 信号，新增 C/D 继续训练。与 OPD-only 版本相比，只改训练目标：开启 GRPO 项，`ppo-loss-weight=1.0`，`opd-loss-weight=1.0`，retention 保持 NLL 保护；继续启用 PCGrad gate gradients。为了让新目标的结果更干净，gate 从 C/D iter15 接上，但 optimizer state 重置，不沿用 OPD-only 阶段的 momentum。

| 实验 | 新 run_dir | 起点 | objective | GPU |
|---|---|---|---|---|
| C continue | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdC_gc_pcguard_grpo_opd_ret_continue5_20260515` | C iter15 gate | GRPO 1.0 + OPD 1.0 + retention NLL | `2,3` |
| D continue | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdD_gp_pcguard_grpo_opd_ret_continue5_20260515` | D iter15 gate | GRPO 1.0 + OPD 1.0 + retention NLL | `4,5` |

统一监控前端：

```text
http://127.0.0.1:8787
```
