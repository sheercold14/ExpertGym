# 2026-05-18 no-R1 3-Expert Layer-Band Memory Push

## 目的

启动一个无 R1 的三专家对照：只使用 `tool / memory / code` task vectors，通过 OPD + NLL retention、无 GRPO，把 layer-band memory gate 在 10 iterations 内推到 `0.55+`。

该实验回答：

```text
如果不引入 DeepSeek-R1 异质 expert，仅靠原三个 agent expert 的结构化 task vector，
OPD bootstrapping 能否快速恢复/增强 memory 能力？
```

## Run

```text
run_name: expE_noR1_3expert_layerband_memorypush_nogrpo_20260518
run_dir: /tmp/shared-storage/OnPolicy/runs/gated_grpo/expE_noR1_3expert_layerband_memorypush_nogrpo_20260518
tmux: train_expE_noR1_3expert_memorypush_20260518
GPUs: 0,1
```

## 核心配置

```text
config: configs/gated_grpo.yaml
mode: /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
experts: tool, memory, code
strategy: layer-band, default early/mid/late
init gate: /tmp/shared-storage/OnPolicy/data/init_gates/init_layer_band_3expert_c033333_20260518.json
init effective: tool=1/3, memory=1/3, code=1/3
num_iters: 10
num_prompts: 96
samples_per_prompt: 4
calibration: /tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
```

## Loss

```text
GRPO/PPO: off, PPO_LOSS_WEIGHT=0
OPD: on, OPD_LOSS_WEIGHT=1.0
retention: on, objective=NLL, target=0.5
prior: off
loss_granularity: sequence
optimizer_step_scope: epoch
old-logprob fill: skipped by loop because PPO=0 and retention=NLL
```

Dynamic OPD 使用三专家原始 expert rollouts：

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
```

## Memory Push 设置

```text
TRAIN_COEFFICIENTS=*.memory
TASK_WEIGHT_TOOL=0.2
TASK_WEIGHT_MEMORY=5.0
TASK_WEIGHT_CODE=0.5
LR=0.4
MAX_COEFF_DELTA=1.0
```

含义：

- 三专家都参与 bake 和 rollout；
- 只训练每个 layer band 的 memory effective coefficient；
- tool/code effective coefficient 通过 projection 锚定在 `1/3`；
- memory 任务 loss 主导更新，目标是快速验证 memory gate 是否可被 OPD bootstrapping 推到 `0.55+`。

## 判据

继续运行：

```text
memory effective coefficient 持续上升；
memory reward / all-fail 有改善；
tool/code 没有因 memory 增大出现明显 proxy 崩溃。
```

早停：

```text
memory gate 连续两轮不上升；
memory gate 上升但 memory reward 不动或下降；
memory gate 过冲导致 tool/code proxy 明显坍塌。
```

## 2026-05-18 18:45 结果更新

本实验按上述设置启动后，iter1 update 出现反方向移动，已停止，不继续跑满 10 轮。

### 原 run: only memory coefficient trainable

```text
run: expE_noR1_3expert_layerband_memorypush_nogrpo_20260518
train_coefficients: *.memory
```

iter1 rollout:

| task | mean reward | rows | any success | all fail | all success | frontier |
|---|---:|---:|---:|---:|---:|---:|
| code | 0.4023 | 32 | 18 | 14 | 5 | 21 |
| memory | 0.4141 | 32 | 25 | 7 | 1 | 24 |
| tool | 0.3980 | 32 | 20 | 12 | 0 | 25 |

dynamic OPD:

```text
selected_rows=16
selected_task_counts: code=2, memory=5, tool=9
skipped: current_not_failure=63, no_expert_positive=17
```

iter1 gate:

| band | tool | memory | code |
|---|---:|---:|---:|
| early | 0.3333 | 0.1897 | 0.3333 |
| mid | 0.3333 | -0.1582 | 0.3333 |
| late | 0.3333 | 0.2770 | 0.3333 |

结论：只训练 memory coefficient 会被 OPD/NLL 梯度强烈往下推，不能作为 memory gate push 方法。

### Probe: all coefficients trainable

为确认是否是“冻结 tool/code 导致投影异常”，复用同一份 rollout/OPD 做了一次 all-coefficients update probe：

```text
run: expE_noR1_3expert_layerband_memorypush_allcoeff_probe_20260518
train_coefficients: all
lr: 0.25
```

iter1 gate:

| band | tool | memory | code |
|---|---:|---:|---:|
| early | 0.2862 | 0.2444 | 0.2918 |
| mid | 0.1960 | 0.0258 | 0.2007 |
| late | 0.3167 | 0.2981 | 0.3149 |

结论：放开三专家一起学也仍然把 memory 往下推，说明核心不是 `TRAIN_COEFFICIENTS=*.memory`，而是当前 OPD/NLL 目标本身在此 batch 上不支持提高 memory coefficient。

## 启发

当前 no-GRPO + OPD NLL 不是“按能力方向推 gate”的稳定目标。它优化的是专家轨迹在当前混合模型下的似然，而不是直接优化 memory reward 或 memory coefficient。对三专家 no-R1 来说，memory expert trajectory 的 NLL 梯度并不等价于“增大 memory task vector”。

如果目标是科学地让 memory gate 学到 `0.55+`，需要改训练信号，而不是只调 LR：

1. 引入 GRPO/frontier reward，让 memory reward 对 coefficient 有直接偏好；
2. 或构造 pairwise/self-compare：同一 prompt 下高 memory gate rollout reward > 低 memory gate rollout reward，再做 preference；
3. 或做 explicit target-prior / coefficient floor，但这只能作为控制实验，不能声称是 executable feedback 学出来的。
