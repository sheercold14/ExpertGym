# 20260519 TRC Round Harness

## 目标

今晚按 round 迭代 TRC stage-1：每轮 4 个实验并行训练，训练结束后只按 loss/gate 选一个 gate bake，不做每 epoch 大模型保存；正式评测先跑 Tool/Memory，满足门槛才跑 Code。

门槛：

| metric | threshold |
|---|---:|
| Tool mean acc | >= 0.79 |
| Memory mean F1 | >= 0.76 |

## Round1 实验

| id | GPU | calibration | 变量 | 训练意图 |
|---|---|---|---|---|
| `trc_r1_e0_anchor` | 0,1 | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl` | 旧 TRC96 + conservative directional | anchor 复核 |
| `trc_r1_e1_code_rf` | 2,3 | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round1/e1_code_eval_rf/trc96_expert_trajectories.jsonl` | ReasonFlux-only eval-aligned Code32 | 看新 code 轨迹是否提升 |
| `trc_r1_e2_code_multi` | 4,5 | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round1/e2_code_eval_multiteacher/trc96_expert_trajectories.jsonl` | Code32 多 teacher，优先 L5 CURE Code16 / DeepSeek | 看 DeepSeek trajectory 是否补算法 |
| `trc_r1_e3_tool_code_multi` | 6,7 | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round1/e3_tool_code_eval_multiteacher/trc96_expert_trajectories.jsonl` | Tool16 + Code32 多 teacher | 保护 Tool 并补 Code |

共同训练设置：

```text
init = 1.0
gate = layer-band-coefficient, 28 layers x 3 experts
epochs = 12
optimizer = AdamW
lr = 0.02
accumulation_steps = 96
hidden_layers = 8,16,24,28
residual_objective = directional
projection_floor = 0.8
projection_weight = 0.1
beta_base = 0.05
gamma_gate = 0.005
coefficient_floor = 0.95
coefficient_floor_weight = 0.1
task_balanced_loss = on
```

E1/E2/E3 的 task-specific code 设置：

```text
code hidden layers = 4,8,12,16,20,24,28
code topk = 192
code projection floor = 0.85
code projection weight = 0.15
code loss multiplier = 1.1
```

E3 额外 Tool 设置：

```text
tool topk = 96
tool loss multiplier = 1.1
```

选择 gate：

```text
selection_mode = loss-plateau
plateau_relative_improvement = 0.01
plateau_patience = 2
plateau_min_epoch = 4
gate_penalty = 0
```

Gate 只作为诊断信号，不参与 checkpoint selection。默认 `SELECT_GATE_PENALTY=0` 时，run 脚本不会向 selector 传入 gate min/max 区间；若全程未出现 plateau，则选 loss 最低的 epoch。

## Round1x 延长训练

目的：Round1 到 epoch 12 时 loss 仍在下降，因此补跑 E1/E3 的 24 epoch 版本，检查是否需要更长的 TRC stage-1 训练。仍然只按 loss-plateau/最低 loss 选 checkpoint，不使用 gate 安全区间。

| ID | EXP_ID | base config | epochs | GPUs | calibration |
|---|---|---|---:|---|---|
| E1x | `trc_r1x_e1_e24_code_rf_20260519` | E1 | 24 | 4,5 | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round1/e1_code_eval_rf/trc96_expert_trajectories.jsonl` |
| E3x | `trc_r1x_e3_e24_tool_code_multi_20260519` | E3 | 24 | 6,7 | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round1/e3_tool_code_eval_multiteacher/trc96_expert_trajectories.jsonl` |

## Round2 Code Recovery

动机：Round1 的 E3 多教师 code trajectory 出现 `span_tokens=1` 的坏样本，DeepSeek/code 轨迹会污染 code hidden-state 目标；E1 RF-only 的 LiveBench acc 反而更高。因此 Round2 构建 clean RF code calibration，同时保留 E3 的 BFCL Tool augment。

数据：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round2/e4_tool_aug_code_rf/trc96_expert_trajectories.jsonl
```

数据审计：

- Tool: 32 unique prompts, 16 BFCL augment + 16 paper96 ToolRL。
- Memory: 32 rows, paper96 MemAgent。
- Code: 32 unique prompts, ReasonFlux-only positives, exclude DeepSeek；32/32 code responses have code-block/code hints, no 1-token span pollution.

共同设置：

```text
config = configs/gated_grpo_layer28_wide.yaml
gate_bounds.coefficient = [-0.20, 1.80]
epochs = 18
selection = loss-plateau / min loss, gate penalty = 0
```

| ID | EXP_ID | GPUs | 关键变量 | 目的 |
|---|---|---|---|---|
| R2A | `trc_r2a_cleanrf_wide_dir_20260519` | 0,1 | directional, code topk=256, code/tool/memory loss=1.2/1.1/1.2 | 干净 RF 数据的保守主线 |
| R2B | `trc_r2b_cleanrf_wide_resp_20260519` | 2,3 | code response span, loss=1.3/1.1/1.3, gamma=0.015, floor_w=0.2 | 修 code span 并增强 memory 约束 |
| R2C | `trc_r2c_cleanrf_wide_relmse_20260519` | 4,5 | relative-MSE, code response span, loss=1.1/1.1/1.2 | 检查 MSE amplitude 对 code 是否更有效 |
| R2D | `trc_r2d_cleanrf_wide_codeheavy_20260519` | 6,7 | code response span, code topk=384, loss=1.6/1.1/1.2, gamma=0.02 | code-heavy 上限探索，同时用 loss 正则保 memory/tool |

运行中调整：

- R2C 在 epoch 4 出现 `mean_total_loss > 7` 且 Tool gate 下行，判断 relative-MSE 尺度不适合当前 TRC 目标，已停止。
- 释放 4,5 卡后启动 R2E：`trc_r2e_cleanrf_wide_codelayers_20260520`。R2E 使用 directional objective、code=response、code topk=384、code hidden layers=2,4,...,28、loss multiplier code/tool/memory=1.8/1.2/1.5、`gamma_gate=0.03`、`coefficient_floor_weight=0.3`，目标是在加强 code 可执行能力的同时通过 loss 正则保住 Memory/Tool。

## 数据审计

Round1 calibration 输出：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round1
```

E1:

- code unique prompts: 32
- code source: L5 ReasonFlux positives + eval_targeted ReasonFlux + old ReasonFlux
- excludes DeepSeek samples

E2:

- code unique prompts: 32
- code source: L5 CURE Code16 + eval_targeted DeepSeek/ReasonFlux + old pools

E3:

- tool unique prompts: 32
- code unique prompts: 32
- tool source: BFCL Tool16 official-answer anchors + paper96 ToolRL
- code source: same as E2

## 评测计划

训练完成后每个实验产物：

```text
run dir: /tmp/shared-storage/OnPolicy/runs/trc/<id>
selected gate: /tmp/shared-storage/OnPolicy/runs/trc/<id>/selected.gates.json
baked model: /tmp/shared-storage/OnPolicy/checkpoints/<id>-selected
```

评测顺序：

1. Tool + Memory only。
2. 若 Tool mean >= 0.79 且 Memory F1 >= 0.76，再跑 Code CURE。
3. 结果写入 `docs/evaluation/20260519_trc_rounds_eval.md`。
