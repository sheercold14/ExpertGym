# 20260519 TRC Stage-1 Harness

## 目标

第一阶段不再追求直接 reward update，而是用 expert 成功轨迹把 gate 推到更有能力的结构化初始点。最终目标是产出可 bake、可评测、能力最强的 gate checkpoint，后续再接 GRPO/OPD reward refinement。

## 当前高优先级原则

1. 训练信号必须密集，但不能被 Code 长轨迹或大 residual 主导。
2. 对齐 span 应尽量贴近能力行为，而不是整段解释文本。
3. 旧 TRC v1 必须保持可复现；v2 只通过显式开关启用。
4. 每个 run 都保留 `trc_run_manifest.json`、`trc_metrics.jsonl`、`epoch_xxx.gates.json`、`trc_gates.json`。

## v2 Loss 修改

新增开关：

```text
--normalize-residual-by-target
--response-span-mode auto
--task-balanced-loss
--residual-weight-power 0.5
```

含义：

- `normalize-residual-by-target`：把 residual MSE 改成相对误差，避免 Code 因目标 residual 大而天然主导。
- `response-span-mode auto`：Tool 只对齐 `<tool_call>...</tool_call>`，Code 只对齐最长 code block，Memory 仍对齐完整 response。
- `task-balanced-loss`：任务数不均衡时每个 task 等权；当前 32/32/32 下 scale=1，但保留为 harness 标准。
- `residual-weight-power=0.5`：仍关注 expert residual 大的 token，但减弱大 residual token 的支配性。

## 默认 v2 启动

```bash
CUDA_VISIBLE_DEVICES=0,1 \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v2_20260519 \
bash skill/command/run_20260519_trc_layer_init1_v2.sh
```

默认参数：

```text
init=1.0
epochs=5
lr=0.01
optimizer=adamw
hidden_layers=8,16,24,28
max_seq_length=1536
max_response_tokens=512
topk_tokens=128
beta_base=0.02
gamma_gate=0.001
```

## 监控标准

优先看三件事：

1. `mean_residual_loss` 是否稳定下降。
2. Memory gate 是否不再被快速压到 0.7 以下。
3. Tool/Code 是否能形成层结构，而不是全层同步过冲。

如果 v2 仍压低 Memory：

- 提高 `BETA_BASE` 到 `0.05`；
- 提高 `GAMMA_GATE` 到 `0.005`；
- 或新增 `memory` task coefficient anchor，但这会引入更强先验，先不默认启用。

## 候选实验

| run | 设置 | 目的 |
|---|---|---|
| `trc_layer_init1_v2_20260519` | normalized + span-aware + beta 0.02 | 主候选，修复 v1 的尺度偏置 |
| `trc_layer_init1_v2_anchor_20260519` | v2 + beta 0.05 + gamma 0.005 | 检查更强保持项是否保护 Memory/Tool |

两者完成后，优先选择：不崩 Tool/Memory、Code 有提升迹象、gate 不贴边的 checkpoint 进入 bake/eval。
