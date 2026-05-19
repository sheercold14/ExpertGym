# 20260519 TRC Layer Init1

## 目标

按 `docs/memory/TRC_Merging_Method.md` 做一条和 GRPO/OPD 主线隔离的 TRC-Merging 原型：用 expert 成功轨迹提供隐藏层 residual target，只训练 OP-VEC gate，不改 reward、不改 rollout、不改旧训练脚本。第一版从 `init=1.0` 开始，直接学习 28 层 × 3 expert 的 layer-band direct coefficient。

当前代码快照已先提交：

- commit: `591aff5`
- message: `Snapshot ExpertGym experiments and gate training tools`

## 数据

构造脚本：

```bash
bash skill/command/build_20260519_trc_calibration_v1.sh
```

输出目录：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1
```

主文件：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl
```

数据定义：96 条 successful expert trajectories，三任务均衡，每 task 32 条。注意这不是 96 个完全不同 prompt；当前可用 expert rollout 中 Tool/Memory 严格成功 prompt 分别只有 27/28 个，所以 builder 先取 unique prompt，再用同 prompt 的其他成功 sample 补足 32 条。

来源策略：

- Tool: `paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl`
- Memory: `paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl`
- Code: 优先 `ReasonFlux-Coder-7B` 的 `seed20260517` 成功轨迹，再 fallback 到 `seed20260516`、DeepSeek-R1、旧 code expert。

## 方法

新增独立脚本：

```text
scripts/trc/train_trc_layer_gates.py
```

训练参数化：

- `--gate-parameterization layer-band-coefficient`
- 可学习参数：`28 layers × 3 experts = 84` 个 direct coefficient
- 初始化：所有 effective coefficient = `1.0`
- mode manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`
- base: `Qwen2.5-7b-instruct`
- experts: tool / memory / code 三个 task vector

TRC v1 loss：

```text
L = L_residual + beta_base * L_base_drift + gamma_gate * L_gate_anchor
```

`L_residual`：对每条 expert 成功轨迹，计算同一 prompt+expert response 上的隐藏层 residual：

```text
r_expert = h(base + target_expert_delta) - h(base)
r_merge  = h(base + gated_deltas) - h(base)
L_residual = weighted_mse(r_merge, r_expert)
```

默认只取 hidden layers `8,16,24,28`，每层取 response token residual norm 最大的 `128` 个 token。这样先抓住 expert 轨迹中 task vector 真正改变表示的位置，避免全 token 平均把信号冲淡。

`L_base_drift`：同一 forward 中 prompt tokens 上约束 merged hidden 不要过度偏离 base hidden，默认 `beta_base=0.02`。它是轻量行为保持项，不是强行把 gate 拉回 0。

`L_gate_anchor`：约束 coefficient 不要离初始 `1.0` 过远，默认 `gamma_gate=0.001`。这个项只防止无界漂移，不主导训练。

## 默认启动

```bash
bash skill/command/run_20260519_trc_layer_init1.sh
```

默认占两张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1
DEVICE_MAP=auto
MAX_MEMORY_ENTRIES="0=70GiB 1=70GiB"
```

默认训练：

```text
epochs=3
lr=0.01
optimizer=adamw
accumulation_steps=8
max_seq_length=1536
max_response_tokens=512
hidden_layers=8,16,24,28
topk_tokens=128
beta_base=0.02
gamma_gate=0.001
```

常用覆盖示例：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_try2 \
EPOCHS=5 \
LR=0.02 \
bash skill/command/run_20260519_trc_layer_init1.sh
```

## 已做验证

- `py_compile` 通过：`scripts/trc/build_trc_calibration_v1.py`、`scripts/trc/train_trc_layer_gates.py`。
- calibration builder 已实际生成 96 条轨迹：Tool/Memory/Code 各 32 条，全部 `reward_train=1.0`。
- trainer dry-run 已通过，能识别 96 行、3 tasks、hidden layers `8,16,24,28`。
- 两卡真实 smoke 已通过：`CUDA_VISIBLE_DEVICES=4,5`、1 row、1 epoch、hidden layers `8,16,24,28`，能完成真实 forward/backward 并写出 `trc_gates.json`。

单卡 smoke 会在安装三专家 delta 时 OOM；正式运行请使用至少两张 80G 卡或进一步缩小 mode/delta。

## 产物

每个 run 输出：

```text
trc_run_manifest.json
trc_metrics.jsonl
epoch_001.gates.json
epoch_002.gates.json
...
trc_gates.json
trc_summary.json
```

`epoch_xxx.gates.json` 和 `trc_gates.json` 都是标准 `{ "gates": ... }` 格式，可以进入现有 bake/eval 链路。

## 判断标准

第一版不是替代 GRPO，而是验证一个问题：expert 成功轨迹的隐藏层 residual 是否能给 gate 一个更直接、更密集、更可解释的学习信号。

优先看：

- gate 是否从 init1 学到 task/layer 结构，而不是所有系数同步平移；
- Tool/Memory/Code 的 gate means 是否与已知能力需求相符；
- bake 后 proxy / official eval 是否至少不崩，并能补足 Code 或 Tool 的短板；
- `L_residual` 是否在 1-3 epoch 内稳定下降。
