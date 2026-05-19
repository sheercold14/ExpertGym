# 2026-05-19 Tool Behavior Null-Space v1

## 目标

本实验解决 L1/L2/L3/L4/L5 中反复出现的 Tool 崩溃问题：Memory/Code 的 OPD 梯度会持续推动 gate，而 Tool 在全对样本上的 NLL retention 太弱，无法保护 tool-call 行为格式。v1 的设计是：在 update 阶段用 Tool 正确轨迹的 behavior span 梯度构造保护子空间，将最终 gate 梯度投影到该子空间的正交补；同时保持 OPD + retention 的主训练逻辑不变。

## 代码改动

| item | path | 说明 |
|---|---|---|
| update null-space | `scripts/train/opvec_update_gates_from_rollouts.py` | 新增可选 `--tool-nullspace-gate-gradients`，默认关闭；开启时在 `clip_grad_norm_` 和 `optimizer.step()` 之前投影当前累计 gate gradient |
| loop passthrough | `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` | 把 null-space 参数传给 update 脚本 |
| launcher env | `skill/command/run_qbank_c033333_gate_strategy.sh` | 暴露 `TOOL_NULLSPACE_*` 环境变量 |
| data builder | `scripts/data/build_tool_nullspace_calibration_v1.py` | 构建 Tool32/Memory32/Code40 calibration 与 Tool replay bank |
| experiment launcher | `skill/command/run_20260519_tool_nullspace_v1.sh` | M1 实验专用启动脚本 |
| tests | `tests/test_pcgrad_gate_gradients.py` | 增加 null-space 投影数学与 Tool behavior span 检测单测 |

默认关闭时不会进入 null-space 分支，不改变 reward、dynamic OPD、retention、task weight、PCGrad、loss 计算或 optimizer 路径。

## 数据

| artifact | path |
|---|---|
| merged prompt manifest | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/tool32_memory32_code40_toolnullspace_seed20260519.prompts.jsonl` |
| extra expert rollouts | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/toolnullspace_extra_expert_rollouts_seed20260519.jsonl` |
| Tool null-space replay | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/toolnullspace_tool_replay_rollouts_seed20260519.jsonl` |
| selection blueprints | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/toolnullspace_selection_blueprints.jsonl` |
| summary | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/toolnullspace_seed20260519.summary.json` |

数据分布：

| task | count | 来源 |
|---|---:|---|
| Tool | 32 | paper96 原始 Tool 16 + BFCL live 16 |
| Memory | 32 | paper96 原始 Memory 32 |
| Code | 40 | paper96 原始 Code 32 + CURE LiveBench/LiveCodeBench hard-vs-TA 8 |

Tool 增量 16 条来自 BFCL live 历史成功轨迹：

| BFCL category | count |
|---|---:|
| live_parallel | 8 |
| live_parallel_multiple | 8 |

Code 增量 8 条：

| source | count |
|---|---:|
| LiveBench | 1 |
| LiveCodeBench | 7 |
| ReasonFlux positive | 7 |
| DeepSeek-R1 positive | 1 |

构建命令：

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_tool_nullspace_calibration_v1.py
```

## 初始 Gate

| expert | init coefficient |
|---|---:|
| tool | 0.333333 |
| memory | 0.333333 |
| code | 0.333333 |
| reasoning | 0.050000 |

路径：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/init_gates/init_layer_band_parameter_28layer_tmc033_r005.json
```

构建命令：

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/modes/build_constant_gate_checkpoint.py \
  --config configs/gated_grpo_4expert_r1math_layer28.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json \
  --gate-parameterization layer-band-parameter \
  --value 0.3333333333333333 \
  --expert-value reasoning=0.05 \
  --output /tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/init_gates/init_layer_band_parameter_28layer_tmc033_r005.json
```

## M1 配置

| field | value |
|---|---|
| run name | `expM1_toolnull_r1init005_layer28_opdret_nullspace_20260519` |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json` |
| config | `configs/gated_grpo_4expert_r1math_layer28.yaml` |
| strategy | `layer-band-parameter` |
| trainable | tool/memory/code/reasoning all trainable |
| samples per prompt | 4 |
| prompts | 104 |
| update scope | epoch |
| optimizer | SGD, lr 0.25, momentum 0.2 |
| GRPO | off, `PPO_LOSS_WEIGHT=0.0` |
| OPD | on, `OPD_LOSS_WEIGHT=1.0`, length-normalized |
| retention | NLL, `RETENTION_LOSS_WEIGHT=1.0`, length-normalized |
| task weights | tool/memory/code = 1/1/1 |
| dynamic OPD require-all | on |
| null-space rows | 16 |
| null-space min rows | 16 |
| null-space rank | numerical rank, `TOOL_NULLSPACE_RANK=0` |

## M1 / M2 对照

两个实验除初始 gate 外完全一致：同一份 104 prompt calibration、同一份 extra expert rollouts、同一份 Tool null-space replay、同一套 OPD + NLL retention + null-space 设置。

| id | run name | init tool/memory/code | init reasoning | GPU | run dir | train log |
|---|---|---:|---:|---|---|---|
| M1 | `expM1_toolnull_r1init005_layer28_opdret_nullspace_20260519` | 0.333333 | 0.05 | 2,3 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expM1_toolnull_r1init005_layer28_opdret_nullspace_20260519` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expM1_toolnull_r1init005_layer28_opdret_nullspace_20260519.train.log` |
| M2 | `expM2_toolnull_init05_r0_layer28_opdret_nullspace_20260519` | 0.500000 | 0.00 | 6,7 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expM2_toolnull_init05_r0_layer28_opdret_nullspace_20260519` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expM2_toolnull_init05_r0_layer28_opdret_nullspace_20260519.train.log` |

M2 init gate:

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/init_gates/init_layer_band_parameter_28layer_tmc05_r0.json
```

M2 启动命令：

```bash
RUN_NAME=expM2_toolnull_init05_r0_layer28_opdret_nullspace_20260519 \
INIT_GATE_CHECKPOINT=/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/init_gates/init_layer_band_parameter_28layer_tmc05_r0.json \
GPU_LIST=6,7 NUM_ITERS=20 \
  bash skill/command/run_20260519_tool_nullspace_v1.sh
```

## 当前状态

更新时间：2026-05-19 16:52 CST

| item | status |
|---|---|
| code implementation | completed |
| data builder | completed; Tool live category balanced 8/8 |
| init gate | completed; R1 reasoning init = 0.05 |
| dry-run | completed; bake / 2-shard vLLM rollout / dynamic OPD / update command all resolved |
| independent review | completed; no blocking issue |
| M1 training | launched in tmux `train_M1_toolnull_20260519`, GPU `2,3` |
| M2 dry-run | completed; resolved init `0.5/0.5/0.5/0.0` and same update args |
| M2 training | launched in tmux `train_M2_toolnull_20260519`, GPU `6,7` |

Run artifacts:

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/expM1_toolnull_r1init005_layer28_opdret_nullspace_20260519
/tmp/shared-storage/OnPolicy/runs/gated_grpo/expM1_toolnull_r1init005_layer28_opdret_nullspace_20260519.train.log
```

启动命令：

```bash
GPU_LIST=0,1 NUM_ITERS=20 \
  bash skill/command/run_20260519_tool_nullspace_v1.sh
```

Dry-run：

```bash
DRY_RUN=1 GPU_LIST=0,1 NUM_ITERS=1 \
  bash skill/command/run_20260519_tool_nullspace_v1.sh
```

## 判断标准

优先看三个指标：

1. Tool proxy 不再在 Memory 上涨阶段快速坍塌到接近 0。
2. Memory gate 可以继续被推到 `0.55+`，同时 Tool gate 不出现异常下降。
3. Code reward 至少不低于 L1/L2 同阶段，并观察新增 CURE hard anchors 是否产生 OPD 动力。

如果 Tool 仍崩，优先检查：

- `tool_nullspace_rows` 是否达到 16；
- `tool_nullspace_rank` 是否过低或过高；
- `tool_nullspace_removed_fraction` 是否接近 0；
- retention loss 是否仍比 OPD 小一个数量级以上。

## 验证

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/train/opvec_update_gates_from_rollouts.py \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  scripts/data/build_tool_nullspace_calibration_v1.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  tests/test_pcgrad_gate_gradients.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  tests/test_update_gates_objectives.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  tests/test_gated_grpo_trust_region.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  tests/test_gated_grpo_utils.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  tests/test_bake_global_coefficients.py
```

当前验证结果：`py_compile` 通过；上述 direct unittest 入口通过。当前 BFCL/easyrl/system Python 均未安装 `pytest`，所以没有使用 `python -m pytest` 入口。
