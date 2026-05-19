# 2026-05-18 R1 Layer Hierarchy 与 3band 续训

## 目标

验证 28-layer gate 训练不动是否因为缺少类似 `global-parameter` 的全局 expert 聚合项；同时把当前最有效的 R1 3band run 从最新 checkpoint 继续训练 10 个 iteration，观察 memory / reasoning gate 是否继续上升、reward 是否饱和或回落。

## 实验 A：28layer hierarchical

| 项 | 值 |
|---|---|
| run | `expD_r1scaled_layer28_hier_20260518` |
| tmux | `train_r1_layer28_hier_20260518` |
| GPU | `2,3` |
| mode | `/tmp/shared-storage/OnPolicy/modes/opvec4_r1scaled_20260518/mode_manifest.json` |
| config | `configs/gated_grpo_4expert_r1scaled_layer28.yaml` |
| strategy | `layer-band-parameter` |
| gate 形式 | `coeff[layer, expert] = global[expert] + residual[layer, expert]` |
| init | tool/memory/code `1/3`，reasoning `0` |
| LR | `0.25` |
| loss | OPD `1.0` + NLL retention `0.5`，GRPO off |
| task weight | tool `0.5` / memory `2.0` / code `1.5` |
| num iters | `10` |

命令：

```bash
PHASE=layer28_hier \
GPU_LIST=2,3 \
RUN_NAME_LAYER28=expD_r1scaled_layer28_hier_20260518 \
NUM_ITERS=10 \
bash scripts/train/run_4expert_r1scaled.sh
```

## 实验 B：3band continue10

| 项 | 值 |
|---|---|
| run | `expD_r1scaled_3band_continue10_20260518` |
| tmux | `train_r1_3band_continue10_20260518` |
| GPU | `4,5` |
| base checkpoint | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expD_r1scaled_3band_noold_20260518/iter_010/gate_updates.gates.json` |
| config | `configs/gated_grpo_4expert_r1scaled.yaml` |
| strategy | `layer-band` |
| LR | `0.25` |
| loss | OPD `1.0` + NLL retention `0.5`，GRPO off |
| task weight | tool `0.5` / memory `2.0` / code `1.5` |
| num iters | `10` |

命令：

```bash
PHASE=3band \
GPU_LIST=4,5 \
RUN_NAME_3BAND=expD_r1scaled_3band_continue10_20260518 \
INIT_GATE_CHECKPOINT_3BAND=/tmp/shared-storage/OnPolicy/runs/gated_grpo/expD_r1scaled_3band_noold_20260518/iter_010/gate_updates.gates.json \
NUM_ITERS=10 \
bash scripts/train/run_4expert_r1scaled.sh
```

## 重点监控

- 28layer hierarchical 的 `global` 聚合项是否能让 gate delta 从旧版 `~0.0016/iter` 提升到 `0.01+ / iter`。
- 28layer hierarchical 是否复制 3band 的 early reward jump；如果 gate 能动但 reward 不涨，说明 layer 级 credit assignment 仍然不可靠。
- 3band continue 是否继续推高 memory / reasoning；若 tool 或 code 开始回落，需要考虑 retention 或 task weight 收紧。
