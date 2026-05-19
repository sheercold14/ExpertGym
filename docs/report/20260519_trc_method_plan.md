# 20260519 TRC-Merging 方法计划

## 核心判断

过去 GRPO/OPD 路线的主要问题不是“没有 loss”，而是 gate 的训练信号经常被 reward 饱和、all-fail 稀疏、任务间轨迹长度和 expert 轨迹覆盖率共同削弱。TRC-Merging 换一个角度：不直接问当前输出 reward 是否变好，而是用 expert 已经成功的轨迹，监督 merged model 在关键 token/layer 上复现目标 expert task vector 带来的隐藏状态 residual。

这条线的价值在于：它仍然只学习 gate，仍然利用 task vector，但 calibration 信号从 sparse scalar reward 变成 dense representation target。

## 第一版实验设计

### Calibration

保持 96 条规模，避免变成大训练集：

- Tool: 32 条 ToolRL expert 成功轨迹。
- Memory: 32 条 MemAgent expert 成功轨迹。
- Code: 32 条 code expert 成功轨迹，优先 ReasonFlux，必要时用 R1/旧 code expert fallback。

这里的 96 是 expert trajectory 数，不强制 96 个 unique prompt。原因是当前 Tool/Memory 的 paper96 expert rollout 中严格成功 prompt 少于 32 个，如果强制 unique prompt，会引入失败轨迹，和 TRC 的“成功轨迹 residual target”定义冲突。

### 参数化

使用 `layer-band-coefficient`，但 layer band 是 28 个单层 band：

```text
theta ∈ R^(28×3)
experts = [tool, memory, code]
init(theta) = 1.0
```

不用 common+residual，不接旧 GRPO update，不加 R1 reasoning expert。这样第一版只验证 TRC 本身是否能学出结构化 gate。

### Loss

对一条 `(prompt, expert_response, target_expert)`：

```text
h0 = hidden(base, prompt + response)
hi = hidden(base + delta_i, prompt + response)
hm = hidden(base + Σ_j theta_j delta_j, prompt + response)

r_i = hi - h0
r_m = hm - h0
L_res = mean_topk_token_layer(||r_m - r_i||^2)
```

token 权重使用 `||r_i||`，直觉是：目标 expert task vector 自己改变越大的 token，越应该成为 gate 学习的关键证据。默认只取 hidden layers `8,16,24,28` 和每层 top-128 response tokens，降低显存与噪声。

辅助项：

```text
L_base = prompt token hidden drift from base
L_gate = ||theta - theta_init||^2
L = L_res + 0.02 L_base + 0.001 L_gate
```

`L_base` 防止在 prompt 行为空间过度破坏，`L_gate` 防止系数无界漂移；二者都不应压过 `L_res`。

## 复现命令

先构造 calibration：

```bash
bash skill/command/build_20260519_trc_calibration_v1.sh
```

dry-run 检查：

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
$PY scripts/trc/train_trc_layer_gates.py \
  --config configs/gated_grpo_layer28.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --calibration /tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl \
  --output-dir /tmp/shared-storage/OnPolicy/runs/trc/dryrun \
  --dry-run
```

正式启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash skill/command/run_20260519_trc_layer_init1.sh
```

## 预期观察

成功迹象：

- `trc_metrics.jsonl` 中 epoch summary 的 `mean_residual_loss` 连续下降；
- gate means 不只是一起下降/上升，而是在任务和层上分化；
- Tool gate 在 tool response 相关层保持，不被 Code/Memory residual 牵走；
- Memory gate 不低于强 memory 能力需要的区域；
- Code gate 若仍无效，说明 code delta 本身或 code residual target 与评测能力不一致，而不是 reward sparse 问题。

失败迹象：

- 所有系数从 1.0 同步向 0 或边界移动：`L_base` 过强或 residual target 不区分；
- loss 下降但 eval 崩：hidden residual 拟合了轨迹表面，不等价于解题能力；
- Code 系数仍无结构：ReasonFlux task vector 可能太稀疏，或 code 能力不在当前 OP-VEC mode 中。

## 后续分支

如果第一版 loss 能下降但 eval 不涨：

1. 把 response token top-k 从 residual norm 改为 logprob-gain span，即 expert 相对 base 更确定的 token。
2. 对 Tool 单独抽 tool_call span 做 residual target，避免普通解释文本主导。
3. 对 Code 使用 unit-test positive 的 code block span，而不是整段解释。
4. 加 offline cache：预先缓存 `h0` 和 `hi`，训练时只 forward merged model，把 3 forward/row 降到 1 forward/row。

如果第一版 gate 有合理结构且 eval 有提升：

1. 和 GRPO/OPD 组合：TRC 作为第一阶段结构化初始化，GRPO 作为第二阶段 reward refinement。
2. 加 R1/code 异质 expert，但先做 delta norm 缩放，再加入 TRC target。

## 首次 Init1 运行结果

运行目录：

```text
/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_20260519
```

命令设置：

```text
init_value=1.0
epochs=3
calibration=trc96_expert_trajectories.jsonl
gate_parameterization=layer-band-coefficient
hidden_layers=8,16,24,28
max_seq_length=1536
max_response_tokens=512
topk_tokens=128
beta_base=0.02
gamma_gate=0.001
lr=0.01
accumulation_steps=8
```

训练动态：

| epoch | mean residual loss | mean total loss | tool mean | memory mean | code mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6378 | 0.6540 | 1.0743 | 0.8946 | 1.0980 |
| 2 | 0.4713 | 0.4857 | 1.1233 | 0.7940 | 1.1990 |
| 3 | 0.3300 | 0.3427 | 1.1426 | 0.7116 | 1.2871 |

结论：

- TRC hidden-residual signal 是可优化的，3 epoch 内 residual loss 单调下降。
- gate 出现明显分化，不是所有系数同步平移。
- 当前 v1 会强推 Code、轻推 Tool、压低 Memory；这不一定符合最终能力最优，需要 bake + official eval 判断。
- 如果 Memory eval 下降，下一版应加入 task-balanced residual scale 或 Memory-specific floor/anchor，而不是直接沿用该 checkpoint。
