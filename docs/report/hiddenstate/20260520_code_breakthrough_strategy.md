# 20260520 Code 破局策略

## 背景

当前 R5A/R11B/R12E 等候选都显示：

- Tool/Memory 可以通过 quick gate；
- Code single Acc 卡在约 `0.32-0.33`；
- Code BoN 可到 `0.43+`，LiveBench 上甚至有 `0.50`。

因此 Code 不是完全没有能力，而是“同一题能采到正确答案，但默认输出不稳定”。继续只推高 code gate 或继续靠近 expert positive residual，收益有限。

## 主线目标

把 Code ability loss 从：

```text
merged residual 靠近 expert positive residual
```

升级为：

```text
同 prompt 下 pass code block 的方向 > fail code block 的方向
```

评估目标不是训练 loss 更低，而是：

1. Tool BFCL mean >= `0.79`；
2. Memory mean F1 >= `0.76`；
3. Code mean Acc 明显超过当前强候选 `0.327`；
4. BoN-to-Acc gap 变小，不能只让 BoN 涨。

## 当前 contrast 分支的问题

R15B 已经启用 `negative_response` hinge：

```text
relu(margin + L_positive_residual - L_negative_residual)
```

但它有三个不足：

1. negative rows 只有 8 条，训练后期 active rate 降到约 `0.0625`，信号太弱；
2. loss 是 residual-loss margin，不是直接的 direction/projection score margin；
3. negative_response 多来自 formal diagnostic anchors，不能直接作为 paper-main 主结果。

R15B 因此更像“干净诊断”，不是最终 Code 破局版本。

## 下一轮实验设计

### R16A: Non-leak Code pass/fail contrast

数据：

- Tool32/Memory32 复用 R5/R11 稳定版本；
- Code32 使用 train-only / CodeP0 prompt；
- 每个 Code prompt 尽量保留：
  - 一个 expert/pass trajectory；
  - 一个 same-prompt fail trajectory；
  - 失败轨迹来自当前强 merged model 或 ReasonFlux/DeepSeek 失败样本；
  - pass/fail 由 train-only generated tests 或原始 public/private-like guard tests 判定。

loss：

- 保留 positive directional residual；
- 加强 same-prompt negative contrast；
- contrast rows 目标至少 24/32，而不是 8/32。

判据：

- ToolRL all80 不低于 `0.6125`；
- BFCL Tool quick gate 通过；
- Memory quick gate 通过；
- 若 Code BoN-to-Acc gap 缩小，进入 CURE。

### R16B: Score-margin direction loss

把 contrast 从 loss-margin 改为 score-margin：

```text
score_pos = projection/cosine(merged_residual, pass_expert_residual)
score_neg = projection/cosine(merged_residual, fail_residual)
L = relu(margin - score_pos + score_neg)
```

优势：

- 直接表达“pass direction 排在 fail direction 前面”；
- 不依赖 residual loss 的绝对尺度；
- 更适合解释 BoN-to-Acc。

风险：

- 需要轻微改 `scripts/trc/train_trc_layer_gates.py`；
- negative residual 的定义要谨慎，第一版可继续用 forced hidden on negative code block，作为可控 baseline。

### R16C: 受限 588 / module-level coefficients

不建议直接全 588 自由学习。最小可行版本：

- 先只放开 Code 相关模块，或只放开 selected layers 的 q/k/v/o + mlp；
- 参数仍从 init=1；
- 加 L1/group sparsity 或 residual prior；
- calibration 至少需要更强 pass/fail 信号，否则 96 条样本对 588 系数欠约束。

判断：

- 如果 R16A/B 没有让 Code proxy/Code quick eval 上升，588 不应优先；
- 如果 R16A/B 有效，再用 588 做结构化定位，解释哪些层/模块负责 Code 稳定性。

## 结论

今晚主线不要被 588 打断。先证明 Code 的训练信号能把 BoN 转成 Acc；一旦 pass/fail direction loss 有效，再让 588 作为结构发现和性能上限探索。

