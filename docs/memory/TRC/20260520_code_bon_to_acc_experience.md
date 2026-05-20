# 20260520 Code BoN-to-Acc 经验沉淀

## 结论摘要

R5A 与 R11B 的 Code 结果都呈现同一现象：BoN 明显高于 single-sample Acc。当前最有价值的方向不是继续无脑推高 code coefficient，而是把“偶尔能采到正确答案”转化为“默认一次生成更稳定正确”。

代表结果：

| model | Tool | Memory F1 | LiveBench Acc/BoN | LiveCodeBench Acc/BoN | mean Acc/BoN |
|---|---:|---:|---:|---:|---:|
| R5A | 0.8035 | 0.7638 | 0.3672 / 0.4844 | 0.2715 / 0.3777 | 0.3194 / 0.4310 |
| R11B | 0.7944 | 0.7715 | 0.3750 / 0.5000 | 0.2794 / 0.3620 | 0.3272 / 0.4310 |

R11B 的 single-sample Acc 更高，R5A 的 Tool 更稳。两者都不是“Code 完全没能力”，而是存在明显 BoN-to-Acc gap。

## 当前实验策略

### R15A: 复现 R5A

目的：排除训练随机性，确认 R5A 是稳定配置而不是偶然 checkpoint。

关键设置：完全复用 R5A 的 calibration、loss、span、floor、LR、epoch。

判据：

- 如果 R15A gate/loss/Tool/Memory/Code 接近 R5A，说明 R5A 可作为稳定 anchor。
- 如果 Code repeat 波动明显，后续需要把评测方差纳入模型选择。

### R15B: R5A + contrast hard anchors

目的：测试同一 prompt 的失败轨迹能否把 BoN 转成 single-sample Acc。

实现：在 R5A Tool/Memory 基础上，Code 使用 24 条原训练成功轨迹 + 8 条 formal hard contrast anchors。训练侧新增 `negative_response` hinge：

```text
L_contrast = relu(margin + L_positive_residual - L_negative_residual)
```

当前观察：contrast loss 有信号但幅度很小，说明第一版 contrast 是干净的诊断分支，但不一定足够强。

## 经验

1. Code gate 高不是充分条件。R5A/R11B/R14B 都能把 code gate 推到 1.1-1.24 区间，但 formal Acc 仍卡在 0.32-0.33 左右。
2. Code data 需要保留 train-only distribution，否则 Tool/Memory 容易被 formal/eval-like Code 牵扯；但只用 train distribution 又不能解释 LiveBench/LiveCodeBench。
3. Negative contrast 应优先选同 prompt、同格式、同 evaluator 下的失败回答；随机失败回答的方向性太弱。
4. 如果 BoN 高而 Acc 低，方法上应引入稳定输出/selection/contrastive preference，而不是只加更多 successful trajectory。
5. Tool/Memory quick gate 仍是硬约束。任何 Code 方案如果 Tool < 0.79 或 Memory F1 < 0.76，不进入昂贵 Code 评测。

## 下一步

- 等 R5A/R11B repeat：判断 BoN-to-Acc gap 是否稳定。
- 等 R15A：确认 R5A 复现性。
- 等 R15B：若 Tool/Memory pass，送 Code，直接比较 R5A/R11B/R15B 的 Acc 与 BoN。
- 如果 R15B contrast 太弱，下一版应提高 hard anchors 数量或改成 pairwise selection-style loss，而不是继续加 code gate floor。

## 2026-05-20 14:06 CST 追加观察

- R5A repeat 的 LiveBench 单次 Acc 为 `0.3906`，BoN 为 `0.4688`。这比旧记录 `0.3672 / 0.4844` 有可见波动，说明 Code formal eval 的随机性足以影响一次实验判断。
- 因此以后筛选 Code 模型时不能只看一次 LiveBench 子项；至少要看完整 LiveBench + LiveCodeBench mean，关键候选要做 repeat 或看 BoN gap。
- R14B Tool quick gate 通过，均值约 `0.7944`；说明 mixed train+contrast 没立刻破坏 Tool。Memory 和 Code 正在补测。
- Tool 评测需要同时看 BFCL 与 ToolRL all80。BFCL live 子类样本很少，`live_parallel` 只有 16 题，单题影响 `0.0625`；`parallel` 有 200 题，单题只影响 `0.005`。因此“live_parallel 提升但 parallel 小降”可能是真实泛化差异，也可能只是小样本波动。ToolRL all80 用来判断源 ToolRL tool-call 行为是否还在。
- R14B/R15A/R15B 的 ToolRL all80 全部为 `0.6375` success，mean reward 约 `0.835-0.836`。这高于 TA0.75 历史参考 `0.6125` 和 best-ever 参考 `0.6250`，说明当前 TRC/contrast 设置没有破坏 ToolRL 源分布能力；BFCL parallel/live 的差异应单独作为 heldout 泛化问题处理。

## 关于 588 系数与 Code contrast 的当前判断

- 当前 TRC trainer 只支持 `layer-band-coefficient`，也就是 `28 layer x 3 expert = 84` 个直接系数；mode manifest 有 `196` 个 mergeable parameter，每个 3 expert，即 `588` 个系数。
- 直接切 588 需要把 `scripts/trc/train_trc_layer_gates.py` 放开 `parameter` manager，并确保 `temporary_direct_coefficients`、floor loss、gate summary、bake/selection 都支持 parameter-key。实现成本不大，但实验风险高：96 条 calibration 对 588 参数明显欠约束，容易记住 span 而不是得到可泛化能力。
- 更合理路线：先用 same-prompt pass/fail direction loss 证明 Code signal 有效；如果有效，再做 588 的受限版本，例如只放开 Code-relevant 层/模块或加 group sparsity/L1，而不是一次性全 588 自由学习。
