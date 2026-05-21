# 20260520 Init=1 Code Residual Learning Review

## 核心问题

当前目标不是让 Code gate 固定上升或下降，而是判断 TRC hidden residual 是否抓到了正式 Code 评测需要的关键子能力。如果 residual 只是让 gate 有方向、loss 下降，但 CURE / LiveBench / LiveCodeBench Acc 不升，它就不是有效能力信号。

## 已有规律

### 1. Init=1 下 Code residual 有稳定梯度

R5/R16/R17/R19 都显示 Code 行的 hidden residual loss 明显大于 Tool/Memory：

- Code residual 常在 `1.3-1.8` 区间；
- Tool residual 约 `0.45-0.50`；
- Memory residual 极小，约 `0.005-0.02`。

这说明 Code 不是“没有梯度”，而是“梯度是否对应正式评测能力”仍未解决。

### 2. Code gate 上升不是充分条件

历史强信号实验中，Code gate 常从 `1.00` 稳定涨到 `1.20-1.24`：

- R5A/R5C/R16B/R17A/R17B 都有类似 gate 轨迹；
- 但 Code formal Acc 多在 `0.31-0.33`；
- 静态 / TA / TAME 类 baseline 可到约 `0.34-0.36`。

因此不能把“Code gate 被推高”当作成功，必须以正式 Code eval 决策。

### 3. Response span 比 code-block-only 更可能包含能力，但噪声也更大

R5C 的 `code=response` 比 R5A 的 `code=code-block` 在 Code Acc 上略好；R19 的早期 loss 也显示：

- `code=response` span tokens 更长，覆盖 reasoning + final code；
- `code=code-block` residual 更容易下降，但可能只学最终代码风格；
- 正式 Code 需要算法理解、边界条件、输出格式、稳定实现，不一定只在最终 code block。

### 4. Prompt residual 很强，但可能主导训练

R19B 的 prompt residual 在 Code 行可到 `~2.0`，明显大于 response residual 的辅助量级。它可能学习 prompt 理解，也可能变成校准 prompt overfit。需要正式评测验证，不能只看 loss。

### 5. Memory 稳定依赖 task-expert coefficient floor

R19A/B/D 保留 R5/R16 风格的 `TASK_EXPERT_COEFFICIENT_FLOOR=1.0, weight=50`，Memory gate 在第 2-3 轮会回到约 `0.99`。

R19C 关闭该 floor 后，Memory gate 第 3 轮已经降到约 `0.94`。这说明 pure residual 会牺牲 Memory，之前稳定的三项能力部分来自 floor 保护，不是 residual 自动解决了多任务平衡。

## R19 当前实验

Calibration:

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round19_init1_code_steer_balanced/trc96_balanced_code_mixed.jsonl
```

四组均从 init=1 出发，Code 使用混合 train/eval-like prompt，Tool/Memory 使用稳定轨迹。

| ID | 核心设置 | 诊断问题 |
|---|---|---|
| R19A | `code=response`, topK256, contrast 1.5 | reasoning+code span 是否比 code-block 更有效 |
| R19B | R19A + prompt residual 0.15 | prompt 理解 residual 是否帮助 Acc |
| R19C | R19A 但关闭 task floor | gate 动力是否来自 residual 本身 |
| R19D | `code=code-block`, topK384, contrast 3.0 | final code + 强 pass/fail contrast 是否更有效 |

## 第 1-3 轮观测

- 四组 Code gate 都约 `+0.02/epoch`，说明当前 residual/contrast 仍在推高 Code expert direction。
- R19A/R19C 第 1 轮 loss 完全一致，说明第一步不是 floor 主导。
- R19C 第 3 轮 Memory gate 下降到 `0.94`，说明关闭 floor 会快速损伤 Memory telemetry。
- R19D 的 code-block residual 更低，但这可能是“更容易拟合”，不代表更强能力。
- R19B 的 prompt residual loss 很大，需要警惕 prompt loss 抢占 response/code 能力学习。

## 第 7 轮 layer-band 观测

R19A/B/D 的 Code gate 在 28 层几乎等幅上升：

| run | code mean | code min | code max | spread |
|---|---:|---:|---:|---:|
| R19A | 1.1397 | 1.1379 | 1.1401 | 0.0022 |
| R19B | 1.1397 | 1.1376 | 1.1401 | 0.0025 |
| R19D | 1.1398 | 1.1385 | 1.1401 | 0.0016 |

Memory / Tool 的 layer spread 明显更大，而 Code 几乎是全层同向同幅移动。这是当前最重要的诊断信号：当前 AdamW + residual objective 更像在做 global Code scaling，而不是学习 Code 子能力所在的关键层。

因此，如果正式 Code 仍不升，失败原因更可能是优化器/目标把 Code residual 压成了“全层统一缩放”，没有让 layer-band 结构发挥作用。后续需要用 SGD 或更显式的 layer/ability residual selection 来保留梯度幅度差异。

## 下一步决策原则

1. 让 R19A/B/D 跑完并 bake；R19C 作为 pure-residual 对照，若 Memory gate 继续下滑可提前停止。
2. 先做 Tool/Memory quick eval，只有通过 `Tool mean >= 0.79`、`Memory F1 >= 0.76` 的模型进入 expensive Code。
3. Code eval 后对比：
   - 如果 R19A > R19D，说明 reasoning span 比 final code span 更重要；
   - 如果 R19B > R19A，说明 prompt residual 是有效子能力；
   - 如果 R19D > R19A，说明强 pass/fail + final code block 更接近评测能力；
   - 如果都不升，说明 hidden residual 当前目标仍未对齐 Code eval，需要转向 execution-aware / test-aware residual selection。
