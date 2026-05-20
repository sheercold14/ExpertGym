# 20260520 Eval-Leak Residual Direction Plan

## 核心判断

当前 Code 的主要问题不是“不允许用评测集”，而是训练信号仍然接近：

```text
merged residual -> successful expert residual
```

这个目标只能说明“靠近某个 expert 正轨迹”，不能说明当前 gate 是否已经加过头。因此它天然倾向于把 Code gate 往 code expert 的 full delta 方向推。如果真实最优组合在 `0.5-0.75`，单向 positive residual 会把 Code gate 推过头，或者在不正确 span 上推高，最终表现为 gate 高但 Code official acc 不涨。

新的原则是：评测集泄露可以使用，但监督对象必须是能力方向和局部最优系数，而不是答案本身。也就是说，eval prompt / eval correct trajectory 可以作为“能力探针”，但训练目标要回答：

```text
在这个 prompt 的关键 span/layer 上，哪个 expert delta 的残差方向能解释成功轨迹？
当前 gate 相对该方向是偏低、偏高，还是方向错了？
```

## 目标

1. 不从 `0.75` checkpoint 开始训练。
2. 允许使用 eval prompt / eval successful trajectory / eval failure trajectory。
3. 不把答案监督当作主线，而把轨迹 residual 转成 gate 的可签名动力。
4. Code gate 必须可以被压低；如果高 Code gate 与失败轨迹更一致，loss 应该反向推动。
5. 最终 checkpoint 仍按 official Tool / Memory / Code 评测选择，而不是按 gate 系数选择。

## Residual 动力设计

### 1. 局部系数拟合

对同一个 prompt 的成功轨迹，取关键 response span 或 reasoning+code span，计算：

```text
r_success = hidden(success_model, prompt + success_response) - hidden(base, prompt + success_response)
v_tool    = hidden(base + delta_tool, same text) - hidden(base, same text)
v_memory  = hidden(base + delta_memory, same text) - hidden(base, same text)
v_code    = hidden(base + delta_code, same text) - hidden(base, same text)
```

然后在每个 row / layer / span 上解一个小的 ridge least-squares：

```text
alpha* = argmin_alpha || alpha_tool v_tool + alpha_memory v_memory + alpha_code v_code - r_success ||^2
         + lambda ||alpha - alpha_init||^2
```

这一步给出的 `alpha*_code` 是轨迹层面的“局部最优 Code 系数估计”。如果 `alpha*_code < current_alpha_code`，该样本会产生压低 Code gate 的信号；如果 `alpha*_code > current_alpha_code`，才推高 Code gate。

关键点：`success_model` 不必只用 code expert，可以是 TA0.75、best-ever model、ReasonFlux、R1、或者当前模型 BoN 中 pass 的 sample。TA0.75 在这里不是初始化点，而是“成功能力方向 witness”。

### 2. Pass/Fail 方向排序

对同一个 prompt 同时保存成功轨迹和失败轨迹：

```text
r_pass = hidden(pass_response) - hidden(base)
r_fail = hidden(fail_response) - hidden(base)
r_gate = hidden(current_gate on pass/fail text) - hidden(base)
```

用方向排序而不是单纯 residual MSE：

```text
score_pass = cos_or_projection(r_gate, r_pass)
score_fail = cos_or_projection(r_gate, r_fail)
L_margin = relu(margin - score_pass + score_fail)
```

如果高 Code gate 更接近失败轨迹，该 loss 会推动 gate 远离失败方向。这个机制比“靠近 positive residual”更适合把 BoN 转成 single-sample acc。

### 3. Span 选择

Code 不应只看 final code block。LiveBench / LiveCodeBench 还需要：

- prompt 约束理解；
- edge-case reasoning；
- IO parsing；
- complexity / algorithm selection；
- final code formatting。

建议第一版 span：

```text
prompt constraint span: 题目中输入输出、限制、边界条件附近 token
reasoning span: 成功轨迹中计划、算法选择、关键不变量附近 token
final code span: imports / parse input / main logic / output formatting
```

Prompt span 不要拉回 base。它要么不加，要么和 response 一样做 success residual / pass-fail residual，因为 task vector 可能本来就包含题目理解方向。

## Calibration 构造

评测泄露可以分两类使用。

### A. 诊断型泄露

用于分析，不一定进入主实验：

- 直接从 LiveBench / LiveCodeBench / BFCL official eval 中取错题；
- 找 TA0.75 / best-ever / ReasonFlux / R1 / BoN pass 的正确轨迹；
- 同 prompt 保存 current fail trajectory；
- 计算 `alpha*` 分布和 pass/fail cosine gap。

输出应该回答：

```text
Code 正确轨迹需要的 alpha_code 中位数是多少？
失败轨迹是否对应 alpha_code 过高？
Memory/Tool 的 alpha 是否被 Code span 错误解释？
```

### B. 训练型泄露

允许进入实验，但要保持 residual 解释合理：

- 每条样本必须有 correct trajectory；
- 尽量有 same-prompt fail trajectory；
- 不要求 unique prompt 绝对少，但要按能力桶均衡；
- 训练集报告必须写明来自 eval prompt，不能伪装成 non-leak 主结果。

Code 建议 32-48 条，按能力桶：

| bucket | 目的 |
|---|---|
| IO parsing / format | 对齐 LiveCodeBench 常见 stdin/stdout 错误 |
| edge case | 修复 public example 过、hidden-like 不过 |
| DP / graph / math / greedy | 覆盖算法选择 |
| long constraint | 覆盖长题意与限制理解 |
| pass/fail BoN gap | 把能采到但默认不稳定的题变成排序信号 |

Tool 建议继续保 32 条，但 BFCL non-live 要更重：

- `parallel` / `parallel_multiple` 多放；
- 保存 tool-call span；
- success trajectory 必须是严格可 parse、argument match、count/order correct；
- fail trajectory 标注 `cannot_find_match` / `wrong_count` / `decoder_failed`。

Memory 要保留完整或近似完整 trajectory，不能只看 final answer，否则会丢 update-turn 行为 span。

## 训练阶段建议

### Phase 0: Residual 诊断

先不训练，跑一个诊断脚本：

1. 对 Code eval-leak rows 计算 `alpha*`。
2. 统计每个 bucket 的 `alpha*_tool / alpha*_memory / alpha*_code`。
3. 统计 pass vs fail 的 residual cosine gap。
4. 对比当前 strong model gate，判断 Code 是“该降”还是“该换 span / 换 teacher”。

判据：

- 如果 `alpha*_code` 大量低于当前 gate，说明应加入 coefficient-fit / margin，不能继续单向推高。
- 如果 `alpha*_code` 仍高，但 acc 不涨，说明 code delta 本身或 teacher trajectory 不对应 official ability。
- 如果 pass/fail cosine gap 很小，说明 hidden layer/span 不对，需要换 attention/constraint/code span。

### Phase 1: Gate 训练

用当前 TRC 训练框架增加一个可选目标：

```text
L_total =
  w_dir * L_directional_positive
+ w_fit * ||alpha_gate - alpha*||^2 或 ||sum alpha v - r_success||^2
+ w_margin * relu(margin - score_pass + score_fail)
+ w_sparse * L1/group sparsity
```

默认不影响旧实验。新目标只在 leak residual experiment 中开启。

### Phase 2: 选择与评测

选择规则仍然是：

1. Tool quick >= 0.79；
2. Memory F1 >= 0.76；
3. Code official acc / BoN；
4. 如果 Code BoN 高但 Acc 低，继续加强 pass/fail sorting，而不是单纯加 Code gate。

## 为什么这符合方法角度

这个方案不是“用评测集把模型训会答案”，而是把评测轨迹当作能力显微镜：

- 正确轨迹提供目标能力方向；
- 失败轨迹提供反方向或错误方向；
- task vectors 提供可解释 basis；
- gate 学到的是哪个 basis 在哪个层/任务 span 上解释成功行为；
- 如果某个 expert delta 过强，残差拟合会自然压低它。

这比从 `0.75` 开始训练更合理，因为 `0.75` 只是经验系数；这里学习的是每条轨迹、每个 span 对 expert residual basis 的局部投影。

## 下一步最小可执行

1. 写 `alpha*` 诊断，不改训练。
2. 用 R16B / best-ever / TA0.75 / ReasonFlux 成功轨迹做 Code 诊断。
3. 若 `alpha*_code` 支持“过高”，再加 coefficient-fit loss。
4. 若 `alpha*_code` 不支持“过高”，优先改 span / teacher / Code ability bucket。
5. 任何训练结果都必须进入 Tool/Memory gate，再决定是否做 Code official。
