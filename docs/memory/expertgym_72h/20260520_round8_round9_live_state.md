# 20260520 Round8/Round9 Live State Memory

## 当前目标

目标不是单点把某个 gate 推高，而是在固定 Tool/Memory 快速门槛下寻找能转化到正式 Code/CURE 的 TRC 合并模型。

快速门槛：

- Tool BFCL category mean `>= 0.79`
- Memory HotpotQA mean F1 `>= 0.76`
- 通过后才进入昂贵 Code/CURE 评测

## 已知规律

1. R5C/R7B 这类 v2 response span 已经能稳定保 Tool/Memory，但 Code 提升有限。
2. R6B eval-leak diagnostic 能把 LiveBench Code 提到 `0.3750`，说明 Code 能力可以被对齐轨迹学习到；问题在于非泄露 calibration 的 Code trajectory 与 CURE/LiveCodeBench 能力仍不完全一致。
3. R8A/B/R8E 的 CodeP0-v3 训练动力健康，Code/Tool gate 可推到约 `1.24/1.23`，Memory gate 维持在 `0.993` 附近，但 R8A e12 Memory 均值只有 `0.7563`，说明 e12 可能过推，不能只看 gate mean。
4. R8C relative-MSE 是负结果：loss 尺度过大，epoch2 就把 Memory gate 拉到 `0.9618`，不作为主线。

## 当前实验组

- R8A：RF-only CodeP0 response，e12 quick eval 已完成，Tool `0.8035`，Memory `0.7563`，e12 暂不进 Code。
- R8A-e08/e10：已 bake，正在做 Memory-only，验证 R8A 是否早停更好。
- R8B：RF + DeepSeek fallback CodeP0 response，Tool `0.7944`，Memory running。
- R8D：RF-only CodeP0 code-block span/topK384，Tool `0.7944`，Memory running。
- R8E：role-quota CodeP0 response，Tool `0.7944`，Memory running。
- R9A：RF-only CodeP0 response topK128，training。
- R9B：role-quota CodeP0 response topK128，training。

## 决策逻辑

1. 如果 R8B/R8D/R8E 任意一个 Memory `>=0.76`，立即进入 Code/CURE。
2. 如果 R8A-e08/e10 Memory 明显高于 e12，补跑 Tool；若 Tool 过线，进入 Code。这检验“过推导致 Memory 掉线”的假设。
3. 如果 R9A/R9B 的 Tool/Memory 通过，再和 R8A/R8E 比较 Code，判定 Code response topK 是否应该收缩。
4. 若多个候选 Tool/Memory 相近，优先送 Code 的顺序：R8E/R9B role-quota > R8B RF+DS > R8D code-block > R8A early。

## 下一轮可选实验

优先级从高到低：

1. role-quota + topK128 的 R9B 当前正在跑，是最直接组合 R8E 和 R9A 的候选。
2. 若 R9A/R9B Memory 仍掉，开早停选择而不是继续增加 epoch。
3. 若 Code 仍不上升，下一轮应改 calibration selection，而不是继续调 gate loss：用非泄露 CodeP0 中“当前 merged model 错、ReasonFlux 至少一次对”的 frontier bucket，提高可恢复比例。
4. 不再使用 relative-MSE 作为主 objective；只保留 directional residual / projection 系列。
