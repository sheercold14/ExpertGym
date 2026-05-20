# 20260520 Round8 记忆：CodeP0-v3 非泄漏校准

## 当前问题

R3D 是目前非泄漏主评测 Code 最好点：Tool `0.7944`，Memory `0.7636`，Code mean acc `0.3289`。R5/R7 的 v2 校准能把 Tool/Memory 保住，并提高部分 Code BoN，但主采样 Code acc 没有超过 R3D。Case study 显示 Code 不是单纯“代码块 token”问题，而是算法选择、IO 约束、边界样例和失败修复等 execution-style 能力没有被校准轨迹充分覆盖。

## Round8 假设

换数据比继续调 gate 更关键。`code_p0_v3` 是非泄漏、通过专家多采样验证的可恢复 Code prompt 集，ReasonFlux 有 29 个唯一成功 prompt、161 条成功样本；DeepSeek 合并池提供额外覆盖到 31 个唯一成功 prompt。Round8 先测试两条线：

- `rf_only_late3`：Code 只用 ReasonFlux 成功轨迹，和当前 `code` task vector 的专家来源一致，减少异质专家轨迹对 ReasonFlux delta 的误导。
- `rf_then_ds_late3`：ReasonFlux 优先，不够的用 DeepSeek 补齐，获得 32 个唯一 Code prompt，测试覆盖度是否比专家一致性更重要。

## 训练设置

沿用 R5C/R7 的稳定配置：init `1.0`、layer-band coefficient、Tool tool-call span、Memory late3 trajectory turns + final、Code response span topK 256、directional residual、task floor weight 50。这样 Round8 的主要变量就是 Code 轨迹源，而不是 loss 配方。

## 决策规则

先跑 Tool/Memory。Tool mean `>=0.79` 且 Memory mean F1 `>=0.76` 才送 CURE Code。若 R8A 明显优于 R8B，说明专家-vector 一致性更重要；若 R8B 更强，说明 Code prompt 覆盖度更重要。若二者 Code 都不上升，下一步应引入更结构化的 non-leak execution trajectory：算法选择、IO 格式、失败修复/反例解释，而不是继续增大 code gate。
