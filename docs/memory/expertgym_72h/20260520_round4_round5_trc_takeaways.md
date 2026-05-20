# 20260520 Round4/Round5 TRC Takeaways

## 当前主线

目标不是让 task vector 系数机械变大，而是在 Tool/Memory/Code 三个能力上同时提高正式评测。TRC 当前比 GRPO/OPD 更快，适合作为第一阶段能力抽取方法；正式 checkpoint 仍必须经过 Tool BFCL、Memory HotpotQA、Code CURE 验证。

## 已验证结论

1. Memory 不能只用 final answer hidden span。MemAgent 的核心能力在 update turns；`late3/uniform4 update turns + final` 明显比 final-only 更合理。
2. 完整 memory trajectory 训练开销太大，当前 7B merged forward 在 full trajectory 上 OOM；预算化 `late3` 或 `uniform4` 是可行折中。
3. 系数保护不能只做全局平均 floor。task-aware expert coefficient floor 能更直接保护对应任务 expert 的表达。
4. 直接放大某个任务的 loss scale 风险很高。R4B code loss multiplier=1.4 和 R4F memory loss multiplier=2.0 都导致 Tool collapse，说明 loss scale 会破坏 tool-call behavior span。
5. Code gate 更大不等于 Code acc 更高。R4A code gate 已高于 R3D，但 LiveBench 低于 R3D；Code bottleneck 更可能是 calibration data / span target 不对。
6. Code full-response span 不是充分条件。R3K Memory 最好，但 Code mean acc 仍低于 R3D。
7. Code directional projection 比 task loss scale 更安全。R4D 在 Tool/Memory 上最好：Tool mean `0.8048`，Memory F1 `0.7669`，Code 等待 CURE 完成。

## 当前最佳候选

- Round3 best Code known: R3D, mean Code acc `0.3289`, Tool `0.7944`, Memory `0.7636`。
- Round3 best Memory known: R3K, Memory `0.7715`, Code mean `0.3223`。
- Round4 best Tool/Memory candidate: R4D, Tool `0.8048`, Memory `0.7669`, but Code mean only `0.3076`。
- Round5 first v2-calibration candidate: R5A, Tool `0.8035`, Memory `0.7638`, Code mean `0.3194`, Code BoN `0.4310`。
- Round5 stronger projection candidate: R5B, Tool `0.7931`, Memory `0.7677`, Code mean `0.3198`, Code BoN `0.4095`。

## Round5 设计

Round5 不再继续调大 task loss multiplier，而改成数据分布实验：

- 用 `sota_calib_v2_20260518` 的 expert rollout 构造 96-row TRC bank。
- 每任务 32 条，且全部 unique prompt。
- Tool: v2 ToolRL 为主，paper96 fallback。
- Memory: v2 RL-MemoryAgent，late3 update turns + final。
- Code: v2 ReasonFlux/R1 + 20260516 ReasonFlux fallback。

R5A 固定 R4D objective，只换 calibration；R5B 只增强 code directional projection；R5C/R5D 把 code span 从 code-block 改为 response。R5A/R5B 已证明 v2 calibration 不会立即破坏 Tool/Memory。R5A 提高了 Code BoN 但没有提高 primary acc，说明当前 hidden-state TRC 更像提高候选分布上限，还缺 execution-aware repair/selection 信号。

## 下一步决策规则

1. R5 每个候选训练完成后先跑 Tool/Memory 快评。
2. Tool mean `<0.79` 或 Memory F1 `<0.76`，删除 baked checkpoint，只保留 logs。
3. 过线后再跑 CURE Code，避免浪费长评测。
4. 如果 R5 仍不能提升 Code，下一步不再调 span/loss，而是构造更严格的 Code recovery bank：当前 merged fail、ReasonFlux/R1 success、覆盖 LiveBench/LiveCodeBench 的 IO、edge case、算法推理、hidden unit-test 风格。
