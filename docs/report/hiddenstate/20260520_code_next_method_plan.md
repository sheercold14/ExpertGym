# 2026-05-20 Code 下一步方法计划

## 当前判断

Code 的瓶颈不是单纯 gate 没有被推高。R8/R11/R12 多轮实验已经能把 Code gate 推到 `1.16-1.24`，但 CURE formal Code mean Acc 仍主要停在 `0.31-0.33`；历史较高点也只在 `0.34` 左右。R13 直接使用 formal Code16 expert-positive 轨迹做 eval-leak diagnostic，目的是验证“如果 calibration 分布对齐 formal eval，正样本 hidden-state TRC 是否足够”。若 R13 仍不上涨，说明正样本 residual alignment 不是充分信号。

## 已发现的问题

1. R13 自动 checkpoint selection 偏保守：R13A/R13B 都被 loss-plateau 选到 epoch 4，而 epoch 8 gate 继续上升。已强制 bake epoch 8 checkpoint 做 Tool/Memory quick gate，避免把 selection 问题误判成方法问题。
2. 当前 TRC 只看 expert positive trajectory：它告诉 gate “靠近专家轨迹”，但没有告诉 gate “远离当前模型失败轨迹”。Code 评测中同一题的失败往往来自 edge case、IO parsing、格式约束、长条件理解，这些在 hidden state 上可能和正轨迹很近，仅靠正样本余弦对齐不足以形成可泛化判别边界。
3. Code reward 与 formal eval 的关键差异在 execution robustness：训练样本多是能通过某组 tests 的正轨迹；formal eval 关心 hidden-like tests、稳定输出和 BoN 中至少一次正确。没有 failed trajectory / guard tests，loss 很难区分“会写一段合理代码”和“能过隐藏用例”。

## 短期诊断

- `R13A-e8-forced`：不带 R1，只用 RF/Mem positive formal Code16。若 Tool/Memory quick gate 通过，送 Code formal eval，对比 R13A auto-selected e4。
- `R13B-e8-forced`：带 scaled correct-R1 task vector，memory final-turn low-memory。若 Tool/Memory 不崩，送 Code formal eval，判断 R1 hidden direction 是否真实贡献 Code。
- 解析 CURE temp/result 输出，抽取同一 prompt 的 pass / fail response。若 eval temp 文件缺少完整失败输出，则从现有 calibration rollout 和 formal eval logs 中补建失败池。

## 下一轮方法：Execution-Aware Contrastive TRC

目标不是增加 calibration 数量，而是在同一 Code prompt 上构造更强的方向信号：

- positive：ReasonFlux / MemoryAgent / R1 中通过 reward tests 或 guard tests 的轨迹；
- negative：当前 merge / TA / 弱专家在同一 prompt 上失败的轨迹；
- span：`critical_reasoning_span + final_code_span`，必要时加入 IO parsing、edge-case reasoning、format-sensitive 片段；
- loss：在现有 positive TRC residual alignment 外，加 margin / contrastive 项，让 merged residual 更接近 positive residual，同时远离 failed residual；
- guard：每条 Code prompt 至少保留 train-only generated tests 或 hidden-like guard tests，避免只拟合 public examples。

这个方向符合论文主线：ExpertGym 不是 sweep 系数，而是用少量能力代表样本和专家轨迹，学习 task-vector 组合在能力空间中的可执行方向。

## 优先实验

1. R14A eval-leak contrastive Code16：用 formal Code16 的 positive/failed responses 做诊断。若这个都不能涨，说明当前 hidden-state loss 表达能力不足，需要改 objective。
2. R14B non-leak CodeP0 contrastive：用 CodeContests/train 或非 formal 数据构造 guard-test positive/negative，预算 32-48 条 Code trajectory，保证 Tool/Memory 各 24-32 条稳定保护样本。
3. R14C R1-controlled contrastive：只在 R14B 基础上加入 scaled R1 task vector，初始 R1 gate 小值，验证异质 reasoning expert 是否能补 Code。

## 2026-05-20 13:10 CST 执行状态

- 已实现默认关闭的 negative contrastive TRC 分支；不开 `--contrastive-negative-loss-weight` 时旧训练路径不进入该分支。
- 已构建 `trc_round14_code_contrast_v1`：Tool32/Memory32/Code10，其中 Code 10/10 均有来自 R11B CURE temp output 的同题 failed trajectory。
- 已启动 `trc_r14a_evalleak_contrast_rfmem_e8_20260520`：R13A 设置 + Code negative contrastive，`weight=0.5`，`margin=0.05`。这是 eval-leak diagnostic，不进入论文主结果，只用于判断 contrastive objective 是否能把 formal Code 能力推出来。

## 2026-05-20 13:45 CST 混合 train prompt 修正

R14A 的问题是 Code 只有 10 条 formal/eval-like prompt，能做机制诊断，但会把 calibration 分布推得过窄。已按“Code prompt 也要混合原来的 train 数据”的原则构建 R14B：

- 数据：`/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_mixed_train_contrast_v1/trc96_expert_trajectories.jsonl`
- 总量：Tool32 / Memory32 / Code32。
- Code32 组成：24 条原始 CodeP0 / CodeContests_train 成功轨迹 + 8 条 formal contrast hard anchors。
- Code expert 分布：`code=28, memory=4`；8 条 hard anchors 都有 `negative_response`。
- 训练：`trc_r14b_mixed_train24_contrast8_e8_20260520`，`CONTRASTIVE_NEGATIVE_LOSS_WEIGHT=1.5`。权重比 R14A 高，是因为只有 8/32 个 Code 行有 negative branch；不放大会被 24 条 train 正样本稀释。

R14B 是更合理的诊断：既保留原始 train 能力分布，又给少量 CURE-like failure contrast 信号。若它比 R14A 泛化更好，下一步应做非泄漏版本：用 CodeP0 train/guard 中的 pass/fail pair 替代 formal anchors。

## 2026-05-20 13:50 CST R5A / R11B 高 BoN 复查线

R5A 和 R11B 都表现出较高 BoN，说明模型候选分布中有正确代码，但 primary Acc 没同步上去。当前把它们作为 Code 稳定性 anchor：

- R5A 原始结果：mean Acc `0.3194`，mean BoN `0.4310`；Tool `0.8035`，Memory F1 `0.7638`。
- R11B 原始结果：mean Acc `0.3272`，mean BoN `0.4310`；Tool `0.7944`，Memory F1 `0.7715`。
- 差异：R5A 使用 v2 SOTA/recovery calibration，Code 17 RF v2 + 3 DeepSeek v2 + 12 RF fallback；R11B/R8D 使用 RF-only CodeP0-v3，epoch 8 早停。

已启动四个跟进：

1. `trc_r15a_r5a_repro_e12_20260520`：严格复现 R5A 训练设置，检查训练 gate/loss 是否可复现。
2. `trc_r15b_r5a_train24_contrast8_e12_20260520`：R5A 设置 + 24 条原始 R5A Code train + 8 条 formal contrast hard anchors，尝试把 BoN 转成 Acc。
3. `eval_r5a_code_repeat_20260520`：原 R5A checkpoint 重新跑 CURE Code，估计 stochastic eval 稳定性。
4. `eval_r11b_code_repeat_20260520`：原 R11B checkpoint 重新跑 CURE Code，与 R5A 对比稳定性。

判断逻辑：如果 repeat eval 波动很大，优先做 selection/decoding 稳定性；如果 repeat 稳定但 Acc 始终低于 BoN，则说明要加入 pass/fail contrastive 或 guard-test-aware loss，而不是继续推 gate。

## 判定标准

先跑 Tool/Memory quick gate：Tool mean `>=0.79`，Memory mean F1 `>=0.76`。通过后才跑 Code，节省 GPU。Code 成功信号不是 calibration loss 下降，而是 LiveBench/LiveCodeBench mean Acc 超过当前稳定上界 `0.34`，并且 BoN 不下降。
