# 2026-05-20 Overnight 35-Attempt Harness

目标：在不污染主评测、不泄露 Code 测试集的前提下，高吞吐寻找 Tool / Memory / Code 同时强的 TRC 模型。计数单位为 attempt：训练候选、checkpoint 门控、正式评测、数据构造诊断都计入，但只有通过 Tool/Memory 门控的模型进入 Code。

## 门控规则

- Quick Tool: BFCL 四项均值 >= 0.79；若 live_parallel 掉到 0.75 一律标记风险，不优先跑 Code。
- Quick Memory: 四个 HotpotQA split mean F1 >= 0.76；优先 >= 0.77。
- Code: 只测通过 quick gate 的模型；记录 LiveBench / LiveCodeBench acc 与 BoN。
- Checkpoint selection: 对每个训练候选优先评 e8/e10/e12 三个点，而不是只相信 loss-plateau selected。

## 当前高价值结论

1. Code response topK128 在 R9A/R9B 都伤 Tool，除非 tag-quota 明显逆转，否则不再扩展 topK128。
2. response topK256 是当前更平衡的 span；code-block384 对 Tool/Memory 稳定但 Code 不一定强。
3. CodeP0-v3 的算法 tag 覆盖比盲目 role quota 更接近 LiveBench/LiveCodeBench 能力面。
4. 目前 Code 上限受 calibration 能力覆盖约束，而不是 gate 推不动；下一步重点是 tag 覆盖、span selection、早停点。

## 今晚 attempt 池

### A. 正在运行/等待收口

- R8A-e08 Code：早停高 Memory 候选，等待 Code。
- R8B Code：RF+DS coverage 候选，等待 LiveCodeBench。
- R8D Code：code-block384 候选，等待 Code。
- R7B Code：旧强门控候选，等待 Code。
- R8A-e10 Memory：确认早停 e10 是否优于 e08。
- R10A：tag-quota topK128，完成后只做 Tool 快速否决。
- R10B：tag-quota topK256，完成后评 e8/e10/e12 quick gate。

### B. 下一批训练候选

- R10C：tag-quota + code-block384，验证 tag 覆盖是否能修复 R8D Code 弱点。
- R10D：tag-quota + response256 + memory loss 1.8，测试 Memory 保护是否能提高 quick gate 稳定性。
- R10E：tag-quota + response256 + tool loss 1.5，测试 Tool live_parallel 保护。
- R10F：tag-quota + response256 + code projection floor 1.05，测试 Code 方向更强但不过度抬 gate。
- R10G：tag-quota + response384，作为 response256/codeblock384 中间对照。
- R10H：tag-quota + response256 + e16 延长，只在 R10B e12 quick gate 合格后启动。

### C. 评测策略

- 每个训练候选优先 bake e8/e10/e12 三个 gate checkpoint 做 Tool/Memory quick gate。
- 若 Tool/Memory 都合格，进入 Code；否则只保留 run metrics，删除无效 baked checkpoint。
- 对 Code 通过者补写 docs/evaluation 和 best_ever_model 对照。

## 风险控制

- 不再启动会占用正在跑 Code eval GPU 的训练，避免 CURE 评测中途 OOM。
- 不对 CURE hidden 测试构造训练数据；R6B 只保留 diagnostic，不进入论文主线。
- 每个新增实验写 run.env 和 ledger；所有临时输出放 /tmp/shared-storage/OnPolicy，repo 只留脚本和文档。
