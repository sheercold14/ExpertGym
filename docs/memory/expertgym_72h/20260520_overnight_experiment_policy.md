# 20260520 Overnight Experiment Policy

## 当前目标

训练出 Tool / Memory / Code 都强的 TRC merged model。今晚不追单项最强，而追可进入论文主线的闭环：非泄露 calibration -> 快速训练 -> Tool/Memory quick gate -> Code 正式评测 -> 根据失败面更新 calibration / span / objective。

## 评测调度原则

- Code 评测最慢，只给 Tool mean >= 0.79 且 Memory mean F1 >= 0.76 的模型。
- 每个训练候选优先看 e8/e10/e12，不只相信 selected.gates.json。
- CURE/LiveBench/LiveCodeBench 的 hidden-test diagnostic 只能解释上限，不进入论文主训练数据。
- GPU 调度避免训练抢正在进行 CURE eval 的下一阶段生成卡；这类 OOM 会污染评测时间和日志。

## 当前核心经验

1. Gate 能推高不等于 Code 能涨。多轮实验里 code gate 到 1.2+ 后，Code mean 仍常在 0.31-0.33，说明瓶颈更像 calibration 能力覆盖，而不是 coefficient magnitude。
2. response topK128 在 R9A/R9B 都伤 Tool，R10A 因 tag-quota 暂时 Tool 过线，需要 Memory/Code 再验证；若 R10A Code 不好，不继续扩展 topK128。
3. response topK256 是目前最有希望的平衡点：它比 topK128 更不容易破坏 Tool，又比 code-block 更能覆盖思考/算法选择 span。
4. code-block384 对 Tool/Memory 较稳，但历史 Code 提升有限；R10C 用 tag-quota 复测它是否只是缺少算法覆盖。
5. Memory 的强度主要来自 late3 完整轨迹和 trajectory-turn loss；不要随意压 Memory span，否则 quick gate 可能过但正式 Memory F1 不稳。

## 今晚正在验证的问题

- R8A-e08：早停点 Memory 明显强，检查 Code 是否因不过推 gate 而更好。
- R8B：RF+DS coverage 是否比 RF-only 更能提升 LiveCodeBench。
- R8D：code-block384 在 CodeP0-v3 非泄露数据上是否仍弱。
- R10A：tag-quota 是否能拯救 topK128 的 Tool 退化。
- R10B：tag-quota + response256 是否成为当前主线。
- R10C：tag-quota + code-block384 是否能保住 Tool/Memory 并修复 Code。

## 当前强候选排序

1. `R8A-e08`: 非泄露 RF-only 早停点，Tool `0.7931`、Memory `0.7716`，Code running。优点是最干净且 Memory 稳，缺点是 Code 未确认。
2. `R10B`: tag-quota + response topK256，selected e12 gate 为 Code `1.2404` / Memory `0.9930` / Tool `1.2136`，Tool quick gate `0.7944`，Memory running。优点是最有方法增量，缺点是 Code 未确认。
3. `R8B`: RF+DeepSeek coverage，Tool `0.7944`、Memory `0.7687`、LiveBench `0.3477`，LiveCodeBench running。优点是 Code prompt 覆盖更完整，缺点是 LiveBench 只中等。
4. `R8D`: RF-only code-block384，Tool `0.7944`、Memory `0.7668`，Code running。用于判断 code-block span 是否能补 Code。
5. `R10A`: tag-quota + response topK128，Tool `0.7944`、Memory running。它证明 tag-quota 缓解了 R9 topK128 Tool collapse，但是否强还取决于 Memory/Code。

负例：`R7B` 虽然 Tool `0.8048`、Memory `0.7821`，但 Code mean 只有 `0.3010`，说明 Tool/Memory gate 强不代表 Code transfer；不能作为全面强模型候选。

## 10:45 阶段更新

- `R8B` 完整 Code 弱：LiveBench `0.3477`、LiveCodeBench `0.2642`、mean `0.3059`。DeepSeek fallback/coverage 没有转成 Code 泛化。
- `R8D` 完整 Code：LiveBench `0.3730`、LiveCodeBench `0.2676`、mean `0.3203`。code-block384 对 LiveBench 有帮助，但不解决 LiveCodeBench。
- `R8A-e08` 完整 Code：LiveBench `0.3594`、LiveCodeBench `0.2842`、mean `0.3218`。早停确实有价值，目前是 Round8 mean Acc 最好。
- `R10B` tag-quota response256 因 Memory `0.7572` 淘汰；`R10D` 把 Memory multiplier 提到 `1.8` 后修复到 `0.7679`，Tool 仍 `0.7944`，已进入 Code。
- `R10E` Tool multiplier `1.5` 不是好修复：Memory `0.7737`，但 Tool 掉到 `0.7788`，live_parallel 只有 `0.75`。
- `R11B` 即 R8D e08 早停，Tool `0.7944`、Memory `0.7715`，已进入 Code；它用于判断 R8D code-block 分支是否早停优于 e12。
- `R11G` 是当前最重要的新数据尝试：hybrid Code bank + response256 + Memory multiplier `1.8`，训练完成，gate C/M/T=`1.2405/0.9951/1.2276`，正在 quick gate。
- `R11H` 是 R11G 的 code-block384 对照，正在训练。

当前主判断：只推 code gate 已经不够，非泄露 Code 仍卡在 `0.32` 左右。下一步最值得押的是数据侧：RF-only/tag-quota/hybrid calibration 与 span/early-stop 的组合，而不是继续增大 loss 或 gate。

## 下一步实验池

- R10D：tag-quota response256 + memory multiplier 1.8，目标是稳住 Memory gate。
- R10E：tag-quota response256 + tool multiplier 1.5，目标是保护 BFCL live_parallel。
- R10F：tag-quota response256 + code projection floor 1.05，目标是增强 Code 方向但不过度改变 span。
- R10G：tag-quota response384，检查更大 response span 是否增加 Code 或破坏 Tool。
- R10H：若 R10B quick gate 合格，则延长到 e16，看 Code 是否继续受益或开始过拟合。

## 论文口径

主线 claim 不应是“sweep 出一个好系数”，而是：ExpertGym 用少量、有专家轨迹的 calibration signals，结构化地发现哪些 expert behavior span 应该被保留/增强。失败样本同样有价值：当 Code gate 上升但 Code 不涨，说明 calibration proxy 与正式能力不一致，需要能力覆盖而非盲目推高系数。
