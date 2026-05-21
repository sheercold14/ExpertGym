# 2026-05-21 Tool / Memory / Code 的 Attention + MLP 模式分析

## 目标

本轮不做调参，不直接优化评测分数，而是回答三个第一性问题：

1. 不同任务的 token 信息流是否有稳定差异？
2. OP-VEC task vector 的 residual 主要落在 attention 还是 MLP？
3. 哪些 residual 只是“幅值大”，哪些 residual 在一阶近似下真的能降低对应轨迹 loss？

结论先行：**attention matrix 能描述能力表达位置，但不能单独决定 gate；MLP 是实际 residual 幅值的主通道；Memory 的 residual 方向最清晰，Tool 次之，Code 当前最不清晰。**

## 产物位置

### Attention Matrix Probe

- 脚本：`scripts/attention_pauh/probe_attention_matrix_patterns.py`
- 输出：`/tmp/shared-storage/ExpertGym/attention_matrix/pattern_probe_20260521_s4_len1536/`
- 抽样：每任务 4 条，`max_seq_length=1536`

### Linear / MLP Exposure Probe

- 脚本：`scripts/attention_pauh/probe_linear_module_exposure_patterns.py`
- delta-norm 输出：`/tmp/shared-storage/ExpertGym/attention_matrix/linear_exposure_s4_len2048_20260521/`
- raw 输出：`/tmp/shared-storage/ExpertGym/attention_matrix/linear_exposure_raw_s4_len2048_20260521/`
- 抽样：每任务 4 条，`max_seq_length=2048`

### Signed Utility Pilot

- 脚本：`scripts/attention_pauh/probe_signed_utility.py`
- 输出：`/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_s1_layers0_10_20_20260521/`
- 抽样：每任务 1 条，层 `{0,10,20}`，`all-linear`
- 说明：这是 pilot，只用于判断量级和方向，不作为最终统计。

## 1. Attention Matrix：三类任务的信息流不同

| task | group | prompt mass | prompt tail | local response | long response | sink mass | entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | early | 0.4547 | 0.1954 | 0.4815 | 0.0638 | 0.3075 | 0.4490 |
| code | middle | 0.4670 | 0.1929 | 0.4537 | 0.0793 | 0.3307 | 0.4244 |
| code | late | 0.5153 | 0.2086 | 0.4050 | 0.0797 | 0.3692 | 0.4001 |
| memory | early | 0.7810 | 0.2828 | 0.2190 | 0.0000 | 0.3478 | 0.3952 |
| memory | middle | 0.7987 | 0.2012 | 0.2013 | 0.0000 | 0.4454 | 0.3210 |
| memory | late | 0.8138 | 0.1855 | 0.1862 | 0.0000 | 0.4568 | 0.3183 |
| tool | early | 0.6030 | 0.2372 | 0.3970 | 0.0001 | 0.2724 | 0.4645 |
| tool | middle | 0.6321 | 0.2435 | 0.3678 | 0.0001 | 0.2892 | 0.4138 |
| tool | late | 0.7019 | 0.2065 | 0.2980 | 0.0001 | 0.3671 | 0.3729 |

### Memory

Memory 的 response token 对 prompt 的注意力极高：`0.78-0.81`。这说明 Memory 本质上是 prompt-anchored retrieval / evidence reading：生成 final answer 时仍强依赖前文事实、检索记录、更新轨迹。

中后层 sink mass 高达 `0.445-0.457`，意味着 memory 的长上下文处理不只是“看 prompt tail”，还包含强 global routing / attention sink 行为。直接用 prompt exposure ratio 会把这种 shared global routing 错判为跨任务 harm。

### Code

Code 的 local response attention 最高：`0.405-0.482`，并且有稳定 long-response mass：`0.064-0.080`。这说明 code 能力更像生成过程中的局部结构维护、constraint propagation、code block 内一致性，而不是只读取 prompt。

因此，只用 prompt hidden state / prompt attention 去决定 code gate 不够。Code 的关键 span 应该包含：

- reasoning span；
- final code span；
- pass/fail 轨迹差异 span；
- 与 unit test / hidden-like guard test 相关的约束 span。

### Tool

Tool 介于 Memory 和 Code 之间。late prompt mass 从 `0.603` 增到 `0.702`，说明后层更读取 tool schema / user constraint；但 response-local mass 仍有 `0.298-0.397`，说明 tool-call 格式和参数生成行为也重要。

Tool 的 head prompt std 最高，暗示它可能依赖少数 specialized heads，而不是全层平均能力。全局 NLL retention 太弱时，tool 格式很容易被其他任务 residual 干扰。

## 2. Linear / MLP Exposure：MLP 是 residual 主通道

### delta-norm exposure：单位 task vector 方向的敏感度

| expert | span | all | mlp_up | mlp_down | attn_v |
| --- | --- | ---: | ---: | ---: | ---: |
| tool | prompt | 0.6813 | 1.0268 | 0.4241 | 0.6769 |
| tool | response | 0.6160 | 0.9230 | 0.2657 | 0.6346 |
| memory | prompt | 0.7514 | 1.3481 | 0.5030 | 0.6086 |
| memory | response | 0.6934 | 1.1903 | 0.3069 | 0.6426 |
| code | prompt | 2.4030 | 6.2841 | 5.1586 | 1.6781 |
| code | response | 1.6352 | 6.1060 | 0.7980 | 1.4495 |

Code 的单位 residual 对 MLP 极其敏感，尤其 `mlp_up` 和 prompt-side `mlp_down`。这解释了为什么 code delta 稀疏但某些组合能影响 code：它不是均匀分布在全模型，而是高度集中在 MLP 子空间。

### raw exposure：真实注入幅值

| expert | span | all | mlp_up | mlp_down | attn_v |
| --- | --- | ---: | ---: | ---: | ---: |
| tool | prompt | 0.00642 | 0.01598 | 0.00761 | 0.00029 |
| tool | response | 0.00545 | 0.01418 | 0.00459 | 0.00028 |
| memory | prompt | 0.13011 | 0.35163 | 0.15022 | 0.00498 |
| memory | response | 0.11119 | 0.31424 | 0.09120 | 0.00525 |
| code | prompt | 0.00547 | 0.01609 | 0.01764 | 0.00026 |
| code | response | 0.00319 | 0.01566 | 0.00288 | 0.00023 |

raw exposure 里 Memory 比 Tool/Code 大约高一个数量级到二十倍。这个结果很关键：

- Memory task vector 真实注入幅值很大，容易被训练信号推出来，也容易主导合并。
- Tool 和 Code raw residual 都很小，所以 gate 增大不一定带来足够强的行为变化。
- Code 的 delta-norm exposure 高，但 raw exposure 小，说明“方向可能敏感，但实际 task vector 幅值弱”。这和我们观察到 code gate 推高但正式 code acc 不稳定是吻合的。

## 3. Signed Utility Pilot：幅值大不等于有用

设置：每任务 1 条，层 `{0,10,20}`，all-linear，response span。

| expert | best layer | owner utility | protected harm | utility - harm |
| --- | ---: | ---: | ---: | ---: |
| memory | 10 | 0.005398 | 0.0000006 | 0.005397 |
| tool | 20 | 0.000100 | 0.000000006 | 0.000100 |
| code | 0 | 0.000000023 | 0.000000061 | -0.000000038 |

### Memory

Memory 的 signed utility 明显为正，且 protected harm 很低。代表层里 `mlp_up/mlp_down/mlp_gate` 都有强正向贡献。说明 memory task vector 的 residual 方向和 memory expert trajectory 是高度对齐的。

这解释了为什么 Memory 容易训练出提升，也解释了为什么如果训练算法不加约束，memory 可能压制其他任务。

### Tool

Tool 的 signed utility 为正，但量级比 Memory 小约 50 倍。layer 20 的贡献最大，top modules 包括：

- `layer20.mlp_down`
- `layer20.mlp_up`
- `layer20.mlp_gate`
- `layer20.self_attn.o_proj`

Tool 更像“弱幅值但格式敏感”的行为能力。保护 tool 不能只靠全局 retention，需要针对 tool-call behavior span / late-layer behavior residual 做保护。

### Code

Code 的 signed utility 在这组 pilot 里接近 0，甚至 owner utility - protected harm 为负。结合 raw exposure 小，可以得到一个更谨慎的判断：

当前 code expert delta 对当前 calibration 的 teacher-forced code trajectory **没有形成稳定一阶下降方向**。这不等于 code task vector 完全没用，因为 TA sweep 的 code 结果确实能涨；但它说明当前 code loss / trajectory / span 选择不足以稳定 inform gate。

Code 的问题可能不是“gate 没推够”，而是：

- code calibration trajectory 没有对齐正式评测能力；
- positive trajectory 的 teacher-forced residual 不等于 pass@1 能力方向；
- code delta raw 幅值太弱，只有特定 MLP 子空间有效；
- 需要 pass/fail contrast，而不是单独靠 positive NLL / residual alignment。

## 4. 三任务模式总结

### Tool 模式

- Attention：prompt-schema + response behavior 混合；late layer 更 prompt anchored。
- MLP：真实 residual 幅值小，主要在 `mlp_gate/mlp_up/mlp_down`。
- Utility：正向但弱，late layer 更有效。
- 风险：其他任务 residual 很容易破坏 tool-call 格式；普通 NLL retention 量级不够。
- 简单原则：Tool 不适合大幅稀疏选择，应该保持 late attention + MLP behavior span 的完整性。

### Memory 模式

- Attention：强 prompt anchored，强 sink/global routing。
- MLP：raw residual 最大，`mlp_gate/mlp_up/mlp_down` 都强。
- Utility：一阶方向最清晰，harm 低。
- 风险：Memory 容易成为 dominant expert，过强时会牺牲 Tool/Code。
- 简单原则：Memory 不应该被 PAUH ratio 压低 late layer；应保留 prompt/sink-heavy 层，但用 harm 约束防止过注入。

### Code 模式

- Attention：response-local construction，prompt 不是唯一能力来源。
- MLP：delta-norm exposure 高，但 raw exposure 小；主要集中在 MLP。
- Utility：当前 positive trajectory 对 code gate 的一阶信号很弱。
- 风险：看起来 gate 变了，但正式 code acc 不动；训练 reward 和评测能力错位。
- 简单原则：Code 需要 contrastive pass/fail direction 和 code-span / reasoning-span 选择，而不是只做 positive residual matching。

## 5. 对下一步“最简单方法”的约束

目前最干净的方法不应该是复杂训练器，而应该是一个可解释的二阶段规则：

1. **保留能力预算**：从 `init=1` 出发，不用 0.75 作为先验，不让模型天然少注入专家能力。
2. **attention 只做 span/layer selector**：用 attention matrix 找 task signature，不直接把 attention mass 当 gate。
3. **residual utility 决定缩放方向**：只在 signature span 上算 signed utility；owner utility 正、protected harm 低的层保留；harm 高的层收缩。
4. **MLP 不做默认剪枝**：MLP 是三类任务 residual 主通道，尤其 Memory/Code。除非 signed harm 明确，否则 MLP gate/up/down 应该保持接近 1。
5. **Tool behavior span 强保护**：Tool raw residual 弱，格式敏感；应保护 tool-call span，而不是靠全局 NLL。
6. **Code 需要单独验证可学习性**：若 signed utility 仍接近 0，说明当前 code trajectory 不能 inform gate，应先换 code calibration / contrast，再谈 gate 优化。

这个路径符合奥卡姆剃刀：先解释 observed failure，再用最少机制修复，不引入大规模 RL 调参。

## 6. 需要继续补的诊断

- Signed utility 扩大到每任务 4-8 条、更多代表层，确认 pilot 是否稳定。
- 对 Tool 单独切 tool-call span，而不是整段 response span。
- 对 Code 单独切 reasoning span / code block span / failing span，并加入 pass-fail contrast。
- 对 Memory 使用更长上下文或分段 prompt windows，避免 `max_seq_length` 截断低估远端 evidence。
- 计算 expert-pair induced residual cosine，确认冲突主要发生在 Code-Memory 还是 Tool-Memory。
