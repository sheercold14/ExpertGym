# 2026-05-21 泛化 RL 能力归因框架

## 目标

当前真正要解决的问题不是“把某个 expert 系数调到多大”，而是：

> 给定多个 RL / SFT expert 的 task vector，如何判断每个残差位置表达的是能力、噪声、协同还是冲突，并据此组合出能够泛化的 merged model。

这个框架的核心要求：

- 能解释 mode conflict：为什么某些 task vector 加进去会伤害其他任务。
- 能解释 mode composition：哪些 residual 可以共同保留，哪些必须压制或分开处理。
- 能闭环验证：每个判断都要能通过 per-sample 结果、局部 residual 统计和 bake 后评测验证。
- 尽量不依赖训练和调参，先建立机制性归因。

## 当前经验事实

### 1. 全量注入不是最优能力表达

历史 TA scale sweep 中，`c=1.0` 是一个强 init1 baseline：

```text
theta = theta_base + 1.0 * (Delta_tool + Delta_memory + Delta_code)
```

它在 Memory 上很强，但 Tool live 泛化不如 RCRF：

| model | Tool quick mean | Tool live mean | Memory eval50 F1 | Memory eval100 F1 |
|---|---:|---:|---:|---:|
| TA init1 | 0.7644 | 0.6562 | 0.7688 | 0.7536 |
| RCRF | 0.7956 | 0.7188 | 0.7708 | 0.7567 |

这说明“把 expert delta 完整加上去”不等于能力最优表达。某些 residual 可能携带源任务能力，同时也带来泛化伤害。

### 2. ToolRL 源分布没有整体崩，只是少数 exact case 掉了

ToolRL all80 对照：

| model | success_rate | mean_reward | exact_tool_rate | parseable_rate | zero_call_rate |
|---|---:|---:|---:|---:|---:|
| TA init1 | 0.6375 | 0.8363 | 0.5500 | 0.9000 | 0.1000 |
| RCRF | 0.6250 | 0.8318 | 0.5250 | 0.9000 | 0.1000 |

逐样本比较结果：

| transition | count |
|---|---:|
| both exact | 42 |
| both non-exact | 36 |
| TA init1 exact only | 2 |
| RCRF exact only | 0 |

因此 ToolRL 差距不是格式能力退化，也不是调用意愿下降；`parseable_rate` 和 `zero_call_rate` 完全一样。问题集中在 2 条源分布 exact case，说明 RCRF 过滤掉了少数 ToolRL 精确参数/调用细节。

对照文件：

```text
/tmp/shared-storage/ExpertGym/rcrf/eval/toolrl_rlla4k_20260521/ta_init1_vs_rcrf_toolrl80_comparison.md
```

### 3. Code 不能只靠 code block teacher-forced utility 解释

RCRF Code quick：

| dataset | code acc | accumulate acc | BoN `(4,4)` acc |
|---|---:|---:|---:|
| LiveBench | 0.3789 | 0.4819 | 0.4297 |
| LiveCodeBench | 0.2862 | 0.4023 | 0.3464 |

TA init1 历史结果：

| dataset | code acc | accumulate acc | BoN `(4,4)` acc |
|---|---:|---:|---:|
| LiveBench | 0.3809 | 0.4829 | 0.4844 |
| LiveCodeBench | 0.3038 | 0.4379 | 0.3562 |

Code 的关键能力不只是“最终代码块像 expert”，还包括 prompt 理解、算法选择、约束跟踪、边界条件和 hidden-test 鲁棒性。因此当前只看 code-block 一阶 NLL 下降，不足以定位 Code 能力残差。

## 框架：从结果到 residual 的四层归因

### 层 1：任务结果归因

先不要看 gate 系数，而是把每个任务拆成官方评测可解释指标：

| task | 结果指标 | 作用 |
|---|---|---|
| Tool | BFCL parallel/live、ToolRL exact、parseable、zero-call | 区分泛化工具调用、源分布工具调用、格式稳定性、调用意愿 |
| Memory | HotpotQA EM/F1、长上下文子集 | 区分最终答案、轨迹记忆和长上下文稳定性 |
| Code | LiveBench/LiveCodeBench acc、accumulate acc、BoN | 区分单次正确率、测试覆盖、候选上限和稳定输出 |

原则：如果一个模型在某任务上下降，必须先知道下降的是哪种能力，而不是立刻归因到某个 expert coefficient。

### 层 2：行为 span 归因

每个任务真正表达能力的位置不同：

| task | span | 当前判断 |
|---|---|---|
| Tool | tool-call span | 与工具名、参数、调用格式最直接相关 |
| Memory | update turns + final answer | 不能只看 final answer，轨迹 span 很重要 |
| Code | prompt understanding + reasoning span + final code span | 只看 code block 不够，hidden-test 能力依赖题意理解和算法选择 |

原则：先确定能力在哪些 token span 上表达，再分析 residual。否则会把无关文本的 imitation 当成能力。

### 层 3：局部 residual 归因

对 expert `e`、模块 `m`、token `t`，功能单元定义为：

```text
u_{e,m,t} = Delta W_{e,m} h_{m,t}
```

当前使用 teacher-forced 一阶效应：

```text
signed_effect = mean_t - < dL/dz_{m,t}, Delta W_{e,m} h_{m,t} >
```

解释：

- 正值：该 residual 局部降低目标轨迹 loss。
- 负值：该 residual 局部伤害目标轨迹。
- 表达能量：`||Delta W h||^2`，只说明这个 residual 被激活，不说明方向好坏。

对 Code 需要升级为 outcome-aware 形式：

```text
contrast_score = utility(pass trajectory) - utility(fail trajectory)
```

否则模型可能学到“更像某个 expert 的代码风格”，但不提升 hidden-test pass rate。

### 层 4：因果 bake 验证

任何 residual 归因都必须通过最小干预验证：

| 干预 | 目的 |
|---|---|
| restore top helpful residuals | 验证这些 residual 是否真的恢复能力 |
| suppress harmful residuals | 验证这些 residual 是否造成冲突 |
| source-preserve subset | 验证源分布 exact case 是否需要特殊保护 |
| span ablation | 验证能力归因是否来自正确 span |

如果一个统计量不能通过 bake 后评测改变结果，它最多是相关性，不是机制。

## 模式冲突的定义

我们不把冲突定义成参数余弦冲突，而定义成输入条件下的功能冲突：

```text
cosine(Delta W_a h, Delta W_b h) < 0
```

更强的冲突定义是 outcome conflict：

```text
residual 对任务 A pass trajectory 有正 utility，
但对任务 B pass trajectory 或同任务 fail/pass contrast 有负 utility。
```

这能解释当前现象：

- TA init1 源分布能力保留更完整，但 BFCL live 更差。
- RCRF 提升 BFCL live，但 ToolRL 少数 exact case 下降。

这不是简单谁更强，而是不同 residual 对不同能力子分布的 utility 不一致。

## 模式组合的原则

### 1. 共享能力 residual

如果某 residual 对多个任务 pass span 都有正 utility，且没有明显冲突，应保留或轻放大。

### 2. 专属能力 residual

如果某 residual 只对 owner task 有正 utility，对其他任务中性，可以保留，但不应靠跨任务 agreement 放大。

### 3. 源分布细节 residual

如果某 residual 只对源分布 exact case 有用，但对泛化评测不稳定，需要建立 source-preserve 对照，而不是直接删掉。

ToolRL 当前的 2 条 `TA init1 exact only` 就是候选。

### 4. 冲突 residual

如果某 residual 对一个任务正、对另一个任务负，或者 induced residual 方向负相关，应压制或分任务条件化。

### 5. 噪声 residual

低表达、符号不稳定、只在少数样本上随机正的 residual，不应因为来自 expert 就保留。

## 当前可闭环实验设计

### 实验 A：Tool 源分布保护验证

问题：

> RCRF 在 ToolRL 80 上少掉的 2 条 exact 是否由被压低的 tool residual 导致？

步骤：

1. 从对照文件提取 `TA init1 exact only` 两条 prompt。
2. 对这两条和若干 BFCL live prompt 做 signed utility probe。
3. 找出 RCRF 中被压低、但在这两条 ToolRL prompt 上 owner utility 高的 tool residual。
4. bake 一个 `tool_source_preserve`：只恢复这些 residual 到 1.0。
5. 同时评测 ToolRL 80 和 BFCL quick。

判据：

- ToolRL exact 回升且 BFCL live 不掉：source-preserve 是合理补丁。
- ToolRL 回升但 BFCL live 掉：说明源分布细节与 live 泛化冲突，需要任务条件化或保守折中。
- ToolRL 不回升：说明差异不是这些 residual 造成的，当前 RCRF 解释不充分。

### 实验 B：Code span 归因

问题：

> Code 能力是否主要来自 prompt 理解，而不是 final code block imitation？

快速回归集：

```text
docs/report/RCRF/20260521_code_hurt_subset.md
```

该集合只保留 TA0.75/TA1.0 能通过、RCRF 未通过或 hidden-test 点明显掉分的 Code 样本，先用于低成本判断方法是否修复“受伤样本”，再进入完整 CURE。

当前 case pack 的第一轮结论：

- LiveBench hurt16：10/16 是 near-miss，说明很多错误集中在边界条件、约束细节或局部实现。
- LiveCodeBench hurt16：11/16 是 0 hidden-test pass，说明更多是 prompt 解析、输入格式、算法选择或题型识别失败。
- 因此 Code 不能只用一个 final-code residual 目标解释。LiveBench 可优先做 pass/fail code span contrast；LiveCodeBench 必须加入 prompt span / reasoning span 的 residual attribution。

已落地的输入层：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs
```

它把每条 hurt case 拆为同题 positive / negative：

- positive full：参考模型成功完整生成；
- negative full：当前模型失败完整生成；
- positive code：参考模型成功 extracted code；
- negative code：当前模型失败 extracted code；
- contrast code：同 prompt 的 pass code 与 fail code。

这使下一步 probe 可以明确回答：

```text
prompt_utility(pass) - prompt_utility(fail)
code_utility(pass) - code_utility(fail)
```

如果 prompt contrast 强而 code contrast 弱，说明 Code 受伤主要来自题意/算法方向；如果 code contrast 强，说明 residual 在具体实现 span 上可修复。

第一轮小规模真实验证：

```text
docs/report/RCRF/20260521_code_hurt_signed_utility_contrast.md
```

结果支持一个更强结论：

- 单独 `utility(pass_code)` 很弱，不能可靠解释 Code 正确性。
- 若干 residual 对 fail code 的 signed utility 反而很强，说明 teacher-forced NLL 能被失败轨迹“吸引”。
- 因此 Code 的第一性信号应是 `utility(pass) - utility(fail)`，而不是 expert positive imitation。
- late MLP down、late attention q/k 是当前 hurt cases 上区分度最大的模块族，值得作为下一轮机制验证对象。

完整 code-span probe 后的更新：

- LiveBench hurt16 的 final-code contrast 很弱，整体接近 0；这些 near-miss 更可能依赖 prompt/reasoning span 或局部边界条件。
- LiveCodeBench hurt16 的 contrast 很强，且 memory expert 的平均正 contrast 最大；这说明“Code 能力 residual”不必只来自 Code expert。
- 因此模式组合不能按 expert identity 简单分配，必须按输入条件下的 pass/fail utility 判断 residual 是能力、噪声还是冲突。

生成四个 training-free gate：

| 版本 | probe span |
|---|---|
| code_block_only | 只看代码块 |
| prompt_only | 只看 prompt token |
| prompt_plus_code | prompt + code block |
| full_response | 完整响应 |

判据：

- 如果 `prompt_plus_code` > `code_block_only`，说明 Code 评测能力需要题意理解 residual。
- 如果 `prompt_only` 也有效，说明 task vector 中有 prompt-side steering 能力。
- 如果都不如 TA init1，说明当前一阶 NLL utility 不是 Code pass rate 的充分代理。

### 实验 C：Code pass/fail contrastive residual

问题：

> 哪些 residual 真正区分能过 hidden tests 的代码和失败代码？

步骤：

1. 对同一批 code prompt 收集 pass trajectory 和 fail trajectory。
2. 对每个 module residual 计算：

```text
contrast_score = utility(pass) - utility(fail)
```

3. 只保留 contrast_score 高的 residual，或者恢复 TA init1 中对应 residual。
4. 评测 LiveBench / LiveCodeBench。

判据：

- 如果 Code acc 提升，说明 Code 需要 outcome-aware residual attribution。
- 如果 BoN 高但 acc 不高，说明能力上限存在但单次稳定性不足，需要 selection / robustness span。
- 如果 BoN 也不高，说明当前 experts 没有足够 Code 能力残差，应换专家或扩大 code calibration。

### 实验 D：跨任务冲突定位

问题：

> 哪些模块是 Tool / Memory / Code 冲突集中位置？

步骤：

1. 对各任务 pass span 计算每个 expert 的 `DeltaW h`。
2. 统计同层同模块 expert-pair cosine 和 signed utility。
3. 找出：
   - Tool 正、Memory 负；
   - Memory 正、Tool 负；
   - Code pass 正、Code fail 也正；
   - 三任务都正。
4. 对这些模块做 restore/suppress bake 验证。

判据：

- 如果 suppress 冲突模块提高多任务均衡，说明 mode conflict 被正确定位。
- 如果 restore 三任务都正模块提高整体，说明 mode composition 可以由 shared residual 决定。

## 当前阶段结论

1. RCRF 已经证明“响应条件 residual 选择”比纯能量或纯全量注入更能解释 Tool/Memory 的泛化差异。
2. ToolRL 差距已经缩小到 2 条 exact case，下一步应做 source-preserve 因果验证，而不是重训。
3. Code 需要从 code-block imitation 升级到 span-aware 和 pass/fail contrastive 归因；否则无法解释 hidden-test 正确性。
4. 泛化框架的核心不是固定某个 gate 公式，而是建立从官方结果到 residual 功能、再到 bake 因果验证的闭环。

## 已落地工具

- RCRF gate builder：`scripts/attention_pauh/build_response_conditioned_residual_filtering_gates.py`
- Signed utility probe：`scripts/attention_pauh/probe_signed_utility.py`
- Code pass/fail contrast comparer：`scripts/analysis/compare_signed_utility_contrast.py`
- Contrast-aware gate overlay：`scripts/attention_pauh/build_contrast_aware_residual_gates.py`
- Code hurt quick eval wrapper：`scripts/eval/run_cure_code_hurt_eval.sh`
- ToolRL rollout summary：`scripts/eval/summarize_rollouts.py`
- ToolRL per-sample comparison：`scripts/eval/compare_toolrl_rollouts.py`
- RCRF 机制报告：`docs/report/RCRF/20260521_tool_memory_mechanism_review.md`
- ToolRL 对照输出：
  - `/tmp/shared-storage/ExpertGym/rcrf/eval/toolrl_rlla4k_20260521/ta_init1_vs_rcrf_toolrl80_comparison.json`
  - `/tmp/shared-storage/ExpertGym/rcrf/eval/toolrl_rlla4k_20260521/ta_init1_vs_rcrf_toolrl80_comparison.md`

## 2026-05-21 更新：从诊断到可验证干预

为了避免 Code 研究停留在“看起来相关”的分析，已经把 Code hurt subset 变成三步闭环：

1. 结果层：筛出 `TA0.75/TA1.0 能做、RCRF 掉点` 的 32 条 Code hurt 样本。
2. 机制层：对同 prompt 的 pass / fail code trajectory 计算 residual signed-utility contrast。
3. 干预层：生成 `rcrf_code_contrast_v1` gate overlay，只在同 expert 内部重分配 residual 预算。

当前 candidate：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_contrast_v1/gates.json
```

它保持 expert 平均系数不变：

| expert | mean coefficient |
|---|---:|
| code | 0.900703 |
| memory | 0.987386 |
| tool | 1.004154 |

因此这个实验的科学问题很清楚：

> 不靠 global scale，不靠训练调参，只利用 pass/fail 行为差分，能否把 residual 预算从失败方向移到成功方向？

如果这个候选在 Code hurt subset 上提升，同时 Tool/Memory quick 不明显下降，就说明“outcome-aware residual routing”是有效机制；如果无效，则说明 final-code span 不足，下一步应优先补 prompt/reasoning span，而不是继续调 gate 系数。

快速验证结果已经给出更细的结论：

| subset | RCRF pass_any | contrast candidate pass_any | 读法 |
|---|---:|---:|---|
| LiveBench hurt16 | 0.0000 | 0.2500 | final-code contrast 有少量恢复，但不足以解释 near-miss |
| LiveCodeBench hurt16 | 0.0000 | 0.7500 | final-code pass/fail contrast 是有效能力信号 |

因此下一步不是继续调 `max_delta`，而是做 span 完整性验证：

```text
LiveBench: prompt/reasoning span contrast
LiveCodeBench: final-code contrast already useful; next test Tool/Memory side effect
```

这让方法主线更清楚：

> 能力归因必须先找到任务对应的行为 span；选错 span 时，即使 residual contrast 正确，也只能修复一部分样本。

## 2026-05-21 更新：span-aware conservative residual routing

补完 prompt / reasoning span 后，构造了第二个不训练候选：

```text
rcrf_code_spanaware_conservative_v2
```

核心不是调参，而是把机制判断写成一个最小规则：

```text
如果同一 residual 在多个行为来源上方向一致，则按 pass/fail contrast 轻微重分配；
如果来源之间符号冲突，则保守压低干预幅度；
始终保持每个 expert 的平均系数不变。
```

这版用到的行为来源：

| source | 用途 |
|---|---|
| LiveBench prompt span | 约束理解、输入格式、题意解析 |
| LiveBench reasoning span | 算法选择、边界条件推理 |
| LiveCodeBench final-code span | 具体实现正确性 |
| LiveCodeBench prompt span | 题型识别、函数/输入约束理解 |

关键诊断：

| pair | pearson | sign conflicts / 588 | 解释 |
|---|---:|---:|---|
| LiveBench prompt vs LiveCodeBench prompt | -0.9948 | 376 | 两个 Code 子分布的 prompt residual 方向几乎相反 |
| LiveBench prompt vs LiveCodeBench code | -0.0145 | 332 | prompt 理解与 final-code 实现不能混成单一方向 |
| LiveBench code vs LiveCodeBench code | -0.0382 | 300 | Code span 内部也存在子分布冲突 |

这给出当前最重要的 insight：

> Code 不是一个平滑的 task-vector scaling 问题；同一个 residual 可能对 LiveBench prompt 是能力、对 LiveCodeBench prompt 是伤害。因此能力组合必须是 span-aware 和 outcome-aware 的局部 residual routing。

最小干预结果：

| subset | RCRF pass_any | final-code v1 pass_any | span-aware v2 pass_any | span-aware v2 hidden test-point rate |
|---|---:|---:|---:|---:|
| LiveBench hurt16 | 0.0000 | 0.2500 | 0.4375 | 0.3398 |
| LiveCodeBench hurt16 | 0.0000 | 0.7500 | 0.8125 | 0.5189 |

副作用检查：

| metric | RCRF v1 | span-aware v2 | 判断 |
|---|---:|---:|---|
| BFCL Tool mean | 0.7956 | 0.7931 | 基本保持 |
| BFCL Tool live mean | 0.7188 | 0.7188 | 不掉 |
| HotpotQA eval50 F1 | 0.7708 | 0.7650 | 小幅下降 |
| HotpotQA eval100 F1 | 0.7567 | 0.7478 | 小幅下降 |

结论：

- v2 在两个 hurt 子集上都优于 final-code-only v1，说明 prompt/reasoning span 提供了真实增量。
- LiveCodeBench 的 hidden-test 点接近 TA0.75 上限，说明局部 residual routing 可以恢复一部分被 RCRF 伤到的 Code 能力。
- Tool 没有明显下降，说明 Code 修复不是简单牺牲 tool-call span 换来的。
- Memory 有可见小幅下降，说明 shared residual 的 Code 修复会触碰 Memory 长上下文稳定性；下一版必须加入 Memory-preserve span，而不是只优化 Code hurt subset。
- LiveBench 仍明显低于 TA0.75，说明它的 near-miss 更依赖细粒度边界条件/测试选择，下一步应做 case-level 局部解释，而不是继续调全局 gate。

这一步把论文主线从“训练 gate 系数”推进到更可辩护的问题：

```text
How can we attribute and route expert residuals by behavioral outcome and token span
when task-vector composition is non-smooth and source-conflicting?
```

## 2026-05-21 更新：expert-level hard preservation 是不够的

为了处理 v2 的 Memory side-effect，测试了一个非常直接的 v3：

```text
对 memory expert 设置负向保护：
Code contrast 可以上调 memory residual，但不能压低 memory residual。
```

这不是最终方法，而是验证“能力保护能否按 expert 做”的科学实验。

结果：

| subset | v2 pass_any | v3 pass_any | v2 test-point | v3 test-point | 结论 |
|---|---:|---:|---:|---:|---|
| LiveBench hurt16 | 0.4375 | 0.4375 | 0.3398 | 0.4121 | Memory floor 帮助 near-miss 稳定性 |
| LiveCodeBench hurt16 | 0.8125 | 0.6250 | 0.5189 | 0.3424 | Memory floor 破坏 LCB final-code 修复 |

这说明：

```text
Memory expert residual != Memory-only ability.
```

同一个 Memory expert 的 residual 可能：

- 对 HotpotQA / LiveBench 题意理解是有用能力；
- 对 LiveCodeBench final-code 或 prompt routing 是冲突干扰。

因此下一步的能力保护必须更细：

```text
protect residuals that are useful on Memory behavior spans,
not all residuals belonging to the Memory expert.
```

这进一步收敛了方法假设：模式组合不应以 expert coefficient 为基本单位，而应以“expert residual x behavior span x outcome utility”为基本单位。
