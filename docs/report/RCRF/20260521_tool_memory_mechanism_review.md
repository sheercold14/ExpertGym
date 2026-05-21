# 2026-05-21 RCRF Tool / Memory 机制复盘

## 一句话结论

这版方法最有价值的发现不是“某个全局系数更大更好”，而是：**不同专家的 task vector 只有在具体输入响应上真正产生有用残差时才应该被保留；只看参数大小或盲目放大专家残差不够可靠。**

目前 Tool 和 Memory 的快速评测支持这个判断：

- 只看 residual 能量的版本，Tool live 能力和 Memory 长上下文稳定性明显弱。
- 只看本任务有用残差的版本已经很强，说明“响应条件下的 residual 选择”是核心。
- 在此基础上加入跨任务冲突抑制，可以避免把一些看似有用、但会伤害其他任务的 residual 一起放大。

## 方法流程

### 研究问题

给定一个 base model 和多个 task expert，我们希望合并 task vector：

```text
Delta_e = theta_e - theta_base
```

传统做法通常给每个 expert 一个全局系数，或者给每层 / 每个参数一个可学习系数。我们这次要验证一个更细的假设：

> 一个 task vector 是否应该被保留，不应该只看它来自哪个 expert，也不应该只看参数范数，而要看它在当前任务响应 token 上实际诱导出的 hidden-state 改变量是否有用。

也就是说，真正的功能单元是某个线性模块上的局部残差：

```text
u_{e,m,t} = Delta W_{e,m} h_{m,t}
```

其中 `e` 是 expert，`m` 是某个 attention / MLP 线性模块，`t` 是被选中的响应 token，`h_{m,t}` 是该模块输入 hidden state。

### 输入

当前实现不做训练，也不使用 rollout reward 更新 gate。它只使用两类离线输入：

| 输入 | 含义 |
|---|---|
| OP-VEC mode manifest | 每个 expert 的 task vector 参数块，当前是 `tool / memory / code` 三个 expert，28 层，每层 7 个线性模块，共 588 个 expert-module 条目 |
| 少量带响应轨迹的 probe 样本 | 每个任务的 prompt + expert response，用于判断 residual 在真实响应 span 上是否有用 |

当前使用的路径：

```text
mode manifest:
/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json

signed utility summary:
/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521/signed_utility_summary.json
```

这里的 probe 不是评测，也不是训练。它只是用 teacher forcing 计算一阶局部效应。

### 第一步：选择任务相关响应 span

不同任务的关键行为 span 不一样，因此 probe 不对整段输出一视同仁：

| task | 选取的响应 span | 理由 |
|---|---|---|
| Tool | `<tool_call>...</tool_call>` | Tool 能力主要体现在函数名、参数、调用格式 |
| Code | markdown code block | 当前先看最终代码实现；从结果看，这不足以解释 hidden-test 正确性 |
| Memory | response span | Memory 任务需要完整轨迹和最终答案，当前先用响应整体近似 |

这一步的作用是减少无关 token 的干扰。例如 Tool 任务里普通解释文字不应该主导 tool-call 参数能力；Code 任务里代码块外的文本也不应该主导代码能力。

### 第二步：用 base model 做 teacher-forced probe

对每条 `(prompt, response)`，使用 base model 做 teacher forcing，计算监督损失：

```text
L = - log p_base(response | prompt)
```

然后在每个目标线性模块上保存：

- 模块输入 hidden state：`h_{m,t}`
- 模块输出梯度：`g_{m,t} = dL / dz_{m,t}`

对于 expert `e` 在模块 `m` 的 task vector `Delta W_{e,m}`，它会诱导一个局部输出变化：

```text
u_{e,m,t} = Delta W_{e,m} h_{m,t}
```

如果这个变化沿着降低 loss 的方向，说明该 residual 对当前轨迹有帮助。代码中用下面的一阶量衡量：

```text
signed_effect(e,m,task) = mean_t [ - < g_{m,t}, Delta W_{e,m} h_{m,t} > ]
```

解释：

- `signed_effect > 0`：加上这个 residual 会一阶降低当前响应的 teacher-forced loss，认为有用。
- `signed_effect < 0`：加上这个 residual 会一阶升高 loss，认为有害。
- `expression = mean_t || Delta W_{e,m} h_{m,t} ||^2`：该 residual 在当前响应上是否真的被激活。

这一步把“参数空间的 task vector”转成了“响应条件下的功能方向”。

### 第三步：聚合每个 expert-module 的统计量

对每个 expert `e`、模块 `m`、任务 `task`，聚合以下统计：

| 统计量 | 含义 |
|---|---|
| `owner_effect` | expert 对自己任务的平均一阶收益 |
| `positive_fraction` | 在自己任务样本上 `signed_effect > 0` 的比例，衡量稳定性 |
| `owner_expression` | residual 在自己任务响应上的表达能量 |
| `cross_positive` | residual 对其他任务也产生正收益的程度，表示潜在协同 |
| `cross_harm` | residual 对其他任务产生负收益的程度，表示潜在伤害 |
| `conflict_score` | 不同 expert 的 induced residual 在同一任务上的余弦冲突 |
| `noise_score` | 低表达能量或符号不稳定的噪声标记 |

其中 conflict 不是看参数余弦，而是看同一个输入上不同 expert 实际诱导的输出变化：

```text
cosine(Delta W_a h, Delta W_b h)
```

如果两个 expert 在同一任务、同一层的 induced residual 经常负相关，则说明它们可能在该位置竞争同一个行为通道。

### 第四步：把统计量转成 gate 系数

每个 expert-module 条目最终得到一个系数 `alpha_{e,m}`。主版本的规则是：

```text
alpha_{e,m} =
  clamp(
    1
    + utility_weight * family_boost * owner_signal
    + synergy_weight * family_boost * synergy_signal
    - harm_weight * conflict_boost * max(direct_harm, conflict_score)
    - noise_weight * noise_score,
    min_coeff,
    max_coeff
  )
```

当前主版本使用：

```text
utility_weight = 0.12
synergy_weight = 0.05
harm_weight = 0.18
noise_weight = 0.10
min_coeff = 0.55
max_coeff = 1.12
```

其中各个 signal 的实现是：

```text
owner_signal   = tanh(max(owner_effect, 0) / owner_effect_scale) * positive_fraction
synergy_signal = tanh(cross_positive / owner_effect_scale)
direct_harm    = tanh((cross_harm + max(0, -owner_effect)) / owner_effect_scale)
noise_score    = 1[expression_ratio < 0.35] or 1[positive_fraction < 0.45 and owner_effect <= 0]
```

`owner_effect_scale` 和 `expression_scale` 用同一个 expert 内正值统计的中位数估计，避免某个 expert 因 task vector 绝对幅值更大而天然占优。

各项含义如下：

| 项 | 作用 |
|---|---|
| `owner_signal` | 本 expert 对自己任务响应有稳定正贡献，则保留或轻微放大 |
| `synergy_signal` | 对其他任务也有正贡献，则视为共享能力，轻微放大 |
| `direct_harm` | 对自己任务为负，或对其他任务为负，则压低 |
| `conflict_score` | 与其他 expert 的 induced residual 方向冲突，则压低 |
| `noise_score` | 低表达或正负不稳定，则压低 |

模块类型也会影响保守程度：

| 模块类型 | 处理 |
|---|---|
| MLP | 更像能力写入通道，允许更充分表达 |
| attention v/o | 作为内容写回通道，温和处理 |
| attention q/k | 影响路由和注意力选择，更容易造成全局行为漂移，因此更保守 |

这不是手动给某个 expert 设大系数，而是让每个参数位置根据实际响应效应决定保留、放大或压低。

### 第五步：生成 OP-VEC gate 并 bake checkpoint

生成的 gate 是逐 expert-module 写入的：

```text
model.layers.i.xxx.weight::expert_name -> alpha
```

然后使用现有 OP-VEC baker 将它物化成 HuggingFace checkpoint：

```text
theta_merge = theta_base + sum_{e,m} alpha_{e,m} Delta W_{e,m}
```

当前生成了四个版本：

| 版本 | 目的 |
|---|---|
| 主版本 | 本任务有效 + 跨任务正贡献 + 冲突/伤害抑制 + 噪声抑制 |
| 只看能量 | 只看 `||Delta W h||` 大不大，不看方向，用来验证“参数动得大”是否足够 |
| 不抑制冲突 | 保留本任务有效和跨任务正贡献，但不压制冲突，用来验证冲突项是否必要 |
| 只看本任务 | 只看 expert 对自己任务是否有用，用来验证跨任务项是否必要 |

### 第六步：快速验证和消融

当前优先用 Tool / Memory 快速验证，因为它们评测成本低、能快速判断方法是否破坏核心能力：

| 任务 | 快速评测 |
|---|---|
| Tool | BFCL quick 四类：parallel / parallel_multiple / live_parallel / live_parallel_multiple |
| Memory | HotpotQA eval_50 / eval_100 F1 |

Code 完整评测成本高，所以当前方法先用 Tool / Memory 建立机制证据，再用 Code quick 检查是否能迁移到 hidden-test 程序正确性。

### 算法伪代码

```text
Input:
  base model theta_base
  expert task vectors {Delta_e}
  small trajectory set D = {(prompt, response, task)}
  target linear modules M

For each trajectory row (x, y, task):
  1. Select task-specific response span S(y)
  2. Run base model with teacher forcing on (x, y)
  3. For each module m in M:
       save module input h_m and output gradient g_m
  4. For each expert e and module m:
       compute induced residual u = Delta W_{e,m} h_m
       compute expression = mean ||u||^2 on S(y)
       compute signed_effect = mean -<g_m, u> on S(y)
       accumulate task-level statistics

For each expert e and module m:
  5. Estimate owner utility, cross-task positive effect, harm, conflict, noise
  6. Convert these statistics into coefficient alpha_{e,m}

Output:
  merged checkpoint theta_base + sum_{e,m} alpha_{e,m} Delta W_{e,m}
```

### 当前实现细节

- 代码入口：`scripts/attention_pauh/build_response_conditioned_residual_filtering_gates.py`
- 轨迹 probe 入口：`scripts/attention_pauh/probe_signed_utility.py`
- 局部一阶效应：`scripts/attention_pauh/core.py::linear_delta_probe`
- 运行脚本：`skill/command/run_20260521_rcrf_v1.sh`
- 当前 signed utility summary 只在一组采样层上 probe；生成 28 层 gate 时，未直接 probe 的层使用最近 probe 层的统计量近似。
- 当前方法是 training-free：不 rollout、不更新 reward、不做 optimizer step，只根据一阶局部效应生成 gate。
- 当前 Memory span 仍是 response-level 近似，后续可以尝试更接近 MemAgent 轨迹更新 span 的版本。

### 复现命令

```bash
# 生成四个 gate 版本
PROFILES=rcrf,energy_only,no_conflict,owner_only \
PHASE=generate \
bash skill/command/run_20260521_rcrf_v1.sh

# bake 成 HF checkpoint
CANDIDATES=rcrf,energy_only,no_conflict,owner_only \
PHASE=bake \
bash skill/command/run_20260521_rcrf_v1.sh

# 单独快速评测，避免 BFCL harness 并发写全局配置
PHASE=quick_eval CANDIDATES=rcrf GPU_LIST=1 TOOL_PORT=8111 \
bash skill/command/run_20260521_rcrf_v1.sh
```

## 当前对照实验

| 版本 | 核心含义 | Tool parallel | Tool parallel_multiple | Tool live_parallel | Tool live_parallel_multiple | Tool 平均 | Memory eval50 F1 | Memory eval100 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 主版本 | 保留本任务有用残差，同时考虑跨任务一致性和冲突 | 0.880 | 0.865 | 0.8125 | 0.625 | 0.7956 | 0.7708 | 0.7567 |
| 只看能量 | 只看 residual 在 hidden state 上是否大，不判断方向好坏 | 0.885 | 0.860 | 0.7500 | 0.625 | 0.7800 | 0.7720 | 0.7296 |
| 不抑制冲突 | 看到本任务有用和跨任务一致就放大，但不压制冲突项 | 0.880 | 0.855 | 0.8125 | 0.625 | 0.7931 | 0.7507 | 0.7372 |
| 只看本任务 | 只保留本任务响应上有用的 residual，不显式处理跨任务关系 | 0.880 | 0.855 | 0.8125 | 0.625 | 0.7931 | 0.7672 | 0.7574 |
| TA init1 | 直接把三个 expert task vector 都以 `c=1.0` 加到 base 上 | 0.880 | 0.865 | 0.6875 | 0.625 | 0.7644 | 0.7688 | 0.7536 |

TA init1 来自历史 scale sweep 报告：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/ta-scale-sweep-20260502-232830/ta_scale_sweep_report_zh.md
```

注意：该 sweep 报告里的 Memory 总 F1 `0.7667` 是四个 HotpotQA 子集的平均；当前 RCRF 快评只跑了 `eval_50 / eval_100`。所以上表只比较 shared 子集，避免口径混用。

## 与 TA init1 的直接比较

TA init1 是一个强基线，因为它代表“不筛 residual，直接完整注入三个专家能力”。和它比较可以回答一个关键问题：

> 我们的响应条件 residual 过滤，是否只是把 task vector 缩小了，还是确实保留了能力并减少了伤害？

shared 子集上的结果：

| 对比项 | TA init1 | 主版本 | 差值 |
|---|---:|---:|---:|
| Tool quick mean | 0.7644 | 0.7956 | +0.0312 |
| Tool live mean | 0.6562 | 0.7188 | +0.0626 |
| Memory eval50 F1 | 0.7688 | 0.7708 | +0.0020 |
| Memory eval100 F1 | 0.7536 | 0.7567 | +0.0031 |

这个对照给出的信息比和 `energy_only` 比更强：

- TA init1 已经把三个 expert 的 task vector 全部加进去，但 Tool live 明显低于主版本，说明“完整注入能力”并不等于“能力表达最好”。
- 主版本在 Memory shared 子集上没有牺牲 TA init1 的强 Memory 能力，说明过滤没有简单删掉 Memory 有效残差。
- 主版本相对 TA init1 的主要增益来自 Tool live_parallel：`0.6875 -> 0.8125`。这符合当前机制解释：Tool 对 attention routing / tool-call span 很敏感，过滤掉冲突或不稳定 residual 后，格式和函数调用行为更稳定。
- 这个结果也说明 RCRF 的论文定位不应该是“比 sweep 找到更大系数”，而是“在同等 task vector 来源下，用响应条件的一阶效应选择更干净的 residual 子空间”。

## 这些数字说明什么

### 1. 只看 residual 大小不够

“只看能量”的版本在 Tool live_parallel 上只有 0.7500，主版本是 0.8125；Memory eval100 也从 0.7296 提到 0.7567。

这说明 task vector 里存在大量“动得很大但不一定有用”的参数位置。它们可能只是专家训练带来的副作用，或者只在专家自己的生成分布里有意义。合并时如果只按大小保留，会把噪声也一起带进 merged model。

### 2. 本任务响应上的有效 residual 是核心信号

“只看本任务”的版本已经非常接近主版本，Memory eval100 甚至略高一点。这说明最重要的判断不是复杂的任务间规则，而是：

> 对同一个 prompt，在模型真正需要输出的 span 上，这个专家残差是否稳定地把 hidden state 推向正确行为。

这也是论文里最应该强调的第一性原则：task vector 的合并单位不应该只是 expert coefficient，而应该是“参数残差在具体任务响应上的功能表达”。

### 3. 冲突抑制仍然有必要

“不抑制冲突”的版本看起来 Tool 还可以，但 Memory 明显下降：eval50 从 0.7708 掉到 0.7507，eval100 从 0.7567 掉到 0.7372。

这说明跨任务一致性不能无脑放大。有些 residual 在某一类任务上看似有用，但它和另一个任务的有效方向相反。如果不抑制这类位置，会把 Memory 这种依赖长轨迹稳定性的能力拉坏。

### 4. 主版本目前更像“稳健折中”

主版本不是每一项都最高，但它在 Tool 和 Memory 上最均衡：

- Tool 平均最高。
- Tool live_parallel 保持最高。
- Memory eval100 保持高位。
- 相比只看本任务，Tool parallel_multiple 多 1 个点。

因此当前更合理的论文表述不是“复杂规则绝对最优”，而是：

> 响应条件下的 residual 选择是主因；冲突抑制提供稳健性，尤其避免某个任务的 residual 放大后破坏另一个任务。

## 当前还需要验证什么

### P0：Code 结果说明当前 span 不够

RCRF Code quick 已完成：

| dataset | code acc | accumulate acc | BoN `(4,4)` acc | BoN `(4,4)` accumulate acc |
|---|---:|---:|---:|---:|
| LiveBench | 0.3789 | 0.4819 | 0.4297 | 0.5544 |
| LiveCodeBench | 0.2862 | 0.4023 | 0.3464 | 0.4962 |

对比 TA init1 历史结果：

| dataset | TA init1 acc | RCRF acc | 判断 |
|---|---:|---:|---|
| LiveBench | 0.3809 | 0.3789 | 基本持平 |
| LiveCodeBench | 0.3038 | 0.2862 | RCRF 更低 |

这说明当前 response/code-block 一阶 utility 没有抓住 Code hidden-test 正确性的关键 residual。Code 的下一步不应该继续调同一个 gate 公式，而应该做 span-aware / pass-fail contrastive probe：

- `prompt_only`：看题意理解 residual 是否关键。
- `prompt_plus_code`：同时看题目理解和最终代码表达。
- `pass_minus_fail`：同一 prompt 下，用能过测试的代码减去失败代码的 utility。

只有当这些 attribution 能通过 bake 后提升 Code acc，Code 才能纳入主方法 claim。

### P0：验证“只看本任务”是不是足够简单的主方法

只看本任务的版本很强，甚至 Memory eval100 略高。下一步应该在完整 Tool / Memory / Code 上比较：

- 主版本：本任务有效 + 跨任务关系。
- 简化版本：只看本任务有效。

如果简化版本整体不差，它可能更符合奥卡姆剃刀原则，更容易写成论文主方法；跨任务冲突抑制可以作为增强项或分析项。

### P1：定位被压制的位置是否真的有害

目前我们知道“压制冲突项会让 Memory 更稳”，但还需要抽样检查被压制的层和模块：

- 它们是否集中在 attention q/k 这类路由位置。
- 它们是否在 Tool/Code 上有局部收益，但对 Memory 有负贡献。
- 它们是否对应三类任务响应差异最大的 token span。

这一步可以把方法从经验规则推进到机制解释。

### P1：Tool 应该继续看两种评测

BFCL live 子集波动大，单次 live 分数不能完全代表 Tool 能力。现在 ToolRL 80 题结果显示主版本和只看能量版本都保持 0.625 success，不是因为源分布能力崩了。

后续 Tool 应同时看：

- BFCL parallel / live：反映泛化和真实工具格式。
- ToolRL 80：反映源任务工具调用能力是否保留。

当前 ToolRL 80 对照：

| method | success_rate | mean_reward | tool_exact_rate | parseable_rate | zero_call_rate |
|---|---:|---:|---:|---:|---:|
| TA init1 | 0.6375 | 0.8363 | 0.5500 | 0.9000 | 0.1000 |
| 主版本 | 0.6250 | 0.8318 | 0.5250 | 0.9000 | 0.1000 |
| 只看能量 | 0.6250 | 0.8315 | 0.5250 | 0.9000 | 0.1000 |

这个结果需要谨慎解释：

- RCRF 的 BFCL live 提升并不是因为 ToolRL 源分布能力更强；在 ToolRL 80 上 TA init1 反而略高。
- RCRF 更像是在保持源分布 tool-call 能力基本不掉的同时，提高 BFCL live / parallel 场景的格式和参数稳定性。
- 因此 Tool 结论不能只写“Tool 能力更强”，而应该写成“源分布能力基本保留，live/generalization 子集更稳”。

进一步逐样本比较显示差距很集中：

| transition | count |
|---|---:|
| both exact | 42 |
| both non-exact | 36 |
| TA init1 exact only | 2 |
| RCRF exact only | 0 |

这意味着 ToolRL 差距不是大面积退化，而是 2 条 exact case 的源分布细节丢失。后续应把这两条作为 source-preserve 因果验证子集：如果恢复被 RCRF 压低的少量 tool residual 能补回这 2 条，且 BFCL live 不掉，则说明需要加入源分布保护项；如果 BFCL live 掉，则说明源分布 exact 和 live 泛化之间存在真实冲突。

### P2：确认是否需要更强的稀疏化

当前主版本只是温和压制和放大。后面可以验证：

- 更强压制低能量噪声是否提升稳定性。
- 对 q/k routing 做更明确的保护或抑制是否更有效。
- 对 MLP 的放大是否主要贡献 Memory。

这些是后续 ablation，不应该抢当前主线优先级。

## 当前论文可写的核心发现

可以先把论文 insight 收敛成三句话：

1. task vector 合并不是专家级别的系数选择问题，而是 residual 在具体任务响应上的功能选择问题。
2. 参数残差的范数或表达能量不能区分能力与噪声，必须看它是否在正确响应 span 上产生有用方向。
3. 多专家合并的主要风险不是“能力不够大”，而是某些局部 residual 对一个任务有用、对另一个任务有害；因此需要在响应条件下过滤冲突位置。

## 当前风险

- Tool / Memory 快评已经正向，但 Code 结果没有超过 TA init1，不能把当前 RCRF 写成三任务统一最优。
- “只看本任务”版本很强，主版本必须证明额外冲突项带来的收益值得保留。
- Code 主线必须转为“机制性 residual 过滤提升 Tool/Memory；Code 需要 outcome-aware / span-aware attribution”，除非后续 contrastive Code 实验补上提升。

## Code Span-Aware v2 的 Tool / Memory 副作用

`rcrf_code_spanaware_conservative_v2` 用 Code hurt pass/fail contrast 加上 prompt/reasoning span 做保守 residual 重分配。它在 Code hurt subset 上比 final-code-only v1 更强：

| Code hurt subset | RCRF pass_any | v1 pass_any | v2 pass_any |
|---|---:|---:|---:|
| LiveBench hurt16 | 0.0000 | 0.2500 | 0.4375 |
| LiveCodeBench hurt16 | 0.0000 | 0.7500 | 0.8125 |

为了判断它是不是合理的模式组合，而不是局部 Code 修补，补跑 Tool / Memory quick eval：

| metric | RCRF v1 | Code span-aware v2 | delta |
|---|---:|---:|---:|
| Tool parallel | 0.8800 | 0.8800 | 0.0000 |
| Tool parallel_multiple | 0.8650 | 0.8550 | -0.0100 |
| Tool live_parallel | 0.8125 | 0.8125 | 0.0000 |
| Tool live_parallel_multiple | 0.6250 | 0.6250 | 0.0000 |
| Tool mean | 0.7956 | 0.7931 | -0.0025 |
| Memory eval50 F1 | 0.7708 | 0.7650 | -0.0058 |
| Memory eval100 F1 | 0.7567 | 0.7478 | -0.0089 |

判断：

- Tool 基本保持，说明 Code pass/fail span-aware routing 没有明显破坏工具调用格式和并行调用行为。
- Memory 有小幅下降，说明 shared residual 的 Code 修复会触碰 Memory 需要的长轨迹稳定 residual。
- 这和前面的结构结论一致：Memory 对 MLP 和自身 attention 都敏感，不能只靠 Code outcome contrast 决定 shared residual。
- 下一版如果要成为三任务主候选，应加入 Memory-preserve 约束：对 Memory 高 utility residual 设 floor，或在同一个 conservative aggregation 中加入 Memory eval/recovery span 的 positive utility。

随后测试了一个最直接的保护对照 `rcrf_code_spanaware_memory_preserve_v3`：所有 memory expert 负向 overlay 归零，memory expert 不参与 recenter。

| Code hurt subset | v2 pass_any | v3 pass_any | v2 test-point | v3 test-point |
|---|---:|---:|---:|---:|
| LiveBench hurt16 | 0.4375 | 0.4375 | 0.3398 | 0.4121 |
| LiveCodeBench hurt16 | 0.8125 | 0.6250 | 0.5189 | 0.3424 |

这个结果说明 expert-level hard floor 不够合理：它帮助 LiveBench near-miss，但破坏 LiveCodeBench final-code 修复。更合理的 Memory-preserve 不能保护所有 Memory expert residual，而应该保护在 Memory behavior span 上有高 utility 的 residual。

## 结果路径

- 主版本 checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v1_20260521/rcrf`
- 只看能量 checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v1_20260521/energy_only`
- 不抑制冲突 checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v1_20260521/no_conflict`
- 只看本任务 checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v1_20260521/owner_only`
- 快评结果目录：`/tmp/shared-storage/ExpertGym/rcrf/eval/rcrf_v1_20260521`
- ToolRL 80 对照目录：`/tmp/shared-storage/ExpertGym/rcrf/eval/toolrl_rlla4k_20260521`
- ToolRL init1 vs RCRF 逐样本对照：`/tmp/shared-storage/ExpertGym/rcrf/eval/toolrl_rlla4k_20260521/ta_init1_vs_rcrf_toolrl80_comparison.md`
- Code quick summary：`/tmp/shared-storage/OnPolicy/eval/cure_feedback/rcrf-v1-rcrf-code/rcrf-v1-rcrf/rcrf_v1_rcrf_code_quick_20260521/summary.json`
- Code span-aware v2 checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2`
- Code span-aware v2 quick Tool/Memory：`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/side_effect_eval/rcrf_code_spanaware_conservative_v2/quick_tool_memory`
- Code span-aware memory-preserve v3 checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_memory_preserve_v3`
