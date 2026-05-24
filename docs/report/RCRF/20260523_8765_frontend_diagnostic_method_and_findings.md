# 8765 前端实验分析：诊断性方法与结论

## 0. 结论先行

`http://127.0.0.1:8765` 当前不是普通训练曲线前端，而是 **RCRF 机制诊断 Workbench**。它的价值不是告诉我们某个 gate 系数该调到多少，而是把模型合并问题拆成一个可审查的残差证据表：

```text
(expert, layer, module / param_name)
```

对每个 residual entry，它同时展示：

- 当前 expert delta 是否在某个任务 span 上真的表达出来；
- 这种表达是一阶降低 loss，还是一阶伤害；
- 不同任务、不同 code source、不同 span 之间是否符号冲突；
- gate 方案到底改了哪些 residual；
- 改动之后 Tool / Memory / Code 评测是否按预期移动。

核心结论是：

> ExpertGym 的 task vector 不是任务纯净方向。能力、噪声、行为保护和跨任务伤害混在同一个 expert delta 里；真正可解释的单位不是 expert-level scalar，而是 residual-level、span-conditioned 的 utility / harm / conflict。

因此下一步主线不应继续围绕全局系数 sweep 或 GRPO gate 调参，而应收束为：

```text
behavior span evidence
+ outcome contrast
+ residual-level conservative routing
```

## 1. 8765 前端读取的数据

进程：

```text
python -m streamlit run scripts/visualization/attn/analysis_platform/app.py --server.port 8765
```

默认真实数据规模：

| 数据表 | 数量 | 含义 |
|---|---:|---|
| residual records | 2016 | `DeltaW h` 在任务 span 上的 signed effect / expression |
| gate records | 2352 | owner-only / energy-only / no-conflict / rcrf 等 gate 的 residual alpha |
| interference records | 504 | expert pair 在任务 span 上的 residual cosine / conflict / cross-harm |
| eval records | 73 | ToolRL、BFCL quick、Memory、Code quick 结果 |

最新 code-hurt 分析目录：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521
```

最新 residual conflict atlas：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522
```

当前论文证据表：

```text
docs/report/RCRF/20260522_rcf_bc_paper_evidence_table.md
```

## 2. 诊断性方法

### 2.1 用 residual response，而不是参数范数，定义能力表达

对 expert `e` 的某个线性模块 delta：

```text
u_{e,p,t} = DeltaW_{e,p} h_{p,t}
```

前端核心读两个量：

| 量 | 含义 |
|---|---|
| `expression = mean ||DeltaW h||^2` | 该 expert residual 在当前 prompt/span 上是否真的被激活 |
| `signed_effect = mean[-<g, DeltaW h>]` | 一阶上是否降低 teacher-forced response loss；正值是 utility，负值是 harm |

这比只看 task vector norm 更合理，因为同一个参数 delta 在不同 prompt/span 上可能完全不表达，或者表达方向相反。

### 2.2 用 span 区分行为，不把整条回答当成一个信号

前端把任务 span 拆开看：

- Tool：tool-call span，重点是 parseable / exact / zero-call；
- Memory：update turns + final behavior，不能只看 final answer；
- Code：prompt / reasoning / final code block，不能只看代码块。

这解释了过去很多实验不稳定：一个 residual 可能对 Code final block 有益，但对 prompt constraint parsing 或 Tool format 有害。

### 2.3 用 pass/fail contrast 定义 Code capability

Code 不再只用 expert positive trajectory，而是看同 prompt 下：

```text
pass trajectory residual - fail trajectory residual
```

这样能避免把“专家常见风格”误当成“会通过测试的能力”。从 8765 的 code-hurt 页面看，span-aware contrast 是目前最干净的 code 修复信号。

### 2.4 用 Tool/Memory behavior 作为 constraint，而不是混成同一个 reward

Tool 和 Memory 的证据不适合只作为一个 scalar reward 混入 Code 优化。它们更像行为约束：

- 如果某个 residual 支持 tool-call 行为，不应因为 Code signal 负向就压掉；
- 如果某个 residual 会伤 Memory full trajectory，Code 正向更新需要 soft veto；
- 如果 evidence 弱，就不要做强决策。

这就是当前 RCF-BC / v18 的核心：continuous Code field + Tool/Memory behavior constraints。

## 3. 关键实验结论

### 3.1 Code 不是一个平滑统一方向

最新 `source_conflict_pairs.csv` 显示，Code 内部 source/span 冲突非常强：

| pair | Pearson | conflict rate |
|---|---:|---:|
| `LB_prompt` vs `LCB_prompt` | -0.9948 | 63.95% |
| `LB_prompt` vs `LCB_code` | -0.0145 | 56.46% |
| `LB_reasoning` vs `LCB_prompt` | -0.0543 | 56.46% |
| `LB_code` vs `LCB_prompt` | -0.0144 | 52.72% |
| `LB_code` vs `LCB_code` | -0.0382 | 51.02% |

这说明：LiveBench 和 LiveCodeBench 对同一 residual key 的需求经常相反。Code gate 拉高不一定提升 Code；Code expert 自身也不是纯 Code 能力包。

### 3.2 Memory 和 Code 的主要冲突发生在 residual key 级别

默认机制数据中，`code-memory` pair 是最强冲突源：

| pair/task | mean conflict |
|---|---:|
| code-memory on memory | 0.3397 |
| code-memory on code | 0.3158 |
| code-memory on tool | 0.2590 |
| code-tool on tool | 0.1554 |
| memory-tool on tool | 0.0032 |

Memory/Code 的冲突不是“memory expert 不该加”或“code expert 不该加”，而是同一个 residual key 在两个行为上有不同符号。

### 3.3 Memory 的能力通道更像 MLP residual，Code 的能力证据更稀疏

默认 residual 表按任务看：

| task/expert | signed mean | expression mean | positive fraction |
|---|---:|---:|---:|
| memory/memory | 2.012e-3 | 4.027 | 0.790 |
| tool/tool | 4.279e-5 | 0.208 | 0.955 |
| code/code | 1.473e-7 | 0.0145 | 0.835 |
| code/memory | -1.042e-7 | 3.081 | 0.411 |

解释：

- Memory expert 在 Memory span 上表达强、方向清楚；
- Tool expert 在 Tool span 上 signed direction 稳，但幅值小；
- Code expert 的 signed effect 很小，容易被其他 expert residual 淹没；
- Memory residual 在 Code span 上表达很强但方向混合，是冲突主体。

这就是为什么 Code 很难靠“提高 code coefficient”解决：Code 的能力残差信号既小又分布条件化。

### 3.4 Hard routing / expert-level protection 过粗

code-hurt 子集 v1-v4：

| variant | LB pass-any | LCB pass-any | 结论 |
|---|---:|---:|---|
| v1 code-only | 0.2500 | 0.7500 | final-code contrast 有信号，但不够 |
| v2 span-aware conservative | 0.4375 | 0.8125 | 当前最干净正结果 |
| v3 memory hard floor | 0.4375 | 0.6250 | 硬保护 Memory 会伤 LCB |
| v4 memory utility floor | 0.3125 | 0.6250 | 小 Memory signature 不足以替代 full behavior evidence |

v2 的结论很重要：加入 prompt/reasoning/code span 的 conservative routing 明显好于只看 final code block。

v3/v4 的负结果同样重要：不能把 Memory 保护做成 expert-level hard rule；Memory residual 里既有需要保护的行为方向，也有 Code 子分布上应该被抑制的方向。

### 3.5 RCF-BC 比 scalar shrink 更像方法

论文证据表中：

| candidate | rule | Tool mean | Memory F1 | LB hurt acc/BoN | LCB hurt acc/BoN |
|---|---|---:|---:|---:|---:|
| v18/v9 | continuous field + soft behavior constraints | 0.7931 | 0.7575 | 0.1406 / 0.2500 | 0.3281 / 0.6250 |
| v19 | strict archetype cleanup | 0.7956 | 0.7793 | 0.1250 / 0.1875 | 0.3281 / 0.4375 |
| v14 | code coefficient ×0.5 | 0.7944 | 0.7774 | 0.2031 / 0.2500 | 0.2500 / 0.1875 |
| v15 | code coefficient =0 | 0.7800 | 0.7841 | 0.0781 / 0.1250 | 0.1719 / 0.1875 |

Scalar code shrink 能提高 Memory，但会明显牺牲 Code。v18/v9 的意义不是绝对分数最高，而是它把 trade-off 暴露在 residual-level field 上，而不是把整个 code expert 当成一个可丢弃 scalar。

## 4. 从 8765 提炼出的论文级诊断协议

### Step 1：定义行为 span

每个任务必须有明确 span：

| task | behavior span |
|---|---|
| Tool | tool-call JSON / function-call span |
| Memory | update turns + final answer span |
| Code | prompt constraint + reasoning + final code block |

### Step 2：构建 outcome-aware residual utility

对同 prompt 的成功/失败轨迹，统计每个 residual entry 的 utility 差异：

```text
utility = signed_effect(pass) - signed_effect(fail)
```

没有 pass/fail 对照的样本，只能做弱证据，不能作为强 gate 决策。

### Step 3：构建跨任务 harm/support 表

对 Tool/Memory 行为 span，统计：

```text
support: residual 是否帮助保持行为
harm: residual 是否破坏行为
```

这一步不是优化 reward，而是给 residual routing 提供约束。

### Step 4：给每个 residual role 分类

当前 atlas 的最有价值角色：

| role | count | 处理原则 |
|---|---:|---|
| code_repair_only | 60 | 可提高 |
| shared_positive | 17 | 优先保留/提高 |
| code_repair_vs_protected_harm | 16 | Pareto 冲突，soft 处理 |
| code_negative_but_protected_support | 58 | 不能为了 Code 压掉 |
| code_source_conflict_with_behavior | 167 | 不能 scalar sweep，需要 span/source 诊断 |
| uninformative | 78 | 不动 |

### Step 5：只用简单保守规则合并

最小规则：

1. 多任务一致正向：保留或轻微增强；
2. 多任务一致负向：抑制；
3. 单任务正向、其他任务中性：保留；
4. 任务间冲突：soft shrink / 不动 / 回 base，不做强推；
5. 证据不足：不动。

这条规则比 sweep 更像论文方法，因为它依赖模型内部行为证据，而不是验证集上找系数。

## 5. 当前最可靠的结论

1. **task vector 不是 task-pure direction**：ExpertGym 中一个 expert residual 可以同时含能力、格式行为、噪声和跨任务伤害。
2. **Code 能力是 span/source-conditioned 的**：LiveBench 与 LiveCodeBench 的 prompt/code span 在 588 个 residual 上大量符号冲突。
3. **Memory-Code 冲突是主矛盾**：code-memory pair 的 conflict 明显高于 memory-tool，Memory residual 既是 Memory 能力通道，也是 Code 子分布冲突源。
4. **Tool 更像 behavior constraint**：ToolRL80 中 RCRF 和 init1 差距很小，parseable/zero-call 相同，说明 Tool 主要要保护 tool-call span；BFCL live 波动更像分布和格式泛化问题。
5. **hard routing 不够，continuous field 必要**：v19 更干净但 Code 下降，说明低置信连续 residual drift 仍可能承载 Code 能力。
6. **global scalar 不够**：code×0.5 或 code=0 可以换 Memory，但会毁 Code；这只能作为 negative control，不能作为方法。

## 6. 下一步应该怎么做

### 6.1 主线方法收束

推荐把论文方法收束为：

```text
Behavior-Constrained Residual Capability Field
```

中文：

```text
行为约束下的残差能力场
```

不要再强调“训练 gate 到最优系数”。真正有价值的是：

> 用少量行为轨迹诊断每个 residual entry 的 capability utility 与 behavior harm，然后用最小规则做结构化合并。

### 6.2 立刻补的证据

1. Tool：用 BFCL live/non-live 和 ToolRL80 共同标注 tool-call span support/harm。
2. Memory：用 full trajectory span，而不是小 final-answer signature。
3. Code：继续用 pass/fail contrast，但把 LB/LCB source/span 冲突显式放进 paper figure。
4. 评测：选 v18/v19/v14/v15 做核心对照，避免铺太多版本。

### 6.3 论文图表建议

| 图/表 | 内容 |
|---|---|
| Figure 1 | task-vector scalar view vs residual capability field view |
| Figure 2 | LB/LCB source-span conflict heatmap |
| Table 1 | residual role atlas counts |
| Table 2 | v18/v19/v14/v15 method/ablation result |
| Figure 3 | Memory-Code conflict by layer/module |

## 7. 一句话版本

8765 前端给出的最重要诊断不是“哪个实验最高”，而是：

> Agent task vector 的能力表达和伤害都发生在 residual entry 与 behavior span 的交互上；因此合并方法应该从 task-level coefficient search 转向 residual-level outcome contrast + behavior constraint。

## 8. 与当前 Eval6 的关系

8765 诊断目前支撑的是 **机制和方法设计**，不是单独支撑 SOTA 结论。当前 paper-main Eval6 队列中，`bcrc_v18_alias_v9` 已完成三项完整评测：

| candidate | Tool | Memory F1 | Code Acc | Avg(T/M/C) | 结论 |
|---|---:|---:|---:|---:|---|
| `bcrc_v18_alias_v9` | 0.7931 | 0.7570 | 0.3301 | 0.6267 | 可作为行为约束残差场的主方法样例，但不是当前最强分数 |

这说明当前论文叙事应谨慎：

1. 可以说：8765 诊断证明了 task vector 的有效单位是 residual-level、span-conditioned、outcome-dependent。
2. 可以说：BCRC 是这个诊断结论导出的简单方法，而不是通过 sweep 得到的黑盒 checkpoint。
3. 不应说：BCRC 当前已经超过所有 TA / TAME-style baseline。
4. 需要等 `no_behavior_v1_code_only` 和 `hard_behavior_v8` 的完整 Code 评测结束后，再决定 behavior constraint 的最终实验 claim。

因此，当前最稳的 ICLR 论文定位是：

> 我们提出一种可审查的 residual-level 诊断协议，并展示它能导出一个简单的 behavior-constrained merging rule；该方法揭示了为什么 expert-level scalar merging 在 AgentGym 任务上不稳定，并给出可泛化的 residual routing 方向。
