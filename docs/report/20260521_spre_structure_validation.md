# 2026-05-21 SPRE 结构验证：Attention / MLP 对 Task Vector 系数的影响

## 结论先行

本轮实验的核心结论是：**task vector 系数不能只由 attention mass 决定。Attention matrix 能告诉我们能力在 token 维度的表达位置；MLP / linear exposure 告诉我们 residual 真正注入在哪里；signed utility 才接近“这个 residual 是否降低目标轨迹 loss”。**

当前证据支持一个简单结构假设：

- Memory 能力主要依赖强 prompt anchoring + 大幅值 MLP residual，因此从 `init=1` 出发保留 Memory MLP 是合理的。
- Tool 的 ToolRL source 能力没有明显崩，问题更多出在 BFCL live / multi-call 分布与格式泛化；对 Tool 做泛化修复应该看 BFCL-like tool-call behavior span，而不是只看 ToolRL train/test。
- Code 的 raw residual 幅值很小，当前 positive trajectory 对 gate 的 signed utility 弱；Code 不适合继续用“推高 code gate”作为唯一优化目标，需要 pass/fail contrast 或评测能力对齐的 trajectory。
- 粗暴缩小 attention 不会自然修复 Tool；`static-code-attn-shrink` 反而降低 BFCL live_parallel，说明 attention 不是可随意压缩的无害通道。

## 产物索引

### 诊断脚本

- `scripts/attention_pauh/probe_attention_matrix_patterns.py`
- `scripts/attention_pauh/probe_linear_module_exposure_patterns.py`
- `scripts/attention_pauh/probe_signed_utility.py`
- `scripts/attention_pauh/build_signature_preserving_gates.py`

### 诊断输出

- Attention matrix：`/tmp/shared-storage/ExpertGym/attention_matrix/pattern_probe_20260521_s4_len1536/`
- Linear exposure, delta-norm：`/tmp/shared-storage/ExpertGym/attention_matrix/linear_exposure_s4_len2048_20260521/`
- Linear exposure, raw：`/tmp/shared-storage/ExpertGym/attention_matrix/linear_exposure_raw_s4_len2048_20260521/`
- Signed utility pilot：`/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_s1_layers0_10_20_20260521/`

### SPRE Gate / Checkpoint

| variant | gate file | checkpoint | 关键设置 |
| --- | --- | --- | --- |
| init1 all | `/tmp/shared-storage/ExpertGym/spre/spre_exposure_shrink_init1_20260521/spre_gates.json` | `/tmp/shared-storage/OnPolicy/checkpoints/spre_init1_all_20260521` | 588 个 gate 全为 1 |
| SPRE-v2 | `/tmp/shared-storage/ExpertGym/spre/spre_exposure_shrink_init1_20260521_v2/spre_gates.json` | `/tmp/shared-storage/OnPolicy/checkpoints/spre_exposure_shrink_init1_v2_20260521` | Memory/Tool 全 1；Code q/k/v 轻微 shrink；MLP 全 1 |
| static code-attn shrink | `/tmp/shared-storage/ExpertGym/spre/spre_static_code_attn_shrink_init1_20260521/spre_gates.json` | `/tmp/shared-storage/OnPolicy/checkpoints/spre_static_code_attn_shrink_init1_20260521` | Code q/k/v=0.75, o=0.9；其他全 1 |
| mlp-preserve attn-calm | `/tmp/shared-storage/ExpertGym/spre/spre_mlp_preserve_attn_calm_20260521/spre_gates.json` | `/tmp/shared-storage/OnPolicy/checkpoints/spre_mlp_preserve_attn_calm_20260521` | 所有 expert 的 attention=0.75，MLP=1 |

## 1. 诊断结果如何约束 gate 设计

### Attention matrix 只提供“能力表达位置”

| task | layer group | prompt mass | local response | sink mass | 解释 |
| --- | --- | ---: | ---: | ---: | --- |
| memory | late | 0.8138 | 0.1862 | 0.4568 | 强 prompt retrieval + global routing |
| tool | late | 0.7019 | 0.2980 | 0.3671 | schema reading + tool-call behavior |
| code | late | 0.5153 | 0.4050 | 0.3692 | response-local construction |

这解释了三个任务为什么不能用同一类 calibration span：

- Memory 要覆盖 evidence / update / final answer 的长上下文轨迹。
- Tool 要覆盖工具 schema、参数选择和 tool-call 输出格式。
- Code 要覆盖 reasoning span、final code span、hidden-like guard test 相关约束。

### MLP 是 residual 注入主通道

Raw exposure 里 Memory MLP 远大于 Tool/Code：

| expert | span | all | mlp_up | mlp_down | attn_v |
| --- | --- | ---: | ---: | ---: | ---: |
| tool | response | 0.00545 | 0.01418 | 0.00459 | 0.00028 |
| memory | response | 0.11119 | 0.31424 | 0.09120 | 0.00525 |
| code | response | 0.00319 | 0.01566 | 0.00288 | 0.00023 |

因此，默认剪 MLP 是危险的。SPRE 的保守原则是：**从 `init=1` 出发，除非 signed harm 明确，否则 MLP 不剪。**

### Signed utility 暂时只支持 Memory 清晰正向

Pilot 结果：

| expert | best layer | owner utility | protected harm | utility-harm |
| --- | ---: | ---: | ---: | ---: |
| memory | 10 | 0.005398 | 0.0000006 | 0.005397 |
| tool | 20 | 0.000100 | 0.000000006 | 0.000100 |
| code | 0 | 0.000000023 | 0.000000061 | -0.000000038 |

解释：

- Memory 的 residual 方向强且干净，能解释为什么 memory gate 容易被推出来。
- Tool 有正向方向，但幅值很弱，格式能力容易被其他 residual 破坏。
- Code 当前 trajectory 对 gate 的一阶信号基本不可用，继续盲目调 code gate 不会自然变成正式 code acc。

## 2. SPRE 候选的系数结构

| variant | tool mean | memory mean | code mean | code attention | MLP |
| --- | ---: | ---: | ---: | --- | --- |
| init1 all | 1.0000 | 1.0000 | 1.0000 | 全 1 | 全 1 |
| SPRE-v2 | 1.0000 | 1.0000 | 0.9458 | q=0.8813, k=0.8615, v=0.8780, o=1.0000 | 全 1 |
| static code-attn shrink | 1.0000 | 1.0000 | 0.8786 | q/k/v=0.75, o=0.90 | 全 1 |
| mlp-preserve attn-calm | 0.8571 | 0.8571 | 0.8571 | 所有 attention=0.75 | 全 1 |
| memory-attn-calm | 1.0000 | 0.8571 | 1.0000 | Memory attention=0.75，其余 attention=1 | 全 1 |

这里的关键不是“某个 shrink 比例最优”，而是验证结构假设：

- 只轻微 shrink Code attention 不破坏 Memory/Tool source 能力；
- 统一压 attention 不会自动提升 Tool；
- 只压 Memory attention 能隔离验证 Memory 是否依赖 attention routing；
- 对 Code attention 压得更重会伤害 BFCL live_parallel；
- MLP 保留是当前最稳的默认策略。

## 3. Tool 评测：ToolRL test 与 BFCL live 要分开解释

### ToolRL rlla_4k test all80

路径：

- prompt：`/tmp/shared-storage/OnPolicy/data/evaluation/toolrl_rlla4k_test_20260518/toolrl_rlla4k_test_all80.prompts.jsonl`
- summary：`/tmp/shared-storage/OnPolicy/eval/toolrl_rlla4k_20260518/`

| model | success | mean reward | exact | parseable | zero call |
| --- | ---: | ---: | ---: | ---: | ---: |
| TA 0.75 | 0.6125 | 0.8310 | 0.5250 | 0.9000 | 0.1000 |
| best-ever reference | 0.6250 | 0.8381 | 0.5375 | 0.9000 | 0.1000 |
| init1 all | 0.6375 | 0.8363 | 0.5500 | 0.9000 | 0.1000 |
| SPRE-v2 | 0.6250 | 0.8333 | 0.5375 | 0.9000 | 0.1000 |
| mlp-preserve attn-calm | 0.6250 | 0.8381 | 0.5375 | 0.9000 | 0.1000 |
| static code-attn shrink | 0.6375 | 0.8354 | 0.5500 | 0.9000 | 0.1000 |

判断：ToolRL source tool 能力没有明显崩。各 SPRE 候选基本在 `0.625-0.6375`，并不比历史 best-ever 弱。

### BFCL quick: parallel / live_parallel

| variant | live_parallel | live_parallel_multiple | parallel | parallel_multiple | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPRE-v2 | 0.7500 | 0.6250 | 0.8850 | 0.8700 | 0.7825 |
| mlp-preserve attn-calm | 0.7500 | 0.6250 | 0.8850 | 0.8650 | 0.7813 |
| memory-attn-calm | 0.7500 | 0.6250 | 0.8850 | 0.8550 | 0.7788 |
| static code-attn shrink | 0.6875 | 0.6250 | 0.8800 | 0.8600 | 0.7631 |

BFCL 的 `parallel/parallel_multiple` 各 200 题，更稳定；`live_parallel/live_parallel_multiple` 分别只有 16/24 题，适合作压力测试，但不能单独作为 Tool 总能力结论。

结构判断：

- SPRE-v2 和 mlp-calm 的 BFCL quick 基本相同，说明“整体压低 attention”不是 Tool live 的核心解法。
- memory-attn-calm 的 BFCL quick 也基本保持，说明只压 Memory attention 的实验没有明显污染 Tool。
- static code-attn shrink 降低 live_parallel，说明 attention 改动会影响 BFCL 格式/参数匹配，不应把 attention 当作无害剪枝位。
- BFCL 错误主要是 `cannot_find_match` 和 `wrong_count`，更像工具选择/参数匹配/调用数量问题，而不是完全不会输出 tool 格式。

## 4. Memory quick 结果

| variant | eval_50 F1 | eval_100 F1 | mean F1 | 解释 |
| --- | ---: | ---: | ---: | --- |
| SPRE-v2 | 0.7805 | 0.7718 | 0.7761 | Memory 强，符合 MLP/full prior 假设 |
| mlp-preserve attn-calm | 0.7596 | 0.7329 | 0.7462 | 统一压 attention 会伤 Memory 长上下文/检索 |
| memory-attn-calm | 0.7406 | 0.7552 | 0.7479 | 只压 Memory attention 也会伤 Memory，Tool 基本不变 |

这个结果很关键：Memory 的 attention sink / prompt anchoring 不是噪声。即便保留 MLP，只把 attention 压到 0.75，也会降低 Memory F1。更干净的 `memory-attn-calm` 说明，Memory F1 下降不是因为 Tool/Code attention 被一并压低造成的副作用，而是 Memory 自身 attention routing 的贡献。Memory 的 gate 设计应保留 prompt/sink-heavy 层，不应做全局 attention calm。

## 5. 对论文方法的直接启发

### 5.1 不要把 attention mass 当 gate

Attention mass 高只说明模型在看某些 token，不说明对应 task vector residual 是正 utility 还是 harm。Memory 的 prompt/sink attention 就是例子：它对 owner task 有用，但容易被 naive exposure-ratio 判成跨任务共享 harm。

更合理的 pipeline：

1. attention matrix 定位 task signature span；
2. MLP / linear exposure 定位 residual 注入通道；
3. signed utility 判断 residual 是否降低 owner loss；
4. protected harm 判断是否伤害其他任务；
5. gate 只对 high-harm / low-utility 子空间收缩。

### 5.2 MLP 应作为能力保留默认通道

三类 expert 的 residual 主通道都在 MLP，尤其 Memory。除非 signed harm 明确，默认不剪 MLP。这个结论比“588 系数全学”更可解释：不是让优化器自己找到，而是基于 residual 表达机制保留能力。

### 5.3 Tool 需要 BFCL-like calibration，不是继续看 ToolRL source

ToolRL all80 已经说明 source tool behavior 没明显差。BFCL live/multiple 弱，说明缺的是：

- live 工具 schema / 参数语义；
- multi-call count control；
- BFCL pythonish call 格式；
- tool selection / argument matching，而不是单纯 tool-call parseability。

下一步 Tool calibration 应从 BFCL train/test-like distribution 里构造少量 behavior span，不应只重复 ToolRL。

### 5.4 Code 要先证明 trajectory 能 inform gate

当前 code signed utility 近 0，不能继续只看 code gate 是否上涨。Code 下一步应做：

- 同 prompt pass/fail contrast；
- code block + reasoning span；
- hidden-like generated tests；
- 正式评测能力桶采样；
- 用 signed utility 验证 trajectory 是否提供正方向，再训练或构造静态 gate。

## 6. 当前可用的最简单方法版本

可以把当前 SPRE-v2 作为“结构保守 baseline”：

- 初始化：`init=1`；
- MLP：全部保留；
- Memory / Tool：全部保留；
- Code：只根据 raw exposure 对 q/k/v 做轻微 shrink；
- 不做 reward sweep，不做 RL 训练，不依赖 0.75 先验。

它的价值不是立刻 SOTA，而是作为论文中一个 clean diagnostic baseline：**只靠结构先验就能保住 Memory 和 Tool source 能力，同时揭示 Code/Tool-BFCL 的真正瓶颈在 calibration 与 behavior span，而不是系数搜索。**

## 7. 后续最小实验

优先级从高到低：

1. 扩大 signed utility：每任务 8 条，层 `{0,4,8,12,16,20,24,27}`，分别看 attention / MLP / all-linear。
2. Tool span 化：只在 tool-call span 上计算 signed utility 和 retention/null-space，而不是整段 response。
3. Code contrast：同 prompt pass/fail trajectory 的 signed utility 差值，验证是否能产生明显 positive direction。
4. Memory 长上下文：对完整或分段 memory trajectory 计算 utility，避免 1536/2048 截断低估远端 evidence。
5. 最后才考虑把 SPRE-v2 与小规模可学习 gate 结合；如果 signed utility 不成立，不应继续盲训。

## 8. 扩展 Signed Utility：signature span, 4 samples/task, 8 layers

本节补充一个更接近 task behavior 的 signed utility 诊断。

命令核心设置：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/attention_pauh/probe_signed_utility.py \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --trajectory-jsonl /tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521 \
  --base-model /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct \
  --tasks tool,memory,code \
  --experts tool,memory,code \
  --scope all-linear \
  --layers 0,4,8,12,16,20,24,27 \
  --samples-per-task 4 \
  --span signature \
  --response-tail-tokens 512 \
  --max-seq-length 2048 \
  --device cuda \
  --torch-dtype bfloat16 \
  --write-row-details
```

`signature` span 的定义：

- Tool：只看 `<tool_call>...</tool_call>` behavior span；
- Code：只看 markdown code block；
- Memory：看 response span。

前 4 条 Tool 均命中 `<tool_call>`；前 4 条 Code 均命中 code block，因此这次不是普通 response probe。

输出：

- `/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521/signed_utility_summary.json`
- `/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521/signed_utility_rows.jsonl`

### 8.1 Top layer 结果

| expert | best layer | owner utility | protected harm | utility-harm |
| --- | ---: | ---: | ---: | ---: |
| memory | 27 | 0.004061 | 0.000004 | 0.004057 |
| tool | 20 | 0.0000768 | 0.0000029 | 0.0000739 |
| code | 24 | 0.000000026 | 0.000000134 | -0.000000108 |

量级差异非常大：

- Memory 是清晰强正方向；
- Tool 是弱正方向，约比 Memory 小 `50x`；
- Code 仍然几乎没有 owner utility，且 utility-harm 为负。

这说明 Code 失败不是因为之前 span 选得太宽；即便只看 code block，当前 code task vector / 当前 positive trajectory 也没有提供稳定的一阶降 loss 方向。

### 8.2 Module family 结果

| expert | family | owner signed | protected harm | positive fraction |
| --- | --- | ---: | ---: | ---: |
| memory | mlp_down | 0.006648 | 0.000004 | 1.000 |
| memory | mlp_up | 0.003070 | 0.000003 | 0.844 |
| memory | mlp_gate | 0.002517 | 0.000002 | 0.969 |
| memory | attn_o | 0.001864 | 0.000001 | 0.969 |
| tool | mlp_down | 0.0000878 | 0.000001 | 1.000 |
| tool | mlp_up | 0.0000774 | 0.000010 | 1.000 |
| tool | mlp_gate | 0.0000630 | 0.000028 | 1.000 |
| tool | attn_o | 0.0000528 | 0.000002 | 1.000 |
| code | mlp_up | 0.000000317 | 0.0000087 | 0.906 |
| code | mlp_down | 0.000000281 | 0.0000771 | 0.875 |
| code | mlp_gate | 0.000000228 | 0.0000052 | 0.969 |

解释：

- Memory 的正向 utility 几乎完全由 MLP 主导，`mlp_down` 最强；这和 raw exposure 一致。
- Tool 的正向 utility 也主要来自 MLP，其次是 `attn_o`。Tool 的 q/k/v owner utility 很弱且 protected harm 更高，说明 Tool 的格式行为不应靠全 attention 放大。
- Code 的 MLP family positive fraction 不低，但 owner signed effect 绝对值太小，而 protected harm 反而更大。换句话说，Code delta 在 code block span 上“有方向感但幅值/对齐太弱”，不足以成为可靠 gate 学习信号。

### 8.3 Expert conflict

最明显的负 cosine 集中在 Code 与 Memory：

| task | layer/pair | cosine | negative frac |
| --- | --- | ---: | ---: |
| code | layer_24:code\|memory | -0.0978 | 1.000 |
| code | layer_27:code\|memory | -0.0832 | 0.929 |
| memory | layer_27:code\|memory | -0.0730 | 0.964 |
| tool | layer_27:code\|memory | -0.0715 | 0.750 |
| memory | layer_24:code\|memory | -0.0617 | 1.000 |

这给了一个很具体的结构结论：late layer 的 Code-Memory residual 更新方向确实存在冲突，尤其在 layer 24/27。它解释了为什么简单把所有 expert gate 同时推高时，Memory 可以涨，Code 却不稳定；Code 的 residual 不仅弱，而且在 late layers 与 Memory 的诱导更新方向相反。

### 8.4 对 gate 设计的更新

扩展诊断后，最合理的结构假设变成：

1. **Memory：保留 MLP 和 late attention。** Memory 的 owner utility 强，harm 低，不应被 attention calm 误伤。
2. **Tool：保护 MLP + `attn_o`，谨慎处理 q/k/v。** Tool 的 source behavior 没崩，但 BFCL live/multi-call 需要更贴近 tool-call span 的 calibration。
3. **Code：不要只靠 positive code block NLL/residual。** 当前 code owner utility 太弱，且 late Code-Memory conflict 明显；下一步必须做同 prompt pass/fail contrast，或寻找更能代表评测能力的 code trajectory。
4. **Late Code-Memory conflict 是核心结构冲突。** 如果后续要做可学习 gate，优先考虑 layer 24/27 的 expert-wise conflict control，而不是全局系数。

## 9. Code pass/fail contrast 诊断

为验证 Code 问题是不是“positive code block 轨迹不够强”，补充同 prompt 的 pass/fail trajectory contrast：

- 数据：`/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl`
- 选择：8 个 Code prompt，每个 prompt 有 `response` positive 和 `negative_response`。
- 正轨迹 probe：`/tmp/shared-storage/ExpertGym/attention_matrix/code_contrast_signed_utility_round16_s8_20260521/positive_probe/signed_utility_summary.json`
- 负轨迹 probe：`/tmp/shared-storage/ExpertGym/attention_matrix/code_contrast_signed_utility_round16_s8_20260521/negative_probe/signed_utility_summary.json`
- 设置：`all-linear`，layers `0,4,8,12,16,20,24,27`，span=`signature`，即 Code 只看 markdown code block。

### 9.1 Code expert pass - fail layer 差分

| layer | positive owner utility | negative owner utility | pass - fail | 判断 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 2.921e-08 | 1.082e-08 | 1.840e-08 | pass > fail |
| 4 | 3.250e-08 | 1.593e-08 | 1.657e-08 | pass > fail |
| 8 | 9.087e-08 | 5.478e-08 | 3.609e-08 | pass > fail |
| 12 | 1.517e-07 | 8.113e-08 | 7.057e-08 | pass > fail |
| 16 | 1.824e-07 | 1.113e-07 | 7.115e-08 | pass > fail |
| 20 | 8.621e-08 | 5.478e-08 | 3.143e-08 | pass > fail |
| 24 | 9.920e-09 | -3.941e-08 | 4.933e-08 | pass > fail |
| 27 | -2.055e-07 | 1.975e-07 | -4.030e-07 | fail >= pass |

中层 `8-20` 确实出现了 pass > fail 的弱正方向；但 layer 27 的反向强到足以抵消整体差分：

| 范围 | positive sum | negative sum | pass - fail |
| --- | ---: | ---: | ---: |
| all probed layers | 3.773e-07 | 4.868e-07 | -1.095e-07 |
| without layer 27 | 5.828e-07 | 2.893e-07 | 2.935e-07 |

这说明 Code contrast 不是完全没信号，而是 **信号集中在中层，late layer 27 明显误导**。如果直接全层推 Code gate，优化器会同时接收“中层应该增大”和“末层反向”的混合信号，最终表现为 gate 能动但 code eval 不稳定。

### 9.2 Family 差分

Code expert 的 family 平均 signed effect：

| family | positive | negative | pass - fail |
| --- | ---: | ---: | ---: |
| attn_k | 5.750e-11 | -3.955e-09 | 4.012e-09 |
| attn_o | 6.102e-08 | 4.150e-08 | 1.952e-08 |
| attn_q | -1.015e-08 | -2.709e-08 | 1.694e-08 |
| attn_v | 2.978e-08 | -3.110e-09 | 3.288e-08 |
| mlp_down | 6.987e-09 | 2.281e-07 | -2.212e-07 |
| mlp_gate | 1.031e-07 | 7.492e-08 | 2.817e-08 |
| mlp_up | 1.394e-07 | 1.155e-07 | 2.386e-08 |

可解释结论：

- `mlp_gate/mlp_up/attn_o/attn_v` 对 pass 轨迹相对更友好；
- `mlp_down` 在这批样本上反而更靠近 fail 轨迹；
- attention 不是整体坏，`attn_o/v` 有弱正 contrast，但 q/k 信号很弱；
- 因此 Code 不适合“所有 MLP 一起加、所有 attention 一起加”的粗粒度 gate。

### 9.3 Protected harm

| protected expert | positive harm sum | negative harm sum | positive - negative |
| --- | ---: | ---: | ---: |
| memory | 1.997e-06 | 1.174e-06 | 8.231e-07 |
| tool | 4.675e-07 | 3.430e-07 | 1.246e-07 |

Positive Code 轨迹不仅 owner utility 弱，还比 negative 轨迹带来更大的 Memory/Tool protected harm。这解释了为什么把 Code 信号当 OPD/NLL 正样本硬推时，容易牺牲 Memory/Tool，却不一定换来 Code eval 增益。

### 9.4 与 Tool / Memory 的量级对比

同一套 signature probe 下，owner utility sum 的量级为：

| expert | owner utility sum | best layer | best layer utility |
| --- | ---: | ---: | ---: |
| memory | 1.610e-02 | 27 | 4.061e-03 |
| tool | 3.423e-04 | 20 | 7.675e-05 |
| code | 1.178e-06 | 16 | 3.651e-07 |

Code 比 Tool 还小约 `290x`，比 Memory 小约 `1.4e4x`。这不是单纯调大学习率能干净解决的问题；调大学习率会同时放大 Code 的 weak positive、late-layer wrong direction 和 protected harm。

### 9.5 对方法设计的直接约束

Code 的结构结论应写成：

1. Code 不是完全不可学；pass/fail contrast 在中层存在弱正信号。
2. Code 的可学信号不是全层一致的；layer 27 是当前最危险的反向层。
3. Code 的 positive trajectory 对 protected tasks 的 harm 高于 negative trajectory，说明“专家正轨迹 NLL”不是天然安全的能力目标。
4. 后续如果继续做 Code gate，应采用结构约束：
   - 优先中层 `8-20`；
   - 谨慎或冻结 layer 27 Code gate；
   - 区分 `mlp_gate/mlp_up/attn_o/attn_v` 与 `mlp_down`；
   - 用 pass/fail margin，而不是只用 positive NLL。

这也给论文一个更清楚的 insight：**task-vector coefficient learning 的困难不只是 reward 稀疏，而是 residual utility 在 layer/family 上非单调，且 owner utility 与 protected harm 可同时上升。**
