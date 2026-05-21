# 2026-05-21 Gate 结构与性能矩阵

## 目的

这份记录把 attention / MLP 诊断与真实候选模型性能接起来，回答一个具体问题：

> task vector 系数变化到底在结构上怎样影响 Tool / Memory / Code？

结论先行：

- **单纯推高 expert 全局系数不等于能力更强。** TRC R5A/R8D 把 Code gate 推到 `1.24`，但 Code mean acc 仍只有 `0.319-0.320`。
- **结构化层分配比全局 scale 更有效。** PAUH layer-all 的 expert 平均只有 `0.75`，但 Code mean acc 达到 `0.3506`，高于 R5/R8 系列多数强推 Code gate 的结果。
- **MLP 是 Memory 的必要能力通道。** PAUH attn-only 与 layer-all 的 attention 分配相近，但没有 MLP 注入，Memory F1 从 `0.736` 掉到 `0.640`。
- **压 Memory attention 会伤 Memory。** SPRE mlp-preserve attn-calm 保留 MLP，但 attention 全压到 `0.75` 后 Memory quick F1 只有 `0.746`；更干净的 memory-attn-calm 只压 Memory attention，也只有 `0.748`，低于 SPRE-v2 的 `0.776`。
- **Code-Memory late-layer conflict 是真实结构冲突。** signed utility 显示 layer 24/27 上 Code-Memory induced residual cosine 为负；这解释了为什么强推 Code gate 不稳定。

## 产物

Gate 结构汇总脚本：

- `scripts/attention_pauh/summarize_gate_structure.py`

输出：

- `/tmp/shared-storage/ExpertGym/attention_matrix/gate_structure_representatives_20260521/gate_structure_summary.json`
- `/tmp/shared-storage/ExpertGym/attention_matrix/gate_structure_representatives_20260521/gate_structure_summary.md`

脚本会把两类 gate 统一展开：

- full gate：`model.layers.N.xxx.weight::expert`
- TRC layer gate：`layerN.expert`，根据 mode manifest 展开到 attention / MLP family

## 1. Gate 结构矩阵

| variant | tool mean | memory mean | code mean | tool attn/mlp | memory attn/mlp | code attn/mlp |
| --- | ---: | ---: | ---: | --- | --- | --- |
| SPRE init1 all | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| SPRE-v2 | 1.0000 | 1.0000 | 0.9458 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9052 / 1.0000 |
| SPRE static code-attn shrink | 1.0000 | 1.0000 | 0.8786 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.7875 / 1.0000 |
| SPRE mlp-preserve attn-calm | 0.8571 | 0.8571 | 0.8571 | 0.7500 / 1.0000 | 0.7500 / 1.0000 | 0.7500 / 1.0000 |
| SPRE memory-attn-calm | 1.0000 | 0.8571 | 1.0000 | 1.0000 / 1.0000 | 0.7500 / 1.0000 | 1.0000 / 1.0000 |
| PAUH layer-all | 0.7500 | 0.7500 | 0.7500 | 0.7500 / 0.7500 | 0.7500 / 0.7500 | 0.7500 / 0.7500 |
| PAUH attn-only | 0.7500 | 0.7500 | 0.7500 | 0.7500 / NA | 0.7500 / NA | 0.7500 / NA |
| TRC R5A selected | 1.2039 | 0.9952 | 1.2393 | 1.2039 / 1.2039 | 0.9952 / 0.9952 | 1.2393 / 1.2393 |
| TRC R8D selected | 1.2096 | 0.9943 | 1.2403 | 1.2096 / 1.2096 | 0.9943 / 0.9943 | 1.2403 / 1.2403 |
| TRC R8D epoch8 | 1.1499 | 0.9974 | 1.1601 | 1.1499 / 1.1499 | 0.9974 / 0.9974 | 1.1601 / 1.1601 |

注意：TRC layer gate 对同一层的 attention 与 MLP 使用同一系数，因此 family 维度不能区分 MLP/attention，只能表示“该层整体注入更强”。

## 2. 性能矩阵

| variant | Tool metric | Memory metric | Code metric | 结构解释 |
| --- | ---: | ---: | ---: | --- |
| SPRE-v2 | BFCL quick mean `0.7825`; ToolRL all80 `0.6250` | HotpotQA quick F1 `0.7761` | 未测 | 保留 Memory/Tool 全 residual；只轻微压 Code attention；Memory 稳 |
| SPRE static code-attn shrink | BFCL quick mean `0.7631`; ToolRL all80 `0.6375` | 未测 | 未测 | Code attention 压得更重，BFCL live_parallel 从 `0.7500` 掉到 `0.6875` |
| SPRE mlp-preserve attn-calm | BFCL quick mean `0.7813`; ToolRL all80 `0.6250` | HotpotQA quick F1 `0.7462` | 未测 | MLP 保留但 attention 全压，Memory 明显低于 SPRE-v2 |
| SPRE memory-attn-calm | BFCL quick mean `0.7788` | HotpotQA quick F1 `0.7479` | 未测 | 只压 Memory attention，Tool 基本保持，但 Memory 仍明显低于 SPRE-v2 |
| PAUH layer-all | BFCL quick mean `0.7954` | HotpotQA quick F1 `0.7362` | CURE mean acc `0.3506`; BoN `0.4047` | 平均 alpha 只有 `0.75`，但层分配带来强 Code；Memory 不够强 |
| PAUH attn-only | BFCL quick mean `0.7930` | HotpotQA quick F1 `0.6400` | 未测 | 无 MLP 注入，Memory 大幅下降 |
| TRC R5A selected | BFCL quick mean `0.8035` | HotpotQA mean F1 `0.7638` | CURE mean acc `0.3194`; BoN `0.4310` | Code gate 高但 deterministic Code 不强；BoN 高说明多样性/上界提升 |
| TRC R8D selected | BFCL quick mean `0.7944` | HotpotQA mean F1 `0.7668` | CURE mean acc `0.3203`; BoN `0.4183` | code-block span 改善 LiveBench，但 LiveCodeBench 仍弱 |
| TRC R8D epoch8 / R11B | BFCL quick mean `0.7944` | HotpotQA mean F1 `0.7668` | CURE clean repeat mean acc `0.3203`; BoN `0.4027` | early stop 保住 Tool/Memory，但 Code 没有突破 |

## 3. 从矩阵得到的结构规律

### 3.1 Code gate 高不代表 Code 能力强

TRC R5A/R8D 的 Code mean gate 都在 `1.24` 左右，但 Code mean acc 只有 `0.319-0.320`。PAUH layer-all 的 Code mean gate 是 `0.75`，却达到 `0.3506`。

这说明 Code 的关键不是“把 code expert residual 整体放大”，而是：

- 哪些层被放大；
- 是否避开 late Code-Memory conflict；
- trajectory 是否真的提供 code pass@1 相关方向；
- 是否对齐 LiveBench / LiveCodeBench 的评测能力，而不是只对齐 code block NLL。

PAUH 的 Code top layers 是：

- top：`27, 9, 6, 5, 4`；
- bottom：`1, 16, 17, 13, 18`。

TRC R8D 的 Code layer 几乎整体同向推到 `1.24`，层间差异极小。这解释了为什么它更像“增加 residual 幅值”，而不是找到结构性能力子空间。

### 3.2 Memory 必须保留 MLP，同时不能粗暴压 attention

两组对照说明 Memory 的结构很清楚：

- PAUH layer-all：attention + MLP 一起注入，Memory F1 `0.7362`；
- PAUH attn-only：只注入 attention，Memory F1 `0.6400`；
- SPRE-v2：Memory attention/MLP 都保留 `1.0`，Memory F1 `0.7761`；
- SPRE mlp-preserve attn-calm：MLP 保留 `1.0`，attention 压到 `0.75`，Memory F1 `0.7462`。
- SPRE memory-attn-calm：只压 Memory attention，Tool/Code 保持 `1.0`，Memory F1 `0.7479`。

因此 Memory 不是单一 MLP 或单一 attention 能力：

- MLP 承载主要 residual 幅值；
- attention prompt/sink routing 对长上下文 retrieval 也很重要；
- Memory signed utility 中 MLP 最强，但 Memory 自身 attention 不能被压低；这不是 Tool/Code attention 的混杂副作用。

### 3.3 Tool source 能力稳定，BFCL live 是单独问题

ToolRL all80：

- TA0.75：`0.6125`
- best-ever reference：`0.6250`
- SPRE init1 all：`0.6375`
- SPRE-v2：`0.6250`
- mlp-calm：`0.6250`
- static code-attn shrink：`0.6375`

ToolRL 说明 source tool-call 能力基本没崩。BFCL quick 的差异更多来自 live/multi-call schema：

- SPRE-v2 live_parallel `0.7500`
- static code-attn shrink live_parallel `0.6875`

所以 Tool 的下一步不是继续看 ToolRL，而是构造 BFCL-like tool-call behavior span：multi-call count、argument matching、live schema reading、pythonish call format。

### 3.4 Layer/family 结构比全局平均更像论文 insight

现有证据支持一个可以写成论文的叙事：

1. Task vector 合并不是平滑全局 scale 问题；
2. 不同 expert 的 residual 在 attention/MLP 和 layer 上有不同表达结构；
3. MLP 是能力幅值主通道，但 attention 决定 prompt/schema/long-context routing；
4. signed utility 能区分 owner utility 与 protected harm；
5. gate 应该先由结构诊断定位，再由少量 trajectory 验证，而不是只做 coefficient sweep。

## 4. 下一步最小验证

为了把这条线变成更强的因果证据，已补完三个小实验中的前两个，第三个可作为后续更细粒度验证：

1. **PAUH layer shuffle / inverse PAUH**：已验证 inverse 明显伤 Tool，说明层方向不是无关变量。
2. **Code pass/fail contrast signed utility**：已验证中层有弱正 signal，但 layer 27 反向。
3. **Memory attention floor ablation**：已验证只压 Memory attention 会让 Memory F1 从 `0.7761` 掉到 `0.7479`。

这三个实验比继续调 OPD/GRPO 超参更接近目标：它们直接测试 task vector 系数的结构性影响。

## 5. PAUH 排序因果对照：alpha1 / inverse / shuffle

为验证 PAUH 的层排序是否只是随机重排，本轮补了两个机制对照：

- `inverse`：把 PAUH score 反向；
- `shuffle`：同样保持每个 expert 的平均 alpha 和系数范围，但随机打乱层排序。

产物：

| variant | gate | baked policy | eval dir |
| --- | --- | --- | --- |
| alpha1 layer-all | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_layer_all_20260521/pauh_gates.json` | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_layer_all_20260521/baked_policy` | `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_alpha1_layer_all_20260521/quick_tool_memory_rerun` |
| alpha1 inverse | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_inverse_20260521/pauh_gates.json` | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_inverse_20260521/baked_policy` | `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_alpha1_inverse_20260521/quick_tool_memory` |
| alpha1 shuffle | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_shuffle_20260521/pauh_gates.json` | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_shuffle_20260521/baked_policy` | `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_alpha1_shuffle_20260521/quick_tool_memory` |

### 5.1 Tool / Memory quick 结果

| variant | BFCL mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory F1 mean | eval50 | eval100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAUH alpha0.75 layer-all | 0.7954 | 0.7500 | 0.6667 | 0.9050 | 0.8600 | 0.7362 | 0.7352 | 0.7371 |
| PAUH alpha1 layer-all | 0.7438 | 0.6250 | 0.6250 | 0.8800 | 0.8450 | 0.7362 | 0.7359 | 0.7365 |
| PAUH alpha1 inverse | 0.6698 | 0.6875 | 0.5417 | 0.7300 | 0.7200 | 0.7577 | 0.7770 | 0.7384 |
| PAUH alpha1 shuffle | 0.7854 | 0.7500 | 0.6667 | 0.8850 | 0.8400 | 0.7563 | 0.7762 | 0.7364 |
| PAUH attn-only | 0.7927 | 0.6875 | 0.7083 | 0.9150 | 0.8600 | 0.6397 | 0.6655 | 0.6138 |

### 5.2 解释

这个对照给了更细的因果判断：

1. **Tool 对错误层排序非常敏感。** `alpha1 inverse` 的 BFCL mean 只有 `0.6698`，parallel / parallel_multiple 从正常 PAUH 的 `0.905/0.860` 掉到 `0.730/0.720`，并出现大量 `ast_decoder:decoder_failed`。这说明 Tool 的格式/调用行为依赖正确层位，不是任意 residual 放大都能保持。
2. **Memory 不完全依赖 PAUH 排序。** `inverse` 和 `shuffle` 的 Memory F1 反而高于 alpha0.75 layer-all。这与前面的 raw exposure/signed utility 一致：Memory 的 residual 幅值和 MLP 通道很强，平均 alpha=1 时即使层排序不理想，也能保住不少 Memory。
3. **alpha 过大伤 Tool。** `alpha1 layer-all` 比 `alpha0.75 layer-all` 的 BFCL mean 低 `0.0516`，主要掉在 live_parallel 和 multi-call。Tool 是最不适合盲目放大 residual 的任务。
4. **shuffle 没有明显伤 Tool，但也没超过 alpha0.75。** 这说明 PAUH 当前的 owner/protected exposure 排序不是对 Tool 的唯一充分条件；随机层排序在这个 seed 下仍能保住 Tool，但 inverse 明显失败，证明“层方向”至少不是完全无关。

### 5.3 对结构 claim 的修正

PAUH 的因果 claim 不能写成“prompt exposure 排序严格决定性能”。更准确的表述是：

> attention-conditioned exposure provides a useful structural prior, but its causal strength is task-dependent. Tool is sensitive to wrong layer ordering and over-scaling; Memory is dominated by MLP/raw residual strength; Code requires separate pass/fail trajectory evidence.

这个结论比单纯报告 PAUH 分数更有价值，因为它解释了不同任务为什么对系数策略反应不同：

- Tool：层位和格式 span 敏感；
- Memory：MLP/residual strength 主导；
- Code：当前 gate/layer signal 仍不足，需要 contrastive ability direction。

## 6. Code pass/fail contrast 补充结论

补充诊断路径：

- input pairs: `/tmp/shared-storage/ExpertGym/attention_matrix/code_contrast_signed_utility_round16_s8_20260521/pair_summary.json`
- positive probe: `/tmp/shared-storage/ExpertGym/attention_matrix/code_contrast_signed_utility_round16_s8_20260521/positive_probe/signed_utility_summary.json`
- negative probe: `/tmp/shared-storage/ExpertGym/attention_matrix/code_contrast_signed_utility_round16_s8_20260521/negative_probe/signed_utility_summary.json`
- 详细表：`docs/report/20260521_spre_structure_validation.md`

核心发现：

| range | positive Code utility | negative Code utility | pass - fail |
| --- | ---: | ---: | ---: |
| all probed layers | 3.773e-07 | 4.868e-07 | -1.095e-07 |
| without layer 27 | 5.828e-07 | 2.893e-07 | 2.935e-07 |

Code 的 pass/fail contrast 给出两个互相约束的事实：

1. 中层 `8-20` 存在 pass > fail 的弱正方向，说明 Code 不是完全没有可学习 residual signal。
2. layer 27 强烈反向，且 positive Code 轨迹对 Memory/Tool 的 protected harm 更大；因此 Code 不能靠全局 gate 或全层 OPD/NLL 硬推。

这解释了结构-性能矩阵中的现象：TRC 训练可以把 Code gate 推到 `1.2+`，但 Code official acc 仍停在 `0.32` 左右；问题不是 gate 没动，而是 gate 动到的 residual 子空间没有稳定对应评测能力。后续 Code 方法应优先尝试中层/模块选择、pass-fail margin、以及冻结或惩罚 late layer 27 的 Code 更新。
