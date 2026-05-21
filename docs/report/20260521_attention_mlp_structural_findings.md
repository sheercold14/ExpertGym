# 2026-05-21 Attention / MLP 结构性结论总报告

## 目标

用 attention matrix、MLP/linear residual exposure、signed utility、以及真实 gate ablation 解释：

> task vector 系数为什么会影响 Tool / Memory / Code 性能，以及哪些系数变化是结构上合理的。

这份报告不是训练日志，而是方法判断依据。核心结论是：**task vector merge 不是一个平滑全局 scaling 问题；不同 agent 能力在 layer、attention/MLP family、owner utility/protected harm 上有明显不同结构。**

## 产物索引

| 模块 | 报告 / 产物 |
| --- | --- |
| Attention matrix pattern | `docs/report/20260521_attention_matrix_diagnostics.md` |
| MLP / linear exposure | `docs/report/20260521_task_pattern_attention_mlp_analysis.md` |
| SPRE gate 与 ablation | `docs/report/20260521_spre_structure_validation.md` |
| Gate-性能矩阵 | `docs/report/20260521_gate_structure_performance_matrix.md` |
| PAUH 排序 / inverse / shuffle | `docs/report/20260521_prompt_attention_utility_harm.md` |
| Code pass/fail contrast | `docs/report/20260521_spre_structure_validation.md#9-code-passfail-contrast-诊断` |
| gate summary utility | `scripts/attention_pauh/summarize_gate_structure.py` |
| signed utility probe | `scripts/attention_pauh/probe_signed_utility.py` |
| SPRE builder | `scripts/attention_pauh/build_signature_preserving_gates.py` |

## 1. Attention 不是能力本身，但揭示任务信息流

Attention matrix 诊断显示三类任务的信息流不同：

| task | prompt-tail / prompt anchoring | response-local | 结构解释 |
| --- | ---: | ---: | --- |
| Memory | 高，late prompt mass 约 `0.81` | 低，约 `0.19` | 需要持续读取问题、证据、memory updates，属于 prompt/sink anchored retrieval |
| Tool | 中高，约 `0.70` | 中等，约 `0.30` | 既要读 schema，也要生成严格 tool call behavior |
| Code | 较低，约 `0.52` | 高，约 `0.41` | 更多依赖 response-local construction，即代码逐步构造 |

推论：

- Memory gate 不能只按“attention 是否跨 prompt”来判 harm；prompt anchoring 是 owner 能力的一部分。
- Tool 的关键不是只会输出 `<tool_call>`，而是 schema reading + call count + argument matching。
- Code 的 attention 行为更像局部构造；只看 prompt attention 很难推出 code acc。

## 2. MLP 是主要 residual 表达通道

Raw linear exposure 显示 MLP 尤其重要：

| task owner | all response exposure | mlp_up | mlp_down | attn_v |
| --- | ---: | ---: | ---: | ---: |
| Tool | `0.00545` | `0.01418` | `0.00459` | `0.00028` |
| Memory | `0.11119` | `0.31424` | `0.09120` | `0.00525` |
| Code | `0.00319` | `0.01566` | `0.00288` | `0.00023` |

结论：

- Memory residual 幅值比 Tool/Code 大一个数量级以上，因此 Memory gate 更容易被训练信号推出来。
- MLP 是能力幅值主通道；attention 更像路由、格式、长上下文对齐通道。
- Code delta 的 raw exposure 很弱，解释了为什么 Code gate 即使变大，也不一定能稳定带来 code acc。

## 3. Signed utility 说明 owner utility 和 protected harm 不能混同

Signature span signed utility 使用 task-relevant span：

- Tool：`<tool_call>...</tool_call>`
- Code：markdown code block
- Memory：response span

核心量级：

| expert | owner utility sum | best layer | best layer utility |
| --- | ---: | ---: | ---: |
| Memory | `1.610e-02` | 27 | `4.061e-03` |
| Tool | `3.423e-04` | 20 | `7.675e-05` |
| Code | `1.178e-06` | 16 | `3.651e-07` |

解释：

- Memory 是强正 residual，且 MLP family 支撑最明显。
- Tool 是弱正 residual，依赖 MLP + `attn_o`，q/k/v 不应盲目放大。
- Code 的 owner utility 极弱，而且 protected harm 相对更大；这就是 Code OPD/NLL 很难直接转化为 official acc 的结构原因。

## 4. Code 的困难来自非单调 residual utility

同 prompt pass/fail contrast 给出更清楚的 Code 结论：

| range | positive Code utility | negative Code utility | pass - fail |
| --- | ---: | ---: | ---: |
| all probed layers | `3.773e-07` | `4.868e-07` | `-1.095e-07` |
| without layer 27 | `5.828e-07` | `2.893e-07` | `2.935e-07` |

解释：

- Code 不是完全没有可学信号；中层 `8-20` 有 pass > fail 的弱正方向。
- layer 27 强烈反向，足以抵消中层信号。
- Positive Code trajectory 对 Memory/Tool 的 protected harm 还更高。

所以 Code 不能靠“全局 Code gate 增大”解决。更合理的结构约束是：

- 优先中层 `8-20`；
- 谨慎或冻结 layer 27 Code gate；
- 区分 `mlp_gate/mlp_up/attn_o/attn_v` 与 `mlp_down`；
- 使用 pass/fail margin，而不是只用 positive NLL。

## 5. Gate 结构与真实性能对齐

| variant | Tool | Memory | Code | 结构解释 |
| --- | ---: | ---: | ---: | --- |
| SPRE-v2 | BFCL quick `0.7825` | HotpotQA F1 `0.7761` | 未测 | 保留 Memory/Tool，全 MLP；轻压 Code attention |
| SPRE static code-attn shrink | BFCL quick `0.7631` | 未测 | 未测 | Code attention 压太重会伤 BFCL live_parallel |
| SPRE mlp-preserve attn-calm | BFCL quick `0.7813` | HotpotQA F1 `0.7462` | 未测 | 全 attention 降低，Memory 明显下降 |
| SPRE memory-attn-calm | BFCL quick `0.7788` | HotpotQA F1 `0.7479` | 未测 | 只压 Memory attention，Tool 保持但 Memory 降 |
| PAUH layer-all | BFCL quick `0.7954` | HotpotQA F1 `0.7362` | CURE acc `0.3506` | 平均 alpha `0.75`，层分配带来 Code 相对优势 |
| PAUH attn-only | BFCL quick `0.7930` | HotpotQA F1 `0.6400` | 未测 | 无 MLP 注入，Memory 大幅下降 |
| TRC R5A selected | BFCL quick `0.8035` | HotpotQA F1 `0.7638` | CURE acc `0.3194` | Code gate 高但 deterministic Code 不强 |
| TRC R8D selected | BFCL quick `0.7944` | HotpotQA F1 `0.7668` | CURE acc `0.3203` | code-block span 改善有限，LiveCodeBench 仍弱 |

关键判断：

- Code gate 高不等于 Code acc 高；R5A/R8D 把 Code gate 推到 `1.24` 左右，但 acc 仍只有 `0.32`。
- PAUH 平均 alpha 只有 `0.75`，但 Code acc 到 `0.3506`，说明层结构比全局幅值更重要。
- Memory 需要 MLP，也需要自身 attention routing；只保留其中一个都会掉。
- Tool source 能力在 ToolRL all80 上稳定，但 BFCL live/multi-call 需要单独看 schema/call-count behavior。

## 6. 对论文方法的结构性 insight

可以形成的核心 insight：

1. **Agent task vectors have task-specific expression channels.** Memory 主要通过 MLP 强残差表达；Tool 需要 MLP + output attention；Code 的有效信号分散且弱。
2. **Coefficient learning is not globally smooth.** 同一 expert 的不同 layer/family 可以有相反 utility；Code layer 27 是典型例子。
3. **Owner utility and protected harm must be separated.** Positive expert trajectory 可能同时提高 owner likelihood 和 protected harm；不能把 OPD/NLL 当天然安全目标。
4. **Attention is a routing constraint, not just a pruning target.** Memory attention 被压低会掉 F1；Tool 的错误层排序会伤 BFCL；Code attention 需要按 family/layer 区分。
5. **Useful gates should be structure-first, reward-verified.** 先由 attention/MLP/signed utility 定位结构，再用少量 executable reward 验证，而不是做无解释 sweep。

## 7. 当前可落地建议

Memory：

- 保留 MLP 和 attention，尤其不要全局压 Memory attention。
- Memory gate 可以高，但应避免用 Code contrast 把 late layer 混合推乱。

Tool：

- ToolRL test 可作为 source 能力 sanity check；BFCL live/multi-call 是独立目标。
- Tool 不适合盲目 alpha=1 全层增强；应重点保护 tool-call span、`attn_o` 和 MLP。

Code：

- 不要再用全局 Code gate 是否上涨判断训练成功。
- 优先试中层 `8-20` 的 pass/fail margin。
- layer 27 Code 更新应冻结、降低、或作为 harm 项惩罚。
- Code calibration 应选择能在 pass/fail contrast 上产生明确 residual 差分的样本，而不是只看 expert 是否做对。

## 8. 验证状态

代码验证：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m py_compile scripts/attention_pauh/probe_attention_matrix_patterns.py \
  scripts/attention_pauh/probe_linear_module_exposure_patterns.py \
  scripts/attention_pauh/probe_signed_utility.py \
  scripts/attention_pauh/build_signature_preserving_gates.py \
  scripts/attention_pauh/summarize_gate_structure.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m unittest discover -s tests -p 'test_attention_pauh.py' -v
```

最近一次结果：`14` 个 attention/PAUH 单测全部通过。

