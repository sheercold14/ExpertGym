# 2026-05-21 Attention Matrix 诊断：Tool / Memory / Code 的结构性差异

## 实验设置

- 目的：先获得诊断性 insight，不直接调参，不用评测指标反向设计方法。
- 模型：`/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct`
- 数据：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl`
- 抽样：每任务 4 条轨迹，`tool/memory/code` 各 4 条。
- 截断：`max_seq_length=1536`，response tail 512 tokens，prompt tail 384 tokens。
- 输出：`/tmp/shared-storage/ExpertGym/attention_matrix/pattern_probe_20260521_s4_len1536/attention_pattern_summary.md`
- 脚本：`scripts/attention_pauh/probe_attention_matrix_patterns.py`

命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/attention_pauh/probe_attention_matrix_patterns.py \
  --base-model /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct \
  --trajectory-jsonl /tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/attention_matrix/pattern_probe_20260521_s4_len1536 \
  --samples-per-task 4 \
  --max-seq-length 1536 \
  --response-tail-tokens 512 \
  --prompt-tail-tokens 384 \
  --local-window 128 \
  --sink-tokens 16 \
  --device cuda \
  --torch-dtype bfloat16
```

## 核心观测

| task | layer group | prompt mass | prompt tail | local response | long response | sink mass | entropy |
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

最强 task contrast：

- Memory 的 response tokens 对 prompt 的注意力最高：middle `0.7987`，late `0.8138`。
- Code 的 response-local 注意力最高：early `0.4815`，middle `0.4537`，late `0.4050`。
- Tool 介于二者之间，并且 late layer 更 prompt anchored：prompt mass 从 `0.6030` 升到 `0.7019`。
- Memory 的 sink mass 在 middle/late 明显高：`0.4454/0.4568`；这说明 memory 轨迹存在强 attention sink / global routing 现象。
- Tool 的 head prompt std 最大，尤其 middle/late：`0.1994/0.2061`；这暗示 tool 能力可能依赖少数 specialized heads 读取 schema / tool-call 相关上下文。

## 解释

### 1. Memory 是 prompt-anchored retrieval，不是 response-local generation

Memory 在所有层组都有 `0.78-0.81` 的 attention mass 指向 prompt，而 response-local 只有 `0.18-0.22`。这和任务本质一致：Hotpot / memory 轨迹最终回答依赖前文事实、检索结果、更新记录，而不是只靠当前生成片段自回归续写。

这解释了 PAUH v1 为什么容易压低 memory late layers：v1 用 exposure ratio 做 utility/harm，memory delta 在 prompt-heavy 层对多个任务都有高 exposure，就容易被误判为 harm。真实判断不能只看“看到了哪里”，必须看该 residual 对 teacher-forced loss 的一阶方向是降 loss 还是升 loss。

### 2. Code 是 response-local construction，prompt attention 不足以表达能力

Code 的 prompt mass 只有 `0.45-0.52`，local response mass 高达 `0.40-0.48`，并且有稳定 long-response mass `0.06-0.08`。这说明 code 能力更像“在生成中的 reasoning/code block 内保持局部一致性、约束传递、代码结构续写”，而不是只在 prompt 侧读取题面。

因此，只用 prompt attention 或 prompt hidden state 去定 code gate 很可能不充分。Code 的 calibration 更应该包含：

- pass trajectory 和 fail trajectory 的同 prompt contrast；
- reasoning span + final code span；
- code block 内局部 token 的 residual direction；
- generated tests / hidden-like guard tests 作为 trajectory 质量筛选。

### 3. Tool 是 prompt-schema + behavior span 的混合任务

Tool 的 prompt mass 中等偏高，并且 late layer 从 `0.6030` 增到 `0.7019`；marker mass 也从 `0.0255` 增到 `0.0335`。这符合 tool-call 输出需要读取工具定义、参数 schema、用户约束，再在 response/tool-call span 中保持格式。

Tool 的 high head_prompt_std 表明不是所有 head 平均工作，而是少数 head 更强地读 prompt/schema。保护 tool 时，全局 NLL retention 可能太弱；更合理的是对 tool-call behavior span 或 specialized attention subspace 做强保护。

## 对方法的启发

Attention matrix 可以提供“能力表达位置”，但不能单独决定 gate 方向。

更合理的下一步不是用 attention mass 直接当系数，而是分两步：

1. 用 attention matrix 找 task signature：
   - memory：prompt anchored / sink-heavy / retrieval span；
   - code：response-local / code-block construction；
   - tool：prompt-schema + tool-call behavior span。
2. 在这些 signature span 上计算 signed residual utility：
   - `utility(e,l,t) = - grad(loss_t) · delta_e,l`；
   - 非 owner task 上 `max(0, -utility)` 作为 harm；
   - gate 由 signed utility-harm 决定，而不是由 attention exposure ratio 决定。

这能把“看哪里”和“这个 residual 是否真的有用”分开，符合第一性原理，也比直接 sweep scaling factor 更像论文方法。

## 需要继续验证

- 当前结果是每任务 4 条、1536 tokens 的小样本诊断，不是最终统计。
- Memory 原 prompt 很长，左截断会保留 prompt tail + response；完整长上下文 attention 还需要分段窗口或稀疏统计。
- Attention sink 很强，后续需要把 sink mass 从 prompt utility 里分离，否则容易把全局 sink 误认为 task-specific prompt usage。
- 下一步应运行 `scripts/attention_pauh/probe_signed_utility.py`，在上述 signature span 上验证每个 expert/layer 的 residual 是否真的降低 owner-task loss，并度量跨任务 harm。
