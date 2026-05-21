# 2026-05-21 Code Hurt Pass/Fail Signed-Utility Contrast

## 目的

这一步验证 Code 能力归因是否能从“靠近 expert 成功代码”升级为：

```text
同 prompt 下，residual 对 pass trajectory 的 utility
是否大于 residual 对 fail trajectory 的 utility
```

即：

```text
contrast = signed_effect(pass) - signed_effect(fail)
```

这比单独看 expert positive NLL 更合理，因为 Code 的失败常常不是风格不像 expert，而是输入格式、边界条件、算法选择或局部实现错。

## 数据

使用刚构造的 LiveCodeBench hurt subset：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs/LiveCodeBenchCodeHurtRcrfVsTa16.positive_code.jsonl
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs/LiveCodeBenchCodeHurtRcrfVsTa16.negative_code.jsonl
```

本次只做小规模真实 probe：

```text
samples = 4
layers = 20-27
scope = all-linear
span = response
entries = 8 layers x 7 modules x 3 experts = 168
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/livecodebench_positive_code_l20_27_s4_20260521
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/livecodebench_negative_code_l20_27_s4_20260521
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livecodebench_code_l20_27_s4_20260521
```

## 命令

Positive pass-code probe：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/attention_pauh/probe_signed_utility.py \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --trajectory-jsonl /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs/LiveCodeBenchCodeHurtRcrfVsTa16.positive_code.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/livecodebench_positive_code_l20_27_s4_20260521 \
  --tasks code --experts tool,memory,code --scope all-linear \
  --layers 20-27 --samples-per-task 4 --span response --max-seq-length 4096 --write-row-details
```

Negative fail-code probe：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/attention_pauh/probe_signed_utility.py \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --trajectory-jsonl /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs/LiveCodeBenchCodeHurtRcrfVsTa16.negative_code.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/livecodebench_negative_code_l20_27_s4_20260521 \
  --tasks code --experts tool,memory,code --scope all-linear \
  --layers 20-27 --samples-per-task 4 --span response --max-seq-length 4096 --write-row-details
```

Pass/fail contrast：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/analysis/compare_signed_utility_contrast.py \
  --positive-rows /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/livecodebench_positive_code_l20_27_s4_20260521/signed_utility_rows.jsonl \
  --negative-rows /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/livecodebench_negative_code_l20_27_s4_20260521/signed_utility_rows.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livecodebench_code_l20_27_s4_20260521 \
  --top-k 16
```

## 结果

| expert | module count | mean contrast | mean pass utility | mean fail utility |
|---|---:|---:|---:|---:|
| code | 56 | 1.2288e-05 | 6.1613e-10 | -1.2288e-05 |
| memory | 56 | 2.6602e-05 | 1.5179e-07 | -2.6450e-05 |
| tool | 56 | -1.2334e-05 | 4.8468e-08 | 1.2383e-05 |

Top layer/module contrast magnitude：

| layer | module | mean contrast | mean abs contrast |
|---:|---|---:|---:|
| 27 | down | 5.5397e-05 | 5.9040e-04 |
| 26 | q | 3.6503e-04 | 4.0789e-04 |
| 24 | q | 1.1594e-04 | 1.9457e-04 |
| 26 | down | 5.0374e-05 | 1.4628e-04 |
| 23 | down | -1.1841e-04 | 1.1841e-04 |
| 26 | k | 8.1087e-05 | 1.0842e-04 |
| 22 | down | -1.0548e-04 | 1.0548e-04 |
| 26 | gate | 7.9468e-05 | 1.0012e-04 |

Code expert 的 top positive contrast：

| layer | module | contrast | pass utility | fail utility |
|---:|---|---:|---:|---:|
| 27 | down | 9.6869e-04 | -8.2850e-06 | -9.7698e-04 |
| 21 | gate | 4.3457e-05 | -4.3650e-07 | -4.3894e-05 |
| 20 | gate | 4.2776e-05 | -3.7381e-07 | -4.3150e-05 |
| 20 | down | 3.8788e-05 | -3.4529e-07 | -3.9133e-05 |
| 27 | up | 3.2633e-05 | -5.5414e-07 | -3.3187e-05 |

Code expert 的 top negative contrast：

| layer | module | contrast | pass utility | fail utility |
|---:|---|---:|---:|---:|
| 24 | q | -1.1794e-04 | 1.3588e-06 | 1.1930e-04 |
| 24 | down | -6.6360e-05 | 1.2033e-06 | 6.7563e-05 |
| 26 | q | -6.4295e-05 | 1.2023e-06 | 6.5498e-05 |
| 23 | down | -6.1894e-05 | 9.5926e-07 | 6.2854e-05 |
| 25 | down | -5.5389e-05 | 8.0094e-07 | 5.6190e-05 |

## 解释

这组小 probe 给出一个重要结论：

> 对 Code hurt 样本，单独看 positive pass-code utility 很弱；真正有区分度的是 pass/fail contrast。

原因：

- pass-code 的 signed utility 均值接近 0；
- fail-code 在某些 residual 上反而有很强 signed utility；
- 这说明“这个 residual 能降低 teacher-forced NLL”不等于“这个 residual 支持正确 Code 能力”。

更具体地：

- layer 27 `mlp.down` 对 Code expert 有最大正 contrast，但 pass utility 本身仍是负的；它的意义更像“fail trajectory 上该 residual 很有害，因此 pass-fail 差分能识别出应该避免的失败方向”。
- layer 24/26 `q` 和若干 `mlp.down` 对 Code expert 是负 contrast：它们更支持失败代码，不能因为来自 Code expert 就保留。
- Tool expert 在这个 Code subset 上平均 contrast 为负，说明它对 Code pass/fail 区分不稳定；后续 Code 归因不应把非 owner expert 的高表达直接当作协同。
- Memory expert 在若干 late attention q / mlp down 上有正 contrast，提示 Code 能力不一定只来自 Code expert，可能有通用题意/结构处理 residual。

## 方法含义

下一版 Code attribution 不应使用：

```text
utility(pass_code)
```

作为唯一选择标准，而应使用：

```text
contrast_utility = utility(pass_code) - utility(fail_code)
```

并结合：

```text
expression energy
positive fraction across pairs
cross-task harm/conflict
```

保守规则：

- 放大：owner 或跨 expert residual 的 contrast 为正，且 positive fraction 稳定。
- 压制：contrast 为负，尤其是 fail utility 明显大于 pass utility 的模块。
- 观察：pass/fail 都接近 0 的低表达模块，不作为核心能力 residual。

这正好服务于论文主线：不是学习一个全局 task-vector 系数，而是识别每个 residual 在成功/失败行为上的局部因果方向。

## 限制

本次只是 sanity probe：

- 只取 4 条 LiveCodeBench hurt cases；
- 只看 layers 20-27；
- 只看 final extracted code span；
- 未跑 prompt span / reasoning span；
- 不能直接作为最终实验结论。

下一步需要扩展到：

```text
LiveBench hurt16 + LiveCodeBench hurt16
prompt span + code span
all layers 或分层抽样
```

## Full Code-Span Probe

随后补跑了完整 code-span probe：

```text
LiveBench hurt16 + LiveCodeBench hurt16
samples = 16 each
layers = 0-27
scope = all-linear
span = response on extracted code
module entries = 588 per dataset
pair rows = 9408 per dataset
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livebench_code_alllayers_s16_20260521
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livecodebench_code_alllayers_s16_20260521
```

### Expert-Level Contrast

| dataset | expert | mean contrast | mean pass utility | mean fail utility |
|---|---|---:|---:|---:|
| LiveBench | code | 4.2779e-09 | 5.7154e-08 | 5.2876e-08 |
| LiveBench | memory | -9.0544e-08 | 4.2069e-07 | 5.1123e-07 |
| LiveBench | tool | 4.2332e-08 | 1.2956e-08 | -2.9376e-08 |
| LiveCodeBench | code | -3.8514e-06 | -3.5853e-08 | 3.8155e-06 |
| LiveCodeBench | memory | 2.4812e-05 | 3.1529e-07 | -2.4496e-05 |
| LiveCodeBench | tool | -5.1399e-06 | 5.1412e-08 | 5.1913e-06 |

Interpretation：

- LiveBench 的整体 contrast 近似 0，说明这些 near-miss 题的 pass/fail 差别不强烈集中在 final code span；更可能需要测试选择、边界条件局部修补，或 prompt/reasoning span 解释。
- LiveCodeBench 的 contrast 很强，尤其 memory expert 正 contrast、code/tool 平均负 contrast，说明在这些 0/8 hurt case 上，不同 expert residual 对成功/失败行为的方向确实可区分。
- 这进一步说明 Code 不能用统一的 imitation 信号：LiveBench 和 LiveCodeBench 需要分开归因。

### Dataset-Level Hot Modules

| dataset | hot layer/module | mean contrast | mean abs contrast |
|---|---|---:|---:|
| LiveBench | L27 down | -2.6755e-06 | 4.9247e-06 |
| LiveBench | L27 up | -1.2433e-06 | 1.2554e-06 |
| LiveBench | L26 q | 9.2425e-07 | 1.0524e-06 |
| LiveBench | L14 up | -5.7623e-08 | 7.1878e-07 |
| LiveBench | L26 down | -6.9615e-07 | 6.9615e-07 |
| LiveCodeBench | L27 down | 1.2245e-04 | 5.8822e-04 |
| LiveCodeBench | L26 q | 3.7697e-04 | 4.2257e-04 |
| LiveCodeBench | L24 q | 1.3125e-04 | 2.2186e-04 |
| LiveCodeBench | L26 down | 5.9785e-05 | 1.8652e-04 |
| LiveCodeBench | L26 gate | 1.0019e-04 | 1.2799e-04 |

Stable observations：

- L27 `mlp.down` remains the largest contrast-magnitude module in both datasets, but sign and expert decomposition differ.
- L26/L24 attention `q` is consistently high contrast, especially on LiveCodeBench.
- MLP `down/up/gate` modules dominate most large-magnitude contrast positions; this supports treating MLP residuals as behavior execution/implementation channels rather than only attention routing.

### LiveCodeBench Expert-Specific Findings

For LiveCodeBench hurt16:

- `memory` expert has strongest positive mean contrast: `2.4812e-05`.
- `code` expert mean contrast is negative: `-3.8514e-06`.
- `tool` expert mean contrast is negative: `-5.1399e-06`.

This is a nontrivial result: the residuals that distinguish pass/fail Code behavior are not simply “the Code expert residuals”. Memory expert residuals appear to encode useful general structure for these Code failures, while many Code expert residuals are more aligned with fail code than pass code.

Top LiveCodeBench memory positive contrast:

| layer | module | contrast |
|---:|---|---:|
| 26 | q | 1.0026e-03 |
| 24 | q | 4.6314e-04 |
| 26 | down | 3.6946e-04 |
| 18 | o | 2.7188e-04 |
| 26 | k | 2.7117e-04 |

Top LiveCodeBench code negative contrast:

| layer | module | contrast |
|---:|---|---:|
| 24 | q | -1.3590e-04 |
| 17 | up | -1.2706e-04 |
| 24 | down | -8.7200e-05 |
| 16 | up | -8.6548e-05 |
| 23 | down | -8.0701e-05 |

### Updated Method Implication

The practical rule should not be:

```text
keep owner expert residual if it helps owner task positive trajectory
```

It should be:

```text
keep residuals with stable positive pass/fail contrast;
suppress residuals with stable negative pass/fail contrast;
then regularize with cross-task conflict/harm.
```

For Code specifically:

```text
Code ability = prompt/reasoning attribution + final-code pass/fail contrast
```

The final-code contrast alone explains LiveCodeBench much better than LiveBench. Therefore next probe should add prompt/reasoning spans, especially for LiveBench near-miss cases.

## Contrast-Aware Gate Overlay v1

基于上面的 pass/fail contrast，新增了一个不训练的 gate candidate：

```text
scripts/attention_pauh/build_contrast_aware_residual_gates.py
```

核心规则：

```text
base_gate = RCRF gate
contrast = utility(pass_code) - utility(fail_code)
normalized_score = contrast / robust_q90_abs_contrast
new_gate = base_gate + max_delta * clipped(normalized_score) * confidence
```

其中 `confidence` 来自同 prompt pair 上 contrast 方向稳定性。默认额外做 per-expert mean recenter：

```text
mean(new_gate[expert]) = mean(base_gate[expert])
```

因此它不是调大/调小某个 expert 的全局系数，而是只回答一个问题：

> 在相同 expert 总预算下，把 residual 预算从 fail-aligned 位置转到 pass-aligned 位置，是否能修复 Code 受伤样本？

生成命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/attention_pauh/build_contrast_aware_residual_gates.py
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_contrast_v1/gates.json
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_contrast_v1/decision_rows.jsonl
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_contrast_v1/summary.md
```

结构审计：

| expert | mean | std | min | max |
|---|---:|---:|---:|---:|
| code | 0.900703 | 0.125513 | 0.649871 | 1.114642 |
| memory | 0.987386 | 0.107896 | 0.668810 | 1.119044 |
| tool | 1.004154 | 0.121268 | 0.667309 | 1.120000 |

相对 RCRF 的变化：

| group | mean abs delta | max abs delta | positive | negative |
|---|---:|---:|---:|---:|
| code | 0.006022 | 0.017771 | 144 | 52 |
| memory | 0.010592 | 0.029258 | 82 | 114 |
| tool | 0.004526 | 0.024656 | 111 | 46 |

读法：

- 变化幅度很小，最大单参数系数变化约 `0.029`，适合作为 first validation，不是强干预。
- Code expert 内部多数位置被轻微上调，但不会改变 code expert 总预算。
- Memory expert 在 LiveCodeBench pass/fail contrast 中承担了大量正/负分流，是这版候选最值得验证的部分。
- Tool 也有轻微重分配；如果 Tool 快速评测下降，说明 Code contrast 对 Tool 行为 span 有副作用，需要加入 Tool source-preserve contrast。

下一步验证顺序：

1. bake 这个 `gates.json`；
2. 先测 Tool quick + Memory quick，确认没有明显破坏；
3. 再跑 32 条 Code hurt subset；
4. 只有 hurt subset 有收益时，才进入完整 CURE。

## Quick Validation

已 bake：

```text
/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_contrast_v1
```

`--plan-only` 和实际 bake 都识别到：

```text
num_delta_entries = 588
```

Code hurt subset 快速结果：

| dataset | RCRF pass_any | contrast pass_any | RCRF pass_rate | contrast pass_rate | contrast BoN `(4,4)` |
|---|---:|---:|---:|---:|---:|
| LiveBench hurt16 | 0.0000 | 0.2500 | 0.0000 | 0.1094 | 0.1250 |
| LiveCodeBench hurt16 | 0.0000 | 0.7500 | 0.0000 | 0.2500 | 0.5000 |

对照 TA 上限：

| dataset | TA0.75 pass_any | TA0.75 pass_rate | TA1.0 pass_any | TA1.0 pass_rate |
|---|---:|---:|---:|---:|
| LiveBench hurt16 | 0.6250 | 0.2344 | 0.5625 | 0.1562 |
| LiveCodeBench hurt16 | 0.8750 | 0.4844 | 0.7500 | 0.3750 |

解释：

- 这不是完整 Code 提升结论，但它证明了“同 prompt pass/fail contrast”不是空信号：在原 RCRF 全 fail 的 32 条上，候选恢复了一批 pass-any 样本。
- LiveCodeBench 的恢复更强，符合之前诊断：它的 final-code contrast 信号大而稳定。
- LiveBench 恢复弱，且 test-point rate 不理想，说明 near-miss 题不能只看 code span；下一步应对 LiveBench hurt16 做 prompt span / reasoning span contrast。

## Span-Aware Conservative Overlay v2

在 full code-span 之后，补跑了 prompt / reasoning span，并构造第二个 gate overlay：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_conservative_v2/gates.json
/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2
```

输入来源：

```text
LiveBench prompt span
LiveBench reasoning span
LiveCodeBench final-code span
LiveCodeBench prompt span
```

构造规则：

| item | value |
|---|---|
| base gate | RCRF v1 |
| normalization | `per-file` |
| aggregation | `conservative` |
| conflict penalty | `0.35` |
| max delta | `0.05` |
| preserve expert mean | `true` |

结构变化：

| expert | mean | std | min | max |
|---|---:|---:|---:|---:|
| code | 0.900703 | 0.128507 | 0.648780 | 1.112169 |
| memory | 0.987386 | 0.104991 | 0.672243 | 1.120000 |
| tool | 1.004162 | 0.120750 | 0.662548 | 1.120000 |

相对 RCRF：

| group | changed | positive | negative | mean abs delta | max abs delta |
|---|---:|---:|---:|---:|---:|
| overall | 562 | 271 | 291 | 0.006417 | 0.041807 |
| code | 196 | 122 | 74 | 0.005943 | 0.031544 |
| memory | 195 | 66 | 129 | 0.007477 | 0.036293 |
| tool | 171 | 83 | 88 | 0.005831 | 0.041807 |

conservative aggregation 发现：

```text
source_sign_conflict_suppress = 261
source_sign_agreement = 218
no_informative_source = 109
```

这说明 Code 的 residual 不是单一方向。许多位置在不同 span 或不同子数据集上符号相反，直接按平均 signed utility 放大容易把一个子分布的正确方向变成另一个子分布的伤害方向。

### Source-Conflict Diagnosis

对五个来源做 pairwise 对比：

```text
LB_code, LB_prompt, LB_reasoning, LCB_code, LCB_prompt
```

最强冲突：

| pair | overlap | pearson | sign conflicts | agreements |
|---|---:|---:|---:|---:|
| LiveBench prompt vs LiveCodeBench prompt | 588 | -0.9948 | 376 | 212 |
| LiveBench prompt vs LiveCodeBench code | 588 | -0.0145 | 332 | 256 |
| LiveBench reasoning vs LiveCodeBench prompt | 588 | -0.0543 | 332 | 256 |
| LiveBench code vs LiveCodeBench prompt | 588 | -0.0144 | 310 | 278 |
| LiveBench code vs LiveCodeBench code | 588 | -0.0382 | 300 | 288 |

典型冲突：

```text
code / model.layers.0.self_attn.o_proj.weight:
LiveBench prompt contrast = +2.050e-05
LiveCodeBench prompt contrast = -1.127e-03
```

这解释了为什么 Code 不能用一个“code expert 系数”描述：同一个 residual 对不同 Code 子分布的 prompt 理解可能方向相反。

### v2 Quick Validation

Code hurt subset 快速验证：

| dataset | RCRF pass_any | v1 pass_any | v2 pass_any | v2 candidate pass_rate | v2 hidden test-point rate | v2 CURE BoN acc |
|---|---:|---:|---:|---:|---:|---:|
| LiveBench hurt16 | 0.0000 | 0.2500 | 0.4375 | 0.1563 | 0.3398 | 0.1875 |
| LiveCodeBench hurt16 | 0.0000 | 0.7500 | 0.8125 | 0.4375 | 0.5189 | 0.7500 |

解释：

- LiveBench：prompt/reasoning span 确实补回了一部分 v1 做不到的 near-miss，但整体 test-point rate 仍低，说明还缺少更细的边界条件/局部修复信号。
- LiveCodeBench：v2 明显优于 v1，说明 final-code contrast 与 prompt span 的保守融合没有互相抵消，反而把 hidden-test 点从 `0.3164` 提到 `0.5189`。
- 这支持一个更干净的论文主张：能力 residual 选择应当 outcome-aware 且 span-aware；同一 task 内部也需要处理 span/source conflict。

### Side-Effect Check

v2 的 Tool/Memory 快评：

| metric | value |
|---|---:|
| BFCL quick mean | 0.7931 |
| BFCL live mean | 0.7188 |
| parallel | 0.8800 |
| parallel_multiple | 0.8550 |
| live_parallel | 0.8125 |
| live_parallel_multiple | 0.6250 |
| HotpotQA eval_50 F1 | 0.7650 |
| HotpotQA eval_100 F1 | 0.7478 |

相对 RCRF v1：

- Tool mean 只下降约 `0.0025`，live mean 不变；
- Memory eval_50 下降约 `0.0058`，eval_100 下降约 `0.0089`。

因此这版可以作为机制验证：Code hurt 修复不是通过牺牲 Tool 得到的。但它还不是最终模式组合解，因为 Memory 有可见小幅副作用。下一版应把 Memory 的有效 span 也作为 source-preserve 约束加入，而不是只让 Code contrast 决定 shared residual 的方向。

## Memory-Preserve Hard Floor v3

为了验证 v2 的 Memory 副作用是否来自“Code contrast 压低 Memory expert residual”，构造了一个最小保护对照：

```text
rcrf_code_spanaware_memory_preserve_v3
```

规则：

- 继承 v2 的四个 span/source；
- 对 `memory` expert 开启 `--protect-negative-expert memory`；
- 所有 memory 负向 overlay 被置零；
- memory expert 跳过 mean recenter，避免 floor 被抵消。

结构结果：

```text
memory changed = 25
memory positive = 25
memory negative = 0
memory protected_negative_overlay = 130
memory mean: 0.9874 -> 0.9895
```

Code hurt 回归：

| dataset | v2 pass_any | v3 pass_any | v2 test-point | v3 test-point | v2 BoN | v3 BoN |
|---|---:|---:|---:|---:|---:|---:|
| LiveBench hurt16 | 0.4375 | 0.4375 | 0.3398 | 0.4121 | 0.1875 | 0.3750 |
| LiveCodeBench hurt16 | 0.8125 | 0.6250 | 0.5189 | 0.3424 | 0.7500 | 0.3750 |

解释：

- LiveBench near-miss 更喜欢保留/增强 Memory residual，说明部分 Memory residual 参与题意/约束稳定性。
- LiveCodeBench final-code 修复依赖压制一部分 Memory residual；把所有 Memory 负向项硬保护会损失大量 LCB Code 修复。
- 这给出一个反例：保护不能以 expert 为单位做，必须以 span/source/outcome 为单位做。Memory-preserve 应保护“Memory 任务上高 utility 的 residual”，而不是保护所有 Memory expert residual。
