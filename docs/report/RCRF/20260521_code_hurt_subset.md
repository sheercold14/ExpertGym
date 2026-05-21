# 2026-05-21 Code 受伤样本快速回归集

## 目的

当前 Code 迭代最大的瓶颈是完整 CURE 评测太慢，而且大量题目对方法判断没有信息：

- 大家都能做对：无法区分哪种 residual 更好。
- 大家都做不对：短期内难以判断能力来源。
- 参考模型能做、当前模型做错：最适合分析“能力被伤在哪里”。

因此先构造一个小集合，只保留 TA 参考模型能 pass、RCRF 当前模型不能 pass 或 hidden-test 点明显掉分的样本，用于快速评测和下一阶段机制分析。

## 抽取规则

目标模型：

```text
rcrf_v1 = /tmp/shared-storage/OnPolicy/checkpoints/rcrf_v1_20260521/rcrf
```

参考模型：

```text
ta_c075 = /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_scale_sweep_c075/model
ta_c100 = /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_scale_sweep_c100/model
```

每个 CURE 样本使用逐题输出里的 `test_bool_table` 计算：

```text
pass_any        = 4 个 code candidate 中至少一个通过全部 hidden tests
pass_count      = 通过全部 hidden tests 的 candidate 数
test_point_rate = 所有 candidate x hidden tests 的平均通过率
```

保留条件：

```text
至少一个参考模型 pass_any=True
且 RCRF pass_any=False
或参考模型 test_point_rate - RCRF test_point_rate >= 0.25
```

排除：

```text
所有模型都 pass
所有模型都 fail
```

排序：

```text
hurt_score = 1[reference_pass and target_fail] + max(0, ref_test_point_rate - target_test_point_rate)
```

## 产物

根目录：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521
```

| subset | 原始候选数 | 选中数 | CURE dataset name | 明细 |
|---|---:|---:|---|---|
| LiveBench | 16 | 16 | `LiveBenchCodeHurtRcrfVsTa16` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/LiveBenchCodeHurtRcrfVsTa16.md` |
| LiveCodeBench | 53 | 16 | `LiveCodeBenchCodeHurtRcrfVsTa16` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/LiveCodeBenchCodeHurtRcrfVsTa16.md` |

已安装到 CURE 数据目录：

```text
/mnt/cache/wuruixiao/users/lsc/CURE/data/LiveBenchCodeHurtRcrfVsTa16.json
/mnt/cache/wuruixiao/users/lsc/CURE/data/LiveCodeBenchCodeHurtRcrfVsTa16.json
```

## 快速评测命令

推荐使用固定 wrapper，一次跑两个 16 题 hurt subset：

```bash
source /mnt/cache/wuruixiao/miniconda3/etc/profile.d/conda.sh
conda activate CURE
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

MODEL=/path/to/model GPU=0 scripts/eval/run_cure_code_hurt_eval.sh
```

只跑其中一个集合：

```bash
MODEL=/path/to/model GPU=0 \
DATASETS="LiveCodeBenchCodeHurtRcrfVsTa16" \
scripts/eval/run_cure_code_hurt_eval.sh
```

单模型快速测 LiveBench hurt 集：

```bash
source /mnt/cache/wuruixiao/miniconda3/etc/profile.d/conda.sh
conda activate CURE
cd /mnt/cache/wuruixiao/users/lsc/CURE/evaluation

MODEL=/path/to/model
CUDA_VISIBLE_DEVICES=0 python eval.py --use_api False \
  --pretrained_model "$MODEL" \
  --single_eval False \
  --dataset LiveBenchCodeHurtRcrfVsTa16 \
  --k_code 4 \
  --k_case 4 \
  --scale_tuple_list "[(4, 4)]" \
  --temp 1.0 \
  --max_model_len 32768 \
  --max_generation_token 10000 \
  --max_test 8 \
  --num_chunks 16 \
  --gpu_groups "[[0]]" \
  --is_final_eval False \
  --exe_verbose True
```

LiveCodeBench hurt 集只替换：

```bash
--dataset LiveCodeBenchCodeHurtRcrfVsTa16
```

说明：这是快速回归集，不替代最终 CURE 全量评测。它只用于判断某个机制改动是否修复了已知 Code 受伤样本。

## First Candidate Result: `rcrf_code_contrast_v1`

候选 checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_contrast_v1
```

对应 gate：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_contrast_v1/gates.json
```

快速回归命令：

```bash
PYTHON_BIN=/mnt/cache/wuruixiao/miniconda3/envs/CURE/bin/python \
MODEL=/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_contrast_v1 \
GPU=1 \
scripts/eval/run_cure_code_hurt_eval.sh
```

同一 hurt subset 上的对比：

| dataset | model | pass_any | pass_rate / CURE acc | hidden test-point rate | pass_count |
|---|---|---:|---:|---:|---:|
| LiveBench hurt16 | RCRF | 0.0000 | 0.0000 | 0.3262 | 0 |
| LiveBench hurt16 | TA0.75 | 0.6250 | 0.2344 | 0.4785 | 15 |
| LiveBench hurt16 | TA1.0 | 0.5625 | 0.1562 | 0.3867 | 10 |
| LiveBench hurt16 | rcrf_code_contrast_v1 | 0.2500 | 0.1094 | 0.3105 | 7 |
| LiveCodeBench hurt16 | RCRF | 0.0000 | 0.0000 | 0.0553 | 0 |
| LiveCodeBench hurt16 | TA0.75 | 0.8750 | 0.4844 | 0.5664 | 31 |
| LiveCodeBench hurt16 | TA1.0 | 0.7500 | 0.3750 | 0.4551 | 24 |
| LiveCodeBench hurt16 | rcrf_code_contrast_v1 | 0.7500 | 0.2500 | 0.3164 | 16 |

结论：

- `rcrf_code_contrast_v1` 能从原 RCRF 的 `0/16 pass_any` 恢复到 LiveBench `4/16`、LiveCodeBench `12/16`，说明 pass/fail contrast 确实包含可操作的能力信号。
- LiveCodeBench 恢复明显更强，和 full code-span contrast 的诊断一致：LiveCodeBench 的成功/失败差异集中在 final-code span。
- LiveBench 只部分恢复，且 hidden test-point rate 低于原 RCRF，说明 LiveBench near-miss 更可能依赖 prompt/reasoning span 或测试选择稳定性，不能只靠 final-code contrast。
- 这个候选还没有进入完整 CURE；它只是证明小集合上“Code 受伤样本可以被机制性干预修复一部分”。

## 后续分析入口

下一阶段对这 32 条样本做机制分析：

1. 对比 RCRF、TA0.75、TA1.0 的 pass / fail candidate，提取同题成功轨迹和失败轨迹。
2. 分别在 prompt span、reasoning span、final code span 上计算 residual utility。
3. 判断 Code 受伤来自哪类能力：
   - 题意理解和约束跟踪；
   - 算法选择；
   - 边界条件；
   - 输出格式和函数签名；
   - final code 局部实现。
4. 只在这个小集合上快速 bake / eval，看到明确收益后再进入完整 LiveBench + LiveCodeBench。

## Case Pack

进一步把每条 hurt case 整理为“目标模型最佳候选 vs 参考模型最佳候选”的分析包：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/case_pack/LiveBenchCodeHurtRcrfVsTa16.case_pack.md
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/case_pack/LiveCodeBenchCodeHurtRcrfVsTa16.case_pack.md
```

每条 case 包含：

- 题目原文；
- RCRF 最佳候选代码和 hidden-test 通过数；
- TA0.75/TA1.0 中最佳参考候选代码和 hidden-test 通过数；
- 初步能力标签；
- 失败桶：`target_zero_hidden_tests` / `target_low_partial` / `target_near_miss`。

初步统计：

| subset | zero hidden tests | low partial | near miss | 主要能力桶 |
|---|---:|---:|---:|---|
| LiveBench hurt16 | 2 | 4 | 10 | greedy/operation、math/set、string |
| LiveCodeBench hurt16 | 11 | 3 | 2 | string、math/set、greedy、grid、dp/bitwise |

解释：

- LiveBench 受伤题多数是 near-miss，说明模型常常已经接近正确，主要可能是边界条件、约束细节或局部实现错误。
- LiveCodeBench 受伤题多数是 0/8，说明很多题不是简单修补代码块，而是 prompt 解析、输入格式、算法选择或题型识别失败。
- 下一步 residual 归因应分开做：LiveBench 更适合 pass/fail code span contrast；LiveCodeBench 更需要 prompt span + reasoning span 归因。

## 复现命令

抽取工具：

```text
scripts/eval/build_cure_hurt_subset.py
```

LiveBench：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/build_cure_hurt_subset.py \
  --dataset-name LiveBench \
  --dataset-json /mnt/cache/wuruixiao/users/lsc/CURE/data/LiveBench.json \
  --target-name rcrf_v1 \
  --target-output /tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data/outputs-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_v1_20260521.rcrf-LiveBench.json \
  --reference ta_c075=/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data/outputs-eval-.tmp.shared-storage.AgentMerging_plan.experiments.task_arithmetic.ta_scale_sweep_c075.model-LiveBench.json \
  --reference ta_c100=/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data/outputs-eval-.tmp.shared-storage.AgentMerging_plan.experiments.task_arithmetic.ta_scale_sweep_c100.model-LiveBench.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521 \
  --subset-name LiveBenchCodeHurtRcrfVsTa16 \
  --top-k 16 \
  --min-point-delta 0.25 \
  --install-to-cure-data /mnt/cache/wuruixiao/users/lsc/CURE/data
```

LiveCodeBench：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/build_cure_hurt_subset.py \
  --dataset-name LiveCodeBench \
  --dataset-json /mnt/cache/wuruixiao/users/lsc/CURE/data/LiveCodeBench.json \
  --target-name rcrf_v1 \
  --target-output /tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data/outputs-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_v1_20260521.rcrf-LiveCodeBench.json \
  --reference ta_c075=/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data/outputs-eval-.tmp.shared-storage.AgentMerging_plan.experiments.task_arithmetic.ta_scale_sweep_c075.model-LiveCodeBench.json \
  --reference ta_c100=/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data/outputs-eval-.tmp.shared-storage.AgentMerging_plan.experiments.task_arithmetic.ta_scale_sweep_c100.model-LiveCodeBench.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521 \
  --subset-name LiveCodeBenchCodeHurtRcrfVsTa16 \
  --top-k 16 \
  --min-point-delta 0.25 \
  --install-to-cure-data /mnt/cache/wuruixiao/users/lsc/CURE/data
```

Case pack：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/build_cure_hurt_case_pack.py \
  --manifest /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/LiveBenchCodeHurtRcrfVsTa16.manifest.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/case_pack

PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/build_cure_hurt_case_pack.py \
  --manifest /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/LiveCodeBenchCodeHurtRcrfVsTa16.manifest.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/case_pack
```

## Span-Pair Manifest

为了让 hurt subset 直接接 residual attribution，又生成了 span-pair manifest：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs
```

每个 subset 输出 6 个文件：

| 文件类型 | 用途 |
|---|---|
| `*.span_pairs.jsonl` | pair-level 元数据，连接同题 positive / negative |
| `*.positive_full.jsonl` | 参考模型成功 full generation，用于 prompt/reasoning span |
| `*.negative_full.jsonl` | 当前模型失败 full generation，用于 prompt/reasoning span |
| `*.positive_code.jsonl` | 参考模型成功 extracted code，用于 final-code span |
| `*.negative_code.jsonl` | 当前模型失败 extracted code，用于 final-code span |
| `*.contrast_code.jsonl` | 同题 pass/fail code contrast，含 `negative_response` |

设计原则：

- prompt / reasoning probe 使用 full generation，因为它要看题意理解和解题思路如何被 teacher-forced response 监督。
- final-code probe 使用 extracted code，并用 `--span response`，避免 CURE response 没有 markdown code fence 时把解释文本混入 code span。
- contrast 文件保留同 prompt 的 pass code 与 fail code，后续可以计算 `utility(pass) - utility(fail)`，避免只学 expert 风格。

已做兼容性检查：

```text
probe_signed_utility.py --plan-only
row_counts = {"code": 16}
num_target_params = 196
num_target_entries = 588
```

这说明 span-pair JSONL 可以直接进入现有 signed utility probe。

第一轮真实 contrast probe：

```text
docs/report/RCRF/20260521_code_hurt_signed_utility_contrast.md
```

核心发现：

```text
utility(pass_code) alone is weak;
utility(pass_code) - utility(fail_code) is the more useful Code signal.
```

## Second Candidate Result: `rcrf_code_spanaware_conservative_v2`

候选 checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2
```

对应 gate：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_conservative_v2/gates.json
```

这版和 `rcrf_code_contrast_v1` 的关键区别：

- 不再只使用 final-code span；
- 加入 LiveBench prompt span、LiveBench reasoning span、LiveCodeBench prompt span；
- 使用 `per-file` normalization，避免 LiveCodeBench code-span 的大幅值完全压过其它 span；
- 使用 `conservative` aggregation，多个来源符号冲突时压低干预幅度；
- 仍保持每个 expert 的平均系数基本不变，验证的是结构化 residual 重分配，而不是 global scale sweep。

快速回归命令：

```bash
PYTHON_BIN=/mnt/cache/wuruixiao/miniconda3/envs/CURE/bin/python \
MODEL=/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2 \
GPU=0 \
scripts/eval/run_cure_code_hurt_eval.sh
```

Code hurt subset 结果：

| dataset | model | pass_any | candidate pass_rate / CURE code acc | hidden test-point rate | CURE BoN `(4,4)` acc | CURE BoN accumulate |
|---|---|---:|---:|---:|---:|---:|
| LiveBench hurt16 | RCRF | 0.0000 | 0.0000 | 0.3262 | - | - |
| LiveBench hurt16 | `rcrf_code_contrast_v1` | 0.2500 | 0.1094 | 0.3105 | 0.1250 | - |
| LiveBench hurt16 | `rcrf_code_spanaware_conservative_v2` | 0.4375 | 0.1563 | 0.3398 | 0.1875 | 0.3672 |
| LiveBench hurt16 | TA0.75 | 0.6250 | 0.2344 | 0.4785 | - | - |
| LiveCodeBench hurt16 | RCRF | 0.0000 | 0.0000 | 0.0553 | - | - |
| LiveCodeBench hurt16 | `rcrf_code_contrast_v1` | 0.7500 | 0.2500 | 0.3164 | 0.5000 | - |
| LiveCodeBench hurt16 | `rcrf_code_spanaware_conservative_v2` | 0.8125 | 0.4375 | 0.5189 | 0.7500 | 0.7899 |
| LiveCodeBench hurt16 | TA0.75 | 0.8750 | 0.4844 | 0.5664 | - | - |

读法：

- v2 在 LiveBench hurt16 上从 v1 的 `4/16 pass_any` 提升到 `7/16`，说明 prompt/reasoning span 对 near-miss 修复确实有增量。
- v2 在 LiveCodeBench hurt16 上从 v1 的 `12/16 pass_any` 提升到 `13/16`，且 hidden test-point rate 从 `0.3164` 提升到 `0.5189`，接近 TA0.75 的 `0.5664`。
- 这仍不是完整 Code 主评测结论；它证明的是：在已知 RCRF 受伤样本上，span-aware pass/fail residual contrast 能产生可复现的局部修复。
- 下一步不能直接宣称 Code SOTA，应先测 Tool/Memory side effect，再选择是否进入完整 CURE。

### Tool / Memory Side Effect

已用相同 full-eval wrapper 跑 Tool BFCL 四类和 Memory HotpotQA eval_50/eval_100：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/side_effect_eval/rcrf_code_spanaware_conservative_v2/quick_tool_memory
```

| metric | RCRF v1 | span-aware v2 | delta |
|---|---:|---:|---:|
| Tool BFCL mean | 0.7956 | 0.7931 | -0.0025 |
| Tool live mean | 0.7188 | 0.7188 | 0.0000 |
| Tool parallel | 0.8800 | 0.8800 | 0.0000 |
| Tool parallel_multiple | 0.8650 | 0.8550 | -0.0100 |
| Tool live_parallel | 0.8125 | 0.8125 | 0.0000 |
| Tool live_parallel_multiple | 0.6250 | 0.6250 | 0.0000 |
| Memory eval_50 F1 | 0.7708 | 0.7650 | -0.0058 |
| Memory eval_100 F1 | 0.7567 | 0.7478 | -0.0089 |

判断：

- Tool 基本没有被破坏，说明 Code span-aware contrast 没有明显伤害 tool-call 行为 span。
- Memory 有轻微下降，尤其 eval_100 更明显；这与 source-conflict 诊断一致：v2 为修复 Code 对一批 memory residual 做了正负重分配，长上下文稳定性会更敏感。
- 因此 v2 是一个有效机制验证，不是最终三任务最优候选。下一步应做 Memory-preserve 版本：对 Memory 高 utility 的 attention/MLP residual 设置更强 floor，或把 Memory span 纳入同样的 pass/fail/source-preserve contrast。

## Third Candidate Result: `rcrf_code_spanaware_memory_preserve_v3`

为验证 Memory 副作用是否来自“Code contrast 压低 Memory residual”，新增一个最小保护版本：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_memory_preserve_v3/gates.json
/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_memory_preserve_v3
```

规则：

```text
沿用 v2 的 span-aware conservative contrast；
但对 memory expert 开启 protect-negative floor：
如果 Code contrast 要压低 memory residual，则保持原 RCRF 系数不变；
正向 memory residual 仍允许上调；
memory expert 不参与 mean recenter，避免保护 floor 被 recenter 抵消。
```

结构变化：

| expert | mean | changed | positive | negative | max abs delta |
|---|---:|---:|---:|---:|---:|
| code | 0.9007 | 196 | 122 | 74 | 0.0315 |
| memory | 0.9895 | 25 | 25 | 0 | 0.0313 |
| tool | 1.0042 | 171 | 83 | 88 | 0.0418 |

其中 memory 有 `130` 个本来会被压低的位置被标记为 `protected_negative_overlay`。

Code hurt subset 结果：

| dataset | v2 pass_any | v3 pass_any | v2 candidate pass_rate | v3 candidate pass_rate | v2 test-point | v3 test-point | v2 BoN acc | v3 BoN acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LiveBench hurt16 | 0.4375 | 0.4375 | 0.1563 | 0.1719 | 0.3398 | 0.4121 | 0.1875 | 0.3750 |
| LiveCodeBench hurt16 | 0.8125 | 0.6250 | 0.4375 | 0.2813 | 0.5189 | 0.3424 | 0.7500 | 0.3750 |

结论：

- 硬保护 Memory 负向 residual 能提升 LiveBench near-miss 的 test-point 和 BoN，但会明显破坏 LiveCodeBench 的修复。
- 这说明 Memory residual 里有两类成分：一类确实是 Memory 能力保护项，另一类在 LiveCodeBench final-code/prompt 上是需要压制的冲突项。
- 因此下一版不能用 expert-level hard floor；必须做 span/source-level preserve，例如只保护 Memory 自己的高 utility residual，而不是保护所有 Memory expert residual 不被 Code contrast 压低。

生成命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/build_cure_hurt_span_pair_manifest.py \
  --manifest /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/LiveBenchCodeHurtRcrfVsTa16.manifest.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs

PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/build_cure_hurt_span_pair_manifest.py \
  --manifest /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/LiveCodeBenchCodeHurtRcrfVsTa16.manifest.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/span_pairs
```
