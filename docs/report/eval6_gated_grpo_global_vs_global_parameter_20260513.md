# Eval6 Gated-GRPO Final Model Report

## 实验设置

| 项目 | 设置 |
|---|---|
| Run ID | `eval6-20260502-125748` |
| Global 模型 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_i20_20260513_010649/iter_020/baked_policy` |
| Global-Parameter 模型 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_parameter_i20_20260513_010649/iter_020/baked_policy` |
| Tool harness | BFCL: `parallel`, `parallel_multiple`, `live_parallel`, `live_parallel_multiple` |
| Memory harness | HotpotQA: `eval_50`, `eval_100`, `eval_qa_1_32768`, `eval_qa_1_65536` |
| Code harness | CURE full: `LiveBench`, `LiveCodeBench`; `k_code=4`, `k_case=4`, `max_model_len=32768` |

## 汇总结果

| 模型 | Tool 平均 Acc | Memory 平均 F1 | Memory 平均 EM | Code 平均 Acc | Code BoN 平均 Acc | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Global | 0.7952 | 0.6897 | 0.5566 | 0.3536 | 0.3978 | 完整 |
| Global-Parameter | 0.7965 | 0.7029 | 0.5781 | 0.3345 | 0.3910 | 完整 |

## Tool/BFCL

| 模型 | live_parallel | live_parallel_multiple | parallel | parallel_multiple | 平均 |
|---|---:|---:|---:|---:|---:|
| Global | 0.6875 | 0.7083 | 0.9100 | 0.8750 | 0.7952 |
| Global-Parameter | 0.6875 | 0.7083 | 0.9150 | 0.8750 | 0.7965 |

## Memory/HotpotQA

| 模型 | 数据集 | F1 | EM | Sub-EM |
|---|---|---:|---:|---:|
| Global | eval_50 | 0.6588 | 0.5078 | 0.6641 |
| Global | eval_100 | 0.7173 | 0.5547 | 0.7266 |
| Global | eval_qa_1_32768 | 0.6708 | 0.5625 | 0.7031 |
| Global | eval_qa_1_65536 | 0.7119 | 0.6016 | 0.7812 |
| Global-Parameter | eval_50 | 0.6973 | 0.5469 | 0.7266 |
| Global-Parameter | eval_100 | 0.7327 | 0.5938 | 0.7344 |
| Global-Parameter | eval_qa_1_32768 | 0.6615 | 0.5547 | 0.7109 |
| Global-Parameter | eval_qa_1_65536 | 0.7201 | 0.6172 | 0.7656 |

## Code/CURE

| 模型 | 数据集 | Code Acc | Code Accumulate Acc | Unit Test Acc | Unit Test Accumulate Acc | BoN Acc | BoN Accumulate Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| Global | LiveBench | 0.4004 | 0.5103 | 0.2892 | 0.3234 | 0.4453 | 0.5798 |
| Global | LiveCodeBench | 0.3068 | 0.4578 | 0.3684 | 0.4006 | 0.3503 | 0.5191 |
| Global-Parameter | LiveBench | 0.3691 | 0.4944 | 0.3447 | 0.3823 | 0.4219 | 0.5573 |
| Global-Parameter | LiveCodeBench | 0.2999 | 0.4570 | 0.4110 | 0.4309 | 0.3601 | 0.5312 |

## 未完成项

| 项目 | 状态 | 证据 |
|---|---|---|
| 无 | 全部完成 | `Global` 与 `Global-Parameter` 均已生成 Tool、Memory、Code summary |

## 结论

Global-Parameter 在 Tool 与 Memory 上略优于 Global；Global 在 Code/CURE 上更好。两者完整对比后，Global-Parameter 的收益主要体现在 Tool/Memory，小幅牺牲 Code。

## 结果文件

| 模型 | Tool | Memory | Code |
|---|---|---|---|
| Global | `eval-batch/eval6-20260502-125748/logs/opvec_qbank_c033333_global_i20_20260513_010649.bfcl.log` | `eval6-memory-hotpotqa/opvec_qbank_c033333_global_i20_20260513_010649/eval6-20260502-125748/summary.json` | `eval6-code-cure-full/opvec_qbank_c033333_global_i20_20260513_010649/eval6-20260502-125748/summary.json` |
| Global-Parameter | `eval-batch/eval6-20260502-125748/logs/opvec_qbank_c033333_global_parameter_i20_20260513_010649.bfcl.log` | `eval6-memory-hotpotqa/opvec_qbank_c033333_global_parameter_i20_20260513_010649/eval6-20260502-125748/summary.json` | `eval6-code-cure-full/opvec_qbank_c033333_global_parameter_i20_20260513_010649/eval6-20260502-125748/summary.json` |
