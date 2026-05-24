# Paper-Main Eval6 Aggregate

Root: `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523`
Run id: `iclr_main_eval6_20260523`

This file is generated from existing evaluation artifacts only. Missing cells mean the corresponding full Eval6 leg has not been run or the expected log is absent.

| candidate | role | status | Tool leg | Memory leg | Code leg | Tool | Tool live | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN | Avg(T/M/C) | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bcrc_v18_alias_v9 | main method: soft behavior-constrained residual field | ready | ready | ready | ready | 0.7931 | 0.7188 | 0.6270 | 0.7570 | 0.3301 | 0.4373 | 0.3939 | 0.6267 | 0.3301 |
| no_behavior_v1_code_only | ablation: no behavior constraint | ready | ready | ready | ready | 0.7956 | 0.7188 | 0.6387 | 0.7650 | 0.3260 | 0.4355 | 0.4076 | 0.6289 | 0.3260 |
| hard_behavior_v8 | ablation: hard behavior constraint | ready | ready | ready | ready | 0.7919 | 0.7188 | 0.6289 | 0.7568 | 0.3274 | 0.4381 | 0.4047 | 0.6254 | 0.3274 |

## Missing / Partial Items
All selected candidates have Tool, Memory, and Code logs.

## Detail Pointers

| candidate | Tool log | Memory log | Code log |
| --- | --- | --- | --- |
| bcrc_v18_alias_v9 | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/bcrc_v18_alias_v9/iclr_main_eval6_20260523/tool_memory/logs/tool_bfcl.log | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/bcrc_v18_alias_v9/iclr_main_eval6_20260523/tool_memory/logs/memory_hotpotqa.log | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/bcrc_v18_alias_v9/iclr_main_eval6_20260523/code/logs/code_cure.log |
| no_behavior_v1_code_only | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/no_behavior_v1_code_only/iclr_main_eval6_20260523/tool_memory/logs/tool_bfcl.log | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/no_behavior_v1_code_only/iclr_main_eval6_20260523/tool_memory/logs/memory_hotpotqa.log | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/no_behavior_v1_code_only/iclr_main_eval6_20260523/code/logs/code_cure.log |
| hard_behavior_v8 | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/hard_behavior_v8/iclr_main_eval6_20260523/tool_memory/logs/tool_bfcl.log | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/hard_behavior_v8/iclr_main_eval6_20260523/tool_memory/logs/memory_hotpotqa.log | /tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/hard_behavior_v8/iclr_main_eval6_20260523/code/logs/code_cure.log |

## Paper Use Rule

Use this table as the final RCF-BC full Eval6 block only when the minimum queue is `ready`: `bcrc_v18_alias_v9`, `no_behavior_v1_code_only`, and `hard_behavior_v8`.
