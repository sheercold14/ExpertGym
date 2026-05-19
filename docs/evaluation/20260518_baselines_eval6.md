# 2026-05-18 Baseline Eval6

## Scope

本表记录论文 baseline 复现与评测进展。WUDI / ExpertMerging 本轮不重跑，只记录已有模型目录；本轮实际构建并评测的是 OP-VEC 静态合并 baseline 与 Qwen AdaMerging。

复现配置见：

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260518_baseline_reproduction.md`

## Summary

### Reference Models (Individual / Pre-merge)

| model | type | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | Base | 0.7500 | 0.6250 | 0.3906 | 0.5288 | 0.2800 | 0.4096 | 0.3304 | done |
| Qwen2.5-7B-Instruct-ToolRL-grpo-cold | Expert (Tool) | 0.7890 | 0.6979 | 0.4238 | 0.5549 | 0.2822 | 0.4126 | 0.3265 | done |
| RL-MemoryAgent-7B | Expert (Memory) | 0.7760 | 0.6771 | 0.6289 | 0.7578 | 0.3167 | 0.4650 | 0.3450 | done |
| ReasonFlux-Coder-7B | Expert (Code) | 0.7500 | 0.6250 | 0.4102 | 0.5573 | 0.3208 | 0.4551 | 0.4222 | done |
| DeepSeek-R1-Distill-Qwen-7B | R1 | 0.3660 | 0.3021 | 0.0156 | 0.0355 | 0.4018 | 0.4359 | 0.4840 | done |

### Merged Models

| model | type | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| RAM-Merged ARM-R-v2 | TA-0.75 | 0.7942 | 0.7083 | 0.6152 | 0.7361 | 0.3441 | 0.4803 | 0.3812 | done |
| wudi-qwen7b-3expert | WUDI | 0.7823 | 0.6771 | 0.5410 | 0.6591 | 0.3304 | 0.4657 | 0.4095 | done |
| task-arithmetic-c033333 | TA-1/3 | 0.7848 | 0.6771 | 0.5195 | 0.6465 | 0.3409 | 0.4688 | 0.4173 | done |
| ties-c033333-k02 | TIES | 0.7642 | 0.6458 | 0.5098 | 0.6359 | 0.3355 | 0.4679 | 0.3880 | done |
| dare-ta-c033333-d08 | DARE-TA | 0.7952 | 0.6979 | 0.5625 | 0.6901 | 0.3365 | 0.4700 | 0.3900 | done |
| dare-ties-c033333-k02-d08 | DARE-TIES | 0.7952 | 0.6979 | 0.5625 | 0.6891 | 0.3426 | 0.4788 | 0.4007 | done |
| adamerging-taskwise-len1024 | AdaMerging | 0.7835 | 0.6771 | 0.5293 | 0.6678 | 0.3406 | 0.4828 | 0.4242 | done |
| adamerging-layerwise-len1024 | AdaMerging | 0.7848 | 0.6771 | 0.5391 | 0.6674 | 0.3350 | 0.4653 | 0.3949 | done |
| adamergingpp-taskwise-len1024 | AdaMerging++ | 0.7629 | 0.6458 | 0.5176 | 0.6407 | 0.3309 | 0.4647 | 0.3773 | done |
| adamergingpp-layerwise-len1024 | AdaMerging++ | 0.7654 | 0.6458 | 0.4922 | 0.6199 | 0.3287 | 0.4593 | 0.3841 | done |
| mixture-grpo-ta13-l96-step1 | Mixture GRPO | 0.7823 | 0.6771 | 0.5313 | 0.6643 | 0.3384 | 0.4716 | 0.3782 | done |

## Tool / BFCL

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | valid run |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 0.9100 | 0.8400 | 0.6250 | 0.6250 | 0.7500 | `eval6-20260502` |
| Qwen2.5-7B-Instruct-ToolRL-grpo-cold | 0.9050 | 0.8550 | 0.6875 | 0.7083 | 0.7890 | `eval6-20260502` |
| RL-MemoryAgent-7B | 0.9000 | 0.8500 | 0.6875 | 0.6667 | 0.7760 | `eval6-20260502` |
| ReasonFlux-Coder-7B | 0.9150 | 0.8350 | 0.6250 | 0.6250 | 0.7500 | `eval6-20260502` |
| DeepSeek-R1-Distill-Qwen-7B | 0.5200 | 0.3400 | 0.4375 | 0.1667 | 0.3660 | `eval6-20260502` |
| RAM-Merged ARM-R-v2 | 0.9050 | 0.8550 | 0.7500 | 0.6667 | 0.7942 | `eval6-20260502` |
| wudi-qwen7b-3expert | 0.9150 | 0.8600 | 0.6875 | 0.6667 | 0.7823 | `eval6-20260502` |
| task-arithmetic-c033333 | 0.9150 | 0.8700 | 0.6875 | 0.6667 | 0.7848 | `20260517_p0_ta13_eval6` |
| ties-c033333-k02 | 0.9150 | 0.8500 | 0.6250 | 0.6667 | 0.7642 | `20260518_baseline_ties_eval6_rerun01` |
| dare-ta-c033333-d08 | 0.9150 | 0.8700 | 0.6875 | 0.7083 | 0.7952 | `20260518_baseline_dare_ta_eval6_tool` |
| dare-ties-c033333-k02-d08 | 0.9150 | 0.8700 | 0.6875 | 0.7083 | 0.7952 | `20260518_baseline_dare_ties_eval6` |
| adamerging-taskwise-len1024 | 0.9150 | 0.8650 | 0.6875 | 0.6667 | 0.7835 | `20260518_baseline_adamerging_tw_len1024_eval6_tool` |
| adamerging-layerwise-len1024 | 0.9150 | 0.8700 | 0.6875 | 0.6667 | 0.7848 | `20260518_baseline_adamerging_lw_len1024_eval6` |
| adamergingpp-taskwise-len1024 | 0.9150 | 0.8450 | 0.6250 | 0.6667 | 0.7629 | `20260518_baseline_adamergingpp_tw_len1024_eval6` |
| adamergingpp-layerwise-len1024 | 0.9150 | 0.8550 | 0.6250 | 0.6667 | 0.7654 | `20260518_baseline_adamergingpp_lw_len1024_eval6` |
| mixture-grpo-ta13-l96-step1 | 0.9150 | 0.8600 | 0.6875 | 0.6667 | 0.7823 | `mixture_grpo_l96_eval6_20260518` |

## Memory / HotpotQA

| model | eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 | status |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 0.5720 | 0.4893 | 0.5460 | 0.5078 | 0.3906 | 0.5288 | done |
| Qwen2.5-7B-Instruct-ToolRL-grpo-cold | 0.5737 | 0.5695 | 0.5142 | 0.5623 | 0.4238 | 0.5549 | done |
| RL-MemoryAgent-7B | 0.7683 | 0.7634 | 0.7663 | 0.7331 | 0.6289 | 0.7578 | done |
| ReasonFlux-Coder-7B | 0.6176 | 0.5063 | 0.5376 | 0.5677 | 0.4102 | 0.5573 | done |
| DeepSeek-R1-Distill-Qwen-7B | 0.0484 | 0.0496 | 0.0167 | 0.0274 | 0.0156 | 0.0355 | done |
| RAM-Merged ARM-R-v2 | 0.7262 | 0.7230 | 0.7593 | 0.7361 | 0.6152 | 0.7361 | done |
| wudi-qwen7b-3expert | 0.6859 | 0.6271 | 0.6710 | 0.6524 | 0.5410 | 0.6591 | done |
| task-arithmetic-c033333 | 0.6876 | 0.6191 | 0.6281 | 0.6511 | 0.5195 | 0.6465 | from P0 table |
| ties-c033333-k02 | 0.6677 | 0.6042 | 0.6384 | 0.6332 | 0.5098 | 0.6359 | done |
| dare-ta-c033333-d08 | 0.7196 | 0.6753 | 0.6990 | 0.6664 | 0.5625 | 0.6901 | done |
| dare-ties-c033333-k02-d08 | 0.7008 | 0.6529 | 0.6767 | 0.7259 | 0.5625 | 0.6891 | done |
| adamerging-taskwise-len1024 | 0.6748 | 0.6234 | 0.6736 | 0.6993 | 0.5293 | 0.6678 | done |
| adamerging-layerwise-len1024 | 0.7048 | 0.6619 | 0.6487 | 0.6540 | 0.5391 | 0.6674 | done |
| adamergingpp-taskwise-len1024 | 0.6457 | 0.6596 | 0.6275 | 0.6301 | 0.5176 | 0.6407 | done |
| adamergingpp-layerwise-len1024 | 0.6388 | 0.6240 | 0.5914 | 0.6252 | 0.4922 | 0.6199 | done |
| mixture-grpo-ta13-l96-step1 | 0.6948 | 0.6124 | 0.6658 | 0.6840 | 0.5313 | 0.6643 | done |

## Code / CURE

| model | run id | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | `eval6-20260502` | 0.2969 | 0.4153 | 0.3594 | 0.2632 | 0.4040 | 0.3014 | 0.2800 | 0.4096 | 0.3304 | done |
| Qwen2.5-7B-Instruct-ToolRL-grpo-cold | `eval6-20260502` | 0.3223 | 0.4346 | 0.3594 | 0.2422 | 0.3906 | 0.2935 | 0.2822 | 0.4126 | 0.3265 | done |
| RL-MemoryAgent-7B | `eval6-20260502` | 0.3418 | 0.4841 | 0.3594 | 0.2916 | 0.4459 | 0.3307 | 0.3167 | 0.4650 | 0.3450 | done |
| ReasonFlux-Coder-7B | `eval6-20260502` | 0.3457 | 0.4709 | 0.4609 | 0.2960 | 0.4393 | 0.3836 | 0.3208 | 0.4551 | 0.4222 | done |
| DeepSeek-R1-Distill-Qwen-7B | `eval6-20260502` | 0.3785 | 0.4156 | 0.4579 | 0.4250 | 0.4561 | 0.5100 | 0.4018 | 0.4359 | 0.4840 | done |
| RAM-Merged ARM-R-v2 | `eval6-20260502` | 0.3691 | 0.4961 | 0.3984 | 0.3190 | 0.4645 | 0.3640 | 0.3441 | 0.4803 | 0.3812 | done |
| wudi-qwen7b-3expert | `eval6-20260502` | 0.3594 | 0.4809 | 0.4531 | 0.3014 | 0.4505 | 0.3659 | 0.3304 | 0.4657 | 0.4095 | done |
| task-arithmetic-c033333 | `20260517_p0_ta13_code_rerun01` | 0.3789 | 0.4836 | 0.4766 | 0.3028 | 0.4539 | 0.3581 | 0.3409 | 0.4688 | 0.4173 | from P0 table |
| ties-c033333-k02 | `20260518_baseline_ties_eval6_rerun01` | 0.3711 | 0.4922 | 0.4219 | 0.2999 | 0.4436 | 0.3542 | 0.3355 | 0.4679 | 0.3880 | done |
| dare-ta-c033333-d08 | `20260518_baseline_dare_ta_eval6_mc` | 0.3711 | 0.4895 | 0.4375 | 0.3019 | 0.4505 | 0.3425 | 0.3365 | 0.4700 | 0.3900 | done |
| dare-ties-c033333-k02-d08 | `20260518_baseline_dare_ties_eval6` | 0.3848 | 0.4956 | 0.4531 | 0.3004 | 0.4620 | 0.3483 | 0.3426 | 0.4788 | 0.4007 | done |
| adamerging-taskwise-len1024 | `20260518_baseline_adamerging_tw_len1024_eval6_mc` | 0.3789 | 0.5108 | 0.4844 | 0.3023 | 0.4547 | 0.3640 | 0.3406 | 0.4828 | 0.4242 | done |
| adamerging-layerwise-len1024 | `20260518_baseline_adamerging_lw_len1024_eval6` | 0.3672 | 0.4785 | 0.4375 | 0.3028 | 0.4521 | 0.3523 | 0.3350 | 0.4653 | 0.3949 | done |
| adamergingpp-taskwise-len1024 | `20260518_baseline_adamergingpp_tw_len1024_eval6` | 0.3574 | 0.4762 | 0.3984 | 0.3043 | 0.4532 | 0.3562 | 0.3309 | 0.4647 | 0.3773 | done |
| adamergingpp-layerwise-len1024 | `20260518_baseline_adamergingpp_lw_len1024_eval6` | 0.3672 | 0.4767 | 0.4297 | 0.2901 | 0.4419 | 0.3386 | 0.3287 | 0.4593 | 0.3841 | done |
| mixture-grpo-ta13-l96-step1 | `mixture_grpo_l96_eval6_20260518` | 0.3789 | 0.4905 | 0.4297 | 0.2979 | 0.4527 | 0.3268 | 0.3384 | 0.4716 | 0.3782 | done |

## Raw Outputs

| item | path |
|---|---|
| Static checkpoints | `/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/` |
| AdaMerging checkpoint | `/tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/task_wise_adamerging/qwen_task_wise_adamerging_len1024_20260518/model` |
| TIES eval | `/tmp/shared-storage/ExpertGym/baselines/eval/ties_c033333_k02_d08_seed20260518/20260518_baseline_ties_eval6_rerun01` |
| DARE-TA Tool | `/tmp/shared-storage/ExpertGym/baselines/eval/dare_ta_c033333_k02_d08_seed20260518/20260518_baseline_dare_ta_eval6_tool` |
| DARE-TA Memory/Code | `/tmp/shared-storage/ExpertGym/baselines/eval/dare_ta_c033333_k02_d08_seed20260518/20260518_baseline_dare_ta_eval6_mc` |
| DARE-TIES eval | `/tmp/shared-storage/ExpertGym/baselines/eval/dare_ties_c033333_k02_d08_seed20260518/20260518_baseline_dare_ties_eval6` |
| AdaMerging Tool | `/tmp/shared-storage/ExpertGym/baselines/eval/adamerging_tw_len1024_20260518/20260518_baseline_adamerging_tw_len1024_eval6_tool` |
| AdaMerging Memory/Code | `/tmp/shared-storage/ExpertGym/baselines/eval/adamerging_tw_len1024_20260518/20260518_baseline_adamerging_tw_len1024_eval6_mc` |
| Mixture GRPO HF checkpoint | `/tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518/hf_merged_global_step_1` |
| Mixture GRPO Eval6 | `/tmp/shared-storage/ExpertGym/baselines/eval/mixture_grpo_ta13_evaltarget_l96_n1_step1/mixture_grpo_l96_eval6_20260518` |
| Mixture GRPO Memory raw artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/opvec-gated-grpo-full-eval-memory/mixture-grpo-ta13-l96-step1/mixture_grpo_l96_eval6_20260518` |
| Mixture GRPO Code raw artifacts | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/mixture-grpo-ta13-l96-step1/mixture_grpo_l96_eval6_20260518` |
| AdaMerging layer-wise Eval6 | `/tmp/shared-storage/ExpertGym/baselines/eval/adamerging_lw_len1024_20260518/20260518_baseline_adamerging_lw_len1024_eval6` |
| AdaMerging layer-wise Code raw artifacts | `/tmp/shared-storage/ExpertGym/baselines/eval/cure_feedback/baseline-adamerging-lw-len1024-eval6-code/adamerging-layerwise-len1024/20260518_baseline_adamerging_lw_len1024_eval6` |
| AdaMerging++ task-wise Eval6 | `/tmp/shared-storage/ExpertGym/baselines/eval/adamergingpp_tw_len1024_20260518/20260518_baseline_adamergingpp_tw_len1024_eval6` |
| AdaMerging++ task-wise Code raw artifacts | `/tmp/shared-storage/ExpertGym/baselines/eval/cure_feedback/baseline-adamergingpp-tw-len1024-eval6-code/adamergingpp-taskwise-len1024/20260518_baseline_adamergingpp_tw_len1024_eval6` |
| AdaMerging++ layer-wise Eval6 | `/tmp/shared-storage/ExpertGym/baselines/eval/adamergingpp_lw_len1024_20260518/20260518_baseline_adamergingpp_lw_len1024_eval6` |
| AdaMerging++ layer-wise Code raw artifacts | `/tmp/shared-storage/ExpertGym/baselines/eval/cure_feedback/baseline-adamergingpp-lw-len1024-eval6-code/adamergingpp-layerwise-len1024/20260518_baseline_adamergingpp_lw_len1024_eval6` |

## Notes

- BFCL Tool harness uses shared `model_config.py` and shared `.env`; multiple Tool models must not be evaluated concurrently. Invalid concurrent attempts for DARE-TA / AdaMerging are audit logs only and are excluded from this table.
- AdaMerging `MAX_LENGTH=2048` OOM on one 80G GPU; reported checkpoint is the same task-wise AdaMerging recipe with `MAX_LENGTH=1024`.
- Fisher baseline remains lower priority: a faithful diagonal Fisher merge needs per-expert backward statistics and should not be mixed into the current GPU-heavy eval batch without a separate run plan.
- Mixture GRPO is a one-step full-parameter VeRL baseline from TA-1/3 with the same OP-VEC RewardRouter. It is not a gated task-vector method and does not use OPD or retention.
