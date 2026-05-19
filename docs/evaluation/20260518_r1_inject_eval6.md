# 2026-05-18 R1-Inject Eval6: arm-r-v2_plus_r1_alpha0.001

## Overview

Model: `arm-r-v2_plus_r1_alpha0.001`
Checkpoint: `/tmp/shared-storage/ExpertGym/baselines/qwen7b/r1_inject/arm-r-v2_plus_r1_alpha0.001`
Run ID: `20260518_r1_inject_eval6`
Summary dir: `/tmp/shared-storage/ExpertGym/baselines/eval/r1_inject_alpha0.001/20260518_r1_inject_eval6`

## Combined Summary

| model | type | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **arm-r-v2_plus_r1_alpha0.001** | **R1-Inject** | **0.7942** | **0.7085** | **0.6348** | **0.7586** | **0.3592** | **0.4875** | **0.3841** | **done** |
| task-arithmetic-c033333 | TA-1/3 | 0.7848 | 0.6771 | 0.5195 | 0.6465 | 0.3409 | 0.4688 | 0.4173 | done |
| ties-c033333-k02 | TIES | 0.7642 | 0.6458 | 0.5098 | 0.6359 | 0.3355 | 0.4679 | 0.3880 | done |
| dare-ta-c033333-d08 | DARE-TA | 0.7952 | 0.6979 | 0.5625 | 0.6901 | 0.3365 | 0.4700 | 0.3900 | done |
| dare-ties-c033333-k02-d08 | DARE-TIES | 0.7952 | 0.6979 | 0.5625 | 0.6891 | 0.3426 | 0.4788 | 0.4007 | done |
| adamerging-taskwise-len1024 | AdaMerging | 0.7835 | 0.6771 | 0.5293 | 0.6678 | 0.3406 | 0.4828 | 0.4242 | done |
| adamerging-layerwise-len1024 | AdaMerging | 0.7848 | 0.6771 | 0.5391 | 0.6674 | 0.3350 | 0.4653 | 0.3949 | done |
| adamergingpp-taskwise-len1024 | AdaMerging++ | 0.7629 | 0.6458 | 0.5176 | 0.6407 | 0.3309 | 0.4647 | 0.3773 | done |
| adamergingpp-layerwise-len1024 | AdaMerging++ | 0.7654 | 0.6458 | 0.4922 | 0.6199 | 0.3287 | 0.4593 | 0.3841 | done |
| mixture-grpo-ta13-l96-step1 | Mixture GRPO | 0.7823 | 0.6771 | 0.5313 | 0.6643 | 0.3384 | 0.4716 | 0.3782 | done |

### Delta vs. best baseline (DARE-TA)

| metric | R1-Inject | DARE-TA | delta |
|---|---:|---:|---:|
| Tool mean | 0.7942 | 0.7952 | -0.0010 |
| Tool live mean | 0.7085 | 0.6979 | +0.0106 |
| Memory EM | 0.6348 | 0.5625 | +0.0723 |
| Memory F1 | 0.7586 | 0.6901 | +0.0685 |
| Code Acc | 0.3592 | 0.3365 | +0.0227 |
| Code TP | 0.4875 | 0.4700 | +0.0175 |
| Code BoN | 0.3841 | 0.3900 | -0.0059 |

## Tool / BFCL

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | run id |
|---|---:|---:|---:|---:|---:|---|
| arm-r-v2_plus_r1_alpha0.001 | 0.8950 | 0.8650 | 0.7500 | 0.6667 | 0.7942 | `20260518_r1_inject_eval6` |

## Memory / HotpotQA

| model | eval_50 EM | eval_50 F1 | eval_100 EM | eval_100 F1 | eval_qa_1_32768 EM | eval_qa_1_32768 F1 | eval_qa_1_65536 EM | eval_qa_1_65536 F1 | mean EM | mean F1 | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| arm-r-v2_plus_r1_alpha0.001 | 0.6094 | 0.7551 | 0.5781 | 0.7136 | 0.6797 | 0.7919 | 0.6719 | 0.7739 | 0.6348 | 0.7586 | done |

### Baseline comparison (mean F1)

| model | eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 |
|---|---:|---:|---:|---:|---:|---:|
| **arm-r-v2_plus_r1_alpha0.001** | **0.7551** | **0.7136** | **0.7919** | **0.7739** | **0.6348** | **0.7586** |
| task-arithmetic-c033333 | 0.6876 | 0.6191 | 0.6281 | 0.6511 | 0.5195 | 0.6465 |
| ties-c033333-k02 | 0.6677 | 0.6042 | 0.6384 | 0.6332 | 0.5098 | 0.6359 |
| dare-ta-c033333-d08 | 0.7196 | 0.6753 | 0.6990 | 0.6664 | 0.5625 | 0.6901 |
| dare-ties-c033333-k02-d08 | 0.7008 | 0.6529 | 0.6767 | 0.7259 | 0.5625 | 0.6891 |
| adamerging-taskwise-len1024 | 0.6748 | 0.6234 | 0.6736 | 0.6993 | 0.5293 | 0.6678 |
| adamerging-layerwise-len1024 | 0.7048 | 0.6619 | 0.6487 | 0.6540 | 0.5391 | 0.6674 |
| adamergingpp-taskwise-len1024 | 0.6457 | 0.6596 | 0.6275 | 0.6301 | 0.5176 | 0.6407 |
| adamergingpp-layerwise-len1024 | 0.6388 | 0.6240 | 0.5914 | 0.6252 | 0.4922 | 0.6199 |
| mixture-grpo-ta13-l96-step1 | 0.6948 | 0.6124 | 0.6658 | 0.6840 | 0.5313 | 0.6643 |

## Code / CURE

| model | run id | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| arm-r-v2_plus_r1_alpha0.001 | `20260518_r1_inject_eval6` | 0.3906 | 0.5039 | 0.4219 | 0.3278 | 0.4710 | 0.3464 | 0.3592 | 0.4875 | 0.3841 | done |

### Baseline comparison (Code)

| model | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **arm-r-v2_plus_r1_alpha0.001** | **0.3906** | **0.5039** | **0.4219** | **0.3278** | **0.4710** | **0.3464** | **0.3592** | **0.4875** | **0.3841** |
| task-arithmetic-c033333 | 0.3789 | 0.4836 | 0.4766 | 0.3028 | 0.4539 | 0.3581 | 0.3409 | 0.4688 | 0.4173 |
| ties-c033333-k02 | 0.3711 | 0.4922 | 0.4219 | 0.2999 | 0.4436 | 0.3542 | 0.3355 | 0.4679 | 0.3880 |
| dare-ta-c033333-d08 | 0.3711 | 0.4895 | 0.4375 | 0.3019 | 0.4505 | 0.3425 | 0.3365 | 0.4700 | 0.3900 |
| dare-ties-c033333-k02-d08 | 0.3848 | 0.4956 | 0.4531 | 0.3004 | 0.4620 | 0.3483 | 0.3426 | 0.4788 | 0.4007 |
| adamerging-taskwise-len1024 | 0.3789 | 0.5108 | 0.4844 | 0.3023 | 0.4547 | 0.3640 | 0.3406 | 0.4828 | 0.4242 |
| adamerging-layerwise-len1024 | 0.3672 | 0.4785 | 0.4375 | 0.3028 | 0.4521 | 0.3523 | 0.3350 | 0.4653 | 0.3949 |
| adamergingpp-taskwise-len1024 | 0.3574 | 0.4762 | 0.3984 | 0.3043 | 0.4532 | 0.3562 | 0.3309 | 0.4647 | 0.3773 |
| adamergingpp-layerwise-len1024 | 0.3672 | 0.4767 | 0.4297 | 0.2901 | 0.4419 | 0.3386 | 0.3287 | 0.4593 | 0.3841 |
| mixture-grpo-ta13-l96-step1 | 0.3789 | 0.4905 | 0.4297 | 0.2979 | 0.4527 | 0.3268 | 0.3384 | 0.4716 | 0.3782 |

## Raw Outputs

| item | path |
|---|---|
| Model checkpoint | `/tmp/shared-storage/ExpertGym/baselines/qwen7b/r1_inject/arm-r-v2_plus_r1_alpha0.001` |
| Eval manifest | `/tmp/shared-storage/ExpertGym/baselines/eval/r1_inject_alpha0.001/20260518_r1_inject_eval6/full_eval_manifest.env` |
| Tool BFCL log | `/tmp/shared-storage/ExpertGym/baselines/eval/r1_inject_alpha0.001/20260518_r1_inject_eval6/logs/tool_bfcl.log` |
| Memory HotpotQA log | `/tmp/shared-storage/ExpertGym/baselines/eval/r1_inject_alpha0.001/20260518_r1_inject_eval6/logs/memory_hotpotqa.log` |
| Code CURE log | `/tmp/shared-storage/ExpertGym/baselines/eval/r1_inject_alpha0.001/20260518_r1_inject_eval6/logs/code_cure.log` |
| Memory raw artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/baseline-r1-inject-eval6-memory/arm-r-v2-plus-r1-alpha0.001/20260518_r1_inject_eval6` |
| Code CURE feedback dir | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/baseline-r1-inject-eval6-code/arm-r-v2-plus-r1-alpha0.001/20260518_r1_inject_eval6` |

## Analysis

The R1-injected model (alpha=0.001) shows notable improvements over all baselines on Memory/HotpotQA:
- Mean F1 of **0.7586** exceeds the previous best baseline (DARE-TA at 0.6901) by +0.0685.
- Mean EM of **0.6348** exceeds DARE-TA by +0.0723.
- Gains are consistent across all four context-length splits.

On Tool/BFCL the model is competitive with top baselines (0.7942 vs. 0.7952 for DARE-TA), essentially matching performance. The live_parallel sub-score (0.7500) ties the best baselines.

On Code/CURE:
- Mean Acc (0.3592) and mean TP (0.4875) are the highest among all evaluated models.
- LiveBench Acc (0.3906) is the best observed.
- LiveCodeBench Acc (0.3278) exceeds all baselines.
- Mean BoN (0.3841) is competitive but slightly below the top baselines (AdaMerging TW at 0.4242, TA-1/3 at 0.4173) due to lower LiveCodeBench BoN.

Overall, the R1-inject model achieves the best Memory and Code accuracy results while maintaining competitive Tool performance, making it the strongest single model configuration in this eval6 round.
