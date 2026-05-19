# 2026-05-18 P1 Eval-Targeted Formal Eval6

## Scope

本表记录 72h ExpertGym P1 eval-targeted 训练候选的正式 Eval6 结果。候选队列见：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260518_p1_evaltarget_candidates.md
```

训练配置与监控报告见：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260518_eg72_p1_evaltarget_fast.md
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/expertgym_72h/20260518_p1_evaltarget_fast.md
```

## Summary

| model | type | Tool mean | Tool live mean | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| main-global-i7 | common+residual, init 1/3, iter007 gate -> iter008 baked | 0.7860 | 0.6771 | 0.5645 | 0.6896 | 0.3370 | 0.4687 | 0.3978 | done |
| main-global-i3 | common+residual, init 1/3, iter003 gate -> iter004 baked | 0.7927 | 0.6979 | 0.5195 | 0.6518 | 0.3414 | 0.4604 | 0.3978 | done |
| main-gc-i7 | global-coefficient, init 1/3, iter007 gate -> iter008 baked | 0.7915 | 0.6979 | 0.5352 | 0.6662 | 0.3423 | 0.4709 | 0.3959 | done |
| main-gc-i5 | global-coefficient, init 1/3, iter005 gate -> iter006 baked | 0.7927 | 0.6979 | 0.5547 | 0.6824 | 0.3250 | 0.4653 | 0.3959 | done |
| opd-gc-i4 | OPD-only global-coefficient, init 1/3, iter004 gate -> iter005 baked | 0.7940 | 0.6979 | 0.5430 | 0.6778 | 0.3323 | 0.4705 | 0.4017 | done |
| init1-gc-i3 | init=1 upper-init global-coefficient, iter003 gate -> iter004 baked | 0.7813 | 0.6875 | 0.6426 | 0.7664 | 0.3377 | 0.4587 | 0.3978 | done |

## Tool / BFCL

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | mean | live mean | valid run |
|---|---:|---:|---:|---:|---:|---:|---|
| main-global-i7 | 0.9150 | 0.8750 | 0.6875 | 0.6667 | 0.7860 | 0.6771 | `expertgym_p1_main_global_i7_eval6_tool_20260518` |
| main-global-i3 | 0.9150 | 0.8600 | 0.6875 | 0.7083 | 0.7927 | 0.6979 | `expertgym_p1_main_global_i3_eval6_tool_20260518` |
| main-gc-i7 | 0.9100 | 0.8600 | 0.6875 | 0.7083 | 0.7915 | 0.6979 | `expertgym_p1_main_gc_i7_eval6_tool_20260518` |
| main-gc-i5 | 0.9100 | 0.8650 | 0.6875 | 0.7083 | 0.7927 | 0.6979 | `expertgym_p1_main_gc_i5_eval6_tool_20260518` |
| opd-gc-i4 | 0.9150 | 0.8650 | 0.6875 | 0.7083 | 0.7940 | 0.6979 | `expertgym_p1_opd_gc_i4_eval6_tool_20260518` |
| init1-gc-i3 | 0.8850 | 0.8650 | 0.7500 | 0.6250 | 0.7813 | 0.6875 | `expertgym_p1_init1_gc_i3_eval6_tool_20260518` |

## Memory / HotpotQA

| model | eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 | status |
|---|---:|---:|---:|---:|---:|---:|---|
| main-global-i7 | 0.6681 | 0.6898 | 0.6775 | 0.7233 | 0.5645 | 0.6896 | done |
| main-global-i3 | 0.6703 | 0.6173 | 0.6285 | 0.6909 | 0.5195 | 0.6518 | done |
| main-gc-i7 | 0.6789 | 0.6848 | 0.6400 | 0.6611 | 0.5352 | 0.6662 | done |
| main-gc-i5 | 0.6992 | 0.6567 | 0.6420 | 0.7315 | 0.5547 | 0.6824 | done |
| opd-gc-i4 | 0.7178 | 0.6214 | 0.6764 | 0.6957 | 0.5430 | 0.6778 | done |
| init1-gc-i3 | 0.7593 | 0.7519 | 0.7934 | 0.7611 | 0.6426 | 0.7664 | done |

## Code / CURE

| model | run id | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| main-global-i7 | `expertgym_p1_main_global_i7_eval6_mc_20260518` | 0.3711 | 0.4865 | 0.4375 | 0.3028 | 0.4508 | 0.3581 | 0.3370 | 0.4687 | 0.3978 | done |
| main-global-i3 | `expertgym_p1_main_global_i3_codeonly_20260518` | 0.3848 | 0.4831 | 0.4609 | 0.2979 | 0.4377 | 0.3346 | 0.3414 | 0.4604 | 0.3978 | done |
| main-gc-i7 | `expertgym_p1_main_gc_i7_codeonly_20260518` | 0.3789 | 0.4819 | 0.4375 | 0.3058 | 0.4599 | 0.3542 | 0.3423 | 0.4709 | 0.3959 | done |
| main-gc-i5 | `expertgym_p1_main_gc_i5_codeonly_20260518` | 0.3457 | 0.4711 | 0.4297 | 0.3043 | 0.4595 | 0.3620 | 0.3250 | 0.4653 | 0.3959 | done |
| opd-gc-i4 | `expertgym_p1_opd_gc_i4_codeonly_20260518` | 0.3613 | 0.4873 | 0.4453 | 0.3033 | 0.4538 | 0.3581 | 0.3323 | 0.4705 | 0.4017 | done |
| init1-gc-i3 | `expertgym_p1_init1_gc_i3_codeonly_20260518` | 0.3672 | 0.4807 | 0.4297 | 0.3082 | 0.4368 | 0.3659 | 0.3377 | 0.4587 | 0.3978 | done |

## Raw Outputs

| item | path |
|---|---|
| main-global-i7 Tool summary | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i7/tool/logs/tool_bfcl.log` |
| main-global-i7 Memory/Code summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i7/memory_code` |
| main-global-i7 Memory raw artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/expertgym-p1-evaltarget-20260518-memory/expertgym-p1-main-global-i7/expertgym_p1_main_global_i7_eval6_mc_20260518` |
| main-global-i7 Code raw artifacts | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/expertgym-p1-evaltarget-20260518-code/expertgym-p1-main-global-i7/expertgym_p1_main_global_i7_eval6_mc_20260518` |
| main-global-i3 Memory raw artifacts | `/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/expertgym-p1-evaltarget-20260518-memory/expertgym-p1-main-global-i3/expertgym_p1_main_global_i3_eval6_mc_20260518` |
| main-global-i3 Code-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i3/code_only` |
| main-global-i3 Code raw artifacts | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/expertgym-p1-evaltarget-20260518-code/expertgym-p1-main-global-i3/expertgym_p1_main_global_i3_codeonly_20260518` |
| main-global-i3 Tool summary | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i3/tool/logs/tool_bfcl.log` |
| main-gc-i7 Tool summary | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i7/tool/logs/tool_bfcl.log` |
| main-gc-i7 Memory-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i7/memory_only` |
| main-gc-i7 Code-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i7/code_only` |
| main-gc-i7 Code raw artifacts | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/expertgym-p1-evaltarget-20260518-code/expertgym-p1-main-gc-i7/expertgym_p1_main_gc_i7_codeonly_20260518` |
| main-gc-i5 Tool summary | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i5/tool/logs/tool_bfcl.log` |
| main-gc-i5 Memory-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i5/memory_only` |
| main-gc-i5 Code-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i5/code_only` |
| main-gc-i5 Code raw artifacts | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/expertgym-p1-evaltarget-20260518-code/expertgym-p1-main-gc-i5/expertgym_p1_main_gc_i5_codeonly_20260518` |
| opd-gc-i4 Tool summary | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/opd_gc_i4/tool/logs/tool_bfcl.log` |
| opd-gc-i4 Memory-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/opd_gc_i4/memory_only` |
| opd-gc-i4 Code-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/opd_gc_i4/code_only` |
| opd-gc-i4 Code raw artifacts | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/expertgym-p1-evaltarget-20260518-code/expertgym-p1-opd-gc-i4/expertgym_p1_opd_gc_i4_codeonly_20260518` |
| init1-gc-i3 Tool summary | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/init1_gc_i3/tool/logs/tool_bfcl.log` |
| init1-gc-i3 Memory-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/init1_gc_i3/memory_only` |
| init1-gc-i3 Code-only summary dir | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/init1_gc_i3/code_only` |
| init1-gc-i3 Code raw artifacts | `/tmp/shared-storage/OnPolicy/eval/cure_feedback/expertgym-p1-evaltarget-20260518-code/expertgym-p1-init1-gc-i3/expertgym_p1_init1_gc_i3_codeonly_20260518` |
| main-global-i7 BFCL score root | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/expertgym-p1-main-global-i7-tool-expertgym-p1-main-global-i7-eval6-tool-20260518-expertgym_p1_main_global_i7_eval6_tool_20260518` |
| main-global-i3 BFCL score root | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/expertgym-p1-main-global-i3-tool-expertgym-p1-main-global-i3-eval6-tool-20260518-expertgym_p1_main_global_i3_eval6_tool_20260518` |
| main-gc-i7 BFCL score root | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/expertgym-p1-main-gc-i7-tool-expertgym-p1-main-gc-i7-eval6-tool-20260518-expertgym_p1_main_gc_i7_eval6_tool_20260518` |
| main-gc-i5 BFCL score root | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/expertgym-p1-main-gc-i5-tool-expertgym-p1-main-gc-i5-eval6-tool-20260518-expertgym_p1_main_gc_i5_eval6_tool_20260518` |
| opd-gc-i4 BFCL score root | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/expertgym-p1-opd-gc-i4-tool-expertgym-p1-opd-gc-i4-eval6-tool-20260518-expertgym_p1_opd_gc_i4_eval6_tool_20260518` |
| init1-gc-i3 BFCL score root | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/expertgym-p1-init1-gc-i3-tool-expertgym-p1-init1-gc-i3-eval6-tool-20260518-expertgym_p1_init1_gc_i3_eval6_tool_20260518` |

## Notes

- `main-global-i7` is the current proxy-high common+residual candidate. Its Tool score is close to static baselines but not a Tool improvement.
- `main-global-i7` has strong Memory but Code is close to TA-1/3 rather than clearly improved; this confirms the proxy-to-CURE gap remains important.
- `main-global-i3` has slightly better Tool heldout than `main-global-i7`; its proxy code was also higher, so it remains an important paired candidate.
- All priority P1 candidates in this table now have Tool, Memory, and Code completed.
- `main-global-i3` has stronger Tool and LiveBench Code than `main-global-i7`, but Memory is much lower. This is a useful early-iteration trade-off point rather than the main candidate.
- `main-gc-i5` is not competitive on Code Acc, despite decent Memory/Tool. It remains useful as a global-coefficient paired ablation rather than a balanced best model.
- `main-gc-i7` is the strongest Code row among the evaluated global-coefficient P1 candidates, but its Memory is lower than `init1-gc-i3`; `init1-gc-i3` is still best Memory but not a learned-from-1/3 main-method result.
