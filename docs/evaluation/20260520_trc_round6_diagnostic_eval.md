# 20260520 TRC Round6 Diagnostic Evaluation

## Warning

Round6 uses CURE hidden-test-passing trajectories from prior evaluated models. It is an eval-leak diagnostic and must not be reported as a paper main result.

## Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R6A | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r6a_curediag_lb_codeblock_e8_20260520-selected` | 32 LiveBench-passing code rows, code-block span | 0.7800 | 0.7500 | 0.6250 | 0.8850 | 0.8600 | skipped | skipped | skipped | skipped | skipped | rejected: Tool below threshold |
| R6B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r6b_curediag_bal_response_e8_20260520-selected` | 16 LiveBench + 16 LiveCodeBench passing rows, response span | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7604 | 0.7599 | 0.7475 | 0.7664 | 0.7677 | promoted to Code |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R6B | `code_20260520_0805_rerun` | 0.3750 | 0.4609 | 0.2671 | 0.3620 | 0.3211 | 0.4115 | done |

## Current Takeaways

- R6A shows that directly fitting LiveBench code-block trajectories can hurt Tool behavior, even with task-aware coefficient floor.
- R6B is safer: balanced CURE-aligned response trajectories keep Tool and Memory barely above threshold. Its Code result will decide whether execution-aligned calibration can solve the Code bottleneck.
- 2026-05-20 monitor update: R6B Code rerun improves LiveBench primary to `0.3750`, but LiveCodeBench is weak (`0.2671`), yielding mean Acc `0.3211` and mean BoN `0.4115`. This is useful as an eval-leak diagnostic, but it does not clearly beat the best non-leak primary Code baselines.
