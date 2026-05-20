# 20260520 TRC Round8 Evaluation

## Purpose

Round8 tests whether non-leak CodeP0-v3 recoverable trajectories can improve Code while preserving Tool/Memory. R8A and R8B keep the stable R5C objective and only change Code trajectory source. R8C/R8E are follow-up probes for residual objective and role-balanced Code selection.

## Training Summary

| ID | checkpoint | calibration | objective | selected epoch | code gate | memory gate | tool gate | status |
|---|---|---|---|---:|---:|---:|---:|---|
| R8A | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8a_codep0_rf_response_e12_20260520-selected` | CodeP0 RF-only | directional response | 12 | 1.2406 | 0.9928 | 1.2303 | quick eval done |
| R8A-e08 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8a_codep0_rf_response_e08_20260520-selected` | CodeP0 RF-only | directional response | 8 | 1.1602 | 0.9977 | 1.1574 | promote to Code |
| R8A-e10 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8a_codep0_rf_response_e10_20260520-selected` | CodeP0 RF-only | directional response | 10 | 1.2004 | 0.9946 | 1.1947 | Memory-only done |
| R8B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8b_codep0_rfds_response_e12_20260520-selected` | CodeP0 RF + DeepSeek fallback | directional response | 12 | 1.2405 | 0.9933 | 1.2296 | promote to Code |
| R8B-e08 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8b_codep0_rfds_response_e08_20260520-selected` | CodeP0 RF + DeepSeek fallback | directional response | 8 | 1.1601 | 0.9978 | 1.1573 | baked; quick gate skipped |
| R8C | not baked | CodeP0 RF-only | relative-MSE response | 2 stopped | 1.0400 | 0.9618 | 0.9839 | rejected: objective unstable |
| R8D | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8d_codep0_rf_codeblock_e12_20260520-selected` | CodeP0 RF-only | directional code-block | 12 | 1.2403 | 0.9943 | 1.2096 | promote to Code |
| R8D-e08 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8d_codep0_rf_codeblock_e08_20260520-selected` | CodeP0 RF-only | directional code-block | 8 | 1.1601 | 0.9974 | 1.1499 | baked here; duplicate quick gate running as R11B |
| R8D-e10 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8d_codep0_rf_codeblock_e10_20260520-selected` | CodeP0 RF-only | directional code-block | 10 | 1.2002 | 0.9953 | 1.1814 | baked; quick gate skipped |
| R8E | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8e_codep0_rolequota_response_e12_20260520-selected` | CodeP0 RF role quotas | directional response | 12 | 1.2405 | 0.9929 | 1.2283 | rejected by Memory |

## Tool / Memory

| ID | run id | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R8A | `r8a_tm_20260520` | 0.8035 | 0.8125 | 0.6667 | 0.8850 | 0.8500 | 0.7563 | 0.7555 | 0.7533 | 0.7258 | 0.7905 | reject e12; test earlier e8/e10 |
| R8B | `r8b_tm_20260520` | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7687 | 0.7759 | 0.7442 | 0.7918 | 0.7631 | promote to Code |
| R8E | `r8e_tm_20260520` | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7580 | 0.7630 | 0.7517 | 0.7820 | 0.7352 | reject e12 |
| R8D | `r8d_tm_20260520` | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7668 | 0.7846 | 0.7269 | 0.7851 | 0.7709 | promote to Code |
| R8A-e08 | `r8a_e08_tool_20260520` + `r8a_e08_mem_20260520` | 0.7931 | 0.8125 | 0.6250 | 0.8850 | 0.8500 | 0.7716 | 0.7711 | 0.7609 | 0.7755 | 0.7790 | promote to Code |
| R8A-e10 | `r8a_e10_mem_20260520` | not run | not run | not run | not run | not run | 0.7602 | 0.7720 | 0.7342 | 0.7780 | 0.7566 | Memory done; below e08, do not expand |
| R8D-e08 | `r11b_tm_20260520` on duplicate `trc_r11b_r8d_codeblock_e08_20260520-selected` | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | pending | pending | pending | pending | pending | Tool pass; Memory running in another worker; no Code recommendation yet |
| R8D-e10 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | baked only; quick gate skipped due no safe free GPU pair |
| R8B-e08 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | baked only; quick gate skipped due no safe free GPU pair |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R8A | pending | pending | pending | pending | pending | pending | pending | gated out by Memory e12; early checkpoint check pending |
| R8B | `code_20260520_0907` | 0.3477 | 0.4141 | 0.2642 | 0.3542 | 0.3059 | 0.3841 | done; below R5/R4 code anchors |
| R8D | `code_20260520_0918` | 0.3730 | 0.4766 | 0.2676 | 0.3601 | 0.3203 | 0.4183 | done; strong LiveBench, weak LiveCodeBench |
| R8A-e08 | `code_20260520_0921` | 0.3594 | 0.4297 | 0.2842 | 0.3640 | 0.3218 | 0.3968 | done; current best Round8 mean Acc |
| R8D-e08 | skipped | skipped | skipped | skipped | skipped | skipped | skipped | no Code eval: Memory quick gate still pending |
| R8D-e10 | skipped | skipped | skipped | skipped | skipped | skipped | skipped | no Code eval: Tool/Memory quick gate not run |
| R8B-e08 | skipped | skipped | skipped | skipped | skipped | skipped | skipped | no Code eval: Tool/Memory quick gate not run |
| R8E | skipped | skipped | skipped | skipped | skipped | skipped | skipped | rejected by Memory |

## Current Takeaways

- R8A/B/R8E training dynamics are stable: Code/Tool rise to about `1.24/1.23`, while Memory recovers near `0.993` instead of collapsing.
- R8A e12 fails the Memory quick gate by a small margin (`0.7563` vs target `0.76`), so its earlier epochs are more likely to be useful than pushing the same objective longer.
- R8B passes the quick gate (`Tool=0.7944`, `Memory=0.7687`) and should be sent to CURE Code. Compared with R8A, the DeepSeek fallback improves Memory enough to clear the gate without changing gate means much.
- R8B's full Code result is weak (`mean Acc=0.3059`, `mean BoN=0.3841`) despite passing Tool/Memory. The RF+DeepSeek coverage calibration improves Memory gate but does not transfer to CURE Code.
- R8D has the strongest Round8 LiveBench (`0.3730` Acc, `0.4766` BoN), but LiveCodeBench remains weak (`0.2676`), so code-block384 improves one Code face without solving overall Code.
- R8A-e08 reaches the best Round8 mean Acc so far (`0.3218`) by trading lower LiveBench than R8D for better LiveCodeBench (`0.2842`). Early stopping is meaningful for Code, not just Memory.
- R8D-e08, R8D-e10, and R8B-e08 were baked on 2026-05-20 from the source `epoch_008/epoch_010.gates.json` files. This worker did not launch a new quick gate because the 10:18 resource check had no safe consecutive free GPU pair: 4/6 were reserved for R11F, 1/2/7 were occupied, and 0/5 were in the protected quick-gate pool. Another worker's duplicate R8D-e08 eval (`eval_r11b_tm_20260520`) has Tool pass (`0.7944`) and Memory still running; no Code recommendation for the duplicate until Memory passes.
- R8A-e10 Memory is only barely above threshold (`0.7602`) and below R8A-e08 (`0.7716`), so keep e08 as the early-stop representative.
- R8A and R8B gate trajectories are nearly identical. If Code eval differs, the cause is likely trajectory content/coverage rather than coefficient magnitude.
- R8A keeps expert-vector consistency; R8B improves unique Code prompt coverage. This comparison is central for deciding how to build the next non-leak Code calibration bank.
- R8C shows relative-MSE is not a safe default objective: after only two epochs, Memory gate dropped to `0.9618` and Tool stayed below `1.0`, while loss scale was about 10x larger than the directional objective. This supports keeping directional residual alignment as the main TRC objective.
- R8E preserves Tool but misses the Memory gate (`0.7580`), mainly from `qa_65536=0.7352`; keep it out of Code unless the Memory gate is intentionally relaxed.
