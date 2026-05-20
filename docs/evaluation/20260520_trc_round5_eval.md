# 20260520 TRC Round5 Evaluation

## Purpose

Round5 tests whether calibration distribution, not task loss scaling, is the main Code bottleneck. All candidates use the v2 96-row SOTA/recovery TRC bank and are screened by Tool/Memory before Code.

## Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R5A | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r5a_v2_late3_codeproj_taskfloor50_e12_20260520-selected` | v2 calibration + code-block384 + code projection floor=0.95 | 0.8035 | 0.8125 | 0.6667 | 0.8800 | 0.8550 | 0.7638 | 0.7614 | 0.7726 | 0.7702 | 0.7511 | promoted to Code |
| R5B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r5b_v2_late3_codeproj105_taskfloor50_e12_20260520-selected` | R5A + code projection floor=1.05 | 0.7931 | 0.8125 | 0.6250 | 0.8850 | 0.8500 | 0.7677 | 0.7818 | 0.7480 | 0.7990 | 0.7420 | promoted to Code |
| R5C | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r5c_v2_late3_coderesponse_taskfloor50_e12_20260520-selected` | v2 calibration + code=response topk256 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7690 | 0.7732 | 0.7651 | 0.7686 | 0.7691 | promoted to Code |
| R5D | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r5d_v2_late3_coderesponse384_taskfloor50_e12_20260520-selected` | v2 calibration + code=response topk384 | 0.7931 | 0.8125 | 0.6250 | 0.8800 | 0.8550 | 0.7634 | 0.7712 | 0.7420 | 0.7726 | 0.7678 | promoted to Code |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R5A | `code_20260520_0528` | 0.3672 | 0.4844 | 0.2715 | 0.3777 | 0.3194 | 0.4310 | done |
| R5B | `code_20260520_0556` | 0.3672 | 0.4609 | 0.2725 | 0.3581 | 0.3198 | 0.4095 | done |
| R5C | `code_20260520_0610` | 0.3711 | 0.4453 | 0.2803 | 0.3659 | 0.3257 | 0.4056 | done |
| R5D | `code_20260520_0640` | 0.3477 | 0.4219 | 0.2608 | 0.3620 | 0.3042 | 0.3920 | done |

## Gate Summary

| ID | selected epoch | code gate | memory gate | tool gate |
|---|---:|---:|---:|---:|
| R5A | 12 | 1.2393 | 0.9952 | 1.2039 |
| R5B | 12 | 1.2394 | 0.9950 | 1.1971 |
| R5C | 12 | 1.2397 | 0.9936 | 1.2174 |
| R5D | 12 | 1.2396 | 0.9935 | 1.2140 |

## Current Takeaways

- V2 calibration does not immediately break Tool: R5A Tool mean is `0.8035`, comparable to R4D `0.8048`.
- All R5 variants converged to similar code gate scale around `1.24`; the key comparison will be Code eval, not gate magnitude.
- Response-span variants R5C/R5D produce higher tool gate than R5A/R5B, but this must pass Tool/Memory first.
- R5A improves Code BoN mean (`0.4310`) over R3D (`0.4154`) but not primary mean (`0.3194` vs `0.3289`). The v2 calibration increases candidate solution diversity/upper bound more than deterministic correctness.
- R5B's stronger code projection floor does not improve Code; it slightly hurts Tool and BoN relative to R5A. Code projection `1.05` is not the main path.
- R5C's response span gives the best Round5 primary Code so far (`0.3257`) and best Memory (`0.7690`), but still does not beat R3D primary (`0.3289`). It suggests response/algorithm span helps slightly, but without execution-aware selection the gain is not enough.
- R5D shows that widening response topk to 384 is harmful for Code: mean acc drops to `0.3042`. The useful response signal is not “more tokens”; it needs cleaner algorithm/repair spans.
