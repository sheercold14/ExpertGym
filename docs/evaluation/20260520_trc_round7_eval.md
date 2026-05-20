# 20260520 TRC Round7 Evaluation

## Purpose

Round7 follows R5C, the best Round5 primary Code candidate. It tests whether either longer response-span optimization or a stronger Code projection floor improves the final model without losing Tool/Memory.

## Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R7B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r7b_v2_response_proj105_e12_20260520-selected` | R5C + Code projection floor 1.05 | 0.8048 | 0.8125 | 0.6667 | 0.8900 | 0.8500 | 0.7821 | 0.7898 | 0.7721 | 0.7721 | 0.7943 | promoted to Code |
| R7A | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r7a_v2_response_e16_20260520-selected` | R5C extended to 16 epochs | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7553 | 0.7776 | 0.7209 | 0.7640 | 0.7589 | rejected: Memory below threshold |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R7B | `code_20260520_0820` | 0.3379 | 0.4141 | 0.2642 | 0.3483 | 0.3010 | 0.3812 | done; rejected for Code |
| R7A | skipped | skipped | skipped | skipped | skipped | skipped | skipped | rejected by Memory |

## Current Takeaways

- R7B is the strongest Tool/Memory candidate in the R5/R7 branch so far: Tool mean `0.8048`, Memory mean F1 `0.7821`.
- R7A's longer optimization did not improve Tool over R5C/R7B and dropped Memory below threshold. Extending response-span optimization to 16 epochs is not a safe route.
- R7B is a clean negative example: even with the strongest Round7 Tool/Memory quick metrics, Code mean is only `0.3010`. Stronger Tool/Memory gates and higher Code projection floor do not imply Code transfer; Code still depends on calibration coverage and span choice.
