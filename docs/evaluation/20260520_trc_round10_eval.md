# 20260520 TRC Round10 Evaluation

## Purpose

Round10 evaluates whether CodeP0-v3 algorithm tag quotas improve Code transfer while preserving Tool and Memory. The main comparison is response topK128 vs response topK256 vs code-block384, all under the same non-leak 96-row TRC calibration bank.

## Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R10A | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r10a_codep0_tag_response128_e12_20260520-selected` | tag-quota, response topK128 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7625 | 0.7861 | 0.7847 | 0.7357 | 0.7436 | promote to Code; long-context Memory weak |
| R10B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r10b_codep0_tag_response256_e12_20260520-selected` | tag-quota, response topK256 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7572 | 0.7637 | 0.7354 | 0.7716 | 0.7581 | reject by Memory |
| R10C | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r10c_codep0_tag_codeblock384_e12_20260520-selected` | tag-quota, code-block topK384 | 0.7788 | 0.7500 | 0.6250 | 0.8850 | 0.8550 | 0.7570 | 0.7703 | 0.7378 | 0.7738 | 0.7460 | reject by Tool+Memory |
| R10D | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r10d_tag_response256_mem18_e12_20260520-selected` | tag-quota, response topK256, memory multiplier 1.8 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7679 | 0.7622 | 0.7477 | 0.7823 | 0.7794 | promote to Code |
| R10E | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r10e_tag_response256_tool15_e12_20260520-selected` | tag-quota, response topK256, tool multiplier 1.5 | 0.7788 | 0.7500 | 0.6250 | 0.8850 | 0.8550 | 0.7737 | 0.7814 | 0.7901 | 0.7642 | 0.7590 | reject by Tool |
| R11A | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11a_r10b_tag_response256_e08_20260520-selected` | R10B epoch 8 early-stop bake | 0.7931 | 0.8125 | 0.6250 | 0.8800 | 0.8550 | 0.7335 | 0.7622 | 0.7337 | 0.7387 | 0.6995 | reject by Memory |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R10A | `code_20260520_0955` | 0.3477 | 0.4453 | 0.2715 | 0.3581 | 0.3096 | 0.4017 | done; weak Code, not a lead candidate |
| R10B | skipped | skipped | skipped | skipped | skipped | skipped | skipped | rejected by Memory |
| R10C | skipped | skipped | skipped | skipped | skipped | skipped | skipped | rejected by Tool+Memory |
| R10D | `code_20260520_1022` | 0.3672 | 0.4453 | 0.2652 | 0.3601 | 0.3162 | 0.4027 | done; best Round10 Code Acc |
| R10E | skipped | skipped | skipped | skipped | skipped | skipped | skipped | rejected by Tool quick gate |
| R11A | skipped | skipped | skipped | skipped | skipped | skipped | skipped | rejected by Memory quick gate |

## Live Takeaways

- Tag-quota fixes the immediate topK128 Tool regression seen in R9A/R9B: R10A Tool returns to `0.7944`.
- R10A Memory passes only barely and weakens on long-context splits (`qa_32768=0.7357`, `qa_65536=0.7436`), so it is not yet a robust all-task model.
- R10B was the more important main candidate by design, but its Memory quick gate failed (`0.7572`), mainly from `eval_100=0.7354`. This motivates R10D, which keeps the same Code setting but raises Memory loss multiplier to `1.8`.
- R10D validates that diagnosis: increasing Memory loss multiplier to `1.8` restores Memory to `0.7679` while preserving Tool `0.7944`; it is now the main tag-quota response256 Code candidate.
- R10E shows that simply raising Tool loss multiplier to `1.5` is not a safe Tool fix: Memory improves to `0.7737`, but Tool drops to `0.7788` because live_parallel falls to `0.75`.
- R10C shows tag-quota + code-block384 is not automatically safer: Tool drops to `0.7788` and Memory is `0.7570`. The strong R8D LiveBench result likely comes from RF-only data / branch-specific early stopping rather than code-block span alone.
- R11A shows early stopping at R10B epoch 8 is not enough: Tool passes (`0.7931`) but Memory collapses to `0.7335`.
- R10A final Code is weak: mean Acc `0.3096`, below R8A-e08/R8D and below the stronger Round3/Round5 references.
- R10D is the best Round10 Code candidate (`mean Acc=0.3162`, `mean BoN=0.4027`) and also passes Tool/Memory. It improves over R10A but still trails R8A-e08 on mean Acc and R8D on BoN/LiveBench.
