# 20260520 TRC Round12 Evaluation

## Scope

Round12 starts RF-only tag-quota follow-ups after Round10/Round11. R12D is the
first selected checkpoint to enter Tool/Memory quick gate.

## Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R12D | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520-selected` | RF-only tag-quota, code-block topK384, Memory x1.6 | 0.7788 | 0.7500 | 0.6250 | 0.8850 | 0.8550 | 0.7590 | 0.7779 | 0.7329 | 0.7465 | 0.7787 | reject by Tool+Memory; do not run Code |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R12D | skipped | skipped | skipped | skipped | skipped | skipped | skipped | no Code launch: Tool mean `0.7788` and Memory mean F1 `0.7590` both miss quick gates |

## Current Takeaways

- R12D's RF-only tag-quota code-block384 branch does not preserve Tool: the
  quick-gate mean is `0.7788`, with live_parallel down to `0.7500`.
- Memory also misses the quick gate at `0.7590`, mainly from `eval_100=0.7329`
  and `qa_32768=0.7465`.
- This is a clean reject before Code; do not launch CURE Code for R12D under the
  current thresholds.
