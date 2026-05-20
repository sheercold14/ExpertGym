# 20260520 TRC Round4 Evaluation

## Gate

Round4 promotes candidates only if Tool mean `>= 0.79` and Memory mean F1 `>= 0.76`.

## Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R4A | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r4a_late3_taskfloor50_e12_20260520-selected` | late3 + code-block384 + task-aware floor50, 12 epochs | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7638 | 0.7644 | 0.7522 | 0.7829 | 0.7558 | promoted to Code |
| R4B | deleted | R4A + code loss multiplier 1.4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | skipped | skipped | skipped | skipped | skipped | rejected: Tool collapse |
| R4C | deleted | uniform4 + code full-response384 + global floor50, 12 epochs | 0.7892 | 0.7500 | 0.6667 | 0.8850 | 0.8550 | 0.7612 | 0.7515 | 0.7662 | 0.7406 | 0.7864 | rejected: Tool below threshold |
| R4D | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r4d_late3_codeproj_taskfloor50_e12_20260520-selected` | R4A + code projection floor=0.95,w=0.25 | 0.8048 | 0.8125 | 0.6667 | 0.8850 | 0.8550 | 0.7669 | 0.7727 | 0.7633 | 0.7650 | 0.7666 | promoted to Code |
| R4E | deleted | uniform4 + code-block384 + task-aware floor50 | 0.7788 | 0.7500 | 0.6250 | 0.8850 | 0.8550 | skipped | skipped | skipped | skipped | skipped | rejected: Tool below threshold |
| R4F | deleted | R4A + memory loss multiplier 2.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | skipped | skipped | skipped | skipped | skipped | rejected: Tool collapse |
| R4G | deleted | R4A epoch10 early-stop bake | 0.7892 | 0.7500 | 0.6667 | 0.8850 | 0.8550 | skipped | skipped | skipped | skipped | skipped | rejected: Tool below threshold |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R4A | `code_20260520_0410` | 0.3613 | 0.4609 | 0.2872 | 0.3738 | 0.3243 | 0.4174 | done |
| R4D | `code_20260520_0436` | 0.3496 | 0.4531 | 0.2657 | 0.3718 | 0.3076 | 0.4125 | done |

## Current Takeaways

- Extending R3O-style training from 8 to 12 epochs can keep Tool and Memory above threshold while pushing code/tool gates higher.
- Increasing code task loss multiplier to `1.4` is unsafe: R4B Tool collapses to zero despite similar aggregate gate means to R4A. The failure is likely behavior-format interference, not simply low tool coefficient.
- Full-response code span plus global floor (R4C) nearly preserves Tool but falls just below threshold; it is not a clean mainline.
- Increasing memory task loss multiplier to `2.0` is also unsafe under this span mix: R4F collapses Tool to zero. Direct loss-scale pushes are therefore lower priority than span design, projection/floor constraints, and eval-aligned calibration.
- R4D is the current Round4 main candidate: code projection preserved the best Tool mean (`0.8048`) and Memory F1 (`0.7669`) while still pushing code gates. Its Code result is the main pending decision point.
- R4D's Code result is weak despite the best Tool/Memory proxy: mean Code acc `0.3076`, below R3D/R4A. This reinforces that higher code gate and stronger proxy preservation do not imply better CURE performance.
