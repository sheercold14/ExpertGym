# 2026-05-20 TRC Round17 Prompt Span Evaluation

## Decision Rule

Gate coefficients are telemetry only. Promotion uses official evaluation:

- quick stage: BFCL Tool mean and HotpotQA Memory mean F1;
- expensive stage: CURE Code only if Tool mean >= 0.79 and Memory F1 >= 0.76.

## Runs

| ID | run id | prompt base drift | prompt expert residual | checkpoint | status |
|---|---|---:|---:|---|---|
| R17A | `trc_r17a_no_prompt_drift_contrast22_w15_e12_20260520` | 0.0 | 0.0 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r17a_no_prompt_drift_contrast22_w15_e12_20260520-selected` | baked; quick eval running |
| R17B | `trc_r17b_prompt_residual_contrast22_w15_e12_20260520` | 0.0 | 0.15 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r17b_prompt_residual_contrast22_w15_e12_20260520-selected` | baked; quick eval pending |

## Quick Gate

| ID | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| R17A | 0.7969 | 0.8125 | 0.6250 | 0.8950 | 0.8550 | 0.7645 | pass; Code candidate |
| R17B | 0.7969 | 0.8125 | 0.6250 | 0.8950 | 0.8550 | 0.7643 | pass; Code candidate |

## Code / CURE

| ID | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---:|---:|---:|---:|---:|---:|---|
| R17A | pending | pending | pending | pending | pending | pending | recommended for Code; not launched by this monitor |
| R17B | pending | pending | pending | pending | pending | pending | recommended for Code; not launched by this monitor |

## Notes

- R17A tests the user's concern that prompt drift-to-base is conceptually wrong because task vectors may encode task-specific prompt understanding.
- R17B tests whether prompt hidden states should be treated like output spans by aligning merged-base residuals to expert-base residuals on prompt-tail tokens.
- R17B uses low prompt residual weight because prompt hidden states are shared across tasks and can overfit calibration distributions more easily than output spans.
- R17A training completed at epoch 12 with lower hidden loss but strong Memory telemetry decline. Quick Tool/Memory eval was launched to verify whether the telemetry corresponds to real Memory degradation.
- R17A Tool quick passes by mean (`0.7969`). The main risk remains Memory: the selected checkpoint has memory gate telemetry around `0.7571`, so HotpotQA F1 is the deciding metric.
- R17A Memory quick gate passes with mean F1 `0.7645` (`0.7494/0.7549/0.7963/0.7574`). This shows removing prompt-base drift can pass official Memory despite low memory-gate telemetry, so R17A is a valid Code candidate.
- R17B reached epoch 10/12 with gate telemetry roughly `code=1.2018, memory=0.7984, tool=1.1961`. Prompt residual reduces hidden loss smoothly but still pushes Memory telemetry down, so it should not enter Code eval unless Memory passes.
- R17B completed at epoch 12 and was baked. Selected telemetry is `code=1.2429, memory=0.7573, tool=1.2319`; this is almost the same Memory-risk pattern as R17A, so quick eval is required before any Code run.
- R17B Tool quick passes with mean `0.7969` (`0.8125/0.6250/0.8950/0.8550`), essentially matching R17A. Memory mean F1 is `0.7643`
  (`0.7477/0.7770/0.7933/0.7391`), so prompt residual is a Code candidate; note that long-context `qa_65536` is weak.
