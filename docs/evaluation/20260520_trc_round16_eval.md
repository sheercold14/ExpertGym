# 20260520 TRC Round16 Evaluation

## Scope

Round16 是 Code BoN-to-Acc 主线的第一版非泄漏实验。与 Round15 相比，核心变化是移除 formal Code eval anchors，改用 CodeP0-v3 `CodeContests_train` same-prompt pass/fail contrast。

## Calibration

Path:

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl
```

Composition:

| task | rows | source |
|---|---:|---|
| Tool | 32 | R5A stable Tool bank |
| Memory | 32 | R5A stable Memory late3 trajectory bank |
| Code | 22 | CodeP0-v3 train pass/fail contrast |
| Code | 10 | CodeP0-v3 train positive fill |

Code rows are all `CodeContests_train`; no LiveBench, LiveCodeBench, or formal CURE eval prompt/output/test is used.

## Training

| ID | run id | checkpoint | contrast weight | selected epoch | code gate | memory gate | tool gate | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| R16A | `trc_r16a_nonleak_contrast22_w15_e12_20260520` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r16a_nonleak_contrast22_w15_e12_20260520-selected` | 1.5 | 12 | 1.2399 | 0.9945 | 1.1887 | quick gate failed on Memory |
| R16B | `trc_r16b_nonleak_contrast22_w30_e12_20260520` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r16b_nonleak_contrast22_w30_e12_20260520-selected` | 3.0 | 12 | 1.2403 | 0.9945 | 1.1809 | Code done; hold diagnostic |
| R16C | `trc_r16c_rfonly_contrast16_w15_e12_20260520` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r16c_rfonly_contrast16_w15_e12_20260520-selected` | 1.5 | 12 | 1.2399 | 0.9946 | 1.2006 | Code done; reject primary |
| R16D | `trc_r16d_nonleak_contrast22_response256_w15_e12_20260520` | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r16d_nonleak_contrast22_response256_w15_e12_20260520-selected` | 1.5 | 12 | 1.2398 | 0.9937 | 1.2195 | Tool/Memory passed; Code candidate |

## Quick Gate

Run Tool/Memory before expensive Code formal eval.

Decision rule: gate coefficients are diagnostics only. Promotion/rejection is decided by evaluation metrics, not by whether code/tool/memory gates rise or fall.

| ID | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| R16A | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7506 | reject: Memory below 0.76 |
| R16B | 0.7944 | 0.8125 | 0.6250 | 0.8800 | 0.8600 | 0.7724 | pass; Code done |
| R16C | 0.7931 | 0.8125 | 0.6250 | 0.8800 | 0.8550 | 0.7632 | pass; Code done |
| R16D | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7604 | pass; Code candidate |

Promotion thresholds:

- BFCL Tool quick mean `>= 0.79`
- Memory mean F1 `>= 0.76`

## Code / CURE

Run only if quick gate passes.

| ID | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---:|---:|---:|---:|---:|---:|---|
| R16A | n/a | n/a | n/a | n/a | n/a | n/a | skipped: Memory quick gate failed |
| R16B | 0.3770 | 0.4375 | 0.2808 | 0.3620 | 0.3289 | 0.3998 | done; keep diagnostic, not primary promote |
| R16C | 0.3477 | 0.4219 | 0.2754 | 0.3562 | 0.3115 | 0.3890 | reject: below mainline Code anchors |
| R16D | 0.3477 | 0.4297 | 0.2568 | 0.3366 | 0.3023 | 0.3831 | done; reject primary |

## Live Notes

- Epoch 1-2 show dense Code contrast signal: Code task contrast active rate is about `0.53`, much stronger than Round15B's late active rate around `0.06`.
- Code gate increases from `1.00` to about `1.04` by epoch 2 in both R16A/R16B.
- Gate coefficients are logged only to interpret dynamics. They should not directly decide stopping or promotion.
- Continue training unless the run crashes or produces invalid artifacts; after baking, run Tool/Memory quick eval and decide from metrics.
- R16A/B completed and baked. R16A quick Tool/Memory eval launched first because BFCL Tool harness should not be run concurrently.
- R16A Tool quick passed with mean `0.7944`, but Memory mean F1 is only `0.7506`
  (`0.7647/0.7299/0.7708/0.7369`), below the `0.76` threshold. Do not run expensive Code eval for R16A.
- R16B quick Tool/Memory launched after the R16A BFCL Tool stage finished; BFCL Tool stages are not overlapped.
- R16B Memory mean F1 is `0.7724` (`0.7636/0.7637/0.7963/0.7658`), so R16B passes quick gate. CURE Code eval launched on GPUs 6,7.
- R16B Code completed. LiveBench is `Acc=0.3770, BoN=0.4375`; LiveCodeBench is `Acc=0.2808, BoN=0.3620`; mean is `Acc=0.3289, BoN=0.3998`.
  This marginally matches the single-Acc target but loses too much BoN versus the stronger R5A/R11B-style anchors, so keep the checkpoint as a diagnostic for high contrast weight rather than promoting it as the primary Code mainline.
- R16C/R16D completed training. A launcher continuation issue caused them to miss automatic select/bake, so selection+bake were run manually from their `epoch_012.gates.json`; no training rows were changed.
- R16C quick Tool passed with mean `0.7931`; however non-live remains below the strongest historical Tool runs (`parallel=0.8800`, `parallel_multiple=0.8550`). Memory mean F1 is `0.7632`
  (`0.7633/0.7608/0.7936/0.7353`), just above the `0.76` threshold. Code eval launched on GPUs 4,5 as `eval_r16c_code_20260520`.
- R16C Code completed. LiveBench is `Acc=0.3477, BoN=0.4219`; LiveCodeBench is `Acc=0.2754, BoN=0.3562`; mean is `Acc=0.3115, BoN=0.3890`.
  RF-only Code rows are not enough here: Tool/Memory pass, but Code is below both R16B and the stronger historical Code anchors, so reject as a primary candidate and retain only as a purity ablation.
- R16C/D use the freed GPUs for two orthogonal controls:
  - R16C: ReasonFlux-only Code rows to test expert-vector purity.
  - R16D: Code `response` span topK256 to test whether reasoning span helps Code Acc.
- R16D Tool quick passed with mean `0.7944` (`0.8125/0.6250/0.8850/0.8550`); Memory mean F1 is `0.7604`
  (`0.7555/0.7263/0.7978/0.7619`), barely above threshold. It is a Code candidate, but less robust than R16B/R16C on Memory and should be compared against R16A/B as the response-span control.
- R16D Code full CURE eval launched at 2026-05-20 23:00 CST as `r16d_code_20260520`, using checkpoint
  `/tmp/shared-storage/OnPolicy/checkpoints/trc_r16d_nonleak_contrast22_response256_w15_e12_20260520-selected` and GPU group `[[4,5]]`.
- R16D LiveBench completed at 2026-05-20 23:14 CST: `Acc=0.3477`, `BoN=0.4297`. This is below R16B on LiveBench; wait for LiveCodeBench before final mean.
- R16D full Code completed at 2026-05-21 00:14 CST. LiveCodeBench is `Acc=0.2568`, `BoN=0.3366`; mean Code is `Acc=0.3023`, `BoN=0.3831`.
  Response-span topK256 does not improve formal Code transfer in this setting; reject R16D as primary and keep it only as a response-span control.
