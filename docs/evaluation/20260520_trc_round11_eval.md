# 20260520 TRC Round11 Evaluation

## Scope

Round11 tracks early-stop/code-block follow-ups and the hybrid calibration bank:
R11B/R11F/R11G/R11H Code evals are launched from passing quick gates.

## Tool / Memory

| ID | checkpoint | key config | Tool mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory mean F1 | eval_50 | eval_100 | qa_32768 | qa_65536 | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R11B | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11b_r8d_codeblock_e08_20260520-selected` | R8D code-block e08 alias | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7715 | 0.7755 | 0.7285 | 0.8139 | 0.7683 | promoted to Code |
| R11C | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8d_codep0_rf_codeblock_e10_20260520-selected` | R8D code-block e10 canonical bake | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7567 | 0.7709 | 0.7444 | 0.7683 | 0.7433 | reject by Memory |
| R11F | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11f_tag_response320_e12_20260520-selected` | tag-quota response topK320 | 0.8048 | pending detail | pending detail | pending detail | pending detail | 0.7726 | 0.7780 | 0.7524 | 0.8030 | 0.7571 | promoted to Code |
| R11G | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11g_hybrid_response256_mem18_e12_20260520-selected` | hybrid stable Tool/Memory + response topK256 + Memory x1.8 | 0.7944 | 0.8125 | 0.6250 | 0.8850 | 0.8550 | 0.7600 | 0.7616 | 0.7646 | 0.7559 | 0.7576 | promoted to Code; exactly on Memory gate |
| R11H | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11h_hybrid_codeblock384_mem18_e12_20260520-selected` | hybrid stable Tool/Memory + code-block topK384 + Memory x1.8 | 0.8048 | 0.8125 | 0.6667 | 0.8850 | 0.8550 | 0.7619 | 0.7746 | 0.7575 | 0.7725 | 0.7429 | promoted to Code |

## Code / CURE

| ID | run id | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | mean Acc | mean BoN | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R10A | `code_20260520_0955` | 0.3477 | 0.4453 | 0.2715 | 0.3581 | 0.3096 | 0.4017 | done; recorded in Round10 |
| R10D | `code_20260520_1022` | 0.3672 | 0.4453 | 0.2652 | 0.3601 | 0.3162 | 0.4027 | done; recorded in Round10 |
| R11B | `code_20260520_1039` | 0.3750 | 0.5000 | 0.2794 | 0.3620 | 0.3272 | 0.4310 | done |
| R11F | `code_20260520_1057` | 0.3555 | 0.4688 | 0.2657 | 0.3444 | 0.3106 | 0.4066 | done |
| R11G | `code_20260520_1110` | 0.3516 | 0.4297 | 0.2784 | 0.3542 | 0.3150 | 0.3919 | done |
| R11H | `code_20260520_1123` | 0.3477 | 0.4375 | 0.2652 | 0.3640 | 0.3064 | 0.4007 | done |

## Training / Bake

| ID | run id | selected epoch | code gate | memory gate | tool gate | status |
|---|---|---:|---:|---:|---:|---|
| R11G | `trc_r11g_hybrid_response256_mem18_e12_20260520` | 12 | 1.2405 | 0.9951 | 1.2276 | baked; quick gate running |
| R11H | `trc_r11h_hybrid_codeblock384_mem18_e12_20260520` | 12 | 1.2401 | 0.9962 | 1.2063 | baked at 2026-05-20 10:56 CST |
| R12B | `trc_r12b_tag_codeblock384_mem18_e12_20260520` | pending | 1.1600 | 0.9991 | 1.1434 | training epoch 8 observed at 2026-05-20 11:19 CST |
| R12D | `trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520` | 12 | 1.2403 | 0.9945 | 1.2110 | baked; quick gate tracked in Round12 |

## Live Monitor Snapshot

- 2026-05-20 11:40 CST tmux sessions present: `eval_r11b_code_20260520`,
  `eval_r11f_code_20260520`, `eval_r11g_code_20260520`,
  `eval_r11h_code_20260520`, `eval_r12d_tm_20260520`; `eval_r10d_code_20260520`
  has exited after producing final Code metrics.
- GPU occupancy at 2026-05-20 11:17 CST: GPU0 `42761/81559 MiB`, GPU1
  `51431/81559 MiB`, GPU2 `51063/81559 MiB`, GPU3 `67223/81559 MiB`, GPU4
  `67749/81559 MiB`, GPU6 `76031/81559 MiB`, GPU7 `55759/81559 MiB`; GPU5
  reports `2/81559 MiB`.
- R11H Tool/Memory pass is confirmed (`0.8048` / `0.7619`), but Code finished
  at mean Acc `0.3064`; code-block topK384 does not improve formal Code despite
  better Tool quick-gate.
- R12B is still training and has no selected/baked checkpoint in this monitor
  snapshot.
- R12D quick gate is complete in Round12: Tool mean `0.7788`, Memory mean F1
  `0.7590`; reject by Tool+Memory and do not launch Code.
