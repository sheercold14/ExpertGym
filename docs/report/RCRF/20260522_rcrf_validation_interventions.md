# 2026-05-22 RCF-BC Validation Interventions

## 目的

这个文件把 validation cards 物化为可 bake 的最小 OP-VEC gate 干预。它不启动评测；每个候选只改一张卡片中的代表 residual rows，用于后续反事实验证。

## 生成候选

| candidate | card | operation | rows | changed | mean delta before | mean delta after |
|---|---|---|---:|---:|---:|---:|
| `card01_high-retain-continuous-field-code-source-conflict-code-early-00-09-attention__drop-delta` | `high-retain-continuous-field-code-source-conflict-code-early-00-09-attention` | `drop-delta` | 6 | 6 | -0.010415 | 0.000000 |
| `card02_high-retain-continuous-field-code-source-conflict-code-early-00-09-mlp__drop-delta` | `high-retain-continuous-field-code-source-conflict-code-early-00-09-mlp` | `drop-delta` | 7 | 7 | -0.009366 | 0.000000 |
| `card03_high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-middle-10-19-mlp__half-delta` | `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-middle-10-19-mlp` | `half-delta` | 1 | 1 | 0.008427 | 0.004213 |
| `card04_high-retain-continuous-field-code-source-conflict-memory-early-00-09-attention__drop-delta` | `high-retain-continuous-field-code-source-conflict-memory-early-00-09-attention` | `drop-delta` | 5 | 5 | -0.006882 | 0.000000 |
| `card05_high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-late-20-27-attention__half-delta` | `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-late-20-27-attention` | `half-delta` | 6 | 6 | 0.006979 | 0.003490 |
| `card06_high-retain-continuous-field-code-source-conflict-memory-early-00-09-mlp__drop-delta` | `high-retain-continuous-field-code-source-conflict-memory-early-00-09-mlp` | `drop-delta` | 4 | 4 | -0.008617 | 0.000000 |

## 后续评测命令模板

对任一候选，先 bake，再跑 Tool/Memory quick；只有行为不过 guardrail 时再跑 Code hurt：

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
CFG=configs/gated_grpo.yaml
GATE=/path/to/validation_card_intervention/gates.json
OUT=/tmp/shared-storage/OnPolicy/checkpoints/<candidate_id>
$PY scripts/eval/opvec_bake_checkpoint.py --config $CFG --mode-manifest $MODE --gate-checkpoint $GATE --output $OUT
```

## 产物

- manifest: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/validation_card_interventions_20260522/validation_interventions_manifest.json`
- output root: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/validation_card_interventions_20260522`
