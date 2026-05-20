# 20260520 TRC Round4 Code-Push Config

## Goal

Round3 established that memory trajectory loss plus coefficient-level retention can produce Tool/Memory-pass candidates. Round4 keeps that core and tests whether Code improves by longer optimization, stronger code loss scale, or full-response code span.

## Shared Settings

- Base command: `skill/command/run_20260519_trc_round_train_one.sh`
- Config: `configs/gated_grpo_layer28_wide.yaml`
- Mode manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`
- Gate parameterization: `layer-band-coefficient`
- Init: `1.0`
- Optimizer: AdamW, `lr=0.02`, `grad_clip_norm=1.0`
- Loss: directional TRC hidden residual + prompt base drift + gate anchor + coefficient retention
- `BETA_BASE=0.05`
- `GAMMA_GATE=0.03`
- `GRADIENT_CHECKPOINTING=1`
- `MAX_SEQ_LENGTH=1536`
- `MAX_RESPONSE_TOKENS=512`
- `TASK_RESPONSE_SPAN_MODE="memory=response code=code-block tool=tool-call"` unless noted.
- `TRAJECTORY_TURN_LOSS_TASKS=memory`
- Evaluation gate: Tool mean >= 0.79 and Memory mean F1 >= 0.76 before Code/CURE.

## Experiments

| ID | run_id | GPU | calibration | epochs | code span/topk | retention | task loss multiplier | hypothesis |
|---|---|---|---|---:|---|---|---|---|
| R4A | `trc_r4a_late3_taskfloor50_e12_20260520` | 0,2 | `mtr_late3_toolaug_code_rf` | 12 | code-block / 384 | task expert floor=1.0,w=50 | code=1.0 memory=1.6 tool=1.2 | R3O may not have converged; longer training may increase Code while preserving Tool/Memory. |
| R4B | `trc_r4b_late3_codeboost_taskfloor50_e12_20260520` | 3,4 | `mtr_late3_toolaug_code_rf` | 12 | code-block / 384 | task expert floor=1.0,w=50 | code=1.4 memory=1.6 tool=1.2 | If Code residual signal is underweighted, a moderate code boost should raise code coefficient without changing data. |
| R4C | `trc_r4c_u4_codefull_globalfloor50_e12_20260520` | 5,6 | `mtr_uniform4_toolaug_code_rf` | 12 | response / 384 | global coefficient floor=1.0,w=50 | code=1.0 memory=1.6 tool=1.2 | R3K had best Memory and full code span; longer training tests whether full-response span is useful for Code. |
| R4D | `trc_r4d_late3_codeproj_taskfloor50_e12_20260520` | 3,4 | `mtr_late3_toolaug_code_rf` | 12 | code-block / 384 | task expert floor=1.0,w=50 | code=1.0 memory=1.6 tool=1.2 | Increase only code directional projection pressure, not code task loss scale, to test safer Code strengthening. |
| R4E | `trc_r4e_u4_codeblock384_taskfloor50_e12_20260520` | 5,6 | `mtr_uniform4_toolaug_code_rf` | 12 | code-block / 384 | task expert floor=1.0,w=50 | code=1.0 memory=1.6 tool=1.2 | Isolate whether R3L Memory drop came from global floor rather than code-block384 itself. |
| R4F | `trc_r4f_late3_memboost_taskfloor50_e12_20260520` | 4,6 | `mtr_late3_toolaug_code_rf` | 12 | code-block / 384 | task expert floor=1.0,w=50 | code=1.0 memory=2.0 tool=1.2 | Try to lift Memory beyond R4A without touching code loss scale. |

## Stop / Promote Rule

- Do not select by desired gate value; select by `loss-plateau`.
- If Tool collapses in quick BFCL eval, delete the baked checkpoint and keep only logs.
- If Tool/Memory pass, queue Code eval after existing R3D/R3J Code jobs finish.
