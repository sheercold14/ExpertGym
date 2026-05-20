# 2026-05-20 TRC Round17 Prompt Span Controls

## Goal

Test whether prompt tokens should be protected from base drift or actively aligned to expert task-vector residuals.

Hypothesis:

- Prompt hidden states encode task-specific understanding: tool schema parsing, code constraints, memory observation/update semantics.
- Pulling prompt hidden states toward base may erase useful expert understanding.
- Expert prompt residual alignment may help if applied conservatively to the prompt tail, but whole-prompt alignment may overfit calibration distribution.

## Shared Setup

- Base method: TRC layer-band coefficient training.
- Init: `INIT_VALUE=1.0`.
- Calibration:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl`
- Rows: 96 = Tool32 + Memory32 + Code22 contrast + Code10 positive.
- Code contrast: enabled for Code only.
- Decision rule: gate telemetry is not used for promotion. Run Tool/Memory quick eval first; run Code only if Tool mean >= 0.79 and Memory F1 >= 0.76.

## Variants

| Variant | Run ID | Change | Purpose |
| --- | --- | --- | --- |
| R17A | `trc_r17a_no_prompt_drift_contrast22_w15_e12_20260520` | `BETA_BASE=0`, `PROMPT_RESIDUAL_WEIGHT=0` | Remove prompt-to-base pull; test whether R16 base drift was suppressing task understanding. |
| R17B | `trc_r17b_prompt_residual_contrast22_w15_e12_20260520` | `BETA_BASE=0`, `PROMPT_RESIDUAL_WEIGHT=0.15`, `PROMPT_RESIDUAL_TOKENS=256` | Treat prompt-tail tokens like output span: align merged-base residual to expert-base residual. |

## R17A Command

```bash
tmux new-session -d -s trc_r17a_train_20260520 '
set -euo pipefail
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export EXP_ID=trc_r17a_no_prompt_drift_contrast22_w15_e12_20260520
export GPU_LIST=0,3
export CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl
export INIT_VALUE=1.0
export EPOCHS=12
export LR=0.02
export ACCUMULATION_STEPS=96
export BETA_BASE=0.0
export PROMPT_RESIDUAL_WEIGHT=0.0
export PROMPT_RESIDUAL_TOKENS=256
export GAMMA_GATE=0.005
export COEFFICIENT_FLOOR=0.0
export COEFFICIENT_FLOOR_WEIGHT=0.0
export TASK_EXPERT_COEFFICIENT_FLOOR=0.0
export TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=0.0
export HIDDEN_LAYERS=8,16,24,28
export TASK_HIDDEN_LAYERS="code=4,8,12,16,20,24,28"
export TASK_RESPONSE_SPAN_MODE="memory=response code=code-block tool=tool-call"
export TASK_TOPK_TOKENS="memory=296 code=384 tool=96"
export TASK_RESIDUAL_WEIGHT_POWER="code=0.5 memory=0.5 tool=0.5"
export TASK_DIRECTIONAL_PROJECTION_FLOOR="code=0.8 memory=0.8 tool=0.8"
export TASK_DIRECTIONAL_PROJECTION_WEIGHT="code=0.1 memory=0.1 tool=0.1"
export TRAJECTORY_TURN_LOSS_TASKS=memory
export CONTRASTIVE_NEGATIVE_LOSS_WEIGHT=1.5
export CONTRASTIVE_NEGATIVE_MARGIN=0.05
export CONTRASTIVE_NEGATIVE_TASKS=code
export GRADIENT_CHECKPOINTING=0
mkdir -p /tmp/shared-storage/OnPolicy/runs/trc/${EXP_ID}
bash skill/command/run_20260519_trc_round_train_one.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/trc/${EXP_ID}/tmux.log
'
```

## R17B Command

```bash
tmux new-session -d -s trc_r17b_train_20260520 '
set -euo pipefail
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export EXP_ID=trc_r17b_prompt_residual_contrast22_w15_e12_20260520
export GPU_LIST=0,3
export CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl
export INIT_VALUE=1.0
export EPOCHS=12
export LR=0.02
export ACCUMULATION_STEPS=96
export BETA_BASE=0.0
export PROMPT_RESIDUAL_WEIGHT=0.15
export PROMPT_RESIDUAL_TOKENS=256
export GAMMA_GATE=0.005
export COEFFICIENT_FLOOR=0.0
export COEFFICIENT_FLOOR_WEIGHT=0.0
export TASK_EXPERT_COEFFICIENT_FLOOR=0.0
export TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=0.0
export HIDDEN_LAYERS=8,16,24,28
export TASK_HIDDEN_LAYERS="code=4,8,12,16,20,24,28"
export TASK_RESPONSE_SPAN_MODE="memory=response code=code-block tool=tool-call"
export TASK_TOPK_TOKENS="memory=296 code=384 tool=96"
export TASK_RESIDUAL_WEIGHT_POWER="code=0.5 memory=0.5 tool=0.5"
export TASK_DIRECTIONAL_PROJECTION_FLOOR="code=0.8 memory=0.8 tool=0.8"
export TASK_DIRECTIONAL_PROJECTION_WEIGHT="code=0.1 memory=0.1 tool=0.1"
export TRAJECTORY_TURN_LOSS_TASKS=memory
export CONTRASTIVE_NEGATIVE_LOSS_WEIGHT=1.5
export CONTRASTIVE_NEGATIVE_MARGIN=0.05
export CONTRASTIVE_NEGATIVE_TASKS=code
export GRADIENT_CHECKPOINTING=0
mkdir -p /tmp/shared-storage/OnPolicy/runs/trc/${EXP_ID}
bash skill/command/run_20260519_trc_round_train_one.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/trc/${EXP_ID}/tmux.log
'
```

## Expected Readout

- R17A improves over R16 if prompt-base drift was suppressing useful expert task understanding.
- R17B improves over R17A only if prompt-tail expert residual contains transferable input-understanding signal.
- R17B failure modes:
  - Tool drops: prompt alignment overfits non-live tool schemas.
  - Memory drops: prompt residual conflicts with long trajectory update semantics.
  - Code BoN rises but Acc stays low: prompt understanding helps candidate quality but not deterministic answer selection.
