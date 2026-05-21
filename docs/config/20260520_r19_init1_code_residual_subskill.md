# 20260520 R19 Init1 Code Residual Subskill Experiments

## Goal

Use the current init=1 TRC method to test whether Code ability can be improved by choosing more capability-aligned residual spans, not by forcing the Code gate to move in a predetermined direction.

Gate coefficients are diagnostics only. Promotion is decided by official Tool/Memory quick eval and CURE Code eval.

## Calibration

Path:

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round19_init1_code_steer_balanced/trc96_balanced_code_mixed.jsonl
```

Composition:

| task | rows | source |
|---|---:|---|
| Tool | 32 | Round16 stable Tool trajectories |
| Memory | 32 | Round16 stable Memory trajectory rows |
| Code | 10 | eval-like LiveBench/LiveCodeBench successful rows |
| Code | 22 | CodeP0 train successful/contrast rows |

Code expert mix: 26 Code expert rows, 6 Memory expert rows. The purpose is to let residual learning discover which expert direction carries the useful Code behavior, rather than assuming the Code task vector alone is sufficient.

## Common Training Settings

All runs start from init=1 and use layer-band-coefficient gates.

```text
CONFIG=configs/gated_grpo_layer28_wide.yaml
INIT_VALUE=1.0
EPOCHS=12
LR=0.02
HIDDEN_LAYERS=8,16,24,28
TASK_HIDDEN_LAYERS="code=4,8,12,16,20,24,28 memory=8,16,24,28 tool=8,16,24,28"
TASK_TOPK_TOKENS="memory=96 code=256 tool=96" unless noted
TASK_RESPONSE_SPAN_MODE="memory=response code=response tool=tool-call" unless noted
TRAJECTORY_TURN_LOSS_TASKS=memory
RESIDUAL_OBJECTIVE=directional
TASK_DIRECTIONAL_PROJECTION_FLOOR="code=0.95 memory=0.8 tool=0.8"
TASK_DIRECTIONAL_PROJECTION_WEIGHT="code=0.25 memory=0.1 tool=0.1"
BETA_BASE=0.05
GAMMA_GATE=0.03
TASK_LOSS_MULTIPLIER="code=1.0 memory=1.6 tool=1.2"
TASK_EXPERT_COEFFICIENT_FLOOR=1.0
TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0
CONTRASTIVE_NEGATIVE_TASKS=code where contrastive loss is enabled
```

The task-expert coefficient floor is retained in R19A/B/D/E to keep the run comparable to the strongest stable TRC family (R5/R16). R19C removes it as a pure-residual control.

## Runs

| ID | Run ID | GPUs | Code span | Prompt residual | Contrastive | Floor | Question |
|---|---|---|---|---:|---:|---:|---|
| R19A | `trc_r19a_init1_mixed_response256_w15_e12_20260520` | 0,1 | response/topK256 | 0.0 | 1.5 | on | Does reasoning+final-code span improve Code over code-block-only? |
| R19B | `trc_r19b_init1_mixed_response256_prompt015_w15_e12_20260520` | 2,3 | response/topK256 | 0.15 | 1.5 | on | Does prompt-understanding residual help convert BoN to Acc? |
| R19C | `trc_r19c_init1_mixed_response256_pureres_w15_e12_20260520` | 4,5 | response/topK256 | 0.0 | 1.5 | off | Is the gate movement coming from residual geometry or coefficient floor? |
| R19D | `trc_r19d_init1_mixed_codeblock384_w30_e12_20260520` | 6,7 | code-block/topK384 | 0.0 | 3.0 | on | Does stronger pass/fail contrast on final code beat response-span alignment? |
| R19E | `trc_r19e_init1_mixed_response256_w30_e12_20260520` | 4,5 | response/topK256 | 0.0 | 3.0 | on | Does stronger pass/fail contrast help response-span alignment? |
| R19G | `trc_r19g_init1_sgd_lr3_response256_w15_e8_20260520` | 4,5 | response/topK256 | 0.0 | 1.5 | on | Does SGD preserve useful layer-wise Code gradient magnitude? |
| R20A | `trc_r20a_init1_codeonly_sgd_lr3_response256_w15_e12_20260520` | 4,5 | response/topK256 | 0.0 | 1.5 | on | If only Code rows and Code gate slice train, can formal Code eval finally rise? |
| R20B | `trc_r20b_init1_codeonly_sgd_lr3_response256_w15_e20_20260520` | 0,1 | response/topK256 | 0.0 | 1.5 | on | Does the same clean Code-only direction keep improving past epoch 12? |
| R20C | `trc_r20c_init1_codeonly_sgd_lr5_response256_w15_e12_20260520` | 2,3 | response/topK256 | 0.0 | 1.5 | on | Can a stronger SGD step reach the useful Code region faster without overshoot? |
| R20D | `trc_r20d_init1_codeonly_sgd_lr3_response256_prompt015_w15_e12_20260520` | 6,7 | response/topK256 | 0.15 | 1.5 | on | Does prompt residual help Code understanding once other tasks are frozen? |
| R20E | `trc_r20e_init1_codeonly_sgd_lr3_codemem_w15_e12_20260520` | 0,1 | response/topK256 | 0.0 | 1.5 | on | On Code rows only, can Code+Memory gates learn a better capability mix? |
| R20F | `trc_r20f_init1_codeonly_sgd_lr3_alltrain_w15_e12_20260520` | 4,5 | response/topK256 | 0.0 | 1.5 | on | On Code rows only, can all expert gates find a better Code capability mix? |

R19C was stopped after epoch 4 because Memory gate dropped to `0.9200` while Code/Tool rose to about `1.08`. This is enough to establish that pure residual without task-floor protection damages Memory balance. R19E replaces it on GPUs 4,5 and isolates contrast strength by matching R19A except `CONTRASTIVE_NEGATIVE_LOSS_WEIGHT=3.0`.

R19G uses `OPTIMIZER=sgd`, `LR=3.0`, and `LOG_GATE_GRADIENTS=1`. It confirmed that AdamW was flattening Code gate dynamics: by epoch 8, Code gate spread reached `0.369` across layers. However, Memory also oscillated strongly, so this is a diagnostic run rather than a balanced final candidate.

R20A uses the same calibration file but filters training to `TRAIN_TASKS=code` and masks gradients with `TRAINABLE_EXPERTS=code`. Memory/Tool gate slices remain at init1; this isolates whether Code residual steering itself is learnable.

R20B/R20C/R20D are follow-ups launched before R20A eval completes to keep GPU utilization high:

- R20B extends the monotonic R20A loss curve to 20 epochs.
- R20C increases LR from `3.0` to `5.0`.
- R20D keeps LR `3.0` but adds `PROMPT_RESIDUAL_WEIGHT=0.15`.
- R20E/R20F keep `TRAIN_TASKS=code` but widen `TRAINABLE_EXPERTS`, testing whether Code ability comes from a combination of task vectors rather than the Code expert slice alone.

## Metrics To Review

For every epoch:

- task-wise `residual_loss`, `contrastive_negative_loss`, active contrast rate;
- task-wise loss scale;
- gate means and layer-band gate values;
- whether Code loss decreases while Tool/Memory losses remain controlled.

Selection:

- First screen by Tool mean >= 0.79 and Memory F1 >= 0.76.
- Run expensive CURE Code eval only for candidates passing the quick screen, unless a run has a clearly superior Code-loss/gate pattern worth diagnostic evaluation.
