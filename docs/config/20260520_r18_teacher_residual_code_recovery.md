# 20260520 R18 Teacher-Residual Code Recovery

## 目标

验证 Code 能力是否可通过 trajectory residual steering 学回来，而不是继续把 Code expert residual 单向推到 `1.0`。

本轮不从 `0.75` checkpoint 初始化。训练从 `init=1.0` 开始，但 residual target 改为强 teacher / baseline 行为 residual。

## 关键假设

当前旧 TRC:

```text
target = row expert full residual
```

因此 Code row 会天然推向 `code=1.0`。

R18 改成:

```text
target = fixed teacher coefficient residual
```

第一版 teacher 用 `tool=0.75,memory=0.75,code=0.75`，不是作为 init，而是作为成功 merged behavior witness。这样如果当前 init1 过强，MSE residual 会产生压低 gate 的梯度。

## 实验 A: R18A c075 teacher residual

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

EXP_ID=trc_r18a_teacher_c075_relmse_e6_20260520 \
GPU_LIST=6,7 \
CONFIG=configs/gated_grpo_layer28.yaml \
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl \
EPOCHS=6 \
INIT_VALUE=1.0 \
OPTIMIZER=adamw \
LR=0.02 \
ACCUMULATION_STEPS=96 \
MAX_SEQ_LENGTH=1536 \
MAX_RESPONSE_TOKENS=512 \
TOPK_TOKENS=128 \
RESPONSE_SPAN_MODE=response \
RESIDUAL_OBJECTIVE=relative-mse \
RESIDUAL_WEIGHT_POWER=0.5 \
RESIDUAL_TARGET_SOURCE=coefficients \
RESIDUAL_TARGET_COEFFICIENTS=tool=0.75,memory=0.75,code=0.75 \
BETA_BASE=0.0 \
GAMMA_GATE=0.0001 \
COEFFICIENT_FLOOR=0.0 \
COEFFICIENT_FLOOR_WEIGHT=0.0 \
TASK_EXPERT_COEFFICIENT_FLOOR=0.0 \
TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=0.0 \
DIRECTIONAL_PROJECTION_FLOOR=0.0 \
DIRECTIONAL_PROJECTION_WEIGHT=0.0 \
MAX_MEMORY_ENTRIES="0=70GiB 1=70GiB cpu=180GiB" \
bash skill/command/run_20260519_trc_round_train_one.sh
```

## 判据

先看是否可学：

1. residual loss 是否下降；
2. gate 是否从 init1 向 teacher/better region 移动；
3. Code gate 是否可以被压低，而不是只能推高；
4. Tool/Memory gate 是否没有被异常压坏。

如果 A 能正常收敛，再做：

- B: target `tool=0.50,memory=0.75,code=0.75`，近似 TAME best 的 cg-tool-extra020 主系数；
- C: target from selected strong layer-band gate checkpoint；
- D: alpha-fit loss，把 offline alpha* 写入 row metadata 后直接训练 gate。

## 注意

这不是 fixed sweep。训练从 init1 出发，teacher coefficient 只是 hidden-residual witness；真正优化仍在 TRC gate parameter space 内进行。

## 实验 P/PC: Code Prompt Steering

用户提出 Code 可能主要缺 prompt 约束理解，而不是 final code block 风格。因此新增 Code-only mixed calibration：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round18_code_prompt_mixed/trc32_code_prompt_mixed.jsonl
```

构成：

| source | rows |
|---|---:|
| eval-leak CURE LiveBench/LiveCodeBench code rows | 10 |
| non-leak CodeP0 train rows | 22 |

Teacher target 第一版用 TAME best 的主系数近似：

```text
tool=0.50,memory=0.75,code=0.75
```

注意这不是 exact TAME baked model hidden residual；exact TAME 还包含 conflict-gated tool residual 和 64 个 R1 micro-modes。若 P/PC 显示 Code 可学，再做 exact TAME hidden cache。

### R18P: prompt-only

目的：只对齐 prompt span 的 teacher residual，验证 Code 是否主要是题意/约束理解 steering。

```bash
EXP_ID=trc_r18p_code_promptonly_tamemain_e6_20260520 \
GPU_LIST=0,1 \
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round18_code_prompt_mixed/trc32_code_prompt_mixed.jsonl \
EPOCHS=6 \
INIT_VALUE=1.0 \
OPTIMIZER=adamw \
LR=0.02 \
ACCUMULATION_STEPS=32 \
MAX_SEQ_LENGTH=1536 \
MAX_RESPONSE_TOKENS=512 \
TOPK_TOKENS=128 \
RESPONSE_RESIDUAL_WEIGHT=0.0 \
PROMPT_RESIDUAL_WEIGHT=1.0 \
PROMPT_RESIDUAL_TOKENS=512 \
RESPONSE_SPAN_MODE=code-block \
RESIDUAL_OBJECTIVE=relative-mse \
RESIDUAL_WEIGHT_POWER=0.5 \
RESIDUAL_TARGET_SOURCE=coefficients \
RESIDUAL_TARGET_COEFFICIENTS=tool=0.50,memory=0.75,code=0.75 \
BETA_BASE=0.0 \
GAMMA_GATE=0.0001 \
COEFFICIENT_FLOOR=0.0 \
COEFFICIENT_FLOOR_WEIGHT=0.0 \
DIRECTIONAL_PROJECTION_FLOOR=0.0 \
DIRECTIONAL_PROJECTION_WEIGHT=0.0 \
MAX_MEMORY_ENTRIES="0=70GiB 1=70GiB cpu=180GiB" \
bash skill/command/run_20260519_trc_round_train_one.sh
```

### R18PC: prompt + code span

目的：对照 prompt-only，检查加入 final code block 是否恢复 execution behavior，还是把模型拉回表面代码风格。

与 R18P 只有两处不同：

```text
EXP_ID=trc_r18pc_code_prompt_plus_codespan_tamemain_e6_20260520
GPU_LIST=2,3
RESPONSE_RESIDUAL_WEIGHT=1.0
PROMPT_RESIDUAL_WEIGHT=1.0
```

### 快速判据

- 如果 P 的 prompt residual loss 快速下降，且 gate 出现稳定收缩/分化，说明 prompt steering 可学。
- 如果 PC 明显比 P 更不稳，说明 final code block residual 可能引入风格噪声。
- 如果 P/PC 都不动，说明只靠 TAME-main coefficient residual 不足，需要 exact TAME baked-model hidden cache 或 pass/fail residual ranking。
