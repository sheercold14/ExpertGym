# 2026-05-18 DeepSeek-R1 Scaled Layer28 配置

## 目标

验证 DeepSeek-R1-Distill-Qwen-7B 是否能作为小幅 reasoning/code task vector prior，在 ExpertGym 的 executable feedback 下提升 Code，同时保持 Memory/Tool。

核心控制：

```text
tool=memory=code=1.0
reasoning=0.001
reasoning max delta from init = 0.002  # 即 reasoning 系数限制在约 [0, 0.003]
```

## 文件

```text
config: configs/gated_grpo_reasoning_layer28.yaml
mode: /tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json
report: docs/report/20260518_deepseek_r1_scaled_task_vector_plan.md
```

## 构建 init gate

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
export ROOT=/tmp/shared-storage/OnPolicy

$PY scripts/modes/build_constant_gate_checkpoint.py \
  --config configs/gated_grpo_reasoning_layer28.yaml \
  --mode-manifest $ROOT/modes/opvec4_reasoning_20260516/mode_manifest.json \
  --gate-parameterization layer-band \
  --value 1.0 \
  --expert-value reasoning=0.001 \
  --output $ROOT/data/init_gates/r1_scaled_20260518/all1_r1_z001.layer-band.json
```

语义：

- 每个 layer band 都有 `tool/memory/code/reasoning` 四个 effective coefficient；
- T/M/C effective coefficient 为 `1.0`；
- R1 effective coefficient 为 `0.001`；
- YAML 中 residual bound 放宽到 `[-1, 1]`，用于表达异尺度 expert，不代表允许 R1 大幅漂移；真正 R1 安全边界由 `MAX_COEFF_DELTA_BY_EXPERT` 控制。

已生成：

```text
/tmp/shared-storage/OnPolicy/data/init_gates/r1_scaled_20260518/all1_r1_z001.layer-band.json
```

plan-only bake 验证：

```text
/tmp/shared-storage/OnPolicy/data/init_gates/r1_scaled_20260518/plan_only_bake/bake_plan.json
gate_parameterization = layer-band
num_delta_entries = 784
layer0/layer14/layer27 coefficients = tool 1.0, memory 1.0, code 1.0, reasoning 0.001
```

## 推荐短跑

先用现有 recoverable101 做可行性短跑；等 Code P0 v3 bank 完成后再替换 calibration。

```bash
tmux new -d -s train_r1_layer28_codeprobe_20260518 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && \
   GPU_LIST=0,1 \
   CONFIG=configs/gated_grpo_reasoning_layer28.yaml \
   MODE=/tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json \
   CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl \
   RUN_NAME=r1_scaled_layer28_z001_recoverable101_20260518 \
   RUN_DIR=/tmp/shared-storage/OnPolicy/runs/gated_grpo/r1_scaled_layer28_z001_recoverable101_20260518 \
   STRATEGY=layer-band \
   INIT_GATE_CHECKPOINT=/tmp/shared-storage/OnPolicy/data/init_gates/r1_scaled_20260518/all1_r1_z001.layer-band.json \
   NUM_ITERS=6 \
   NUM_PROMPTS=101 \
   SAMPLES_PER_PROMPT=4 \
   CODE_MAX_NEW_TOKENS=10000 \
   MAX_MODEL_LEN=24576 \
   MAX_LOGPROB_TOKENS=24576 \
   PPO_LOSS_WEIGHT=1.0 \
   OPD_LOSS_WEIGHT=1.0 \
   USE_RETENTION=1 \
   RETENTION_OBJECTIVE=nll \
   RETENTION_LOSS_WEIGHT=0.05 \
   ADVANTAGE_NORMALIZATION=zscore \
   OPTIMIZER=sgd \
   SGD_MOMENTUM=0.2 \
   OPTIMIZER_STEP_SCOPE=epoch \
   LOSS_GRANULARITY=sequence \
   LR=0.05 \
   PRIOR_LOSS_WEIGHT=0.0 \
   MAX_COEFF_DELTA=0.15 \
   MAX_COEFF_DELTA_BY_EXPERT=reasoning=0.002 \
   DYNAMIC_OPD_EXPERT_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/tool_expert_toolrl_qwen25_7b_sota_v2_train128_s4_seed20260518.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/memory_expert_rl_memoryagent7b_sota_v2_train128_s4_seed20260518.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/code_expert_reasonflux_coder7b_sota_v2_train128_s8_seed20260518.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_sota_v2_train128_s8_seed20260518.jsonl \
   DYNAMIC_OPD_TASKS=tool,memory,code \
   DYNAMIC_OPD_PER_TASK=32 \
   bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/r1_scaled_layer28_z001_recoverable101_20260518/train.log'
```

## 监控标准

短跑只看是否值得进入正式 P0：

| 条件 | 晋级判断 |
|---|---|
| Code train reward | 需要上升 |
| monitor64 Code reward | 需要同步上升，不能只涨 train |
| Memory gate | 目标 `>0.55`，或者 formal memory 不低于当前强模型 |
| reasoning gate | 大多数 layer 应在 `[0, 0.003]` 内，不应被投影外溢 |
| Tool reward | 不应出现 tool-call 格式崩溃 |

若 Code reward 不涨，优先修 Code P0 v3 bank/reward，而不是继续放大 R1。
