# 2026-05-20 Round13 Formal-Code Eval-Leak Hidden-State Diagnostic

## 目的

Round13 是诊断实验，不进入论文主结果。它回答一个问题：如果 Code calibration 直接使用 formal LiveBench/LiveCodeBench 分布中 expert 已做对的轨迹，当前 TRC hidden-state loss 是否能把 Code formal eval 拉起来。

## 数据

Builder:

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
$PY scripts/trc/build_trc_round13_evalleak_code16_calibration.py
```

输出根目录：

`/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round13_evalleak_code16`

| variant | rows | Tool | Memory | Code | Code source | Code gate target |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `rfmem_only` | 74 | 32 | 32 | 10 | ReasonFlux 4, MemoryAgent 6 | ReasonFlux -> code, MemoryAgent -> memory |
| `all_with_r1` | 87 | 32 | 32 | 23 | ReasonFlux 4, MemoryAgent 6, DeepSeek/R1 13 | ReasonFlux -> code, MemoryAgent -> memory, DeepSeek -> reasoning |

Code 轨迹来自：

`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_expert_success_rollouts_seed20260519.jsonl`

Tool/Memory 轨迹来自稳定 bank：

`/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl`

重要处理：

- 这是 eval-leak diagnostic，只用于判断上限和定位目标错配。
- 不带 R1 task vector 的 `rfmem_only` 明确排除 DeepSeek/R1 轨迹。
- 带 R1 对照 `all_with_r1` 使用全部 verified positive 轨迹。
- Code response 被压缩成 `critical_reasoning_span + final_code_span`，span 写入 `sample_metadata.ability_spans`。
- 2026-05-20 11:46 CST 修正：reasoning context 从 1800 chars 改为 600 chars。原因是 3072/1024 首轮 OOM；回退到 R12 稳定的 1536/512 预算时，必须保证 final code block 进入前 512 response tokens。
- DeepSeek/R1 delta 使用已缩放的 correct-R1 math-base mode：`/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json`，scale factor `0.0069105254`。

## 训练设置

共同设置：

- method: TRC hidden-state residual alignment
- init: `INIT_VALUE=1.0`
- epochs: `8`
- optimizer: inherited from `run_20260519_trc_round_train_one.sh`, `LR=0.02`
- max seq: `1536`
- max response tokens: `512`
- code span: `response`，但 response 已由 builder 压缩为能力 span
- code topK tokens: `384`
- hidden layers: default `8,16,24,28`; code override `8,16,24,28`
- task balanced loss: on
- task multipliers: `code=1.2 memory=1.8 tool=1.2`
- task expert floor: `1.0`, weight `50.0`
- memory uses trajectory-turn loss

### R13A: no-R1 task vector, RF/Mem trajectories only

```bash
tmux new-session -d -s train_r13a_evalleak_rfmem_20260520 \
  "cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && \
  EXP_ID=trc_r13a_evalleak_rfmem_compact600_e8_20260520 \
  GPU_LIST=2,3 \
  CONFIG=configs/gated_grpo_layer28_wide.yaml \
  MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round13_evalleak_code16/rfmem_only/trc_expert_trajectories.jsonl \
  RUN_DIR=/tmp/shared-storage/OnPolicy/runs/trc/trc_r13a_evalleak_rfmem_compact600_e8_20260520 \
  BAKED_DIR=/tmp/shared-storage/OnPolicy/checkpoints/trc_r13a_evalleak_rfmem_compact600_e8_20260520-selected \
  EPOCHS=8 LR=0.02 MAX_SEQ_LENGTH=1536 MAX_RESPONSE_TOKENS=512 TOPK_TOKENS=128 \
  TASK_EXPERT_COEFFICIENT_FLOOR=1.0 TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0 \
  TASK_HIDDEN_LAYERS='code=8,16,24,28' TASK_TOPK_TOKENS='code=384' \
  TASK_DIRECTIONAL_PROJECTION_FLOOR='code=0.95' TASK_DIRECTIONAL_PROJECTION_WEIGHT='code=0.25' \
  TASK_RESPONSE_SPAN_MODE='tool=tool-call code=response memory=response' \
  TASK_LOSS_MULTIPLIER='code=1.2 memory=1.8 tool=1.2' \
  TRAJECTORY_TURN_LOSS_TASKS=memory \
  bash skill/command/run_20260519_trc_round_train_one.sh"
```

### R13B: all trajectories + scaled R1 task vector

```bash
tmux new-session -d -s train_r13b_evalleak_all_r1_20260520 \
  "cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && \
  EXP_ID=trc_r13b_evalleak_all_r1_compact600_e8_20260520 \
  GPU_LIST=4,7 \
  CONFIG=configs/gated_grpo_4expert_r1math_layer28.yaml \
  MODE=/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json \
  CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round13_evalleak_code16/all_with_r1/trc_expert_trajectories.jsonl \
  RUN_DIR=/tmp/shared-storage/OnPolicy/runs/trc/trc_r13b_evalleak_all_r1_compact600_e8_20260520 \
  BAKED_DIR=/tmp/shared-storage/OnPolicy/checkpoints/trc_r13b_evalleak_all_r1_compact600_e8_20260520-selected \
  EPOCHS=8 LR=0.02 MAX_SEQ_LENGTH=1536 MAX_RESPONSE_TOKENS=512 TOPK_TOKENS=128 \
  TASK_EXPERT_COEFFICIENT_FLOOR=1.0 TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0 \
  TASK_HIDDEN_LAYERS='code=8,16,24,28' TASK_TOPK_TOKENS='code=384' \
  TASK_DIRECTIONAL_PROJECTION_FLOOR='code=0.95' TASK_DIRECTIONAL_PROJECTION_WEIGHT='code=0.25' \
  TASK_RESPONSE_SPAN_MODE='tool=tool-call code=response memory=response' \
  TASK_LOSS_MULTIPLIER='code=1.2 memory=1.8 tool=1.2' \
  TRAJECTORY_TURN_LOSS_TASKS=memory \
  bash skill/command/run_20260519_trc_round_train_one.sh"
```

## 判定

训练结束后先跑 Tool/Memory quick gate：

- Tool mean >= `0.79`
- Memory mean F1 >= `0.76`

通过后再跑 Code formal eval。若 R13B Code 明显优于 R13A，说明 DeepSeek/R1 task vector 与 formal-code 正轨迹能提供新能力；若 R13A/R13B 都不涨，说明当前 TRC hidden-state objective 仍不能把 formal code ability 转成可泛化 gate，需要进入 contrastive / guard-test-aware loss。

## 运行状态

2026-05-20 12:09 CST:

- R13A `trc_r13a_evalleak_rfmem_compact600_e8_20260520` 已完成并 bake。自动 selection 实际选中 epoch 4，gate mean：Code `1.0389`，Memory `1.0177`，Tool `1.0685`；epoch 8 gate mean 为 Code `1.0821`，Memory `1.0332`，Tool `1.1346`。这说明 loss-plateau selection 对 Code diagnostic 偏保守；已额外 bake epoch 8 forced checkpoint `/tmp/shared-storage/OnPolicy/checkpoints/trc_r13a_evalleak_rfmem_compact600_e08_forced_20260520-selected` 并启动 Tool/Memory quick gate `eval_r13a_e08_forced_tm_20260520`。
- R13B `trc_r13b_evalleak_all_r1_compact600_e8_20260520` 两卡 OOM，没有 baked checkpoint。直接原因是 4-expert correct-R1 scaled mode 的 delta 常驻显存高于 3-expert mode，叠加 memory trajectory-turn 的 base/expert/merged 三次 forward。后续若继续测 R1，需要 4 卡 device_map，或把 memory 改为 final-turn/fewer-turn low-memory diagnostic。
- 已启动 low-memory R1 control：`trc_r13b_evalleak_all_r1_finalmem_e8_20260520`。该 run 保持 all+R1 formal-code bank 和 scaled correct-R1 mode，但关闭 `TRAJECTORY_TURN_LOSS_TASKS=memory`，让 Memory 使用 final response。它不是 R13A 的严格同配置对照，但可以快速判断 R1+formal-code 方向是否值得上 4 卡完整版本。自动 selection 也选中 epoch 4；epoch 8 forced checkpoint 已 bake 到 `/tmp/shared-storage/OnPolicy/checkpoints/trc_r13b_evalleak_all_r1_finalmem_e08_forced_20260520-selected`，等待 BFCL quick gate 空档。
