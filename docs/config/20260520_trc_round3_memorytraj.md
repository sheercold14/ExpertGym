# 2026-05-20 TRC Round3 Memory-Trajectory Experiments

## Goal

用 MemAgent 完整/近似完整轨迹替代 final-answer-only memory proxy，验证是否能在保持 Tool 强度的前提下恢复 Memory，并筛出进入 Code 评测的候选。

## Shared Settings

- Base config: `configs/gated_grpo_layer28_wide.yaml`
- Mode manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`
- Trainer: `scripts/trc/train_trc_layer_gates.py`
- Runner: `skill/command/run_20260519_trc_round_train_one.sh`
- Gate parameterization: `layer-band-coefficient`
- Init: `1.0`
- Residual objective: `directional`
- Selection: `loss-plateau`, no gate interval penalty
- Memory trajectory loss: `TRAJECTORY_TURN_LOSS_TASKS="memory"`
- Checkpoint policy: bake selected only; delete baked dirs that fail Tool/Memory.

## Calibration Sets

Root: `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round3_memorytraj`

| set | memory turns | purpose |
|---|---:|---|
| `mtr_uniform4_toolaug_code_rf` | 4 memory_update + final | main, balanced speed/coverage |
| `mtr_late3_toolaug_code_rf` | last 3 memory_update + final | test late evidence integration |
| `mtr_full_toolaug_code_rf` | all memory_update + final | test upper-bound memory coverage |

## First Wave

| id | gpus | calib | epochs | max_seq | memory topk | task multiplier | hypothesis |
|---|---|---|---:|---:|---:|---|---|
| `trc_r3a_mtr_u4_main_20260520` | 1,2 | uniform4 | 10 | 4096 | 256 | code=1.5 memory=1.8 tool=1.2 | near-full memory span should lift memory gate/F1 without killing tool |
| `trc_r3b_mtr_late3_20260520` | 3,4 | late3 | 10 | 4096 | 256 | code=1.5 memory=1.8 tool=1.2 | late memory updates may be the highest-signal span |
| `trc_r3c_mtr_u4_memlayers_20260520` | 5,7 | uniform4 | 10 | 4096 | 384 | code=1.4 memory=2.0 tool=1.2 | memory may need broader token and layer coverage |

GPU 0/6 当前给 R2D/R2E memory eval 使用；空出后补 `trc_r3d_mtr_full_20260520`。

## 01:30 Retention Wave

第一轮 trajectory run 证明 memory turn-level loss 已经进入训练，但 memory residual loss 太小，必须补一个直接约束 task-vector 系数幅度的项。下面四个 run 保持数据与主体 loss 不变，只改 coefficient retention 形式与权重：

| id | calib | coefficient floor | task-aware floor | selected epoch | selected gates | status |
|---|---|---:|---:|---:|---|---|
| `trc_r3c_globalfloor20_u4_20260520` | uniform4 | `1.0 * 20` | off | 8 | T=1.1519 / M=0.9798 / C=1.1601 | trained, low priority |
| `trc_r3d_globalfloor50_u4_20260520` | uniform4 | `1.0 * 50` | off | 8 | T=1.1520 / M=0.9996 / C=1.1599 | baked, Tool+Memory eval |
| `trc_r3e_taskfloor20_u4_20260520` | uniform4 | off | `1.0 * 20` | 8 | T=1.1519 / M=0.9884 / C=1.1600 | trained, low priority |
| `trc_r3f_taskfloor50_u4_20260520` | uniform4 | off | `1.0 * 50` | 8 | T=1.1520 / M=1.0019 / C=1.1599 | baked, Tool+Memory eval |

当前判断：

- floor weight 20 只能减缓 memory 下降。
- floor weight 50 能把 memory 稳在 expert 幅度附近，同时不阻止 tool/code loss 继续下降。
- task-aware floor 更符合论文表述：只保护当前 row 对应 task 的 expert 系数，不把 unrelated expert 系数统一往上拉。

## 01:35 Next Wave

这批在 GPU 2-7 上并行运行；GPU 0/1 同时跑 R3D/R3F Tool+Memory 快评测。

| id | gpus | calib | key change | hypothesis |
|---|---|---|---|---|
| `trc_r3g_full_taskfloor50_20260520` | 2,3 | `mtr_full_toolaug_code_rf` | full memory trajectory, task-aware floor50, memory topk=96 | full trajectory may recover official memory F1 beyond uniform4 |
| `trc_r3h_full_globalfloor50_20260520` | 4,5 | `mtr_full_toolaug_code_rf` | full memory trajectory, global floor50, memory topk=96 | compare global vs task-aware retention under full trajectory |
| `trc_r3i_u4_codefull_taskfloor50_20260520` | 6,7 | `mtr_uniform4_toolaug_code_rf` | code span from `code-block` to full `response`, code topk=384 | Code eval may require reasoning / problem-understanding span, not only code block |

## 01:58 Eval-Gated Follow-Up

| id | type | path / gpus | key setting | current decision |
|---|---|---|---|---|
| `trc_r3d_globalfloor50_u4_20260520-selected` | eval | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3d_globalfloor50_u4_20260520-selected` | uniform4 + global floor50 | Tool mean `0.7944`, Memory F1 mean `0.7636`; promoted to Code |
| `trc_r3f_taskfloor50_u4_20260520-selected` | eval | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3f_taskfloor50_u4_20260520-selected` | uniform4 + task-aware floor50 | Tool mean `0.7775`; Memory pending, Code not prioritized |
| `trc_r3g2_full_probe_taskfloor50_20260520` | probe | GPU 2,3 | full trajectory, 8 rows/task, seq1536 | OOM; full trajectory dropped from tonight's main path |
| `trc_r3i_u4_codefull_taskfloor50_20260520-selected` | eval | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3i_u4_codefull_taskfloor50_20260520-selected` | code span=`response`, task floor50 | Tool+Memory running |
| `trc_r3j_late3_taskfloor50_20260520-selected` | eval | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3j_late3_taskfloor50_20260520-selected` | late3 memory, task floor50 | Tool+Memory running |
| `trc_r3k_u4_codefull_globalfloor50_20260520` | train | GPU 6,7 | code span=`response`, global floor50 | running |

## Candidate Commands

```bash
REPO=/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
ROOT=/tmp/shared-storage/OnPolicy
cd "$REPO"
```

```bash
EXP_ID=trc_r3a_mtr_u4_main_20260520 \
GPU_LIST=1,2 \
CONFIG=configs/gated_grpo_layer28_wide.yaml \
CALIB=$ROOT/data/calibration/20260520_trc_round3_memorytraj/mtr_uniform4_toolaug_code_rf/trc96_expert_trajectories.jsonl \
EPOCHS=10 LR=0.02 INIT_VALUE=1.0 \
MAX_SEQ_LENGTH=4096 MAX_RESPONSE_TOKENS=512 \
TASK_TOPK_TOKENS="memory=256 code=384 tool=96" \
TASK_HIDDEN_LAYERS="code=2,4,6,8,10,12,14,16,18,20,22,24,26,28 memory=8,12,16,20,24,28" \
TASK_RESPONSE_SPAN_MODE="memory=response code=code-block tool=tool-call" \
TASK_LOSS_MULTIPLIER="code=1.5 memory=1.8 tool=1.2" \
TRAJECTORY_TURN_LOSS_TASKS="memory" \
BETA_BASE=0.05 GAMMA_GATE=0.03 COEFFICIENT_FLOOR=0.0 COEFFICIENT_FLOOR_WEIGHT=0.0 \
ACCUMULATION_STEPS=96 \
skill/command/run_20260519_trc_round_train_one.sh
```

```bash
EXP_ID=trc_r3b_mtr_late3_20260520 \
GPU_LIST=3,4 \
CONFIG=configs/gated_grpo_layer28_wide.yaml \
CALIB=$ROOT/data/calibration/20260520_trc_round3_memorytraj/mtr_late3_toolaug_code_rf/trc96_expert_trajectories.jsonl \
EPOCHS=10 LR=0.02 INIT_VALUE=1.0 \
MAX_SEQ_LENGTH=4096 MAX_RESPONSE_TOKENS=512 \
TASK_TOPK_TOKENS="memory=256 code=384 tool=96" \
TASK_HIDDEN_LAYERS="code=2,4,6,8,10,12,14,16,18,20,22,24,26,28 memory=8,12,16,20,24,28" \
TASK_RESPONSE_SPAN_MODE="memory=response code=code-block tool=tool-call" \
TASK_LOSS_MULTIPLIER="code=1.5 memory=1.8 tool=1.2" \
TRAJECTORY_TURN_LOSS_TASKS="memory" \
BETA_BASE=0.05 GAMMA_GATE=0.03 COEFFICIENT_FLOOR=0.0 COEFFICIENT_FLOOR_WEIGHT=0.0 \
ACCUMULATION_STEPS=96 \
skill/command/run_20260519_trc_round_train_one.sh
```

```bash
EXP_ID=trc_r3c_mtr_u4_memlayers_20260520 \
GPU_LIST=5,7 \
CONFIG=configs/gated_grpo_layer28_wide.yaml \
CALIB=$ROOT/data/calibration/20260520_trc_round3_memorytraj/mtr_uniform4_toolaug_code_rf/trc96_expert_trajectories.jsonl \
EPOCHS=10 LR=0.02 INIT_VALUE=1.0 \
MAX_SEQ_LENGTH=4096 MAX_RESPONSE_TOKENS=512 \
TASK_TOPK_TOKENS="memory=384 code=320 tool=96" \
TASK_HIDDEN_LAYERS="code=4,8,12,16,20,24,28 memory=4,8,12,16,20,24,28 tool=8,16,24,28" \
TASK_RESPONSE_SPAN_MODE="memory=response code=code-block tool=tool-call" \
TASK_LOSS_MULTIPLIER="code=1.4 memory=2.0 tool=1.2" \
TRAJECTORY_TURN_LOSS_TASKS="memory" \
BETA_BASE=0.05 GAMMA_GATE=0.03 COEFFICIENT_FLOOR=0.0 COEFFICIENT_FLOOR_WEIGHT=0.0 \
ACCUMULATION_STEPS=96 \
skill/command/run_20260519_trc_round_train_one.sh
```
