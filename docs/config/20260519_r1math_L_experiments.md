# 2026-05-19 Correct-R1 Task Vector L 系列实验

## 核心修正

旧 R1 delta 是错误语义：

```text
DeepSeek-R1-Distill-Qwen-7B - Qwen2.5-7B-Instruct
```

DeepSeek-R1-Distill-Qwen-7B 的官方 base 是 Qwen2.5-Math-7B，因此今晚先构建：

```text
Delta_R1_correct = DeepSeek-R1-Distill-Qwen-7B - Qwen2.5-Math-7B
```

runtime merge anchor 仍然是 Qwen2.5-7B-Instruct。tool / memory / code 仍减 Instruct base；只有 reasoning 使用 per-expert delta base。

## 产物路径

| item | path |
|---|---|
| Math base | `/mnt/cache/wuruixiao/models/Qwen2.5-Math-7B` |
| config | `configs/gated_grpo_4expert_r1math_layer28.yaml` |
| raw correct R1 modes | `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519` |
| scaled correct R1 modes | `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519` |
| init gates | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_r1math_init_gates/init_layer_band_parameter_28layer_tmc033_r0.json` |
| run script | `skill/command/run_20260519_r1math_L_experiments.sh` |
| build script | `skill/command/build_20260519_r1math_modes.sh` |

## 当前执行状态

更新时间：2026-05-19 10:27 CST

| item | status |
|---|---|
| Qwen2.5-Math-7B 下载 | completed; safetensors shard 可读取 |
| correct-R1 raw modes | completed: `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519/mode_manifest.json` |
| correct-R1 scaled modes | completed: `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json` |
| reasoning scale factor | `0.006910525387901668` |
| L1 | completed; iter20 gate produced; final is over-pushed and not a candidate |
| L2 | running in tmux `train_L2_r1math_20260519`, GPU `6,7`; R1 coefficient frozen by `TRAIN_COEFFICIENTS=*.tool,*.memory,*.code`; iter19 gate completed; iter19 rollout tool near zero |
| L3 | running in tmux `train_L3_r1math_fastopd_20260519`, GPU `4,5`; all coefficients trainable; fast OPD+retention with `LR=0.35`; iter17 gate completed; iter17 rollout tool near zero |
| L4 | running in tmux `train_L4_r1math_bfcltool_reqall_20260519`, GPU `0,1`; L1 setting + BFCL Tool16 + dynamic OPD require-all; iter1 collect started |
| monitor | `http://127.0.0.1:8798` or `http://<server-ip>:8798`; includes L1/L2/L3/L4 |

First rollout snapshot:

| run | iter | tool reward | memory reward | code reward | frontier tool/memory/code | OPD selected tool/memory/code | note |
|---|---:|---:|---:|---:|---|---|---|
| L1 | 1 | 0.4220 | 0.4141 | 0.3721 | 21 / 23 / 23 | 10 / 7 / 10 | rollout done; update running |
| L2 | 1 | pending parse | pending parse | pending parse | pending parse | 8 / 3 / 13 | rollout done; first update needed train-coefficient projection repair |

Gate snapshot:

| run | iter | tool mean | memory mean | code mean | reasoning mean | note |
|---|---:|---:|---:|---:|---:|---|
| L1 | 1 | 0.3309 | 0.3694 | 0.3377 | 0.0005 | correct R1 trainable |
| L1 | 2 | 0.3316 | 0.4005 | 0.3450 | 0.0016 | iter2 update repaired on GPU0 |
| L1 | 3 | 0.3322 | 0.4339 | 0.3501 | 0.0024 | single-GPU continuation |
| L1 | 4 | 0.3317 | 0.4632 | 0.3555 | 0.0030 | switch to dual-GPU after this checkpoint |
| L1 | 5 | 0.3312 | 0.4944 | 0.3609 | 0.0035 | dual-GPU continuation |
| L1 | 6 | 0.3298 | 0.5220 | 0.3665 | 0.0043 | memory passed 0.52 |
| L1 | 7 | 0.3286 | 0.5554 | 0.3728 | 0.0050 | memory passed 0.55 |
| L1 | 8 | 0.3283 | 0.5752 | 0.3797 | 0.0060 | memory continues rising |
| L1 | 9 | 0.3287 | 0.5942 | 0.3869 | 0.0068 | high memory; tool proxy in next rollout drops |
| L1 | 10 | 0.3346 | 0.6089 | 0.3946 | 0.0077 | over-push region |
| L1 | 11 | 0.3382 | 0.6219 | 0.4023 | 0.0087 | over-push continues |
| L1 | 12 | 0.3393 | 0.6362 | 0.4096 | 0.0094 | over-push continues |
| L1 | 13 | 0.3397 | 0.6504 | 0.4171 | 0.0104 | over-push continues |
| L1 | 14 | 0.3404 | 0.6603 | 0.4254 | 0.0114 | over-push continues |
| L1 | 15 | 0.3412 | 0.6692 | 0.4331 | 0.0122 | over-push continues |
| L1 | 16 | 0.3421 | 0.6772 | 0.4410 | 0.0131 | over-push continues |
| L1 | 19 | 0.3450 | 0.6964 | 0.4637 | 0.0156 | final region over-pushed |
| L1 | 20 | 0.3462 | 0.7069 | 0.4699 | 0.0163 | completed; final over-pushed |
| L2 | 1 | 0.3320 | 0.3597 | 0.3359 | 0.0000 | reasoning frozen |
| L2 | 2 | 0.3319 | 0.3941 | 0.3410 | 0.0000 | freeze repair verified |
| L2 | 3 | 0.3321 | 0.4199 | 0.3464 | 0.0000 | reasoning remains frozen |
| L2 | 4 | 0.3318 | 0.4516 | 0.3531 | 0.0000 | reasoning remains frozen |
| L2 | 5 | 0.3310 | 0.4756 | 0.3598 | 0.0000 | reasoning remains frozen |
| L2 | 6 | 0.3300 | 0.5042 | 0.3658 | 0.0000 | memory passed 0.50; reasoning remains frozen |
| L2 | 7 | 0.3286 | 0.5395 | 0.3718 | 0.0000 | memory continues rising; reasoning remains frozen |
| L2 | 8 | 0.3273 | 0.5715 | 0.3781 | 0.0000 | memory continues rising; reasoning remains frozen |
| L2 | 9 | 0.3270 | 0.5959 | 0.3834 | 0.0000 | high memory; currently best-balanced proxy before next rollout |
| L2 | 10 | 0.3339 | 0.6129 | 0.3895 | 0.0000 | over-push region; reasoning remains frozen |
| L2 | 11 | 0.3370 | 0.6277 | 0.3960 | 0.0000 | over-push continues; reasoning remains frozen |
| L2 | 12 | 0.3380 | 0.6401 | 0.4037 | 0.0000 | over-push continues; reasoning remains frozen |
| L2 | 13 | 0.3391 | 0.6518 | 0.4104 | 0.0000 | over-push continues; reasoning remains frozen |
| L2 | 14 | 0.3397 | 0.6619 | 0.4181 | 0.0000 | over-push continues; reasoning remains frozen |
| L2 | 15 | 0.3407 | 0.6723 | 0.4242 | 0.0000 | over-push continues; reasoning remains frozen |
| L2 | 16 | 0.3422 | 0.6862 | 0.4308 | 0.0000 | over-push continues; reasoning remains frozen |
| L2 | 19 | 0.3473 | 0.7213 | 0.4499 | 0.0000 | final region over-pushed; reasoning remains frozen |
| L3 | 1 | 0.3327 | 0.3769 | 0.3392 | 0.0009 | fast OPD LR=0.35 |
| L3 | 2 | 0.3338 | 0.4112 | 0.3465 | 0.0017 | iter3 restarted after shard failure |
| L3 | 3 | 0.3326 | 0.4645 | 0.3527 | 0.0025 | restarted path healthy |
| L3 | 4 | 0.3314 | 0.5020 | 0.3607 | 0.0035 | LR=0.35 crosses 0.50 memory earlier |
| L3 | 5 | 0.3280 | 0.5477 | 0.3701 | 0.0048 | fast variant catches up to 0.55 memory |
| L3 | 6 | 0.3262 | 0.5943 | 0.3770 | 0.0055 | memory rises fast; watch tool retention |
| L3 | 7 | 0.3329 | 0.6209 | 0.3862 | 0.0065 | over-push region |
| L3 | 8 | 0.3364 | 0.6421 | 0.3953 | 0.0073 | over-push continues |
| L3 | 9 | 0.3377 | 0.6564 | 0.4063 | 0.0086 | over-push continues |
| L3 | 10 | 0.3397 | 0.6731 | 0.4168 | 0.0096 | over-push continues |
| L3 | 11 | 0.3419 | 0.6890 | 0.4278 | 0.0110 | over-push continues |
| L3 | 12 | 0.3448 | 0.7030 | 0.4378 | 0.0120 | over-push continues |
| L3 | 13 | 0.3476 | 0.6870 | 0.4490 | 0.0132 | over-push continues |
| L3 | 16 | 0.3494 | 0.6856 | 0.4814 | 0.0170 | final region over-pushed |
| L3 | 17 | 0.3498 | 0.6975 | 0.4909 | 0.0180 | final region over-pushed |

Recent proxy reward snapshot:

| run | iter | tool reward / success | memory reward / success | code reward / success | note |
|---|---:|---:|---:|---:|---|
| L1 | 20 | 0.033 / 0.00 | 0.820 / 0.82 | 0.395 / 0.29 | final rollout still loses tool |
| L2 | 19 | 0.031 / 0.00 | 0.820 / 0.82 | 0.386 / 0.30 | final region still loses tool |
| L3 | 17 | 0.040 / 0.00 | 0.852 / 0.85 | 0.396 / 0.30 | final region still loses tool |

Scale diagnostic:

```text
agent experts mean per-param norm = 0.1473
reasoning mean per-param norm     = 21.3087
reasoning scale factor            = 0.006911

tool      count=196 mean_norm=0.0765 total_norm=1.2485
memory    count=196 mean_norm=0.3272 total_norm=5.3376
code      count=196 mean_norm=0.0381 total_norm=0.6199
reasoning count=196 mean_norm=21.3087 total_norm=346.1319
```

## 实验表

| id | 目的 | mode | trainable coefficients | init | loss | iters | GPU | run dir |
|---|---|---|---|---|---|---:|---|---|
| L1 | 复现旧 `r1_layer28_hier` 的训练设置，但替换为正确 R1 delta | correct R1 scaled | tool/memory/code/reasoning all trainable | T/M/C=1/3, R=0 | OPD + NLL retention, no GRPO | 20 | 0,1 优先 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expL1_r1math_layer28_hier_20it_20260519` |
| L2 | L1 对照：冻结 R1，只训练三个 agent expert | correct R1 scaled | `*.tool,*.memory,*.code`; reasoning anchored at 0 | T/M/C=1/3, R=0 | same as L1 | 20 | 6,7 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expL2_r1math_layer28_hier_freezeR1_20it_20260519` |
| L3 | 机动策略：更强 OPD 推动，要求今晚可跑完 | correct R1 scaled | tool/memory/code/reasoning all trainable | T/M/C=1/3, R=0 | OPD + NLL retention, no GRPO, `LR=0.35` | 20 | 4,5 after iter3 restart | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expL3_r1math_layer28_hier_opd_lr035_20it_20260519` |
| L4 | L1 设置 + BFCL Tool calibration + dynamic OPD 全任务齐备保护，测试能否避免 Tool 崩 | correct R1 scaled | tool/memory/code/reasoning all trainable | T/M/C=1/3, R=0 | OPD + NLL retention, no GRPO, require all tasks for dynamic OPD | 20 | 0,1 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expL4_r1math_layer28_hier_bfcltool16_reqallopd_20it_20260519` |

## 运行管理

- 不删除旧诊断实验；若它们占用 2-5 卡但仍在正常结束阶段，先等待完成。
- L1 使用断点续跑：`START_ITERATION=2`，`INIT_GATE_CHECKPOINT=.../L1/iter_001/gate_updates.gates.json`。
- L1 iter2 rollout 完成后 update 因 GPU1 残留上下文 OOM；当前只重跑 iter2 update，不重复 rollout。
- L1 在 GPU1 恢复后从 `iter_005` 切回双卡，删除了单卡刚启动的 partial `iter_005`。
- L2 修复后先重跑 `iter_001` update，产出 gate 后用 `START_ITERATION=2` 继续。
- L2 第二轮曾发现 frozen reasoning 被错误投到 `0.5`；原因是 direct coefficient checkpoint anchor 未被 `_anchor_expert_coefficients` 识别。已修复并删除坏的 `iter_002/iter_003` 后重跑。
- L3 iter3 首次在 GPU `2,3` rollout 时遇到 vLLM startup free-memory check failure；已删除 partial `iter_003`，改用 GPU `4,5` 从 iter3 续跑。
- 若有外部非本计划任务重新抢占 L 系列 GPU 并导致 vLLM 初始化失败，优先清理该阻塞进程，保证 L1/L2/L3 主计划完成。

## L3 决策规则

L3 最初尝试过 `PPO_LOSS_WEIGHT=1.0` 的 GRPO 机动对照，但 iter1 update 接近 30 分钟仍未产出 gate，不满足今晚 20 iter 的时限目标，已停止并清理。当前 L3 改为 fast OPD：保持全系数可学，不加 GRPO，把 LR 从 `0.25` 提到 `0.35`，检验更强 OPD 是否能更快推动 memory/code/reasoning 系数。

后续仍按以下规则调整：

- 如果 L1 reasoning coefficient 有上涨且 monitor/code proxy 不掉，L3 用更激进 R1 bound/LR 或只开放 code+reasoning 相关系数。
- 如果 L1 reasoning 不动但 memory/code 有收益，L3 改为 no-R1 三专家强化，验证 R1 是否应仅作为 teacher 不作为 delta。
- 如果 L1/L2 reward 均不涨，L3 引入少量 GRPO (`PPO_LOSS_WEIGHT>0`) 或切换到更高信息量 calibration bank。
- 如果 tool retention 明显下降，L3 提高 retention 或冻结 tool 下界。

## 执行命令

下载 Math base：

```bash
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download Qwen/Qwen2.5-Math-7B \
  --local-dir /mnt/cache/wuruixiao/models/Qwen2.5-Math-7B \
  --max-workers 8
```

构建 correct-R1 modes：

```bash
bash skill/command/build_20260519_r1math_modes.sh
```

启动 L1：

```bash
PHASE=L1 GPU_LIST=0,1 NUM_ITERS=20 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

启动 L2：

```bash
PHASE=L2 GPU_LIST=2,3 NUM_ITERS=20 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

启动 L3 示例，实际参数按 L1/L2 动态再定：

```bash
PHASE=L3 GPU_LIST=4,5 NUM_ITERS=20 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

构建 L4 BFCL Tool augmentation：

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_bfcl_tool_calibration.py
```

启动 L4：

```bash
PHASE=L4 GPU_LIST=0,1 NUM_ITERS=20 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

L4 数据：

| artifact | path |
|---|---|
| merged manifest | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/qbank_c033333_paper96_plus_bfcl_tool16_seed20260519.prompts.jsonl` |
| BFCL Tool prompts | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_nonlive8_live8_seed20260519.prompts.jsonl` |
| BFCL official-answer expert rollout | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_official_answer_expert_rollouts_seed20260519.jsonl` |
| summary | `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/qbank_c033333_paper96_plus_bfcl_tool16_seed20260519.summary.json` |

L4 dry-run verified on 2026-05-19:

- manifest rows: `112`, task counts `tool=48, memory=32, code=32`;
- BFCL rows: `parallel=4`, `parallel_multiple=4`, `live_parallel=4`, `live_parallel_multiple=4`;
- live hard rows: `8`, exceeding the required `6`;
- rollout shards: two single-GPU vLLM shards on `0,1`, `56` prompts each;
- dynamic OPD expert rollouts include original paper96 experts, code augmentation experts, and BFCL official-answer Tool anchors;
- `DYNAMIC_OPD_REQUIRE_ALL_TASKS=1`, so an iteration with missing Tool/Memory/Code OPD rows skips dynamic OPD entirely and keeps only retention/other active losses.

## 监控指标

- reward: overall/tool/memory/code mean reward
- frontier rows / all-fail / all-success
- dynamic OPD selected rows by task
- effective gate means: tool/memory/code/reasoning
- gate delta / grad norm
- best checkpoint rule: monitor reward 优先，其次 proxy overall；如 proxy 与正式评测冲突，以正式评测为准。

## 临时选择规则

- 当前 `memory≈0.59-0.62` 后出现 tool proxy 下滑迹象，不能默认取 final checkpoint。
- 05:21 观察：L1/L2/L3 均出现并持续该模式，确认这是 over-push 区间，而不是单次采样噪声。
- 若后续 rollout 无恢复，正式评测优先候选应从 `L1/L2/L3` 的中间 checkpoint 中按 proxy overall 选择。
- 截至 04:36，`L2 iter009` 是最平衡候选：tool `0.956`、memory `0.820`、code `0.447`，且 R1 冻结对照更干净。

## 注意

旧 `opvec4_r1scaled_20260518` 及其训练不再作为论文主结果，只保留为“错误 base 的诊断对照”。
