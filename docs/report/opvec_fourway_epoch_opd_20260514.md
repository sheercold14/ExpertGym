# OpVec Epoch Accumulation + OPD 四路实验报告

日期：2026-05-14

## 目标

本轮实验验证两条线：

1. **epoch-scope 梯度累计**：同一轮 rollout 后，遍历本轮 frontier/OPD rows 累计梯度，最后只做一次 `optimizer.step()`，避免小 batch 逐步更新导致方向被早期 batch 牵引。
2. **OPD loss 利用全错/弱信号样本**：从专家 rollout 中选正样本、当前策略负样本，对 calibration 中 raw reward 不足的样本施加 best-response / pairwise distillation，观察是否能把 task vector 系数从 `1/3` 推向更高能力区间。

## 公共设置

| 项 | 设置 |
|---|---|
| 初始化 | `ta_avgvec_c033333_hotpotqa_v1`，三专家系数从 `1/3` 起步 |
| prompt 数 | 48 |
| samples/prompt | 4 |
| rollout | vLLM，两卡分片，每张卡 24 prompts |
| update | HF logprob/backward，2 GPU `device_map=auto` |
| optimizer | SGD, momentum=0.8, persistent optimizer state |
| step scope | `optimizer_step_scope=epoch` |
| loss granularity | sequence |
| task normalize | 关闭，仅使用 centered advantage |
| frontier quota | tool/memory/code 各 24 |
| OPD replay | compact 21 rows，tool/memory/code 各 7 rows，每行 1 expert positive + 2 current negatives |

## 四路实验

| run | GPU | 参数化 | LR | OPD best-response | OPD pairwise | 状态 |
|---|---:|---|---:|---:|---:|---|
| `gc_opdc` | 0,1 | global-coefficient | 0.08 | 0.12 | 0.06 | iter3 update 运行中 |
| `gp_opdc` | 2,3 | global-parameter | 0.04 | 0.12 | 0.06 | iter3 完成，进入 iter4 |
| `gc_opdc_push` | 4,5 | global-coefficient | 0.12 | 0.18 | 0.09 | 已停止在 iter4 checkpoint，防止继续过冲 |
| `gp_opdc_push` | 6,7 | global-parameter | 0.06 | 0.18 | 0.09 | iter4 rollout 运行中 |
| `gc_opdc_guarded` | 4,5 | global-coefficient | 0.10 | 0.18 | 0.09 | replacement run，`MAX_COEFF_DELTA=0.35` |

## Iter 1 已完成结果

| run | rollout 时间 | update 时间 | kept frontier | updates | optimizer steps | code | memory | tool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gc_opdc` | 224.9s | 980.4s | 39 | 53 | 1 | 0.3591 | 0.3583 | 0.3312 |
| `gp_opdc` | 227.9s | 809.9s | 36 | 50 | 1 | 0.3469 global / 0.3469 mean | 0.3465 global / 0.3466 mean | 0.3319 global / 0.3319 mean |
| `gc_opdc_push` | 225.2s | 726.6s | 33 | 47 | 1 | 0.3978 | 0.3976 | 0.3249 |
| `gp_opdc_push` | 235.4s | 759.3s | 33 | 47 | 1 | 0.3655 global / 0.3657 mean | 0.3651 global / 0.3653 mean | 0.3293 global / 0.3293 mean |

## 初步判断

1. **OPD 对 task vector 有明确推动**：普通 `global-coefficient` 一轮把 code/memory 推到约 `0.359`，普通 `global-parameter` 推到约 `0.347`；push 版 `global-coefficient` 推到约 `0.398`，push 版 `global-parameter` 推到约 `0.366`。幅度随 LR/OPD 权重增强而增大。
2. **tool 被轻微压低**：两个 push run 的 tool 都从 `0.3333` 降到约 `0.325-0.329`。这需要在 iter2/iter3 观察是否只是 code/memory 恢复导致的短期再平衡，还是 tool 能力持续被牺牲。
3. **瓶颈在 update，不在 rollout**：两卡 vLLM rollout 约 225-235s；HF update 约 12-13 分钟。OPD pairwise 会额外引入多次正负样本 logprob backward，是当前主要耗时项。
4. **epoch-scope 确实避免了 batch 级“固定步长同涨”问题**：一轮只做一次 optimizer step，最终方向来自全轮 frontier + OPD 累积梯度；从结果看 code/memory/tool 已出现分化，不再是纯同向步长。

## Iter 2 Rollout 快照

使用 iter1 更新后的 gate 进行 rollout：

| run | iter1 overall | iter2 overall | tool | memory | code | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `gc_opdc` | 0.4391 | 0.5176 | 0.3915 -> 0.7344 | 0.4219 -> 0.3594 | 0.5039 -> 0.4590 | 整体上涨，主要来自 tool |
| `gp_opdc` | 0.4464 | 0.4409 | 0.5005 -> 0.5084 | 0.4219 -> 0.4062 | 0.4168 -> 0.4082 | 基本持平，信号弱 |
| `gc_opdc_push` | 0.3944 | 0.5493 | 0.4194 -> 0.9505 | 0.3750 -> 0.3281 | 0.3887 -> 0.3691 | 涨幅最大，但 tool 主导 |
| `gp_opdc_push` | 0.3497 | 0.4834 | 0.4222 -> 0.7072 | 0.2656 -> 0.3438 | 0.3613 -> 0.3992 | 三任务中 memory/code 也回升 |

当前结论：OPD 能让 overall reward 上升，但提升来源不均衡。`global-coefficient` 推系数最快，容易优先释放 tool；`global-parameter push` 较慢但 memory/code 的同步改善更健康，值得继续观察。

## Iter 2 Update 快照

| run | update 时间 | updates | loss normalizer | code | memory | tool |
|---|---:|---:|---:|---:|---:|---:|
| `gc_opdc` | 793.8s | 49 | 56 | 0.4077 | 0.4034 | 0.3261 |
| `gp_opdc` | 待补 | 48 | 55 | 0.3718 global / 0.3720 mean | 0.3716 global / 0.3718 mean | 0.3291 global / 0.3291 mean |
| `gc_opdc_push` | 711.3s | 40 | 47 | 0.5260 | 0.4965 | 0.3093 |
| `gp_opdc_push` | 692.3s | 44 | 51 | 0.4258 global / 0.4263 mean | 0.4208 global / 0.4213 mean | 0.3220 global / 0.3219 mean |

判断：push OPD 的系数在第二轮继续增长，说明 momentum + epoch 累积没有失效；但 tool 系数持续下降。下一步需要用 iter3 rollout 判断 tool reward 是否仍然稳定。如果 tool reward 仍高，说明较低 tool 系数已足够维持 tool 能力；如果 tool 开始掉，需要引入 tool retention 或降低 OPD 对 memory/code 的相对推动。

## Iter 3 Rollout 快照

| run | iter3 overall | tool | memory | code | 判断 |
|---|---:|---:|---:|---:|---|
| `gc_opdc_push` | 0.6822 | 0.9724 | 0.6562 | 0.4180 | 强正向，memory 开始显著回升 |
| `gp_opdc_push` | 0.5848 | 0.9848 | 0.4219 | 0.3477 | tool 稳定，memory 上升，code 回落 |

当前最强证据来自 `gc_opdc_push`：系数已经到 code/memory 约 `0.5`，iter3 reward 同时出现 tool 稳定高分与 memory 回升。这说明 OPD 不是只推系数，也开始改善 rollout reward。但 code 仍弱，需要后续检查 code 样本的 reward proxy 是否过严或 OPD 正样本覆盖不足。

## Iter 3 Update / Iter 4 Rollout 快照

| run | iter3 gate code | iter3 gate memory | iter3 gate tool | iter4 overall | iter4 tool | iter4 memory | iter4 code |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gp_opdc` | 0.4066 global | 0.4040 global | 0.3256 global | 运行中 | 运行中 | 运行中 | 运行中 |
| `gc_opdc_push` | 0.7139 | 0.5746 | 0.2867 | 0.6754 | 0.8738 | 0.7188 | 0.4336 |
| `gp_opdc_push` | 0.5131 global / 0.5140 mean | 0.4861 global / 0.4869 mean | 0.3114 global / 0.3113 mean | 运行中 | 运行中 | 运行中 | 运行中 |

判断：`gc_opdc_push` 已经进入用户希望观察的 `0.6-0.8` task-vector 区间。iter4 reward 没有崩，memory 继续提高，code 小幅提高；tool reward 从高峰回落但仍显著高于初始。`gp_opdc` 低强度版本也在推高 code/memory，但 iter3 rollout 主要仍是 tool 提升，memory/code 下降。后续 stop condition：若 tool gate 低于 `0.25` 或 overall reward 连续两轮下降，应停止对应 run 并保留 checkpoint 做评测。

已执行：`gc_opdc_push` 在 iter4 后停止，原因是 code gate 已达 `0.9456`，明显超过目标观察区间，tool gate 到 `0.2616`，接近 `0.25` 保护线。保留 checkpoint：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_opdcompact_push_sgd_m08_n48_i10_20260514_0154/iter_004/gate_updates.gates.json`

补充 replacement：在 4/5 卡启动 `gc_opdc_guarded`，run name 为 `qbank_c033333_gc_epoch_opdcompact_guarded_sgd_m08_n48_i10_20260514_0300`。该 run 保持 OPD push 强度不变，但将 LR 降到 `0.10`，并设置 `MAX_COEFF_DELTA=0.35`，目标是验证“有 OPD 推力但不让 global coefficient 过冲”的可行性。

## 03:05 Live Status

| run | latest gate | latest rollout | code | memory | tool | overall reward | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gc_opdc` | iter3 | iter4 | 0.4802 | 0.4619 | 0.3181 | 0.6129 | 温和上涨，继续 |
| `gp_opdc` | iter4 | iter4 | 0.4524 global / 0.4530 mean | 0.4425 global / 0.4431 mean | 0.3207 global / 0.3206 mean | 0.6410 | 稳定上涨，继续 |
| `gp_opdc_push` | iter4 | iter4 | 0.6243 global / 0.6258 mean | 0.5403 global / 0.5413 mean | 0.2978 global / 0.2976 mean | 0.6329 | 当前最稳 promising run，继续 |
| `gc_opdc_guarded` | 无 | iter1 | 0.3333 | 0.3333 | 0.3333 | 0.4061 | replacement 刚启动，等待 iter1 update |

当前判断：`gp_opdc_push` 比已经停止的 `gc_opdc_push` 更稳，code/memory 进入目标区间同时 tool gate 尚未跌破 `0.30` 太多，reward 没有崩。若后续 tool gate 低于 `0.25` 或 iter5/iter6 overall 连续下降，需要停止并保留 checkpoint；否则继续到 10 iter，若仍稳定再做 15/20 iter continuation。

## 03:12 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | trend |
|---|---:|---:|---:|---:|---:|---:|---|
| `gc_opdc` | iter4 | iter4 | 0.5742 | 0.5151 | 0.3075 | 0.6129 | 温和上升，继续 |
| `gp_opdc` | iter4 | iter5 | 0.4524 global / 0.4530 mean | 0.4425 global / 0.4431 mean | 0.3207 global / 0.3206 mean | 0.6067 | iter5 回落一次，继续观察 |
| `gp_opdc_push` | iter4 | iter5 | 0.6243 global / 0.6258 mean | 0.5403 global / 0.5413 mean | 0.2978 global / 0.2976 mean | 0.7032 | 当前最 promising，继续 |
| `gc_opdc_guarded` | 无 | iter1 | 0.3333 | 0.3333 | 0.3333 | 0.4061 | 等待 iter1 update |

`gp_opdc_push` iter5 reward 细分：tool `0.9809`，memory `0.7188`，code `0.4098`。这满足“overall 持续上涨 + task vector 系数分化上涨”的主要预期，当前优先保留到 10 iter。`gp_opdc` iter5 从 `0.6410` 回落到 `0.6067`，但不是连续回落，暂不停止。

## 03:20 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_opdc_push` | iter5 | iter5 | 0.7608 global / 0.7630 mean | 0.5702 global / 0.5714 mean | 0.2813 global / 0.2811 mean | 0.7032 | 主候选，继续但严密监控 tool |
| `gc_opdc` | iter4 | iter5 | 0.5742 | 0.5151 | 0.3075 | 0.6716 | 普通 global-coefficient 也稳定上升，继续 |
| `gp_opdc` | iter4 | iter5 | 0.4524 global / 0.4530 mean | 0.4425 global / 0.4431 mean | 0.3207 global / 0.3206 mean | 0.6067 | 一轮回落，继续观察 |
| `gc_opdc_guarded` | iter1 | iter1 | 0.3826 | 0.3815 | 0.3273 | 0.4061 | bounded 版本启动正常，未过冲 |

`gp_opdc_push` 已达到最符合预期的区间：code 接近 `0.76`，memory 约 `0.57`，tool 仍保持在 `0.28`。下一轮若 tool 低于 `0.25` 或 iter6 reward 明显下滑，应停止并保留 iter5 checkpoint；否则继续到 10 iter，并准备 continuation 到 15/20 iter。

已执行：`gp_opdc_push` 在 iter6 后停止。原因是 iter6 gate 变为 code `0.9117`、memory `0.5728`、tool `0.2664`，iter6 overall reward 从 iter5 的 `0.7032` 回落到 `0.6564`，且 tool reward 明显下降。保留 iter5 checkpoint 作为主候选：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_opdcompact_push_sgd_m08_n48_i10_20260514_0154/iter_005/gate_updates.gates.json`

补充 replacement：在 6/7 卡启动 `gp_opdc_push_guarded`，run name 为 `qbank_c033333_gp_epoch_opdcompact_push_guarded_sgd_m08_n48_i10_20260514_0327`。该 run 复用 `gp_opdc_push` 的 LR/OPD 强度，但设置 `MAX_COEFF_DELTA=0.45`，目标是保留 iter5 的好区间并避免 code 继续冲到 `0.9+`。

## 03:35 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | 动作 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gc_opdc` | iter6 | iter6 | 0.8249 | 0.5697 | 0.2783 | 0.7157 | 已停止，保留 checkpoint |
| `gp_opdc` | iter5 | iter6 | 0.5065 global / 0.5074 mean | 0.4799 global / 0.4807 mean | 0.3146 global / 0.3146 mean | 0.6884 | 继续 |
| `gc_opdc_guarded` | iter1 | iter2 | 0.3826 | 0.3815 | 0.3273 | 0.5292 | 继续 |
| `gp_opdc_push_guarded` | 无 | iter1 | 0.3333 | 0.3333 | 0.3333 | 0.4451 | 等待 iter1 update |

`gc_opdc` 在 iter6 达到目前最高 overall reward `0.7157`，但 code 已到 `0.8249`、tool 到 `0.2783`，继续更新很可能重复强 push 的过冲问题，因此停止并保留 checkpoint：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_opdcompact_sgd_m08_n48_i10_20260514_0154/iter_006/gate_updates.gates.json`

补充 continuation：在 0/1 卡启动 `gc_best_finetune`，run name 为 `qbank_c033333_gc_best_iter6_finetune_sgd_m08_n48_i10_20260514_0336`。该 run 从 `gc_opdc` iter6 checkpoint 继续，降低为 `LR=0.025`、`OPD=0.06/0.03`，并设置 `MAX_COEFF_DELTA=0.05`（相对 iter6 checkpoint），目标是在不继续压 tool 的情况下验证 reward 能否稳定在 `0.70+`。

## 03:45 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_opdc` | iter6 | iter6 | 0.5678 global / 0.5690 mean | 0.5096 global / 0.5105 mean | 0.3076 global / 0.3074 mean | 0.6884 | 温和稳定，继续 |
| `gc_opdc_guarded` | iter2 | iter3 | 0.4830 | 0.4658 | 0.3154 | 0.6253 | bounded 生效，继续 |
| `gp_opdc_push_guarded` | 无 | iter1 | 0.3333 | 0.3333 | 0.3333 | 0.4451 | 等待 iter1 update |
| `gc_best_finetune` | 无 | iter1 | 0.8249 起点 | 0.5697 起点 | 0.2783 起点 | 0.6905 | 续训稳定性初步成立 |

`gc_best_finetune` 第一轮 rollout：tool `0.9184`，memory `0.7344`，code `0.4188`。这说明从 `gc_opdc` iter6 最佳点小步续训并没有立即崩，且 memory 保持高位；等待第一轮 update 后判断是否可以作为 15/20 iter continuation 的主线。

## 03:55 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_opdc` | iter6 | iter7 | 0.5678 global / 0.5690 mean | 0.5096 global / 0.5105 mean | 0.3076 global / 0.3074 mean | 0.7063 | 当前最稳长跑线，继续到 10 |
| `gc_opdc_guarded` | iter2 | iter3 | 0.4830 | 0.4658 | 0.3154 | 0.6253 | 有效但 code 偏弱，继续 |
| `gp_opdc_push_guarded` | iter1 | iter2 | 0.3649 global / 0.3651 mean | 0.3652 global / 0.3653 mean | 0.3296 global / 0.3296 mean | 0.4936 | 初期上升，继续 |
| `gc_best_finetune` | iter1 | iter1 | 0.8310 | 0.5670 | 0.2784 | 0.6905 | 稳定续训，等待 iter2 rollout |

重要观察：低强度 `gp_opdc` 没有强 push 的过冲问题，iter7 overall 已到 `0.7063`，且 tool gate 仍在 `0.30+`。如果它到 10 iter 仍不崩，优先对这条做 15/20 iter continuation。

## 04:05 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_opdc` | iter7 | iter7 | 0.6365 global / 0.6380 mean | 0.5305 global / 0.5315 mean | 0.2995 global / 0.2993 mean | 0.7063 | 稳定主线，继续 |
| `gc_opdc_guarded` | iter3 | iter4 | 0.6307 | 0.5486 | 0.2979 | 0.6992 | bounded global-coeff 成功，继续 |
| `gp_opdc_push_guarded` | iter1 | iter2 | 0.3649 global / 0.3651 mean | 0.3652 global / 0.3653 mean | 0.3296 global / 0.3296 mean | 0.4936 | 仍在早期，继续 |
| `gc_best_finetune` | iter2 | iter2 | 0.8418 | 0.5624 | 0.2788 | 0.6929 | 从最佳点稳定，没有继续崩 |

当前有两个可作为明早主结果的候选：`gp_opdc` 和 `gc_opdc_guarded`。二者都在 overall `0.70` 左右，tool gate 仍接近 `0.30`，没有强 push 的过冲；优先继续到 10 iter。`gc_best_finetune` 证明高点小步续训可稳定，但没有明显超过原 iter6 best。

## 04:15 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | 动作 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_opdc` | iter7 | iter8 | 0.6365 global / 0.6380 mean | 0.5305 global / 0.5315 mean | 0.2995 global / 0.2993 mean | 0.6645 | 一轮回落，继续观察 |
| `gc_opdc_guarded` | iter4 | iter4 | 0.8283 | 0.5893 | 0.2745 | 0.6992 | 已停止，保留 checkpoint |
| `gp_opdc_push_guarded` | iter2 | iter3 | 0.4223 global / 0.4228 mean | 0.4198 global / 0.4203 mean | 0.3230 global / 0.3229 mean | 0.5877 | 继续 |
| `gc_best_finetune` | iter2 | iter3 | 0.8418 | 0.5624 | 0.2788 | 0.7086 | 当前最佳稳定续训，继续 |

已执行：`gc_opdc_guarded` 停止在 iter4，保留 checkpoint：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_opdcompact_guarded_sgd_m08_n48_i10_20260514_0300/iter_004/gate_updates.gates.json`

注意：`MAX_COEFF_DELTA` 在当前 bake loop 中不是“全 run 相对 1/3 的累计硬上限”。每轮 update 的 `init_gate_checkpoint` 会变成上一轮 gate，所以它实际约束的是**单轮相对上一 checkpoint 的最大位移**。因此 `gc_opdc_guarded` 累计到 code `0.8283` 不是投影代码失效，而是该参数无法防止多轮累计漂移；真正的累计保护需要另加“相对原始 1/3 anchor”的 global cap。

## 04:25 Live Status

| run | latest gate | latest rollout | code | memory | tool | latest overall | 动作 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_opdc` | iter8 | iter9 | 0.7117 global / 0.7136 mean | 0.5428 global / 0.5439 mean | 0.2906 global / 0.2903 mean | 0.7189 | 当前最佳，继续到 10 |
| `gp_opdc_push_guarded` | iter2 | iter3 | 0.4223 global / 0.4228 mean | 0.4198 global / 0.4203 mean | 0.3230 global / 0.3229 mean | 0.5877 | 继续观察 |
| `gc_best_finetune` | iter3 | iter4 | 0.8572 | 0.5571 | 0.2783 | 0.6681 | 已停止，保留 iter3 |

当前主结果是 `gp_opdc`：iter9 overall `0.7189`，tool `0.9809`，memory `0.7188`，code `0.4570`，且 gate 没有明显过冲。`gc_best_finetune` iter4 从 `0.7086` 回落到 `0.6681`，code reward 下降明显，因此停止并保留 iter3 checkpoint：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_best_iter6_finetune_sgd_m08_n48_i10_20260514_0336/iter_003/gate_updates.gates.json`

## 04:22 2x2 Matrix Relaunch

为严格回答“epoch-scope 梯度累计”和“在此基础上加入 OPD 是否更好”，重新整理为 2x2 对照。每个实验占 2 张卡；参数除 gate 形态和 OPD 开关外保持一致：`NUM_PROMPTS=48`、`SAMPLES_PER_PROMPT=4`、`NUM_ITERS=10`、`OPTIMIZER_STEP_SCOPE=epoch`、`LOSS_GRANULARITY=sequence`、`OPTIMIZER=sgd`、`SGD_MOMENTUM=0.8`、`LR=0.04`、`PPO_LOSS_WEIGHT=6.0`、`PRIOR_LOSS_WEIGHT=0.0`、`MAX_COEFF_DELTA=1.0`、`TASK_NORMALIZE_ADVANTAGES=0`、`ADVANTAGE_NORMALIZATION=centered`、`FRONTIER_ORDER=task-interleaved`。

| cell | GPU | gate | OPD | run dir |
|---|---:|---|---|---|
| epoch-only global-coeff | 6,7 | `global-coefficient` | off | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_noopd_sgd_m08_n48_i10_20260514_0440` |
| epoch-only 588 common+residual | 4,5 | `global-parameter` | off | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_noopd_sgd_m08_n48_i10_20260514_0440` |
| epoch+OPD global-coeff | 0,1 | `global-coefficient` | `0.12/0.06` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_opdcompact_sgd_m08_n48_i10_20260514_0440` |
| epoch+OPD 588 common+residual | 2,3 | `global-parameter` | `0.12/0.06` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_20260514_0154` |

已停止 `gp_opdc_push_guarded`，原因是它与 `gp_opd` 属于同一 OPD + global-parameter 类别，但早期 reward 明显更弱；释放 6/7 用于 epoch-only global-coeff 对照。新矩阵前端监控端口：`8766`，tmux session 为 `opvec_monitor_epoch_opd_0440_view`。保留旧前端 `8765` 用于昨晚四策略曲线。

OPD 数据固定为：

`/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill_balanced21_compact_pos1_neg2_seed20260514.jsonl`

该文件为 compact distillation rows：每个 task 7 条，共 21 条；每条含 1 个 expert-positive 与 2 个 current-negative。当前 OPD 只作为辅助 loss 进入 update，不改变 rollout 数据分布。

## 04:35 gp_opd Finished

`global-parameter + OPD` 已完成 10 iter。最佳点不是最终 iter10，而是 iter9：

| iter | overall | tool | memory | code | global tool | global memory | global code | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | 0.7189 | 0.9809 | 0.7188 | 0.4570 | 0.2812 | 0.5482 | 0.7914 | 当前 best |
| 10 | 0.7032 | 0.9806 | 0.7031 | 0.4258 | 0.2715 | 0.5483 | 0.8735 | code 继续过推，overall 回落 |

结论：`global-parameter + OPD` 能从 `1/3` 推到 high-reward 区间，但 OPD/GRPO 仍会沿 code 方向累计过推；后续不要继续这条 run。iter9 rollout 使用的是 iter8 gate，因此保留 iter8 checkpoint 作为当前 best：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_20260514_0154/iter_008/gate_updates.gates.json`

## 04:52 First Update Comparison

三条新实验的 iter1 update 已完成。核心结论：epoch-scope 本身只解决“小 batch 连续改 gate”的噪声问题，但如果没有 OPD，全局信号在 `1/3` 起点仍高度抵消，gate 几乎不动；OPD 会立刻提供非零方向。

| run | iter1 rollout overall | tool | memory | code | frontier tool/mem/code | OPD rows | gate after iter1 | gate_delta_max |
|---|---:|---:|---:|---:|---|---:|---|---:|
| `gc_noopd` | 0.2972 | 0.3421 | 0.2656 | 0.2840 | 13/11/9 | 0 | tool 0.3334, memory 0.3332, code 0.3335 | 0.00017 |
| `gp_noopd` | 0.3870 | 0.4343 | 0.4062 | 0.3203 | 12/14/11 | 0 | global tool 0.3330, memory 0.3333, code 0.3333 | 0.00034 |
| `gc_opd` | 0.3741 | 0.4290 | 0.2656 | 0.4277 | 13/13/12 | 21 | tool 0.3315, memory 0.3467, code 0.3464 | 0.01337 |

判断：`no-OPD` 的两条可以作为反证，即“只把 optimizer step 改成 epoch-scope 不足以快速推动系数”；`gc_opd` 是当前最值得继续验证的 OPD 线。下一步看 iter2 rollout 是否把 reward 从 `0.3741` 推高，如果没有，说明 OPD 只改变 gate 但没有改善当前 qbank reward。

## Eval6 Formal Evaluation: OPD Best Checkpoints

评测完成时间：2026-05-14

评测对象：

| model | 训练候选 | 实际评测模型路径 | 说明 |
|---|---|---|---|
| `opvec-gp-opd-best-iter9` | `global-parameter + OPD` iter9 rollout best | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_20260514_0154/iter_009/baked_policy` | iter9 baked policy 对应上一轮 gate，用于验证 588 common+residual 候选 |
| `opvec-gc-opd-best-iter8` | `global-coefficient + OPD` iter8 rollout best | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_opdcompact_sgd_m08_n48_i10_20260514_0440/iter_008/baked_policy` | iter8 baked policy 对应上一轮 gate，用于验证三系数候选 |

### Tool / BFCL

| model | live_parallel | live_parallel_multiple | parallel | parallel_multiple | weighted overall |
|---|---:|---:|---:|---:|---:|
| `opvec-gp-opd-best-iter9` | 11/16 = 0.6875 | 16/24 = 0.6667 | 181/200 = 0.9050 | 175/200 = 0.8750 | 383/440 = 0.8705 |
| `opvec-gc-opd-best-iter8` | 11/16 = 0.6875 | 16/24 = 0.6667 | 181/200 = 0.9050 | 175/200 = 0.8750 | 383/440 = 0.8705 |

结论：两个候选 Tool 能力打平，且未出现格式崩坏；主要错误仍是 `cannot_find_match` / `wrong_count`，不是空输出或大规模解析失败。

### Memory / HotpotQA

| model | eval_50 F1 / EM | eval_100 F1 / EM | eval_qa_1_32768 F1 / EM | eval_qa_1_65536 F1 / EM | avg F1 | avg EM | avg sub-EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `opvec-gp-opd-best-iter9` | 0.7414 / 0.6016 | 0.7741 / 0.6250 | 0.7486 / 0.6406 | 0.7956 / 0.6875 | 0.7649 | 0.6387 | 0.7969 |
| `opvec-gc-opd-best-iter8` | 0.7249 / 0.5938 | 0.7326 / 0.5859 | 0.7312 / 0.5938 | 0.7559 / 0.6328 | 0.7361 | 0.6016 | 0.7637 |

结论：`global-parameter + OPD` 在四个 memory split 上稳定优于 `global-coefficient + OPD`，avg F1 高 `+0.0288`，avg EM 高 `+0.0371`。

### Code / CURE Full

评测协议：`LiveBench` + `LiveCodeBench`，`k_case=4`，`k_code=4`，`max_generation_token=10000`，`max_model_len=32768`，`temp=1.0`。

| model | dataset | code_acc | code_accumulate_acc | estimated_unit_test_acc | estimated_unit_test_accumulate_acc | BoN(4,4) acc / accumulate |
|---|---|---:|---:|---:|---:|---:|
| `opvec-gp-opd-best-iter9` | LiveBench | 0.3828 | 0.5051 | 0.4945 | 0.5390 | 0.4453 / 0.5798 |
| `opvec-gp-opd-best-iter9` | LiveCodeBench | 0.3146 | 0.4590 | 0.4515 | 0.4841 | 0.3836 / 0.5390 |
| `opvec-gc-opd-best-iter8` | LiveBench | 0.3730 | 0.4907 | 0.4104 | 0.4436 | 0.4375 / 0.5710 |
| `opvec-gc-opd-best-iter8` | LiveCodeBench | 0.3165 | 0.4643 | 0.4513 | 0.4900 | 0.3796 / 0.5388 |

平均结果：

| model | avg code_acc | avg code_accumulate_acc | avg estimated_unit_test_acc | avg estimated_unit_test_accumulate_acc | avg BoN(4,4) acc | avg BoN(4,4) accumulate |
|---|---:|---:|---:|---:|---:|---:|
| `opvec-gp-opd-best-iter9` | 0.3487 | 0.4821 | 0.4730 | 0.5115 | 0.4144 | 0.5594 |
| `opvec-gc-opd-best-iter8` | 0.3448 | 0.4775 | 0.4309 | 0.4668 | 0.4086 | 0.5549 |

结论：Code 上 `global-parameter + OPD` 小幅优于 `global-coefficient + OPD`，主要优势来自 LiveBench estimated unit-test 指标；LiveCodeBench 两者接近。

### Overall Decision

| model | Tool weighted | Memory avg F1 | Code avg code_acc | Code avg unit-test acc | 结论 |
|---|---:|---:|---:|---:|---|
| `opvec-gp-opd-best-iter9` | 0.8705 | 0.7649 | 0.3487 | 0.4730 | 当前正式评测主候选 |
| `opvec-gc-opd-best-iter8` | 0.8705 | 0.7361 | 0.3448 | 0.4309 | 三系数简化 baseline，能力弱于 gp |

最终判断：昨晚 proxy reward 最高的两个 OPD checkpoint 中，`global-parameter + OPD` 是更强的正式候选。它在 Tool 不损失的前提下，Memory 明显优于 `global-coefficient`，Code 也略优；因此后续论文主线应优先报告 `opvec-gp-opd-best-iter9`，`opvec-gc-opd-best-iter8` 作为低参数量对照。

Artifact summary:

- Tool:
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/opvec-gp-opd-best-iter9/tool/summary.json`
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/opvec-gc-opd-best-iter8/tool/summary.json`
- Memory:
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/opvec-gp-opd-best-iter9/eval6-20260502-125748/summary.json`
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/opvec-gc-opd-best-iter8/eval6-20260502-125748/summary.json`
- Code:
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/opvec-gp-opd-best-iter9/eval6-20260502-125748/summary.json`
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/opvec-gc-opd-best-iter8/eval6-20260502-125748/summary.json`

## 05:00 Iter2 Rollout

| run | iter1 overall | iter2 overall | delta | iter2 tool | iter2 memory | iter2 code | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| `gc_noopd` | 0.2972 | 0.3738 | +0.0766 | 0.4008 | 0.2969 | 0.4238 | gate 几乎没动，主要可能是采样/seed 波动 |
| `gp_noopd` | 0.3870 | 0.4082 | +0.0212 | 0.5156 | 0.2812 | 0.4277 | gate 几乎没动，memory 掉明显 |
| `gc_opd` | 0.3741 | 0.4266 | +0.0525 | 0.5318 | 0.3125 | 0.4355 | 最高 overall，且 gate 已有实质位移 |

下一步策略：等待 iter2 update。若 `no-OPD` 仍 `gate_delta_max < 1e-3`，停止两条 no-OPD，对照结论已足够；保留 `gc_opd` 继续到至少 5 iter，看 OPD 是否能稳定把 reward 推过 `0.5`。

## 05:08 gc_noopd Stopped

`gc_noopd` iter2 update 后 gate 仍几乎不动：tool `0.33396`、memory `0.33296`、code `0.33363`，`gate_delta_max=0.00052`。该结果说明 `global-coefficient` 在 no-OPD、epoch-scope 下无法获得足够推动信号；已停止该 run，释放 GPU 6/7。保留目录用于审计：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_noopd_sgd_m08_n48_i10_20260514_0440`

## 05:14 gp_noopd Stopped

`gp_noopd` iter2 update 后同样几乎不动：global tool `0.333785`、memory `0.333321`、code `0.333427`，effective mean 也基本一致，`gate_delta_max=0.00079`。该结果说明 `global-parameter/common+residual` 形态在 no-OPD、epoch-scope 下也没有足够推动信号；已停止该 run。保留目录用于审计：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_noopd_sgd_m08_n48_i10_20260514_0440`

## 05:20 gc_opd Iter2 Update

`gc_opd` 第二次 update 仍然健康，继续作为主线。

| iter | rollout overall | tool | memory | code | gate tool | gate memory | gate code | grad_norm | gate_delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3741 | 0.4290 | 0.2656 | 0.4277 | 0.3315 | 0.3467 | 0.3464 | 0.4704 | 0.0134 |
| 2 | 0.4266 | 0.5318 | 0.3125 | 0.4355 | 0.3293 | 0.3713 | 0.3703 | 0.4821 | 0.0246 |

判断：OPD 正在把 memory/code 从 `1/3` 推出，reward 同步上升，tool gate 只小幅下降且仍在 `0.329`。继续到 5-6 iter；停止条件为 tool gate `<0.30`、overall 连续两轮回落，或 code/memory 单边过推导致 task reward 明显失衡。

## 05:27 gc_opd Iter3 Rollout

`gc_opd` iter3 rollout 继续上升：

| iter | overall | tool | memory | code | frontier tool/mem/code | all-success | all-failure |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 0.3741 | 0.4290 | 0.2656 | 0.4277 | 13/13/12 | 0/0/1 | 8/3/7 |
| 2 | 0.4266 | 0.5318 | 0.3125 | 0.4355 | 12/13/12 | 2/0/2 | 6/3/6 |
| 3 | 0.4895 | 0.7588 | 0.2656 | 0.4441 | 8/9/14 | 7/0/0 | 4/7/4 |

解释：overall 已接近 0.5，但 iter3 的增益主要来自 tool；memory 没跟上，仍是 0.2656，且 memory all-failure 增加到 7。继续等待 iter3 update，重点看 gate 是否仍然主要推 memory/code，还是开始因为 tool frontier 收缩而进一步牺牲 tool。

## 05:37 gc_opd Iter3 Update

| iter | rollout overall | gate tool | gate memory | gate code | grad_norm | gate_delta | frontier tool/mem/code |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.3741 | 0.3315 | 0.3467 | 0.3464 | 0.4704 | 0.0134 | 13/13/12 |
| 2 | 0.4266 | 0.3293 | 0.3713 | 0.3703 | 0.4821 | 0.0246 | 12/13/12 |
| 3 | 0.4895 | 0.3263 | 0.4040 | 0.4043 | 0.4980 | 0.0340 | 8/9/14 |

判断：仍可继续。tool gate 只从 `0.333` 降到 `0.326`，memory/code 已推到 `0.404`，说明 OPD 能在 global-coefficient 下快速把 task vector 从 `1/3` 拉出。但 memory reward 尚未同步改善，iter4 rollout 是关键验证点。

## 05:45 gc_opd Iter4 Rollout

`gc_opd` iter4 overall 继续上升到 `0.5543`。

| iter | overall | tool | memory | code | frontier tool/mem/code | all-success tool/mem/code | all-failure tool/mem/code |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 0.3741 | 0.4290 | 0.2656 | 0.4277 | 13/13/12 | 0/0/1 | 8/3/7 |
| 2 | 0.4266 | 0.5318 | 0.3125 | 0.4355 | 12/13/12 | 2/0/2 | 6/3/6 |
| 3 | 0.4895 | 0.7588 | 0.2656 | 0.4441 | 8/9/14 | 7/0/0 | 4/7/4 |
| 4 | 0.5543 | 0.9736 | 0.3125 | 0.3770 | 3/9/15 | 11/2/0 | 2/5/5 |

解释：这条 OPD 线已经证明能把 overall reward 从 `0.3741` 推到 `0.5543`。但 tool 已接近饱和，后续梯度会主要来自 code/memory；如果 iter4 update 继续压 tool 或 code 明显掉，需要准备降低 OPD pairwise 或加入 retention。

## 05:57 gc_opd Iter4 Update

| iter | rollout overall | gate tool | gate memory | gate code | grad_norm | gate_delta | frontier tool/mem/code |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.3741 | 0.3315 | 0.3467 | 0.3464 | 0.4704 | 0.0134 | 13/13/12 |
| 2 | 0.4266 | 0.3293 | 0.3713 | 0.3703 | 0.4821 | 0.0246 | 12/13/12 |
| 3 | 0.4895 | 0.3263 | 0.4040 | 0.4043 | 0.4980 | 0.0340 | 8/9/14 |
| 4 | 0.5543 | 0.3221 | 0.4408 | 0.4483 | 0.4977 | 0.0440 | 3/9/15 |

判断：继续等 iter5 rollout。虽然 overall 仍升，但 gate 已开始明显偏向 memory/code，tool 剩余 frontier 很少；若 iter5 overall 不涨或 code/memory 仍未改善，则停在 iter4/iter5 最高点，不继续硬推。

## 06:05 gc_opd Iter5 Rollout

`gc_opd` iter5 rollout 继续升到 `0.5794`，这是当前该线最高值；对应使用的是 iter4 gate checkpoint。

| iter | overall | tool | memory | code | frontier tool/mem/code | all-success tool/mem/code | all-failure tool/mem/code |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 0.3741 | 0.4290 | 0.2656 | 0.4277 | 13/13/12 | 0/0/1 | 8/3/7 |
| 2 | 0.4266 | 0.5318 | 0.3125 | 0.4355 | 12/13/12 | 2/0/2 | 6/3/6 |
| 3 | 0.4895 | 0.7588 | 0.2656 | 0.4441 | 8/9/14 | 7/0/0 | 4/7/4 |
| 4 | 0.5543 | 0.9736 | 0.3125 | 0.3770 | 3/9/15 | 11/2/0 | 2/5/5 |
| 5 | 0.5794 | 0.9882 | 0.3906 | 0.3594 | 2/11/11 | 12/1/0 | 2/4/8 |

判断：OPD 线已明确验证“全错/低信号样本 + expert-positive OPD”能持续推动 overall reward。风险也很清楚：tool 已饱和，code reward 下降且 all-failure 增到 8。继续让 iter5 update 完成，但若 iter6 rollout 不提升或 code 继续掉，保留 iter4 gate 为该线 best checkpoint。

## 06:17 gc_opd Iter5 Update

| iter | rollout overall | gate tool | gate memory | gate code | grad_norm | gate_delta | frontier tool/mem/code |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.5543 | 0.3221 | 0.4408 | 0.4483 | 0.4977 | 0.0440 | 3/9/15 |
| 5 | 0.5794 | 0.3168 | 0.4734 | 0.5016 | 0.4609 | 0.0533 | 2/11/11 |

判断：tool 仍高于 `0.30`，因此继续等 iter6 rollout。风险加大：code coefficient 已过 `0.50`，但 code reward 在 iter5 只有 `0.3594`，说明继续推 code 未必改善 code proxy。iter6 若不升，应停止并保留 iter4 或 iter5 gate。

## 06:25 gc_opd Iter6 Rollout

iter6 rollout 大幅上升到 `0.6420`，memory 明显恢复，说明 iter5 gate 没有过推。

| iter | overall | tool | memory | code | frontier tool/mem/code | all-success tool/mem/code | all-failure tool/mem/code |
|---:|---:|---:|---:|---:|---|---|---|
| 4 | 0.5543 | 0.9736 | 0.3125 | 0.3770 | 3/9/15 | 11/2/0 | 2/5/5 |
| 5 | 0.5794 | 0.9882 | 0.3906 | 0.3594 | 2/11/11 | 12/1/0 | 2/4/8 |
| 6 | 0.6420 | 0.9768 | 0.5625 | 0.3867 | 1/9/12 | 12/4/2 | 3/3/6 |

判断：继续到 iter7。当前 best 可审计 checkpoint 是 iter5 gate：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_opdcompact_sgd_m08_n48_i10_20260514_0440/iter_005/gate_updates.gates.json`

## 06:39 gc_opd Iter6 Update

| iter | rollout overall | gate tool | gate memory | gate code | grad_norm | gate_delta | frontier tool/mem/code |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.5794 | 0.3168 | 0.4734 | 0.5016 | 0.4609 | 0.0533 | 2/11/11 |
| 6 | 0.6420 | 0.3103 | 0.5023 | 0.5628 | 0.4735 | 0.0612 | 1/9/12 |

判断：iter6 update 后 tool gate 接近 `0.30` 阈值，memory/code 继续上升。允许 iter7 rollout 验证一次，但暂不承诺继续 update；若 iter7 reward 不升或 tool/code 退化，停止该 run。

## 06:48 gc_opd Iter7 Rollout

iter7 rollout 继续上升到 `0.6950`，且 memory/code 同时改善。

| iter | overall | tool | memory | code | frontier tool/mem/code | all-success tool/mem/code | all-failure tool/mem/code |
|---:|---:|---:|---:|---:|---|---|---|
| 5 | 0.5794 | 0.9882 | 0.3906 | 0.3594 | 2/11/11 | 12/1/0 | 2/4/8 |
| 6 | 0.6420 | 0.9768 | 0.5625 | 0.3867 | 1/9/12 | 12/4/2 | 3/3/6 |
| 7 | 0.6950 | 0.9768 | 0.6250 | 0.4832 | 1/7/12 | 12/6/2 | 3/3/5 |

当前 best checkpoint 是 iter6 gate：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gc_epoch_opdcompact_sgd_m08_n48_i10_20260514_0440/iter_006/gate_updates.gates.json`

判断：允许 iter7 update 完成，但以 tool gate `0.30` 作为硬停止线；如果 update 后 tool 低于 `0.30`，停止并保留 iter6 gate。

## 06:57 gc_opd Iter7 Update

| iter | rollout overall | gate tool | gate memory | gate code | grad_norm | gate_delta | frontier tool/mem/code |
|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 0.6420 | 0.3103 | 0.5023 | 0.5628 | 0.4735 | 0.0612 | 1/9/12 |
| 7 | 0.6950 | 0.3028 | 0.5244 | 0.6311 | 0.4875 | 0.0683 | 1/7/12 |

判断：tool gate 已贴近 `0.30`，memory/code 继续上涨。允许 iter8 rollout 做最后验证；不建议继续多轮硬推，除非 iter8 明显提升且 tool 仍不崩。

## 07:04 Final Stop

`gc_opd` iter8 rollout 达到 `0.7051` 后停止。iter8 rollout 使用的是 iter7 gate；iter7 gate 的 tool 已到 `0.3028`，继续 update 大概率破 `0.30`，因此不再硬推。

| run | best rollout | overall | tool | memory | code | best gate checkpoint | gate tool | gate memory | gate code |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `gp_opd` | iter9 | 0.7189 | 0.9809 | 0.7188 | 0.4570 | `iter_008/gate_updates.gates.json` | 0.2906 | 0.5428 | 0.7117 |
| `gc_opd` | iter8 | 0.7051 | 0.9689 | 0.6406 | 0.5059 | `iter_007/gate_updates.gates.json` | 0.3028 | 0.5244 | 0.6311 |
| `gp_noopd` | iter2 | 0.4082 | 0.5156 | 0.2812 | 0.4277 | `iter_001/gate_updates.gates.json` | 0.3330 | 0.3333 | 0.3333 |
| `gc_noopd` | iter2 | 0.3738 | 0.4008 | 0.2969 | 0.4238 | `iter_001/gate_updates.gates.json` | 0.3334 | 0.3332 | 0.3335 |

最终判断：

1. `epoch-scope` 解决了小 batch 连续 step 的方向抖动问题，但**单独使用不够**：两条 no-OPD 对照两轮后 gate 仍停在 `1/3` 附近，`gate_delta_max < 1e-3`。
2. OPD 是本轮有效信号的关键：`gc_opd` 从 overall `0.3741` 连续推到 `0.7051`，且 global coefficients 从 `1/3` 变为 tool `0.3028`、memory `0.5244`、code `0.6311`。
3. `gp_opd` 上限最高但更容易过推：best rollout `0.7189`，但 tool gate 已低于 `0.30`，code gate 已到 `0.71+`；这说明 common+residual 588 参数更灵活，也更需要 cumulative cap / retention。
4. `gc_opd` 更适合作为论文故事的主证据：只有 3 个 direct coefficients，仍能靠 OPD+GRPO 自动发现从 `1/3` 推向高 reward 区间；趋势清楚、可解释性强。

所有训练进程已停止，仅保留监控前端进程 `opvec_monitor_epoch_opd_0440_view`。

## 继续监控标准

1. 对 2x2 矩阵优先比较同参数形态下 OPD on/off 的 reward slope，而不是单轮最高点。
2. 若 OPD on 的 overall 上升更快但 tool gate 被压到 `0.25` 以下，需要降 `OPD_PAIRWISE_LOSS_WEIGHT` 或加入 tool retention。
3. 若 epoch-only 的 gate 只推某一任务且其他任务掉明显，说明 frontier 信号仍偏单任务，需要调整 qbank composition 或 OPD/retention。
4. 后续报告必须同时记录：每 iter 的 rollout reward 分任务均值、kept frontier、gate coefficient、rollout/update/bake 时间。
