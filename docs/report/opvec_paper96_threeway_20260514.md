# OP-VEC Paper96 Three-Way Run 2026-05-14

## 目标

在不占用 eval harness 的 GPU2/3 前提下，用剩余六张卡并行启动三组 96-prompt 训练实验。目标不是最大化训练集 proxy reward，而是在保持 Tool、Memory、Code 三种能力不被破坏的情况下，找到可送入正式评测的强 gate checkpoint。

## 数据

- 训练 prompt manifest: `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl`
- prompt 组成: `tool=32, memory=32, code=32`
- OPD distill manifest: `/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill_balanced21_paperfix_rewardtrain_len_seed20260514.jsonl`
- OPD 组成: 每任务 7 rows，每 row 为 1 expert positive + 2 current negatives
- OPD 修正: 所有样本显式写入 `reward_train`，positive 为 `1.0`，negative 为 `0.0`；重算 `length`，避免 length-normalized OPD 尺度异常。
- 数据准备 summary: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_inputs_seed20260514.summary.json`

训练 reward 统计口径保持为 rollout 全部样本，不只看 frontier。Loss 口径中 frontier、retention、OPD 分别限额以保持三任务平衡。

## 三组实验

| Run | GPU | Gate | OPD | Retention | 作用 |
|---|---|---|---|---|---|
| A | 0,1 | `global-coefficient` | on | on | 主实验，3 个直接系数 |
| B | 4,5 | `global-coefficient` | off | on | OPD ablation |
| C | 6,7 | `global-parameter` | on | on | 588 参数容量对照 |

GPU2/3 保留给 eval harness。

## 共享训练设置

- `NUM_ITERS=8`
- `NUM_PROMPTS=96`
- `SAMPLES_PER_PROMPT=4`
- `INIT_VALUE=0.3333333333333333`
- `OPTIMIZER_STEP_SCOPE=epoch`
- `UPDATE_BATCH_SIZE=4`
- `LOSS_GRANULARITY=sequence`
- `STORE_TOKEN_LOGPROBS=0`
- `ADVANTAGE_NORMALIZATION=centered`
- `TASK_NORMALIZE_ADVANTAGES=0`
- `USE_FRONTIER_WEIGHT=0`
- `FRONTIER_ORDER=task-interleaved`
- `FRONTIER_TOOL_QUOTA=32`
- `FRONTIER_MEMORY_QUOTA=32`
- `FRONTIER_CODE_QUOTA=32`
- `MAX_FRONTIER_ROWS_PER_TASK=32`

## Loss 与约束

- `PPO_LOSS_WEIGHT=6.0`
- `OPD_LOSS_WEIGHT=0.12` for A/C
- `OPD_PAIRWISE_LOSS_WEIGHT=0.06` for A/C
- `MAX_OPD_DISTILL_ROWS=21`
- `MAX_OPD_PAIRWISE_PAIRS_PER_ROW=2`
- `OPD_POSITIVE_REWARD_THRESHOLD=1.0`
- `USE_RETENTION=1`
- `RETENTION_LOSS_WEIGHT=0.03`
- `MAX_RETENTION_ROWS_PER_TASK=8`
- `MAX_RETENTION_ROWS=24`
- `LENGTH_NORMALIZE_LOGPROB=1`
- `LENGTH_NORMALIZE_POLICY_LOGPROB=1`
- `PRIOR_LOSS_WEIGHT=0.005`
- `MAX_COEFF_DELTA=0.40`

## 生成长度

- `MAX_MODEL_LEN=16384`
- `MAX_LOGPROB_TOKENS=12288`
- `TOOL_MAX_NEW_TOKENS=768`
- `MEMORY_UPDATE_MAX_NEW_TOKENS=1536`
- `MEMORY_FINAL_MAX_NEW_TOKENS=768`
- `CODE_MAX_NEW_TOKENS=2048`

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
bash skill/command/run_paper96_threeway_20260514.sh
```

如需固定 run tag:

```bash
RUN_TAG=20260514_paper96_i8 bash skill/command/run_paper96_threeway_20260514.sh
```

## 监控

默认前端端口:

```text
http://127.0.0.1:8768
```

SSH tunnel:

```bash
ssh -L 8768:127.0.0.1:8768 <server>
```

每个 run 的 `run.log`、`rollouts.summary.json`、`gate_updates.summary.json`、`gate_updates.jsonl`、`strategy_summary.json` 都保留在对应 run dir。

## 启动记录

启动时间：2026-05-14 11:24 Asia/Shanghai

启动命令：

```bash
RUN_TAG=20260514_paper96_i8 MONITOR_PORT=8768 \
  bash skill/command/run_paper96_threeway_20260514.sh
```

Run directories:

- A: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_A_gc_opd_i8_20260514_paper96_i8`
- B: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_B_gc_noopd_i8_20260514_paper96_i8`
- C: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_C_gp_opd_i8_20260514_paper96_i8`

Initial sanity check:

- `bash -n` passed for both launchers.
- `py_compile` passed for data prep, monitor, bake-vLLM loop, and update script.
- Iter1 bake completed for all three runs in about 72s.
- Iter1 rollout started on GPU0/1/4/5/6/7; GPU2/3 were not used by the new training jobs.
- vLLM loaded successfully with FlashAttention 3; each rollout GPU used about 67-68GB during generation.
- At first check, each run had completed `10/48` prompts per rollout shard.

## 选择 Checkpoint 标准

1. 先看 heldout/eval harness，不能只看 train proxy reward。
2. 若 Tool reward 或 Tool gate 快速塌陷，优先保守选择早停 checkpoint。
3. 若 Code gate 持续上涨但 Code proxy 不涨，判定为过推。
4. A 是主论文候选；B 用于证明 OPD 增益；C 用于讨论高容量 gate 的收益与过拟合风险。

## D: Dynamic OPD 追加实验

启动时间：2026-05-14 11:54 Asia/Shanghai

目的：

- 对齐“OPD 只修复当前 policy 在 96 prompt 上全错的样本”的设定。
- 先为同一 96 prompt 离线生成 Tool/Memory/Code expert trajectories。
- 每轮 policy rollout 后，筛选 `num_success=0` / `all_failure` 的 prompt，再匹配同 prompt expert positive，生成本轮 `opd_distill_from_allfail.jsonl`。
- Gate 使用 `global-coefficient`，即只学习 `tool/memory/code` 三个直接系数。

启动命令：

```bash
RUN_TAG=20260514_paper96_dynopd_i8 \
BASE_RUN_TAG=20260514_paper96_i8 \
MONITOR_PORT=8769 \
  bash skill/command/run_paper96_dynamic_opd_gc_20260514.sh
```

Run directory:

- D: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_D_gc_dynamic_opd_i8_20260514_paper96_dynopd_i8`

Expert rollout cache:

- Tool: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl`
- Memory: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl`
- Code: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl`

关键差异：

- A/C 使用固定 21 条 OPD rows。
- D 每轮按当前 policy 的全错样本动态重建 OPD rows。
- D 的 OPD row 上限是每任务 32，实际数量由当前轮全错样本和 expert 是否做对共同决定。
- D 继续保留 retention、length normalization、epoch-level optimizer step、官方 reward 训练口径。

监控：

```text
http://127.0.0.1:8769
```

SSH tunnel:

```bash
ssh -L 8769:127.0.0.1:8769 <server>
```

## Eval6 正式评测调度

评测口径沿用 `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/roadmap.md`：

- Tool/BFCL: `parallel`, `parallel_multiple`, `live_parallel`, `live_parallel_multiple`
- Memory/HotpotQA: `eval_50`, `eval_100`, `eval_qa_1_32768`, `eval_qa_1_65536`
- Code/CURE: `LiveBench`, `LiveCodeBench`
- 聚合脚本: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/append_eval6_models.py`

评测 runner:

- `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_paper96_final_gates_20260514.py`

当前送评 checkpoint 与实验设置：

| ID | Eval model name | Gate | OPD | OPD 类型 | Retention | Train checkpoint | Baked model | Eval GPU | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| A | `paper96-a-gc-fixedopd-final-iter8` | `global-coefficient` | on | fixed 21 rows | on | `iter_008/gate_updates.gates.json` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_A_gc_opd_i8_20260514_paper96_i8/eval_baked_policy_iter008_final_gate` | 6 | CURE running |
| B | `paper96-b-gc-noopd-final-iter8` | `global-coefficient` | off | none | on | `iter_008/gate_updates.gates.json` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_B_gc_noopd_i8_20260514_paper96_i8/eval_baked_policy_iter008_final_gate` | 4 | CURE running |
| C | `paper96-c-gp-fixedopd-final-iter8` | `global-parameter` | on | fixed 21 rows | on | `iter_008/gate_updates.gates.json` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_C_gp_opd_i8_20260514_paper96_i8/eval_baked_policy_iter008_final_gate` | 5 | CURE running |
| D | `paper96-d-gc-dynopd-final-iter8` | `global-coefficient` | on | dynamic all-fail OPD | on | pending final `iter_008` | pending bake | pending | training |

已完成 Tool/BFCL 结果：

| Model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Tool mean |
|---|---:|---:|---:|---:|---:|
| `paper96-a-gc-fixedopd-final-iter8` | 0.9150 | 0.8650 | 0.6875 | 0.6667 | 0.7835 |
| `paper96-b-gc-noopd-final-iter8` | 0.9150 | 0.8600 | 0.6875 | 0.6667 | 0.7823 |
| `paper96-c-gp-fixedopd-final-iter8` | 0.9150 | 0.8650 | 0.6875 | 0.7083 | 0.7939 |

已完成 Memory/HotpotQA 结果：

| Model | eval_50 EM/F1 | eval_100 EM/F1 | eval_qa_1_32768 EM/F1 | eval_qa_1_65536 EM/F1 | Memory F1 mean |
|---|---:|---:|---:|---:|---:|
| `paper96-a-gc-fixedopd-final-iter8` | 0.5312/0.6820 | 0.5078/0.6587 | 0.5000/0.6034 | 0.5469/0.6558 | 0.6500 |
| `paper96-b-gc-noopd-final-iter8` | 0.5156/0.6680 | 0.4922/0.6426 | 0.5781/0.6842 | 0.4844/0.6075 | 0.6506 |
| `paper96-c-gp-fixedopd-final-iter8` | 0.5391/0.7072 | 0.4766/0.6200 | 0.5547/0.6785 | 0.5391/0.6583 | 0.6660 |

当前日志：

- B/C runner: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/logs/paper96_bc_eval_runner.log`
- A runner: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/logs/paper96_a_eval_runner.log`

待补：

- A/B/C 的 Code summary。
- D 完成 `iter_008` 后 bake 并进入同一 Eval6 表。

## 16:35 Stop And Postmortem

停止时间：2026-05-14 16:35 Asia/Shanghai

动作：

- 停止 `paper96_A/B/C` 的 Eval6 runner 与 CURE 评测。
- 停止 `paper96_D_gc_dynamic_opd_i8` 训练与 D-only Eval6 watcher。
- 停止旧的 paper96 tmux 监控 session。
- 停止后 `nvidia-smi` 显示 0-7 卡均无 paper96 compute 进程。

保留产物：

- A/B/C 的训练 run dir、final gate、baked eval model、Tool/BFCL summary、Memory/HotpotQA summary。
- A/B/C 的 Code/CURE 评测未完成，因此本次 paper96 不报告 Code 正式结果。
- D 停在 `iter_008` update/rollout 附近，未形成可用于正式 Eval6 的 final checkpoint。

### 当前结果是否成功

结论：**相较昨晚最佳实验，本次 paper96 版本不成功**。

原因不是 Tool 坏掉。A/B/C 的 Tool/BFCL 与昨晚持平或略高：

| model | Tool weighted | Memory avg F1 | Code |
|---|---:|---:|---|
| `paper96-a-gc-fixedopd-final-iter8` | 0.8705 | 0.6500 | stopped before summary |
| `paper96-b-gc-noopd-final-iter8` | 0.8682 | 0.6506 | stopped before summary |
| `paper96-c-gp-fixedopd-final-iter8` | 0.8727 | 0.6660 | stopped before summary |
| `opvec-gp-opd-best-iter9` | 0.8705 | 0.7649 | 0.3487 code acc |

| `opvec-gc-opd-best-iter8` | 0.8705 | 0.7361 | 0.3448 code acc |

主要失败点是 Memory：当前最好的 C 也只有 `0.6660`，比昨晚 `global-parameter + OPD` best 的 `0.7649` 低约 `0.099`。这已经足以判定 paper96 这版没有复现昨晚的主效果。

### 训练动态对比

昨晚最佳 `global-parameter + OPD` 的关键状态：

| item | value |
|---|---:|
| best checkpoint | `qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_20260514_0154/iter_008/gate_updates.gates.json` |
| best rollout | iter9 |
| iter9 overall | 0.7189 |
| iter9 tool / memory / code | 0.9809 / 0.7188 / 0.4570 |
| gate global tool / memory / code | 0.2906 / 0.5428 / 0.7117 |
| gate mean tool / memory / code | 0.2903 / 0.5439 / 0.7136 |
| iter8 update grad_norm_max | 0.5442 |
| iter8 gate_delta_max | 0.0774 |
| frontier task counts | code=11, memory=6, tool=1 |
| OPD rows | 21, balanced 7/7/7 |
| retention rows | 0 |

本次 paper96 的典型状态：

| item | A: global-coefficient fixed OPD | C: global-parameter fixed OPD |
|---|---:|---:|
| final rollout overall | 0.2535 | 0.2348 |
| final tool / memory / code | 0.0446 / 0.3594 / 0.3564 | -0.0505 / 0.3594 / 0.3955 |
| final gate tool / memory / code | 0.3370 / 0.3346 / 0.3336 | 0.3375 / 0.3345 / 0.3338 |
| iter8 update grad_norm_max | 0.0077 | 0.0119 |
| iter8 gate_delta_max | 0.00027 | 0.00048 |
| frontier task counts | code=23, memory=21, tool=26 | code=23, memory=25, tool=26 |
| OPD rows | 21, balanced 7/7/7 | 21, balanced 7/7/7 |
| retention rows | 3 | 4 |

核心差异是梯度尺度和有效信号比例。昨晚 best 的 OPD/GRPO update 能产生 `~0.077` 级别的单轮 gate 位移；paper96 的同类 update 只有 `~0.0003-0.0005`，差了两个数量级。系数没有进入 `0.5-0.8` 的有效区间，模型自然不会出现 Memory/Code 的明显提升。

### 为什么这次梯度被稀释

1. **96 prompts 的 frontier 太多，但不等于高质量方向更多。**

昨晚 best 的 iter8 frontier 是 `code=11, memory=6, tool=1`，OPD 21 rows 在 loss 中占很大比例，且主要驱动 memory/code task vector 上升。paper96 每轮 frontier 接近 `70+` rows，再加 retention，OPD 仍然只有 21 rows。OPD 从“主导方向”变成“弱辅助项”，被大量 on-policy frontier 的近零/噪声优势抵消。

2. **paper96 的 calibration 更均衡，但更保守。**

paper96 使用 `tool=32, memory=32, code=32` 的均衡题库，目标是论文口径更规范；但这个集合里大量样本对 `1/3` gate 已经没有强可分梯度，或者 reward 受采样波动主导。昨晚 high-info / frontier 型数据更像“压力测试集”，更容易暴露增加 memory/code task vector 的收益。

3. **fixed OPD 没有绑定 paper96 当前失败样本。**

A/C 使用固定 21 条 OPD rows，来源是已有 high-info distill 数据，不是当前 96 prompts 中 policy 失败、expert 成功的样本。这样 OPD 方向和 paper96 rollout frontier 的方向不一定一致。D 尝试 dynamic all-fail OPD，但仍沿用了较保守的 `LR=0.04, OPD=0.12/0.06, retention=0.03`，且 OPD rows 规模仍小，没能形成强推动。

4. **retention 与 prior 把探索进一步拉回 1/3 附近。**

paper96 版本启用了 `retention_loss_weight=0.03` 和 `prior_loss_weight=0.005`。昨晚 best 的有效 update 中 retention rows 为 0，OPD 方向更纯。paper96 加入 retention 是为了防止能力破坏，但在当前 gate 尚未进入高能力区间时，它更像阻尼项，压低了 task-vector 位移。

5. **length normalization 设置改变了 OPD/trajectory 信号的相对尺度。**

昨晚 best 的 update summary 显示 `length_normalize_logprob=False, length_normalize_policy_logprob=True`；paper96 A/C 显示两者都为 true。对于 Memory 这种长轨迹任务，额外的 length normalization 会进一步压低长回答/轨迹样本的 NLL 或 policy logprob 贡献，使 memory/code OPD 很难像昨晚那样把系数推开。

6. **checkpoint 选择也更弱。**

昨晚不是盲目评最终 checkpoint，而是根据 proxy reward 和 gate 区间选择 best：`iter9 rollout` 对应 `iter8 gate`。paper96 送评的是 final iter8 gate；A/C 在训练 proxy 上早已回落，C 的 best proxy 在 iter4 附近，final iter8 不是最优点。不过即使只看 C 的 best proxy，也没有达到昨晚 `0.7+` 的有效区间。

### 对 B/noOPD 的解释

B 的 iter8 overall `0.5458` 看起来最好，但主要来自 Tool reward spike：

| task | iter8 reward |
|---|---:|
| tool | 0.8669 |
| memory | 0.3828 |
| code | 0.3877 |

B 的 gate 仍几乎停在 `tool=0.3382, memory=0.3335, code=0.3335`。这不是“无 OPD 更好”，而是 Tool 子集高方差/采样波动让 overall reward 被抬高。正式 Eval6 也支持这个判断：B 的 Tool 与 A/C 接近，但 Memory 只有 `0.6506`，没有形成能力增强。

### 这次实验给出的负结论

1. **“论文式均衡 96 prompts + 保守 OPD/retention”不能直接复现昨晚的强推动。**
2. **OPD rows 数量固定为 21 时，随着 frontier rows 增多，OPD 梯度会被明显稀释。**
3. **如果目标是自动找到高 reward task-vector 组合，训练初期必须让 OPD 或 high-info frontier 拥有足够权重，否则 gate 会被困在 `1/3` 附近。**
4. **正式评测上，当前 paper96 版本保住了 Tool，但 Memory 明显退化，不适合作为主论文结果。**

### 梯度量级缩小的核心原因

更精确地说，paper96 A/C 并不是没有 OPD。A/C 每轮确实读入固定 `21` 条 OPD rows，且始终是 `tool=7, memory=7, code=7`。B 没有 OPD。D 才是 dynamic all-fail OPD，它不是固定 21 条，实际每轮约 `15-21` 条，并且需要同时满足当前 policy 全错、同 prompt expert 有 positive、当前 rollout 有 negative、未超过 task quota。

真正把梯度压小的首要差异是 `LENGTH_NORMALIZE_LOGPROB=1`。昨晚 best 的 update 是 `length_normalize_logprob=False, length_normalize_policy_logprob=True`；paper96 A/C 是两者都为 true。代码里 OPD best-response / pairwise 都通过 `_sample_score(logp, length_normalize=True)` 计算，即 `logp / length`。paper96 固定 OPD 的 response 平均长度约 `275` tokens，positive 平均约 `321` tokens；因此 OPD 对 gate 的梯度被按 token 数除掉，从 sequence-sum 信号变成 per-token 平均信号。因为昨晚的有效推动主要来自 OPD，这个开关足以把 OPD 从“主梯度”压成“几乎看不见的辅助项”。

日志也支持这个判断。昨晚 best iter1 的 `grad_norm_max` 约 `0.49`，paper96 A/C iter1 只有 `0.013/0.010`；paper96 B/noOPD iter1 是 `0.018`。也就是说，paper96 中“开 OPD”和“不开 OPD”的梯度量级已经非常接近，说明 OPD 没有像昨晚那样贡献主导梯度。

第二层原因是 loss normalizer 和样本组成。昨晚 best iter1 是 `frontier=36, OPD=21, retention=0, loss_normalizer=57`；paper96 A iter1 是 `frontier=66, OPD=21, retention=6, loss_normalizer=93`，C iter1 是 `frontier=73, OPD=21, retention=4, loss_normalizer=98`。在 `batch_loss_reduction=mean` 下，OPD 的相对权重进一步降低，并被更多低方差或方向冲突的 GRPO frontier rows 平均掉。

retention 当前只来自 all-success rows：实现上 `_is_retention_candidate(row)` 检查 `frontier.all_success` 或 `skip_reason == "all_success"`。paper96 启用了 `USE_RETENTION=1, RETENTION_LOSS_WEIGHT=0.03`，每轮实际 retention rows 只有几条到十条。它不是梯度量级缩小的主因，但当 OPD 被 length normalization 压小后，retention/prior 会在方向上更容易把 gate 留在 `1/3` 附近。

所以核心因果链是：

`OPD sequence loss 被 length-normalized` -> `OPD 梯度按 response length 缩小` -> `开 OPD 和不开 OPD 的 grad_norm 接近` -> `剩余 GRPO frontier 信号弱且冲突` -> `retention/prior 阻尼下 gate 基本停在 1/3`。

### 下一步应该怎么改

优先方向不是继续跑 paper96 这个配置，而是回到昨晚 best 的有效机制，并逐步规范化：

1. 先复现昨晚 best 的推动强度：允许 `global-parameter + OPD` 把 memory/code 推到 `0.5-0.75`，同时用 tool retention 防止 tool gate 低于约 `0.28`。
2. 用 paper96 题库时提高 OPD 占比：例如每轮 `frontier:OPD` 至少接近 `1:1`，或按 task 分别做 loss normalizer，而不是让 70+ frontier rows 淹没 21 OPD rows。
3. dynamic OPD 应该绑定当前 96 prompts，但权重要更强：对 all-fail 且 expert 成功样本，OPD loss 不能只是弱辅助，应成为该样本的主学习信号。
4. early phase 关闭或降低 retention/prior；等 gate 进入有效能力区间后再开 retention 做能力保持。
5. checkpoint 选择要按 rollout proxy + Eval6 子集共同早停，不再默认评 final iter。
6. 前端必须同时显示 raw rollout reward、OPD/retention loss 权重、gate delta、grad norm；其中 `gate_delta_max` 和 `grad_norm_max` 是判断是否“推得动”的第一指标。

## Formal Eval6 Results

Update time: 2026-05-14 18:05 Asia/Shanghai.

This section supersedes the B/C Code status in the 16:35 stop note above. B and C had completed vLLM generation before the interruption; their LiveCodeBench temp JSON files were valid, so the missing CURE execution and summaries were recovered CPU-only from those existing temp artifacts. A's LiveCodeBench temp JSON is truncated (`JSONDecodeError` at char `3277489015`), so A still needs a rerun of Code/LiveCodeBench generation if a full Eval6 result is required. D has no final `iter_008/gate_updates.gates.json`; its training stopped during the iter 008 update after rollout and dynamic OPD construction.

Evaluation runner paths:

- Eval6 runner: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_paper96_final_gates_20260514.py`
- CURE recovery runner: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/recover_paper96_cure_livecodebench_from_temp_20260514.py`
- Aggregate updater: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/append_eval6_models.py`

### Models And Settings

| ID | Eval model name | Gate | OPD setting | Retention | Run dir | Gate checkpoint | Baked model | Eval status |
|---|---|---|---|---|---|---|---|---|
| A | `paper96-a-gc-fixedopd-final-iter8` | `global-coefficient` | fixed OPD, 21 rows | on | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_A_gc_opd_i8_20260514_paper96_i8` | `iter_008/gate_updates.gates.json` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_A_gc_opd_i8_20260514_paper96_i8/eval_baked_policy_iter008_final_gate` | Tool+Memory complete; Code incomplete |
| B | `paper96-b-gc-noopd-final-iter8` | `global-coefficient` | no OPD | on | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_B_gc_noopd_i8_20260514_paper96_i8` | `iter_008/gate_updates.gates.json` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_B_gc_noopd_i8_20260514_paper96_i8/eval_baked_policy_iter008_final_gate` | Full Eval6 complete |
| C | `paper96-c-gp-fixedopd-final-iter8` | `global-parameter` | fixed OPD, 21 rows | on | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_C_gp_opd_i8_20260514_paper96_i8` | `iter_008/gate_updates.gates.json` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_C_gp_opd_i8_20260514_paper96_i8/eval_baked_policy_iter008_final_gate` | Full Eval6 complete |
| D | `paper96-d-gc-dynopd-final-iter8` | `global-coefficient` | dynamic all-fail OPD | on | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_D_gc_dynamic_opd_i8_20260514_paper96_dynopd_i8` | missing final `iter_008/gate_updates.gates.json` | pending | Not evaluable |

All four runs use 96 prompts, init 1/3, epoch-scope optimizer step, sequence loss, SGD momentum 0.8, and retention on.

### Tool / BFCL

Weighted overall is total correct divided by total count across the four BFCL subsets.

| Model | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Weighted overall |
|---|---:|---:|---:|---:|---:|
| `paper96-a-gc-fixedopd-final-iter8` | 0.6875 | 0.6667 | 0.9150 | 0.8650 | 0.8705 |
| `paper96-b-gc-noopd-final-iter8` | 0.6875 | 0.6667 | 0.9150 | 0.8600 | 0.8682 |
| `paper96-c-gp-fixedopd-final-iter8` | 0.6875 | 0.7083 | 0.9150 | 0.8650 | 0.8727 |

### Memory / HotpotQA

Cells are EM/F1.

| Model | eval_50 | eval_100 | eval_qa_1_32768 | eval_qa_1_65536 | Avg EM/F1 |
|---|---:|---:|---:|---:|---:|
| `paper96-a-gc-fixedopd-final-iter8` | 0.5312/0.6820 | 0.5078/0.6587 | 0.5000/0.6034 | 0.5469/0.6558 | 0.5215/0.6500 |
| `paper96-b-gc-noopd-final-iter8` | 0.5156/0.6680 | 0.4922/0.6426 | 0.5781/0.6842 | 0.4844/0.6075 | 0.5176/0.6506 |
| `paper96-c-gp-fixedopd-final-iter8` | 0.5391/0.7072 | 0.4766/0.6200 | 0.5547/0.6785 | 0.5391/0.6583 | 0.5273/0.6660 |

### Code / CURE

Cells are `code_acc / accumulated / estimated unit-test acc / estimated unit-test accumulated / BoN(4,4) acc / BoN(4,4) accumulated`.

| Model | LiveBench | LiveCodeBench | Code status |
|---|---:|---:|---|
| `paper96-a-gc-fixedopd-final-iter8` | 0.3398 / 0.4667 / 0.2707 / 0.3098 / 0.3984 / 0.5446 | n/a | LiveBench printed in log; LiveCodeBench JSON truncated, no summary |
| `paper96-b-gc-noopd-final-iter8` | 0.3672 / 0.4907 / 0.2964 / 0.3474 / 0.4375 / 0.5426 | 0.3014 / 0.4480 / 0.3592 / 0.4016 / 0.3503 / 0.5055 | complete |
| `paper96-c-gp-fixedopd-final-iter8` | 0.3770 / 0.4985 / 0.2750 / 0.3238 / 0.4531 / 0.5886 | 0.3043 / 0.4553 / 0.3330 / 0.3633 / 0.3444 / 0.5085 | complete |

### Summary Paths

| Model | Tool summary | Memory summary | Code summary |
|---|---|---|---|
| `paper96-a-gc-fixedopd-final-iter8` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/paper96-a-gc-fixedopd-final-iter8/tool/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/paper96-a-gc-fixedopd-final-iter8/eval6-20260502-125748/summary.json` | missing |
| `paper96-b-gc-noopd-final-iter8` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/paper96-b-gc-noopd-final-iter8/tool/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/paper96-b-gc-noopd-final-iter8/eval6-20260502-125748/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/paper96-b-gc-noopd-final-iter8/eval6-20260502-125748/summary.json` |
| `paper96-c-gp-fixedopd-final-iter8` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/paper96-c-gp-fixedopd-final-iter8/tool/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/paper96-c-gp-fixedopd-final-iter8/eval6-20260502-125748/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/paper96-c-gp-fixedopd-final-iter8/eval6-20260502-125748/summary.json` |

### Interpretation

- C (`global-parameter` with fixed OPD) is the best of A/B/C on Tool weighted overall and Memory average F1, but the margin over B is small on Tool and moderate on Memory.
- B (`global-coefficient`, no OPD) is slightly behind C on Tool/Memory, but its LiveCodeBench BoN is slightly higher than C while C is higher on raw LiveCodeBench code_acc and accumulated accuracy.
- Fixed OPD does not show a clean win in this partial comparison: C improves Memory and LiveBench versus B, but B is competitive on LiveCodeBench BoN and A cannot be fully compared because Code is incomplete.
- D cannot support a dynamic-OPD conclusion yet because no final gate checkpoint was produced.
