# 2026-05-14 昨晚最佳实验复现记录：Global-Parameter + Compact OPD

## 一句话结论

昨晚最佳实验是 `global-parameter + epoch-scope update + compact OPD`。它从标准 `1/3` task-vector 初始化出发，用 48 个 calibration prompts 和 21 条 compact OPD rows，把 gate 有效均值推到 `tool=0.2903, memory=0.5439, code=0.7136`，对应 iter9 proxy reward 达到 `overall=0.7189`。正式 Eval6 中 Tool 保持、Memory 明显增强、Code 略优，是当前最强候选。

最关键的复现点：**OPD 使用 sequence-sum logprob，不做 OPD length normalization**：

```text
LENGTH_NORMALIZE_LOGPROB=0
LENGTH_NORMALIZE_POLICY_LOGPROB=1
```

今天 paper96 失败的核心区别正是在这里：paper96 把 `LENGTH_NORMALIZE_LOGPROB=1` 打开后，OPD loss 从 sequence-sum 变成 per-token average，梯度量级下降两个数量级。

## 原始 Run

| 项 | 值 |
|---|---|
| run dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_20260514_0154` |
| 训练时间 | 2026-05-14 01:54 - 04:34 Asia/Shanghai |
| 总耗时 | `9569.7s`，约 `2h39m` |
| 参数化 | `global-parameter` |
| 可学习参数 | 588 个 task-vector gate 参数，`common + residual` 形式 |
| 初始化 | `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/init_gates/init_global_parameter_c033333.json` |
| best checkpoint | `iter_008/gate_updates.gates.json` |
| best rollout | iter9，使用 iter8 gate |
| best baked model | `iter_009/baked_policy` |

## 数据

| 类型 | 路径 | 说明 |
|---|---|---|
| calibration prompts | `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.prompts.jsonl` | 使用 manifest 前 48 条，三任务均衡 |
| OPD replay | `/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill_balanced21_compact_pos1_neg2_seed20260514.jsonl` | 21 rows，tool/memory/code 各 7 rows，每 row 1 positive + 2 negative |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` | 588 个可门控模块系数 |

OPD 文件的注意点：

- 该旧版 compact OPD 文件没有显式 `opd_role`。
- 部分 sample 没有 `reward_train`，update 侧会 fallback 到 `reward`。
- 对 tool 来说，fallback 会把 raw reward 映射到 `[0,1]`，在 `OPD_POSITIVE_REWARD_THRESHOLD=1.0` 下，tool OPD 基本不产生非零 best-response/pairwise loss。
- 因此昨晚这次“21 条 OPD”在实际梯度上更像 **memory/code 强 OPD + tool 主要由 on-policy GRPO 保持**。
- 这反而解释了为什么 memory/code 被强推，而 tool gate 可以下降但 Tool 能力仍保持。

## 训练设置

| 项 | 值 |
|---|---|
| `NUM_ITERS` | `10` |
| `NUM_PROMPTS` | `48` |
| `SAMPLES_PER_PROMPT` | `4` |
| `GPU_LIST / ROLLOUT_GPUS` | `2,3` |
| rollout | 每轮 bake checkpoint，然后 2 个 vLLM shard 并行 rollout |
| `ROLLOUT_BATCH_SIZE` | `32` |
| `MAX_MODEL_LEN` | `12288` |
| `MAX_LOGPROB_TOKENS` | `12288` |
| `TOOL_MAX_NEW_TOKENS` | `512` |
| `MEMORY_UPDATE_MAX_NEW_TOKENS` | `2048` |
| `MEMORY_FINAL_MAX_NEW_TOKENS` | `2048` |
| `CODE_MAX_NEW_TOKENS` | `4096` |
| optimizer | SGD |
| `LR` | `0.04` |
| `SGD_MOMENTUM` | `0.8` |
| `PERSIST_OPTIMIZER_STATE` | `1` |
| `OPTIMIZER_STEP_SCOPE` | `epoch` |
| `UPDATE_BATCH_SIZE` | `4`，只用于 loss 累计分块；每 epoch 只 step 一次 |
| `LOSS_GRANULARITY` | `sequence` |
| `PPO_LOSS_WEIGHT` | `6.0` |
| `OPD_LOSS_WEIGHT` | `0.12` |
| `OPD_PAIRWISE_LOSS_WEIGHT` | `0.06` |
| `MAX_OPD_DISTILL_ROWS` | `21` |
| `MAX_OPD_PAIRWISE_PAIRS_PER_ROW` | `2` |
| `OPD_POSITIVE_REWARD_THRESHOLD` | `1.0` |
| `USE_RETENTION` | `0` |
| `PRIOR_LOSS_WEIGHT` | `0.0` |
| `MAX_COEFF_DELTA` | `1.0` |
| `TASK_NORMALIZE_ADVANTAGES` | `0` |
| `ADVANTAGE_NORMALIZATION` | `centered` |
| `LENGTH_NORMALIZE_POLICY_LOGPROB` | `1` |
| `LENGTH_NORMALIZE_LOGPROB` | `0` |

## 复现命令

从项目根目录运行：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

RUN_NAME=qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_repro_20260514 \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_repro_20260514 \
STRATEGY=global-parameter \
GPU_LIST=2,3 \
ROLLOUT_GPUS=2,3 \
NUM_ITERS=10 \
NUM_PROMPTS=48 \
SAMPLES_PER_PROMPT=4 \
INIT_VALUE=0.3333333333333333 \
OPTIMIZER=sgd \
SGD_MOMENTUM=0.8 \
PERSIST_OPTIMIZER_STATE=1 \
LR=0.04 \
PPO_LOSS_WEIGHT=6.0 \
PRIOR_LOSS_WEIGHT=0.0 \
MAX_COEFF_DELTA=1.0 \
UPDATE_EPOCHS=1 \
UPDATE_BATCH_SIZE=4 \
BATCH_LOSS_REDUCTION=mean \
OPTIMIZER_STEP_SCOPE=epoch \
LOSS_GRANULARITY=sequence \
STORE_TOKEN_LOGPROBS=0 \
TASK_NORMALIZE_ADVANTAGES=0 \
ADVANTAGE_NORMALIZATION=centered \
USE_FRONTIER_WEIGHT=0 \
FRONTIER_ORDER=task-interleaved \
FRONTIER_SHUFFLE_SEED=20260514 \
FRONTIER_TOOL_QUOTA=24 \
FRONTIER_MEMORY_QUOTA=24 \
FRONTIER_CODE_QUOTA=24 \
MAX_FRONTIER_ROWS_PER_TASK=24 \
USE_RETENTION=0 \
OPD_DISTILL_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill_balanced21_compact_pos1_neg2_seed20260514.jsonl \
OPD_LOSS_WEIGHT=0.12 \
OPD_PAIRWISE_LOSS_WEIGHT=0.06 \
OPD_PAIRWISE_MARGIN=0.0 \
OPD_POSITIVE_REWARD_THRESHOLD=1.0 \
MAX_OPD_DISTILL_ROWS=21 \
MAX_OPD_PAIRWISE_PAIRS_PER_ROW=2 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
LENGTH_NORMALIZE_LOGPROB=0 \
MAX_NEW_TOKENS=1024 \
TOOL_MAX_NEW_TOKENS=512 \
MEMORY_UPDATE_MAX_NEW_TOKENS=2048 \
MEMORY_FINAL_MAX_NEW_TOKENS=2048 \
CODE_MAX_NEW_TOKENS=4096 \
MAX_PROMPT_TOKENS=8192 \
MAX_MODEL_LEN=12288 \
MAX_LOGPROB_TOKENS=12288 \
ROLLOUT_SHARDS=auto \
ROLLOUT_BATCH_SIZE=32 \
TENSOR_PARALLEL_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.82 \
POST_BAKE_SLEEP_SECONDS=5 \
TEMPERATURE=0.7 \
TOP_P=0.95 \
SEED_VALUE=20260512 \
GRADIENT_CHECKPOINTING=1 \
MAX_MEMORY_PER_GPU=70GiB \
CPU_MAX_MEMORY=180GiB \
bash skill/command/run_qbank_c033333_gate_strategy.sh
```

复现时最应该检查的日志行：

```text
[run] strategy=global-parameter init=0.3333333333333333
[run] calibration=/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.prompts.jsonl
[run] optimizer=sgd sgd_momentum=0.8 persist_optimizer_state=1
[run] lr=0.04 prior=0.0 max_delta=1.0
[run] loss_granularity=sequence update_batch_size=4 optimizer_step_scope=epoch store_token_logprobs=0
[run] retention=0
[run] opd_rollout=/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill_balanced21_compact_pos1_neg2_seed20260514.jsonl opd_loss=0.12 opd_pairwise=0.06 max_opd_rows=21
```

## 训练曲线

Tool reward 这里写的是归一化后的 Tool reward，计算方式为 `(raw_tool_reward + 3) / 7`，因此可与 Memory/Code 的 `[0,1]` reward 平均。

| iter | overall | tool | memory | code | gate tool | gate memory | gate code | frontier | grad norm | gate delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4464 | 0.5005 | 0.4219 | 0.4168 | 0.3319 | 0.3466 | 0.3469 | 36 | 0.4865 | 0.0143 |
| 2 | 0.4409 | 0.5084 | 0.4062 | 0.4082 | 0.3291 | 0.3718 | 0.3720 | 34 | 0.5195 | 0.0270 |
| 3 | 0.4845 | 0.8245 | 0.2656 | 0.3633 | 0.3256 | 0.4043 | 0.4070 | 32 | 0.4986 | 0.0358 |
| 4 | 0.6410 | 0.9776 | 0.5312 | 0.4141 | 0.3206 | 0.4431 | 0.4530 | 24 | 0.5682 | 0.0471 |
| 5 | 0.6067 | 0.9841 | 0.4531 | 0.3828 | 0.3146 | 0.4807 | 0.5074 | 25 | 0.4868 | 0.0556 |
| 6 | 0.6884 | 0.9813 | 0.6094 | 0.4746 | 0.3074 | 0.5105 | 0.5690 | 24 | 0.4749 | 0.0631 |
| 7 | 0.7063 | 0.9724 | 0.6406 | 0.5059 | 0.2993 | 0.5315 | 0.6380 | 20 | 0.5181 | 0.0706 |
| 8 | 0.6645 | 0.9799 | 0.6406 | 0.3730 | 0.2903 | 0.5439 | 0.7136 | 18 | 0.5442 | 0.0774 |
| 9 | 0.7189 | 0.9809 | 0.7188 | 0.4570 | 0.2809 | 0.5493 | 0.7938 | 18 | 0.5250 | 0.0820 |
| 10 | 0.7032 | 0.9806 | 0.7031 | 0.4258 | 0.2712 | 0.5494 | 0.8763 | 19 | 0.4937 | 0.0845 |

说明：

- 表中 iter9 的 `overall=0.7189` 是本 run 采用的 best rollout 口径。
- iter9 rollout 使用的是 iter8 gate，因此 best checkpoint 是 `iter_008/gate_updates.gates.json`。
- iter10 继续更新后 code gate 过高、tool gate 更低，不作为 best。

## 正式 Eval6 结果

Eval model name: `opvec-gp-opd-best-iter9`

| 任务 | 指标 | 结果 |
|---|---|---:|
| Tool/BFCL | parallel | 0.9050 |
| Tool/BFCL | parallel_multiple | 0.8750 |
| Tool/BFCL | live_parallel | 0.6875 |
| Tool/BFCL | live_parallel_multiple | 0.6667 |
| Tool/BFCL | mean | 0.7835 |
| Memory/HotpotQA | eval_50 F1 | 0.7414 |
| Memory/HotpotQA | eval_100 F1 | 0.7741 |
| Memory/HotpotQA | eval_qa_1_32768 F1 | 0.7486 |
| Memory/HotpotQA | eval_qa_1_65536 F1 | 0.7956 |
| Memory/HotpotQA | mean F1 | 0.7649 |
| Code/CURE | LiveBench code_acc | 0.3828 |
| Code/CURE | LiveCodeBench code_acc | 0.3146 |
| Code/CURE | mean code_acc | 0.3487 |

评测产物：

- Tool: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/opvec-gp-opd-best-iter9/tool/summary.json`
- Memory: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/opvec-gp-opd-best-iter9/eval6-20260502-125748/summary.json`
- Code: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/opvec-gp-opd-best-iter9/eval6-20260502-125748/summary.json`

## 今天 OPD 与昨晚 Best 的明显差异

| 维度 | 昨晚 best | 今天 paper96 A/C |
|---|---|---|
| OPD 数据 | `distill_balanced21_compact_pos1_neg2_seed20260514.jsonl` | `distill_balanced21_paperfix_rewardtrain_len_seed20260514.jsonl` |
| OPD rows | 固定 21，7/7/7 | 固定 21，7/7/7 |
| 实际非零 OPD loss | 主要 memory/code 14 rows；tool 基本不贡献 | tool/memory/code 21 rows 都贡献 |
| OPD length normalize | `False` | `True` |
| policy GRPO length normalize | `True` | `True` |
| retention | 0 | on，少量 all-success rows |
| prior | 0 | 0.005 |
| frontier rows | iter1 36，iter8 18 | 每轮约 66-75 |
| loss normalizer | iter1 57，iter8 39 | iter1 93-98，iter8 94-99 |
| grad norm | `0.49-0.56` | `0.007-0.019` |
| gate delta | iter8 `0.0774` | iter8 `0.0003-0.0005` |

### OPD loss 数值证据

iter1 update 的 OPD row loss：

| run | 非零 OPD rows | OPD raw loss 量级 |
|---|---:|---:|
| 昨晚 best | memory 7 + code 7 | memory 合计约 `201`，code 合计约 `134` |
| paper96 A | tool 7 + memory 7 + code 7 | 三任务合计约 `2.13` |

两者都设置 `OPD_LOSS_WEIGHT=0.12, OPD_PAIRWISE_LOSS_WEIGHT=0.06`。差异不是 OPD 权重，而是 OPD score 是否除以 response length。

代码路径：

```text
scripts/train/opvec_update_gates_from_rollouts.py
_sample_score(logp, length_normalize=True) = logp / length
```

昨晚 best 中 OPD 用 sequence-sum logprob，长轨迹的 likelihood 差异会形成强梯度；今天 paper96 中 OPD 用 per-token average logprob，memory/code 长轨迹的强监督信号被按长度压小。再叠加更多 frontier rows、retention、prior，gate 就停在 `1/3` 附近。

## 复现成功判据

复现 run 不需要逐 token 完全一致，但应满足：

1. iter1 `grad_norm_max` 在 `0.45-0.55` 附近。
2. iter1 `gate_delta_max` 在 `0.01` 量级。
3. iter5 前后 memory/code gate 明显离开 `1/3`。
4. iter8 gate effective mean 接近：
   - tool `0.29-0.31`
   - memory `0.52-0.56`
   - code `0.65-0.75`
5. 若 iter1 `grad_norm_max < 0.05`，优先检查：
   - 是否误开 `LENGTH_NORMALIZE_LOGPROB=1`
   - 是否误开 retention/prior
   - OPD 文件是否换成 paperfix 版本
   - `OPD_LOSS_WEIGHT` / `OPD_PAIRWISE_LOSS_WEIGHT` 是否没有传入

## 结论

昨晚 best 的本质不是“OPD rows 数量多”，而是 **memory/code compact OPD 在 sequence-sum logprob 口径下提供了强、持续、方向一致的 gate 梯度**。今天 paper96 版本虽然把 OPD 数据格式修得更规范，也让 tool OPD 真正进入 loss，但同时打开了 OPD length normalization，并加入更多 frontier/retention/prior，导致 OPD 从主驱动力退化为弱辅助项。

后续如果要在 paper96 上复现昨晚效果，第一步应该保持：

```text
LENGTH_NORMALIZE_LOGPROB=0
USE_RETENTION=0 或显著降低 retention
PRIOR_LOSS_WEIGHT=0
OPD/frontier loss normalizer 接近昨晚比例
```

然后再逐步加入 tool retention、dynamic OPD 和更规范的论文设置。
