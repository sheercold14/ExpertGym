# 2026-05-14 Paper96 Dynamic OPD No-Length-Norm ABCD

## 实验目的

本实验重跑 paper96 ABCD，但把所有 OPD 都改成 **dynamic same-prompt OPD**，并关闭 OPD/best-response 的 length normalization。目标是验证今天 paper96 失败是否主要来自 `LENGTH_NORMALIZE_LOGPROB=1` 把 OPD 梯度压小。

核心假设：

```text
dynamic OPD + LENGTH_NORMALIZE_LOGPROB=0
```

应该让 `grad_norm_max` 和 `gate_delta_max` 恢复到昨晚 best 的量级；如果仍然推不动，再检查 dynamic OPD 覆盖率和任务分布。

## 启动时间

记录时间：2026-05-14 17:47:47 CST

## 与前序实验的区别

| 维度 | 昨晚 best | 今天 paper96 A/C | 本实验 |
|---|---|---|---|
| OPD 类型 | fixed compact OPD | fixed OPD | dynamic same-prompt OPD |
| OPD prompt 来源 | historical high-info | historical high-info | 当前 paper96 96 prompts |
| all-fail 筛选 | 否 | 否 | 是 |
| `LENGTH_NORMALIZE_LOGPROB` | 0 | 1 | 0 |
| `LENGTH_NORMALIZE_POLICY_LOGPROB` | 1 | 1 | 1 |
| retention/prior | off | on | A/C off，B/D on |
| 主要观测 | 强推动 | 梯度塌缩 | 判断 length norm 与 dynamic OPD 的作用 |

## 数据

| 项 | 路径 | 行数 |
|---|---|---:|
| paper96 prompts | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` | 96 |
| tool expert rollout | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl` | 32 |
| memory expert rollout | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl` | 32 |
| code expert rollout | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl` | 32 |

Dynamic OPD 每轮从当前 rollout 中筛：

```text
current policy all-fail prompt
+ same-prompt expert positive
+ current policy negative samples
```

输出到每轮目录：

```text
iter_xxx/opd_distill_from_allfail.jsonl
iter_xxx/opd_distill_from_allfail.summary.json
```

## ABCD 设计

| Run | GPU | Gate | Dynamic OPD | Retention | Prior | `MAX_COEFF_DELTA` | 目的 |
|---|---:|---|---|---|---:|---:|---|
| A | 0,1 | `global-coefficient` | on | off | 0.0 | 1.0 | 三系数强推动，最干净 |
| B | 2,3 | `global-coefficient` | on | on | 0.005 | 0.40 | 三系数保护项 |
| C | 4,5 | `global-parameter` | on | off | 0.0 | 1.0 | 最接近昨晚 best |
| D | 6,7 | `global-parameter` | on | on | 0.005 | 0.40 | 高容量保护项 |

## 公共设置

| 参数 | 值 |
|---|---|
| `NUM_ITERS` | 8 |
| `NUM_PROMPTS` | 96 |
| `SAMPLES_PER_PROMPT` | 4 |
| 初始化 | `1/3` |
| optimizer | SGD |
| `LR` | 0.04 |
| `SGD_MOMENTUM` | 0.8 |
| `PERSIST_OPTIMIZER_STATE` | 1 |
| `OPTIMIZER_STEP_SCOPE` | epoch |
| `UPDATE_BATCH_SIZE` | 4 |
| `LOSS_GRANULARITY` | sequence |
| `PPO_LOSS_WEIGHT` | 6.0 |
| `OPD_LOSS_WEIGHT` | 0.12 |
| `OPD_PAIRWISE_LOSS_WEIGHT` | 0.06 |
| `MAX_OPD_PAIRWISE_PAIRS_PER_ROW` | 2 |
| `OPD_POSITIVE_REWARD_THRESHOLD` | 1.0 |
| `TASK_NORMALIZE_ADVANTAGES` | 0 |
| `ADVANTAGE_NORMALIZATION` | centered |
| `LENGTH_NORMALIZE_LOGPROB` | 0 |
| `LENGTH_NORMALIZE_POLICY_LOGPROB` | 1 |
| `TOOL_MAX_NEW_TOKENS` | 768 |
| `MEMORY_UPDATE_MAX_NEW_TOKENS` | 1536 |
| `MEMORY_FINAL_MAX_NEW_TOKENS` | 768 |
| `CODE_MAX_NEW_TOKENS` | 2048 |

## Launcher

脚本：

```text
skill/command/run_paper96_dynamic_opd_nolen_abcd_20260514.sh
```

启动命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
RUN_TAG=20260514_dynopd_nolen_abcd_i8 \
MONITOR_PORT=8771 \
  bash skill/command/run_paper96_dynamic_opd_nolen_abcd_20260514.sh
```

Dry-run 命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
DRY_RUN=1 RUN_TAG=20260514_dynopd_nolen_abcd_i8_dryrun \
  bash skill/command/run_paper96_dynamic_opd_nolen_abcd_20260514.sh
```

## Run Directories

| Run | Directory |
|---|---|
| A | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_A_gc_dynopd_nolen_noret_i8_20260514_dynopd_nolen_abcd_i8` |
| B | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_B_gc_dynopd_nolen_ret_i8_20260514_dynopd_nolen_abcd_i8` |
| C | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_C_gp_dynopd_nolen_noret_i8_20260514_dynopd_nolen_abcd_i8` |
| D | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_D_gp_dynopd_nolen_ret_i8_20260514_dynopd_nolen_abcd_i8` |

## 监控

前端：

```text
http://127.0.0.1:8771
```

SSH tunnel：

```bash
ssh -L 8771:127.0.0.1:8771 <server>
```

## 第一轮强制检查标准

iter1 update 完成后立刻看：

| 指标 | 预期 |
|---|---|
| `dynamic_opd.selected_rows` | 每任务都有样本；若严重偏 Tool，需要记录 |
| `grad_norm_max` | 若 `<0.05`，判定仍然没恢复强 OPD |
| `gate_delta_max` | 若 `<0.005`，不要盲跑满 8 iter |
| `LENGTH_NORMALIZE_LOGPROB` | 必须为 false |
| `LENGTH_NORMALIZE_POLICY_LOGPROB` | 必须为 true |
| `OPD best_response/pairwise loss` | 应明显大于今天 paper96 A/C 的 `~0.1` 量级 |

## 当前状态

- 17:47:47：GPU 0-7 空闲。
- 17:47:47：paper96 prompt 与 expert rollout cache 已确认存在。
- 17:47:47：launcher `bash -n` 通过。
- 17:48:54：dry-run 通过。
  - A/B/C/D 的 update 命令都没有 `--length-normalize-logprob`。
  - A/B/C/D 的 update 命令都有 `--length-normalize-policy-logprob`。
  - A/B/C/D 每轮都插入 `build_opd_distill_from_expert_rollouts.py` dynamic OPD builder。
  - B/D 正确带 `--use-retention`、`retention_loss_weight=0.03`、`prior_loss_weight=0.005`。
  - A/C 正确为 `retention=0`、`prior_loss_weight=0.0`。
- 17:49:33：正式 ABCD 已启动，四个训练 session 与 monitor session 均已创建。
  - A: `paper96_A_gc_dynopd_nolen_20260514_dynopd_nolen_abcd_i8`
  - B: `paper96_B_gc_dynopd_nolen_20260514_dynopd_nolen_abcd_i8`
  - C: `paper96_C_gp_dynopd_nolen_20260514_dynopd_nolen_abcd_i8`
  - D: `paper96_D_gp_dynopd_nolen_20260514_dynopd_nolen_abcd_i8`
  - Monitor: `opvec_monitor_paper96_dynopd_nolen_20260514_dynopd_nolen_abcd_i8`
- 17:49:44：A/B/C/D 均已进入 iter1 bake 阶段；启动日志确认 A/C 为 no-retention，B/D 为 retention protected。
- 17:52-17:58：iter1 rollout 正常完成；四组均为 2 shard，每组 96 prompts / 384 samples。
- 17:58：iter1 dynamic OPD 生成完成，样本来自当前 policy all-fail prompt 与离线 expert rollout cache 的 same-prompt positive 匹配，不重新跑 expert。
  - A：16 rows，`tool=9,memory=5,code=2`；skip: `current_not_failure=60,no_expert_positive=20`。
  - B：18 rows，`tool=10,memory=7,code=1`；skip: `current_not_failure=58,no_expert_positive=20`。
  - C：13 rows，`tool=6,memory=5,code=2`；skip: `current_not_failure=67,no_expert_positive=16`。
  - D：20 rows，`tool=12,memory=5,code=3`；skip: `current_not_failure=58,no_expert_positive=18`。
- 18:28：iter1 update 全部完成，耗时约 30.5 min。该耗时显著偏高，原因是当前 update 路径对约 70-77 个 frontier rows、384 个 policy samples 和 OPD/retention rows 逐样本 HF logprob/backward；这不是 vLLM 批量推理路径。

## Iter1 Update 结果

| Run | Frontier rows | Frontier task | Dynamic OPD | Retention | `grad_norm_max` | `gate_delta_max` | Gate summary |
|---|---:|---|---|---:|---:|---:|---|
| A | 75 | code 27 / memory 23 / tool 25 | code 2 / memory 5 / tool 9 | 0 | 4.1579 | 0.03999 | code 0.33317 / memory 0.37333 / tool 0.33396 |
| B | 71 | code 27 / memory 21 / tool 23 | code 1 / memory 7 / tool 10 | 3 | 5.0522 | 0.04000 | code 0.33311 / memory 0.37333 / tool 0.33365 |
| C | 77 | code 24 / memory 24 / tool 29 | code 2 / memory 5 / tool 6 | 0 | 3.4021 | 0.04080 | 591 params, min 0.33347 / mean 0.34695 / max 0.37413 |
| D | 70 | code 29 / memory 20 / tool 21 | code 3 / memory 5 / tool 12 | 4 | 3.3689 | 0.04079 | 591 params, min 0.33334 / mean 0.34684 / max 0.37413 |

结论：

- `LENGTH_NORMALIZE_LOGPROB=0` 后，OPD/GRPO 梯度从今天 paper96 失败时的弱梯度恢复到强梯度量级；首轮不是信号塌缩。
- 三系数 global-coefficient 的首步主要推高 memory，code/tool 基本留在 1/3 附近。这与 dynamic OPD 的任务分布一致：OPD positive 里 code 太少，memory/tool 多。
- global-parameter 版本出现大范围参数上移，均值约 0.347，最大约 0.374；需要继续看后续 rollout reward 是否跟着提升，而不是只看 gate 上移。
- 当前主要工程问题是 update 太慢；如果继续按 8 iter 跑，单轮预计约 38-40 min，其中 rollout 约 6.7 min，update 约 30.5 min。
- 18:29：四组均进入 iter2 bake/collect；由于 iter1 梯度有效，暂不停止。

## Iter2 Rollout 观测

iter2 rollout 已完成，说明 iter1 gate update 对当前 calibration reward 有明显正向作用。

| Run | Tool train reward | Memory train reward | Code train reward | Tool success | Memory success | Code success | Iter2 dynamic OPD |
|---|---:|---:|---:|---:|---:|---:|---|
| A iter1 | 0.4113 | 0.3594 | 0.3842 | 0.2578 | 0.3594 | 0.2891 | 16 |
| A iter2 | 0.8601 | 0.4375 | 0.3750 | 0.6250 | 0.4375 | 0.2734 | 13 |
| B iter1 | 0.4472 | 0.3047 | 0.3920 | 0.2734 | 0.3047 | 0.2734 | 18 |
| B iter2 | 0.8132 | 0.3750 | 0.4082 | 0.5703 | 0.3750 | 0.3047 | 13 |
| C iter1 | 0.3822 | 0.4375 | 0.4199 | 0.2734 | 0.4375 | 0.3125 | 13 |
| C iter2 | 0.7836 | 0.4531 | 0.4721 | 0.5547 | 0.4531 | 0.3594 | 8 |
| D iter1 | 0.3878 | 0.3125 | 0.3637 | 0.2422 | 0.3125 | 0.3125 | 20 |
| D iter2 | 0.8810 | 0.3203 | 0.4062 | 0.6406 | 0.3203 | 0.3125 | 11 |

Iter2 dynamic OPD 分布：

- A：13 rows，`code=4,memory=7,tool=2`。
- B：13 rows，`code=2,memory=8,tool=3`。
- C：8 rows，`code=3,memory=3,tool=2`。
- D：11 rows，`code=3,memory=6,tool=2`。

解释：

- Tool reward 大幅上升，说明首轮 gate update 没有只是在无意义地推系数，而是实质改善了 proxy reward。
- Dynamic OPD 中 tool rows 从 iter1 的 6-12 降到 iter2 的 2-3，说明很多 tool all-fail prompt 已被修复。
- Memory 在 A/B/C 上上升，D 基本持平；Code 在 B/C/D 上不降反升，A 轻微下降。暂时没有观察到能力被明显牺牲。
- OPD rows 下降是正向信号，但也意味着后续 OPD 可用信号会减少，需要依靠 GRPO frontier 和 retention 继续校正。

## Iter2 Update 结果

iter2 update 约在 19:06 完成，四组均进入 iter3 bake/rollout。

| Run | Frontier rows | Dynamic OPD | Retention | `grad_norm_max` | `gate_delta_max` | Gate summary |
|---|---:|---|---:|---:|---:|---|
| A | 56 | code 4 / memory 7 / tool 2 | 0 | 5.4913 | 0.07198 | code 0.33382 / memory 0.44531 / tool 0.33500 |
| B | 57 | code 2 / memory 8 / tool 3 | 14 | 5.8888 | 0.07199 | code 0.33290 / memory 0.44532 / tool 0.33458 |
| C | 68 | code 3 / memory 3 / tool 2 | 0 | 2.7610 | 0.07340 | 591 params, min 0.33424 / mean 0.37156 / max 0.44754 |
| D | 59 | code 3 / memory 6 / tool 2 | 10 | 3.5661 | 0.07339 | 591 params, min 0.33389 / mean 0.37129 / max 0.44749 |

解释：

- 第二步仍有强梯度，未出现梯度消失。
- Global-coefficient A/B 的主变化是 memory 从 0.373 推到 0.445；tool/code 只小幅移动。这符合 iter2 OPD 中 tool all-fail 已显著减少、memory/code 成为主要剩余失败信号的现象。
- Global-parameter C/D 的 591 个参数均值从约 0.347 推到约 0.371，最大值到约 0.447；高容量版本也在持续上移。
- B/D 的 retention 没有阻止上涨，但也没有明显改变前两步主方向。后续要看 retention 是否能在更高系数区间保护 tool/code 不回撤。
- 当前实验继续运行；下一检查点是 iter3 rollout reward。如果 iter3 reward 继续提升，说明“dynamic OPD + no OPD length norm + epoch-step SGD”可以持续推动；如果 iter3 reward 回撤，则说明第二步可能已过推，需要调小 lr 或增加 retention/KL。

## Iter3-8 动态异常

截至 2026-05-14 晚间检查，A/C 已完成 iter7 update 并进入 iter8 update，B/D 已完成 iter5 update 并进入 iter6 update。代码路径没有出现 OOM/Traceback，但训练动态出现一致的能力崖：

| Run | Iter3 tool train / succ | Iter4 tool train / succ | Iter5 tool train / succ | Iter6 tool train / succ | 备注 |
|---|---:|---:|---:|---:|---|
| A global-coeff no-ret | 0.965 / 0.766 | 0.959 / 0.750 | 0.041 / 0.008 | 0.027 / 0.000 | iter5 起 tool collapse |
| B global-coeff ret | 0.959 / 0.742 | 0.956 / 0.742 | 0.056 / 0.016 | 0.027 / 0.000 | retention 未阻止 collapse |
| C global-param no-ret | 0.947 / 0.727 | 0.963 / 0.758 | 0.038 / 0.000 | 0.033 / 0.000 | 高容量版本同样 collapse |
| D global-param ret | 0.962 / 0.773 | 0.964 / 0.750 | 0.049 / 0.016 | 0.027 / 0.000 | retention 未阻止 collapse |

同时，memory reward 在 iter4-6 明显上升到约 0.79-0.84，code 基本维持在 0.36-0.43 区间。global-coefficient A 的 gate 从 iter4 的 `code=0.341,memory=0.639,tool=0.335` 继续到 iter6 的 `code=0.374,memory=0.704,tool=0.381`，说明主要风险不是 tool 系数不动，而是 memory/code/task-vector 组合跨过某个区间后破坏了 tool 输出格式。

抽样检查 A 的同一 tool prompt：

- iter4：输出 `<tool_call> ... </tool_call>`，reward=4，success=True。
- iter5：输出 `<tool_call>` 后缺少闭合，或者继续生成 `<response>`，reward=-3，success=False。
- iter8：仍然常见缺闭合或混入 `<response>`，说明 tool collapse 不是单轮采样噪声。

判断：

- 这版前 1-4 轮是有效训练：tool/memory/code proxy reward 均被推动，OPD rows 下降。
- 第 5 轮开始不是健康收敛，而是出现跨任务干扰导致的 tool 格式崖；A/B/C/D 同步出现，基本排除随机性。
- 去掉 OPD length normalization 确实恢复了梯度，但配合 `lr=0.04`、SGD momentum、sequence-level loss 与较弱 retention，会把 memory 方向推得过快；当前 retention 设置不足以约束 tool 格式保持。
- 当前 run 的价值主要是定位阈值：tool 在 iter3-4 最好，iter5 后显著坏。后续若复现，应优先保留 iter3/iter4 checkpoint 送评测，而不是使用最终 iter8。

下一步建议：

- 不把本轮最终 checkpoint 作为候选强模型；优先评测 iter3/iter4。
- 新一轮训练应加入 early-stop 或 per-task guard：当 tool reward 从上一轮下降超过阈值时停止或回滚。
- 参数上优先降低 `lr`/momentum，或对 memory/task-vector 总量加上更强 KL/retention；仅靠当前 dynamic OPD 无法在 tool collapse 后快速拉回。
