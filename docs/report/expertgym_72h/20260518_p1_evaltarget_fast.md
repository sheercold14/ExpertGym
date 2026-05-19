# ExpertGym 72h P1 eval-targeted fast 监控记录

## 监控对象

| run | run_dir | tmux |
|---|---|---|
| main_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518` | `train_eg72_main_gc_c033_fast_20260518` |
| main_global | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518` | `train_eg72_main_global_c033_fast_20260518` |
| init1_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518` | `train_eg72_main_gc_init1_fast_20260518` |
| opd_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_opd_gc_c033_evaltarget_fast_20260518` | `train_eg72_opd_gc_c033_fast_20260518` |
| monitor | `127.0.0.1:8796` | `opvec_monitor_eg72_p1_evaltarget_20260518` |

## 2026-05-18 02:36 CST 巡检

### 存活/GPU

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个 `train_eg72_*_fast_20260518` session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 34-49 C |
| GPU 显存 | 高占用且一致 | 8 卡约 67.8/81.6 GB |
| GPU 利用率 | 6 卡满载，2 卡瞬时空闲 | GPU 0/1/2/3/5/7 为 99-100%；GPU 4/6 为 0%，需继续观察是否为 rollout/同步阶段瞬时波动 |

### 迭代进度

| run | 当前目录进度 | summary/gates 状态 | tmux 近端证据 |
|---|---|---|---|
| main_gc | `iter_001` 已建 | `rollouts.summary.json`、`opd_distill_from_allfail.summary.json`、`gate_updates.summary.json`、`gate_updates.gates.json` 未落盘 | rollout 采集中，已见 `completed 10/48 prompts` |
| main_global | `iter_001` 已建 | 同上，未落盘 | rollout 采集中，已见 `completed 20/48 prompts` |
| init1_gc | `iter_001` 已建 | 同上，未落盘 | vLLM 初始化完成，近端未见错误 |
| opd_gc | `iter_001` 已建 | 同上，未落盘 | rollout 采集中，已见 `completed 10/48 prompts` |

### 异常/风险

| 类别 | 判断 | 处置 |
|---|---|---|
| 崩溃 | 暂未发现 | 不停止、不重启 |
| summary 缺失 | 当前不是异常，目录已建但 rollout 仍在进行 | 等待 iter_001 完整产物后再抽取 reward/frontier/OPD/gate/grad/timing |
| GPU 4/6 利用率 0% | 暂定为瞬时阶段差异；显存仍高占用，训练 session 存活 | 下一轮复查，若持续空闲且日志停滞再升级记录 |

### 待补指标

| 指标 | 触发条件 | 来源文件 |
|---|---|---|
| reward、frontier rows、timing | `rollouts.summary.json` 落盘 | `iter_xxx/rollouts.summary.json` |
| OPD selected_task_counts、retention rows | `opd_distill_from_allfail.summary.json` 落盘 | `iter_xxx/opd_distill_from_allfail.summary.json` |
| gate 变化、grad_norm | gate update 文件落盘 | `iter_xxx/gate_updates.summary.json`、`iter_xxx/gate_updates.gates.json` |
| iter_003 趋势判断 | 至少完整到 `iter_003` | 对比 `iter_001`-`iter_003` |

## 2026-05-18 02:42 CST 巡检

### 存活/GPU

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个 `train_eg72_*_fast_20260518` session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 30-55 C |
| GPU 利用率 | 阶段性轮换 | GPU 4/5 为 100%，GPU 3 为 70%，GPU 2 为 14%，GPU 0/1/6/7 瞬时 0%；结合显存与 tmux，未见全局停滞 |
| GPU 显存 | 正常波动 | 约 13.6-68.0/81.6 GB；部分卡已释放大模型显存，符合 rollout 完成后进入 update/等待的阶段差异 |

### 迭代/文件进度

| run | 当前完整度 | 已落盘 | 未落盘/待观察 |
|---|---|---|---|
| main_gc | `iter_001` rollout+OPD 完成，gate update 进行中 | `rollouts.summary.json`、`rollouts.jsonl`、`opd_distill_from_allfail.summary.json` | `gate_updates.summary.json`、`gate_updates.gates.json` |
| main_global | `iter_001` rollout+OPD 完成，gate update 进行中 | 同 main_gc | 同 main_gc；tmux 出现 PyTorch stream mismatch warning，暂未中断 |
| init1_gc | `iter_001` rollout shard 已完成，merge/summary 尚未落盘 | shard 端 stdout 显示 48/48 prompts 完成 | `rollouts.summary.json`、OPD、gate 文件；tmux 有 vLLM shutdown 后 `Engine core died unexpectedly`，因出现在 shutdown 之后且 session 存活，暂不判定崩溃 |
| opd_gc | `iter_001` rollout+OPD 完成，gate update 进行中 | `rollouts.summary.json`、`rollouts.jsonl`、`opd_distill_from_allfail.summary.json` | `gate_updates.summary.json`、`gate_updates.gates.json` |

### iter_001 rollout/frontier/reward

| run | rows | reward_mean | raw_mean | kept_frontiers | has_variance | all_failure | all_success | success/sample | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_gc | 96 | 0.4247 | 0.4492 | 52 | 54 | 41 | 12 | 129/384 | 413.6s |
| main_global | 96 | 0.4304 | 0.4832 | 56 | 56 | 39 | 12 | 132/384 | 413.5s |
| init1_gc | 待 merge | - | - | - | - | - | - | - | shard00 473.6s |
| opd_gc | 96 | 0.4468 | 0.3927 | 59 | 60 | 35 | 12 | 141/384 | 401.6s |

### iter_001 分任务 reward/frontier

| run | task | reward_mean | raw_mean | kept/rows | variance_rows | all_failure | all_success | success/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| main_gc | tool | 0.6736 | 0.7473 | 15/32 | 15 | 12 | 9 | 62/128 |
| main_gc | memory | 0.2734 | 0.2734 | 19/32 | 19 | 12 | 1 | 35/128 |
| main_gc | code | 0.3270 | 0.3270 | 18/32 | 20 | 17 | 2 | 32/128 |
| main_global | tool | 0.6866 | 0.8451 | 16/32 | 16 | 10 | 10 | 65/128 |
| main_global | memory | 0.3281 | 0.3281 | 23/32 | 23 | 8 | 1 | 42/128 |
| main_global | code | 0.2764 | 0.2764 | 17/32 | 17 | 21 | 1 | 25/128 |
| opd_gc | tool | 0.6378 | 0.4757 | 18/32 | 18 | 10 | 9 | 61/128 |
| opd_gc | memory | 0.3828 | 0.3828 | 22/32 | 22 | 8 | 2 | 49/128 |
| opd_gc | code | 0.3197 | 0.3197 | 19/32 | 20 | 17 | 1 | 31/128 |

### iter_001 OPD all-fail distill

| run | selected_rows | selected_task_counts | skipped.current_not_failure | skipped.no_expert_positive | current_max_success | 判断 |
|---|---:|---|---:|---:|---:|---|
| main_gc | 19 | code 5 / memory 10 / tool 4 | 55 | 22 | 0 | OPD rows 足够进入 update；memory 占比偏高但未单任务独占 |
| main_global | 18 | code 9 / memory 7 / tool 2 | 57 | 21 | 0 | OPD rows 足够；tool OPD 偏少，后续看 gate 是否压制/补偿 |
| init1_gc | 待产出 | - | - | - | - | 待 merge 后复查 |
| opd_gc | 14 | code 5 / memory 6 / tool 3 | 61 | 21 | 0 | OPD rows 可用但较少，仍覆盖三任务 |

### 当前判断

## 2026-05-18 04:22 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个 `train_eg72_*_fast_20260518` session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 状态 | 8 卡仍被现有训练/进程占用 | GPU 0/1、3、4/5、6/7 有训练相关显存；GPU 2 瞬时显存低但 main_global shard/merge 尚未完全结束 |
| baseline 调度 | 暂不抢卡 | WUDI/ExpertMerging 不重跑；Mixture/full-GRPO 已有 launcher，等卡释放后跑 smoke |

### 最新 next-rollout validation

候选口径仍是：`iter_k/gate_updates.gates.json` 由下一轮 `iter_{k+1}/rollouts.jsonl`
验证，不能直接看同轮 rollout。

| run | validated gate | validation rollout | overall | tool | memory | code | 判断 |
|---|---|---|---:|---:|---:|---:|---|
| main_gc | iter005 gate | iter006 rollout | 0.5426 | 0.8969 | 0.3906 | 0.3402 | 当前 global-coefficient 最佳点 |
| main_gc | iter006 gate | iter007 rollout | 0.5388 | 0.9015 | 0.4219 | 0.2930 | overall 小降，code 明显回落；不替换 iter005 gate |
| main_gc | iter007 gate | iter008 rollout | 0.5498 | 0.9063 | 0.4453 | 0.2979 | global-coefficient overall 新高，但 code 明显低于 iter005 gate |
| main_global | iter003 gate | iter004 rollout | 0.5539 | 0.9093 | 0.4219 | 0.3305 | 当前 1/3 主线最佳候选；common+residual 可解释 |
| main_global | iter004 gate | iter005 rollout | 0.5384 | 0.8991 | 0.3984 | 0.3178 | 低于 iter003 gate |
| main_global | iter005 gate | iter006 rollout | 0.5383 | 0.9108 | 0.4375 | 0.2666 | tool/memory 继续上升但 code 下滑，提示后续过推风险 |
| main_global | iter007 gate | iter008 rollout | 0.5674 | 0.9045 | 0.4922 | 0.3057 | common+residual overall 新高，memory 提升明显，但 code 低于 iter003 gate |
| init1_gc | iter003 gate | iter004 rollout | 0.7274 | 0.9292 | 0.8750 | 0.3779 | upper-init ablation 最佳点；不能作为主方法初始化证据 |
| opd_gc | iter004 gate | iter005 rollout | 0.5430 | 0.8586 | 0.4531 | 0.3174 | OPD-only 最佳防御候选 |
| opd_gc | iter006 gate | iter007 rollout | 0.4998 | 0.8940 | 0.3203 | 0.2852 | OPD-only 后续回落，不替换 iter004 gate |

补充 1：04:25 CST 后 `main_global/iter_007/rollouts.jsonl` 完整落盘，验证
`iter006 gate` 的 overall 为 `0.5486`，tool `0.9075` / memory `0.4297` /
code `0.3086`。它低于 `iter003 gate -> iter004 rollout` 的 `0.5539`，
因此 priority 1 不变。

补充 2：04:35 CST 后 `main_global/iter_008/rollouts.jsonl` 完整落盘，验证
`iter007 gate` 的 overall 为 `0.5674`，tool `0.9045` / memory `0.4922` /
code `0.3057`。这是 common+residual 当前 proxy 新高，但 code 低于
`iter003 gate` 的 `0.3305`，因此正式 eval 应成对比较 `main_global_i7`
和 `main_global_i3`。

### 当前候选排序

| priority | run | checkpoint | reason |
|---:|---|---|---|
| 1 | main_global | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_007/gate_updates.gates.json` | common+residual overall 新高，memory 提升明显；需验证 heldout code 是否被牺牲 |
| 2 | main_global | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_003/gate_updates.gates.json` | common+residual early stable point，code proxy 更高，和 iter007 gate 成对评测 |
| 3 | main_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_007/gate_updates.gates.json` | global-coefficient overall 新高，但需要验证 code 是否在 heldout 上受损 |
| 4 | main_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_005/gate_updates.gates.json` | global-coefficient code 更稳点，和 iter007 gate 成对评测 |
| 5 | opd_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_opd_gc_c033_evaltarget_fast_20260518/iter_004/gate_updates.gates.json` | OPD-only 防御 baseline，说明 offline recovery 可推动但后续不稳 |
| 6 | init1_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_003/gate_updates.gates.json` | init=1 upper-initialization ablation，验证强 task-vector 起点上限 |

### 风险判断

| 风险 | 观察 | 当前处理 |
|---|---|---|
| code 泛化不足 | main_global 后续 gate 提升 tool/memory 时 code proxy 从 0.3305/0.3178 降到 0.2666 | 正式 eval 优先评 iter003 gate，而不是 final gate |
| OPD-only 不稳定 | opd_gc iter005 最高后 iter006/007 下降 | 保留为防御实验，不作为主方法 |
| init1 不是主证据 | init1 proxy 明显高，但起点是强 task-vector 叠加 | 只作为 upper-init ablation |
| baseline 资源 | 当前 GPU 仍被四路训练占用 | 不抢占；等卡释放后启动 Mixture/full-GRPO smoke |

## 2026-05-18 04:45 CST 调度决策

### 停止低优先级后续训练

为了尽快拿主候选 heldout Eval6，而不是继续消耗 GPU 做低边际 rollout，已停止：

| run | stopped session | reason | 保留候选 |
|---|---|---|---|
| init1_gc | `train_eg72_main_gc_init1_fast_20260518` | upper-init ablation 已有最佳点，后续 iter005/006 未超过 iter004 proxy | `init1_gc_i3` |
| opd_gc | `train_eg72_opd_gc_c033_fast_20260518` | OPD-only 防御实验已有最佳点，后续回落后无新增主线价值 | `opd_gc_i4` |

继续保留：

| run | session | reason |
|---|---|---|
| main_global | `train_eg72_main_global_c033_fast_20260518` | 当前主线 common+residual 已出现新 proxy 高点，仍可能给出 final/late insight |
| main_gc | `train_eg72_main_gc_c033_fast_20260518` | global-coefficient 对照仍可观察是否恢复 code |

### 正式 Eval6 Dispatch

已启动 priority 1 `main_global_i7`：

| split | tmux | GPU | output |
|---|---|---|---|
| Tool/BFCL | `eval_p1_main_global_i7_tool_20260518` | 4 | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i7/tool` |
| Memory+Code | `eval_p1_main_global_i7_mc_20260518` | memory 5, code 6/7 | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i7/memory_code` |

下一步：`main_global_i7` 完成后回填 evaluation 表；若 Tool/BFCL 未占用，立刻启动
`main_global_i3` 对照。

| 问题 | 判断 |
|---|---|
| reward 是否有效上升 | 需等 `iter_002/003`，当前只有初始 iter_001 基线 |
| 是否 task collapse | iter_001 未见 collapse：三任务 rollout rows 均 32；frontier/OPD 覆盖三任务，但 code all_failure 偏高，main_global tool OPD 偏少 |
| gate 是否过冲 | gate update 产物未落盘，暂不能判断 |
| OPD rows 是否足够 | main_gc/main_global/opd_gc 分别 19/18/14，均可用；opd_gc 略少，需看 retention/gate update 是否稳定 |
| 崩溃风险 | 暂未发现需停止/重启的风险；init1_gc 的 vLLM shutdown 文案需下一轮确认是否继续推进 |

## 2026-05-18 02:49 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个训练 session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 31-59 C |
| GPU 利用率 | 正常阶段性轮换 | GPU 1 97%、GPU 4 66%、GPU 5 47%、GPU 0 5%，GPU 2/3 高显存低利用；GPU 6/7 已释放显存 |
| 当前 iter | main_global/opd_gc 已进入 `iter_002`；main_gc 完成 `iter_001` update；init1_gc 正在 `iter_001` update | 文件与 tmux 一致 |

### iter_001 gate/update 指标

| run | update 状态 | final_gates | gate_delta_max | grad_norm_max | optimizer_steps | updates | frontier rows | raw_frontier_task_counts | retention rows | retention_task_rows | OPD rows |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| main_gc | 完成 | code 0.3355 / memory 0.3433 / tool 0.3298 | 0.00998 | 0.05770 | 1 | 42 | 12 | code 18 / memory 19 / tool 15 | 11 | code 2 / memory 1 / tool 8 | 19 = code 5 / memory 10 / tool 4 |
| main_global | 完成 | common 0.3403 / code_res -0.0002 / memory_res +0.0070 / tool_res -0.0068 | 0.00702 | 0.06412 | 1 | 40 | 12 | code 17 / memory 23 / tool 16 | 10 | code 1 / memory 1 / tool 8 | 18 = code 9 / memory 7 / tool 2 |
| init1_gc | 进行中 | 待产出 | - | - | - | - | 待产出 | raw frontier 已知 kept 21 | 待产出 | 待产出 | 11 = code 9 / memory 1 / tool 1 |
| opd_gc | 完成 | code 0.3356 / memory 0.3425 / tool 0.3316 | 0.00920 | 0.05137 | 1 | 37 | 12 | code 19 / memory 22 / tool 18 | 11 | code 1 / memory 2 / tool 8 | 14 = code 5 / memory 6 / tool 3 |

### init1_gc iter_001 rollout/frontier/reward 补录

| run | rows | reward_mean | raw_mean | kept_frontiers | has_variance | all_failure | all_success | success/sample | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| init1_gc | 96 | 0.6923 | 1.1824 | 21 | 21 | 32 | 49 | 230/384 | 618.1s |

| run | task | reward_mean | raw_mean | kept/rows | variance_rows | all_failure | all_success | success/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| init1_gc | tool | 0.9294 | 2.3998 | 1/32 | 1 | 7 | 24 | 97/128 |
| init1_gc | memory | 0.8438 | 0.8438 | 4/32 | 4 | 4 | 24 | 108/128 |
| init1_gc | code | 0.3037 | 0.3037 | 16/32 | 16 | 21 | 1 | 25/128 |

### 当前判断

| 问题 | 判断 |
|---|---|
| reward 是否有效上升 | 仍需等 `iter_002/003` 对比；iter_001 中 init1_gc reward 明显高，但 frontier rows 仅 21，说明大量 tool/memory 已 all-success，不代表可训练信号更强 |
| 是否 task collapse | rollout rows 仍三任务均衡；训练 frontier 被 quota 固定为每任务 4。init1_gc 的 OPD rows 11 且 code 9/memory 1/tool 1，OPD 在 init1_gc 上明显偏 code，但这是由 tool/memory 高成功率造成，暂不等同 collapse |
| gate 是否过冲 | main_gc/main_global/opd_gc 的 `gate_delta_max` 均 < 0.01，grad_norm 正常，未见过冲 |
| OPD rows 是否足够 | main_gc/main_global/opd_gc 够用；init1_gc 仅 11 且偏 code，后续需看是否长期 OPD 稀疏 |
| 崩溃风险 | 未发现需停止/重启的风险；PyTorch stream mismatch warning 出现在 update backward，未导致 session 退出 |

## 2026-05-18 02:55 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个训练 session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 30-60 C |
| GPU 利用率 | 正常轮换 | GPU 1/5/6/7 约 98-100%，GPU 4 30%，GPU 0/2/3 低利用；无全局 idle |
| 当前 iter | main_gc/main_global/opd_gc 已进入 `iter_002`；init1_gc 仍在 `iter_001` update | 文件进度与 tmux 一致 |

### 文件进度

| run | 当前状态 | 新增产物 |
|---|---|---|
| main_gc | `iter_002` rollout 进行中 | `iter_001/gate_updates.summary.json`、`gate_updates.gates.json` 已确认 |
| main_global | `iter_002` rollout+OPD 完成，update 进行中 | `iter_002/rollouts.summary.json`、`opd_distill_from_allfail.summary.json` |
| init1_gc | `iter_001` update 仍在运行 | 暂无 gate 文件；tmux 停在 backward warning 后，未退出 |
| opd_gc | `iter_002` rollout 进行中 | 暂无 `iter_002` summary；tmux 已到 40/48 prompts |

### main_global iter_002 rollout/frontier/reward

| run | iter | rows | reward_mean | raw_mean | kept_frontiers | has_variance | all_failure | all_success | success/sample | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_global | 001 | 96 | 0.4304 | 0.4832 | 56 | 56 | 39 | 12 | 132/384 | 413.5s |
| main_global | 002 | 96 | 0.4867 | 0.6461 | 57 | 57 | 32 | 15 | 151/384 | 409.8s |

| run | iter | task | reward_mean | raw_mean | kept/rows | variance_rows | all_failure | all_success | success/sample |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| main_global | 002 | tool | 0.7305 | 1.2089 | 13/32 | 13 | 10 | 13 | 67/128 |
| main_global | 002 | memory | 0.3828 | 0.3828 | 24/32 | 24 | 7 | 1 | 49/128 |
| main_global | 002 | code | 0.3467 | 0.3467 | 20/32 | 20 | 15 | 1 | 35/128 |

### main_global iter_002 OPD

| run | iter | selected_rows | selected_task_counts | skipped.current_not_failure | skipped.no_current_negative | skipped.no_expert_positive | current_max_success | 判断 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| main_global | 002 | 9 | code 3 / memory 5 / tool 1 | 64 | 1 | 22 | 0 | OPD rows 较 iter_001 的 18 降低，和 reward/成功率上升一致；tool 仅 1 行，后续看是否持续稀疏 |

### 当前判断

| 问题 | 判断 |
|---|---|
| reward 是否有效上升 | main_global 从 iter_001 0.4304 升到 iter_002 0.4867，success 132/384 升到 151/384，初步有效上升；仍需 iter_003 验证 |
| 是否 task collapse | main_global iter_002 rows 仍三任务均衡；frontier task 中 memory/code 多于 tool，未见采样 collapse |
| gate 是否过冲 | main_global iter_002 gate update 未落盘；iter_001 无过冲 |
| OPD rows 是否足够 | main_global iter_002 OPD=9，低于 iter_001；仍覆盖三任务但 tool 稀疏，需关注 |
| init1_gc 风险 | update 时间偏长但 session 存活、GPU 有负载；未达到自行停止/重启条件，继续观察 |

## 2026-05-18 03:01 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个训练 session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 43-58 C |
| GPU 利用率 | 正常运行 | 0/3/4 高负载，5/6/7 中高负载，1/2 低利用但显存仍占用；无全局 idle |
| 当前 iter | 四路均已进入 `iter_002` | main_gc/main_global/opd_gc 已有 iter_002 rollout+OPD；init1_gc 已完成 iter_001 gate 并创建 iter_002 |

### iter_002 rollout/frontier/reward

| run | iter | rows | reward_mean | raw_mean | kept_frontiers | has_variance | all_failure | all_success | success/sample | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_gc | 001 | 96 | 0.4247 | 0.4492 | 52 | 54 | 41 | 12 | 129/384 | 413.6s |
| main_gc | 002 | 96 | 0.4346 | 0.4381 | 56 | 56 | 40 | 12 | 130/384 | 382.8s |
| main_global | 001 | 96 | 0.4304 | 0.4832 | 56 | 56 | 39 | 12 | 132/384 | 413.5s |
| main_global | 002 | 96 | 0.4867 | 0.6461 | 57 | 57 | 32 | 15 | 151/384 | 409.8s |
| opd_gc | 001 | 96 | 0.4468 | 0.3927 | 59 | 60 | 35 | 12 | 141/384 | 401.6s |
| opd_gc | 002 | 96 | 0.4274 | 0.4667 | 55 | 55 | 38 | 13 | 132/384 | 421.0s |
| init1_gc | 002 | 待产出 | - | - | - | - | - | - | - | - |

### iter_002 分任务 reward/frontier

| run | task | reward_mean | raw_mean | kept/rows | variance_rows | all_failure | all_success | success/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| main_gc | tool | 0.6572 | 0.6678 | 16/32 | 16 | 11 | 10 | 59/128 |
| main_gc | memory | 0.3750 | 0.3750 | 22/32 | 22 | 9 | 1 | 48/128 |
| main_gc | code | 0.2715 | 0.2715 | 18/32 | 18 | 20 | 1 | 23/128 |
| main_global | tool | 0.7305 | 1.2089 | 13/32 | 13 | 10 | 13 | 67/128 |
| main_global | memory | 0.3828 | 0.3828 | 24/32 | 24 | 7 | 1 | 49/128 |
| main_global | code | 0.3467 | 0.3467 | 20/32 | 20 | 15 | 1 | 35/128 |
| opd_gc | tool | 0.6759 | 0.7936 | 15/32 | 15 | 9 | 10 | 66/128 |
| opd_gc | memory | 0.3281 | 0.3281 | 22/32 | 22 | 9 | 1 | 42/128 |
| opd_gc | code | 0.2783 | 0.2783 | 18/32 | 18 | 20 | 2 | 24/128 |

### iter_002 OPD all-fail distill

| run | iter | selected_rows | selected_task_counts | skipped.current_not_failure | skipped.no_current_negative | skipped.no_expert_positive | current_max_success | 判断 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| main_gc | 002 | 17 | code 8 / memory 7 / tool 2 | 56 | 1 | 22 | 0 | OPD rows 足够；tool 偏少 |
| main_global | 002 | 9 | code 3 / memory 5 / tool 1 | 64 | 1 | 22 | 0 | OPD 稀疏，仍覆盖三任务 |
| opd_gc | 002 | 15 | code 8 / memory 7 / tool 0 | 58 | 1 | 22 | 0 | OPD rows 可用但 tool 为 0，需关注是否连续缺失 |
| init1_gc | 002 | 待产出 | - | - | - | - | - | 待观察 |

### init1_gc iter_001 gate/update 补录

| run | update 状态 | final_gates | gate_delta_max | grad_norm_max | optimizer_steps | updates | frontier rows | raw_frontier_task_counts | retention rows | retention_task_rows | OPD rows |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| init1_gc | 完成 | code 1.0042 / memory 0.9877 / tool 0.9935 | 0.01228 | 0.07727 | 1 | 37 | 9 | code 16 / memory 4 / tool 1 | 17 | code 1 / memory 8 / tool 8 | 11 = code 9 / memory 1 / tool 1 |

### 当前判断

| 问题 | 判断 |
|---|---|
| reward 是否有效上升 | main_global 明显上升；main_gc 小幅上升；opd_gc 从 0.4468 回落到 0.4274。总体不能下最终结论，需 iter_003 |
| 是否 task collapse | rollout rows 仍均衡；但 OPD 分布开始稀疏，opd_gc iter_002 tool=0、main_global tool=1，需要后续确认是否连续缺失 |
| gate 是否过冲 | init1_gc delta 0.0123 略高于其他路但仍小；已完成四路 iter_001 gate 均未见过冲 |
| OPD rows 是否足够 | main_gc/opd_gc iter_002 rows 15-17 足够；main_global 仅 9，偏低；opd_gc tool 缺失是当前最大分布风险 |
| 崩溃风险 | 无崩溃；init1_gc update 已完成，前一轮长 update 不是死锁 |

## 2026-05-18 03:08 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个训练 session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 29-47 C |
| GPU 利用率 | 正常运行 | GPU 0/1/2/3/6/7 为 100%，GPU 5 4%，GPU 4 0%；符合多路交替 rollout/update |
| 当前 iter | main_gc/main_global/opd_gc 已进入 `iter_003`；init1_gc 在 `iter_002` update 前后 | iter_003 目录已出现 3 路 |

### iter_002 gate/update 指标

| run | update 状态 | final_gates | gate_delta_max | grad_norm_max | optimizer_steps | updates | frontier rows | raw_frontier_task_counts | retention rows | retention_task_rows | OPD rows |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| main_gc | 完成 | code 0.3384 / memory 0.3543 / tool 0.3283 | 0.01102 | 0.04999 | 1 | 39 | 12 | code 18 / memory 22 / tool 16 | 10 | code 1 / memory 1 / tool 8 | 17 = code 8 / memory 7 / tool 2 |
| main_global | 完成 | common 0.3523 / code_res -0.0007 / memory_res +0.0110 / tool_res -0.0103 | 0.01194 | 0.05900 | 1 | 31 | 12 | code 20 / memory 24 / tool 13 | 10 | code 1 / memory 1 / tool 8 | 9 = code 3 / memory 5 / tool 1 |
| opd_gc | 完成 | code 0.3388 / memory 0.3515 / tool 0.3306 | 0.00900 | 0.04101 | 1 | 38 | 12 | code 18 / memory 22 / tool 15 | 11 | code 2 / memory 1 / tool 8 | 15 = code 8 / memory 7 / tool 0 |
| init1_gc | 待产出 | - | - | - | - | - | - | - | - | - | iter_002 OPD=9 |

### init1_gc iter_002 rollout/frontier/reward

| run | iter | rows | reward_mean | raw_mean | kept_frontiers | has_variance | all_failure | all_success | success/sample | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| init1_gc | 001 | 96 | 0.6923 | 1.1824 | 21 | 21 | 32 | 49 | 230/384 | 618.1s |
| init1_gc | 002 | 96 | 0.7108 | 1.1890 | 21 | 21 | 29 | 51 | 236/384 | 569.9s |

| run | iter | task | reward_mean | raw_mean | kept/rows | variance_rows | all_failure | all_success | success/sample |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| init1_gc | 002 | tool | 0.9262 | 2.3608 | 4/32 | 4 | 6 | 22 | 96/128 |
| init1_gc | 002 | memory | 0.8438 | 0.8438 | 3/32 | 3 | 4 | 25 | 108/128 |
| init1_gc | 002 | code | 0.3623 | 0.3623 | 14/32 | 14 | 19 | 4 | 32/128 |

### init1_gc iter_002 OPD

| run | iter | selected_rows | selected_task_counts | skipped.current_not_failure | skipped.no_current_negative | skipped.no_expert_positive | current_max_success | 判断 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| init1_gc | 002 | 9 | code 7 / memory 1 / tool 1 | 67 | 1 | 19 | 0 | 继续偏 code，OPD 总量偏少；tool/memory 高成功率导致可恢复 all-fail 少 |

### 当前判断

| 问题 | 判断 |
|---|---|
| reward 是否有效上升 | main_global、init1_gc 到 iter_002 呈上升；main_gc 小幅上升；opd_gc 回落。等待 iter_003 完整后给趋势结论 |
| 是否 task collapse | rollout 三任务仍均衡；frontier quota 三任务均 4（除 init1_gc iter_001 受可用 frontier 限制 tool 1）。OPD 分布有稀疏风险，但未见采样层 collapse |
| gate 是否过冲 | iter_002 delta 0.0090-0.0119，grad_norm 0.041-0.059；仍未见过冲 |
| OPD rows 是否足够 | main_gc/opd_gc 尚可；main_global 与 init1_gc iter_002 均为 9，偏少但可更新；opd_gc tool=0 是连续关注点 |
| 崩溃风险 | 无崩溃；六卡满载，训练推进到 iter_003 |

## 2026-05-18 03:14 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个训练 session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 31-58 C |
| GPU 利用率 | 正常运行 | GPU 2/6 满载，GPU 4/5/7 中高负载，GPU 0/1/3 低利用但显存占用；无全局 idle |
| 当前 iter | main_gc/main_global/opd_gc 已完成 `iter_003` rollout+OPD，正在 iter_003 update；init1_gc 仍在 `iter_002` update | 文件与 tmux 一致 |

### iter_003 rollout/frontier/reward

| run | iter | rows | reward_mean | raw_mean | kept_frontiers | has_variance | all_failure | all_success | success/sample | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_gc | 001 | 96 | 0.4247 | 0.4492 | 52 | 54 | 41 | 12 | 129/384 | 413.6s |
| main_gc | 002 | 96 | 0.4346 | 0.4381 | 56 | 56 | 40 | 12 | 130/384 | 382.8s |
| main_gc | 003 | 96 | 0.4354 | 0.4123 | 59 | 59 | 37 | 13 | 136/384 | 417.0s |
| main_global | 001 | 96 | 0.4304 | 0.4832 | 56 | 56 | 39 | 12 | 132/384 | 413.5s |
| main_global | 002 | 96 | 0.4867 | 0.6461 | 57 | 57 | 32 | 15 | 151/384 | 409.8s |
| main_global | 003 | 96 | 0.5255 | 0.9314 | 49 | 50 | 32 | 22 | 163/384 | 403.3s |
| opd_gc | 001 | 96 | 0.4468 | 0.3927 | 59 | 60 | 35 | 12 | 141/384 | 401.6s |
| opd_gc | 002 | 96 | 0.4274 | 0.4667 | 55 | 55 | 38 | 13 | 132/384 | 421.0s |
| opd_gc | 003 | 96 | 0.4755 | 0.5823 | 56 | 56 | 36 | 14 | 145/384 | 406.6s |
| init1_gc | 003 | 待产出 | - | - | - | - | - | - | - | - |

### iter_003 分任务 reward/frontier

| run | task | reward_mean | raw_mean | kept/rows | variance_rows | all_failure | all_success | success/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| main_gc | tool | 0.6506 | 0.5812 | 16/32 | 16 | 11 | 10 | 61/128 |
| main_gc | memory | 0.3828 | 0.3828 | 22/32 | 22 | 7 | 3 | 49/128 |
| main_gc | code | 0.2729 | 0.2729 | 21/32 | 21 | 19 | 0 | 26/128 |
| main_global | tool | 0.8631 | 2.0810 | 10/32 | 10 | 8 | 17 | 84/128 |
| main_global | memory | 0.4141 | 0.4141 | 20/32 | 20 | 7 | 5 | 53/128 |
| main_global | code | 0.2992 | 0.2992 | 19/32 | 20 | 17 | 0 | 26/128 |
| opd_gc | tool | 0.7127 | 1.0329 | 13/32 | 13 | 11 | 12 | 66/128 |
| opd_gc | memory | 0.3906 | 0.3906 | 24/32 | 24 | 7 | 1 | 50/128 |
| opd_gc | code | 0.3232 | 0.3232 | 19/32 | 19 | 18 | 1 | 29/128 |

### iter_003 OPD all-fail distill

| run | iter | selected_rows | selected_task_counts | skipped.current_not_failure | skipped.no_expert_positive | current_max_success | 判断 |
|---|---:|---:|---|---:|---:|---:|---|
| main_gc | 003 | 16 | code 7 / memory 6 / tool 3 | 59 | 21 | 0 | OPD rows 足够，三任务覆盖恢复 |
| main_global | 003 | 9 | code 5 / memory 4 / tool 0 | 64 | 22 | 0 | OPD 仍偏少且 tool 缺失；tool reward 已高，缺失可解释但需防止长期无 OPD tool |
| opd_gc | 003 | 15 | code 6 / memory 6 / tool 3 | 60 | 21 | 0 | OPD rows 足够，tool 覆盖从 iter_002 的 0 恢复到 3 |
| init1_gc | 003 | 待产出 | - | - | - | - | 待观察 |

### iter_003 趋势判断

| 问题 | main_gc | main_global | opd_gc | init1_gc |
|---|---|---|---|---|
| reward 是否有效上升 | 弱上升：0.4247 -> 0.4346 -> 0.4354，success 129 -> 130 -> 136；主要来自 memory 改善，code 仍弱 | 明显上升：0.4304 -> 0.4867 -> 0.5255，success 132 -> 151 -> 163；tool 拉动显著，memory/code 小幅波动 | 先降后升：0.4468 -> 0.4274 -> 0.4755，success 141 -> 132 -> 145；iter_003 已高于基线 | 仅到 iter_002：0.6923 -> 0.7108，success 230 -> 236；tool/memory 接近饱和，code 是主要剩余难点 |
| 是否 task collapse | 未见采样 collapse，rows 仍 32/32/32；code all_failure 仍高 | 未见采样 collapse；tool 成功率快速升高导致 frontier/OPD tool 变少 | 未见采样 collapse；iter_003 OPD 三任务覆盖恢复 | 未见采样 collapse；但 OPD 长期偏 code，因 tool/memory all-success 高 |
| gate 是否过冲 | 到 iter_002 delta <= 0.0110，未见过冲；iter_003 update 正在跑 | 到 iter_002 delta <= 0.0119，未见过冲；global common 持续上调，需看 iter_003 gate | 到 iter_002 delta <= 0.0090，未见过冲；iter_003 update 正在跑 | iter_001 delta 0.0123，略高但可接受；iter_002 gate 待产出 |
| OPD rows 是否足够 | 足够：19 -> 17 -> 16，且 iter_003 三任务覆盖 | 偏少：18 -> 9 -> 9，iter_003 tool=0；可更新但需重点关注 | 足够：14 -> 15 -> 15，iter_003 三任务覆盖 | 偏少：11 -> 9，且 code 占主导；等 iter_003 |
| 当前优先关注 | code reward 未改善，gate memory 上升可能挤压 code/tool | tool 可能过快饱和，OPD tool 缺失；需看 gate 是否继续抬 common/压 tool_res | iter_002 回落已恢复，继续看稳定性 | update 速度慢于其他三路，等 iter_002 gate 和 iter_003 rollout |

### 异常/风险

| 类别 | 判断 | 处置 |
|---|---|---|
| 崩溃 | 暂未发现 | 不停止、不重启 |
| warning | 多路 update 出现 PyTorch `AccumulateGrad stream mismatch` warning | 未导致退出，继续记录 |
| 训练节奏 | init1_gc 明显慢于其他三路 | 继续只读观察，不干预 |

## 2026-05-18 03:21 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个训练 session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 27-52 C |
| GPU 利用率 | 正常轮换 | GPU 0/4/5 满载，GPU 1 中显存低利用，GPU 2/3/6/7 已释放显存 |
| 当前 iter | main_global/opd_gc 已进入 `iter_004`；init1_gc 进入 `iter_003`；main_gc 仍在 `iter_003` update | 无崩溃 |

### 新增 gate/update 指标

| run | iter | update 状态 | final_gates | gate_delta_max | grad_norm_max | optimizer_steps | updates | frontier rows | raw_frontier_task_counts | retention rows | retention_task_rows | OPD rows |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| main_global | 003 | 完成 | common 0.3640 / code_res -0.0014 / memory_res +0.0172 / tool_res -0.0158 | 0.01175 | 0.06295 | 1 | 34 | 12 | code 19 / memory 20 / tool 10 | 13 | code 0 / memory 5 / tool 8 | 9 = code 5 / memory 4 / tool 0 |
| opd_gc | 003 | 完成 | code 0.3420 / memory 0.3631 / tool 0.3279 | 0.01158 | 0.05549 | 1 | 37 | 12 | code 19 / memory 24 / tool 13 | 10 | code 1 / memory 1 / tool 8 | 15 = code 6 / memory 6 / tool 3 |
| init1_gc | 002 | 完成 | code 1.0092 / memory 0.9724 / tool 0.9854 | 0.01535 | 0.08106 | 1 | 40 | 11 | code 14 / memory 3 / tool 4 | 20 | code 4 / memory 8 / tool 8 | 9 = code 7 / memory 1 / tool 1 |
| main_gc | 003 | 进行中 | 待产出 | - | - | - | - | - | - | - | - | iter_003 OPD=16 |

### 当前判断补充

| 问题 | 判断 |
|---|---|
| gate 是否过冲 | main_global/opd_gc iter_003 delta 约 0.0116-0.0117，仍可控；init1_gc iter_002 delta 0.01535、grad 0.08106，为目前最大但未见异常跳变 |
| gate 方向 | main_global common 持续上升，memory_res 上升、tool_res 下降；opd_gc memory gate 持续上升、tool 轻微下降；init1_gc code 上升、memory/tool 下降，符合 code 难点被加强 |
| OPD/retention | main_global iter_003 OPD tool=0 且 retention code=0，信号进一步偏 memory/code；opd_gc OPD 覆盖正常；init1_gc OPD 继续偏 code |
| 崩溃风险 | 仍无崩溃；未停止或重启任何进程 |

## 2026-05-18 03:27 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个训练 session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 温度 | 正常 | H800 31-49 C |
| GPU 利用率 | 正常轮换 | GPU 0/2/3/6 约 98-99%，GPU 1/5/7 高显存低利用，GPU 4 已释放 |
| 当前 iter | main_gc/main_global/opd_gc 均已进入 `iter_004`；init1_gc 在 `iter_003`，summary 未落盘 | 无崩溃 |

### main_gc iter_003 gate/update 补录

| run | iter | update 状态 | final_gates | gate_delta_max | grad_norm_max | optimizer_steps | updates | frontier rows | raw_frontier_task_counts | retention rows | retention_task_rows | OPD rows |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| main_gc | 003 | 完成 | code 0.3420 / memory 0.3640 / tool 0.3263 | 0.00963 | 0.04379 | 1 | 39 | 12 | code 21 / memory 22 / tool 16 | 11 | code 0 / memory 3 / tool 8 | 16 = code 7 / memory 6 / tool 3 |

### 当前判断补充

| 问题 | 判断 |
|---|---|
| gate 是否过冲 | main_gc iter_003 delta 0.00963、grad 0.04379；四路已知 gate update 均未见过冲 |
| gate 方向 | main_gc memory gate 继续上升到 0.3640，tool 下降到 0.3263，code 小幅升到 0.3420；与 memory reward 改善、code 仍弱一致 |
| OPD/retention | main_gc iter_003 OPD 覆盖三任务；retention code=0，说明 code all-success retention 不足，后续看 code reward 是否恢复 |
| init1_gc | 已有 `iter_003` 目录但 summary 未落盘；仍慢于其他三路，未见崩溃 |
| 崩溃风险 | 无崩溃；未停止或重启任何进程 |

## 2026-05-18 02:58 CST 巡检

### 存活/GPU/进度

| 项 | 状态 | 证据 |
|---|---:|---|
| 训练 tmux | 4/4 alive | 四个 `train_eg72_*_fast_20260518` session 均存在 |
| 前端 tmux | alive | `opvec_monitor_eg72_p1_evaltarget_20260518` 存在 |
| GPU 利用 | 正常轮换 | GPU 0/1/7 高显存；GPU 2/3 正在 update；GPU 4/5 在 init1 bake/collect 过渡后继续推进 |
| 当前 iter | main_gc/opd_gc 进入 `iter_002` update；main_global 进入 `iter_002` update；init1_gc 完成 `iter_001` update 并已写出 `iter_002/baked_policy` |

### iter_002 rollout 已知对比

| run | iter | reward_mean | raw_mean | frontier | variance_rows | all_failure | all_success | success/sample | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| main_gc | 001 | 0.4247 | 0.4492 | 52/96 | 54 | 41 | 12 | 129/384 | 初始点 |
| main_gc | 002 | 0.4346 | 0.4381 | 56/96 | 56 | 40 | 12 | 130/384 | overall 微升；memory 升、code 降 |
| main_global | 001 | 0.4304 | 0.4832 | 56/96 | 56 | 39 | 12 | 132/384 | 初始点 |
| main_global | 002 | 0.4867 | 0.6461 | 57/96 | 57 | 32 | 15 | 151/384 | 最明显有效上升，三任务都有正向变化 |
| opd_gc | 001 | 0.4468 | 0.3927 | 59/96 | 60 | 35 | 12 | 141/384 | 初始点 |
| opd_gc | 002 | 0.4274 | 0.4667 | 55/96 | 55 | 38 | 13 | 132/384 | OPD-only 方向暂不稳，需看 iter2 update/iter3 |

### 分任务变化

| run | iter | tool reward | memory reward | code reward | 主要信号 |
|---|---:|---:|---:|---:|---|
| main_gc | 001 -> 002 | 0.6736 -> 0.6572 | 0.2734 -> 0.3750 | 0.3270 -> 0.2715 | memory 明显上升，code 被牺牲 |
| main_global | 001 -> 002 | 0.6866 -> 0.7305 | 0.3281 -> 0.3828 | 0.2764 -> 0.3467 | 三任务同步上升，是当前最优先保留候选 |
| opd_gc | 001 -> 002 | 0.6378 -> 0.6759 | 0.3828 -> 0.3281 | 0.3197 -> 0.2783 | tool 上升但 memory/code 下降，说明无 GRPO 时 OPD/retention 不够稳 |

### init1_gc iter_001 update 补录

| run | final_gates | frontier_task_counts | retention_rows | OPD rows | updates | 判断 |
|---|---|---|---:|---:|---:|---|
| init1_gc | code 1.0042 / memory 0.9877 / tool 0.9935 | code 4 / memory 4 / tool 1 | 17 | 11 | 37 | init1 高初始能力导致 tool/memory frontier 稀疏；第一步主要推 code、略压 memory/tool |

### 当前判断

| 问题 | 判断 |
|---|---|
| 是否停止无效 run | 暂不停止。`opd_gc` iter2 下降但只完成 rollout，需看 update 后 iter3；`main_gc` 轻微上升；`main_global` 明显有希望 |
| 是否启动新 baseline | 暂不启动 GPU-heavy baseline。当前四路训练仍在占用/即将占用 8 卡；WUDI/ExpertMerging 已记录本地目录，Fisher/Mixture 需要独立复现计划 |
| 优先候选 | `main_global` 当前最符合论文主线：global common+residual 可解释、三任务 reward 同步上升、OPD rows 随成功率上升而自然减少 |

## 2026-05-18 03:03 CST 巡检

### iter_002 update 新进展

| run | iter | update 状态 | final_gates | OPD selected | 判断 |
|---|---:|---|---|---|---|
| main_global | 002 | 完成 | common 0.3523 / code_res -0.0007 / memory_res +0.0110 / tool_res -0.0103 | code 3 / memory 5 / tool 1 | 当前最强候选：reward 已在 rollout 侧三任务同步上升，gate 继续沿 common+memory_residual 上升 |
| main_gc | 002 | 进行中 | 待产出 | code 8 / memory 7 / tool 2 | rollout overall 微升但 code 下降；等 gate update 后看是否修复 |
| init1_gc | 002 | rollout/bake 阶段 | 待产出 | 待产出 | init1 第一轮 gate 主要推 code、压 memory/tool；因 tool/memory all-success 多，训练信号稀疏 |
| opd_gc | 002 | 进行中 | 待产出 | code 8 / memory 7 / tool 0 | OPD-only 这一轮没有 tool OPD，且 rollout reward 下降，作为防御实验继续观察但主线优先级降低 |

### main_global gate 轨迹

| iter | common | code_residual | memory_residual | tool_residual | effective code | effective memory | effective tool |
|---:|---:|---:|---:|---:|---:|---:|---:|
| init | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.3333 | 0.3333 |
| 001 | 0.3403 | -0.0002 | +0.0070 | -0.0068 | 0.3401 | 0.3474 | 0.3335 |
| 002 | 0.3523 | -0.0007 | +0.0110 | -0.0103 | 0.3516 | 0.3633 | 0.3420 |

### 判断

| 问题 | 判断 |
|---|---|
| 主线候选 | `main_global` 目前最值得保留到完整 12 iter 或至少送中间 checkpoint 评测；它比纯 global coefficient 更有可解释分解：common 承担整体提升，memory residual 额外补偿，tool residual 抑制但 effective tool 仍小幅上升 |
| OPD-only 价值 | `opd_gc` 更适合作为“只靠 offline distillation 不稳定”的防御实验；iter2 rollout 下降，且 tool OPD=0，后续若 iter3 仍下降可考虑提前停止以释放 GPU |
| init1 价值 | `init1_gc` 是高初始能力对照，但可训练 frontier 明显稀疏；如果后续 code 不明显补齐，它更像 upper-initialization ablation，不适合作为主方法 |

## 2026-05-18 03:08 CST 巡检

### 关键进展

| run | 最新完整 iter | latest gates | latest rollout | 判断 |
|---|---:|---|---|---|
| main_gc | iter_002 update 完成 | code 0.3384 / memory 0.3543 / tool 0.3283 | iter002 reward 0.4346 | memory/code 被推高，tool 持续轻压；overall 只小幅升 |
| main_global | iter_003 rollout 完成，update 进行中 | iter002: common 0.3523 / code_res -0.0007 / memory_res +0.0110 / tool_res -0.0103 | iter003 reward 0.5255, success 163/384 | 目前最强主线候选，reward 已连续两轮上升 |
| init1_gc | iter_002 rollout 完成，update 进行中 | iter001: code 1.0042 / memory 0.9877 / tool 0.9935 | iter002 reward 0.7108 | 高初始点继续高 reward，但 frontier 稀疏，论文主线价值偏 ablation |
| opd_gc | iter_002 update 完成 | code 0.3388 / memory 0.3515 / tool 0.3306 | iter002 reward 0.4274 | OPD-only 在 iter2 下降，但 gate 仍推 memory/code；继续作为防御对照 |

### main_global reward 轨迹

| iter | overall reward | raw reward | frontier | all_failure | all_success | success/sample | OPD selected |
|---:|---:|---:|---:|---:|---:|---:|---|
| 001 | 0.4304 | 0.4832 | 56/96 | 39 | 12 | 132/384 | code 9 / memory 7 / tool 2 |
| 002 | 0.4867 | 0.6461 | 57/96 | 32 | 15 | 151/384 | code 3 / memory 5 / tool 1 |
| 003 | 0.5255 | 0.9314 | 49/96 | 32 | 22 | 163/384 | code 5 / memory 4 / tool 0 |

### 判断

| 问题 | 判断 |
|---|---|
| 是否已经有主实验雏形 | 是。`main_global` 在 common+residual 参数化下连续提升，且不是只提升单一任务；这是目前最能支撑“executable feedback learns composition”的曲线 |
| 需要关注的风险 | `main_global` iter003 OPD 已无 tool 行，说明 tool 主要靠 GRPO/frontier/retention 维护；后续若 tool reward 回落，需要提前选取 best checkpoint 而不是盲目跑满 |
| 下一步 | 等 `main_global` iter003 update 后继续看 iter004 rollout；如果仍上升，保留 run 到 12 iter，并把 iter003/004 作为中间候选 checkpoint 标记 |

## 2026-05-18 03:18 CST 巡检

### 最新状态

| run | 最新完整 iter | latest gates | latest rollout reward | 判断 |
|---|---:|---|---:|---|
| main_gc | iter003 update 完成 | code 0.3420 / memory 0.3640 / tool 0.3263 | 0.4354 | memory 持续上升，code/tool 弱；overall 基本平台 |
| main_global | iter003 update 完成 | common 0.3640 / code_res -0.0014 / memory_res +0.0172 / tool_res -0.0158 | 0.5255 | 主候选继续成立；overall 连续上升，但 code 在 iter003 回落 |
| init1_gc | iter002 update 完成 | code 1.0092 / memory 0.9724 / tool 0.9854 | 0.7108 | 高初始点 reward 高，code 有补齐趋势；但 memory/tool 被压，主要做 init upper ablation |
| opd_gc | iter003 update 完成 | code 0.3420 / memory 0.3631 / tool 0.3279 | 0.4755 | OPD-only iter003 反弹，仍可继续作为防御曲线 |

### main_global 轨迹细分

| iter | overall | tool | memory | code | success/sample | gates / effective coeff |
|---:|---:|---:|---:|---:|---:|---|
| 001 | 0.4304 | 0.6866 | 0.3281 | 0.2764 | 132/384 | effective code 0.3401 / memory 0.3474 / tool 0.3335 |
| 002 | 0.4867 | 0.7305 | 0.3828 | 0.3467 | 151/384 | effective code 0.3516 / memory 0.3633 / tool 0.3420 |
| 003 | 0.5255 | 0.8631 | 0.4141 | 0.2992 | 163/384 | effective code 0.3626 / memory 0.3812 / tool 0.3483 |

### 当前判断

| 问题 | 判断 |
|---|---|
| 是否过拟合到 tool | 有风险。iter003 overall 上升主要来自 tool + memory，code 从 iter002 回落；后续选择 checkpoint 必须看 code 不再继续掉 |
| 是否停止任何 run | 暂不停止。四路都还在 5 小时预算内，且 `opd_gc` iter003 反弹；继续到至少 iter004/005 再决定 |
| 评测候选 | 暂记 `main_global` iter003 为第一个强候选；若 iter004/005 code 不恢复，需要同时保留 iter002 作对照候选 |

## 2026-05-18 03:29 CST 巡检

### 最新 rollout 趋势

| run | latest rollout iter | overall | tool | memory | code | success/sample | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| main_gc | 004 | 0.4779 | 0.8187 | 0.3594 | 0.2557 | 148/384 | overall 上升主要来自 tool，code 继续偏低 |
| main_global | 004 | 0.5539 | 0.9093 | 0.4219 | 0.3305 | 171/384 | 主候选强化：overall 继续升，code 从 iter003 回升 |
| init1_gc | 003 | 0.6764 | 0.9247 | 0.8203 | 0.2842 | 226/384 | 从 iter002 回落；高初始点仍强但 code 不稳 |
| opd_gc | 004 | 0.4808 | 0.7851 | 0.3438 | 0.3135 | 147/384 | OPD-only 继续反弹，但 memory 下降，整体弱于 main_global |

### main_global 主线曲线

| rollout iter | overall | tool | memory | code | all_failure | all_success | OPD selected |
|---:|---:|---:|---:|---:|---:|---:|---|
| 001 | 0.4304 | 0.6866 | 0.3281 | 0.2764 | 39 | 12 | code 9 / memory 7 / tool 2 |
| 002 | 0.4867 | 0.7305 | 0.3828 | 0.3467 | 32 | 15 | code 3 / memory 5 / tool 1 |
| 003 | 0.5255 | 0.8631 | 0.4141 | 0.2992 | 32 | 22 | code 5 / memory 4 / tool 0 |
| 004 | 0.5539 | 0.9093 | 0.4219 | 0.3305 | 36 | 27 | code 6 / memory 8 / tool 0 |

### 判断

| 问题 | 判断 |
|---|---|
| main_global 是否牺牲 code | 目前不像单调牺牲。iter003 code 回落后，iter004 回升到 0.3305，接近 iter002 的 0.3467；但仍需正式 CURE eval 验证 |
| 是否需要提前评测 | 需要排队但不立即抢卡。`main_global` iter003 gate 已产生，并由 iter004 rollout 验证继续上升，是目前第一候选；等待训练到 iter005/006 或出现回落后再切评测更划算 |
| OPD 分布风险 | main_global iter003/004 OPD tool=0，说明 tool 不是由 OPD 直接推，而由 GRPO/frontier/retention 维护；这有利于论文区分 OPD-only，但需要报告中明确说明 |

## 2026-05-18 03:39 CST 巡检

### 最新状态

| run | latest complete | latest gate | latest rollout | 判断 |
|---|---|---|---|---|
| main_gc | iter004 update | code 0.3459 / memory 0.3740 / tool 0.3271 | iter004 overall 0.4779 | overall 升但 code 降到 0.2557，偏 tool/memory |
| main_global | iter004 update | common 0.3745 / code_res -0.0017 / memory_res +0.0228 / tool_res -0.0212 | iter004 overall 0.5539 | 主线第一候选；effective code/memory/tool 均高于 1/3 |
| init1_gc | iter003 update | code 1.0154 / memory 0.9576 / tool 0.9768 | iter003 overall 0.6764 | init1 从 iter002 回落，code 不稳；仅作 upper-init ablation |
| opd_gc | iter005 rollout | iter004: code 0.3457 / memory 0.3703 / tool 0.3285 | iter005 overall 0.5430 | OPD-only 反弹明显，适合作为“OPD 能提供动力但非主方法”的防御对照 |

### 关键曲线对比

| run | iter001 | iter002 | iter003 | iter004 | iter005 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| main_gc overall | 0.4247 | 0.4346 | 0.4354 | 0.4779 | pending | 升幅主要由 tool 带来，code 低 |
| main_global overall | 0.4304 | 0.4867 | 0.5255 | 0.5539 | pending | 当前最稳定主线 |
| init1_gc overall | 0.6923 | 0.7108 | 0.6764 | pending | pending | 高初始点但不稳定 |
| opd_gc overall | 0.4468 | 0.4274 | 0.4755 | 0.4808 | 0.5430 | OPD-only 先降后升 |

### 当前判断

| 问题 | 判断 |
|---|---|
| 是否提前停 main_gc | 暂不立即停。虽然 code 下降，但它提供 pure global-coefficient 对照；若 iter005 仍只涨 tool、code 继续 <0.28，可考虑提前结束并保留到 iter004 |
| 是否提前停 init1_gc | 倾向继续到 iter004 再判断。它是 init1 upper ablation，训练信号稀疏但有论文价值 |
| 是否提前评测 | 不抢当前训练卡。候选队列中保留 `main_global` iter003/iter004；等 run 到 iter005/006 或出现明显回落后再启动正式 eval6 |

## 2026-05-18 03:51 CST 巡检

### 最新 rollout/update

| run | latest complete | latest gate | latest rollout | 判断 |
|---|---|---|---|---|
| main_gc | iter005 update | code 0.3493 / memory 0.3835 / tool 0.3263 | iter005 overall 0.5309 | code 从低点恢复到 0.3227，成为 global-coefficient 强候选 |
| main_global | iter005 update | common 0.3886 / code_res -0.0028 / memory_res +0.0286 / tool_res -0.0258 | iter005 overall 0.5384 | 从 iter004 0.5539 小幅回落，但仍是主线强候选 |
| init1_gc | iter004 rollout，update 进行中 | iter003: code 1.0154 / memory 0.9576 / tool 0.9768 | iter004 overall 0.7274 | init1 反弹，code 0.3779；但不是主方法初始化 |
| opd_gc | iter005 update | code 0.3496 / memory 0.3778 / tool 0.3292 | iter005 overall 0.5430 | OPD-only 已追上 main_global proxy，需正式 eval 验证是否只是 proxy 上升 |

### 最新曲线

| run | iter001 | iter002 | iter003 | iter004 | iter005 | 主要风险 |
|---|---:|---:|---:|---:|---:|---|
| main_gc | 0.4247 | 0.4346 | 0.4354 | 0.4779 | 0.5309 | tool 拉升明显，tool coeff 被压，code 仍低于 0.327 起点 |
| main_global | 0.4304 | 0.4867 | 0.5255 | 0.5539 | 0.5384 | iter005 小回落，tool OPD 长期稀疏 |
| init1_gc | 0.6923 | 0.7108 | 0.6764 | 0.7274 | pending | 高初始点，不说明从 weak prior 学 composition |
| opd_gc | 0.4468 | 0.4274 | 0.4755 | 0.4808 | 0.5430 | OPD-only 曲线不稳，可能更像 distillation baseline |

### 判断

| 问题 | 判断 |
|---|---|
| 是否停止 main_global | 不停止。iter005 小回落但仍显著高于起点，且 common+residual 是最贴近论文 claim 的参数化 |
| 是否将 OPD-only 当主方法 | 不。`opd_gc` proxy 已追上，但它缺少 GRPO/frontier 的 on-policy 解释，适合作防御实验：OPD 有帮助，但主方法需要 executable feedback |
| 是否评测多个候选 | 是。正式 eval6 至少应包含 `main_global` best、`main_gc` best、`opd_gc` best；否则无法支撑参数化/目标函数 ablation |

### Checkpoint 选择口径修正

`iter_k/gate_updates.gates.json` 会用于下一轮 `iter_{k+1}` rollout。
因此候选 checkpoint 的 proxy 应按下一轮 rollout 读数判断：

| candidate gate | validating rollout | validating overall | 结论 |
|---|---|---:|---|
| main_global iter003 gate | main_global iter004 rollout | 0.5539 | 当前第一候选 |
| main_global iter004 gate | main_global iter005 rollout | 0.5384 | 高但低于 iter003 gate |
| main_gc iter004 gate | main_gc iter005 rollout | 0.5309 | global-coefficient 强对照 |
| opd_gc iter004 gate | opd_gc iter005 rollout | 0.5430 | OPD-only 强对照 |

## 2026-05-18 03:59 CST 巡检

### 新增确认

| run | latest complete | latest gate | latest rollout | 判断 |
|---|---|---|---|---|
| init1_gc | iter004 update | code 1.0208 / memory 0.9420 / tool 0.9689 | iter004 overall 0.7274 | init1 best proxy 对应 iter003 gate；iter004 gate 需等 iter005 验证 |
| opd_gc | iter006 rollout，update 进行中 | iter005: code 0.3496 / memory 0.3778 / tool 0.3292 | iter006 overall 0.5028 | OPD-only 从 iter005 0.5430 回落，best 仍是 iter004 gate |

### Checkpoint 口径补充

| candidate gate | validating rollout | validating overall | 结论 |
|---|---|---:|---|
| init1_gc iter003 gate | init1_gc iter004 rollout | 0.7274 | init1 upper-init 第一候选 |
| init1_gc iter004 gate | init1_gc iter005 rollout | pending | 暂不作为正式候选 |
| opd_gc iter004 gate | opd_gc iter005 rollout | 0.5430 | OPD-only 第一候选 |
| opd_gc iter005 gate | opd_gc iter006 rollout | 0.5028 | 低于 iter004 gate |

### 判断

| 问题 | 判断 |
|---|---|
| OPD-only 是否继续上升 | 否，iter006 回落；这支持“OPD 能推动但不稳定，需要 on-policy feedback/retention/frontier”这个论文防御点 |
| init1 是否可作为主方法 | 不适合。虽然 proxy 最高，但它从强初始点出发，tool/memory 大量 all-success，不能回答从 structured prior 学 composition 的核心问题 |

## 2026-05-18 04:10 CST 巡检

### 最新趋势

| run | latest rollout | overall | tool | memory | code | 判断 |
|---|---:|---:|---:|---:|---:|---|
| main_gc | iter006 | 0.5426 | 0.8969 | 0.3906 | 0.3402 | pure global-coefficient 新最佳，code 已恢复并超过起点 |
| main_global | iter006 | 0.5383 | 0.9108 | 0.4375 | 0.2666 | overall 高但 code 明显回落；best 仍是 iter003 gate |
| init1_gc | iter005 | 0.7201 | 0.9255 | 0.8672 | 0.3676 | 低于 iter004；best 仍是 iter003 gate |
| opd_gc | iter006 | 0.5028 | 0.8624 | 0.3438 | 0.3021 | 从 iter005 回落；best 仍是 iter004 gate |

### Checkpoint 口径更新

| candidate gate | validating rollout | validating overall | 结论 |
|---|---|---:|---|
| main_global iter003 gate | main_global iter004 rollout | 0.5539 | main_global 第一候选 |
| main_gc iter005 gate | main_gc iter006 rollout | 0.5426 | global-coefficient 第一候选 |
| opd_gc iter004 gate | opd_gc iter005 rollout | 0.5430 | OPD-only 第一候选 |
| init1_gc iter003 gate | init1_gc iter004 rollout | 0.7274 | init1 upper-init 第一候选 |

### 判断

| 问题 | 判断 |
|---|---|
| main_global 是否继续主线第一 | 是，但必须用 iter003 gate 评测；后续 gate 虽然 common 继续涨，但 code 可能被压，说明不要盲目选最后一轮 |
| main_gc 是否应纳入正式 eval | 是。iter006 proxy 已与 main_global 接近，且 code 更健康；这是参数化对照必须评测 |
| OPD-only 是否足够作为防御 | 是。iter005 有强 proxy，iter006 回落，正好说明 OPD-only 能推但不稳定，不能替代 on-policy GRPO 主线 |
