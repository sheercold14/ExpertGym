# 2026-05-18 P1 Eval-Targeted Candidate Queue

本文件只登记待正式 eval6 的候选 checkpoint，不代表已经完成评测。

配置来源：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260518_eg72_p1_evaltarget_fast.md`
- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/expertgym_72h/20260518_p1_evaltarget_fast.md`

## Candidate Checkpoints

| priority | run | checkpoint | reason | proxy caveat | eval status |
|---:|---|---|---|---|---|
| 1 | main_global | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_007/gate_updates.gates.json` | common+residual 新 proxy 高点：该 gate 生成 iter008 proxy 0.5674，tool 0.9045 / memory 0.4922 / code 0.3057 | code 低于 iter003 gate；必须和 main_global_i3 成对评测 heldout | Tool+Memory+Code done |
| 2 | main_global | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_003/gate_updates.gates.json` | common+residual 早期稳点：该 gate 生成 iter004 rollout proxy 0.5539，tool 0.9093 / memory 0.4219 / code 0.3305；可解释 | 正式 eval 前不能只看 proxy；CURE/Tool heldout 可能与 calibration proxy 不一致 | Tool+Memory+Code done |
| 3 | main_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_007/gate_updates.gates.json` | global-coefficient 新 proxy 高点：该 gate 生成 iter008 proxy 0.5498，memory/tool 高 | code proxy 0.2979，低于 iter005 gate；可能牺牲 Code heldout | Tool+Memory+Code done |
| 4 | main_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_005/gate_updates.gates.json` | global-coefficient code 更稳点：该 gate 生成 iter006 proxy 0.5426，code 0.3402，超过起点 | overall 低于 iter007 gate；需要正式 eval 检查是否泛化 | Tool+Memory+Code done |
| 5 | opd_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_opd_gc_c033_evaltarget_fast_20260518/iter_004/gate_updates.gates.json` | OPD-only 对照：该 gate 生成的 iter005 proxy 到 0.5430，可检验 offline distillation baseline 上限 | 缺少 GRPO/frontier on-policy 解释，不能作为主方法 claim | Tool+Memory+Code done |
| 6 | main_global | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_006/gate_updates.gates.json` | common+residual 后续接近高点：该 gate 生成 iter007 proxy 0.5486 | 低于 iter007/iter003 gate；code 0.3086，仍低于 iter003 gate code 0.3305 | queued |
| 7 | main_global | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518/iter_004/gate_updates.gates.json` | 该 gate 生成的 iter005 proxy 仍高：overall 0.5384；effective code/memory/tool 均高于 init | 低于 iter003/iter006/iter007 gate 的 next-rollout proxy；iter006 code 下滑提示后续 gate 可能过推 | queued |
| 8 | init1_gc | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_003/gate_updates.gates.json` | init1 upper-initialization 对照：该 gate 生成 iter004 proxy 0.7274，code 0.3779 / memory 0.8750 / tool 0.9292 | 不是主方法初始化；tool/memory frontier 稀疏，不能作为 learned-composition 主证据 | Tool+Memory+Code done |

## 2026-05-18 04:22 CST Update

当前仍不把 final checkpoint 自动作为最佳候选，继续使用 next-rollout validation：

- `main_gc/iter_006/gate_updates.gates.json` 已由 `iter_007` rollout 验证，overall `0.5388`，低于 `iter_005` gate 对应的 `iter_006` overall `0.5426`；暂不替换 priority 3。
- `main_gc/iter_007/gate_updates.gates.json` 已由 `iter_008` rollout 验证，overall `0.5498`，高于该 run 早先点，但 code proxy `0.2979` 低于 `iter_005` gate 的 `0.3402`；加入候选但需要和 `main_gc_i5` 成对评测。
- `main_global/iter_006/gate_updates.gates.json` 已由 `iter_007` rollout 验证，overall `0.5486`，低于 priority 1 的 iter003 gate 对应 `0.5539`；暂不替换。
- `main_global/iter_007/gate_updates.gates.json` 已由 `iter_008` rollout 验证，overall `0.5674`，成为 common+residual 新 proxy 高点；但 code proxy `0.3057` 低于 iter003 gate 的 `0.3305`，必须成对评测。
- `init1_gc/iter_005/gate_updates.gates.json` 已落盘但 `iter_006` rollout 未完成；init1 仍只作为 upper-initialization ablation。
- `opd_gc/iter_006/gate_updates.gates.json` 已由 `iter_007` rollout 验证，overall `0.4998`，低于 `iter_004` gate 对应的 `iter_005` overall `0.5430`；暂不替换 priority 2。

## 2026-05-18 04:45 CST Dispatch

为优先拿主表 heldout 结果，停止低优先级后续训练：

- stopped: `train_eg72_main_gc_init1_fast_20260518`
- stopped: `train_eg72_opd_gc_c033_fast_20260518`

保留其候选 checkpoint：

- `init1_gc_i3`: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_004/baked_policy`
- `opd_gc_i4`: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_opd_gc_c033_evaltarget_fast_20260518/iter_005/baked_policy`

已启动 priority 1 `main_global_i7` 正式 eval6：

| split | tmux | GPUs | summary_dir |
|---|---|---|---|
| Tool/BFCL | `eval_p1_main_global_i7_tool_20260518` | 4 | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i7/tool` |
| Memory+Code | `eval_p1_main_global_i7_mc_20260518` | memory 5, code 6/7 | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i7/memory_code` |

## 2026-05-18 04:56 CST Tool Results

已完成 Tool/BFCL：

| candidate | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Tool mean | live mean |
|---|---:|---:|---:|---:|---:|---:|
| main_global_i7 | 0.9150 | 0.8750 | 0.6875 | 0.6667 | 0.7860 | 0.6771 |
| main_global_i3 | 0.9150 | 0.8600 | 0.6875 | 0.7083 | 0.7927 | 0.6979 |
| main_gc_i7 | 0.9100 | 0.8600 | 0.6875 | 0.7083 | 0.7915 | 0.6979 |
| main_gc_i5 | 0.9100 | 0.8650 | 0.6875 | 0.7083 | 0.7927 | 0.6979 |
| opd_gc_i4 | 0.9150 | 0.8650 | 0.6875 | 0.7083 | 0.7940 | 0.6979 |
| init1_gc_i3 | 0.8850 | 0.8650 | 0.7500 | 0.6250 | 0.7813 | 0.6875 |

正式表：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260518_p1_evaltarget_eval6.md
```

## 2026-05-18 04:58 CST main_global_i7 Memory

`main_global_i7` Memory formal eval6 completed:

| eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 |
|---:|---:|---:|---:|---:|---:|
| 0.6681 | 0.6898 | 0.6775 | 0.7233 | 0.5645 | 0.6896 |

Code/CURE has started on GPU group `[[6,7]]`.

## 2026-05-18 06:27 CST main_global_i7 Code

`main_global_i7` Code formal eval6 completed:

| LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3711 | 0.4865 | 0.4375 | 0.3028 | 0.4508 | 0.3581 | 0.3370 | 0.4687 | 0.3978 |

Interpretation: this checkpoint improves/keeps Memory but does not clearly
improve CURE Code over TA-1/3; continue paired evaluation for `main_global_i3`
and global-coefficient candidates before choosing paper main.

## 2026-05-18 05:17 CST main_global_i3 Memory

`main_global_i3` Memory formal eval6 completed:

| eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 |
|---:|---:|---:|---:|---:|---:|
| 0.6703 | 0.6173 | 0.6285 | 0.6909 | 0.5195 | 0.6518 |

Operational note: `main_global_i3` Code was intentionally interrupted because
it auto-started while `main_global_i7` LiveCodeBench was already using GPU
6/7. Its Memory result is valid; requeue Code-only after `main_global_i7` Code
finishes.

## 2026-05-18 06:37 CST main_global_i3 Code Requeue

`main_global_i7` Code has completed, so `main_global_i3` Code-only formal eval
was requeued in a separate directory to avoid mixing with the interrupted
Memory+Code log:

```text
tmux: eval_p1_main_global_i3_codeonly_20260518
gpu_groups: [[4,5]]
summary_dir: /tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_global_i3/code_only
run_id: expertgym_p1_main_global_i3_codeonly_20260518
```

## 2026-05-18 08:03 CST main_global_i3 Code Result

`main_global_i3` Code-only formal eval completed:

| LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3848 | 0.4831 | 0.4609 | 0.2979 | 0.4377 | 0.3346 | 0.3414 | 0.4604 | 0.3978 |

Interpretation: `main_global_i3` has a stronger LiveBench / Tool profile than
`main_global_i7`, but substantially weaker Memory. It is a useful
early-iteration trade-off point, not the main balanced candidate.

## 2026-05-18 06:40 CST main_gc_i7 Memory Dispatch

`main_gc_i7` has Tool done and is the global-coefficient high-proxy paired
candidate. Its Memory-only eval was started on a free card while Code evals run
on other cards:

```text
tmux: eval_p1_main_gc_i7_memory_20260518
gpu: 6
summary_dir: /tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i7/memory_only
run_id: expertgym_p1_main_gc_i7_memory_20260518
```

## 2026-05-18 06:53 CST main_gc_i7 Memory Result

`main_gc_i7` Memory-only formal eval completed:

| eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 |
|---:|---:|---:|---:|---:|---:|
| 0.6789 | 0.6848 | 0.6400 | 0.6611 | 0.5352 | 0.6662 |

## 2026-05-18 06:58 CST main_gc_i5 / opd_gc_i4 Memory Results

| candidate | eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 |
|---|---:|---:|---:|---:|---:|---:|
| main_gc_i5 | 0.6992 | 0.6567 | 0.6420 | 0.7315 | 0.5547 | 0.6824 |
| opd_gc_i4 | 0.7178 | 0.6214 | 0.6764 | 0.6957 | 0.5430 | 0.6778 |

## 2026-05-18 07:02 CST main_gc_i5 Code Dispatch

`main_gc_i5` is the strongest remaining global-coefficient candidate on
Memory, so Code-only eval was started after GPU0/1 became free:

```text
tmux: eval_p1_main_gc_i5_codeonly_20260518
gpu_groups: [[0,1]]
summary_dir: /tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i5/code_only
run_id: expertgym_p1_main_gc_i5_codeonly_20260518
```

## 2026-05-18 08:28 CST main_gc_i5 Code Result

`main_gc_i5` Code-only formal eval completed:

| LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3457 | 0.4711 | 0.4297 | 0.3043 | 0.4595 | 0.3620 | 0.3250 | 0.4653 | 0.3959 |

Interpretation: `main_gc_i5` has decent Memory/Tool but is weak on Code Acc.
It should be used as the global-coefficient ablation row, not as the main
balanced model.

## 2026-05-18 08:31 CST Remaining Code Dispatch

All earlier Code evals completed, so the remaining queued P1 candidates were
started as code-only jobs:

| candidate | tmux | gpu_groups | summary_dir | run_id |
|---|---|---|---|---|
| main_gc_i7 | `eval_p1_main_gc_i7_codeonly_20260518` | `[[0,1]]` | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i7/code_only` | `expertgym_p1_main_gc_i7_codeonly_20260518` |
| opd_gc_i4 | `eval_p1_opd_gc_i4_codeonly_20260518` | `[[2,3]]` | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/opd_gc_i4/code_only` | `expertgym_p1_opd_gc_i4_codeonly_20260518` |
| init1_gc_i3 | `eval_p1_init1_gc_i3_codeonly_20260518` | `[[4,5]]` | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/init1_gc_i3/code_only` | `expertgym_p1_init1_gc_i3_codeonly_20260518` |

Only CURE is enabled; Tool/Memory are disabled for these jobs.

## 2026-05-18 09:58 CST Remaining Code Results

The remaining code-only jobs completed:

| candidate | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_gc_i7 | 0.3789 | 0.4819 | 0.4375 | 0.3058 | 0.4599 | 0.3542 | 0.3423 | 0.4709 | 0.3959 |
| opd_gc_i4 | 0.3613 | 0.4873 | 0.4453 | 0.3033 | 0.4538 | 0.3581 | 0.3323 | 0.4705 | 0.4017 |
| init1_gc_i3 | 0.3672 | 0.4807 | 0.4297 | 0.3082 | 0.4368 | 0.3659 | 0.3377 | 0.4587 | 0.3978 |

Interpretation: `main_gc_i7` is the strongest global-coefficient P1 row on
mean Code Acc, while `init1_gc_i3` remains a high-Memory upper-initialization
ablation. None of these rows clearly resolves the Code generalization gap.

## 2026-05-18 06:43 CST Additional Memory Dispatch

Since the two Code jobs moved into CPU unit-test execution and released their
vLLM GPUs, two more Memory-only jobs were started to fill the P1 table:

| candidate | tmux | gpu | summary_dir | run_id |
|---|---|---:|---|---|
| main_gc_i5 | `eval_p1_main_gc_i5_memory_20260518` | 0 | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/main_gc_i5/memory_only` | `expertgym_p1_main_gc_i5_memory_20260518` |
| opd_gc_i4 | `eval_p1_opd_gc_i4_memory_20260518` | 1 | `/tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/opd_gc_i4/memory_only` | `expertgym_p1_opd_gc_i4_memory_20260518` |

## 2026-05-18 06:49 CST init1_gc_i3 Memory Dispatch

GPU7 was free after the Code jobs moved to CPU unit tests. The init=1
upper-initialization ablation Memory-only eval was started:

```text
tmux: eval_p1_init1_gc_i3_memory_20260518
gpu: 7
summary_dir: /tmp/shared-storage/ExpertGym/eval/p1_evaltarget_20260518/init1_gc_i3/memory_only
run_id: expertgym_p1_init1_gc_i3_memory_20260518
```

## 2026-05-18 07:08 CST init1_gc_i3 Memory Result

`init1_gc_i3` Memory-only formal eval completed:

| eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | mean EM | mean F1 |
|---:|---:|---:|---:|---:|---:|
| 0.7593 | 0.7519 | 0.7934 | 0.7611 | 0.6426 | 0.7664 |

Interpretation: init=1 remains a strong upper-initialization ablation for
Memory, but it is not the canonical learned-from-1/3 setting.

## Dispatch Rule

候选选择口径：`iter_k/gate_updates.gates.json` 的 proxy 表现由下一轮
`iter_{k+1}/rollouts.jsonl` 观测；不要把 `iter_k` rollout 误记为
`iter_k` gate 的结果。

固定候选 eval6 launcher：

```bash
DRY_RUN=1 bash skill/command/run_20260518_p1_candidate_eval6.sh main_global_i7
DRY_RUN=1 bash skill/command/run_20260518_p1_candidate_eval6.sh main_global_i3
DRY_RUN=1 bash skill/command/run_20260518_p1_candidate_eval6.sh main_gc_i7
DRY_RUN=1 bash skill/command/run_20260518_p1_candidate_eval6.sh main_gc_i5
DRY_RUN=1 bash skill/command/run_20260518_p1_candidate_eval6.sh opd_gc_i4
DRY_RUN=1 bash skill/command/run_20260518_p1_candidate_eval6.sh init1_gc_i3
```

正式 eval6 不与 BFCL Tool 任务并发：

1. Tool / BFCL 一次只跑一个模型。
2. Memory/Code 可以与非 Tool 任务并发，但结果必须回填到本文件和主表。
3. 若后续 iter004/005 的 code proxy 恢复并 overall 更高，替换或追加更高优先级候选。
