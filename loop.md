---
title: 2026-05-11 OP-VEC Gated-GRPO Loop
project: OnPolicyMerge_gated_grpo
date: 2026-05-11
tags:
  - opvec
  - gated-grpo
  - calibration
  - loop
---

# 2026-05-11 Loop

## 目标

选择有梯度信号、三任务均衡、可固定复现的 calibration data；用官方 reward 和规范参数训练 global gate；观察梯度是否收敛、系数是否落在合理区间；若合理则 bake checkpoint 并进入完整评测；若不合理则分析数据或优化设置；后续再考虑 588 系数细粒度版本。

## 代码改动

- `scripts/train/opvec_collect_vllm_rollouts.py`
  - 新增 bake checkpoint 后用 vLLM rollout 的采样路径。
  - vLLM 只负责生成和 reward，`old_logprob=None`，由 update 阶段按当前初始 gate 补齐。
- `scripts/train/opvec_update_gates_from_rollouts.py`
  - 新增 `--fill-missing-old-logprob`：对 vLLM rollout 的 kept samples 补 old logprob。
  - 新增 `--gradient-checkpointing`：长 memory trajectory 反传时降显存。
  - 新增 `--early-stop-grad-norm / --early-stop-gate-delta / --early-stop-patience`：按 epoch 记录收敛信号。
- `opvec/modeling/gated_linear.py`
  - delta buffer 按 base layer 的 device/dtype 固定，避免 forward 里重复 cast 大矩阵。
  - 将 `linear(x, coeff * delta)` 改为 `coeff * linear(x, delta)`，避免构造临时 delta weight。
- `scripts/data/select_balanced_frontier_calibration.py`
  - 从多个 rollout JSONL 中筛选 `keep_for_policy_loss=true`、reward 有方差、frontier weight/std 达标的样本。
  - 按任务 quota 做确定性选择，默认去重 prompt，输出固定 calibration JSONL 和 summary。
- `skill/command/run_fixed_frontier7_global.sh`
  - 固定 calibration global gate 训练 + bake 的可复现脚本。

验证：

```bash
$PY -m py_compile \
  scripts/train/opvec_update_gates_from_rollouts.py \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  scripts/data/select_balanced_frontier_calibration.py \
  opvec/modeling/gated_linear.py

bash -n skill/command/run_fixed_frontier7_global.sh
```

`pytest` 未运行：当前 BFCL 环境没有安装 `pytest`。

## 方案一耗时基准

方案一：每轮先 bake 当前 gate 为普通 HF checkpoint，用 vLLM 生成 rollout，再用 HF gate model 补 logprob 并 update。

试验目录：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/official_global_vllm_tp1_48x4_i1_seed20260511
```

设置：

```text
prompts: 48
samples_per_prompt: 4
tasks: code=16, memory=16, tool=16
max_new_tokens: 2048
max_logprob_tokens: 8192
vLLM TP: 1
```

结果：

```text
bake:    52.9s
collect: 434.3s
update:  179s
total:   666s ~= 11.1min
```

注意：

- vLLM rollout 成功，TP=4 曾因 custom allreduce/CUDA invalid argument 失败，所以当前稳妥配置是 TP=1。
- HF update 的 4 卡 8192-token 版本 OOM；修掉 delta 临时拷贝后，使用 6 卡 + gradient checkpointing 跑通。
- 该 48x4 只留下 9 条 frontier：`code=7, memory=1, tool=1`，不适合作为均衡 calibration。

## Calibration Data 固定

使用 0.75 初始 baked policy 补采样：

```text
mixed 48x4:  /tmp/shared-storage/OnPolicy/runs/gated_grpo/official_global_vllm_tp1_48x4_i1_seed20260511/iter_001/rollouts.jsonl
tool 64x4:   /tmp/shared-storage/OnPolicy/runs/gated_grpo/calibration_frontier_search_20260511/tool_64x4/rollouts.jsonl
tool 64x4:   /tmp/shared-storage/OnPolicy/runs/gated_grpo/calibration_frontier_search_20260511/tool_64x4_seed20260512/rollouts.jsonl
tool 64x4:   /tmp/shared-storage/OnPolicy/runs/gated_grpo/calibration_frontier_search_20260511/tool_64x4_seed20260513/rollouts.jsonl
memory 64x4: /tmp/shared-storage/OnPolicy/runs/gated_grpo/calibration_frontier_search_20260511/memory_64x4/rollouts.jsonl
code 64x4:   /tmp/shared-storage/OnPolicy/runs/gated_grpo/calibration_frontier_search_20260511/code_64x4/rollouts.jsonl
```

耗时：

```text
tool 64x4 seed20260510: 96.8s, kept_frontiers=8
memory 64x4:            1045.4s, kept_frontiers=8
code 64x4:              488.7s, kept_frontiers=24
tool 64x4 seed20260512: 82.8s, kept_frontiers=9
tool 64x4 seed20260513: 84.3s, kept_frontiers=10
```

严格去重后可用 frontier：

```text
code:   15
memory: 8
tool:   7
```

Tool 官方 routed pool 只有 39 条，64 prompt 会覆盖全池；多 seed 后仍只有 7 个 unique tool prompt 有 reward 方差信号。因此正式固定 calibration 先采用 `7/7/7`，不使用 duplicate prompt 硬凑 `8/8/8`。

固定 calibration：

```text
/tmp/shared-storage/OnPolicy/data/calibration/frontier_balanced_7each_seed20260511.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/frontier_balanced_7each_seed20260511.summary.json
```

summary：

```text
selected_rows: 21
selected_counts: code=7, memory=7, tool=7
candidate_counts: code=15, memory=8, tool=7
filters: keep_for_policy_loss=true, min_frontier_weight=0.20, min_reward_std=0.05, duplicate_prompt=false
```

Reward 方差：

```text
code   mean_reward_avg=0.576, std_reward_avg=0.458
memory mean_reward_avg=0.536, std_reward_avg=0.452
tool   mean_reward_avg=1.315, std_reward_avg=0.716
```

Tool reward 数值大于 1，说明 ToolRL verifier 的 reward scale 与 Code/Memory 不同。当前训练先用 equal task weight，后续需要考虑 task 内归一化或按 task reward scale 重标定。

## Global Gate 训练

训练命令已固化：

```bash
bash skill/command/run_fixed_frontier7_global.sh
```

从上一轮 gate 继续：

```bash
INIT_GATE_CHECKPOINT=/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_e3_20260511/gate_updates.gates.json \
RUN_NAME=fixed_frontier7_global_continue \
MAX_STEPS=6 \
bash skill/command/run_fixed_frontier7_global.sh
```

实际运行目录：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_e3_20260511
```

设置：

```text
calibration: frontier_balanced_7each_seed20260511.jsonl
gate_parameterization: global
trainable coefficients: common + tool_residual + memory_residual + code_residual
max_steps: 3
max_logprob_tokens: 8192
lr: 0.005
prior_loss_weight: 0.02
task_weight: tool=1.0, memory=1.0, code=1.0
max_coefficient_delta_from_init: 0.15
gradient_checkpointing: enabled
device_map: auto, 6 GPUs
```

结果：

```text
elapsed: 1100s ~= 18.3min
filled_missing_old_logprobs: 84
updates: 63
frontier_task_counts: code=7, memory=7, tool=7
stopped_early_at_step: null
```

Epoch summary：

```text
epoch 1: grad_norm_max=71.77, gate_delta_max=0.03584
epoch 2: grad_norm_max=23.25, gate_delta_max=0.01314
epoch 3: grad_norm_max=87.23, gate_delta_max=0.01664
```

没有收敛：第 3 个 epoch 的最大梯度又上升，说明固定数据上仍有强梯度信号，不能说“梯度消失”。

Final gates：

```json
{
  "common": 0.7779730558395386,
  "tool_residual": 0.03696972131729126,
  "memory_residual": -0.012355685234069824,
  "code_residual": -0.024614036083221436
}
```

折算 task coefficient：

```text
tool   ~= 0.81494
memory ~= 0.76562
code   ~= 0.75336
```

判断：

- 三个能力系数都在 0.75 附近，均大于 0.5，没有出现某个能力被学到极小。
- Tool 方向被上调，Memory/Code 小幅下调；这可能来自 Tool reward scale 较大，也可能说明初始 0.75 对 Tool 仍不足。
- 当前还没收敛，不适合直接声明最终实验结论；可以作为“固定 balanced calibration 能推动 gate 且系数合理”的第一版证据。

Baked checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_e3_20260511
```

Bake summary：

```text
num_delta_entries: 588
plan_only: false
```

## 方案一单轮耗时实测

方案一指每轮先把当前 gate bake 成普通 HF checkpoint，再用 vLLM 从 baked checkpoint rollout，最后用 HF/torch 对 gate 做 GRPO update。

本次 benchmark 目录：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/plan1_bake_vllm_one_iter_20260511_031029
```

实测分段：

```text
bake 588 个 delta 到 HF checkpoint: 51.4s
完整 48 prompt x 4 samples vLLM rollout: 411.55s
完整 48-row rollout 的 6GPU global-gate update: 171s
估算完整一轮: 51.4 + 411.55 + 171 = 633.95s ~= 10.6min
```

资源结论：

- vLLM rollout 用 `tensor_parallel_size=1` 时主要占 GPU0，baked checkpoint 加载约 4.3s，vLLM engine warmup/profile 约 18s。
- 2GPU update 在完整 48-row rollout 上 OOM：GPU1 占到约 79.1GiB 后还差 268MiB。
- 6GPU update 使用 `max_memory=55GiB` 能稳定跑通，最高显存约 53GiB。
- 本次从头跑的 48x4 collect 在 40/48 处被外部 SIGTERM 中断，只写出 40 条 rollout；但此前已有同配置完整 48x4 rollout summary，可用于拼接本次 update 计时。

## Global Gate 继续训练

继续训练目录：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_continue_e3b_20260511
```

Baked checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_continue_e3b_20260511
```

最终 global gates：

```json
{
  "common": 0.7813409566879272,
  "tool_residual": 0.0700899139046669,
  "memory_residual": -0.023759325966238976,
  "code_residual": -0.046330589801073074
}
```

折算 task coefficient：

```text
tool   ~= 0.85143
memory ~= 0.75758
code   ~= 0.73501
```

训练信号：

```text
filled_missing_old_logprobs: 84
updates: 63
frontier_task_counts: code=7, memory=7, tool=7
epoch 1: grad_norm_max=77.65
epoch 2: grad_norm_max=58.47
epoch 3: grad_norm_max=73.32
stopped_early_at_step: null
```

判断：

- 三个 task coefficient 仍然都大于 0.5，区间合理。
- Tool 持续上升，Code/Memory 被轻微压低。
- 梯度没有消失，固定 `7/7/7` calibration 仍在强推参数；这不是收敛态。

## 588 系数细粒度训练

脚本：

```bash
bash skill/command/run_fixed_frontier7_global_parameter.sh
```

该脚本从 global e3b gate 初始化 `global-parameter`，也就是：

```text
global task coefficient: 3
per-module residual: 196 modules x 3 tasks = 588
实际可学习 parameter_coefficients: 588
```

运行目录：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_parameter_e3_from_e3b_20260511
```

Baked checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_parameter_e3_from_e3b_20260511
```

训练设置：

```text
init_gate_checkpoint: fixed_frontier7_global_continue_e3b
calibration: frontier_balanced_7each_seed20260511.jsonl
gate_parameterization: global-parameter
max_steps: 3
lr: 0.003
prior_loss_weight: 0.05
max_coefficient_delta_from_init: 0.05
max_logprob_tokens: 8192
gradient_checkpointing: enabled
device_map: auto, 6 GPUs
```

训练结果：

```text
parameter_coefficients: 588
filled_missing_old_logprobs: 84
updates: 63
frontier_task_counts: code=7, memory=7, tool=7
stopped_early_at_step: null
gate_grad_nonzero: true
```

Epoch summary：

```text
epoch 1: grad_norm_max=58.79, gate_delta_max=0.06487
epoch 2: grad_norm_max=12.89, gate_delta_max=0.03867
epoch 3: grad_norm_max=64.00, gate_delta_max=0.03347
```

全局 task coefficient：

```text
tool   = 0.86473
memory = 0.74328
code   = 0.73138
```

每个 expert 的 196 个 module coefficient 统计：

```text
tool:
  mean=0.87239, min=0.81473, p05=0.82025, median=0.87619, p95=0.91473, max=0.91473
memory:
  mean=0.74063, min=0.69328, p05=0.69328, median=0.74082, p95=0.78317, max=0.79328
code:
  mean=0.72669, min=0.68138, p05=0.68138, median=0.72928, p95=0.76662, max=0.78138
```

判断：

- 588 个细粒度系数全部大于 0.5，没有能力被完全关掉。
- Tool 明显最高，Code 最低；这与 global gate 的趋势一致。
- 多个 coefficient 触到 `max_coefficient_delta_from_init=0.05` 边界，说明固定 calibration 仍在推，训练没有收敛。
- 因为数据只有 `7/7/7`，588 参数相对 calibration 明显过参数化；它能给出可分析分布，但不能单独证明泛化。

## Sanity Eval

这不是完整官方评测，只是同一 source pool、同一 seed 的 48 prompt x 4 samples 快速对比。

Rollout：

```text
initial075:
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/initial075_48x4_seed20260514/rollouts.jsonl
global_e3b:
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/fixed_frontier7_global_continue_e3b_48x4_seed20260514/rollouts.jsonl
global_parameter_588:
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/fixed_frontier7_global_parameter_e3_from_e3b_48x4_seed20260514/rollouts.jsonl
```

汇总文件：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/compare_initial_global_globalparameter588_seed20260514.json
```

指标：

| model | task | mean_reward | success_rate |
|---|---:|---:|---:|
| initial075 | all | 1.59440 | 0.83333 |
| initial075 | code | 0.79883 | 0.76562 |
| initial075 | memory | 0.81250 | 0.81250 |
| initial075 | tool | 3.17188 | 0.92188 |
| global_e3b | all | 1.59466 | 0.83333 |
| global_e3b | code | 0.75273 | 0.71875 |
| global_e3b | memory | 0.81250 | 0.81250 |
| global_e3b | tool | 3.21875 | 0.96875 |
| global_parameter_588 | all | 1.55924 | 0.80208 |
| global_parameter_588 | code | 0.70898 | 0.68750 |
| global_parameter_588 | memory | 0.76562 | 0.76562 |
| global_parameter_588 | tool | 3.20312 | 0.95312 |

相对 `initial075`：

```text
global_e3b:
  all    mean_reward +0.00026, success +0.00000
  code   mean_reward -0.04609, success -0.04688
  memory mean_reward +0.00000, success +0.00000
  tool   mean_reward +0.04688, success +0.04688

global_parameter_588:
  all    mean_reward -0.03516, success -0.03125
  code   mean_reward -0.08984, success -0.07812
  memory mean_reward -0.04688, success -0.04688
  tool   mean_reward +0.03125, success +0.03125
```

判断：

- Global e3b 在 sanity eval 上几乎持平：Tool 提升，Code 下降，Memory 持平。
- 588 版本虽然 coefficient 区间合理，但 sanity eval 退化：整体、Code、Memory 都低于 0.75 初始点，只保留 Tool 小幅收益。
- 当前证据更像“Tool reward scale / frontier data 牵引过强”，不是已经找到泛化最优 gate。
- 不能把 588 版本作为最终结论；它是一个重要负结果，说明细粒度参数需要更大、更稳的 calibration 或更强 task-normalized objective。

## Tool/BFCL 官方子集评测

使用 `skill/command/run_full_eval_suite.sh` 调外部 BFCL harness，评测 category：

```text
parallel, parallel_multiple, live_parallel, live_parallel_multiple
```

这不是 BFCL 全 leaderboard，而是与 ToolRL/parallel tool-call 最相关的 4 个子集，共 440 条。

命令形态：

```bash
RUN_TOOL=1 RUN_MEMORY=0 RUN_CODE=0 TOOL_GPU=0 TOOL_PORT=8001 \
bash skill/command/run_full_eval_suite.sh <checkpoint> <model_name>
```

结果目录：

```text
initial075:
  /tmp/shared-storage/OnPolicy/eval/full_suite/opvec-fixed075/tool_initial075_20260511
  /mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/opvec-fixed075-tool-tool-initial075-20260511-tool_initial075_20260511

global_e3b:
  /tmp/shared-storage/OnPolicy/eval/full_suite/fixed-frontier7-global-e3b/tool_global_e3b_20260511
  /mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/fixed-frontier7-global-e3b-tool-tool-global-e3b-20260511-tool_global_e3b_20260511

global_parameter_588:
  /tmp/shared-storage/OnPolicy/eval/full_suite/fixed-frontier7-globalparameter588/tool_globalparameter588_20260511
  /mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/score/fixed-frontier7-globalparameter588-tool-tool-globalparameter588-20260511-tool_globalparameter588_20260511
```

指标：

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple | weighted subset |
|---|---:|---:|---:|---:|---:|
| initial075 | 180/200 = 90.00% | 173/200 = 86.50% | 12/16 = 75.00% | 16/24 = 66.67% | 381/440 = 86.59% |
| global_e3b | 181/200 = 90.50% | 172/200 = 86.00% | 12/16 = 75.00% | 15/24 = 62.50% | 380/440 = 86.36% |
| global_parameter_588 | 181/200 = 90.50% | 171/200 = 85.50% | 12/16 = 75.00% | 16/24 = 66.67% | 380/440 = 86.36% |

判断：

- 官方 BFCL parallel 子集上，训练后的 Tool 没有提升；global e3b 和 588 都比 0.75 初始少 1/440。
- global/588 提高了 `parallel` 1 条，但损失了 `parallel_multiple`，global 还损失了 `live_parallel_multiple`。
- 这说明 fixed `7/7/7` calibration 里 Tool reward 的上推没有泛化到 BFCL heldout；小 sanity eval 的 Tool 提升不能作为结论。

## Memory/HotpotQA Eval-50

使用 `skill/command/run_full_eval_suite.sh` 调外部 HotpotQA memory harness，先跑 `eval_50` 小集。

命令形态：

```bash
RUN_TOOL=0 RUN_MEMORY=1 RUN_CODE=0 MEMORY_GPU_IDS=0 MEMORY_TP=1 \
MEMORY_DATASETS="eval_50" MEMORY_MAX_NEW_TOKENS=2048 \
MEMORY_MAX_INPUT_LENGTH=32768 MEMORY_VLLM_MAX_MODEL_LEN=32768 \
bash skill/command/run_full_eval_suite.sh <checkpoint> <model_name>
```

注意：`eval_50.json` 实际加载 128 samples，推理日志显示 memory 阶段进度为 256，说明这是轨迹式 memory 生成，不是只看 final answer。

结果目录：

```text
initial075:
  /tmp/shared-storage/OnPolicy/eval/full_suite/opvec-fixed075/memory50_initial075_20260511
  /tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/opvec-gated-grpo-full-eval-memory/opvec-fixed075/memory50_initial075_20260511/eval_50/evaluation_summary.json

global_e3b:
  /tmp/shared-storage/OnPolicy/eval/full_suite/fixed-frontier7-global-e3b/memory50_global_e3b_20260511
  /tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/opvec-gated-grpo-full-eval-memory/fixed-frontier7-global-e3b/memory50_global_e3b_20260511/eval_50/evaluation_summary.json

global_parameter_588:
  /tmp/shared-storage/OnPolicy/eval/full_suite/fixed-frontier7-globalparameter588/memory50_globalparameter588_20260511
  /tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/opvec-gated-grpo-full-eval-memory/fixed-frontier7-globalparameter588/memory50_globalparameter588_20260511/eval_50/evaluation_summary.json
```

指标：

| model | exact_match | exact_match_rate | sub_exact_match_rate | avg_f1 |
|---|---:|---:|---:|---:|
| initial075 | 81/128 | 0.63281 | 0.79688 | 0.76252 |
| global_e3b | 80/128 | 0.62500 | 0.77344 | 0.75113 |
| global_parameter_588 | 78/128 | 0.60938 | 0.79688 | 0.76852 |

判断：

- Memory eval-50 上 global e3b 低于 0.75 初始点：EM、Sub-EM、F1 全降。
- 588 的 EM 更低，Sub-EM 与初始持平，F1 略高；如果以官方 exact match 作为主指标，仍不是正向结果。
- 结合 sanity eval，Memory gate 被压低并没有带来泛化收益。

## Code/CURE Global E3B

使用 `skill/command/run_full_eval_suite.sh` 调外部 CURE harness。当前只跑了 global e3b；同配置跑 initial/588 单模型也约 80 分钟，不建议在当前 fixed `7/7/7` 已经 Tool/Memory 退化的情况下继续盲目消耗。

命令：

```bash
RUN_TOOL=0 RUN_MEMORY=0 RUN_CODE=1 \
CODE_GPU_GROUPS="[[0,1]]" CODE_MAX_TEST=8 CODE_MAX_GENERATION_TOKEN=10000 \
RUN_ID=code_global_e3b_20260511 \
bash skill/command/run_full_eval_suite.sh \
  /tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_continue_e3b_20260511 \
  fixed-frontier7-global-e3b
```

运行耗时：

```text
START_ALL: 2026-05-11 04:45:10
END_ALL:   2026-05-11 06:06:10
elapsed:   ~81min
```

结果目录：

```text
/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/fixed-frontier7-global-e3b/code_global_e3b_20260511/summary.json
```

协议：

```text
datasets: LiveBench, LiveCodeBench
k_code=4, k_case=4, scale_tuple_list=[(4, 4)]
max_test=8
max_generation_token=10000
max_model_len=32768
num_chunks=512
gpu_groups=[[0,1]]
```

指标：

| dataset | code_acc | code_accumulate_acc | unit_test_acc | unit_test_accumulate_acc | BoN acc | BoN accumulate_acc |
|---|---:|---:|---:|---:|---:|---:|
| LiveBench | 0.38867 | 0.49216 | 0.37597 | 0.39924 | 0.41406 | 0.55730 |
| LiveCodeBench | 0.30577 | 0.44789 | 0.43478 | 0.46796 | 0.34442 | 0.50825 |

历史参考：

```text
ta-c075 old CURE run:
  /mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/ta-scale-sweep-code-cure-full/ta-c075/ta-scale-sweep-20260502-232830/summary.json
```

该旧模型路径是 `/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_scale_sweep_c075/model`，不是当前 `opvec-fixed075-parameter-coeffs-full-bf16`，所以不能作为严格 baseline。仅作粗参考：

| model | dataset | code_acc | BoN acc |
|---|---|---:|---:|
| old ta-c075 | LiveBench | 0.40039 | 0.46875 |
| old ta-c075 | LiveCodeBench | 0.31654 | 0.37573 |
| global_e3b | LiveBench | 0.38867 | 0.41406 |
| global_e3b | LiveCodeBench | 0.30577 | 0.34442 |

判断：

- 与旧 `ta-c075` 粗参考相比，global e3b 的 Code/CURE 两个 dataset 都偏低。
- 虽然这不是严格同 checkpoint baseline，但它与 sanity eval 中 Code 下降的方向一致。
- 在 Tool/BFCL、Memory eval-50 已经不升的情况下，继续加训当前 fixed `7/7/7` calibration 的收益证据不足。

## 当前结论

1. 方案一可用：bake + vLLM rollout + HF update 可以跑通，48x4 单轮约 10.6-11 分钟。
2. 固定 balanced calibration 已经构造完成，严格去重后当前最大均衡规模是 `7/7/7`。
3. Global gate 与 588 细粒度 gate 都能学习，且所有能力系数都大于 0.5。
4. 训练没有收敛：global 和 588 的末轮 `grad_norm_max` 都重新升高，没有 early stop。
5. Sanity eval 不支持“已经变好”：global 只是整体持平，588 明显退化。
6. Tool/BFCL 官方子集也不支持“已经变好”：global/588 都比 0.75 初始少 1/440。
7. Memory eval-50 也不支持“已经变好”：global 的 EM/F1 都低于 0.75，588 的 EM 更低。
8. Code/CURE global e3b 已跑通；缺严格 current initial075/588 的 Code baseline，但与旧 `ta-c075` 粗参考相比也偏低。
9. 当前固定 `7/7/7` calibration 可以判定为“不足以泛化”：系数学到合理区间不等于 heldout 提升。

Full-eval wrapper：

```text
skill/command/run_full_eval_suite.sh
```

验证：

```bash
bash -n skill/command/run_full_eval_suite.sh
bash skill/command/run_full_eval_suite.sh --help
```

该 wrapper 不恢复旧 `skill/Evaluation_all` 目录，只调用外部已存在 harness：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/scripts/run_bfcl_tool_harness.py
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/scripts/run_hotpotqa_memory_harness.py
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/scripts/run_cure_full_harness.sh
```

## 下一步

- 不建议继续训练当前 fixed `7/7/7` calibration；优先改数据和 objective。
- 暂停继续加大 588 训练步数：现有 `7/7/7` calibration 太小，继续训练更可能放大 overfit。
- 方案一作为后续主路径：`bake -> vLLM rollout -> HF update`。已确认 dry-run 会正确串起三段命令，并把 `--task-normalize-advantages`、`--length-normalize-policy-logprob` 传入 update；修复 delta 安装显存峰值后，48x4 实测完整一轮约 `9.9-11min`。
- 2026-05-11 06:32 已正式复跑一轮：bake 52.8s，collect 396.7s，retry update 约 145s；详见“方案一正式一轮”。
- 已补两个默认关闭的下一轮 objective 开关：
  - `--task-normalize-advantages`：把 row 内标准化后的 advantage 再按 task 的 mean-abs 做 rescale，避免某个 task 的 frontier weight/reward variance 系统性放大。
  - `--length-normalize-policy-logprob`：PPO ratio/KL 用平均 token logprob，避免不同任务输出长度差异直接进入梯度尺度。
- 两个开关已经接入：
  - `scripts/train/opvec_update_gates_from_rollouts.py`
  - `scripts/train/opvec_gated_grpo_loop.py`
  - `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - `skill/command/run_fixed_frontier7_global.sh`
  - `skill/command/run_fixed_frontier7_global_parameter.sh`
  - `skill/command/run_official_gated_grpo_global.sh`
  - `skill/command/run_official_gated_grpo_global_vllm_one_iter.sh`
- 验证：
  - `python -m py_compile scripts/train/opvec_update_gates_from_rollouts.py scripts/train/opvec_gated_grpo_loop.py scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - `bash -n skill/command/run_fixed_frontier7_global.sh skill/command/run_fixed_frontier7_global_parameter.sh skill/command/run_official_gated_grpo_global.sh skill/command/run_official_gated_grpo_global_vllm_one_iter.sh skill/command/run_full_eval_suite.sh`
  - native loop dry-run 已确认 update command 会带上 `--task-normalize-advantages --length-normalize-policy-logprob`
- 下一轮 objective ablation 命令：
  ```bash
  TASK_NORMALIZE_ADVANTAGES=1 \
  LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
  RUN_NAME=fixed_frontier7_global_tasknorm_lenavg \
  bash skill/command/run_fixed_frontier7_global.sh
  ```
- 优先扩大和重标定 calibration：
  - 增加 Tool 以外的 Code/Memory frontier 数量。
  - 对 Tool reward 做 task 内归一化，避免 reward scale 主导。
  - Memory 需要更长 trajectory 与官方 reward 对齐，不能只看 final answer。
- 下一轮应先重做 calibration selection 和 task-normalized objective，再训练；不要把当前 global e3b 或 588 作为最终 merged model。

## 完成度审计

目标拆解：

1. 选择有梯度信号且尽量均衡的 calibration data，并固定 data。
2. 使用规范配置训练，直到梯度信号消失或模型收敛后停止。
3. 进入完整评测，计算 Tool/Memory/Code 指标。
4. 如果 task vector 系数在合理区间，分析实验结果；如果不合理，分析原因并迭代。
5. 如果整体权重合理，训练 588 个系数的细粒度版本并分析。
6. 把今晚过程记录到项目 `loop.md`。

证据映射：

| 要求 | 当前 artifact | 结论 |
|---|---|---|
| 均衡 calibration | `/tmp/shared-storage/OnPolicy/data/calibration/frontier_balanced_7each_seed20260511.jsonl` | 已固定，`tool=7,memory=7,code=7`，每条都有 reward variance |
| 扩展梯度信号 calibration | `/tmp/shared-storage/OnPolicy/data/calibration/tool_all24_strict_expert_contrast_20260511.jsonl` + `/tmp/shared-storage/OnPolicy/data/calibration/memory_code_independent_posgrad8each_20260511.jsonl` | 已固定，`tool=24,memory=8,code=8`，但不是三任务等量均衡；用于修 Tool 信号不足 |
| 规范训练配置 | `skill/command/run_fixed_frontier7_global.sh`、`run_fixed_frontier7_global_parameter.sh`、`run_official_gated_grpo_global_vllm_one_iter.sh` | 已有脚本；方案一 dry-run 通过，默认可走 bake + vLLM rollout + HF update |
| 训练到收敛 | global e3/e3b、588 summary | 未完成；`early_stop.stopped=false`，末轮梯度仍明显存在 |
| 完整评测 | `run_full_eval_suite.sh`、Tool BFCL 子集、Memory eval_50、Code CURE global e3b | 未完成；已有子集/单模型证据，但缺严格全量三任务全模型对照 |
| 系数合理性分析 | global e3b、588、`tool24/memcode8` variants | 已分析部分；`7/7/7` 系数合理但 heldout 不升，若干 `tool24/memcode8` 旧版本出现 Tool 低系数或 588 撞边界 |
| 588 细粒度版本 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_parameter_e3_from_e3b_20260511` | 已训练并分析；但不收敛且 sanity eval 退化 |
| 今晚 loop 记录 | `loop.md` | 已持续记录 |

当前审计结论：

- 目标不能标记完成。
- 已完成“数据固定、脚本规范化、方案一耗时、初版 global/588、部分评测和问题定位”。
- 未完成“收敛训练”和“完整评测”。
- 当前最佳下一步不是继续 `7/7/7`，而是等待/分析 `tool=24,memory=8,code=8` 的 nonnegative 588 训练，并决定是否需要回退到 global-coefficient 或重新做 task-normalized calibration。

## `tool24/memcode8` 旧结果快速审计

已完成的扩展数据训练结果显示，低系数问题主要集中在 Tool：

| run | param | frontier | updates | early stop | code mean/min/max/<0.5 | memory mean/min/max/<0.5 | tool mean/min/max/<0.5 |
|---|---|---:|---:|---|---|---|---|
| independent_posgrad8_toolstruct_memoryevidence_globalcoeff_continue | global-coefficient | 8/8/8 | 192 | false | 0.7199/0.7199/0.7199/0 | 0.7167/0.7167/0.7167/0 | 0.2874/0.2874/0.2874/1 |
| toolaction_strict_memoryevidence_posgrad8_globalcoeff | global-coefficient | 8/8/8 | 288 | false | 0.5666/0.5666/0.5666/0 | 0.4920/0.4920/0.4920/1 | 0.0030/0.0030/0.0030/1 |
| tool24_memcode8_globalparameter | global-parameter | 8/8/24 | 480 | false | 0.6231/0.3599/0.6599/24 | 0.5846/0.3720/0.6720/58 | 0.2549/0.0524/0.3524/197 |
| tool24_memcode8_globalparameter_wideresidual | global-parameter | 8/8/24 | 480 | false | 0.7856/0.0876/0.8876/24 | 0.6854/0.0930/0.8930/55 | 0.1587/-0.3687/0.4313/197 |
| tool24_memcode8_parameter588 | parameter | 8/8/24 | 480 | false | 0.4514/-0.5/0.5746/34 | 0.3349/-0.5/0.5853/70 | 0.3497/-0.5/0.6099/128 |
| tool24_memcode8_parameter588_continue | parameter | 8/8/24 | 320 | false | 0.7030/-0.5/0.9621/36 | 0.5216/-0.5/0.9814/65 | 0.2287/-0.5/1.0051/134 |

判断：

- 单纯增加 Tool 到 24 条并没有解决 Tool 系数过低；多个 parameterization 都把 Tool 压低。
- `parameter` 版本容易把部分 gate 推到负边界，说明不加非负约束时 588 结果不可直接解释。
- 当前正在跑的 `train_tool24_memcode8_parameter588_nonneg_bronly_accum_m16_lr5e2_20260511` 是必要补充：它检验“低系数是否只是负边界/参数化导致”，但即使 nonnegative 变合理，也仍需 heldout 评测证明。

### nonnegative 588 结果

运行目录：

```text
/tmp/shared-storage/OnPolicy/runs/opvec4/rewardfix_20260510/train_tool24_memcode8_parameter588_nonneg_bronly_accum_m16_lr5e2_20260511
```

设置：

```text
frontier_task_counts: code=8, memory=8, tool=24
gate_parameterization: parameter
parameter_coefficients: 588
coefficient_bounds: 0.0 1.2
updates: 640
early_stop: false
last_grad_norm: 25.38
```

最终系数分布：

| task | n | mean | median | min | max | <0.5 | =0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 196 | 0.6258 | 0.7590 | 0.0 | 0.7703 | 36 | 25 |
| memory | 196 | 0.5025 | 0.7377 | 0.0 | 0.7834 | 69 | 48 |
| tool | 196 | 0.3750 | 0.3879 | 0.0 | 0.8168 | 111 | 36 |

判断：

- 非负约束没有解决低 Tool 系数问题；Tool 仍明显低于 0.5，且大量 gate 被推到 0。
- 训练也未收敛：`early_stop=false`，最后一行 `grad_norm=25.38`。
- 因此当前 `tool=24,memory=8,code=8` + best-response/pairwise 的 588 训练不能作为正结果。
- 低 Tool 系数更可能来自 objective/data mismatch：Tool expert reference 的 reward 信号在“让模型更像 expert 输出”上成立，但对当前 task-vector gate 的似然梯度方向不一定一致；需要回到 global/prob-center 或引入 on-policy self-compare，而不是继续加大 588 步数。

## 方案一正式一轮：tasknorm + length-normalized policy logprob

运行目录：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/plan1_tasknorm_lenavg_one_iter_20260511_0632
```

启动配置：

```bash
RUN_NAME=plan1_tasknorm_lenavg_one_iter_20260511_0632 \
GPU_LIST=0,1,2,3,4,5 \
MAX_MEMORY_PER_GPU=55GiB \
CPU_MAX_MEMORY=200GiB \
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
bash skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
```

结果：

```text
bake:    52.8s
collect: 396.7s wall, rollout internal elapsed=373.5s
rollout: 48 rows, tasks code=16/memory=16/tool=16
frontier kept by rollout: 9
```

第一次 update 失败：

```text
error: CUDA OOM during install_gated_linears_from_manifest
reason: device_map=auto 时 delta 被直接 load 到 target GPU，GatedLinear 再 clone 一份 buffer，安装阶段产生临时双份 delta
```

修复：

```text
opvec/modeling/apply_gates.py
opvec/modeling/gated_linear.py
```

具体修改：

- `device_map=auto` 时 delta 默认先从 CPU 加载，不直接 `torch.load(..., map_location=target_gpu)`。
- `GatedLinear` 注册 delta buffer 时去掉不必要的 `.clone()`，改为 `.contiguous()`。

验证：

```bash
python -m py_compile opvec/modeling/apply_gates.py opvec/modeling/gated_linear.py scripts/train/opvec_update_gates_from_rollouts.py
```

update retry：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train/opvec_update_gates_from_rollouts.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --rollouts /tmp/shared-storage/OnPolicy/runs/gated_grpo/plan1_tasknorm_lenavg_one_iter_20260511_0632/iter_001/rollouts.jsonl \
  --output /tmp/shared-storage/OnPolicy/runs/gated_grpo/plan1_tasknorm_lenavg_one_iter_20260511_0632/iter_001/gate_updates_retry_cpu_delta_4gpu.jsonl \
  --max-steps 1 \
  --max-logprob-tokens 8192 \
  --fill-missing-old-logprob \
  --lr 0.005 \
  --prior-loss-weight 0.02 \
  --ppo-loss-weight 1.0 \
  --device cuda \
  --torch-dtype bfloat16 \
  --gate-parameterization global \
  --device-map auto \
  --max-memory 0=70GiB --max-memory 1=70GiB --max-memory 2=70GiB --max-memory 3=70GiB --max-memory cpu=200GiB \
  --gradient-checkpointing \
  --task-weight tool=1.0 --task-weight memory=1.0 --task-weight code=1.0 \
  --frontier-task-quota tool=16 --frontier-task-quota memory=16 --frontier-task-quota code=16 \
  --length-normalize-policy-logprob \
  --task-normalize-advantages \
  --max-coefficient-delta-from-init 0.15
```

retry 成功：

```text
output: /tmp/shared-storage/OnPolicy/runs/gated_grpo/plan1_tasknorm_lenavg_one_iter_20260511_0632/iter_001/gate_updates_retry_cpu_delta_4gpu.gates.json
filled_missing_old_logprobs: 36
frontier_task_counts: code=6, memory=1, tool=2
updates: 9
task_scales: code=0.9292, memory=1.1799, tool=0.9292
final_gates:
  common=0.772074
  tool_residual=0.018801 -> tool≈0.790875
  memory_residual=-0.004471 -> memory≈0.767603
  code_residual=-0.014330 -> code≈0.757744
```

耗时判断：

- 如果从修复后的代码直接跑，预计一轮为 `52.8 + 396.7 + ~145 = ~594.5s`，约 `9.9min`。
- 这次原始脚本失败不是方案一不可行，而是 delta 安装阶段的显存峰值 bug；修复后 4 张空闲卡即可完成 update。
- 但本轮 rollout 仍只得到 `code=6,memory=1,tool=2` frontier，不适合作为最终 calibration。方案一解决速度问题，不解决数据分布问题。

## parameter588 nonnegative m16 评测进展

评测入口由另一进程启动：

```text
PID 867222
python scripts/eval/opvec_run_full_eval.py \
  --config configs/opvec4.yaml \
  --model-path /tmp/shared-storage/OnPolicy/checkpoints/tool24_memcode8_parameter588_nonneg_m16_20260511 \
  --model-name parameter588-nonneg-m16-20260511 \
  --run-id 20260511_parameter588_nonneg_m16_full \
  --parallel
```

计划文件：

```text
/tmp/shared-storage/OnPolicy/evaluation/parameter588-nonneg-m16-20260511/20260511_parameter588_nonneg_m16_full/full_eval_plan.json
```

并行任务：

| task | artifact / command | status |
|---|---|---|
| Tool/BFCL | `/tmp/shared-storage/OnPolicy/evaluation/bfcl/parameter588-nonneg-m16-20260511/20260511_parameter588_nonneg_m16_full/score/` | 已出分 |
| Memory/HotpotQA | `/tmp/shared-storage/OnPolicy/evaluation/memory/opvec-memory-hotpotqa/parameter588-nonneg-m16-20260511/20260511_parameter588_nonneg_m16_full/` | 运行中，GPU 3/4 |
| Code/CURE | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge/skill/plan/v1—feedback/evaluation/on_policy_merge/opvec-code-cure-full/parameter588-nonneg-m16-20260511/20260511_parameter588_nonneg_m16_full/` | 运行中，GPU 5/6 |

Quick subset trend：

```text
/tmp/shared-storage/OnPolicy/evaluation/subset_trend/20260511_parameter588_nonneg_m16_quick/subset_trend_summary.md
```

| model | Tool success | Tool reward | Memory success | Memory reward | Code selection success | Code relative BoN |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.7500 | 0.9121 | 0.2500 | 0.2500 | 0.8000 | +0.1000 |
| parameter588_nonneg_m16 | 0.8500 | 0.9333 | 0.4500 | 0.4500 | 0.8500 | +0.1500 |

这说明 nonnegative 588 在固定 quick subset 上有正趋势，但它仍不是最终结论，因为 quick subset 太小且系数分布本身显示 Tool 大量低于 0.5。

Tool/BFCL 已出分：

| category | score |
|---|---:|
| Non-Live Parallel AST | 90.00% |
| Non-Live Parallel Multiple AST | 87.50% |
| Live Parallel AST | 68.75% |
| Live Parallel Multiple AST | 66.67% |

按此前四类 BFCL subset 的样本量估算：

```text
correct = 180 + 175 + 11 + 16 = 382
total = 440
weighted subset acc = 86.82%
```

和之前 0.75 初始点 `381/440=86.59%` 相比，只多 `+1/440`；但 live_parallel 从 `75.00%` 降到 `68.75%`。因此 Tool/BFCL 目前只能算“基本持平/弱正”，不能证明泛化显著提升。

Memory/HotpotQA 已出分：

```text
/tmp/shared-storage/OnPolicy/evaluation/memory/opvec-memory-hotpotqa/parameter588-nonneg-m16-20260511/20260511_parameter588_nonneg_m16_full/*/evaluation_summary.json
```

| dataset | total | EM | EM rate | Sub-EM | Sub-EM rate | F1 | boxed |
|---|---:|---:|---:|---:|---:|---:|---:|
| eval_50 | 128 | 78 | 0.60938 | 97 | 0.75781 | 0.73355 | 1.00 |
| eval_100 | 128 | 78 | 0.60938 | 101 | 0.78906 | 0.75464 | 1.00 |
| eval_qa_1_32768 | 128 | 86 | 0.67188 | 106 | 0.82812 | 0.77872 | 1.00 |
| eval_qa_1_65536 | 128 | 82 | 0.64062 | 104 | 0.81250 | 0.75485 | 1.00 |

对照：

- 之前 0.75 初始点 eval_50：EM `81/128=0.63281`，F1 `0.76252`。
- parameter588 nonnegative m16 eval_50：EM `78/128=0.60938`，F1 `0.73355`。

判断：Memory 完整 harness 不支持该 588 版本变好；quick subset 的 Memory 正趋势没有泛化到 HotpotQA eval_50。

Code/CURE 完整结果：

LiveBench 已完成，结果文件：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge/skill/plan/v1—feedback/evaluation/on_policy_merge/opvec-code-cure-full/parameter588-nonneg-m16-20260511/20260511_parameter588_nonneg_m16_full/results/LiveBench.txt
```

| metric | value |
|---|---:|
| code acc | 0.33789 |
| code accumulate acc | 0.45495 |
| unit test acc | 0.43548 |
| unit test accumulate acc | 0.47977 |
| BoN acc | 0.40625 |
| BoN accumulate acc | 0.55044 |

对照：

- 之前 global e3b LiveBench：code_acc `0.38867`，BoN acc `0.41406`。
- parameter588 nonnegative m16 LiveBench：code_acc `0.33789`，BoN acc `0.40625`。

LiveCodeBench 也已完成，完整 summary：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge/skill/plan/v1—feedback/evaluation/on_policy_merge/opvec-code-cure-full/parameter588-nonneg-m16-20260511/20260511_parameter588_nonneg_m16_full/summary.json
```

| dataset | code_acc | code_accumulate_acc | unit_test_acc | unit_test_accumulate_acc | BoN acc | BoN accumulate_acc |
|---|---:|---:|---:|---:|---:|---:|
| LiveBench | 0.33789 | 0.45495 | 0.43548 | 0.47977 | 0.40625 | 0.55044 |
| LiveCodeBench | 0.30490 | 0.45297 | 0.47106 | 0.49878 | 0.35686 | 0.53098 |

对照 global e3b：

| dataset | global e3b code_acc | nonneg 588 code_acc | global e3b BoN | nonneg 588 BoN |
|---|---:|---:|---:|---:|
| LiveBench | 0.38867 | 0.33789 | 0.41406 | 0.40625 |
| LiveCodeBench | 0.30577 | 0.30490 | 0.34442 | 0.35686 |

判断：

- LiveBench 明显退化，尤其 code_acc 从 `0.38867` 降到 `0.33789`。
- LiveCodeBench 基本持平，BoN 小升但 code_acc 几乎不变。
- 结合 Tool/BFCL 只是 `+1/440` 弱正、Memory eval_50 明显低于 initial，该 nonnegative 588 不能作为最终 merged model。

## fixed `7/7/7` + tasknorm/length-normalized global baseline

目的：在同一份固定 balanced calibration 上，检验新增 objective 开关是否能改善旧 `7/7/7` global 训练，而不是继续扩大 588。

运行命令：

```bash
GPU_LIST=0,1,2,3 \
MAX_MEMORY_PER_GPU=70GiB \
CPU_MAX_MEMORY=200GiB \
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
RUN_NAME=fixed_frontier7_global_tasknorm_lenavg_e3_20260511 \
BAKE_OUTPUT=/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_tasknorm_lenavg_e3_20260511 \
bash skill/command/run_fixed_frontier7_global.sh
```

输出：

```text
run:        /tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_tasknorm_lenavg_e3_20260511
checkpoint: /tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_tasknorm_lenavg_e3_20260511
```

训练设置：

```text
frontier_task_counts: code=7, memory=7, tool=7
task_normalize_advantages: true
task_scales: code=0.9796, memory=1.0442, tool=0.9789
length_normalize_policy_logprob: true
updates: 63
steps: 3
```

最终 gates：

```text
common=0.815061
tool_residual=0.084939   -> tool≈0.900000
memory_residual=-0.020187 -> memory≈0.794874
code_residual=-0.064752   -> code≈0.750309
```

收敛状态：

| step | grad_norm_max | gate_delta_max |
|---:|---:|---:|
| 1 | 0.34607 | 0.04056 |
| 2 | 0.27415 | 0.03011 |
| 3 | 0.22490 | 0.01571 |

判断：

- 系数区间合理，且 Tool 被推到 `0.90` 上界附近；没有出现低于 0.5 的能力。
- 训练未收敛：`early_stop` 未触发，`grad_norm_max=0.2249` 仍远高于 `0.02`，`gate_delta_max=0.0157` 也高于 `0.001`。
- 该版本可以作为下一轮 heldout/sanity eval 候选，但不能声明完成；需要继续训练或做 eval 验证是否只是 calibration overfit。

Sanity eval：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/fixed_frontier7_global_tasknorm_lenavg_e3_48x4_seed20260514/rollouts.jsonl
```

设置：`48 prompts x 4 samples`，同 seed `20260514`，vLLM rollout 用时 `428.1s`，`kept_frontiers=15`。

| model | split | n | mean_reward | success_rate |
|---|---|---:|---:|---:|
| initial075 | all | 192 | 1.59440 | 0.83333 |
| initial075 | code | 64 | 0.79883 | 0.76562 |
| initial075 | memory | 64 | 0.81250 | 0.81250 |
| initial075 | tool | 64 | 3.17188 | 0.92188 |
| global_e3b | all | 192 | 1.59466 | 0.83333 |
| global_e3b | code | 64 | 0.75273 | 0.71875 |
| global_e3b | memory | 64 | 0.81250 | 0.81250 |
| global_e3b | tool | 64 | 3.21875 | 0.96875 |
| globalparam588 | all | 192 | 1.55924 | 0.80208 |
| globalparam588 | code | 64 | 0.70898 | 0.68750 |
| globalparam588 | memory | 64 | 0.76562 | 0.76562 |
| globalparam588 | tool | 64 | 3.20312 | 0.95312 |
| tasknorm_lenavg_global | all | 192 | 1.55990 | 0.78646 |
| tasknorm_lenavg_global | code | 64 | 0.67969 | 0.60938 |
| tasknorm_lenavg_global | memory | 64 | 0.81250 | 0.81250 |
| tasknorm_lenavg_global | tool | 64 | 3.18750 | 0.93750 |

判断：

- `tasknorm_lenavg_global` 虽然系数都合理，但 sanity heldout 明显不如 initial/global_e3b。
- 主要问题是 Code：success rate 从 initial 的 `0.76562` 降到 `0.60938`。
- 因此该版本不值得进入完整评测，也不能继续作为最终 merged model 候选。

## 08:05 收尾审计

逐项结论：

| 目标要求 | 当前证据 | 状态 |
|---|---|---|
| 选择有梯度信号的均衡 calibration data | `frontier_balanced_7each_seed20260511.jsonl`，`tool=7,memory=7,code=7`，每条有 reward variance | 已完成 |
| 固定 data | 训练脚本固定读取上述 calibration；扩展版 `tool24/memcode8` 也固定在两个 jsonl | 已完成 |
| 使用规范配置训练 | `run_fixed_frontier7_global.sh`、方案一 vLLM loop、tasknorm/lenavg 开关、CPU delta-load fix | 已完成 |
| 训练到梯度消失/收敛 | global e3b、588、nonnegative 588、tasknorm global 都 `early_stop=false` 或梯度仍明显存在 | 未完成 |
| 进入完整评测并计算指标 | nonnegative 588 完成 Tool/BFCL、Memory/HotpotQA、Code/CURE；fixed `7/7/7` global/588 有 sanity 与部分完整评测 | 部分完成 |
| task vector 系数合理则分析 | global/tasknorm global 系数合理但 sanity/heldout 不支持；nonnegative 588 系数不合理且评测不支持 | 已分析，结果为负 |
| 系数不合理则分析原因并迭代 | 已定位 Tool 低系数不是负边界问题；更可能是 expert-reference objective 与 task-vector gate 梯度方向不匹配，以及 calibration 太小/分布不稳 | 已分析，仍需新迭代 |
| 如果整体权重合理，训 588 并分析 | 已训 fixed `7/7/7` global-parameter 588 和 `tool24/memcode8` nonnegative parameter 588 | 已完成，结果为负 |
| 记录 loop.md | 本文件持续记录 | 已完成 |

最终实验判断：

- `7/7/7` fixed calibration 太小，容易把系数推到看似合理的区间，但 heldout 不提升。
- `tasknorm + length-normalized policy logprob` 能让 global 系数保持合理，但 sanity eval 明显伤 Code。
- `tool24/memcode8` nonnegative 588 没有解决低 Tool gate；Tool mean 仍只有 `0.375`，111/196 个 Tool gate 低于 0.5。
- nonnegative 588 quick subset 有正趋势，但完整评测不支持：Tool/BFCL 只有 `+1/440`，Memory eval_50 低于 initial，Code/CURE 的 LiveBench 明显低于 global e3b，LiveCodeBench 也只是基本持平。

因此当前目标不能标记完成：没有得到“收敛且完整评测正向”的 merged model。

下一轮不应继续堆 step 或扩大 588；优先改 objective/data：

1. 从 expert-reference imitation 改为更接近 on-policy self-compare：用当前 policy/上一阶段 policy 作 baseline，优化相对 improvement，而不是只把 expert answer 当正样本。
2. Calibration 不能只靠 small frontier；需要固定 heldout 分层，并对 Code/Memory/Tool 分别保证足够 task diversity。
3. Tool 需要检查 task-vector 对 tool-call 格式的局部影响：低 Tool gate 可能是 tool expert delta 在当前 base+memory/code delta 上产生负迁移，而不是数据量不足。
4. Memory 应优先使用官方轨迹 reward 的长上下文版本；quick subset 正趋势不能替代 HotpotQA full harness。

## 09:10 self-compare 与 freeze-Tool 迭代

新增代码：

```text
scripts/data/build_self_compare_advantage_calibration.py
scripts/train/opvec_update_gates_from_rollouts.py
scripts/train/opvec_gated_grpo_bake_vllm_loop.py
skill/command/run_fixed_frontier7_global.sh
skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
```

变更点：

- `build_self_compare_advantage_calibration.py`：把 candidate rollout 与 baseline rollout 按 `prompt_id` 对齐，在每个 sample 上写入 `reward_delta_vs_baseline`。
- `opvec_update_gates_from_rollouts.py` 支持 `--advantage-field reward_delta_vs_baseline`，直接用相对 baseline 的 advantage；不再做组内中心化，否则 `reward - baseline` 会被抵消。
- `--train-coefficient` 扩展到 `global` gate，可用 `global.memory,global.code` 冻住 Tool effective coefficient。
- 两个 command 脚本增加 `ADVANTAGE_FIELD`、`ADVANTAGE_FIELD_FRONTIER_WEIGHT`、`TRAIN_COEFFICIENTS` 环境变量透传。

验证：

```bash
python -m py_compile \
  scripts/data/build_self_compare_advantage_calibration.py \
  scripts/train/opvec_update_gates_from_rollouts.py \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py

bash -n \
  skill/command/run_fixed_frontier7_global.sh \
  skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
```

### self-compare diagnostic

用已有 48x4 heldout rollout 做 candidate-vs-initial 诊断：

```text
baseline:  /tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/initial075_48x4_seed20260514/rollouts.jsonl
candidate: fixed_frontier7_global_tasknorm_lenavg_e3 / global_e3b / globalparam588
```

严格要求 cross-baseline 时，`tasknorm_lenavg_global` 只能得到：

```text
tool=1, memory=2, code=6
```

放宽 cross-baseline 后也只有：

```text
tool=3, memory=2, code=10
```

结论：已有 48x4 heldout 不足以构造均衡 self-compare calibration，尤其 Tool/Memory 信号太少。

随后补跑 paired Tool 64x4：

```text
candidate rollout:
/tmp/shared-storage/OnPolicy/runs/gated_grpo/self_compare_pair_e3b_vs_initial_64x4/tool_64x4/rollouts.jsonl

baseline rollout:
/tmp/shared-storage/OnPolicy/runs/gated_grpo/calibration_frontier_search_20260511/tool_64x4/rollouts.jsonl
```

耗时：

```text
Tool 64x4 vLLM rollout: 725.6s
```

同一批 Tool prompt 上的结果：

| model | rows | unique prompts | samples | mean reward | success | kept frontier |
|---|---:|---:|---:|---:|---:|---:|
| initial_tool64 | 64 | 39 | 256 | 3.38509 | 0.92188 | 8 |
| e3b_tool64 | 64 | 39 | 256 | -2.39062 | 0.09375 | 4 |

self-compare 可用行：

```text
tool rows selected: 4/64
mean_delta_avg: -0.46875
positive_samples: 1
negative_samples: 8
```

典型失败：e3b 不再输出 `<tool_call>` JSON，而是普通自然语言回答，ToolRL reward 从 initial 的 `4.0` 掉到 `-3.0`。

判断：

- e3b 在小 sanity eval 上 Tool 看似提升，但在 Tool 64x4 search prompts 上严重退化。
- Tool task vector 方向不是简单“越大越好”；当前 additive OP-VEC 在 Tool prompt 上很容易破坏 tool-call 格式。
- 不能用 e3b 作为 self-compare 正样本来源。

### freeze-Tool global 训练

目的：验证 Tool 崩坏是否只是 Tool coefficient 被上调导致。

训练命令：

```bash
GPU_LIST=0,1 \
MAX_MEMORY_PER_GPU=70GiB \
CPU_MAX_MEMORY=220GiB \
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
TRAIN_COEFFICIENTS=global.memory,global.code \
RUN_NAME=fixed_frontier7_global_freeze_tool_tasknorm_lenavg_e3_20260511 \
BAKE_OUTPUT=/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_freeze_tool_tasknorm_lenavg_e3_20260511 \
bash skill/command/run_fixed_frontier7_global.sh
```

输出：

```text
run:        /tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_freeze_tool_tasknorm_lenavg_e3_20260511
checkpoint: /tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_freeze_tool_tasknorm_lenavg_e3_20260511
```

最终 gates：

```text
common=0.77300358
tool_residual=-0.02300362 -> tool=0.750000
memory_residual=0.03790160 -> memory=0.810905
code_residual=-0.01489798 -> code=0.758106
```

训练状态：

| step | grad_norm_max | gate_delta_max |
|---:|---:|---:|
| 1 | 0.39173 | 0.02926 |
| 2 | 0.38213 | 0.00471 |
| 3 | 0.38963 | 0.00394 |

未收敛：`stopped_early_at_step=null`，梯度仍大。

Sanity eval：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/fixed_frontier7_global_freeze_tool_tasknorm_lenavg_e3_48x4_seed20260514/rollouts.jsonl
```

耗时：

```text
48x4 vLLM rollout: 585.1s
```

同批 48x4 对比：

| model | split | n | mean_reward | success_rate |
|---|---|---:|---:|---:|
| initial075 | all | 192 | 1.59440 | 0.83333 |
| initial075 | code | 64 | 0.79883 | 0.76562 |
| initial075 | memory | 64 | 0.81250 | 0.81250 |
| initial075 | tool | 64 | 3.17188 | 0.92188 |
| global_e3b | all | 192 | 1.59466 | 0.83333 |
| global_e3b | code | 64 | 0.75273 | 0.71875 |
| global_e3b | memory | 64 | 0.81250 | 0.81250 |
| global_e3b | tool | 64 | 3.21875 | 0.96875 |
| tasknorm_lenavg_global | all | 192 | 1.55990 | 0.78646 |
| tasknorm_lenavg_global | code | 64 | 0.67969 | 0.60938 |
| tasknorm_lenavg_global | memory | 64 | 0.81250 | 0.81250 |
| tasknorm_lenavg_global | tool | 64 | 3.18750 | 0.93750 |
| freeze_tool_tasknorm | all | 192 | -0.25104 | 0.48438 |
| freeze_tool_tasknorm | code | 64 | 0.62187 | 0.57812 |
| freeze_tool_tasknorm | memory | 64 | 0.79688 | 0.79688 |
| freeze_tool_tasknorm | tool | 64 | -2.17188 | 0.07812 |

判断：

- 即使 Tool effective coefficient 被精确固定在 `0.75`，只把 Memory 提到 `0.8109`、Code 到 `0.7581`，Tool 也会从 `0.92188` success 崩到 `0.07812`。
- 这说明当前三个 expert delta 不是独立可加的能力方向；Memory/Code delta 的小幅变化会破坏 Tool 格式。
- 因此问题不是“Tool 系数是否低于 0.5”这么简单，而是 additive merge 的干扰项太强。
- 不建议继续沿 `fixed 7/7/7 + global additive OP-VEC + 加 step` 路线。

下一步建议：

1. 先做单轴 sweep：固定 Tool=0.75，分别扫 Memory/Code 在 `[0.70,0.82]` 的小网格，确认是哪一个 delta 破坏 Tool。
2. 如果 Memory/Code 任一轴对 Tool 有强负迁移，训练目标必须加入 Tool 格式 retention 或改成更细粒度/分层 gate，而不是 global 三系数。
3. 588 参数粒度不要继续盲跑；应该先定位破坏 Tool 的层/模块，再只开放低风险模块。

## 09:25 Tool 单轴 sanity sweep

目的：定位 freeze-Tool 失败是否由某一个非 Tool 轴导致。

构造两个手工 gate，并 bake 成普通 HF checkpoint：

```text
tool075_mem08109_code075:
  tool=0.750000
  memory=0.810905
  code=0.750000

tool075_mem075_code07581:
  tool=0.750000
  memory=0.750000
  code=0.758106
```

产物：

```text
gates:
/tmp/shared-storage/OnPolicy/runs/gated_grpo/axis_sweep_tool_sanity_20260511/gates/tool075_mem08109_code075.gates.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/axis_sweep_tool_sanity_20260511/gates/tool075_mem075_code07581.gates.json

checkpoints:
/tmp/shared-storage/OnPolicy/checkpoints/axis_sweep_tool075_mem08109_code075_20260511
/tmp/shared-storage/OnPolicy/checkpoints/axis_sweep_tool075_mem075_code07581_20260511
```

评测：复用 `initial075_48x4_seed20260514` 中的同一批 16 个 Tool prompt，每个 4 samples。

rollout：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/axis_sweep_tool_sanity_20260511/tool075_mem08109_code075/rollouts.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/axis_sweep_tool_sanity_20260511/tool075_mem075_code07581/rollouts.jsonl
```

耗时：

```text
memory-only Tool16x4: 205.5s
code-only Tool16x4:   197.7s
```

同批 Tool16 对比：

| model | samples | mean_reward | success_rate | parse failure |
|---|---:|---:|---:|---:|
| initial075_tool16 | 64 | 3.17188 | 0.92188 | 14/64 |
| memory_only_08109 | 64 | -2.20312 | 0.04688 | 64/64 |
| code_only_07581 | 64 | -2.15625 | 0.09375 | 64/64 |
| freeze_both_08109_07581 | 64 | -2.17188 | 0.07812 | 64/64 |
| global_e3b_tool16 | 64 | 3.21875 | 0.96875 | 14/64 |

判断：

- 只让 Memory 高于 Tool 会破坏 Tool。
- 只让 Code 略高于 Tool 也会破坏 Tool，哪怕幅度只有约 `+0.0081`。
- global_e3b 在同批 Tool16 上反而正常，因为它的 effective coefficients 是 Tool 最高，而不是 freeze Tool：

```text
global_e3b:
  tool≈0.85143
  memory≈0.75758
  code≈0.73501
```

因此更准确的结论是：

- Tool 格式能力对相对系数排序极其敏感。
- 不是某个 Memory/Code delta 单独“有毒”，而是当非 Tool 轴高于或接近压过 Tool 轴时，模型会从 tool-call 模式坍缩到普通对话模式。
- 初始 `0.75/0.75/0.75` 能工作；global_e3b 的 `tool > memory > code` 也能在 Tool16 上工作；但 `memory > tool` 或 `code > tool` 会崩。

后续训练约束应改成：

1. global 层面加入 `tool >= memory + margin`、`tool >= code + margin` 的投影或 penalty。
2. 或者把 Tool 作为 retention/constraint，而不是只靠 Tool reward frontier。
3. 588 参数化前先做分层/模块级 Tool retention，避免某些非 Tool delta 在关键 tool-format 层压过 Tool delta。

## 10:35 Tool margin 与 Tool retention 反例

基于上一节结论，新增训练控制：

```text
scripts/train/opvec_update_gates_from_rollouts.py
scripts/train/opvec_gated_grpo_bake_vllm_loop.py
skill/command/run_fixed_frontier7_global.sh
skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
```

新增参数：

```text
--tool-min-margin-over-memory
--tool-min-margin-over-code
```

含义：每次 optimizer step 后做 hard projection，要求 effective coefficients 满足：

```text
tool >= memory + margin
tool >= code + margin
```

验证：

```bash
python -m py_compile \
  scripts/train/opvec_update_gates_from_rollouts.py \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  scripts/data/build_self_compare_advantage_calibration.py

bash -n \
  skill/command/run_fixed_frontier7_global.sh \
  skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
```

CPU projection unit check：

```text
input approx:  tool=0.75, memory=0.81, code=0.75
margin=0.03
projected: tool=0.795, memory=0.765, code=0.75
```

### margin003 run

训练命令：

```bash
GPU_LIST=0,1,2,3 \
MAX_MEMORY_PER_GPU=70GiB \
CPU_MAX_MEMORY=220GiB \
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
TOOL_MIN_MARGIN_OVER_MEMORY=0.03 \
TOOL_MIN_MARGIN_OVER_CODE=0.03 \
RUN_NAME=fixed_frontier7_global_toolmargin003_tasknorm_lenavg_e3_20260511 \
BAKE_OUTPUT=/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_toolmargin003_tasknorm_lenavg_e3_20260511 \
bash skill/command/run_fixed_frontier7_global.sh
```

产物：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_toolmargin003_tasknorm_lenavg_e3_20260511
/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_toolmargin003_tasknorm_lenavg_e3_20260511
```

最终 effective coefficients：

```text
tool=0.900000
memory=0.782008
code=0.746547
```

训练仍未收敛：

| step | grad_norm_max | gate_delta_max |
|---:|---:|---:|
| 1 | 0.32572 | 0.05812 |
| 2 | 0.25378 | 0.02985 |
| 3 | 0.22031 | 0.00252 |

同批 48x4 sanity：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/fixed_frontier7_global_toolmargin003_tasknorm_lenavg_e3_48x4_seed20260514/rollouts.jsonl
```

结果：

| model | split | n | mean_reward | success_rate |
|---|---|---:|---:|---:|
| initial075 | all | 192 | 1.59440 | 0.83333 |
| initial075 | code | 64 | 0.79883 | 0.76562 |
| initial075 | memory | 64 | 0.81250 | 0.81250 |
| initial075 | tool | 64 | 3.17188 | 0.92188 |
| tasknorm_lenavg_global | all | 192 | 1.55990 | 0.78646 |
| tasknorm_lenavg_global | tool | 64 | 3.18750 | 0.93750 |
| toolmargin003_tasknorm | all | 192 | -0.21940 | 0.51562 |
| toolmargin003_tasknorm | code | 64 | 0.68555 | 0.64062 |
| toolmargin003_tasknorm | memory | 64 | 0.82812 | 0.82812 |
| toolmargin003_tasknorm | tool | 64 | -2.17188 | 0.07812 |

重要反例：

- `tasknorm_lenavg_global` 与 `toolmargin003` 的系数非常接近：

```text
tasknorm:   tool=0.900000, memory=0.794874, code=0.750309
toolmargin: tool=0.900000, memory=0.782008, code=0.746547
```

- 但 Tool 表现完全相反：

```text
tasknorm Tool success:   0.93750
toolmargin Tool success: 0.07812
```

典型失败：`toolmargin` 不输出 `<tool_call>` JSON，而生成带 `**Hadoop Job Status Query** <system> ...` 的普通对话式长文本。

判断：

- `tool >= memory/code + margin` 不是充分条件。
- global 三系数空间存在非常尖锐的非线性区域；极小系数变化可能导致 Tool 格式模式坍缩。
- 仅靠 global coefficient 排序无法稳定控制 Tool。

### Tool retention run

构造 Tool retention buffer：

```text
/tmp/shared-storage/OnPolicy/data/calibration/tool_initial075_all_success_retention_seed20260514.json
```

来源：`initial075_48x4_seed20260514` 中 13 条 Tool all-success rows，只用于 KL retention，不加入 frontier policy loss。

训练命令核心：

```bash
python scripts/train/opvec_update_gates_from_rollouts.py \
  --rollouts /tmp/shared-storage/OnPolicy/data/calibration/frontier_balanced_7each_seed20260511.jsonl \
  --retention-only-replay-buffer /tmp/shared-storage/OnPolicy/data/calibration/tool_initial075_all_success_retention_seed20260514.json \
  --use-retention \
  --retention-loss-weight 1.0 \
  --task-normalize-advantages \
  --length-normalize-policy-logprob \
  ...
```

产物：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_toolret1_tasknorm_lenavg_e3_20260511
/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_toolret1_tasknorm_lenavg_e3_20260511
```

最终 effective coefficients：

```text
tool=0.762172
memory=0.736652
code=0.732811
```

训练状态：

| step | grad_norm_max | gate_delta_max |
|---:|---:|---:|
| 1 | 7.49295 | 0.02192 |
| 2 | 6.31793 | 0.00457 |
| 3 | 2.59195 | 0.01013 |

retention 梯度很大，仍未收敛。

同批 48x4 sanity：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/fixed_frontier7_global_toolret1_tasknorm_lenavg_e3_48x4_seed20260514/rollouts.jsonl
```

结果：

| model | split | n | mean_reward | success_rate |
|---|---|---:|---:|---:|
| toolret1_tasknorm | all | 192 | -0.22331 | 0.51562 |
| toolret1_tasknorm | code | 64 | 0.68945 | 0.65625 |
| toolret1_tasknorm | memory | 64 | 0.79688 | 0.79688 |
| toolret1_tasknorm | tool | 64 | -2.15625 | 0.09375 |

判断：

- Tool KL retention weight `1.0` 没有保住 Tool 格式。
- 它把系数拉近 initial，但生成仍然崩成普通对话模式。
- 这进一步说明当前问题不只是 global coefficient 数值，而是 delta 叠加后的局部表示/格式吸引子改变。

当前路线结论：

- `fixed 7/7/7 + global 三系数` 已经试过 unconstrained、tasknorm、freeze Tool、tool margin、Tool retention，均未得到稳定正向模型。
- 不应继续在 global 三系数上加 step 或继续调 margin。
- 下一步若继续，应该做模块级定位：用 588/global-parameter 但不是全量训练，而是先做 Tool-format sensitive module ablation，找出哪些 layer/module 的 non-Tool delta 破坏 Tool 格式，再只开放低风险模块或给高风险模块强 Tool retention。

## Tool-format sensitive module ablation

目的：判断 Tool 崩坏是否来自少数层段/模块类型的 memory/code delta 扰动，而不是单纯由 global 三系数数值决定。

设置：

- 基准：`initial075`，所有 task vector effective coefficient 为 `0.75`。
- 扰动值取自 freeze-Tool 训练末端：`tool=0.75, memory=0.810905, code=0.758106`。
- 每个 ablation 只把某个层段/模块类型设置为上述 memory/code 值，其它位置保持三系数 `0.75`。
- 评测：同一批 Tool 前 8 个 prompt，`samples_per_prompt=4`，总计 32 samples。
- prompt 来源：`/tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/initial075_48x4_seed20260514/rollouts.jsonl`。

产物：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/module_ablation_tool_sanity_20260511
/tmp/shared-storage/OnPolicy/checkpoints/module_ablation_{group}_mem08109_code07581_20260511
```

模块分组：

| group | changed modules |
|---|---:|
| early_attn | 40 |
| early_mlp | 30 |
| mid_attn | 40 |
| mid_mlp | 30 |
| late_attn | 32 |
| late_mlp | 24 |

同批 Tool8x4 结果：

| model/group | rows | samples | mean_reward | success | parseable | zero_calls | avg_len | kept_frontiers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial075 | 8 | 32 | 2.75000 | 0.87500 | 0.68750 | 0.31250 | 84.8 | 2 |
| global_tasknorm_e3 | 8 | 32 | 2.78125 | 0.90625 | 0.71875 | 0.28125 | 98.3 | 2 |
| freeze_tool_e3 | 8 | 32 | -1.81250 | 0.06250 | 0.00000 | 1.00000 | 1438.8 | 1 |
| toolmargin003_e3 | 8 | 32 | -1.81250 | 0.06250 | 0.00000 | 1.00000 | 1438.8 | 1 |
| toolret1_e3 | 8 | 32 | -1.78125 | 0.09375 | 0.00000 | 1.00000 | 1270.1 | 1 |
| mem_only | 8 | 32 | -1.84375 | 0.03125 | 0.00000 | 1.00000 | 1419.2 | 1 |
| code_only | 8 | 32 | -1.81250 | 0.06250 | 0.00000 | 1.00000 | 1293.2 | 1 |
| early_attn | 8 | 32 | -1.84375 | 0.03125 | 0.00000 | 1.00000 | 1415.2 | 1 |
| early_mlp | 8 | 32 | -1.84375 | 0.03125 | 0.00000 | 1.00000 | 1292.8 | 1 |
| mid_attn | 8 | 32 | -1.81250 | 0.06250 | 0.00000 | 1.00000 | 1302.4 | 1 |
| mid_mlp | 8 | 32 | -1.78125 | 0.09375 | 0.00000 | 1.00000 | 1141.8 | 1 |
| late_attn | 8 | 32 | -1.81250 | 0.06250 | 0.00000 | 1.00000 | 1365.4 | 1 |
| late_mlp | 8 | 32 | -1.81250 | 0.06250 | 0.00000 | 1.00000 | 1222.3 | 1 |

判断：

- 任意一个大块局部 memory/code 扰动都会让 Tool 前 8 题几乎完全失去 parseable tool call，`zero_calls=1.0`。
- 这说明 Tool 格式不是只受 global 三系数排序控制，而是对 non-Tool delta 的局部注入非常敏感。
- `global_tasknorm_e3` 在同批 Tool8 上反而正常，说明“系数接近”不是充分解释；不同模块的联合扰动存在强非线性抵消或吸引子切换。
- 因此下一步不适合直接训 588 全量参数，也不适合继续在 global 三系数上硬推。更稳的路线是先缩小可训练空间：
  - 保持 Tool expert coefficient 不低于初始点；
  - memory/code 先只开放经过 Tool sanity 验证的更细粒度小组，而不是 6 个大块；
  - 每轮 update 后必须立即跑固定 Tool sanity；只要 parseable/zero_calls 崩掉，就拒绝该 checkpoint。

## Fixed frontier 7/7/7: cap025 e10 续训失败

目的：验证固定 balanced calibration data 上，放宽 coefficient cap 后，global 三系数是否能继续训练到收敛并得到更强模型。

训练命令：

```bash
RUN_NAME=fixed_frontier7_global_tasknorm_lenavg_cap025_e10_20260511 \
GPU_LIST=0,1,2,3 \
MAX_MEMORY_PER_GPU=70GiB \
CPU_MAX_MEMORY=200GiB \
MAX_STEPS=10 \
MAX_COEFF_DELTA=0.25 \
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
bash skill/command/run_fixed_frontier7_global.sh
```

产物：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_tasknorm_lenavg_cap025_e10_20260511
/tmp/shared-storage/OnPolicy/checkpoints/fixed_frontier7_global_tasknorm_lenavg_cap025_e10_20260511
```

训练耗时：约 58 分钟。原因是 `opvec_update_gates_from_rollouts.py` 当前只在全部 step 结束后一次性写 `gate_updates.jsonl/summary/gates`，中途没有逐 step flush；长跑监控不友好。

最终 raw gates：

```json
{
  "common": 0.8147098422050476,
  "tool_residual": 0.1852901130914688,
  "memory_residual": -0.1025613322854042,
  "code_residual": -0.0827287808060646
}
```

最终 effective coefficients：

```text
tool   = 1.000000
memory = 0.712149
code   = 0.731981
```

收敛状态：

| step | grad_norm_max | gate_delta_max | tool | memory | code |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.346065 | 0.040591 | 0.829713 | 0.790610 | 0.751451 |
| 2 | 0.274146 | 0.030113 | 0.877724 | 0.795991 | 0.751751 |
| 3 | 0.224902 | 0.027453 | 0.917672 | 0.794950 | 0.750328 |
| 4 | 0.197425 | 0.025467 | 0.951639 | 0.789033 | 0.747777 |
| 5 | 0.182582 | 0.024310 | 0.981419 | 0.778799 | 0.744641 |
| 6 | 0.168350 | 0.017647 | 1.000000 | 0.766497 | 0.741165 |
| 7 | 0.156943 | 0.008135 | 1.000000 | 0.752747 | 0.738070 |
| 8 | 0.158202 | 0.007645 | 1.000000 | 0.739893 | 0.735297 |
| 9 | 0.153271 | 0.009402 | 1.000000 | 0.724905 | 0.733526 |
| 10 | 0.154869 | 0.007989 | 1.000000 | 0.712149 | 0.731981 |

判断：未收敛。`grad_norm_max` 到 step10 仍为 `0.154869`，没有触发 early stop；Tool coefficient 从 step6 起撞到上界 `1.0`，memory/code 继续下降。

固定 48x4 sanity：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eval_rollout/fixed_frontier7_global_tasknorm_lenavg_cap025_e10_48x4_seed20260514/rollouts.jsonl
```

同批对比：

| model | split | n | mean_reward | success | parseable | zero_calls | avg_len |
|---|---|---:|---:|---:|---:|---:|---:|
| initial075 | all | 192 | 1.59440 | 0.83333 | 0.26042 | 0.73958 | 660.3 |
| initial075 | code | 64 | 0.79883 | 0.76562 | 0.00000 | 1.00000 | 465.2 |
| initial075 | memory | 64 | 0.81250 | 0.81250 | 0.00000 | 1.00000 | 1408.1 |
| initial075 | tool | 64 | 3.17188 | 0.92188 | 0.78125 | 0.21875 | 107.6 |
| global_tasknorm_e3 | all | 192 | 1.55990 | 0.78646 | 0.26562 | 0.73438 | 708.5 |
| global_tasknorm_e3 | code | 64 | 0.67969 | 0.60938 | 0.00000 | 1.00000 | 436.9 |
| global_tasknorm_e3 | memory | 64 | 0.81250 | 0.81250 | 0.00000 | 1.00000 | 1572.8 |
| global_tasknorm_e3 | tool | 64 | 3.18750 | 0.93750 | 0.79688 | 0.20312 | 115.9 |
| cap025_e10 | all | 192 | -0.23177 | 0.50521 | 0.00000 | 1.00000 | 1038.6 |
| cap025_e10 | code | 64 | 0.69531 | 0.65625 | 0.00000 | 1.00000 | 443.5 |
| cap025_e10 | memory | 64 | 0.76562 | 0.76562 | 0.00000 | 1.00000 | 1324.2 |
| cap025_e10 | tool | 64 | -2.15625 | 0.09375 | 0.00000 | 1.00000 | 1348.0 |

结论：

- `cap025_e10` 不能进入 full eval；它在小 sanity 上已经完全丢失 Tool 格式。
- 固定 7/7/7 calibration 继续优化会把 Tool 推到上界，但这不是可靠提升，而是进入普通对话/长输出吸引子。
- 仅凭训练集 GRPO 梯度消失作为停止条件不够，必须加入固定 heldout sanity guard，尤其是 Tool parseable / zero-call 指标。
- 当前最好的 global 观察点仍是 `global_tasknorm_e3`，但它 Code 退化明显，也不是最终模型。
- 下一轮应改训练策略：每个 epoch 保存 gate/checkpoint 并立即跑固定 Tool/Code/Memory sanity；一旦 Tool parseable 崩掉或 Code 下降超过阈值，就回退到上一个 checkpoint，而不是继续追求训练梯度下降。

### Tool collapse threshold

从 `cap025_e10` 的 epoch summaries 抽取 step3/4/5/6 gates，分别 bake checkpoint 并只跑同批 Tool 前 8 题：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_tasknorm_lenavg_cap025_e10_20260511/intermediate_gates/step_003.gates.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_tasknorm_lenavg_cap025_e10_20260511/intermediate_gates/step_004.gates.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_tasknorm_lenavg_cap025_e10_20260511/intermediate_gates/step_005.gates.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/fixed_frontier7_global_tasknorm_lenavg_cap025_e10_20260511/intermediate_gates/step_006.gates.json
```

Tool8 结果：

| model | tool | memory | code | mean_reward | success | parseable | zero_calls | avg_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial075 | 0.750000 | 0.750000 | 0.750000 | 2.75000 | 0.87500 | 0.68750 | 0.31250 | 84.8 |
| cap015_e3 | 0.900000 | 0.794874 | 0.750309 | 2.78125 | 0.90625 | 0.71875 | 0.28125 | 98.3 |
| cap025_step003 | 0.917672 | 0.794950 | 0.750328 | -1.84375 | 0.03125 | 0.00000 | 1.00000 | 1252.9 |
| cap025_step004 | 0.951639 | 0.789033 | 0.747777 | -1.81250 | 0.06250 | 0.00000 | 1.00000 | 1282.2 |
| cap025_step005 | 0.981419 | 0.778799 | 0.744641 | -1.84375 | 0.03125 | 0.00000 | 1.00000 | 1310.9 |
| cap025_step006 | 1.000000 | 0.766497 | 0.741165 | -1.84375 | 0.03125 | 0.00000 | 1.00000 | 1456.0 |

关键判断：

- Tool 格式临界点在 `tool effective coefficient ~= 0.90-0.918` 之间，非常窄。
- `tool` 越大不代表 Tool 能力越强；超过临界点后会切到长普通对话模式，官方 Tool reward 直接崩。
- 这解释了为什么 `cap015_e3` 可用而 `cap025_step003/e10` 崩：不是训练更充分，而是越过了 Tool 格式稳定区。
- 后续 global 训练必须把 `tool` 上界显式限制在约 `0.90` 附近，或者把 Tool sanity 作为 hard rejection，而不是放宽到 `1.0`。

## 初始点 0/0.5 与 calibration 目标重新判断

用户提出的新方向：先用基础 task vector 合并让模型具备一定专家能力，再构造约 100 条 high-information calibration data，用 on-policy distillation + GRPO 逐步发现 task vector 的高效组合。

这个方向比昨晚的 fixed `7/7/7` reward-only GRPO 更合理。现在应把 calibration data 定义为“能力组合传感器”，不是普通训练集。

### 子代理并行结论

Agent A：目标函数/梯度分析。

- `0.5` 或 `0` 起点不是天然更稳。它们离 Tool 崩坏上界更远，但是否能学起来取决于低起点附近是否仍有 frontier 方差信号。
- 0.75 下 Tool 越大反而崩，是 task-vector 线性叠加进入了非线性格式吸引子；`tool=0.900` 正常，`tool=0.917672` 已经 Tool8 全崩。
- 现有 reward-only group-relative GRPO 更像在追训练 frontier 的似然最优，不保证 heldout 最优。
- 直接拿 0.75 产生的 calibration 给 0/0.5 训练，会变成 off-policy imitation；PPO ratio 的 on-policy 含义变弱。
- 必须加入 hard sanity guard、on-policy self-compare、Tool 格式约束、分层 calibration。

Agent B：最小执行验证。

- 无代码改动即可使用 `0` 或 `0.5` 初始点。
- `0` 可用 `scripts/modes/build_zero_gate_checkpoint.py` 生成。
- `0.5` 可传普通 `{"gates": ...}` checkpoint。
- 训练/采样入口已经支持：
  - `opvec_collect_hf_rollouts.py --gate-checkpoint`
  - `opvec_update_gates_from_rollouts.py --init-gate-checkpoint`
  - `opvec_gated_grpo_bake_vllm_loop.py --init-gate-checkpoint`

Agent C：代码库清洁审计。

- 当前 worktree 是 gated_grpo 主干切换后的状态：旧 `InternVL/Qwen/global_utils/skill/Evaluation*` 等 `D` 文件是有意清理，不应恢复。
- 新 `configs/ opvec/ scripts/ skill/command/ docs/ tests/ recipes/` 是当前主干，应保留并后续按模块纳入版本控制。
- 明确垃圾只有 ignored `__pycache__/.pyc`，已清理。
- 实验产物继续只放 `/tmp/shared-storage/OnPolicy/...`，不放回代码库。

### Init-anchor probe

目的：验证从 `0` 或 `0.5` 起点是否在当前 fixed calibration 代表 prompt 上具备自举方差信号。

Prompt：从 `frontier_balanced_7each_seed20260511.jsonl` 取每个任务 1 个代表 prompt，每个 4 samples。

产物：

```text
/tmp/shared-storage/OnPolicy/data/calibration/init_anchor_probe_20260511/init_zero_global.gates.json
/tmp/shared-storage/OnPolicy/data/calibration/init_anchor_probe_20260511/init_half_global.gates.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/init_anchor_probe_20260511/init0_hf_calprompt_1each4_skiplogprob/rollouts.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/init_anchor_probe_20260511/init05_hf_calprompt_1each4_skiplogprob/rollouts.jsonl
```

结果：

| init | task | prompt_id | rewards | std | keep_for_policy_loss | skip_reason | notes |
|---:|---|---|---|---:|---|---|---|
| 0.0 | tool | tool__45a7663a38ccd267 | [-3,-3,-3,-3] | 0.0000 | false | all_failure_no_contract_signal | 全部无 tool call，长度打满 512 |
| 0.0 | memory | memory__f9bdda63b0414a55 | [1,1,1,1] | 0.0000 | false | all_success | 全成功但无 GRPO 方差 |
| 0.0 | code | code__ed3efbfa38a67793 | [1,1,1,0.05] | 0.4114 | true | none | 只有 Code 有 frontier |
| 0.5 | tool | tool__45a7663a38ccd267 | [-3,-3,-3,-3] | 0.0000 | false | all_failure_no_contract_signal | 全部无 tool call，长度打满 512 |
| 0.5 | memory | memory__f9bdda63b0414a55 | [0,0,0,0] | 0.0000 | false | all_failure_no_contract_signal | 长输出失败，长度约 1828-2371 |
| 0.5 | code | code__ed3efbfa38a67793 | [0,0,0,1] | 0.4330 | true | none | 只有 Code 有 frontier |

判断：

- `init=0` 和 `init=0.5` 都不能直接靠当前 fixed calibration 的 reward-only GRPO 自举。
- 对 Tool 来说，两者都是 all-failure，无 parseable tool call；没有 on-policy 方差信号。
- 对 Memory 来说，`0` 在这条 prompt 上 all-success，`0.5` 反而 all-failure；都没有 GRPO 方差信号。
- 对 Code 来说，两者都有方差，说明 Code 更容易从低起点自举。
- 因此下一轮不能简单把初始点改成 `0.5/0` 后长跑；必须用 distillation 或 self-compare 给 Tool/Memory 补方向。

### 新 calibration 定义

约 100 条 calibration data 应该是 high-information probes，而不是随机训练样本。

推荐结构：

```text
Tool:   30-35 条
Memory: 30-35 条
Code:   30-35 条
Guard/retention: 额外固定，不一定进入 GRPO loss
```

每条最好包含：

```text
prompt
current-policy samples + official reward
tool expert sample + reward
memory expert sample + reward
code expert sample + reward
anchor sample + reward
parse/format/length diagnostics
```

筛选优先级：

1. 当前混合失败、对应 expert 成功。
2. 当前混合有成功有失败，reward 有方差。
3. 某能力提升会伤害另一能力的 conflict case。
4. Tool/Memory 的格式和轨迹稳定性样本。

### 下一轮目标函数建议

不要再只用：

```text
GRPO(current samples, official reward)
```

应改成：

```text
loss =
  GRPO(current samples, official reward)
+ lambda_distill * logp(best expert / best anchor response)
+ lambda_retention * KL(initial good tool/memory/code behavior)
+ lambda_guard * hard penalty(format collapse / length explosion)
+ prior(gate near safe anchor)
```

其中 Tool 必须有 hard guard：

```text
parseable tool_call
zero_calls
tool_call JSON correctness
length upper bound
```

训练策略：

- 初始点优先试 `0.5` 和 `0.75`，但不直接长跑。
- 每个 step/epoch 保存 gate，bake 后跑固定 sanity。
- Tool parseable 崩、zero_calls 到 1.0、Code 或 Memory 明显退化，立即拒绝该 checkpoint。
- 只有 global 三系数在 sanity 上稳定后，再考虑受限模块级；不要直接盲训 588。

## 2026-05-11 high-information calibration 方案

当前目标不是扩大 calibration data，而是把固定小集合做成能力方向传感器。约 100 条 prompt 的作用不是拟合数据分布，而是回答：

```text
当前 task-vector 组合往 tool / memory / code 哪个方向动，会不会带来真实能力收益？
这个收益是否会破坏已有格式、轨迹和其他任务？
```

因此 calibration data 应固定 prompt，但 rollout 必须 on-policy：每一轮用当前 gate 重新生成。这样同一批 prompt 会随着模型分布变化持续产生不同的 reward/format/length 信号，不等价于持续挖新训练集。

### 推荐组成

```text
Tool:   30 条主 calibration
Memory: 35 条主 calibration
Code:   25 条主 calibration
Guard:  10 条左右，只做 retention / sanity，不进入主 GRPO
```

主 calibration 每条只需要是 prompt 级固定，sample 每轮现采。每个任务内部分为三类：

```text
frontier_grpo:
  当前 policy samples 内 reward 有方差。
  用官方 reward 做 GRPO。

expert_recovery_distill:
  当前 policy 全错或弱，但对应 expert 有正确 action / trajectory / answer。
  用 best-response / pairwise / distill 给无方差任务补方向。

self_compare:
  当前 checkpoint 与上一 checkpoint / anchor 在同一 prompt 上有 reward delta。
  advantage 用 reward(current) - reward(previous/anchor)。
```

Guard 不应该被当作普通正样本推大系数，主要用于拒绝坏 checkpoint：

```text
Tool: parseable tool_call, zero_call rate, JSON 参数正确性
Memory: 轨迹长度、update 是否保留证据、final answer reward
Code: heldout pass/reward 不能明显掉
```

### 为什么纯 GRPO 不够

已有 probe 说明：

- `init=0` 和 `init=0.5` 在 Tool 上都是 all-failure，无 tool call，无方差。
- Memory 在低起点上要么 all-success，要么 all-failure，也没有稳定 GRPO 方差。
- Code 更容易出现 reward 方差，所以纯 GRPO 会天然偏向 Code。
- Tool 的安全窗口很窄，`tool ~= 0.90` 还能正常，推到约 `0.918` 已观察到 tool call collapse。

所以训练目标应是：

```text
loss =
  on_policy_GRPO(current samples, official reward)
+ expert_action_or_trajectory_distill(expert recovery rows)
+ self_compare_advantage(current reward - previous/anchor reward)
+ retention_KL(anchor good behavior)
+ gate_prior_and_hard_bounds
```

其中 Tool / Memory 的 distill 不是为了复制专家答案本身，而是为了在 reward 方差为 0 时提供可微方向。Code 可以更多依赖 frontier GRPO。

### 当前代码支持与缺口

已有支持：

- vLLM bake rollout 可以固定 seed manifest，每轮 on-policy rollout。
- Memory rollout 已按 MemAgent 多轮 trajectory 计算 logprob，不再只看 final answer。
- `opvec_update_gates_from_rollouts.py` 支持：
  - `--task-normalize-advantages`
  - `--length-normalize-policy-logprob`
  - `--advantage-field reward_delta_vs_baseline`
  - `--best-response-loss-weight`
  - `--pairwise-loss-weight`
  - `--use-retention`

关键缺口：

- 现在 `opvec_gated_grpo_bake_vllm_loop.py` 只做一次统一 update，不区分 on-policy GRPO rows 与 expert-distill rows。
- 如果直接把 expert recovery JSONL 当 `--rollouts` 混入，并同时开 PPO，会把离线 expert sample 当 on-policy PPO 样本，目标不规范。
- 更规范的实现应把一次迭代拆成两个更新阶段：

```text
stage A: on-policy rollout rows
  ppo_loss_weight > 0
  best_response_loss_weight 可小
  使用官方 reward / self-compare advantage

stage B: fixed expert recovery rows
  ppo_loss_weight = 0
  best_response_loss_weight / pairwise_loss_weight > 0
  只给 Tool/Memory/Code 的 expert-positive samples 做 distill
```

这可以先用 shell 两次调用 `opvec_update_gates_from_rollouts.py` 实现，不需要马上大改训练器。长期再把它封装进 bake+vLLM loop。

### calibration v1 的数据来源建议

当前可直接复用的高信息池：

```text
/tmp/shared-storage/OnPolicy/data/calibration/fixed_balanced13_codecont_tool_memorysplit_diverse_expertrecovery_20260511.jsonl
  39 rows, 13/tool 13/memory 13/code, expert recovery 样本齐全。

/tmp/shared-storage/OnPolicy/data/calibration/toolstrict_memoryevidence_independent_posgrad8each_20260511.jsonl
  24 rows, 8/tool 8/memory 8/code, 独立正梯度筛选，Tool strict call reward。

/tmp/shared-storage/OnPolicy/data/calibration/tool24_memcode8_strict_train32_memagent_reward_seed20260511.jsonl
  32 rows, Tool 20 / Memory 6 / Code 6，MemAgent reward 重算后更贴近官方轨迹。

/tmp/shared-storage/OnPolicy/data/calibration/codecontests_source_reward_probe48c_excl_prior_seed20260510.jsonl
  Code-only 48 rows，可补 Code frontier / heldout。

/tmp/shared-storage/OnPolicy/data/calibration/memory_answer_evidence_strict_main_seed20260510.jsonl
  Memory-only 86 rows，可补 Memory 证据型 prompt。
```

但不建议简单拼接成 100 条就跑。下一版应先生成一个 manifest summary，明确每条的角色：

```text
role: frontier_grpo | expert_recovery_distill | self_compare | guard_retention
task: tool | memory | code
source_file
prompt_id
reward_rule
why_selected
expected_axis
```

### 下一轮最小可执行协议

1. 固定 `high_info_calibration_v1` prompt set。
2. 从 `0.75` 或 `global_tasknorm_e3` 附近作为 anchor，不从 `0/0.5` 直接长跑。
3. 每轮：

```text
bake current gate
vLLM rollout fixed prompt set
official reward scoring
GRPO update one small step
expert recovery distill one small step
bake candidate
run Tool/Memory/Code sanity guard
accept or reject candidate
```

4. 收敛标准：

```text
连续 2-3 轮：
  kept frontier 数减少或 mean_abs_advantage 接近 0
  gate_delta 很小
  sanity 不退化
```

5. 只有 global 三系数都在合理区间并通过 full eval 后，再跑 588 系数版本。588 版本需要更强 prior / bound / group regularization，否则会把少量 calibration 过拟合到局部模块。

### 对当前 idea 的判断

这条路线是可行的，但重点应从“让 reward 推动系数”改为“让 calibration row 暴露方向”：

```text
有方差的行 -> GRPO
无方差但 expert 可恢复的行 -> distill / pairwise
已会的行 -> retention / hard guard
阶段提升行 -> self-compare advantage
```

这样 100 条数据不是普通训练集，而是一个固定的能力探针面板。它既能保持 on-policy，又不会变成不断挖新数据的直接训练。

## 2026-05-11 high-info v1 artifact

已新增构造脚本：

```text
scripts/data/build_high_info_calibration.py
```

生成的固定数据 bundle：

```text
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.prompts.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.guard.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.bundle.json
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.summary.json
```

配额与实际数量：

```text
on-policy prompts:
  Tool   30
  Memory 35
  Code   25
  total  90

guard prompts:
  Tool   4
  Memory 3
  Code   3
  total  10

offline distill rows:
  Tool   13
  Memory 13
  Code   13
  total  39
```

Prompt 选择：

```text
frontier seeded:
  Tool   7
  Memory 7
  Code   7

official correct fill:
  Tool   23
  Memory 28
  Code   18
```

Memory prompt 检查：

```text
35 / 35 memory prompts have MemAgent trajectory chunks
chunk count: min 5, avg 5.77, max 6
```

Distill 选择：

```text
candidate rows after filtering:
  Tool   21
  Memory 25
  Code   24

selected:
  Tool   13, samples 65, positive 14, nonpositive 51, avg row std 0.4069
  Memory 13, samples 65, positive 17, nonpositive 48, avg row std 0.4207
  Code   13, samples 104, positive 43, nonpositive 61, avg row std 0.3791

skipped:
  all_success 25
```

这里跳过 all-success distill rows 是有意的：Stage B 需要 expert-positive 和 weak/negative 样本形成 contrast；全正样本更适合 guard/retention，不适合 pairwise distill。

### high-info v1 训练入口

已新增脚本：

```text
skill/command/run_high_info_global_two_stage.sh
```

dry-run 已通过：

```text
DRY_RUN=1 NUM_ITERS=1 RUN_NAME=high_info_v1_global_twostage_dryrun \
  bash skill/command/run_high_info_global_two_stage.sh
```

脚本实际执行流程：

```text
for each iteration:
  bake current gate
  vLLM rollout 90 fixed prompts, 4 samples each
  Stage A update:
    current-policy GRPO only
    ppo_loss_weight = 1.0
    task-normalized advantages
    length-normalized PPO logprob
  Stage B update:
    offline expert recovery only
    ppo_loss_weight = 0
    best_response_loss_weight = 0.05
    pairwise_loss_weight = 0.05
    max_pairwise_pairs_per_row = 4
    length-normalized response logprob
  bake candidate
  rollout 10 guard prompts, 2 greedy samples each
  summarize guard metrics
  fail by default if Tool parseable collapses
```

默认关键参数：

```text
NUM_ITERS=3
SAMPLES_PER_PROMPT=4
MAX_NEW_TOKENS=2048
MAX_PROMPT_TOKENS=8192
MAX_LOGPROB_TOKENS=8192
MAX_MODEL_LEN=12288
LR_STAGE_A=0.003
LR_STAGE_B=0.002
PRIOR_LOSS_WEIGHT=0.02
MAX_COEFF_DELTA_STAGE=0.04
ENFORCE_GUARD=1
MIN_TOOL_PARSEABLE_RATE=0.25
MAX_TOOL_ZERO_CALL_RATE=0.75
```

注意：`MAX_COEFF_DELTA_STAGE=0.04` 是 per-stage bound。它避免从 0.75 直接冲过 Tool 的危险窗口；如果增加 `NUM_ITERS`，必须看每轮 guard summary 和最终 gate，不能盲目长跑。

辅助 summary 脚本：

```text
scripts/eval/summarize_rollouts.py
```

它统计：

```text
task mean/std/min/max reward
success_rate
mean/max length
kept_frontier_rows
Tool parseable rate
Tool zero-call rate
Tool exact-call rate
```

下一步建议先跑：

```text
NUM_ITERS=1 RUN_NAME=high_info_v1_global_twostage_i1_probe \
  bash skill/command/run_high_info_global_two_stage.sh
```

目的不是直接收敛，而是检查：

```text
1. 90 prompt rollout 中三任务各有多少 kept frontier。
2. Stage A/B 后 global.tool/global.memory/global.code 分别移动多少。
3. guard 是否拦截 Tool collapse。
4. 单轮耗时是否可接受。
```

如果单轮结果健康，再跑 `NUM_ITERS=3`；如果 Tool 或 Memory 仍然无梯度，就需要调 Stage B loss/数据角色，而不是扩大 prompt 数。

## 2026-05-11 high-info v1 一轮 probe

实际启动：

```text
GPU_LIST=0,1,2,3 NUM_ITERS=1 RUN_NAME=high_info_v1_global_twostage_i1_probe_20260511 \
  bash skill/command/run_high_info_global_two_stage.sh
```

中途情况：

- vLLM rollout 成功完成。
- Stage A 第一次在 `0,1,2,3` 上 OOM，因为物理 GPU2/3 后来被外部进程占用。
- 复用已完成的 `rollouts.jsonl`，改用 `CUDA_VISIBLE_DEVICES=0,1,6,7` 重新跑 Stage A 成功。
- Stage B 第一次发现代码问题：best-response-only 不使用 `old_logprob`，但更新器仍检查 `old_logprob_max_length`，导致离线 distill rows 的 `768 vs 8192` mismatch。
- 已修复：

```text
scripts/train/opvec_update_gates_from_rollouts.py
```

修复点：`best_response_only` 时不校验未使用的 old-logprob max length。

### On-policy rollout 结果

产物：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.summary.json
```

耗时：

```text
90 prompts x 4 samples
elapsed_seconds = 776.3
```

任务信号：

```text
Tool:
  rows 30, samples 120
  kept_frontier_rows 6
  mean_reward 3.3035
  success_rate 0.9583
  parseable_rate 0.8417
  zero_call_rate 0.1583
  exact_rate 0.8000

Memory:
  rows 35, samples 140
  kept_frontier_rows 7
  mean_reward 0.9214
  success_rate 0.9214
  max_length 6933

Code:
  rows 25, samples 100
  kept_frontier_rows 12
  mean_reward 0.7663
  success_rate 0.8600
```

判断：high-info v1 有效。它不是 Code-only frontier；Tool/Memory/Code 都有 kept frontier 和 reward 方差。

### Stage A: on-policy GRPO

配置：

```text
ppo_loss_weight = 1.0
task_normalize_advantages = true
length_normalize_policy_logprob = true
lr = 0.003
max_coefficient_delta_from_init = 0.04
```

结果：

```text
raw:
  common          0.762539
  tool_residual   0.019172
  memory_residual -0.011725
  code_residual   -0.007447

effective coefficients:
  Tool   0.781710
  Memory 0.750814
  Code   0.755091

updates = 25
grad_norm_max = 0.2572
gate_delta_max = 0.0192
```

判断：

- Stage A 是健康信号：Tool 被轻微推高，Memory/Code 仍在 0.75 附近。
- 没有出现 Tool collapse。
- 这比之前盲跑 fixed frontier 更合理，因为三任务都有信号且 Tool 格式未坏。

Stage A guard：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_guard_rollouts.summary.json
```

结果：

```text
Tool guard:
  mean_reward 4.0
  exact_rate 1.0
  parseable_rate 1.0
  zero_call_rate 0.0

Memory guard:
  mean_reward 1.0
  mean_length 1306.7

Code guard:
  mean_reward 1.0
```

正式 Stage A checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/high_info_v1_global_stage_a_i1_probe_20260511
```

### Stage B: original distill probe

原配置：

```text
best_response_loss_weight = 0.05
pairwise_loss_weight = 0.05
lr = 0.002
max_coefficient_delta_from_init = 0.04
```

结果：

```text
raw:
  common          0.736662
  tool_residual   0.005049
  memory_residual -0.025848
  code_residual    0.020799

effective coefficients:
  Tool   0.741710
  Memory 0.710814
  Code   0.757460

updates = 39
grad_norm_max = 3.3163
gate_delta_max = 0.0282
```

判断：

- Distill 梯度远大于 Stage A。
- 它把 Tool/Memory 从 Stage A 的健康位置往下拉，方向与 on-policy reward 不一致。
- 即使 guard 通过，也不应把这个 Stage B 结果作为主候选。

Stage B guard：

```text
Tool exact/parseable = 1.0, zero_call = 0.0
Memory mean_reward = 1.0, mean_length = 1876.5
Code mean_reward = 1.0
```

注意：guard 太小，只能排除 collapse，不能证明 Stage B 更好。

### Stage B: low-weight distill probe

低权重配置：

```text
best_response_loss_weight = 0.005
pairwise_loss_weight = 0.0
lr = 0.001
max_coefficient_delta_from_init = 0.01
```

结果：

```text
raw:
  common          0.756611
  tool_residual   0.015099
  memory_residual -0.015797
  code_residual    0.000698

effective coefficients:
  Tool   0.771710
  Memory 0.740814
  Code   0.757309

grad_norm_max = 0.1725
gate_delta_max = 0.0081
```

判断：

- 降权后不再大幅破坏，但方向仍然是压低 Tool/Memory。
- 说明当前 offline expert-recovery distill 不是简单权重过大，而是目标/数据角色与 on-policy reward 不完全一致。
- 下一步不应继续加强 Stage B；应先用 Stage A-only 进入下一轮或小评测。

### 脚本默认值修正

已修改：

```text
skill/command/run_high_info_global_two_stage.sh
```

新的默认行为：

```text
RUN_STAGE_B=0
```

即默认只跑 Stage A on-policy GRPO + guard。Stage B 变成显式 opt-in：

```text
RUN_STAGE_B=1
```

同时 Stage B opt-in 默认降为：

```text
DISTILL_BEST_RESPONSE_LOSS_WEIGHT=0.005
DISTILL_PAIRWISE_LOSS_WEIGHT=0.0
LR_STAGE_B=0.001
MAX_COEFF_DELTA_STAGE_B=0.01
```

### 当前判断

当前最合理候选是 Stage A-only：

```text
gates:
  Tool   0.7817
  Memory 0.7508
  Code   0.7551
```

它仍然接近初始 0.75，但方向正确、没有破坏 Tool，并且 high-info v1 证明确实能产生三任务梯度信号。下一步应对 Stage A checkpoint 做小规模固定评测，而不是继续把 offline distill 混进去。

## 2026-05-11 high-info v1 第二轮 Stage A-only

启动：

```text
GPU_LIST=0,1,6,7 \
NUM_ITERS=1 \
RUN_STAGE_B=0 \
RUN_NAME=high_info_v1_global_stagea_i2_probe_20260511 \
INIT_GATE_CHECKPOINT=/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_gate_updates.gates.json \
  bash skill/command/run_high_info_global_two_stage.sh
```

产物：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_stagea_i2_probe_20260511/iter_001/rollouts.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_stagea_i2_probe_20260511/iter_001/rollouts.summary.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_stagea_i2_probe_20260511/iter_001/stage_a_gate_updates.gates.json
/tmp/shared-storage/OnPolicy/checkpoints/high_info_v1_global_stagea_i2_probe_20260511
```

Rollout：

```text
90 prompts x 4 samples
elapsed_seconds = 728.9
kept_frontiers = 20
```

任务信号：

```text
Tool:
  kept_frontier_rows 4
  mean_reward 3.4229
  success_rate 0.9750
  parseable_rate 0.8500
  zero_call_rate 0.1500
  exact_rate 0.8167

Memory:
  kept_frontier_rows 2
  mean_reward 0.9500
  success_rate 0.9500

Code:
  kept_frontier_rows 14
  mean_reward 0.7363
  success_rate 0.7800
```

和第一轮对比：

```text
kept frontier:
  Tool   6 -> 4
  Memory 7 -> 2
  Code   12 -> 14
  Total  25 -> 20

mean reward:
  Tool   3.3035 -> 3.4229
  Memory 0.9214 -> 0.9500
  Code   0.7663 -> 0.7363
```

这说明 Tool/Memory 在这批 calibration 上更接近饱和，Code 仍有较多 frontier。

第二轮 Stage A 更新：

```text
updates = 20
grad_norm_max = 0.0332
gate_delta_max = 0.0154
```

系数：

```text
i1 Stage A effective:
  Tool   0.781710
  Memory 0.750814
  Code   0.755091

i2 Stage A effective:
  Tool   0.759280
  Memory 0.735358
  Code   0.759257
```

Guard：

```text
Tool exact/parseable = 1.0, zero_call = 0.0
Memory mean_reward = 1.0
Code mean_reward = 1.0
```

判断：

- 第二轮没有 collapse，但它不再是明确的“各能力增长”方向。
- 因为 Code frontier 占主导，global common 被拉低，Memory effective 下降到 0.735。
- `grad_norm_max` 明显下降，说明信号在减弱，但不是完全收敛；它转成了偏 Code 的剩余信号。
- 不建议盲跑第三轮 global。更合理的选择：
  1. 以 i1 Stage A 作为当前候选做小规模评测。
  2. 或在第二阶段只训练 `global.code`，冻结 Tool/Memory，专门处理剩余 Code frontier。
  3. 或重构 calibration，让 Tool/Memory 后续轮次继续提供 self-compare / hard-negative 信号。

当前不应进入 588 系数训练。理由：global 还没有稳定地朝三能力共同更优方向收敛，细粒度 588 会放大 calibration 偏置。

## 2026-05-11 high-info v1 i1 Stage A full eval

评测 checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/high_info_v1_global_stage_a_i1_probe_20260511
```

评测入口：

```text
RUN_ID=high_info_v1_stagea_i1_full_20260511
MODEL_NAME=high-info-v1-stagea-i1
summary_dir=/tmp/shared-storage/OnPolicy/eval/full_suite/high-info-v1-stagea-i1/high_info_v1_stagea_i1_full_20260511
```

### Tool / BFCL

```text
parallel:               90.00%  (180/200)
parallel_multiple:      86.50%  (173/200)
live_parallel:          75.00%  (12/16)
live_parallel_multiple: 66.67%  (16/24)
```

对比 `opvec-fixed075/tool_initial075_20260511`：

```text
initial 0.75:
  parallel               90.00%
  parallel_multiple      86.50%
  live_parallel          75.00%
  live_parallel_multiple 66.67%

high-info i1 Stage A:
  parallel               90.00%
  parallel_multiple      86.50%
  live_parallel          75.00%
  live_parallel_multiple 66.67%
```

判断：Tool 没有退化，但 full BFCL 没显示超过 0.75 初始点。calibration 内 Tool reward 被推高更像是保持/局部修复，而不是全局 Tool 能力提升证据。

### Memory / HotpotQA

```text
eval_50:
  EM     80/128 = 0.6250
  subEM  102/128 = 0.796875
  avg_f1 0.76556

eval_100:
  EM     79/128 = 0.6171875
  subEM  102/128 = 0.796875
  avg_f1 0.75906

eval_qa_1_32768:
  EM     84/128 = 0.65625
  subEM  106/128 = 0.828125
  avg_f1 0.76193

eval_qa_1_65536:
  EM     84/128 = 0.65625
  subEM  102/128 = 0.796875
  avg_f1 0.74804
```

对比 `opvec-fixed075/memory50_initial075_20260511` 的 `eval_50`：

```text
initial 0.75 eval_50:
  EM     0.6328125
  subEM  0.796875
  avg_f1 0.76252

high-info i1 Stage A eval_50:
  EM     0.6250
  subEM  0.796875
  avg_f1 0.76556
```

判断：Memory 没 collapse，但不能说强于 0.75。EM 略低，F1 略高，属于基本持平。

### Code / CURE

```text
LiveBench:
  code_acc                 0.38477
  code_accumulate_acc      0.47674
  estimated_unit_test_acc  0.41111
  BoN(4,4) acc             0.46875
  BoN(4,4) accumulate_acc  0.55338

LiveCodeBench:
  code_acc                 0.31262
  code_accumulate_acc      0.45627
  estimated_unit_test_acc  0.43833
  BoN(4,4) acc             0.38160
  BoN(4,4) accumulate_acc  0.53018
```

对比 `fixed-frontier7-global-e3b/code_global_e3b_20260511`：

```text
fixed-frontier7 e3b:
  LiveBench code_acc       0.38867
  LiveBench BoN acc        0.41406
  LiveCodeBench code_acc   0.30577
  LiveCodeBench BoN acc    0.34442

high-info i1 Stage A:
  LiveBench code_acc       0.38477
  LiveBench BoN acc        0.46875
  LiveCodeBench code_acc   0.31262
  LiveCodeBench BoN acc    0.38160
```

判断：Code 不是单调全面提升。raw LiveBench 略低于 e3b，但 BoN 和 LiveCodeBench 更好。说明 high-info v1 对代码的采样恢复能力有正信号，但还不能作为单点最优。

### Full Eval 结论

`high-info v1 i1 Stage A` 是目前最干净的候选：三任务均无明显 collapse，系数也合理：

```text
Tool   0.7817
Memory 0.7508
Code   0.7551
```

但它不是“已经证明优于 0.75”的最终 checkpoint：

- Tool full BFCL 与 0.75 完全相同。
- Memory eval_50 与 0.75 基本持平，EM 略低、F1 略高。
- Code 对 fixed-frontier7 e3b 有部分改善，尤其 BoN 和 LiveCodeBench，但不是全部指标占优。

因此下一步不应直接进入 588 gate。更合理的是继续在 global / layer-band 小参数空间里验证 calibration 目标，特别是加入 self-compare 和条件式 distill，先证明 fixed 100 prompt 能稳定选出 Pareto 更好的 task-vector 组合。

## Calibration Data 科研表述

当前目标不应表述成“用 100 条数据继续训练模型”，而应表述成：

```text
用固定约 100 个能力探针 prompt，通过 on-policy rollout + official reward，
自动识别多个 expert task vector 的组合系数。
```

这样 calibration data 的角色是能力测量面板，不是新训练集。为了避免持续挖掘变成普通训练，prompt_id 必须冻结；每轮只更新同一批 prompt 的 response、reward、self-compare delta。

推荐三类训练信号：

```text
1. frontier_grpo
   当前 policy 在同一 prompt 上采样出成功/失败方差。
   用 official reward 做组内相对 advantage。

2. expert_recovery_distill
   当前 policy 全错或无方差，但某个 expert 的官方 reward 为正。
   只作为小权重方向补偿，不混进 PPO。

3. self_compare_delta
   当前 candidate 和 anchor/previous policy 在同一 prompt 上的 reward delta。
   用于无方差或弱方差场景，防止只追 calibration 内部组内噪声。
```

Reward 的角色应分开：

```text
official absolute reward:
  定义能力，所有任务必须回到官方 verifier。

group-relative reward:
  GRPO 主 advantage，避免 Tool reward scale 直接压过 Code/Memory。

base comparison:
  只用于选样和诊断，不直接当训练 reward。

previous/self comparison:
  作为辅助 advantage field，而不是替代 official reward。
```

这解释了为什么当前 Stage B distill 会有风险：离线 expert response 的 logprob 方向不一定等价于当前 task-vector 组合下的 reward 提升方向。已有实验证据：

```text
i1 Stage A:
  Tool   0.7817
  Memory 0.7508
  Code   0.7551

Stage B original:
  Tool   0.7417
  Memory 0.7108
  Code   0.7575

Stage B low-weight:
  Tool   0.7717
  Memory 0.7408
  Code   0.7573
```

即使低权重 distill 仍会压低 Tool/Memory，因此 distill 应该条件触发：

```text
if prompt has on-policy reward variance:
  use GRPO only
elif current policy all-fail/no-variance and expert positive exists:
  use low-weight distill
else:
  skip for update, keep for guard/diagnosis
```

### 固定 100 prompt 的建议构成

```text
Tool:   30
Memory: 35
Code:   25
Guard:  10
```

选择标准：

```text
Tool:
  parallel/multiple/tool-call 格式边界；
  parseable 与 exact correctness 都能区分；
  不能只选当前 0.75 已经全对的题。

Memory:
  使用完整 trajectory reward，不只 final answer；
  长轨迹、证据更新、跨轮依赖优先；
  必须 length-normalize，否则 memory 梯度容易因长输出过大。

Code:
  选择 pass/fail 有方差、public/private 边界、BoN 可恢复题；
  Code frontier 天然多，可少于 Memory。

Guard:
  不参与训练；
  Tool parseable/zero-call、Memory reward/长度、Code heldout pass 作为 hard rejection。
```

### Init 建议

```text
主线：0.75
  当前唯一能同时保留 Tool/Memory/Code 能力并产生三任务 frontier 的起点。

0.5:
  可作为 ablation 或 distill warmup 研究，但不适合 reward-only GRPO 长跑。

0:
  只适合证明 task vector 是否必要，不适合作为 fixed 100 calibration 主起点。
```

### 下一步实验建议

先不要上 588。推荐最小实验矩阵：

```text
A0: 0.75 initial full/sanity baseline
A1: high-info i1 Stage A, 已完成 full eval
A2: high-info i2 Stage A, 只做 sanity/guard，不直接信任
A3: i1 rollout 上限制 frontier quota=4/4/4，降低 Code 主导
A4: self-compare delta probe，用 candidate vs anchor reward_delta_vs_baseline
A5: conditional distill only on all-fail/no-variance expert-recovery prompts
```

接受标准：

```text
Tool:
  BFCL 不低于 0.75，zero-call 不升高。

Memory:
  EM/F1 至少不低于 0.75，trajectory 长度不爆。

Code:
  LiveBench/LiveCodeBench raw 或 BoN 至少有一个稳定改善，不能靠单一 calibration 内 reward 解释。

Gate:
  global 系数不应大幅偏离 0.75；
  多轮后如果只剩 Code frontier，应停止 global 更新或冻结 Tool/Memory。
```

只有当 global / layer-band 在 fixed 100 prompt 上能稳定产生 Pareto 更好的 checkpoint，才进入 588。否则 588 会把 calibration 偏置放大成过拟合。

## 2026-05-11 balanced-quota probe

动机：i1 Stage A 的有效 frontier 分布是：

```text
Tool   6
Memory 7
Code   12
```

为了确认 i1 的方向不是由 Code frontier 数量主导，复用同一个 i1 rollout，不重新生成样本，只在 update 阶段强制每个任务最多使用 4 条 frontier：

```text
run_dir=/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_i1_balquota4_probe_20260511
rollout=/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl
frontier_task_quota: tool=4, memory=4, code=4
lr=0.003
prior_loss_weight=0.02
task_normalize_advantages=true
length_normalize_policy_logprob=true
max_coefficient_delta_from_init=0.04
```

结果：

```text
raw_frontier_task_counts:
  Tool   6
  Memory 7
  Code   12

used_frontier_task_counts:
  Tool   4
  Memory 4
  Code   4

updates = 12
filled_missing_old_logprobs = 48
grad_norm_max = 0.2763
gate_delta_max = 0.01744
```

系数：

```text
raw:
  common          0.761180
  tool_residual   0.017435
  memory_residual -0.008996
  code_residual   -0.008439

effective:
  Tool   0.778615
  Memory 0.752183
  Code   0.752741
```

对比 i1 Stage A：

```text
i1 Stage A effective:
  Tool   0.781710
  Memory 0.750814
  Code   0.755091

balanced-quota effective:
  Tool   0.778615
  Memory 0.752183
  Code   0.752741
```

判断：balanced quota 后仍然得到接近 i1 的方向，说明 i1 的 Tool 上调不是纯粹由 Code frontier 数量造成的。区别是 balanced-quota 对 Memory/Code 更保守，整体仍围绕 0.75。

Guard：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_i1_balquota4_probe_20260511/guard_rollouts.summary.json

Tool:
  mean_reward 4.0
  exact_rate 1.0
  parseable_rate 1.0
  zero_call_rate 0.0

Memory:
  mean_reward 1.0
  max_length 3199

Code:
  mean_reward 1.0
```

结论：balanced-quota probe 通过 guard，但它与 i1 Stage A 太接近，且 i1 已经完成 full eval，因此不值得立即再跑一次 full eval。它更适合作为“均衡 frontier 不改变主方向”的补充证据。

## 2026-05-11 self-compare delta probe

目的：验证“和 anchor / previous policy 比”是否能提供比纯组内 GRPO 更稳定的方向。

构造：

```text
baseline / anchor:
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl
  含义：0.75 初始 policy 在固定 high-info v1 prompts 上的 rollout。

candidate:
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_stagea_i2_probe_20260511/iter_001/rollouts.jsonl
  含义：i1 Stage A gate 在同一批 prompts 上的 rollout。

self-compare field:
  reward_delta_vs_baseline = candidate_sample_reward - mean(anchor_sample_rewards_for_same_prompt)
```

先构造宽松版本：

```text
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_selfcompare_i2_vs_i1_mean_seed20260511.jsonl
```

结果：

```text
selected rows = 19 / 90

candidate_counts:
  Code   14
  Memory 2
  Tool   3

skipped:
  no_self_compare_signal = 71
```

解释：i1 Stage A 之后，固定 high-info prompts 上能相对 0.75 anchor 产生 self-compare delta 的样本非常少，而且明显偏 Code。Tool/Memory 的 remaining signal 很稀疏，说明它们在这批 prompts 上接近饱和或缺少 hard-negative。

为了避免 Code 再主导，构造 balanced 版本：

```text
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_selfcompare_i2_vs_i1_bal2_seed20260511.jsonl
quota: Tool 2 / Memory 2 / Code 2
baseline_agg: mean
min_abs_delta: 0.05
min_delta_std: 0.05
require_cross_baseline: false
```

delta stats：

```text
Tool:
  rows 2
  mean_delta_avg 0.0000
  std_delta_avg 0.7371
  positive_samples 5
  negative_samples 3

Memory:
  rows 2
  mean_delta_avg 0.0000
  std_delta_avg 0.4665
  positive_samples 5
  negative_samples 3

Code:
  rows 2
  mean_delta_avg 0.2031
  std_delta_avg 0.5000
  positive_samples 4
  negative_samples 4
```

Update 设置：

```text
run_dir=/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_selfcompare_i2_vs_i1_bal2_probe_20260511
init_gate_checkpoint=i1 Stage A gate
advantage_field=reward_delta_vs_baseline
task_normalize_advantages=true
length_normalize_policy_logprob=true
lr=0.001
max_coefficient_delta_from_init=0.02
prior_loss_weight=0.02
```

结果：

```text
updates = 6
filled_missing_old_logprobs = 24
grad_norm_max = 0.02562
gate_delta_max = 0.00289

raw:
  common          0.759778
  tool_residual   0.016593
  memory_residual -0.012034
  code_residual   -0.004559

effective:
  Tool   0.776372
  Memory 0.747744
  Code   0.755219
```

对比：

```text
i1 Stage A effective:
  Tool   0.781710
  Memory 0.750814
  Code   0.755091

self-compare bal2 effective:
  Tool   0.776372
  Memory 0.747744
  Code   0.755219
```

Guard：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_selfcompare_i2_vs_i1_bal2_probe_20260511/guard_rollouts.summary.json

Tool:
  mean_reward 4.0
  exact_rate 1.0
  parseable_rate 1.0
  zero_call_rate 0.0

Memory:
  mean_reward 1.0
  max_length 3199

Code:
  mean_reward 1.0
```

结论：

- self-compare 作为机制是可用的：脚本能构造 delta rows，update 能用 `reward_delta_vs_baseline` 反传，guard 不崩。
- 但当前 `i2 vs i1` 的 self-compare 数据太稀疏，且宽松版本明显偏 Code：`Code 14 / Memory 2 / Tool 3`。
- balanced 2/2/2 后，更新方向没有优于 i1；Tool/Memory 反而轻微下降，Code 基本不动。
- 这说明 self-compare 不能简单替代 GRPO。它应该作为诊断和补无方差信号的工具，而不是当前主优化目标。

对 calibration 设计的含义：

```text
1. 固定 prompts 是对的，但当前 Tool/Memory hard-negative 不够。
2. 若要让 self-compare 真正有用，需要在固定 100 prompts 中加入更多 Tool/Memory 的临界题：
   - Tool: parseable/exact 边界、wrong_count、multi-call 参数错题。
   - Memory: 长轨迹 evidence update、非 final-answer-only 的 trajectory reward 边界。
3. 对 i1 之后的下一阶段，不能让 Code-only self-compare 主导 global gate。
4. 588 仍不应启动；self-compare 进一步证明剩余信号不均衡。
```

## 2026-05-11 conditional distill diagnostic

目的：确认当前 `high_info_v1.distill.jsonl` 能否作为“固定 100 prompts 内，当 current policy all-fail/no-variance 时的小权重补信号”。

检查对象：

```text
prompts:
  /tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.prompts.jsonl

distill:
  /tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill.jsonl

current rollout:
  /tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_stagea_i2_probe_20260511/iter_001/rollouts.jsonl
```

结果：

```text
fixed prompt counts:
  Tool   30
  Memory 35
  Code   25

distill counts:
  Tool   13
  Memory 13
  Code   13

prompt_id intersection between fixed prompts and distill:
  Tool   0
  Memory 0
  Code   0
```

i2 rollout 的无方差情况：

```text
i2 rows:
  Tool   30
  Memory 35
  Code   25

all_success rows:
  Tool   28
  Memory 32
  Code   14

all_fail rows:
  Memory 1
  Code   1

no_variance rows:
  Tool   27
  Memory 33
  Code   11

all_fail + no_variance rows:
  Memory 1
  Code   1

all_fail_no_variance intersection with current distill:
  Memory 0
  Code   0
```

结论：

- 当前 distill rows 不是 fixed prompt panel 的专家响应缓存，而是另一批离线 expert-recovery 样本。
- 因此当前 Stage B 不能被解释为“同一 calibration data 上的条件式补信号”；它实际是在另一个数据分布上优化 expert text logprob。
- 这解释了为什么 Stage B original/low-weight 都会把 Tool/Memory 往下拉。
- 若要做 conditional distill，必须重构数据：对固定 90/100 个 prompt 中的 all-fail/no-variance prompt，补同 prompt_id 的 expert positive rollout / trajectory，并用官方 reward 验证。
- 在完成这个重构前，不应启用 Stage B，也不应进入 588。

### Minimal same-prompt expert recovery probe

i2 中 all-fail/no-variance 的固定 prompt：

```text
Memory:
  memory__af530d66348a2c84

Code:
  code__e544f56a6c808c77
```

用对应 expert model 对同一个 prompt_id 重新 rollout：

```text
memory expert:
  model=/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B
  output=/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_conditional_distill_probe_20260511/memory_expert_rollouts.jsonl

code expert:
  model=/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B
  output=/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_conditional_distill_probe_20260511/code_expert_rollouts.jsonl
```

官方 reward 结果：

```text
memory__af530d66348a2c84:
  rewards = [0.0, 1.0, 0.0, 0.0]
  mean_reward = 0.25
  std_reward = 0.4330
  frontier_weight = 0.75
  keep_for_policy_loss = true

code__e544f56a6c808c77:
  rewards = [1.0, 1.0, 0.0, 0.0]
  mean_reward = 0.50
  std_reward = 0.50
  frontier_weight = 1.0
  keep_for_policy_loss = true
```

结论：

- same-prompt expert recovery 是可行的：对 i2 all-fail/no-variance 的 Memory/Code prompt，对应 expert 能产生 official-positive samples。
- 这说明 conditional distill 的正确数据形态应该是：固定 prompt panel 中失败/无方差的 prompt，加上同 prompt_id 的 expert positive samples。
- 但目前只有 2 条验证样本，不能直接作为训练结论；需要把该流程系统化到固定 100 prompt 的构造脚本中。

## Completion Audit

目标拆解：

```text
1. 选择有梯度信号且三任务均衡的 calibration data。
2. 固定 data。
3. 使用规范配置训练。
4. 训练到梯度信号消失/模型收敛后停止。
5. 进入完整评测并计算指标。
6. 如果 task vector 系数合理，分析实验结果。
7. 如果整体权重合理，可以训练 588 细粒度版本并分析。
8. 将今晚 loop 记录到项目文件夹。
```

证据清单：

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 选择 calibration data | `high_info_v1_seed20260511.summary.json`：prompts 90, guard 10, distill 39；prompt quotas Tool 30 / Memory 35 / Code 25；distill quotas 13/13/13；无 deficits | 已完成 |
| 有梯度信号 | i1 rollout kept frontier Tool 6 / Memory 7 / Code 12；balanced-quota probe 使用 4/4/4 后仍有非零梯度；self-compare 仅得到 Code 14 / Memory 2 / Tool 3，说明后续 delta 信号不均衡 | 已完成 |
| 固定 data | `high_info_v1_seed20260511.prompts.jsonl`、`.guard.jsonl`、`.distill.jsonl` 均固定在 `/tmp/shared-storage/OnPolicy/data/calibration/` | 已完成 |
| 规范配置 | Stage A 使用 official reward、`behavior_span_reward_weight=0.0`、task-normalized advantages、length-normalized PPO logprob、prior loss、guard；Stage B 默认关闭 | 已完成 |
| 训练 | i1 Stage A、i2 Stage A、balanced-quota probe、self-compare bal2 probe 均完成；Stage B probe 证明方向冲突后默认关闭 | 已完成 |
| 收敛/停止 | i2 的 `grad_norm_max=0.0332` 相比 i1 明显下降，但没有严格归零；i2 frontier 转为 Code 主导；停止理由是“剩余信号不再均衡”，不是严格数学收敛 | 弱完成 |
| full eval | i1 Stage A 完整跑了 Tool BFCL、Memory HotpotQA、Code CURE | 已完成 |
| 指标计算 | Tool: 90/86.5/75/66.67；Memory: EM 0.617-0.656；Code: LiveBench acc 0.3848, LiveCodeBench acc 0.3126 等 | 已完成 |
| 系数合理性分析 | i1 effective Tool 0.7817 / Memory 0.7508 / Code 0.7551；balanced-quota Tool 0.7786 / Memory 0.7522 / Code 0.7527；self-compare bal2 Tool 0.7764 / Memory 0.7477 / Code 0.7552；均无低于 0.5 | 已完成 |
| 588 细粒度 | 尚未对 high-info v1 训练 588；已有历史 fixed-frontier7 588/parameter 结果不支持直接上 588 | 未触发/不建议 |
| 条件式 distill | 诊断发现当前 distill rows 与固定 prompts 的 prompt_id 交集为 0，不能作为同一 fixed panel 的 all-fail/no-variance 补信号 | 当前数据不支持 |
| loop 记录 | `loop.md` 已记录数据、训练、评测、balanced-quota probe、self-compare、conditional distill 诊断、下一步建议 | 已完成 |

Audit 结论：

```text
不能把整个目标标记为完全完成。
```

原因：

1. 训练停止不是严格“梯度消失/收敛”，而是因为第二轮剩余 frontier 明显转向 Code，继续 global 更新会偏离三任务均衡目标。
2. i1 Stage A full eval 没有证明相对 0.75 初始点 Pareto 改善：Tool 持平，Memory 基本持平，Code 部分指标改善。
3. 588 的进入条件不充分。虽然 global 系数在合理区间，但全评测没有证明 global 已经是更优起点；贸然上 588 会放大 calibration 偏置。
4. 当前 distill 数据与固定 prompts 不重合，不能作为 conditional distill 使用；需要补同 prompt_id 的 expert positive responses。

下一步应补的不是盲训 588，而是：

```text
1. self-compare delta probe：
   在固定 100 prompt 上构造 candidate vs anchor/previous 的 reward_delta_vs_baseline。

2. conditional distill：
   只对当前 policy all-fail/no-variance 但 expert positive 的 prompt 启用小权重 distill。

3. layer-band 或受限模块级：
   如果 global + self-compare 能产生 Pareto 改善，再进入更细粒度；优先 layer-band，不直接全量 588。
```
