---
title: OnPolicy 数据筛选和 Reward 构建
date: 2026-05-11
tags:
  - onpolicy
  - task-vector
  - calibration-data
  - reward
  - grpo
---

# OnPolicy 数据筛选和 Reward 构建

## 0. 当前结论

当前瓶颈不是模型完全不会，而是 calibration 数据的有效梯度比例低。最近 `high_info_v1` 的现象是：大量题在 0.75 task-vector 初始点已经全做对，raw reward 在同一个 prompt 内没有方差，GRPO 只能给出很弱甚至为 0 的更新信号。继续在同一批题上重复训练，会把系数推高一点，但不等价于找到可泛化的最优 task-vector 组合。

新的数据策略应从「直接构造 calibration data」改成两层：

1. 先构建可审计的 **question bank**：用 task-vector baseline 对本地原始/派生题源做 vLLM rollout，按官方 reward 统计每道题的 reward 分布。
2. 再从 question bank 按 bucket 采样 calibration data：重点选「偏低但至少有 rollout 能做对」和「中间成功率」样本；少量保留全错题做 expert recovery，少量保留全对题做 regression guard。

训练目标也应分层：

1. raw reward GRPO：用于 `low_but_solvable / mid_frontier / all_fail_partial`。
2. self-compare reward：用于 raw reward 饱和或需要持续改进时，优化 `delta_reward = reward(candidate_gate) - reward(reference_gate)`。
3. conditional distill：用于 all-fail/no-variance 但 expert 能在同 prompt 上做对的题。

核心原则：calibration data 不追求覆盖原始分布的所有题，而追求覆盖「task-vector 组合决策的边界」。题库负责泛化和审查，calibration 负责产生可用梯度。

---

## 1. 本机原始数据地址

### 1.1 Tool

| 层级 | 路径 | 用途 |
|---|---|---|
| ToolRL raw train | `/tmp/shared-storage/OnPolicy/external_repos/ToolRL/dataset/rlla_4k/train.parquet` | 主扩池源，3920 rows |
| ToolRL raw test | `/tmp/shared-storage/OnPolicy/external_repos/ToolRL/dataset/rlla_4k/test.parquet` | 小规模 sanity / heldout |
| ToolRL mirror | `/tmp/shared-storage/dataset/ToolRL/dataset/rlla_4k/train.parquet` | 同源镜像 |
| BFCL eval | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_*.json` | 官方工具调用评测 |
| BFCL answers | `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_*.json` | 官方答案 |
| ExpertMerging cache | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets/ToolCall.json` | 早期派生池，40 条 |
| routed correct | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1/ToolCall.json` | 当前 high_info 使用源，39 条 |

provenance：

```text
ToolRL repo: /tmp/shared-storage/OnPolicy/external_repos/ToolRL
remote: https://github.com/qiancheng0/ToolRL.git
commit: 8cee13ec0ca72f0461da372a93a6fd8140dbb840
columns: data_source, prompt, ability, reward_model, extra_info

BFCL repo: /mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard
remote: https://github.com/ShishirPatil/gorilla.git
commit: 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
```

建议：Tool 的 question bank 用 ToolRL train 扩池，BFCL 只做 heldout guard。不要把 BFCL 全量混进 calibration 主训，否则容易把评测分布泄漏到训练选择里。

### 1.2 Memory

| 层级 | 路径 | 用途 |
|---|---|---|
| ExpertMerging HotpotQA raw | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/hotpotqa/hotpotqa_train_32k.parquet` | 本轮 v1 主扩池源，32768 rows |
| ExpertMerging HotpotQA dev | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/hotpotqa/hotpotqa_dev.parquet` | heldout sanity，128 rows |
| MemAgent HotpotQA train mirror | `/tmp/shared-storage/dataset/hotpotqa/hotpotqa_train_32k.parquet` | 同源镜像 |
| MemAgent eval | `/tmp/shared-storage/dataset/hotpotqa/eval_*.json` | 官方 eval |
| MemAgent QA eval | `/tmp/shared-storage/dataset/hotpotqa/eval_qa_*.json` | 官方 QA eval |
| HotpotQA fullwiki | `/tmp/shared-storage/dataset/hotpot_qa/fullwiki/*.parquet` | 原始语料层 |
| HotpotQA distractor | `/tmp/shared-storage/dataset/hotpot_qa/distractor/*.parquet` | 原始语料层 |
| ExpertMerging cache | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets/Memory.json` | 589 rows，chunk 502，final 87 |
| routed correct | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1/Memory.json` | 576 rows，chunk 491，final 85 |

provenance：

```text
MemAgent repo: /tmp/shared-storage/OnPolicy/external_repos/MemAgent
remote: https://github.com/BytedTsinghua-SIA/MemAgent.git
commit: ce53a0656328c2fdf338364382635150e801cd74
```

核对结果：`routed_1/Memory.json` 的 85 个 final questions 中，79 个能按 question id/text 匹配到 `hotpotqa_train_32k.parquet`，0 个匹配 `hotpotqa_dev.parquet`。因此它主要是 train-derived correct-sample 小池，不是 eval 主源；可用于 sanity/guard，不适合作为 question-bank 主源。

建议：Memory 不能退化成普通 final-only prompt。筛选时必须保留 recurrent memory update 的全轨迹 rollout，再用 MemAgent 训练流程中的最终 QA reward 打分；扩池从 `hotpotqa_train_32k.parquet` 重新按 5000 token chunk 构造轨迹任务。

### 1.3 Code

| 层级 | 路径 | 用途 |
|---|---|---|
| CodeContests train | `/tmp/shared-storage/dataset/CodeContests_train/train/CodeContests_train.json` | 主扩池源，4529 rows |
| ExpertMerging symlink | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json` | 指向 shared dataset |
| CURE LiveBench | `/mnt/cache/wuruixiao/users/lsc/CURE/data/LiveBenchSmall20.json` | heldout eval |
| CURE LiveBench64 | `/mnt/cache/wuruixiao/users/lsc/CURE/data/LiveBenchSmall64.json` | heldout eval |
| CURE LiveCodeBench | `/mnt/cache/wuruixiao/users/lsc/CURE/data/LiveCodeBenchSmall20.json` | heldout eval |
| CURE LiveCodeBench64 | `/mnt/cache/wuruixiao/users/lsc/CURE/data/LiveCodeBenchSmall64.json` | heldout eval |
| ExpertMerging cache | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets/Code.json` | 25 条 |
| routed correct | `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1/Code.json` | 34 条 |

provenance：

```text
CURE repo: /mnt/cache/wuruixiao/users/lsc/CURE
remote: https://github.com/Gen-Verse/CURE.git
commit: 24f37c295231e460ca17d1a26a15815471476f53
CodeContests fields: question, solutions, example_input/output, test_input/output, test_time_limit, exe_method
```

建议：Code 的扩池用 CodeContests train，LiveBench/LiveCodeBench 保持 heldout。Code reward 可以继续使用当前官方 CURE/unit-test adapter。

---

## 2. 当前 baseline 模型

### 2.1 当前配置

```text
base:   /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct
tool:   /mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold
memory: /mnt/cache/wuruixiao/models/RL-MemoryAgent-7B
code:   /mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B
mode:   /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
```

`opvec4` 当前是 588 个系数：196 个可合并 module × 3 个 expert delta。global 参数化时只是把 588 个位置绑成几个共享系数；细粒度训练时才真正释放 588 个 gate。

### 2.2 推荐 question-bank reference baseline

question bank 的 reference 应使用无调参的静态 Task Arithmetic average-vector baseline，而不是 OP-VEC baked 0.75，也不是 TA sweep 里 validation 选出来的 c=0.50/c=0.75。推荐默认：

```text
/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_avgvec_c033333/model
```

这是三专家 average task-vector baseline。公式：

```text
theta_TA_avg = theta_base + (1/3) * Σ_e (theta_expert_e - theta_base)
```

这个点的科研含义是「不调 scaling factor，只平均三个 expert delta」。它适合作为 question-bank 的 neutral reference，因为不会把 validation sweep 的选择偏置带进数据筛选。

期望 summary：

```text
method: task_arithmetic
base_model: /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct
experts:
  /mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold
  /mnt/cache/wuruixiao/models/RL-MemoryAgent-7B
  /mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B
scaling_coefficient: 0.3333333333333333
exclude_regex: [".*lm_head.*", ".*norm.*", ".*embed_tokens.*", ".*bias.*"]
merged_tensors: 196
copied_excluded_tensors: 143
```

c=0.50 作为本地历史 validated TA baseline 保留为 secondary reference：

```text
/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ITER_009_ta_c050/model
```

它有明确的历史评测记录，BFCL 格式也健康：

```text
Tool BFCL:
  parallel: 91.00%
  parallel_multiple: 86.00%
  live_parallel: 68.75%
  live_parallel_multiple: 66.67%

Memory:
  eval_50 EM/F1: 0.5625 / 0.7055
  eval_100 EM/F1: 0.5859 / 0.7195

Code:
  LiveBench acc: 0.4023
  LiveCodeBench acc: 0.3043
```

### 2.3 TA 复现代码

标准 TA 的大模型可复现构建脚本：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/scripts/build_task_arithmetic_merge.py
```

它是 CPU streaming/sharded 版本，避免旧 CLI 一次性加载 base+3 experts 到 GPU OOM。核心实现是逐 shard、逐 tensor：

```text
merged = base_tensor + scaling_coefficient * Σ_e(expert_tensor_e - base_tensor)
```

对应历史记录：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/memory.zh.md
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/ta-scale-sweep-20260502-232830/ta_scale_sweep_report_zh.md
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/ta-scale-sweep-20260502-232830/ta_scale_sweep_summary.json
```

构建 c=1/3 average-vector baseline 的命令：

```bash
export PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python

$PY /mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/scripts/build_task_arithmetic_merge.py \
  --base-model /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct \
  --expert-models \
    /mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold \
    /mnt/cache/wuruixiao/models/RL-MemoryAgent-7B \
    /mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B \
  --output-dir /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_avgvec_c033333/model \
  --scaling-coefficient 0.3333333333333333 \
  --overwrite
```

### 2.4 其他 TA sweep 点

已有 sweep checkpoints：

```text
c=0.10  /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_scale_sweep_c010/model
c=0.333 /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_avgvec_c033333/model
c=0.50  /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ITER_009_ta_c050/model
c=0.75  /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_scale_sweep_c075/model
c=1.00  /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_scale_sweep_c100/model
c=2.00  /tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_scale_sweep_c200/model
```

c=0.50 和 c=0.75 可以作为 sweep/validated 对照，不作为默认 question-bank reference。OP-VEC baked 0.75 只用于检查当前 on-policy gate 初始化，不作为标准 TA baseline。

---

## 3. Reward 定义

### 3.1 Raw official reward

对任务 `t`、prompt `x_i`、第 `j` 个 rollout `y_ij`：

```text
r_ij = R_t(y_ij, metadata_i)
```

其中 `R_t` 必须是对应 expert 官方训练/评测流程中的 reward：

| Task | Reward 入口 | 说明 |
|---|---|---|
| Tool | ToolRL/BFCL 官方工具调用 reward | 需要校验 tool call 格式、函数名、参数正确性；不能只看自然语言答案 |
| Memory | MemAgent recurrent trajectory + final QA reward | rollout 保存 update/final 全轨迹；当前 reward 是 MemAgent 风格最终答案 exact/boxed 打分，不给每个 update turn 单独 dense reward |
| Code | CURE/CodeContests 单测 reward | 当前可保持 |

当前 `opvec_collect_vllm_rollouts.py` 通过 `RewardRouter()` 打 task reward。做官方 reward rollout 时要把行为 span shaping 关掉：

```bash
--behavior-span-reward-weight 0.0
```

success 的定义优先使用 rollout 中的 `success` 字段，而不是简单 `reward > 0`。原因是 Tool 和 Memory 可能存在 partial reward，positive reward 不一定等价于最终任务成功。

### 3.2 Prompt 级统计

对每道题采样 `K` 次：

```text
mean_reward_i   = mean_j(r_ij)
max_reward_i    = max_j(r_ij)
min_reward_i    = min_j(r_ij)
std_reward_i    = std_j(r_ij)
success_count_i = sum_j(success_ij)
success_rate_i  = success_count_i / K
```

raw GRPO 的有效信号主要来自同 prompt 内的相对差异。如果 `success_count = K` 且 reward 基本无方差，这道题对 raw GRPO 的贡献接近 regression guard，而不是能力增长信号。如果 `success_count = 0` 且 reward 无方差，这道题也不能直接告诉 gate 往哪个 expert 方向移动。

### 3.3 Self-compare reward

当 raw reward 饱和时，目标从「这次输出是否正确」改成「当前 gate 是否比 reference gate 更好」：

```text
b_i = Agg_j(R_t(y_ref_ij, metadata_i))
delta_ij = R_t(y_candidate_ij, metadata_i) - b_i
```

常用 `Agg = mean`。训练时使用：

```bash
--advantage-field reward_delta_vs_baseline
```

已有脚本：

```text
scripts/data/build_self_compare_advantage_calibration.py
scripts/train/opvec_update_gates_from_rollouts.py
```

适用场景：

1. raw reward 全对太多，直接训练没有正向区分度。
2. 想让新 gate 相对上一个 checkpoint 持续提升，而不是永远和 base 或 0.75 比。
3. 需要把 regression 也变成显式负信号。

注意：如果 reference 和 candidate 都全对，self-compare 也没有正信号；它能解决「绝对 reward 饱和但存在退化/提升差异」的问题，不能凭空从完全同分数据中产生方向。

### 3.4 Conditional distill

对于 `all_fail_zero`，raw reward 和 self-compare 都通常无效。处理方式是同 prompt 让对应 expert 重新 rollout：

```text
base/task-vector 当前失败
expert 同 prompt 成功
```

这类样本适合构造 conditional distill 或 best-response loss。最近已经验证同 prompt expert recovery 可行：

```text
Memory expert: /mnt/cache/wuruixiao/models/RL-MemoryAgent-7B
same prompt rewards: [0.0, 1.0, 0.0, 0.0]

Code expert: /mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B
same prompt rewards: [1.0, 1.0, 0.0, 0.0]
```

现有 `high_info_v1` 的 distill prompt 与固定训练 prompts 交集为 0，说明之前 Stage B 的 distill 是离线分布，不是同 prompt 修复。后续应优先做 same-prompt conditional distill。

---

## 4. Question Bank 分桶

每个 prompt 根据 baseline `K` 次 rollout 的 reward/success 分布进入一个 bucket。

| Bucket | 条件 | 训练用途 |
|---|---|---|
| `all_correct` | `success_count == K` | 不进 raw GRPO 主训；做 guard/retention |
| `high_not_all` | `0.75 <= success_rate < 1.0` | 少量保留，检测易退化题 |
| `mid_frontier` | `0.25 <= success_rate < 0.75` | raw GRPO 主力 |
| `low_but_solvable` | `0 < success_rate < 0.25` | 最重点；偏低但有成功轨迹 |
| `all_fail_partial` | `success_count == 0 and std_reward >= 0.05` | 可用于 partial reward 梯度 |
| `all_fail_zero` | `success_count == 0 and std_reward < 0.05` | 少量保留，做 expert recovery |
| `regression_guard` | baseline all_correct/high_not_all 的 heldout 子集 | 训练后硬检查，不主训 |

推荐每个 task 内的采样比例：

```text
low_but_solvable: 35-40%
mid_frontier:     30-35%
all_fail_partial: 10-15%
high_not_all:     10%
all_correct:      5-10%，guard only
all_fail_zero:    <=5%，只用于 expert recovery 可行性
```

这个比例的科研含义：

1. `low_but_solvable` 提供「模型已经偶尔能做对」的正向可达性。
2. `mid_frontier` 提供最大组内方差，是 raw GRPO 最稳定的来源。
3. `all_fail_partial` 保留格式、局部步骤、partial credit 的梯度。
4. `all_correct` 不负责推动能力增长，负责防止 gate 学偏后遗忘。
5. `all_fail_zero` 不直接训练 raw reward，先判断 expert 是否能恢复。

---

## 5. 数据结构

所有实验数据放到 shared storage，不写进代码库：

```text
/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/
  source_reward_raw_tool500_code500_seed20260511.jsonl
  hotpotqa_train_memory500_chunk5000_seed20260511.jsonl
  source_reward_hybrid_tool_code_hotpotqa_memory_traj_seed20260511.jsonl
  run_manifest.json
  commands.sh
  rollouts/
    full/
      tool_ta_avgvec_c033333_n8_offset000_len100.jsonl
      memory_ta_avgvec_c033333_n8_offset000_len100.jsonl
      code_ta_avgvec_c033333_n8_offset000_len100.jsonl
      ...
  diagnostics/
    tool500_question_bank.summary.json
    memory500_question_bank.summary.json
    code500_question_bank.summary.json
  question_bank.jsonl
  question_bank.summary.json
  calibration/
    calib100_seed20260511.prompts.jsonl
    calib100_seed20260511.guard.jsonl
    calib100_seed20260511.summary.json

/tmp/shared-storage/OnPolicy/runs/question_bank/ta_avgvec_c033333_hotpotqa_v1/
  raw_grpo/
  self_compare/
  expert_recovery/
```

`run_manifest.json` 必须记录：

```text
git_commit
worktree_path
config_path
mode_manifest
baseline_policy_model
source_manifest
rollout_seed
samples_per_prompt
temperature
top_p
max_new_tokens
max_prompt_tokens
max_model_len
reward_router_version
behavior_span_reward_weight
created_at
```

question bank 每行建议字段：

```text
question_id
prompt_id
task
source
source_path
source_row
prompt_hash
baseline_model
baseline_rollout_path
samples_per_prompt
temperature
top_p
seed
reward_list
success_list
mean_reward
max_reward
min_reward
std_reward
success_count
success_rate
bucket
bucket_reason
official_reward_adapter
selected_from_manifest
prompt_record
trace
```

`prompt_record` 用于重新导出训练 manifest；`trace` 用于审查数据从原始源到 rollout 到 bucket 的完整链路。

---

## 6. 可复现命令

### 6.1 环境

```bash
cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo

export PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
export ROOT=/tmp/shared-storage/OnPolicy
export ONPOLICY_STORAGE_ROOT=$ROOT
export MODE=$ROOT/modes/opvec4/mode_manifest.json
export QB=$ROOT/data/question_bank/ta_avgvec_c033333_hotpotqa_v1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p $QB/rollouts $QB/calibration $ROOT/runs/question_bank/ta_avgvec_c033333_hotpotqa_v1
```

当前已经整理成一键脚本，推荐优先用它复现，手工命令只作为审查细节：

```bash
cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo

# prepare: 生成 Tool raw + Code raw + HotpotQA train Memory trajectory 的 hybrid seed manifest
PHASE=prepare bash skill/command/run_question_bank_ta_avgvec_c033333.sh

# build-model: 构建 theta_base + 1/3 * sum(expert_delta) 的 TA checkpoint
PHASE=build-model bash skill/command/run_question_bank_ta_avgvec_c033333.sh

# rollout: 用静态 TA checkpoint 做 vLLM n=8 rollout，并用官方 reward 打分
CUDA_VISIBLE_DEVICES=0 PHASE=rollout bash skill/command/run_question_bank_ta_avgvec_c033333.sh

# bank: 聚合 prompt 级 reward/success 分布和 bucket
PHASE=bank bash skill/command/run_question_bank_ta_avgvec_c033333.sh

# sample: 从 question bank 采样 calib100 + guard
PHASE=sample bash skill/command/run_question_bank_ta_avgvec_c033333.sh
```

正式 1500 题 rollout 推荐用分片脚本，避免单进程长时间占一张卡：

```bash
cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo

# 默认 GPU_LIST=0,1,2,3,4；每个 task 500 题，shard size=100，n=8
PHASE=rollout-all bash skill/command/run_question_bank_full_sharded.sh

# 如果想分任务跑：
PHASE=rollout-tool bash skill/command/run_question_bank_full_sharded.sh
PHASE=rollout-memory bash skill/command/run_question_bank_full_sharded.sh
PHASE=rollout-code bash skill/command/run_question_bank_full_sharded.sh

# 合并所有 full shards 并采样 calib100
PHASE=post bash skill/command/run_question_bank_full_sharded.sh
```

复现注意：

```text
run_question_bank_full_sharded.sh 现在按 shard 行数判断是否完整；崩溃后留下半截 JSONL 不会被误判为完成。
opvec_collect_vllm_rollouts.py 会把 --vllm-batch-size 传给 vLLM max_num_seqs，避免默认 1024 dummy requests 在 warmup 阶段 OOM。
Memory full rollout 曾因 GPU4 warmup OOM 重试，最终 offset400 使用修正后的 max_num_seqs=32 成功完成；最终数据文件是完整 100 行。
```

默认 token 上限：

```text
Tool max_new_tokens: 1024
Memory update max_new_tokens: 1024
Memory final max_new_tokens: 1024
Code max_new_tokens: 1024
max_prompt_tokens: 8192
max_model_len: 12288
```

如果要完全复现早期手工命令里的更宽生成，把 `TOOL_MAX_NEW_TOKENS/CODE_MAX_NEW_TOKENS/MEMORY_MAX_NEW_TOKENS` 设为 `2048`。

### 6.2 构造候选 manifest

当前 v1 不是 final-only manifest，而是三任务 raw/hybrid manifest：

```text
Tool:   ToolRL raw train，500 条
Memory: HotpotQA train parquet，500 条问题，按 5000 token 切成 MemAgent recurrent chunks
Code:   CodeContests raw train，500 条
Total:  1500 prompts
```

这样做是因为 Memory final-only reward 不能复现 MemAgent 的训练信号；`routed_1/Memory.json` 是 ExpertMerging correct-sample 派生池，适合作为 sanity/guard 或 expert-recovery 参考，不适合作为 question-bank 主源。主源应使用本地 HotpotQA raw train parquet。

如果手工复现，可以分三步：

```bash
$PY scripts/data/build_source_reward_seed_manifest.py \
  --output $QB/source_reward_raw_tool500_code500_seed20260511.jsonl \
  --tool-limit 500 \
  --memory-final-limit 0 \
  --code-limit 500 \
  --seed 20260511

$PY scripts/data/build_hotpotqa_memory_seed_manifest.py \
  --input /mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/hotpotqa/hotpotqa_train_32k.parquet \
  --output $QB/hotpotqa_train_memory500_chunk5000_seed20260511.jsonl \
  --limit 500 \
  --seed 20260511 \
  --chunk-tokenizer /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct \
  --chunk-size-tokens 5000 \
  --max-chunks 0

$PY scripts/data/merge_seed_manifests.py \
  --input $QB/source_reward_raw_tool500_code500_seed20260511.jsonl::tool,code \
  --input $QB/hotpotqa_train_memory500_chunk5000_seed20260511.jsonl::memory \
  --output $QB/source_reward_hybrid_tool_code_hotpotqa_memory_traj_seed20260511.jsonl
```

### 6.3 baseline vLLM rollout

```bash
export POLICY=/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_avgvec_c033333/model

CUDA_VISIBLE_DEVICES=0 \
$PY scripts/train/opvec_collect_vllm_rollouts.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest $MODE \
  --policy-model $POLICY \
  --policy-id ta-avgvec-c033333 \
  --no-gate-values \
  --seed-manifest $QB/source_reward_hybrid_tool_code_hotpotqa_memory_traj_seed20260511.jsonl \
  --output $QB/rollouts/baseline_ta_avgvec_c033333_s20260511_n8.jsonl \
  --run-id qb-v1-baseline-ta-avgvec-c033333-n8 \
  --num-prompts 1500 \
  --use-manifest-order \
  --samples-per-prompt 8 \
  --max-new-tokens 2048 \
  --max-prompt-tokens 8192 \
  --max-model-len 12288 \
  --vllm-batch-size 32 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.82 \
  --dtype bfloat16 \
  --temperature 0.7 \
  --top-p 0.95 \
  --seed 20260511 \
  --stream-output \
  --progress-every 4 \
  --behavior-span-reward-weight 0.0
```

Memory 的 `max-new-tokens=512` 通常偏短，尤其是 memory update/trajectory。question bank 阶段建议至少 2048；如果遇到截断明显，再提高到 4096，并对应增大 `max-model-len`。

### 6.4 构建 question bank

数据层脚本已落地：

```bash
$PY scripts/data/build_question_bank_from_rollouts.py \
  --rollouts $QB/rollouts/baseline_ta_avgvec_c033333_s20260511_n8.jsonl \
  --seed-manifest $QB/source_reward_hybrid_tool_code_hotpotqa_memory_traj_seed20260511.jsonl \
  --output $QB/question_bank.jsonl \
  --summary $QB/question_bank.summary.json
```

这个脚本只做统计和索引，不重新 reward，不修改 rollout 原文。

### 6.5 从 question bank 采样 calibration

采样脚本已落地：

```bash
$PY scripts/data/sample_question_bank.py \
  --question-bank $QB/question_bank.jsonl \
  --output-prefix $QB/calibration/calib100_seed20260511 \
  --quota tool=34 \
  --quota memory=33 \
  --quota code=33 \
  --guard-per-task 3 \
  --seed 20260511
```

输出：

```text
$QB/calibration/calib100_seed20260511.prompts.jsonl
$QB/calibration/calib100_seed20260511.guard.jsonl
$QB/calibration/calib100_seed20260511.summary.json
```

### 6.6 Self-compare 构造

candidate rollout 和 reference rollout 必须覆盖同一批 `prompt_id`：

```bash
$PY scripts/data/build_self_compare_advantage_calibration.py \
  --candidate-rollouts $CANDIDATE_ROLLOUT \
  --baseline-rollouts $REFERENCE_ROLLOUT \
  --output $ROOT/runs/question_bank/v1/self_compare/self_compare_delta.jsonl \
  --quota tool=16 \
  --quota memory=16 \
  --quota code=16 \
  --baseline-agg mean \
  --min-abs-delta 0.02 \
  --min-delta-std 0.02 \
  --no-require-cross-baseline
```

训练时：

```bash
$PY scripts/train/opvec_update_gates_from_rollouts.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest $MODE \
  --rollouts $ROOT/runs/question_bank/v1/self_compare/self_compare_delta.jsonl \
  --output $ROOT/runs/question_bank/v1/self_compare/gate_updates.jsonl \
  --advantage-field reward_delta_vs_baseline \
  --task-normalize-advantages \
  --length-normalize-policy-logprob \
  --ppo-loss-weight 1.0 \
  --prior-loss-weight 0.02 \
  --lr 0.001 \
  --max-coefficient-delta-from-init 0.02
```

---

## 7. 当前实验给出的约束

### 7.1 high_info_v1 的问题

`high_info_v1` 已经比最早的 seed manifest 好，但仍然存在全对比例高的问题。最近一轮统计中，i2 rollout 的 all-success 大致是：

```text
Tool:   28 / 30
Memory: 32 / 35
Code:   14 / 25
```

这说明 Tool 和 Memory 在当前 calibration 里严重饱和。继续扩大同源 routed correct 数据，不一定会提供新梯度。

### 7.2 Self-compare 初步结果

i2 vs i1 的 self-compare 可用行数：

```text
Code:   14
Memory: 2
Tool:   3
total:  19
```

balanced 2/2/2 probe 能跑通，且 guard 通过，但信号太少、Code 偏重。这证明机制可用，不能证明数据足够。

### 7.3 Conditional distill 诊断

已有 high_info distill prompt 与固定 prompts 的交集为 0，说明之前 distill 不是「当前失败题的同 prompt 修复」。后续应改为：

```text
当前 gate rollout 找到 all_fail/no-variance prompt
同 prompt 调 expert rollout
expert reward > current reward
写入 conditional distill pool
```

这样 distill 才能直接补 raw GRPO 没有方向的题。

### 7.4 HotpotQA pilot 结果

用 `ta_avgvec_c033333_hotpotqa_v1` 跑了一个小规模 pilot，目的只是验证数据源、reward、分桶和采样，不作为最终 calibration。

输入：

```text
Tool:   30 prompts, n=4
Memory: 30 prompts, n=2，来自 HotpotQA train parquet 的三个 offset 分片
Code:   30 prompts, n=4
```

输出：

```text
/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/pilots/pilot90_question_bank.jsonl
/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/pilots/pilot90_question_bank.summary.json
```

分桶结果：

```text
Overall:
  all_correct:      29
  all_fail_zero:    29
  all_fail_partial: 12
  high_not_all:      6
  mid_frontier:     14

Tool:
  all_correct: 3
  all_fail_partial: 6
  all_fail_zero: 8
  high_not_all: 4
  mid_frontier: 9

Memory:
  all_correct: 22
  all_fail_zero: 7
  mid_frontier: 1

Code:
  all_correct: 4
  all_fail_partial: 6
  all_fail_zero: 14
  high_not_all: 2
  mid_frontier: 4
```

这个结果说明：

1. HotpotQA train raw 比 `routed_1` 更合理，但 Memory 在 TA c=1/3 下仍然明显偏饱和。
2. Tool 和 Code 有足够 non-saturated 信号；Code reward 已确认使用 `CodeContests_train` 的 8 条 source tests。
3. Memory 要靠更大 n、更大覆盖面和后续 self-compare/expert-recovery 才能凑出足够 calibration 信号。
4. pilot 采样 `calib30` 时 Memory 只拿到 8/10，缺的是 `low_but_solvable/mid_frontier/high_not_all`，这是数据分布信号，不是采样脚本错误。

### 7.5 HotpotQA v1 full question bank

本轮正式数据池已经完成，reference baseline 是：

```text
/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic/ta_avgvec_c033333/model
theta = theta_base + 1/3 * (delta_tool + delta_memory + delta_code)
```

输入：

```text
Tool:   ToolRL raw train 500 prompts, n=8
Memory: HotpotQA train raw 500 prompts, n=8，5000-token recurrent chunks
Code:   CodeContests train 500 prompts, n=8
Total:  1500 prompts
```

输出：

```text
/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/question_bank.jsonl
/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/question_bank.summary.json
/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.prompts.jsonl
/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.guard.jsonl
```

行数审计：

```text
question_bank.jsonl:                    1500
calib100_seed20260511.prompts.jsonl:     100
calib100_seed20260511.guard.jsonl:         9
```

overall bucket：

```text
all_correct:       326
high_not_all:      191
mid_frontier:      250
low_but_solvable:   91
all_fail_partial:  251
all_fail_zero:     391
```

task bucket：

```text
Tool:
  all_correct: 52
  high_not_all: 79
  mid_frontier: 106
  low_but_solvable: 39
  all_fail_partial: 109
  all_fail_zero: 115
  success_rate_avg: 0.333

Memory:
  all_correct: 222
  high_not_all: 68
  mid_frontier: 61
  low_but_solvable: 16
  all_fail_zero: 133
  success_rate_avg: 0.617

Code:
  all_correct: 52
  high_not_all: 44
  mid_frontier: 83
  low_but_solvable: 36
  all_fail_partial: 142
  all_fail_zero: 143
  success_rate_avg: 0.261
```

`calib100` 采样结果：

```text
Tool 34:
  low_but_solvable 14, mid_frontier 12, all_fail_partial 4, high_not_all 3, all_fail_zero 1

Memory 33:
  low_but_solvable 13, mid_frontier 16, high_not_all 3, all_fail_zero 1
  deficit: all_fail_partial wanted 4 got 0，因为 Memory reward 是最终 exact/boxed，当前没有 partial bucket

Code 33:
  low_but_solvable 13, mid_frontier 12, all_fail_partial 4, high_not_all 3, all_fail_zero 1

Guard:
  Tool 3, Memory 3, Code 3，均来自 high_not_all/all_correct 候选
```

判断：这版数据池已经比 `routed_1` 高信息密度，Tool/Code 的 non-saturated 信号充足；Memory 仍偏饱和，但 full 500 后已有 77 个 `low/mid` 可用样本，足够先跑 global raw GRPO。Memory 的 133 个 `all_fail_zero` 不应直接大量进 raw GRPO，后续应优先做 same-prompt memory expert recovery 或 self-compare。

---

## 8. 训练阶段建议

### 8.1 初始阶段

先不要直接释放 588 个 gate。推荐顺序：

1. global gate：验证 question bank 筛选是否能稳定推动三个 expert 的有效系数。
2. task residual gate：global + tool/memory/code residual，检查三任务是否能分开。
3. layer-band gate：每个 task vector 按 block band 释放少量参数，观察是否有层级结构。
4. 588 gate：只有在前面阶段出现 heldout Pareto 改善后再做。

理由：如果 calibration data 本身信号不足，588 gate 会先学到噪声和 prompt id 偏好，而不是泛化的 task-vector 组合。

### 8.2 100 条 calibration 的推荐组成

```text
Tool:   34
Memory: 33
Code:   33
```

每个 task 内部：

```text
low_but_solvable: 12-14
mid_frontier:     10-12
all_fail_partial: 4-5
high_not_all:     3-4
all_correct:      2-3 guard only
all_fail_zero:    0-2 expert recovery only
```

训练主集不要混太多全对题。全对题应该进入 guard 或 self-compare negative set，用于阻止退化，不负责提供正向梯度。

### 8.3 何时停止

一轮不是按固定 iter 数停止，而是按信号停止：

```text
raw GRPO:
  calibration mean reward 不再提高
  heldout guard 不下降
  gate delta 低于阈值

self-compare:
  mean(delta_reward) 接近 0
  positive-delta prompt 数不再增加
  regression prompt 数不增加
```

如果 calibration 提高但 heldout 下降，说明数据选择过拟合，优先回到 question bank 重新分桶采样，而不是继续调学习率。

---

## 9. 任务清单

### P0 已完成研究

- [x] 找到本机 Tool/Memory/Code 原始数据和派生池地址。
- [x] 确认当前 high_info_v1 主要来自 `datasets_2/correct_samples/routed_1`，不是官方 raw 全量。
- [x] 找到无调参 TA average-vector c=1/3 方案、标准 TA c=0.50 历史 baseline、TA sweep checkpoints、OP-VEC 0.75 baked checkpoint。
- [x] 确认 `RewardRouter + --behavior-span-reward-weight 0.0` 是当前官方 reward rollout 路径。
- [x] 确认 `build_self_compare_advantage_calibration.py` 和 `--advantage-field reward_delta_vs_baseline` 已经支持 self-compare 训练。
- [x] 确认缺口是 question bank 聚合和采样脚本。

### P0 已完成执行

- [x] 新增 `scripts/data/build_question_bank_from_rollouts.py`。
- [x] 新增 `scripts/data/sample_question_bank.py`。
- [x] 生成 `$QB/run_manifest.json` 和 `$QB/commands.sh`，保证筛选过程可复现。
- [x] 生成 Tool raw 500 + Code raw 500 + HotpotQA train Memory trajectory 500 的 hybrid seed manifest。
- [x] 构建 TA average-vector c=1/3 baseline checkpoint。
- [x] 用 HotpotQA hybrid manifest 对 TA average-vector c=1/3 baseline 做 n=8 rollout。
- [x] 聚合 `question_bank.jsonl` 和 `question_bank.summary.json`。
- [x] 采样 `calib100_seed20260511`，并输出 bucket/task 统计。

### P1 训练和诊断

- [ ] 用 `calib100` 跑 global raw GRPO。
- [ ] 对同一批 prompt 做 reference/candidate rollout，构造 self-compare delta。
- [ ] 跑 self-compare update，比较是否比 raw reward 更能避免全对饱和。
- [ ] 对 `all_fail_zero` 做 same-prompt expert recovery。
- [ ] 把 expert recovery 成功样本写入 conditional distill pool。
- [ ] 做 Tool BFCL、Memory HotpotQA、Code CURE heldout eval。

### P2 进入细粒度 gate 的条件

- [ ] global 或 task-residual 阶段在 heldout 上至少一个 task 提升，其他 task 不显著下降。
- [ ] question bank bucket 中三类 task 都有足够 non-saturated prompt。
- [ ] self-compare 能提供跨 task 的正负 delta，而不是只有 Code 有信号。
- [ ] guard set 不被破坏。
- [ ] 满足以上条件后，才释放 layer-band 或 588 gate。

---

## 10. 数据卫生规则

1. 代码库只保留必要脚本和文档，不写 rollout、checkpoint、临时 JSONL。
2. 所有大文件进入 `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/` 或 `/tmp/shared-storage/OnPolicy/runs/question_bank/ta_avgvec_c033333_hotpotqa_v1/`。
3. 执行命令前设置 `PYTHONDONTWRITEBYTECODE=1`，避免生成 `__pycache__`。
4. 每次筛选必须有 `run_manifest.json`，记录输入、模型、seed、reward、脚本版本。
5. 每个 calibration 文件必须能追溯到 question bank row，再追溯到 raw source row 和 baseline rollout。
6. 不覆盖旧版本。新实验用 `v2` 或带日期/seed 的目录。
7. 不把 heldout eval 数据混入 calibration 主训。

---

## 11. 最小可执行下一步

question bank 已经建好，下一步是验证这 100 条 calibration 是否真的能推动 gate，而不是继续调数据脚本：

```text
1. 用 calib100_seed20260511.prompts.jsonl 跑 global raw GRPO
2. 用 calib100_seed20260511.guard.jsonl 做同 checkpoint regression check
3. 观察 gate 是否按 task 出现可解释增长，而不是只推单一 expert
4. 对 all_fail_zero 抽样做 same-prompt expert recovery
5. 用同一批 prompt 构造 self-compare delta，判断 raw reward 饱和题能否重新提供方向
6. 如果 global 有 heldout Pareto 改善，再进入 task residual；否则回到 question bank 调采样比例
```

判断本轮数据是否值得训练的硬标准：

```text
每个 task 至少有 20 个 non-saturated prompt
calib100 中 raw GRPO 主训样本不少于 70%
all_correct 只做 guard，不超过 10%
all_fail_zero 不超过 5%，除非进入 conditional distill
self-compare delta 每个 task 至少有 8 个非零 prompt
```

当前 v1 已满足“每个 task 至少 20 个 non-saturated prompt”和“calib100 主训样本占多数”。风险点是 Memory partial bucket 为 0、all_fail_zero 总量高；因此第一轮 raw GRPO 可以跑，但必须配 guard 和后续 expert recovery/self-compare 诊断，不要只看 calibration reward。
