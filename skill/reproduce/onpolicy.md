---
title: OnPolicyMerge 复现与科研推进笔记
project: OnPolicyMerge
repo: /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge
date: 2026-05-09
tags:
  - reproduce
  - model-merging
  - task-vector
  - on-policy
  - opvec
---

# OnPolicyMerge 复现与科研推进笔记

## TL;DR

OnPolicyMerge 当前不是传统 ExpertMerging 的无监督合并复现，而是在 Qwen2.5-7B base 上，用 Tool / Memory / Code 三个专家的 task vector 做可学习系数合并：

```text
W = W_base
  + alpha_tool   * (W_tool   - W_base)
  + alpha_memory * (W_memory - W_base)
  + alpha_code   * (W_code   - W_base)
```

当前正式 OP-VEC-4 选择 196 个可合并权重：

```text
28 层 * 7 个线性层(q/k/v/o/gate/up/down) = 196
196 个权重 * 3 个专家 = 588 个参数级 task-vector 系数
```

核心结论：

- 工程链路已跑通：构建专家差值、安装 gated linear、生成 rollouts、用 verifier reward 更新系数、烘焙成普通 HF checkpoint、子集评测。
- 当前可复现模型是从 base 出发的 `global-parameter` 系数模型：`/tmp/shared-storage/OnPolicy/checkpoints/opvec-fromzero-base-expert-n40x3-balanced-brpair-lr4e2-steps2`。
- 这个模型还没有超过 stage2：Tool 平均 reward 略高，Memory 持平，Code 选择略低。
- 5 月 9 日已经把数据推进到 `frontier_v2`：Tool kept frontier 56、Memory 26、Code 40，并标准化成 capability / split_group 数据与 5-fold group-disjoint split。这是下一轮科研版复现的主线。

## 当前进展判断

### 当前最佳可用基线

报告里仍把上一轮 stage2 作为更强 baseline：

```text
/tmp/shared-storage/OnPolicy/checkpoints/opvec-global-residual-stage2-code-tool-frontiers-lr3e3-steps2-ret12
```

原因是 2026-05-08 的 source-reward stage3 没超过它，2026-05-09 的 from-zero 模型也没超过它。

### 当前可复现模型

```text
/tmp/shared-storage/OnPolicy/checkpoints/opvec-fromzero-base-expert-n40x3-balanced-brpair-lr4e2-steps2
```

对应 gate：

```text
/tmp/shared-storage/OnPolicy/runs/opvec4/gate_update_fromzero_base_expert_n40x3_balanced_br_pair_lr4e2_steps2.log.gates.json
```

系数幅值：

| 方向 | global 系数 | 参数级平均 | 参数级范围 | 解释 |
| --- | ---: | ---: | ---: | --- |
| Tool | 0.542 | 0.629 | 0.392-0.692 | 接近 0.75，但没到 |
| Memory | 0.909 | 1.001 | 0.759-1.059 | 明显强表达 |
| Code | 0.381 | 0.511 | 0.231-0.531 | 偏弱 |

固定子集评测：

| 模型 | Tool 成功率 | Tool 平均 reward | Memory 成功率 | Code 选择成功率 | Code 相对 BoN |
| --- | ---: | ---: | ---: | ---: | ---: |
| from-zero 当前模型 | 0.850 | 0.946 | 0.500 | 0.800 | +0.100 |
| stage2 | 0.850 | 0.933 | 0.500 | 0.850 | +0.150 |

判断：from-zero 证明“从 0 学 task-vector 系数”可行，但还不是最终胜出模型。

### 5 月 9 日数据层进展

旧 n40x3 单样本 proposal 的问题是 Code 只有 1 条可训练 frontier：

| 任务 | 输入题数 | 可训练题数 |
| --- | ---: | ---: |
| Tool | 40 | 28 |
| Memory | 40 | 13 |
| Code | 40 | 1 |

后续 `frontier_v2` 改成多 producer / 多采样，已有产物：

| 数据 | 输入 | kept frontier | capability 分布 |
| --- | ---: | ---: | --- |
| Tool v2 | 80 | 56 | `tool_math=27`, `tool_multi_call=28`, `tool_nested_argument=1` |
| Memory v2 | 80 | 26 | `memory_final_retrieval=26` |
| Code v2 | 80 | 40 | `code_math=28`, `code_string=7`, `code_graph=4`, `code_algorithmic=1` |

合并后的 standardized frontier：

```text
/tmp/shared-storage/OnPolicy/runs/opvec4/frontier_v2_tmc_balanced_all_sources.summary.json
```

总量：

```text
Tool:   56
Memory: 26
Code:   40
Total: 122
```

已按 `split_group` 做 5-fold group-disjoint split，例如 fold0：

```text
train:   93 rows = code 30 / memory 20 / tool 43
heldout: 29 rows = code 10 / memory 6 / tool 13
```

注意：standardized JSONL 目前主要用于数据审计、分桶、split 设计；`opvec_update_gates_from_rollouts.py` 仍直接吃原始 rollout/frontier 行，因为训练需要 `prompt/rendered_prompt + samples.text/reward`。

## 代码结构地图

### 合并与 gate

| 文件 | 作用 |
| --- | --- |
| `opvec/modeling/gated_linear.py` | 用 `GatedLinear` 包装 `nn.Linear`，forward 时动态计算 `W0 + sum(alpha_i * delta_i)` |
| `opvec/modeling/apply_gates.py` | 根据 mode manifest 把目标 Linear 替换成 `GatedLinear` |
| `opvec/modeling/gate_parameters.py` | 定义 gate manager：global、layer-band、parameter、global-parameter |
| `opvec/modeling/bake.py` | 把 gate 系数烘焙进 safetensors，输出普通 HF checkpoint |
| `scripts/modes/build_opvec4_modes.py` | 构建 OP-VEC-4 专家差值 manifest |
| `scripts/modes/build_zero_gate_checkpoint.py` | 显式生成全 0 gate，避免默认 `common=0.5` |

### 数据与 rollout

| 文件 | 作用 |
| --- | --- |
| `scripts/data/build_source_reward_seed_manifest.py` | 从 ToolRL / Memory / CodeContests 构造 source-reward calibration manifest |
| `scripts/train/opvec_collect_hf_rollouts.py` | 给指定 policy 采样，调用 `RewardRouter` 打分，输出 rollout JSONL |
| `scripts/data/merge_rollout_samples_by_prompt.py` | 按 `prompt_id` 合并多 producer 的 samples |
| `scripts/data/filter_source_reward_frontiers.py` | 筛出同题内有明显好坏差异的 frontier |
| `scripts/data/build_standardized_frontier_dataset.py` | 转成 capability-tagged frontier example |
| `scripts/data/split_standardized_frontier_by_group.py` | 按 `split_group` 做 group-disjoint folds |

### 训练与评测

| 文件 | 作用 |
| --- | --- |
| `scripts/train/opvec_update_gates_from_rollouts.py` | 主训练入口；冻结模型权重，只更新 gate manager |
| `scripts/train/opvec_train_task_vector_global_residual_grpo.py` | global-parameter 训练薄封装 |
| `scripts/eval/opvec_bake_checkpoint.py` | gate -> HF checkpoint |
| `scripts/eval/opvec_subset_trend_monitor.py` | 固定小子集评测 Tool / Memory / Code |
| `scripts/eval/bfcl_eval_hf_manifest.py` | Tool/Memory 的 HF 生成评测 |
| `scripts/eval/cure_eval_candidate_selector_logprob.py` | Code 候选选择评测 |

## 方法机制

### 参数化方式

早期有四种 gate：

| parameterization | 可学习对象 | 备注 |
| --- | --- | --- |
| `global` | `common + expert_residual` | 只有 4 个原始参数，表达力弱 |
| `layer-band` | early/mid/late 分段 gate | 约 12 个参数 |
| `parameter` | 每个权重每个专家一个系数 | 588 个参数，信号分散 |
| `global-parameter` | 专家全局强度 + 参数残差 | 当前主线 |

当前主线：

```text
coef[param, expert] = global[expert] + residual[param, expert]
```

默认约束：

```text
global_coefficient: [0.00, 1.20]
parameter_residual: [-0.15, 0.15]
coefficient: [-0.50, 1.50]
```

这比纯 588 参数稳定，因为每个 expert 有一个可整体移动的主旋钮；也比 layer-band 更细，因为每个真实权重仍有 residual。

### 训练目标

训练时：

1. 加载 base model。
2. 加载三个专家相对 base 的 delta。
3. 用 gate manager 动态组合当前合并模型。
4. 对 calibration prompt 里的候选答案计算当前 logprob。
5. 同一道题里，提高高 reward 答案概率，降低低 reward 答案概率。
6. 只更新 gate manager；base 和 delta 都 `requires_grad=False`。

from-zero 训练使用的是：

```text
--ppo-loss-weight 0.0
--best-response-loss-weight 1.0
--pairwise-loss-weight 1.0
--length-normalize-logprob
```

这意味着不需要 `old_logprob`，可以直接使用 base/expert proposal 数据。

### Reward 信号

| 域 | 主 reward | 当前问题 |
| --- | --- | --- |
| Tool | 结构化工具调用检查：函数名、参数名、参数值、调用数量/顺序 | 信号最干净，容易主导训练 |
| Memory | final answer 与参考答案/关键片段匹配 | 还偏“最终问答”，缺少写入/保持过程 reward |
| Code | 可提取 Python、语法、stdin/stdout、公开样例或测试通过率 | n40x3 太稀疏，v2 已改善到 40 kept |

重要原则：训练 reward 来自 calibration verifier，不是最终测试集答案；heldout/final eval 不回流训练。

## 精确复现：当前 from-zero n40x3 模型

### 环境变量

```bash
cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge
conda activate BFCL

export PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
export ROOT=/tmp/shared-storage/OnPolicy
export RUN=$ROOT/runs/opvec4
export MODE_FULL=$ROOT/modes/opvec4-full-bf16-real/mode_manifest.json
export MODE_SMOKE=$ROOT/modes/opvec4-smoke-real/mode_manifest.json
export SEED_MANIFEST=$ROOT/data/source_reward/source_reward_t80_m80_c80_seed20260508.jsonl

export BASE=/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct
export TOOL=/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold
export MEMORY=/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B
export CODE=/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B
```

关键坑：

- 主训练必须用 `$MODE_FULL`，它含真实 delta 文件。
- `$ROOT/modes/opvec4/mode_manifest.json` 是 dry-run，不能训练或烘焙。
- proposal 采样时 `--disable-gates`，`--mode-manifest` 只是参数占位，可用 `$MODE_SMOKE`。
- 不传 zero gate 就会继承 `configs/opvec4.yaml` 里的 `common=0.5`，那不是从 base 出发。

### 0. 构造 source-reward manifest

如果 `$SEED_MANIFEST` 已存在，可以跳过。需要重建时：

```bash
mkdir -p $ROOT/data/source_reward

$PY scripts/data/build_source_reward_seed_manifest.py \
  --output $ROOT/data/source_reward/source_reward_t80_m80_c80_seed20260508.jsonl \
  --tool-limit 80 \
  --memory-final-limit 80 \
  --code-limit 80 \
  --seed 20260508
```

### 1. 生成 zero gate

```bash
$PY scripts/modes/build_zero_gate_checkpoint.py \
  --mode-manifest $MODE_FULL \
  --output $RUN/zero_init_global_parameter_full_bf16.gates.json \
  --gate-parameterization global-parameter
```

预期：

```text
num_mergeable_params = 196
num_gate_values = 595
learnable effective coefficients = 196 * 3 = 588
```

### 2. 采样 base/expert proposals

每个任务取 40 道题，base 和对应 expert 各采一次。六个命令可并行跑。

完整参数表：

| 输出后缀 | policy | tasks | memory-kind | max-new-tokens | GPU |
| --- | --- | --- | --- | ---: | ---: |
| `base_code` | `$BASE` | code | 空 | 1024 | 2 |
| `expert_code` | `$CODE` | code | 空 | 1024 | 3 |
| `base_tool` | `$BASE` | tool | 空 | 512 | 4 |
| `expert_tool` | `$TOOL` | tool | 空 | 512 | 5 |
| `base_memory` | `$BASE` | memory | final_answer | 128 | 6 |
| `expert_memory` | `$MEMORY` | memory | final_answer | 128 | 7 |

更不容易漏项的写法：

```bash
run_n40_proposal () {
  local gpu="$1"
  local name="$2"
  local model="$3"
  local task="$4"
  local max_new="$5"
  local extra_memory_kind="${6:-}"

  local memory_args=()
  if [ -n "$extra_memory_kind" ]; then
    memory_args=(--memory-kind "$extra_memory_kind")
  fi

  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train/opvec_collect_hf_rollouts.py \
    --config configs/opvec4.yaml \
    --seed-manifest $SEED_MANIFEST \
    --output $RUN/fromzero_proposal_${name}_n40_s1_seed20260508.jsonl \
    --run-id fromzero-${name}-n40-s1 \
    --policy-model "$model" \
    --disable-gates \
    --tasks "$task" \
    "${memory_args[@]}" \
    --num-prompts 40 \
    --samples-per-prompt 1 \
    --max-new-tokens "$max_new" \
    --max-prompt-tokens 2048 \
    --skip-logprob \
    --temperature 0.7 \
    --top-p 0.95 \
    --behavior-span-reward-weight 0.03 \
    --seed 20260508 \
    --progress-every 10 \
    --mode-manifest $MODE_SMOKE \
    --torch-dtype bfloat16
}

run_n40_proposal 2 base_code    $BASE   code   1024
run_n40_proposal 3 expert_code  $CODE   code   1024
run_n40_proposal 4 base_tool    $BASE   tool   512
run_n40_proposal 5 expert_tool  $TOOL   tool   512
run_n40_proposal 6 base_memory  $BASE   memory 128 final_answer
run_n40_proposal 7 expert_memory $MEMORY memory 128 final_answer
```

下面是展开后的单命令示例。

Code base：

```bash
CUDA_VISIBLE_DEVICES=2 $PY scripts/train/opvec_collect_hf_rollouts.py \
  --config configs/opvec4.yaml \
  --seed-manifest $SEED_MANIFEST \
  --output $RUN/fromzero_proposal_base_code_n40_s1_seed20260508.jsonl \
  --run-id fromzero-base-code-n40-s1 \
  --policy-model $BASE \
  --disable-gates \
  --tasks code \
  --num-prompts 40 \
  --samples-per-prompt 1 \
  --max-new-tokens 1024 \
  --max-prompt-tokens 2048 \
  --skip-logprob \
  --temperature 0.7 \
  --top-p 0.95 \
  --behavior-span-reward-weight 0.03 \
  --seed 20260508 \
  --progress-every 10 \
  --mode-manifest $MODE_SMOKE \
  --torch-dtype bfloat16
```

Code expert：把 `--policy-model $BASE` 改成 `$CODE`，输出改为：

```text
$RUN/fromzero_proposal_expert_code_n40_s1_seed20260508.jsonl
```

Tool base / expert：

```bash
# base: --policy-model $BASE
# expert: --policy-model $TOOL
CUDA_VISIBLE_DEVICES=4 $PY scripts/train/opvec_collect_hf_rollouts.py \
  --config configs/opvec4.yaml \
  --seed-manifest $SEED_MANIFEST \
  --output $RUN/fromzero_proposal_base_tool_n40_s1_seed20260508.jsonl \
  --run-id fromzero-base-tool-n40-s1 \
  --policy-model $BASE \
  --disable-gates \
  --tasks tool \
  --num-prompts 40 \
  --samples-per-prompt 1 \
  --max-new-tokens 512 \
  --max-prompt-tokens 2048 \
  --skip-logprob \
  --temperature 0.7 \
  --top-p 0.95 \
  --behavior-span-reward-weight 0.03 \
  --seed 20260508 \
  --progress-every 10 \
  --mode-manifest $MODE_SMOKE \
  --torch-dtype bfloat16
```

Memory base / expert：

```bash
# base: --policy-model $BASE
# expert: --policy-model $MEMORY
CUDA_VISIBLE_DEVICES=6 $PY scripts/train/opvec_collect_hf_rollouts.py \
  --config configs/opvec4.yaml \
  --seed-manifest $SEED_MANIFEST \
  --output $RUN/fromzero_proposal_base_memory_n40_s1_seed20260508.jsonl \
  --run-id fromzero-base-memory-n40-s1 \
  --policy-model $BASE \
  --disable-gates \
  --tasks memory \
  --memory-kind final_answer \
  --num-prompts 40 \
  --samples-per-prompt 1 \
  --max-new-tokens 128 \
  --max-prompt-tokens 2048 \
  --skip-logprob \
  --temperature 0.7 \
  --top-p 0.95 \
  --behavior-span-reward-weight 0.03 \
  --seed 20260508 \
  --progress-every 10 \
  --mode-manifest $MODE_SMOKE \
  --torch-dtype bfloat16
```

### 3. 合并与筛选 frontier

```bash
$PY scripts/data/merge_rollout_samples_by_prompt.py \
  --config configs/opvec4.yaml \
  --rollouts $RUN/fromzero_proposal_base_code_n40_s1_seed20260508.jsonl \
  --rollouts $RUN/fromzero_proposal_expert_code_n40_s1_seed20260508.jsonl \
  --rollouts $RUN/fromzero_proposal_base_tool_n40_s1_seed20260508.jsonl \
  --rollouts $RUN/fromzero_proposal_expert_tool_n40_s1_seed20260508.jsonl \
  --rollouts $RUN/fromzero_proposal_base_memory_n40_s1_seed20260508.jsonl \
  --rollouts $RUN/fromzero_proposal_expert_memory_n40_s1_seed20260508.jsonl \
  --output $RUN/fromzero_base_expert_proposals_n40x3_s1_merged.jsonl
```

```bash
$PY scripts/data/filter_source_reward_frontiers.py \
  --rollouts $RUN/fromzero_base_expert_proposals_n40x3_s1_merged.jsonl \
  --output $RUN/fromzero_base_expert_proposals_n40x3_s1_merged.source_filtered.jsonl \
  --min-reward-gap 0.20 \
  --positive-threshold 0.95 \
  --negative-threshold 0.50 \
  --audit-examples 12
```

预期 kept：

```text
Tool 28
Memory 13
Code 1
```

### 4. 从零训练

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/train/opvec_update_gates_from_rollouts.py \
  --config configs/opvec4.yaml \
  --rollouts $RUN/fromzero_base_expert_proposals_n40x3_s1_merged.source_filtered.jsonl \
  --mode-manifest $MODE_FULL \
  --output $RUN/gate_update_fromzero_base_expert_n40x3_balanced_br_pair_lr4e2_steps2.log.jsonl \
  --max-gated-modules 196 \
  --max-steps 2 \
  --max-logprob-tokens 1536 \
  --lr 0.04 \
  --weight-decay 0.0 \
  --prior-loss-weight 0.0 \
  --ppo-loss-weight 0.0 \
  --best-response-loss-weight 1.0 \
  --pairwise-loss-weight 1.0 \
  --length-normalize-logprob \
  --task-weight code=10.0 \
  --task-weight memory=2.0 \
  --task-weight tool=1.0 \
  --frontier-task-quota tool=13 \
  --frontier-task-quota memory=13 \
  --frontier-task-quota code=13 \
  --gate-parameterization global-parameter \
  --init-gate-checkpoint $RUN/zero_init_global_parameter_full_bf16.gates.json \
  --device-map auto \
  --max-memory 0=70GiB \
  --max-memory 1=70GiB \
  --max-memory 2=70GiB \
  --max-memory 3=70GiB \
  --max-memory 4=70GiB \
  --max-memory 5=70GiB \
  --max-memory cpu=120GiB \
  --torch-dtype bfloat16
```

预期 summary：

```text
updates = 54
frontier_task_counts = {"code": 1, "memory": 13, "tool": 13}
gate_grad_nonzero = true
parameter_coefficients = 588
```

### 5. 检查幅值

```bash
$PY - <<'PY'
import json, statistics as st
p = "/tmp/shared-storage/OnPolicy/runs/opvec4/gate_update_fromzero_base_expert_n40x3_balanced_br_pair_lr4e2_steps2.log.summary.json"
s = json.load(open(p))
g = s["final_gates"]
print("updates:", s["updates"])
print("frontier_task_counts:", s["frontier_task_counts"])
for ex in ["tool", "memory", "code"]:
    vals = sorted(float(v) for k, v in g.items() if k.endswith("::" + ex) and not k.startswith("__global__::"))
    n = len(vals)
    def pct(q): return vals[min(n - 1, max(0, round(q * (n - 1))))]
    print(ex, {
        "global": g[f"__global__::{ex}"],
        "mean": st.mean(vals),
        "median": st.median(vals),
        "std": st.pstdev(vals),
        "min": vals[0],
        "p25": pct(0.25),
        "p75": pct(0.75),
        "max": vals[-1],
    })
PY
```

### 6. 烘焙

```bash
$PY scripts/eval/opvec_bake_checkpoint.py \
  --config configs/opvec4.yaml \
  --mode-manifest $MODE_FULL \
  --gate-checkpoint $RUN/gate_update_fromzero_base_expert_n40x3_balanced_br_pair_lr4e2_steps2.log.gates.json \
  --output $ROOT/checkpoints/opvec-fromzero-base-expert-n40x3-balanced-brpair-lr4e2-steps2
```

预期：

```text
num_delta_entries = 588
```

### 7. 子集评测

```bash
CUDA_VISIBLE_DEVICES=2 $PY scripts/eval/opvec_subset_trend_monitor.py \
  --run \
  --run-id fromzero_balanced_lr4e2_steps2_vs_stage2_20260509_repro \
  --model fromzero_bal=$ROOT/checkpoints/opvec-fromzero-base-expert-n40x3-balanced-brpair-lr4e2-steps2 \
  --model stage2=$ROOT/checkpoints/opvec-global-residual-stage2-code-tool-frontiers-lr3e3-steps2-ret12 \
  --tool-limit-per-category 5 \
  --memory-limit 8 \
  --code-limit 20 \
  --samples-per-prompt 1 \
  --device cuda:0 \
  --torch-dtype bfloat16
```

读结果：

```bash
sed -n '1,120p' \
  $ROOT/evaluation/subset_trend/fromzero_balanced_lr4e2_steps2_vs_stage2_20260509_repro/subset_trend_summary.md
```

### 8. 测试

```bash
$PY -m unittest discover -s tests -v
```

报告里 2026-05-09 通过：

```text
Ran 113 tests
OK
```

## 推荐复现：frontier_v2 科研主线

from-zero n40x3 更像“证明链路可跑”。科研上下一轮应复现和扩展 `frontier_v2`。

### 目标

把训练 frontier 从：

```text
Tool 28 / Memory 13 / Code 1
```

提升到当前已有的：

```text
Tool 56 / Memory 26 / Code 40
```

并继续扩到论文更可信的目标：

```text
每个大域 >= 200 kept frontier
每个 capability bucket >= 30-50 kept frontier
```

### v2 数据构造原则

每道题多 producer、多采样：

```text
base:          4 samples
matched expert:4 samples
current merge: 4 samples
stage2:        4 samples, 主要给 Code 增加 frontier
greedy variants: 可作为 proposal producer
```

筛选：

```text
max_reward - min_reward >= 0.20 或 0.25
至少一个高分候选
至少一个低分候选
verifier 结果可审计
```

Code 可放宽为：

```text
max_reward >= 0.60
min_reward <= 0.20
reward_gap >= 0.30
```

因为代码完全通过样本更稀疏。

### v2 已有路径

Tool：

```text
$RUN/frontier_v2_tool_base_expert_fromzero_n80_s4_merged.source_filtered.jsonl
$RUN/frontier_v2_tool_base_expert_fromzero_n80_s4_merged.standardized.jsonl
```

Memory：

```text
$RUN/frontier_v2_memory_base_expert_fromzero_n80_s4_merged.source_filtered.jsonl
$RUN/frontier_v2_memory_base_expert_fromzero_n80_s4_merged.standardized.jsonl
```

Code：

```text
$RUN/frontier_v2_code_all_sources_merged.source_filtered.jsonl
$RUN/frontier_v2_code_all_sources_merged.standardized.jsonl
```

Merged standardized summary：

```text
$RUN/frontier_v2_tmc_balanced_all_sources.summary.json
```

5 folds：

```text
$RUN/frontier_v2_tmc_balanced_all_sources_fold0_train.jsonl
$RUN/frontier_v2_tmc_balanced_all_sources_fold0_heldout.jsonl
...
$RUN/frontier_v2_tmc_balanced_all_sources_fold4_train.jsonl
$RUN/frontier_v2_tmc_balanced_all_sources_fold4_heldout.jsonl
```

### v2 训练建议

先直接用 raw filtered rollouts 训练，不用 standardized 文件喂训练脚本：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/train/opvec_update_gates_from_rollouts.py \
  --config configs/opvec4.yaml \
  --rollouts $RUN/frontier_v2_tool_base_expert_fromzero_n80_s4_merged.source_filtered.jsonl \
  --rollouts $RUN/frontier_v2_memory_base_expert_fromzero_n80_s4_merged.source_filtered.jsonl \
  --rollouts $RUN/frontier_v2_code_all_sources_merged.source_filtered.jsonl \
  --mode-manifest $MODE_FULL \
  --output $RUN/gate_update_fromzero_frontier_v2_tmc_lrXX_stepsY.log.jsonl \
  --max-gated-modules 196 \
  --max-steps 1 \
  --max-logprob-tokens 1536 \
  --lr 0.01 \
  --weight-decay 0.0 \
  --prior-loss-weight 0.001 \
  --ppo-loss-weight 0.0 \
  --best-response-loss-weight 1.0 \
  --pairwise-loss-weight 1.0 \
  --length-normalize-logprob \
  --task-weight code=3.0 \
  --task-weight memory=1.5 \
  --task-weight tool=1.0 \
  --frontier-task-quota tool=40 \
  --frontier-task-quota memory=26 \
  --frontier-task-quota code=40 \
  --gate-parameterization global-parameter \
  --init-gate-checkpoint $RUN/zero_init_global_parameter_full_bf16.gates.json \
  --device-map auto \
  --max-memory 0=70GiB \
  --max-memory 1=70GiB \
  --max-memory 2=70GiB \
  --max-memory 3=70GiB \
  --max-memory 4=70GiB \
  --max-memory 5=70GiB \
  --max-memory cpu=120GiB \
  --torch-dtype bfloat16
```

建议从保守版开始：

| 参数 | 建议 |
| --- | --- |
| `lr` | 0.005-0.01 起步；n40x3 的 0.04 对 v2 可能过大 |
| `max_steps` | 1 起步，先看系数幅值和子集趋势 |
| `prior-loss-weight` | 0.001 或 0.005，避免小数据把系数推过头 |
| `max-coefficient-delta-from-init` | 可试 0.35-0.50，做 trust region |
| task weights | Code 不再只有 1 条，不需要 10.0；可从 3.0 开始 |

成功标准不是训练 loss 下降，而是：

```text
1. gate_grad_nonzero = true
2. Tool / Memory / Code 系数幅值合理，不单域爆掉
3. 固定子集优于 stage2 或至少 Pareto 不退
4. group-disjoint heldout frontier 上能力分桶不崩
```

## 数据标准化与泛化实验

### 为什么要 standardize

论文叙述不能只说 BFCL / HotpotQA / CodeContests。更稳的叙述是：

```text
verifier-confirmed frontier calibration set
```

每条样本表示“同一输入下多个候选行为，verifier 能确认哪些更正确、更可恢复”。这比“给三个任务写 reward”更像通用方法。

### 当前 standardized schema

`scripts/data/build_standardized_frontier_dataset.py` 输出：

```json
{
  "format": "opvec_capability_frontier_example_v1",
  "example_id": "...",
  "prompt_id": "...",
  "domain": "tool|memory|code",
  "capability": "tool_multi_call|memory_final_retrieval|code_math|...",
  "source_dataset": "...",
  "source_family": "...",
  "split_group": "...",
  "verifier": "...",
  "prompt_hash": "...",
  "reference_hash": "...",
  "keep_for_training": true,
  "frontier": {
    "max_reward": 1.0,
    "min_reward": 0.0,
    "reward_gap": 1.0,
    "num_samples": 12,
    "num_positive": 3,
    "num_negative": 5
  },
  "proposal_stats": {},
  "proposals": []
}
```

### split_group 规则

| 域 | split_group |
| --- | --- |
| Tool | `source_dataset::tools::{tool_names}` |
| Memory | `source_dataset::memory::{question_id/task_id}` |
| Code | `source_dataset::code::{task_id/question_id}` |

这样可以避免同一个工具 schema、同一个 memory question、同一个 code problem 同时进训练和 heldout。

### 推荐实验矩阵

1. **Exact Repro**：复现 n40x3 from-zero，证明链路一致。
2. **V2 Data Scaling**：用 Tool56 / Memory26 / Code40 训练，与 stage2 比。
3. **Group-Disjoint Heldout**：用 v2 fold0 train 训练，在 fold0 heldout 的同源 frontier 上做能力分桶诊断。
4. **Leave-One-Capability-Out**：训练时去掉 `tool_multi_call` 或 `code_graph`，heldout 看迁移。
5. **Cross-Benchmark Eval**：最终仍跑 BFCL / Memory / Code 官方或更大 harness，不用 frontier heldout 替代最终分数。

## 科研 idea

### Idea 1: Verifier-confirmed Frontier Merging

把方法表述成：

```text
不是用验证集扫合并系数，而是用 verifier-confirmed frontier 学 task-vector coefficient。
```

关键 novelty：

- 合并权重是可学习参数，但训练数据不是测试标签。
- 每个训练样本是同一 prompt 下的多个候选行为，reward gap 提供局部偏好。
- 只训练 task-vector coefficient，不训练 base 权重，也不训练 LoRA。

### Idea 2: Global Strength + Local Residual

`global-parameter` 是一个很好的论文点：

```text
alpha_{p,e} = g_e + r_{p,e}
```

解释：

- `g_e` 学每个专家能力整体该多强。
- `r_{p,e}` 允许局部层/矩阵修正。
- prior 可以对 global 和 residual 分别设权重，形成稳定的低维主控制 + 高维微调。

可以和以下 baseline 对比：

| Baseline | 问题 |
| --- | --- |
| scalar task arithmetic | 无法分层处理冲突 |
| full 588 independent coeff | 小数据下信号太分散 |
| layer-band | 过粗，可能漏掉模块差异 |
| global-parameter | 当前推荐 |

### Idea 3: Proposal Pool 不等于训练集

proposal producer 可以很多：

```text
base / expert / current merge / stage2 / fixed 0.75 merge
```

但训练集只保留 verifier-confirmed frontier。固定 0.75 可以用来“产生候选”，但不能用来“扫最优系数”。这能避免审稿人质疑测试集调参。

### Idea 4: Capability-Balanced Curriculum

当前训练按 task quota，下一步应按 capability quota：

```text
tool_multi_call: 40
tool_math: 40
memory_final_retrieval: 40
memory_write: 40
code_graph: 40
code_string: 40
code_math: 40
```

这样能把“任务均衡”推进成“能力均衡”，叙述更科研。

### Idea 5: Trust Region for Task-Vector Coefficients

source-reward stage3 曾出现小批量把 Code 系数推过头的问题。可系统研究：

```text
|alpha - alpha_init| <= delta
```

或者：

```text
L = L_preference + lambda_global * ||g-g0||^2 + lambda_residual * ||r-r0||^2
```

报告时画 Pareto 曲线：

```text
delta / prior strength vs Tool-Memory-Code heldout
```

### Idea 6: Memory 过程 reward

当前 Memory 主要是最终检索。要扩到：

| 子能力 | 检查 |
| --- | --- |
| `memory_write` | 是否写入新事实 |
| `memory_retention` | 干扰后是否保持 |
| `memory_conflict_update` | 新事实是否覆盖旧事实 |
| `memory_final_retrieval` | 最终回答是否用正确记忆 |

这能把“答对 HotpotQA final answer”升级成真正的 memory behavior merging。

### Idea 7: Code 多测试 reward

Code 需要从 binary pass/fail 改为更连续：

```json
{
  "compile": 1,
  "public_pass_rate": 0.75,
  "generated_pass_rate": 0.50,
  "timeout": 0,
  "format_ok": 1,
  "reward": 0.65
}
```

训练 reward 用 public/generated tests，hidden tests 只用于 final eval。

## 最容易踩的坑

1. **dry-run manifest 不能主训练**  
   错误：`$ROOT/modes/opvec4/mode_manifest.json`  
   正确：`$ROOT/modes/opvec4-full-bf16-real/mode_manifest.json`

2. **不传 zero gate 就不是 from-zero**  
   `configs/opvec4.yaml` 默认 `common=0.50`。

3. **单卡挂 196 个模块容易爆显存**  
   用 `--device-map auto` 和多卡 `max-memory`。

4. **standardized 文件不能直接喂当前训练入口**  
   当前训练入口需要原始 rollout 的 prompt 和 samples text。

5. **Code 子集评测不是生成代码能力**  
   `cure_eval_candidate_selector_logprob.py` 是候选选择 logprob，不生成新程序。

6. **当前 from-zero 模型不是最终方法**  
   它证明可训练，但没有超过 stage2。

## 最小验收清单

- [ ] `zero_init_global_parameter_full_bf16.gates.json` 里 588 个参数级系数全为 0。
- [ ] 训练 summary 里 `parameter_coefficients=588`。
- [ ] 训练 summary 里 `gate_grad_nonzero=true`。
- [ ] 烘焙 summary 里 `num_delta_entries=588`。
- [ ] 子集评测至少复现 from-zero vs stage2 表格。
- [ ] v2 数据 summary 复现 Tool56 / Memory26 / Code40。
- [ ] standardized summary 有 capability counts 和 split_group counts。
- [ ] 所有结论区分 calibration frontier、subset trend、final benchmark。

## 推荐下一步

优先级：

1. 用 `frontier_v2` 三域 raw filtered 数据训练一个保守版 from-zero 模型。
2. 与 stage2 跑同一套 subset trend。
3. 如果 subset 不退，再做 fold0 group-disjoint frontier diagnostic。
4. 然后扩 Memory process reward 和 Code generated tests，把 kept frontier 扩到每域 200。
5. 最后才跑完整官方 harness，避免在弱数据上烧评测预算。
