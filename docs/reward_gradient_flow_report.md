# Reward 与梯度流程调试报告

日期：2026-05-11  
项目：`OnPolicyMerge_gated_grpo`

## 0. 一句话结论

当前流程可以理解为：

```text
seed manifest 只提供 prompt/reference/verifier metadata
        ↓
当前 policy 生成多条 samples
        ↓
RewardRouter 给每条 sample 打 task reward
        ↓
同一个 prompt 内 reward 差异变成 GRPO advantage
        ↓
update 阶段重新计算 current logprob
        ↓
old_logprob 与 current logprob 构造 PPO/GRPO loss
        ↓
只更新 OP-VEC gate/coefficient 参数
```

最应该先看懂的是：

```text
rollouts.jsonl 里的 samples[].details
stage_a_gate_updates.jsonl 里的 grad_norm / gates
```

## 1. 关键文件地址

### 1.1 Reward 相关

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/opvec/rewards/router.py
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/opvec/rewards/simple.py
```

`router.py` 负责按 task 分发：

```text
tool   -> ToolRewardAdapter
memory -> MemoryRewardAdapter
code   -> CodeRewardAdapter
```

`simple.py` 负责实际 reward 计算。

### 1.2 Rollout 生成

HF 版本：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/scripts/train/opvec_collect_hf_rollouts.py
```

vLLM baked checkpoint 版本：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/scripts/train/opvec_collect_vllm_rollouts.py
```

当前昨晚主要使用的是 vLLM 路径。

### 1.3 Gate 更新

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/scripts/train/opvec_update_gates_from_rollouts.py
```

这个文件负责：

```text
读取 rollouts
过滤 kept frontier rows
计算 advantage
补 old_logprob
重新计算 current logprob
反传 gate gradient
写 gate_updates.jsonl / summary / gates.json
```

### 1.4 当前最重要的实验产物

High-info bundle：

```text
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.prompts.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.guard.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.summary.json
```

One-iteration probe：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.summary.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_gate_updates.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_gate_updates.summary.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_b_gate_updates.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_b_gate_updates.summary.json
```

## 2. 数据从哪里来

### 2.1 seed manifest

当前官方 reward 对齐 seed manifest：

```text
/tmp/shared-storage/OnPolicy/data/source_reward/routed1_correct_official_seed20260510.jsonl
```

它的作用不是保存训练 rollout，而是提供：

```text
prompt_id
task
prompt / messages
reference
verifier metadata
tags
```

也就是说，manifest 是“题目和参考信息”，不是当前 policy 的采样结果。

### 2.2 high-info prompts

当前 high-info prompt 文件：

```text
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.prompts.jsonl
```

统计：

```text
tool   30
memory 35
code   25
total  90
```

其中有一部分来自之前发现过 reward 方差的 frontier prompt，另一部分来自 official routed correct pool 填充。

summary 地址：

```text
/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.summary.json
```

里面最关键字段：

```json
{
  "prompt_selection": {
    "frontier_seeded": {
      "code": 7,
      "memory": 7,
      "tool": 7
    },
    "fill": {
      "code": 18,
      "memory": 28,
      "tool": 23
    }
  }
}
```

解释：

```text
frontier_seeded = 之前已经确认有 reward 方差的 prompt
fill = 从官方 correct pool 里补进来的 prompt
```

## 3. Rollout 中间输出怎么看

当前最应该看的 rollout：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl
```

每一行是一条 prompt 的 rollout row，结构大致是：

```json
{
  "prompt_id": "...",
  "task": "tool | memory | code",
  "prompt": "...",
  "reference": {...},
  "rendered_prompt": "...",
  "samples": [
    {
      "sample_id": "...__k0",
      "text": "model output",
      "reward": 4.0,
      "task_reward": 4.0,
      "contract_reward": 0.1,
      "success": true,
      "old_logprob": null,
      "details": {...}
    }
  ],
  "frontier": {...},
  "keep_for_policy_loss": true
}
```

对于 vLLM rollout：

```text
old_logprob = null
```

原因是 vLLM 只负责生成文本和打 reward。`old_logprob` 在 update 阶段由 HF gated model 补齐。

快速查看三类样本：

```bash
python - <<'PY'
import json
p="/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl"
seen=set()
for line in open(p):
    row=json.loads(line)
    task=row["task"]
    if task in seen:
        continue
    seen.add(task)
    print("\n===", task, row["prompt_id"], "keep=", row["keep_for_policy_loss"])
    for s in row["samples"]:
        print("reward=", s["reward"], "success=", s.get("success"))
        print("details=", s.get("details"))
    if len(seen)==3:
        break
PY
```

## 4. Reward 是怎么给的

### 4.1 Tool reward

实现位置：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/opvec/rewards/simple.py
ToolRewardAdapter
```

当前对齐 ToolRL 默认 reward：

```text
reward = format_reward + tool_call_correctness_reward
```

范围：

```text
format_reward: 0 或 1
tool_call_correctness_reward: -3 到 3
total reward: -3 到 4
```

也就是说 Tool reward 可以大于 1，这是正常的。

#### Tool format reward

如果 reference 需要 tool call，那么预测必须满足类似格式：

```text
<think> ... </think>
<tool_call>
{"name": "...", "parameters": {...}}
</tool_call>
```

如果 reference 同时要求 tool call 和 response，则格式要是：

```text
<think> ... </think>
<tool_call>
...
</tool_call>
<response> ... </response>
```

通过格式检查：

```text
format_reward = 1
```

否则：

```text
format_reward = 0
```

#### Tool correctness reward

核心逻辑：

```text
完全匹配 reference tool calls -> +3
无法解析 tool call           -> -3
部分匹配 tool name / params   -> 在 -3 到 +3 之间插值
```

当前已经按 ToolRL 官方行为对齐两个细节：

```text
1. 只解析第一个 <tool_call>...</tool_call> block
2. block 内任意 JSON 行非法，或者缺 name/parameters，correctness 直接给 -3
```

调试时重点看：

```json
details.format_score
details.toolrl_correctness_raw
details.parseable
details.prediction_calls
details.reference_calls
details.exact_tool_match
details.name_recall
```

### 4.2 Memory reward

实现位置：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/opvec/rewards/simple.py
MemoryRewardAdapter
```

当前 Memory 是 MemAgent recurrent trajectory：

```text
chunk 1 -> 当前 policy 生成 updated memory
chunk 2 -> 使用上一步 memory，再生成 updated memory
...
final -> 使用最后 memory，生成 boxed final answer
```

主 reward 只打在 final answer 上：

```text
final answer boxed exact match -> 1
否则 -> 0
```

中间的 `memory_update` turn 不单独给主 reward。

调试时重点看：

```json
details.boxed_found
details.prediction
details.ground_truths
details.training_exact_match
details.token_f1
```

Memory rollout 的 sample 里还会有：

```json
samples[].trajectory[]
```

结构类似：

```json
[
  {
    "kind": "memory_update",
    "prompt_text": "...",
    "text": "Updated memory: ...",
    "old_logprob": null
  },
  {
    "kind": "final_answer",
    "prompt_text": "...",
    "text": "\\boxed{...}",
    "old_logprob": null
  }
]
```

update 阶段计算 logprob 时，会把所有 memory update turn 和 final answer turn 的 logprob 加起来。

### 4.3 Code reward

实现位置：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/opvec/rewards/simple.py
CodeRewardAdapter
```

优先级：

```text
1. 如果 reference metadata 里有 source tests，运行 source tests，reward = pass_rate
2. 如果没有 source tests，但 prompt 里能解析 public examples，则用 examples 做近似 pass_rate
3. 如果都没有，则退化成 syntax/input/print/reference overlap 等弱 heuristic
```

Code reward 通常在：

```text
0 到 1
```

调试时重点看：

```json
details.syntax_ok
details.source_tests
details.public_examples
details.input_used
details.output_used
```

## 4.4 三个代表性输出例子

下面三个例子都来自当前 rollout：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl
```

选它们的原因是：每个 prompt 下 4 个 samples 的 reward 有差异，所以这些样本实际会进入 GRPO frontier。

### 4.4.1 Tool 例子：完全 tool call 与漏参数

prompt id：

```text
tool__c94120f65da56fef
```

最后一轮用户请求：

```text
Find the top 100 players by matchmaking rank in TrackMania from page 0.
```

reference 期望的 tool call：

```text
<tool_call>
{"name": "2kLySV", "parameters": {"8ex.lJ": 100, "3FAM": 0}}
</tool_call>
```

这条 prompt 的 4 个 samples：

```text
sample 0 reward = 4.0
sample 1 reward = 4.0
sample 2 reward = 1.75
sample 3 reward = 4.0
```

sample 0 输出：

```text
<think> The user wants to fetch the top 100 players by matchmaking rank in TrackMania from page 0. ... </think>
<tool_call>
{"name": "2kLySV", "parameters": {"8ex.lJ": 100, "3FAM": 0}}
</tool_call>
```

sample 0 reward details：

```json
{
  "format_score": 1.0,
  "toolrl_correctness_raw": 3.0,
  "toolrl_raw_total": 4.0,
  "parseable": true,
  "prediction_calls": 1,
  "reference_calls": 1,
  "exact_tool_match": true,
  "reference_tool_names": ["2kLySV"],
  "prediction_tool_names": ["2kLySV"],
  "tool_call_parse_error": null
}
```

解释：

```text
格式正确 -> format_score = 1
tool name 和两个参数都完全一致 -> correctness = 3
总 reward = 1 + 3 = 4
```

sample 2 输出：

```text
<think> The user wants to find the top 100 players by matchmaking rank in TrackMania. ... </think>
<tool_call>
{"name": "2kLySV", "parameters": {"8ex.lJ": 100}}
</tool_call>
```

sample 2 reward details：

```json
{
  "format_score": 1.0,
  "toolrl_correctness_raw": 0.75,
  "toolrl_raw_total": 1.75,
  "parseable": true,
  "prediction_calls": 1,
  "reference_calls": 1,
  "exact_tool_match": false,
  "reference_tool_names": ["2kLySV"],
  "prediction_tool_names": ["2kLySV"],
  "tool_call_parse_error": null
}
```

解释：

```text
格式正确 -> format_score = 1
tool name 正确
参数 "8ex.lJ": 100 正确
但是漏掉参数 "3FAM": 0
所以 correctness 不是 3，而是部分分 0.75
总 reward = 1 + 0.75 = 1.75
```

这条样本最适合用来理解 Tool reward，因为它清楚展示了：

```text
完全匹配 -> 4.0
漏参数 -> 1.75
```

### 4.4.2 Memory 例子：boxed exact 才给 1

prompt id：

```text
memory__ecc439ade0f80705
```

问题：

```text
What retailer is the second-largest in the United States and has a commercial featuring the American artist who was 1st runner-up in the 2005 USA Weekend Magazine's songwriting competition?
```

reference final answer：

```text
\boxed{Target Corporation}
```

这条 prompt 的 4 个 samples：

```text
sample 0 reward = 1.0
sample 1 reward = 0.0
sample 2 reward = 1.0
sample 3 reward = 1.0
```

sample 0 输出：

```text
\boxed{Target Corporation}
```

sample 0 reward details：

```json
{
  "boxed_found": true,
  "prediction": "Target Corporation",
  "ground_truths": ["Target Corporation"],
  "training_exact_match": true,
  "token_f1": 1.0
}
```

解释：

```text
boxed answer 存在
prediction 与 ground truth 完全一致
所以 reward = 1
```

sample 1 输出：

```text
\boxed{Target}
```

sample 1 reward details：

```json
{
  "boxed_found": true,
  "prediction": "Target",
  "ground_truths": ["Target Corporation"],
  "training_exact_match": false,
  "token_f1": 0.6666666666666666
}
```

解释：

```text
虽然 "Target" 和 "Target Corporation" 很接近
token_f1 也不是 0
但是训练主 reward 用 boxed exact
不是完全一致，所以 reward = 0
```

这条样本最适合用来理解 Memory reward 的稀疏性：

```text
\boxed{Target Corporation} -> 1
\boxed{Target}             -> 0
```

### 4.4.3 Code 例子：source tests pass_rate 直接作为 reward

prompt id：

```text
code__ed3efbfa38a67793
```

问题概要：

```text
Aika 的咖啡店有若干托盘，每个托盘有 Ki 个 scones。
选择连续托盘区间，计算总数对 m 的余数，输出最大 surplus。
```

这条 prompt 的 4 个 samples：

```text
sample 0 reward = 0.375
sample 1 reward = 0.0
sample 2 reward = 0.0
sample 3 reward = 1.0
```

sample 0 reward details：

```json
{
  "syntax_ok": true,
  "source_tests": {
    "total": 8,
    "passed": 3,
    "pass_rate": 0.375,
    "task_id": 2293,
    "source": "CodeContests_train"
  }
}
```

解释：

```text
代码语法正确
source tests 共 8 个
通过 3 个
所以 reward = 3 / 8 = 0.375
```

sample 1 reward details：

```json
{
  "syntax_ok": true,
  "source_tests": {
    "total": 8,
    "passed": 0,
    "pass_rate": 0.0,
    "task_id": 2293,
    "source": "CodeContests_train"
  }
}
```

解释：

```text
语法正确不代表 reward 高
如果 source tests 全部失败
reward = 0
```

sample 3 reward details：

```json
{
  "syntax_ok": true,
  "source_tests": {
    "total": 8,
    "passed": 8,
    "pass_rate": 1.0,
    "task_id": 2293,
    "source": "CodeContests_train"
  }
}
```

解释：

```text
8 个 source tests 全部通过
reward = 1
```

这条样本最适合用来理解 Code reward：

```text
reward 不是由语言描述质量决定
而是由 source tests pass_rate 决定
```

## 5. Frontier 是什么

不是所有 prompt 都会进入 gate update。

一个 prompt 必须满足：

```text
同一个 prompt 下多条 samples 的 reward 有差异
```

例如：

```text
rewards = [1, 1, 1, 1] -> 没有 GRPO 信号
rewards = [0, 1, 0, 1] -> 有 GRPO 信号
```

当前 rollout summary：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.summary.json
```

统计结果：

```text
input prompts:
  code   25
  memory 35
  tool   30

kept frontier rows:
  code   12
  memory 7
  tool   6
```

这说明 90 个 prompts 里，只有 25 个 prompts 在当前 policy 下产生了有效 GRPO 信号。

## 6. Reward 怎么变成 advantage

update 文件：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/scripts/train/opvec_update_gates_from_rollouts.py
```

对每一个 kept prompt：

```text
samples rewards = [r1, r2, r3, r4]
```

先做组内归一化：

```text
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

再乘 frontier weight。

如果打开了：

```text
--task-normalize-advantages
```

还会让不同 task 的 mean absolute advantage 接近，避免 Tool reward scale 明显大于 Code/Memory 时主导训练。

当前 high-info Stage A 使用了 task-normalized advantage：

```json
{
  "task_normalize_advantages": true,
  "task_scales": {
    "code": 1.0082,
    "memory": 1.0310,
    "tool": 0.9632
  }
}
```

## 7. old_logprob 和 current logprob

### 7.1 vLLM rollout 为什么没有 old_logprob

vLLM 路径输出：

```text
samples[].old_logprob = null
```

因为 vLLM 生成时没有在当前代码里保存训练所需的 token-level logprob。

所以 update 时使用：

```text
--fill-missing-old-logprob
```

在优化 gate 之前，用初始 gate policy 重新算一遍 response logprob，并写回内存中的 sample：

```text
sample["old_logprob"] = ...
sample["old_logprob_max_length"] = ...
```

当前 Stage A summary：

```json
{
  "filled_missing_old_logprobs": 100
}
```

解释：

```text
Stage A 有 25 条 kept frontier rows
每条 4 samples
一共补了 100 个 old_logprob
```

### 7.2 current logprob

优化时，对每条 sample 重新计算当前 gate 下的 logprob：

```text
current_logprob = log p_current_policy(sample_text | prompt)
```

然后构造 PPO ratio：

```text
ratio = exp(current_logprob - old_logprob)
```

loss 形式是 clipped PPO/GRPO loss：

```text
loss = - min(ratio * advantage, clipped_ratio * advantage)
```

直观理解：

```text
高 reward sample 的 advantage > 0
  -> 希望 current_logprob 增大

低 reward sample 的 advantage < 0
  -> 希望 current_logprob 减小
```

## 8. Gate 梯度在哪里看

当前最直接的 update 输出：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_gate_updates.jsonl
```

每一行是一条 prompt update 后的日志：

```json
{
  "prompt_id": "...",
  "task": "tool",
  "loss": ...,
  "policy_loss": ...,
  "kl_loss": ...,
  "grad_norm": ...,
  "mean_reward": ...,
  "frontier_weight": ...,
  "mean_abs_advantage": ...,
  "gates": {
    "common": ...,
    "tool_residual": ...,
    "memory_residual": ...,
    "code_residual": ...
  }
}
```

快速查看：

```bash
python - <<'PY'
import json
p="/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_gate_updates.jsonl"
for i,line in zip(range(20), open(p)):
    r=json.loads(line)
    print(r["step"], r["task"], r["prompt_id"])
    print("loss=", r["loss"], "grad_norm=", r["grad_norm"])
    print("mean_reward=", r["mean_reward"], "mean_abs_advantage=", r.get("mean_abs_advantage"))
    print("gates=", r["gates"])
    print()
PY
```

Stage A summary：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_gate_updates.summary.json
```

关键结果：

```json
{
  "frontier_task_counts": {
    "code": 12,
    "memory": 7,
    "tool": 6
  },
  "filled_missing_old_logprobs": 100,
  "updates": 25,
  "epoch_summaries": [
    {
      "grad_norm_max": 0.2572,
      "gate_delta_max": 0.01917
    }
  ],
  "final_gates": {
    "common": 0.76254,
    "tool_residual": 0.01917,
    "memory_residual": -0.01172,
    "code_residual": -0.00745
  }
}
```

折算 task coefficient：

```text
tool   = common + tool_residual   ~= 0.78171
memory = common + memory_residual ~= 0.75081
code   = common + code_residual   ~= 0.75509
```

## 9. Stage A 和 Stage B 的区别

当前 high-info two-stage 设计：

```text
Stage A: on-policy GRPO/PPO
Stage B: offline expert recovery distill
```

### 9.1 Stage A

输入：

```text
high_info_v1_seed20260511.prompts.jsonl
```

流程：

```text
当前 policy 现场生成 samples
reward router 打分
有 reward 方差的 prompt 进入 update
使用 PPO/GRPO loss 更新 gate
```

这是严格 on-policy。

### 9.2 Stage B

输入：

```text
high_info_v1_seed20260511.distill.jsonl
```

流程：

```text
使用已有 expert-recovery rollout rows
ppo_loss_weight = 0
best_response_loss / pairwise_loss > 0
不把这些样本当作 PPO on-policy samples
```

Stage B 当前结果：

```json
{
  "frontier_task_counts": {
    "code": 13,
    "memory": 13,
    "tool": 13
  },
  "updates": 39,
  "final_gates": {
    "common": 0.73666,
    "tool_residual": 0.00505,
    "memory_residual": -0.02585,
    "code_residual": 0.02080
  }
}
```

解释：

```text
Stage A 更像“当前 policy 自己探索出的偏好更新”
Stage B 更像“用专家/正样本恢复能力”
```

两者的梯度含义不同，调试时不要混在一起看。

## 10. 推荐调试顺序

### 第一步：只看 reward

先不要看梯度。

目标：

```text
确认 samples[].details 是否解释了 reward
```

命令：

```bash
python - <<'PY'
import json
p="/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl"
for line in open(p):
    row=json.loads(line)
    rewards=[s["reward"] for s in row["samples"]]
    if row["keep_for_policy_loss"] and len(set(rewards)) > 1:
        print("task=", row["task"])
        print("prompt_id=", row["prompt_id"])
        print("rewards=", rewards)
        for s in row["samples"]:
            print("\nTEXT:\n", s["text"][:1000])
            print("REWARD:", s["reward"])
            print("DETAILS:", s.get("details"))
        break
PY
```

### 第二步：按 task 分开看

不要三类一起 debug。

建议顺序：

```text
1. Tool
2. Memory
3. Code
```

原因：

```text
Tool reward 细节最多，但 details 最可解释
Memory 是 0/1 boxed exact，容易判断 reward 是否合理
Code 涉及运行测试，最后再看
```

### 第三步：看同一个 prompt 的 advantage

找一个 frontier prompt：

```bash
python - <<'PY'
import json
p="/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl"
for line in open(p):
    row=json.loads(line)
    rewards=[float(s["reward"]) for s in row["samples"]]
    if row["keep_for_policy_loss"] and len(set(rewards)) > 1:
        mean=sum(rewards)/len(rewards)
        var=sum((x-mean)**2 for x in rewards)/len(rewards)
        std=var**0.5
        adv=[(x-mean)/std if std else 0 for x in rewards]
        print(row["task"], row["prompt_id"])
        print("rewards=", rewards)
        print("advantages=", adv)
        break
PY
```

### 第四步：看 gate update log

用 prompt_id 对照：

```bash
PROMPT_ID="填上一步看到的 prompt_id"
python - <<PY
import json
p="/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/stage_a_gate_updates.jsonl"
for line in open(p):
    r=json.loads(line)
    if r["prompt_id"] == "$PROMPT_ID":
        print(json.dumps(r, ensure_ascii=False, indent=2))
PY
```

### 第五步：需要看真实 gate grad 时临时加打印

位置：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/scripts/train/opvec_update_gates_from_rollouts.py
```

在这些调用后面加：

```text
loss.backward()
sample_loss.backward()
prior_loss.backward()
```

临时打印：

```python
for name, param in gate_manager.named_parameters():
    if param.grad is not None:
        print("[gate_grad]", name, param.grad.detach().float().cpu().tolist())
```

如果只想看一个 prompt，建议先构造小调试命令，不要跑全量。

## 11. 最小单 prompt 调试建议

先从 rollout 中选一个 prompt_id：

```bash
python - <<'PY'
import json
p="/tmp/shared-storage/OnPolicy/runs/gated_grpo/high_info_v1_global_twostage_i1_probe_20260511/iter_001/rollouts.jsonl"
for line in open(p):
    row=json.loads(line)
    rewards=[s["reward"] for s in row["samples"]]
    if row["task"]=="tool" and row["keep_for_policy_loss"] and len(set(rewards)) > 1:
        print(row["prompt_id"], rewards)
        break
PY
```

然后可以只用这个 prompt 重新 collect：

```bash
cd /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo

PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
ROOT=/tmp/shared-storage/OnPolicy
MODE=$ROOT/modes/opvec4/mode_manifest.json
SEED=$ROOT/data/calibration/high_info_v1_seed20260511.prompts.jsonl

$PY scripts/train/opvec_collect_hf_rollouts.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest $MODE \
  --seed-manifest $SEED \
  --output /tmp/debug_one_prompt_rollouts.jsonl \
  --prompt-id <PROMPT_ID> \
  --num-prompts 1 \
  --samples-per-prompt 4 \
  --max-gated-modules 1 \
  --max-new-tokens 512 \
  --max-prompt-tokens 2048 \
  --max-logprob-tokens 3072 \
  --device cuda \
  --torch-dtype bfloat16
```

再只 update 这个 prompt：

```bash
$PY scripts/train/opvec_update_gates_from_rollouts.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest $MODE \
  --rollouts /tmp/debug_one_prompt_rollouts.jsonl \
  --output /tmp/debug_one_prompt_gate_updates.jsonl \
  --max-steps 1 \
  --max-logprob-tokens 3072 \
  --lr 0.005 \
  --prior-loss-weight 0.02 \
  --ppo-loss-weight 1.0 \
  --best-response-loss-weight 0.0 \
  --pairwise-loss-weight 0.0 \
  --gate-parameterization global \
  --max-gated-modules 1 \
  --device cuda \
  --torch-dtype bfloat16
```

这样可以直接观察：

```text
reward 是否合理
advantage 是否合理
old_logprob 是否存在
grad_norm 是否非零
gate 是否移动
```

## 12. 当前最需要警惕的点

### 12.1 Tool reward scale 大

Tool reward 范围是：

```text
-3 到 4
```

Code/Memory 基本是：

```text
0 到 1
```

所以如果不做 task advantage normalization，Tool 很容易主导训练。

当前 high-info Stage A 已经打开：

```text
--task-normalize-advantages
```

这是合理的。

### 12.2 Memory reward 很稀疏

Memory 只有 final answer boxed exact 给主 reward。

所以它容易出现：

```text
全 0
全 1
```

这两种都没有 GRPO 信号。只有同 prompt 下 samples 有成功有失败，才有有效梯度。

### 12.3 vLLM rollout 与 HF update 是两套模型路径

vLLM 使用 baked checkpoint 生成。

HF update 使用 gated model 重算 logprob。

理论上它们代表同一个初始 gate policy，但实际调试时要警惕：

```text
bake 是否正确
tokenizer 是否一致
max_prompt_tokens / max_logprob_tokens 是否一致
```

### 12.4 Stage A 和 Stage B 不要混淆

Stage A 是 on-policy。

Stage B 是 offline expert recovery。

如果你要研究“GRPO 梯度是否合理”，优先看 Stage A。

如果你要研究“专家正样本是否把 gate 拉向某个 expert”，再看 Stage B。

## 13. 最短阅读路径

如果只想花 30 分钟看懂当前流程，建议按这个顺序：

```text
1. 打开 rollouts.summary.json
   看每个 task 的 mean_reward / kept_frontier_rows

2. 打开 rollouts.jsonl
   找一条 keep_for_policy_loss=true 的 Tool 样本
   看 samples[].text 和 samples[].details

3. 打开 simple.py
   对照 ToolRewardAdapter

4. 打开 stage_a_gate_updates.jsonl
   用同一个 prompt_id 看 grad_norm 和 gates

5. 打开 stage_a_gate_updates.summary.json
   看 final_gates 和 epoch_summaries
```

这条路径可以先避免阅读完整训练框架。
