# VeRL MemAgent Rollout 与 Reward 对齐报告

更新时间：2026-05-12

## 当前 Todo

- [x] 让 VeRL 数据准备阶段把 Memory 轨迹样本自动路由到自定义 agent loop：`agent_name=opvec_memagent`。
- [x] 新增 `opvec_memagent` loop，使用 MemAgent 官方 update/final prompt 模板做 on-policy 轨迹生成。
- [x] 修复 reward wrapper：Memory mixed batch 下优先用 `memagent_final_text` 计分，避免把整段 rollout 拼接文本当最终答案。
- [x] 修复无 `flash_attn` 环境下的 VeRL 启动：默认追加 `attn_implementation=eager`。
- [x] 此前 final-answer scope 跑通过 1 条 Memory tiny smoke，验证自定义 loop、reward、old logprob、actor update、vLLM weight sync 的基础工程路径。
- [x] 对齐 native/MemAgent credit assignment：Memory update turns + final turn 都进入 PPO/GRPO，final reward 的 GRPO advantage 映射回整条轨迹。
- [x] 做 CPU 单元验证：同一 rollout 的 update/final rows 共享同一个 final-answer advantage，组内归一化只用 final rows。
- [ ] 做 `SAMPLES_PER_PROMPT>=2` 的 Memory full smoke，确认存在 reward 方差时 gate 梯度非零。
- [ ] 做 `tool,memory,code` mixed smoke，确认三任务 agent routing 与 reward 统计同时正常。
- [ ] 修复 OP-VEC effective weight sync 的显存问题；当前 full smoke 在首次 actor->vLLM 同步时 OOM，尚未进入 rollout。

## 代码入口

- 启动脚本：`third_party/verl/examples/opvec_gated_grpo/run_verl_grpo_smoke10.sh`
- 自定义 loop 配置：`third_party/verl/examples/opvec_gated_grpo/opvec_agent_loops.yaml`
- MemAgent loop：`third_party/verl/verl/experimental/opvec/memagent_agent_loop.py`
- 数据转换：`third_party/verl/verl/experimental/opvec/prepare_data.py`
- Reward wrapper：`third_party/verl/verl/experimental/opvec/reward_fn.py`
- Reward manager 透传：`third_party/verl/verl/workers/reward_manager/naive.py`
- GRPO advantage 映射：`third_party/verl/verl/trainer/ppo/ray_trainer.py`
- AgentLoop 多行返回支持：`third_party/verl/verl/experimental/agent_loop/agent_loop.py`

## 数据路由

`prepare_data.py` 读取 calibration JSONL，输出 VeRL parquet：

```text
data_source = opvec
prompt      = messages 或 user prompt
ability     = task
agent_name  = opvec_memagent | single_turn_agent
extra_info  = prompt_id/task/source/prompt/reference_json/verifier_json/selection...
```

路由规则：

```text
if task == memory and reference.metadata.memagent_chunks exists:
    agent_name = opvec_memagent
else:
    agent_name = single_turn_agent
```

这保证 tool/code 不被 Memory loop 影响；Memory trajectory 样本不再退化为 single-turn 问答。

## VeRL Rollout 轨迹

### Tool/Code

当前仍走 `single_turn_agent`：

```text
raw_prompt -> chat_template -> vLLM generate -> response
```

Reward 由 `RewardRouter.score(prompt_record, response)` 计算。

### Memory

当前 `opvec_memagent` 的训练范围是 `trajectory_all_turns`：

```text
memory_0 = "No previous memory"

for chunk_i in memagent_chunks:
    prompt_i = MEMAGENT_UPDATE_TEMPLATE(question, memory_{i-1}, chunk_i)
    memory_i = policy.generate(prompt_i)

final_prompt = MEMAGENT_FINAL_TEMPLATE(question, memory_last)
final_answer = policy.generate(final_prompt)
reward = MemAgent HotpotQA reward(final_answer, ground_truths)
```

VeRL PPO tensor 中，一条 logical rollout 被展开成多行：

```text
row 1..T-1:
  prompt_ids    = update_prompt_i tokens
  response_ids  = memory_update_i tokens
  response_mask = 1 for update tokens

row T:
  prompt_ids    = final_prompt tokens
  response_ids  = final_answer tokens
  response_mask = 1 for final answer tokens
```

所有 rows 都带同一个 `memagent_rollout_key`，final row 额外标记：

```text
memagent_training_scope = trajectory_all_turns
memagent_final_mask     = true only for final_answer row
memagent_reward         = final answer reward
```

这与 native MemAgent 的 recurrent rollout 语义一致：update turns 是 on-policy action，不只是 latent state，也直接承担 policy-gradient loss。

## Reward 计算

VeRL 的 `NaiveRewardManager` 会 decode：

```text
prompt_str   = tokenizer.decode(valid prompt ids)
response_str = tokenizer.decode(valid response ids)
```

然后调用：

```python
compute_score(
    data_source="opvec",
    solution_str=response_str,
    ground_truth=reward_model["ground_truth"],
    extra_info=extra_info,
)
```

OP-VEC wrapper 重建 `prompt_record`：

```text
reference = json.loads(reference_json) or ground_truth
task      = extra_info.task
```

Memory 特例现在有两层：

```text
1. opvec_memagent loop 内部先用 final_text 调 RewardRouter.score(...)，把 final reward 写入每个 turn row 的 rm_scores。
2. 若某些路径没有 rm_scores，reward_fn 仍会在 task=memory 时优先用 memagent_final_text 计分，避免 update 文本被当成最终答案。
```

最后统一进入：

```python
RewardRouter.score(prompt_record, score_text)
```

### Memory 官方 reward

当前 Memory reward 仍使用项目里的官方对齐 adapter：

```text
reward_source = MemAgent/verl/utils/reward_score/hotpotqa.py
definition    = compute_score(solution[-300:], ground_truth_list)
```

数学上：

```text
r_memory(y) = max_{a in A} 1[ normalize(extract_boxed(y[-300:])) == normalize(a) ]
```

其中 `A` 是 HotpotQA ground-truth answer list。实现还记录 boxed/exact/f1 等 diagnostic，但训练主 reward 是 scalar。

### GRPO 梯度

VeRL 对同一 prompt 的 `K=SAMPLES_PER_PROMPT` 个响应做组内 advantage：

```text
A_i = normalize_group(r_i)
L_pg = - mean_i min(
    ratio_i * A_i,
    clip(ratio_i, 1-eps, 1+eps) * A_i
)
```

gate 参数通过 actor logprob 反传：

```text
W_eff = W_base + c_tool * Δ_tool + c_memory * Δ_memory + c_code * Δ_code
∂L / ∂c_e = <∂L/∂W_eff, Δ_e>
```

因此 reward 本身不直接“知道”哪个 expert；它通过输出 logprob 对 `W_eff` 的梯度，把有利于高 reward response 的 expert delta 系数推高。

Memory recurrent 对齐后，advantage 的实际计算是：

```text
1. 只取 final rows 的 reward 做同 prompt 组内归一化：
   A_final,k = (r_final,k - mean_prompt(r_final)) / std_prompt(r_final)

2. 按 memagent_rollout_key 把 A_final,k 映射回同一 rollout 的所有 rows：
   A_update_i,k = A_final,k
   A_final,k    = A_final,k

3. 每个 row 内所有 response_mask=1 的 token 都乘同一个 scalar advantage。
```

这避免了一个 subtle bias：如果直接把 final reward 复制到所有 update/final rows 再让普通 GRPO 归一化，`torch.std(unbiased=True)` 会因为每条 rollout 被重复 T 次而改变方差尺度。当前实现只用 final rows 估计 mean/std，和官方 MemAgent `final_batch -> compute_1D_grpo_advantage -> sample_index` 的逻辑对齐。

## 与 Native Pipeline 的差异

已对齐：

- Memory 使用 MemAgent update/final 官方模板。
- Memory final answer 使用官方 HotpotQA reward adapter。
- Memory update 是 on-policy 生成，最终答案依赖当前 gate 模型生成的 memory。
- Memory update turns + final turn 都进入 policy loss。
- Final reward 只按 final answer 计算，再把同一个 scalar advantage 分配给该 rollout 的所有 turns。

未完全等价：

- native 的 recurrent trainer 在一个专门的 `generation_manager.run_llm_loop` 里维护 `final_mask/sample_index`；当前实现是在 VeRL AgentLoop 层把一条 Memory rollout 展开成多行，并在标准 `compute_advantage` 里按 `memagent_rollout_key` 映射 advantage。数学目标一致，工程承载位置不同。
- native 还有 frontier filter、task quota、prior loss、coefficient trust region；当前 VeRL 标准 GRPO 没完整复刻这些控制项。
- OP-VEC actor->vLLM effective weight sync 当前仍是瓶颈：`export_effective_hf_state_dict(model.state_dict())` 在 colocated FSDP+vLLM 下容易 OOM，需要后续改成 streaming/materialize-before-resume 或降低常驻显存。

结论：Memory credit assignment 已从 final-only 修正为 native-style recurrent GRPO；当前 blocker 转移到 OP-VEC 权重同步显存，而不是 reward/advantage 公式。

## 验证结果

已完成验证：

```text
py_compile:
  agent_loop.py
  memagent_agent_loop.py
  ray_trainer.py

CPU unit:
  两条 rollout，每条 update+final 两行
  final rewards = [1.0, 3.0]
  mapped advantages = [-0.707106, -0.707106, +0.707106, +0.707106]
```

full smoke 状态：

```text
1 GPU, vLLM mem util 0.35:
  vLLM init OOM，free 26.5GB < requested 27.8GB。

1 GPU, vLLM mem util 0.20:
  vLLM init 成功，但首次 actor->vLLM update_weights 在 export_effective_hf_state_dict 时 OOM。

2 GPU, TP=1 或 TP=2:
  vLLM init 成功，但仍在首次 update_weights 的 FSDP state_dict/export 阶段 OOM。
```

因此当前代码层面的 Memory credit assignment 已通过静态与单元验证；端到端 full smoke 需要先处理 OP-VEC effective weight sync 显存。

## 对整体构想的意义

当前 VeRL 版本已经能支持我们的核心方向：

```text
用少量 high-information calibration prompts
-> on-policy rollout
-> 官方 reward
-> GRPO 更新 task-vector gate
-> 让模型自动发现 task vector 的组合方向
```

关键实验应避免 reward 饱和：

- 每个任务保留“当前 gate 不稳定、但 expert/task-vector 有机会救回来”的样本。
- `SAMPLES_PER_PROMPT` 至少为 2，最好 4 或 8，否则 GRPO 没有组内差异。
- Memory 的 update 轨迹很长，若 reward 方差过强或过弱，需单独调 `MEMORY_UPDATE_MAX_NEW_TOKENS`、`MEMORY_FINAL_MAX_NEW_TOKENS`、prompt truncation。
- 如果 Memory 梯度持续压制 Tool/Code，需要 task-balanced batch 或 per-task advantage normalization。

## 下一步建议

第一阶段先跑 mixed smoke：

```text
TASKS=tool,memory,code
LIMIT=6 or 12
SAMPLES_PER_PROMPT=2
MAX_GATED_MODULES=1
```

确认三任务 reward 都能进 metrics、Memory `num_turns>2`、gate grad 非零。

第二阶段跑可学习小实验：

```text
LIMIT=30
SAMPLES_PER_PROMPT=4
STRATEGY=global
INIT_VALUE=1/3
TOTAL_TRAINING_STEPS=2
```

判断 reward 是否随 step 上升、`memory_residual/tool_residual/code_residual` 是否出现可解释分化。

第三阶段先修 weight sync：

```text
方案 A：在 update_weights 前先导出/stream effective params，再 resume vLLM weights。
方案 B：让 export_effective_hf_state_dict 按 tensor 流式产出，避免 full state_dict clone 峰值。
方案 C：用更多 GPU 或 CPU/offload 先跑通 full smoke，但这只是绕过，不是根治。
```

当前最务实的是方案 C 起步；如果 Memory 训练信号明显不足，再做 A/B。
