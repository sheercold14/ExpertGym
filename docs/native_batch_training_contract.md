# Native Batch Gated-GRPO 训练契约

日期：2026-05-12  
目标：在保留 native 路径透明、低显存、易审计优势的前提下，把 OP-VEC gated GRPO 从逐样本更新推进到 batch 化、可对齐 VeRL 的训练语义。

## 1. 每轮 On-Policy 边界

一轮训练必须固定一个行为策略：

```text
gate_t
  -> rollout / reward / old_logprob
  -> update
  -> gate_{t+1}
```

契约：

- `rollouts.jsonl` 中的 `policy_id`、`gate_checkpoint`、`gate_values` 描述产生 response 的行为策略。
- 新生成的 rollout row 会写 `group_id`、`gate_id`、`seed`；旧 rollout 允许缺失这些字段，但不应用作最终审计证据。
- `old_logprob` 必须由同一个 `gate_t` 计算；如果 vLLM 只生成文本，update 阶段用 `--fill-missing-old-logprob` 只能在优化前补一次。
- update 期间不能边优化边重新解释同一批 rollout 的 `old_logprob`。
- `gate_updates.gates.json` 是下一轮 rollout 的唯一 gate 输入。

判断是否违规：

- 同一个 `iter_xxx/rollouts.jsonl` 里混有多个 `gate_checkpoint`，需要拆分。
- update 前后重复补 `old_logprob`，会把 behavior policy 改成 moving target。
- rollout 用 baked checkpoint，update 却没有记录对应 gate checkpoint，需要在 summary 中明确 `policy_model` 与 `gate_values` 的来源。

## 2. Rollout Schema 目标

当前 schema 已有：

```text
run_id
created_at
step
policy_id
gate_checkpoint
gate_values
prompt_id
task
prompt
reference
rendered_prompt
samples[]
frontier
keep_for_policy_loss
skip_reason
```

每个 `samples[]` 当前已有：

```text
sample_id
text
old_logprob
old_logprob_max_length
length
reward
task_reward
contract_reward
success
details
trajectory?        # Memory recurrent 样本
```

下一版必须补齐 token-level 字段：

```text
response_token_ids
old_logprobs       # per-token, 与 response_token_ids 对齐
response_mask      # 1 表示参与 policy loss
loss_scope         # final_only | trajectory_all_turns | behavior_span
group_id           # GRPO 同组归一化 key，默认 prompt_id
gate_id            # gate_t 的稳定 id/hash
seed
```

Memory 特例：

- `trajectory[]` 必须完整保存 update turns + final turn。
- 每个 turn 要保存 `kind`、`prompt_text`、`text`、`response_token_ids`、`old_logprobs`、`response_mask`。
- final reward 只由官方 MemAgent HotpotQA final answer reward 决定，但 policy gradient 要覆盖 update turns + final turn。

## 3. Reward 契约

统一入口：

```python
RewardRouter.score(prompt_record, output_text)
RewardRouter.batch_score(prompt_records, output_texts)
```

当前语义：

- `tool`：ToolRL 官方风格 `format_reward + tool_call_correctness_reward`。
- `memory`：MemAgent HotpotQA final boxed answer reward；update turn 本身不直接给主 reward。
- `code`：CURE 风格 source tests pass rate / public example fallback。
- `batch_score()` 目前只是稳定 batch API，内部仍串行调用各官方 adapter；它不改变 reward 数值。

训练时 reward 使用原则：

- raw reward 作为 verifier outcome，不直接跨任务比较强弱。
- GRPO advantage 默认在同一个 `group_id` 内计算。
- 任务间梯度强弱由 task weight、advantage normalization、token normalization 控制，而不是改官方 reward 定义。

## 4. Update 契约

当前 native update 已支持：

```text
--update-batch-size N
--batch-loss-reduction mean|sum
```

当前行为：

- `N=1` 保持旧的 row-by-row `optimizer.step()`。
- `N>1` 会累积多个 kept frontier / retention row 的梯度，再统一 clip、step、project。
- `mean` 按固定 `1 / update_batch_size` 缩放每个 row loss；最后不足一个 batch 的 flush 会偏保守。
- 当前仍是 sequence-level logprob ratio，不是 VeRL token-mask 等价 PPO/GRPO。

目标行为：

```text
old_log_probs: [batch, response_len]
new_log_probs: [batch, response_len]
response_mask: [batch, response_len]
advantages:   [batch] -> broadcast 到 token
```

推荐 loss：

```text
ratio_t = exp(new_logprob_t - old_logprob_t)
policy_loss_t = -min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps) * A)
loss = masked_mean(policy_loss_t + beta_kl * kl_t)
```

Memory 的 `A` 应按 sample / rollout 计算一次，再广播到该 sample 的所有 update/final response tokens。不要把每个 turn 当成独立 GRPO 样本重新归一化。

## 5. 监控契约

每轮至少记录：

- reward histogram：按 task、按 bucket。
- frontier ratio：`kept_frontiers / rows`。
- advantage std / mean abs advantage。
- clip frac、approx KL。
- grad norm、skipped optimizer steps。
- gate mean / min / max / delta。
- 每个 task vector 的系数趋势。

异常判断：

- reward 大量全对：raw GRPO 信号饱和，需要 question bank 或 self-compare。
- reward 大量全错且无可恢复样本：梯度方向不可用，不能靠重复训练硬推。
- Memory 梯度远大于 Tool/Code：优先检查 token 数量、trajectory 长度、task normalization。
- gate 长期不动：检查 reward 方差、old_logprob 是否存在、current logprob 是否参与反传。

## 6. 当前状态

已完成：

- native mini-batch update 的第一版实现。
- `RewardRouter.batch_score()` batch API。
- vLLM rollout 的普通样本和 Memory final reward 已改为走 batch reward 入口。
- vLLM collector 可选 `--store-token-logprobs`，直接保存生成时 sampled token 的 `old_logprobs`，减少 update 阶段重复 old-policy forward。
- token update 优先使用 rollout 里保存的 `response_token_ids` 重算 current logprob，避免 vLLM 文本 detokenize 后再 tokenize 造成长度漂移。
- rollout row/sample schema 校验与 token-level 字段长度校验。
- HF collector 可选 `--store-token-logprobs` 写出 `response_token_ids`、`old_logprobs`、`response_mask`；Memory 会聚合 update turns + final turn。
- token-level clipped GRPO / reverse-KL 纯函数和单测；语义是 group scalar advantage 广播到 response mask token。
- production updater 支持 `--loss-granularity token`，显式启用时 PPO/GRPO 与 KL 走 token-level old/new logprob。
- `--fill-missing-old-logprob` 在 token mode 下会同时补齐 vLLM baked rollout 缺失的 token-level old logprob，包括 Memory trajectory turns。
- update 日志记录 `clip_frac`、`approx_kl`；epoch summary 记录均值。
- HF/vLLM collector 新生成的 row 写入 `group_id`、`gate_id`、`seed`。
- 主 HF loop 和 bake+vLLM loop 已透传 batch/token loss 参数。
- bake+vLLM loop 已透传 `--store-token-logprobs`；official vLLM launcher 在 `LOSS_GRANULARITY=token` 时默认打开。
- `skill/command/run_smoke_sequence_vs_token.sh` 可用同一份 10-prompt rollout 对照 sequence/token update。

未完成：

- sequence/token 两种 loss 在真实 10-prompt smoke 上还需要对照，默认暂不切到 token。
- 常驻 vLLM server 和 gate-only sync。
