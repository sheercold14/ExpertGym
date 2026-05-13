# Native Sequence vs Token Loss Smoke 报告

日期：2026-05-12  
运行目录：`/tmp/shared-storage/OnPolicy/runs/gated_grpo/sequence_vs_token_smoke10_native_20260512_153139`

## 运行目的

验证新版 token-level PPO/GRPO 路径是否能在同一份 native HF rollout 上正常 update，并与旧 native sequence-sum loss 做最小对照。

这不是 vLLM 速度测试；本次 rollout 使用：

```text
scripts/train/opvec_collect_hf_rollouts.py
```

原因是 HF collector 可以直接写出 token-level `old_logprobs/response_mask`，最适合检查 loss 语义。

## 运行情况

原计划采 10 prompts，但 HF native rollout 太慢。中途停止 collector，保留已写出的 2 条 rollout 做 partial update 对照。

已写出 rollout：

| step | task | prompt_id | keep | rewards | lengths |
|---:|---|---|---|---|---|
| 1 | tool | `tool__c1798a2697ae052e` | false | `[4.0, 4.0, 4.0, 4.0]` | `[54, 56, 55, 57]` |
| 2 | code | `code__eb7bc7c33b5c3b07` | true | `[0.0, 1.0, 1.0, 1.0]` | `[595, 602, 694, 642]` |

两条 rollout 的所有 sample 都包含 token-level 字段：

```text
response_token_ids
old_logprobs
response_mask
```

因此 partial update 实际只验证了 1 个 Code frontier row，没有覆盖 Memory。

## 对照结果

### Sequence Loss

```text
updates:          1
optimizer_steps: 1
frontier:         {"code": 1}
grad_norm_max:    1.9693217277526855
clip_frac_mean:   0.0
approx_kl_mean:   1.9073486328125e-06
gate_delta_max:   0.006666660308837891
```

Final gates:

```json
{
  "common": 0.7450000643730164,
  "tool_residual": 0.0033333301544189453,
  "memory_residual": -0.006666660308837891,
  "code_residual": 0.0033333301544189453
}
```

### Token Loss

```text
updates:          1
optimizer_steps: 1
frontier:         {"code": 1}
grad_norm_max:    0.0033086524344980717
clip_frac_mean:   0.0
approx_kl_mean:   0.0
gate_delta_max:   0.006666600704193115
```

Final gates:

```json
{
  "common": 0.7450000643730164,
  "tool_residual": 0.00333327054977417,
  "memory_residual": -0.006666600704193115,
  "code_residual": 0.0033333301544189453
}
```

## 判断

本次 partial smoke 说明：

- 新版 token-level update 能正常跑通。
- sequence/token 都产生了非零 gate 梯度和 1 次 optimizer step。
- 两者最终 gate 更新方向一致，数值几乎相同。
- 两者 loss/grad norm 不应期望数值一致：
  - sequence loss 使用整段 logprob sum；
  - token loss 使用 response mask 上的 token mean；
  - 因此 token loss 的 grad norm 明显小是合理的。

这次不能证明：

- token loss 在 Memory trajectory 上已经合理。
- 10-prompt / 2-3 轮训练稳定。
- vLLM batch rollout 与 native HF rollout 完全一致。

下一步应改用更短 token 上限或 vLLM rollout 路径跑完整 10-prompt，再覆盖 Memory。
