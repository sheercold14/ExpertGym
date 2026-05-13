# OnPolicy Gated-GRPO 后续 Batch Forward/Backward 优化计划

日期：2026-05-12  
状态：**今晚先不改训练核心代码；等正确版实验跑完后再做。**  
目标：在不改变训练语义的前提下，把 update 阶段从逐 sample/turn forward-backward 改成 batch logprob + batch loss，降低 HF update 成本。

## 背景

当前正确主线应为：

```text
vLLM rollout 只生成文本
HF update 侧补 old_logprobs
HF update 侧计算 current_logprobs
token-level GRPO loss
task-level advantage normalization
```

已确认不能用：

```text
vLLM 保存 token old_logprobs + HF 计算 current_logprobs
```

原因：old/current 不同源会污染 PPO ratio，导致 tool/memory gate 被错误压低。

## 当前 update 为什么慢

现在的 `UPDATE_BATCH_SIZE=4` 主要是 optimizer 层面的 batch：

```text
for row in frontier_rows:
  for sample in row.samples:
    for turn in sample.trajectory or final_response:
      forward current logprob
      backward loss

  每累计 4 个 row:
    grad clip
    optimizer.step
    gate projection
```

它减少的是：

```text
optimizer.step 次数
grad clip 次数
gate projection 次数
```

但没有充分减少：

```text
model forward 次数
model backward 次数
Python 循环
tokenizer 调用
Memory 多 turn 的逐条调度
```

真正的 batch forward/backward 应该是：

```text
flatten 多任务样本/轨迹
-> 得到 logprob examples
-> 按长度 bucket
-> pad 成 batch tensors
-> 一次 HF forward 得到 token logprobs
-> 用 response_mask 聚合 sample/turn loss
-> 一次或少量 backward
```

## 设计原则

必须保持的语义：

```text
1. old/current logprob 同源：都由 HF tokenizer + HF model 计算。
2. token-level loss 不变：advantage broadcast 到 response tokens。
3. Memory credit assignment 不丢：update turns + final turn 都参与 policy loss。
4. task advantage normalization 不变。
5. frontier row 选择不变。
6. gate projection / max_delta / prior loss 不变。
```

不做：

```text
不再启用 vLLM token old_logprobs
不把 reward 逻辑揉进 update batcher
不迁移 VeRL 作为第一步
不改变 calibration data schema
不改变 gate 参数化语义
```

## 目标速度

以 `100 prompts * 4 samples * 1024 tokens` 粗估：

```text
当前正确版:
  单轮约 25-40 min
  update 约 18-30 min

优化后目标:
  单轮约 12-20 min
  update 约 6-12 min
```

合理预期：

```text
update 2-3x 提速
做得好 3-4x
总训练 1.5-2.5x 提速
```

不承诺 10x，因为仍有：

```text
bake checkpoint
vLLM reload
Memory rollout/reward
长上下文 padding waste
共享存储 I/O
```

## 数据结构计划

新增内部结构，不改变外部 rollout jsonl：

```python
LogprobExample:
  example_id: str
  row_id: str
  sample_id: str
  task: str
  turn_id: str | None
  prompt_text: str
  response_text: str
  input_ids: Tensor
  attention_mask: Tensor
  response_mask: Tensor
  old_logprobs: Tensor | None
  advantage: float
  frontier_weight: float
  metadata: dict
```

Memory flatten 规则：

```text
sample.trajectory[*]:
  kind = memory_update
  prompt_text = 当前 turn prompt
  text = 当前 turn response

sample final:
  kind = final_answer
  prompt_text = final prompt
  text = final answer

同一个 sample 的多个 turn 共享同一个 scalar advantage。
loss 可按所有 response tokens 聚合，也可先 turn 内 mean 再 sample 内 mean。
第一版建议按所有 response tokens 聚合，和 token-level mask 语义最直接。
```

Tool / Code flatten 规则：

```text
一个 sample 通常对应一个 final response example。
如果 Tool 后续有多 tool-call trajectory，也复用 Memory 的多 turn 结构。
```

## 实现阶段

### Phase 0：锁定基线

输入：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/diag_taskvec_20260512_212029/fixed_rollout_update_ab/shared_stripped_24.jsonl
```

当前可信基线：

```text
token + HF-fill old + batch:
  tool   0.340001
  memory 0.333257
  code   0.333910
```

验收：

```text
新 batch update 在同一 rollout 上 gate 方向一致。
允许数值有小差异，但不能出现 tool/memory 同时大幅下跌。
```

### Phase 1：批量 tokenization 和 logprob examples

候选文件：

```text
opvec/modeling/logprob.py
scripts/train/opvec_update_gates_from_rollouts.py
```

任务：

```text
1. 从 rows 中 flatten 出 LogprobExample。
2. 支持 Memory trajectory。
3. 支持普通 response。
4. 保留 example_id -> row/sample/turn 映射。
5. 加 tokenizer cache，避免 old/current 两遍重复 tokenize。
```

验收：

```text
同一 batch 中 response_mask token 数与旧实现一致。
Memory sample 的 total response tokens = trajectory turns + final turn。
```

### Phase 2：batch old-logprob fill

当前慢点之一：

```text
_fill_missing_old_logprobs 逐 sample / turn 计算。
```

目标：

```text
with torch.no_grad():
  batch forward
  gather response token logprobs
  写回 sample.old_logprobs / response_mask / response_token_ids
  Memory 同时写回每个 trajectory turn
```

注意：

```text
old_logprob 必须在 optimizer step 前一次性补齐。
update 期间不能重新补 old_logprob。
```

验收：

```text
filled_missing_old_logprobs 数量与旧实现一致。
sample.old_logprob = sum(sample.old_logprobs)。
Memory sample.old_logprob = sum(all turn old_logprob + final old_logprob)。
```

### Phase 3：batch current-logprob + token GRPO loss

目标：

```text
同一 mini-batch examples:
  current_logprobs = model forward
  ratio = exp(current - old)
  token_loss = clipped GRPO token loss
  mask response tokens
  aggregate by sample / row
  backward once per mini-batch
```

关键问题：

```text
Adam step 粒度必须与 UPDATE_BATCH_SIZE 语义一致。
如果旧版是每 4 rows step 一次，新版也应每 4 rows step 一次。
```

第一版可保守：

```text
每个 update mini-batch 包含 4 rows。
把这 4 rows 的所有 samples/turns flatten 后 batch forward。
一次 backward。
一次 optimizer.step。
```

这样最容易和现有 `UPDATE_BATCH_SIZE=4` 对齐。

### Phase 4：长度 bucket 和 OOM fallback

长短混合会产生 padding waste。第一版先按粗粒度 bucket：

```text
<=512
<=1024
<=2048
<=4096
<=8192
```

OOM fallback：

```text
batch_size 减半
仍 OOM 则按单 row fallback
记录 warning 到 summary
```

不要静默跳样本。

### Phase 5：日志和 summary

新增 summary 字段：

```text
logprob_batch_enabled
logprob_batch_size
num_logprob_examples
num_memory_turn_examples
old_logprob_fill_seconds
current_logprob_backward_seconds
tokenization_seconds
oom_fallback_count
padding_efficiency
```

保留旧字段：

```text
filled_missing_old_logprobs
frontier_task_counts
optimizer_steps
task_scales
loss_granularity
```

## 测试计划

### 单元测试

候选文件：

```text
tests/test_logprob.py
tests/test_update_gates_objectives.py
tests/test_rollout_schema.py
```

新增覆盖：

```text
1. 单 response token mask 对齐。
2. Memory trajectory 多 turn flatten / restore。
3. old_logprob sum 与 token old_logprobs sum 一致。
4. batch 与逐条 logprob 在小模型/短文本上接近。
5. OOM fallback 不丢样本。
```

### Fixed rollout 对照

必须跑：

```bash
shared_stripped_24.jsonl
```

对照目标：

```text
旧 token + HF-fill + batch:
  tool   0.340001
  memory 0.333257
  code   0.333910

新 batch-forward:
  tool 应上涨
  memory/code 不应显著塌
```

### 小规模真实训练

```text
NUM_PROMPTS=10
SAMPLES_PER_PROMPT=4
NUM_ITERS=1
```

检查：

```text
rollout 正常
update 正常
filled_missing_old_logprobs > 0
gate_updates.gates.json 存在
summary timing 合理
```

### 正式验证

```text
NUM_PROMPTS=100
SAMPLES_PER_PROMPT=4
NUM_ITERS=2
ROLLOUT_SHARDS=auto
```

比较：

```text
旧正确版 vs 新 batch-forward:
  reward trend
  gate trend
  update time
  peak memory
```

## 主要风险

### 1. 梯度尺度变化

batch loss 聚合方式可能改变梯度尺度。必须明确：

```text
按 row 平均？
按 sample 平均？
按 token 平均？
按 turn 平均？
```

建议第一版对齐当前 token loss：

```text
每 row 内 samples 平均。
每 sample 内 response_mask token mean。
UPDATE_BATCH_SIZE=4 时再对 4 rows mean。
```

### 2. Memory 长轨迹 credit assignment

风险：

```text
只训练 final answer，丢 update turns。
或者 update turns 权重过大，压倒 final answer。
```

验收必须检查：

```text
Memory sample 的 update turns + final turn 都有 response_mask。
```

### 3. Adam step 顺序不同

旧 row-by-row 和新 batch-forward 的 Adam step 顺序可能不同。不要追求 bitwise 一致，要追求：

```text
同一 rollout 下 gate 方向一致
无明显任务塌陷
reward trend 不退化
```

### 4. 长度 padding waste

Memory 长 prompt 会让 batch padding 变大。需要 bucket，否则 batch forward 可能不快。

### 5. 代码侵入过大

第一版不要重写 updater。推荐局部新增：

```text
batch logprob helper
batch old fill path
batch token objective path
保留旧逐条 fallback
```

## 建议执行顺序

1. 跑完今晚正确版实验，确认 `token + HF-fill old` 多轮趋势。
2. 固定 24-row rollout 作为 regression fixture。
3. 做 batch old-logprob fill。
4. 做 batch current-logprob + token loss。
5. 跑 fixed rollout 对照。
6. 跑 10 prompts smoke。
7. 跑 100 prompts 两轮。
8. 若速度收益不足，再做长度 bucket / tokenization cache。

## 成功标准

最低成功：

```text
同一 fixed rollout 下不改变 gate 方向。
update time 有可见下降。
无 vLLM old-logprob 参与训练。
```

可用于正式实验：

```text
100 prompts * 4 samples * 2 iters 稳定跑完。
task-vector 不再因为 logprob 路径错误整体下跌。
summary 能解释每轮 reward / gate / timing。
```

理想状态：

```text
update 2-3x 提速。
单轮 100 prompts 从 25-40 min 降到 12-20 min。
```

