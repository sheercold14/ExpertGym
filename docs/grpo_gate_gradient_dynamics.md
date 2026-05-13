# GRPO Gate 梯度动力学：为什么 token-level grad norm 小，以及如何让 gate 动起来

## 0. 结论先行

在我们当前的 OP-VEC gated GRPO 里，`gate` 不是普通 LM 权重，而是控制 task vector 混合强度的低维参数。训练信号链路是：

```text
gate -> merged model 权重 -> rollout logprob -> GRPO loss -> gate 梯度 -> gate 更新
```

因此，`grad_norm` 小不等价于 gate 一定推不动。真正决定 gate 能不能学的是：

1. **reward 是否有组内方差**：同一题多条 sample 如果 reward 全一样，GRPO 的 advantage 接近 0。
2. **advantage 是否压得太小**：reward 标准化、token 平均、batch mean 都会缩小数值。
3. **logprob 对 gate 是否敏感**：task vector 改变后，如果目标 token 概率几乎不变，梯度自然很小。
4. **优化器是否把小梯度归一化掉**：AdamW 第一步主要看梯度方向，不看绝对尺度；多步以后才更依赖二阶矩和持续信号。
5. **gate 参数化是否限制变化**：global/shared/grouped/full gate、delta clamp、prior loss 都会影响可学习空间。

如果 token-level loss 的 `grad_norm` 小，但 reward 持续上涨、gate 方向稳定，就不需要强行放大。  
如果 reward 不涨、gate 长期贴近初始化、有效样本 advantage 很小，才需要干预训练动力。

---

## 1. 我们当前训练的数学对象

设第 `i` 个 prompt 下采样 `G` 个回答，回答记为 `y_i^j`。当前 gate 为 `g`，它决定合并模型：

```text
theta(g) = theta_base + sum_e c_e(g) * delta_e
```

其中：

- `theta_base`：base model。
- `delta_e`：第 `e` 个 expert task vector。
- `c_e(g)`：由 gate 参数生成的 expert 系数。

模型生成回答的概率是：

```text
pi_g(y | x) = product_t pi_g(y_t | x, y_<t)
```

sequence logprob：

```text
log pi_g(y | x) = sum_t log pi_g(y_t | x, y_<t)
```

官方 reward 给每条回答一个分数：

```text
r_i^j = R(x_i, y_i^j)
```

GRPO 不训练 value model，而是在同一 prompt 的多条回答之间做组内标准化：

```text
A_i^j = (r_i^j - mean_j(r_i^j)) / (std_j(r_i^j) + eps)
```

如果同一题的 reward 是：

```text
[1, 1, 1, 1]
```

那么：

```text
A = [0, 0, 0, 0]
```

这题对 gate 没有有效 RL 梯度。  
所以我们一直强调 calibration data 不能全是“全对题”，也不能全是“全错且无人能做对题”。最有价值的是：

```text
[0, 0, 1, 1]
[0, 1, 0, 1]
[0, 0.2, 0.8, 1]
```

这种组内有方差、且模型行为有可改善空间的题。

---

## 2. GRPO 梯度到底怎么推 gate

PPO/GRPO 常用 clipped objective：

```text
rho = exp(log pi_g(y|x) - log pi_old(y|x))

L = - mean( min(
  rho * A,
  clip(rho, 1 - eps, 1 + eps) * A
))
```

这里 `pi_old` 是 rollout 时的行为策略，`pi_g` 是当前更新时的策略。

对 gate 求梯度：

```text
dL/dg
  = dL/dlogpi * dlogpi/dtheta * dtheta/dg
```

展开看，gate 能被推动需要三件事同时成立：

```text
reward 方差 -> A 非零
模型概率可变 -> dlogpi/dtheta 非零
task vector 对权重有影响 -> dtheta/dg 非零
```

其中：

- `A > 0` 的样本：提高它的 logprob。
- `A < 0` 的样本：降低它的 logprob。
- 如果某个 expert delta 让高 reward 样本更容易生成，它对应的 gate 会被推高。
- 如果某个 expert delta 让低 reward 样本更容易生成，它对应的 gate 会被压低。

这就是 gate 自动发现 task vector 组合的核心机制。

---

## 3. sequence-level 和 token-level 的区别

### 3.1 sequence-level loss

sequence-level 把整段回答看成一个动作：

```text
logpi_seq = sum_t logpi_t
rho_seq = exp(logpi_seq - old_logpi_seq)
loss_seq = - rho_seq * A
```

优点：

- 和“整条回答 reward”匹配。
- 长回答天然有更大的 logprob 变化空间。
- 对长轨迹 Memory 这类任务推动更强。

缺点：

- 长回答会天然贡献更大的梯度。
- Memory 轨迹可能压过 Code/Tool。
- 不同长度样本之间 credit 不公平。

### 3.2 token-level loss

token-level 把每个 response token 都作为优化位置：

```text
rho_t = exp(logpi_t - old_logpi_t)
loss_token = - mean_t(rho_t * A)
```

或者更准确地说，是在有效 response mask 上平均：

```text
loss_token = - sum_t mask_t * rho_t * A / sum_t mask_t
```

优点：

- 长回答不会因为 token 多而天然支配梯度。
- Memory 长轨迹和 Code 短回答之间更公平。
- 更接近很多 RLHF/VeRL 框架的 token 级实现。

缺点：

- 梯度范数通常会比 sequence-level 小很多。
- 因为 `sum_t` 变成了 `mean_t`，长度维度被平均掉了。
- 如果 reward 只在 final answer 上给，所有 token 共享同一个 advantage，credit assignment 仍然粗糙。

---

## 4. 为什么 token-level grad norm 会小

假设一条回答有 `T=1000` 个 token。

sequence-level 近似：

```text
grad_seq ≈ sum_t grad(logpi_t) * A
```

token-level 平均：

```text
grad_token ≈ (1/T) * sum_t grad(logpi_t) * A
```

所以粗略看：

```text
grad_token ≈ grad_seq / T
```

实际不会完全除以 `T`，因为不同 token 梯度方向会抵消，AdamW 也会归一化，但数量级变小是正常的。

这不是 bug，而是 token-level 选择了“长度归一化”。它牺牲了长回答的天然梯度优势，换来跨任务更公平。

---

## 5. grad norm 小，为什么 gate 仍可能更新一致

AdamW 第一步近似：

```text
m = grad
v = grad^2
update ≈ lr * m / sqrt(v)
       ≈ lr * sign(grad)
```

所以如果 sequence-level 和 token-level 的梯度方向一致：

```text
grad_seq   = [100, -50, 20]
grad_token = [0.1, -0.05, 0.02]
```

AdamW 第一步可能给出接近的更新：

```text
update ≈ [+lr, -lr, +lr]
```

因此：

- `grad_norm` 不一致：说明 raw gradient scale 不同。
- `gate_delta` 一致：说明梯度方向一致，且 AdamW 对尺度不敏感。

这只在早期尤其明显。多步以后，AdamW 的 moving average 会积累历史，sequence/token 如果方向或相对比例不同，gate 轨迹就会分叉。

---

## 6. token-level 推不动 gate 时，先诊断什么

### 6.1 看 reward 是否有组内方差

最重要的表：

```text
每个 prompt 的 rewards:
[1, 1, 1, 1] -> 无 RL 信号
[0, 0, 0, 0] -> 无组内方差信号，但可用于 hard negative 分析
[0, 1, 1, 1] -> 有信号
[0, 0, 1, 1] -> 强信号
```

优先统计：

```text
frontier_ratio = 有组内 reward 方差的 prompt 数 / prompt 总数
mean_abs_adv   = 平均 |advantage|
reward_std     = 组内 reward std
```

如果 `frontier_ratio` 很低，调学习率没有用，应该先改数据筛选。

### 6.2 看 gate 梯度方向是否稳定

不要只看 `grad_norm`，要看：

```text
每轮 gate delta 的方向
不同任务 residual 的符号
sequence/token 梯度 cosine
同一任务多 batch 的梯度符号一致率
```

如果方向乱跳，说明 calibration data 冲突、reward 噪声大或 batch 太小。

### 6.3 看 logprob 对 gate 是否敏感

可以固定一批样本，比较不同 gate 下的 logprob：

```text
gate = 0.00
gate = 0.33
gate = 0.50
gate = 0.75
gate = 1.00
```

如果 reward 或 logprob 基本不变，说明这些样本对 task vector 不敏感，不能作为好 calibration data。

### 6.4 看 prior / clamp 是否压住 gate

常见阻力：

```text
prior-loss-weight 太大
max-coefficient-delta-from-init 太小
lr 太小
batch loss 被多次 mean 缩小
token loss 又除以 token 数
```

如果 gate 每步都贴着 trust region 边界，说明不是推不动，而是被限制住。  
如果 gate 完全不动，才是梯度/优化器/数据问题。

---

## 7. 让 token-level gate 动起来的调参优先级

### 优先级 A：先修数据，而不是先放大学习率

目标 calibration data 应该提高：

```text
frontier prompt 比例
reward std
expert-recoverable 样本比例
任务覆盖均衡性
```

推荐采样结构：

```text
40%: 当前 baseline 部分失败，但至少一个 rollout 成功
30%: expert 明显优于 baseline 的 recovery 样本
20%: 中等难度、reward 分布不饱和样本
10%: 全错 hard negative，用于保留探索边界
```

不推荐：

```text
大量全对题
大量全错且 expert 也不会的题
只筛单任务高方差题，导致 gate 牺牲其他任务
```

### 优先级 B：保持 token-level，但调 loss scaling

如果 token-level 梯度太小，可以使用显式 loss scale：

```text
loss = token_loss * loss_scale
```

常见选择：

```text
loss_scale = sqrt(avg_response_len)
loss_scale = fixed 16 / 32 / 64
loss_scale = per-task tuned scale
```

注意：如果用 AdamW，单纯整体乘一个常数，对第一步影响可能很小；但对多步 Adam moment、weight decay、grad clipping、混合辅助 loss 的相对比例仍然有影响。

### 优先级 C：调学习率和 batch

如果方向稳定但步子太小：

```text
lr: 0.005 -> 0.01 -> 0.02
update_batch_size: 4 -> 8/16
num_epochs: 1 -> 3/5
```

判断标准：

```text
reward 上升
gate 不震荡
KL 不爆
clip_frac 不长期接近 1
```

### 优先级 D：放松 gate 约束

如果 gate 总被限制：

```text
max-coefficient-delta-from-init: 0.15 -> 0.25 -> 0.35
prior-loss-weight: 0.01 -> 0.003 -> 0.0
```

但这一步有风险：可能快速破坏已有能力。必须同时监控 Code/Memory/Tool 分任务 reward。

### 优先级 E：换 gate 参数化

从弱到强：

```text
global 4 参数
grouped module gate
588 full gate
per-layer/per-module structured gate
```

更强参数化能表达更细，但更容易过拟合，也更依赖 calibration data 的覆盖。

---

## 8. self-compare reward 为什么能缓解 reward 饱和

raw reward 问的是：

```text
这个输出对不对？
```

如果很多题 raw reward 都是 1，就没有差异。

self-compare 问的是：

```text
当前 gate 是否比 reference gate 更好？
```

构造：

```text
delta_reward = reward(candidate_gate) - reward(reference_gate)
```

好处：

- 全对题中，如果 candidate 更短、更稳定、格式更好，可以通过细粒度 reward 重新产生差异。
- 当前模型和上一阶段模型比较，可以持续制造“变好/变差”的方向。
- 更符合 on-policy iterative improvement。

风险：

- reference 选不好会导致目标漂移。
- 如果只在小 calibration set 上 self-compare，容易过拟合这批题。
- 需要固定 held-out set 判断是否真的泛化。

---

## 9. Memory 任务的特殊问题

Memory 的 rollout 通常包含：

```text
update turns + final answer
```

sequence-level 如果把所有 turn 的 logprob 求和，会让 Memory 梯度天然更大：

```text
grad_memory ≈ update_turns + final_turn
```

这可能让 Memory gate 快速上涨，同时牺牲 Code/Tool。

token-level 会把长轨迹平均掉，减少 Memory 长度优势。  
但如果官方 MemAgent 的 credit assignment 本来就希望 update turns 也承担 credit，那么 token-level 必须保证：

```text
update turns 的 token logprob 被纳入 loss
final turn 的 token logprob 被纳入 loss
mask 正确
reward/advantage 分配逻辑和官方训练一致
```

否则会出现“reward 对了，但梯度没有打到记忆更新轨迹”的问题。

---

## 10. 什么时候该用 sequence-level，什么时候该用 token-level

推荐判断：

```text
如果目标是快速确认 gate 能不能被推动：
  用 sequence-level smoke。

如果目标是公平比较 Code/Tool/Memory：
  用 token-level。

如果 Memory 长轨迹压制其他任务：
  用 token-level 或 per-task normalization。

如果 token-level 完全推不动：
  先看 frontier 数据比例，再调 loss scale/lr。
```

更稳的折中方案：

```text
loss = alpha * sequence_loss + (1 - alpha) * token_loss
```

例如：

```text
alpha = 0.2
```

这样保留整条回答 reward 的强信号，同时避免长轨迹完全支配。

---

## 11. 实验监控表

每轮至少记录：

```text
reward_mean
reward_std
frontier_prompt_ratio
mean_abs_adv
grad_norm
grad_sign_by_gate
gate_common
gate_tool_residual
gate_memory_residual
gate_code_residual
clip_frac
approx_kl
prior_loss
policy_loss
per_task_reward_mean
per_task_gate_delta
```

关键解释：

```text
reward 不涨 + frontier_ratio 低:
  数据无信号。

reward 不涨 + frontier_ratio 高 + grad_norm 小:
  loss scale / lr / gate sensitivity 问题。

reward 涨 + gate 不怎么变:
  可能初始 gate 已经接近局部最优，或 Adam 小步足够。

gate 大幅变 + reward 不涨:
  reward 噪声、过拟合、任务冲突或 KL 太大。

Memory 涨 + Code 掉:
  长轨迹/任务权重不平衡。

clip_frac 长期很高:
  lr 太大或每轮更新太多，PPO clipping 在阻止学习。
```

---

## 12. 当前项目的建议策略

现阶段不要简单因为 token-level `grad_norm` 小就放弃 token-level。更合理的推进顺序：

1. 用 sequence-level 做短 smoke，确认 gate 方向能被推动。
2. 用 token-level 对齐更严格的 credit assignment，尤其是 Memory update turns。
3. 统计 calibration data 的 frontier ratio，剔除大量全对饱和题。
4. 每个任务分别看 reward std 和 gate delta，不让 Memory 长轨迹吃掉全部梯度。
5. 如果 token-level reward 不涨，再加 loss scale 或提高 lr。
6. 保留 held-out question bank，防止 calibration set 上自我强化。

推荐初始设置：

```text
loss_granularity = token
update_batch_size = 4 或 8
lr = 0.005 起步
num_epochs = 3
prior_loss_weight = 0.003 到 0.01
max_coefficient_delta_from_init = 0.15 到 0.25
samples_per_prompt = 4 或 8
```

如果 1-2 轮后：

```text
frontier_ratio >= 30%
mean_abs_adv 正常
gate 方向稳定
reward 不涨
```

再尝试：

```text
lr 提到 0.01
token_loss_scale = 16 或 sqrt(avg_len)
prior_loss_weight 降低
```

如果：

```text
frontier_ratio < 10%
```

优先重做数据筛选，不要调优化器。

---

## 13. 一句话理解

token-level 不是“梯度更弱的 GRPO”，而是“去掉长度偏置后的 GRPO”。  
如果去掉长度偏置以后 gate 推不动，通常不是 token-level 错了，而是 calibration data、advantage 方差、gate 敏感性或约束项没有给出足够清晰的方向。

