# VeRL 迁移可行性判断

生成时间：2026-05-11

## 结论

官方 VeRL 已克隆到：

```text
third_party/verl
```

当前 commit：

```text
f2659862afff9edc353aa9550628dbaf52196f94
```

判断：VeRL 能显著改善标准 GRPO 的 rollout/logprob/update 工程效率，但不能无改动替换当前 OP-VEC gated-GRPO。原因不是 reward 或数据格式，而是 actor 表达不同。VeRL 默认训练普通 HF actor 权重，并把 actor state_dict 同步给 vLLM；我们训练的是少量 gate 系数，真实推理权重是：

```text
W_eff = W_base + c_tool * Delta_tool + c_memory * Delta_memory + c_code * Delta_code
```

所以 VeRL 要完整复刻当前逻辑，必须在 actor 到 vLLM 的权重同步路径上导出 `W_eff`，而不是直接同步 `GatedLinear` 的内部 state_dict。

## 当前 OP-VEC 管线

当前训练环路：

```text
gate checkpoint
  -> bake 成普通 HF checkpoint
  -> vLLM rollout
  -> 官方 reward router 打分
  -> HF 重新算 old/current logprob
  -> 只更新 gate 参数
  -> 下一轮继续 bake
```

最近 global 2 轮实测耗时：

```text
iter1 bake 55s, rollout 671s, update 1368s
iter2 bake 51s, rollout 688s, update 1135s
```

瓶颈是 update，不是 vLLM rollout。update 慢的直接原因是 `opvec_update_gates_from_rollouts.py` 逐 row、逐 sample、逐 trajectory turn 调 HF forward，缺少 batch 化和 no-padding/FSDP 训练引擎。

## VeRL 能直接复用的部分

1. 数据格式可以复用

已有适配：

```text
scripts/frameworks/opvec_prepare_verl_data.py
```

它可以把 OP-VEC seed manifest 转成 VeRL 常见字段：`data_source`、`prompt`、`ability`、`reward_model`、`extra_info`。

2. reward 可以复用

已有适配：

```text
scripts/frameworks/opvec_verl_reward_fn.py
```

它调用当前 `RewardRouter`，所以 Tool/BFCL、Memory/MemAgent、Code reward 的语义可以保持一致。

3. actor gate 安装可以部分复用

已有适配：

```text
scripts/frameworks/opvec_verl_external_lib.py
opvec/frameworks/verl_gated_actor.py
```

它能在 HF model load 后替换 Linear 为 `GatedLinear`，冻结 base LM，只暴露 gate manager 参数。

## 不能直接跑通的关键问题

### 1. vLLM 权重同步不兼容 GatedLinear

VeRL 的 vLLM rollout 会通过 `actor.engine.get_per_tensor_param()` 拿 actor state_dict，然后调用 vLLM `load_weights()`。

普通 HF actor 的 key 是：

```text
model.layers.0.self_attn.q_proj.weight
...
```

我们的 GatedLinear 替换后，state_dict 会出现：

```text
...q_proj.base_linear.weight
...q_proj.delta_tool
...q_proj.delta_memory
...q_proj.delta_code
opvec_gate_manager.*
```

这不是 vLLM Qwen loader 能直接消费的正常权重。因此 VeRL 可以训练 gate，但 rollout 侧不会自动得到正确的 gated policy。

需要实现一个专门的权重导出层：

```text
for each mergeable Linear:
  emit original_param_name.weight = base_weight + sum(coeff * delta)
for non-mergeable params:
  emit original HF param
drop gate manager and delta buffers from vLLM sync payload
```

### 2. VeRL 模板字段已对齐官方版本

官方当前配置里 model hook 字段是：

```text
actor_rollout_ref.model.external_lib
```

不是旧模板里写的 `external_libs`。

当前仓库的 experimental config 已修成 `external_lib`，但整体仍是草稿，因为 actor->vLLM 的 gated weight sync 还没有实现。

### 3. Memory 多轮轨迹需要用 VeRL agent/reward-loop 复刻

VeRL 有 multi-turn rollout 和 reward loop，但默认 reward manager 通常只看最后一段输出。Memory 如果要与官方 MemAgent 完全一致，reward manager 必须能看到完整 trajectory，并把每轮 update/final answer 按官方逻辑打分。

也就是说，Memory 不能只接普通 single-turn custom reward。需要定制：

```text
multi_turn agent loop
custom reward manager
trajectory-aware score extraction
```

## VeRL 是否会让 update 更高效

会，但前提是解决上面的 `W_eff` 同步问题。

VeRL 的优势：

1. actor old_log_prob/current_log_prob 是 batch 化的，不是当前逐 sample HF forward。
2. 支持 no-padding、动态 batch、FSDP/sequence parallel。
3. rollout/update 权重同步和 sleep/wake 已经工程化。
4. reward loop 可以并行打分，适合 code sandbox 和 memory trajectory reward。

对我们当前 100 prompts、4 samples 设置，保守判断：

```text
rollout 约 11 min，目前已足够快
update 约 19-23 min，是主要收益点
VeRL 化后 update 有机会降到数分钟级，但不是只换框架就能得到
```

如果不改权重同步，只把 reward/data 接入 VeRL，收益有限，甚至会因为 actor/vLLM policy 不一致导致训练目标错误。

## 建议的 VeRL 复刻路线

第一阶段：不要直接全量迁移。先做 gate-only weight export smoke。

目标：

```text
给定 gate checkpoint
从 GatedLinear actor 导出普通 HF key 的 W_eff
与当前 bake checkpoint 的同名权重逐层比较
max_abs_diff 接近 0
```

第二阶段：接入 VeRL actor update，但 rollout 仍用每轮 bake checkpoint。

目标：

```text
VeRL 只负责 batch 化 logprob/update
rollout 仍保持当前可审计 bake-vLLM 路径
验证 gate 梯度方向与当前 update 脚本一致
```

第三阶段：替换 VeRL actor->vLLM 权重同步。

目标：

```text
VeRL update gate
同步 W_eff 到 vLLM
不再每轮写完整 baked checkpoint
```

第四阶段：Memory trajectory-aware reward manager。

目标：

```text
Memory rollout/reward 与官方 MemAgent 训练流程一致
reward manager 能访问完整轨迹
```

## Code 为什么会被牺牲

最近 global 2 轮结果：

```text
iter1 code mean_reward 0.4271, success 0.5379
iter2 code mean_reward 0.4117, success 0.5000

iter1 effective coeff:
  tool   0.3833
  memory 0.3833
  code   0.3491

iter2 effective coeff:
  tool   0.4046
  memory 0.4333
  code   0.3735
```

Code 不是完全没涨，而是相对 Tool/Memory 被压低。原因有三层：

1. global 参数化耦合太强

当前 global gate 实际是一个 common 加三个 residual。Memory/Tool 强梯度会先推 common 上涨，然后 code residual 被学成负数来抵消一部分 common。这会形成“整体更像多专家平均，但 Code 方向保守”的结果。

2. Tool reward 在第二轮快速饱和，改变了共享参数的梯度环境

Tool success 从 0.507 到 0.985，frontier 从 24 降到 8。它前期给 common 提供很强正向信号；当 Tool 饱和后，剩下 Memory/Code 的 frontier 仍在，但 Code 的收益没有 Memory 稳定。

3. 不同 gate 组合下样本可解性会变

当前训练只在“当前 gate”上采样。同一个 Code prompt 在 1/3 gate、0.37 gate、0.43 gate 下可解性不同。GRPO 看到的是当前 gate 的 stochastic outputs，不直接比较其他 gate 组合。因此如果某些 Code 题在更高 common/memory 下变难，训练要等 rollout 分布变化后才感知，容易出现一两轮的 Code 回撤。

所以 Code 被牺牲不一定表示 code delta 无效，更可能是参数化和 calibration objective 没有显式保护 Code Pareto 前沿。

## 全错样本当前是否处理

题库构造已经处理了全错样本：

```text
all_fail_partial: success_count == 0 但 reward 有连续方差
all_fail_zero:    success_count == 0 且 reward 几乎无方差
```

全量题库：

```text
code   all_fail_partial 142, all_fail_zero 143
memory all_fail_partial   0, all_fail_zero 133
tool   all_fail_partial 109, all_fail_zero 115
```

calib100 也采了少量全错样本：

```text
code   all_fail_partial 4, all_fail_zero 1
memory all_fail_partial 0, all_fail_zero 1
tool   all_fail_partial 4, all_fail_zero 1
```

但训练 update 的策略是：

```text
all_fail_partial: 如果 reward std >= 阈值，可以保留为 frontier
all_fail_zero: 没有 reward 方差，通常丢到 low_info_failure，不进 policy loss
```

这是合理的，因为纯 GRPO 需要同 prompt 多样本之间有 advantage 差异。`all_fail_zero` 如果没有外部专家恢复信号、自比较 reward delta、过程 reward 或 distillation target，放进 policy loss 只会提供 0 advantage。

## 对全错样本的建议

不要把 `all_fail_zero` 直接塞进 raw GRPO。建议拆成三类：

1. expert-recovery data

同一题 baseline 全错，但 expert 或某个 gate 组合能做对。用作 distillation/pairwise/best-response，解决“当前 policy 完全不会”的冷启动。

2. self-compare data

对不同 gate 或上一阶段 reference gate 做：

```text
delta_reward = reward(candidate_gate) - reward(reference_gate)
```

这样原本绝对 reward 全 0 或全 1 的题，在 gate 比较中仍可能产生信号。

3. guard/anti-regression data

全错且所有已知 gate 都全错的题，不适合当前阶段训练，但适合保留为诊断集合：如果训练后突然变好，说明有新能力；如果训练后高质量题掉分，说明有副作用。

## 下一步推荐

优先级最高的不是完整迁移 VeRL，而是做一个最小可验证 bridge：

```text
GatedLinear actor -> effective HF weights export -> 与当前 bake checkpoint 对齐
```

这个 bridge 一旦成立，VeRL 的 batch old_logprob、actor update、vLLM 权重同步才有意义。否则 VeRL 会让 rollout policy 和训练 actor policy 不一致，速度再快也不可信。

并行地，继续当前 native loop 做三类参数化对照：

```text
global
layer-band
parameter / global-parameter
```

重点看 Code 的 Pareto 曲线，而不是只看总 reward。Code 的保护策略应至少包括：

```text
task-balanced frontier quota
Code guard set
per-task reward/advantage normalization
Code drop 超阈值时停止或降 LR
self-compare reward 用于饱和/全错样本
```
