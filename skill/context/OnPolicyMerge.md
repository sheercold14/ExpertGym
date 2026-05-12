# OnPolicyMerge 方法与代码框架报告

代码分支：
`/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge`

当前分支：
`on-policy-merge`

报告日期：
`2026-05-08`

## 1. 一句话结论

当前分支的核心方法已经从旧的静态 Expert Merging 推进到 OP-VEC on-policy 系数学习：冻结 base 模型和三个专家相对 base 的 task vector，只训练 task vector 的少量系数；当前合并模型先在校准题上自己采样，再用工具、记忆、代码任务的验证信号给采样打分，筛出同题内有好坏差异的 frontier 样本，用 GRPO/PPO 风格目标更新系数，最后把系数 bake 成标准 HuggingFace checkpoint。

这个框架已经跑通真实模型、真实 expert delta、真实多卡采样、梯度更新、checkpoint bake 和小规模趋势评测。它还没有完成严格泛化证明：Tool/Memory 小集合偏小且部分复用 calibration，Code 小评测只有每类 8 题，完整 Tool/HotpotQA/CURE 评测还没有作为最终结论闭环。

## 2. 方法定位

这条线不是 test-time router，不是输出后处理替换，也不是 LoRA 候选选择器。它优化的是合并模型内部参数化：

```text
theta(alpha) = theta_base + sum_e alpha[p,e] * Delta[p,e]

Delta[p,e] = theta_expert_e[p] - theta_base[p]
e in {tool, memory, code}
```

其中：

- `theta_base`：`/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct`
- `tool expert`：`Qwen2.5-7B-Instruct-ToolRL-grpo-cold`
- `memory expert`：`RL-MemoryAgent-7B`
- `code expert`：`ReasonFlux-Coder-7B`
- `Delta[p,e]` 冻结，只训练 `alpha[p,e]`
- 最终产物是普通模型权重，不需要推理时路由器

它和旧 ExpertMerging 的差别在于：旧主线用 teacher/student 蒸馏或固定系数找静态合并点；OP-VEC 让“当前合并模型自己采样”，只用当前模型生成出的成功/失败差异来更新合并系数。这一点是 on-policy 的核心。

## 3. 参数化设计

当前代码支持四类 gate/系数参数化：

| 参数化 | 含义 | 入口 |
| --- | --- | --- |
| `global` | 一个 common 强度加三个 zero-mean residual | `TorchGateManager` |
| `layer-band` | early/mid/late 每个层段一组 common/residual | `TorchLayerBandGateManager` |
| `parameter` | 每个可合并权重、每个专家一个独立系数 | `TorchParameterCoefficientManager` |
| `global-parameter` | 每个专家一个全局强度，加每个权重的小 residual | `TorchGlobalParameterCoefficientManager` |

当前更合理的主线是 `global-parameter`：

```text
alpha[p,e] = global[e] + residual[p,e]
```

优势是：

- 比 `global` 和 `layer-band` 更细，能表达不同层、不同矩阵的差异。
- 比 588 个完全独立 `parameter` 系数更稳，因为每个专家有一个可整体移动的主强度。
- prior 可以分别约束全局强度和参数 residual，避免小批量校准把所有参数推散。

配置中的主要边界：

```yaml
gate_bounds:
  global_coefficient: [0.00, 1.20]
  parameter_residual: [-0.15, 0.15]
  coefficient: [-0.50, 1.50]
loss:
  global_coefficient_prior_scale: 0.10
  parameter_residual_prior_scale: 1.00
```

## 4. Expert Delta Basis

mode 构建入口：

- `scripts/modes/build_opvec4_modes.py`
- `opvec/modes/build_modes.py`

选择的 Qwen 权重：

```text
model.layers.{0..27}.(self_attn|mlp).(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).weight
```

显式排除：

```text
lm_head / norm / embed_tokens / bias
```

因此 full 配置下：

```text
28 层 * 7 个线性权重 = 196 个可合并权重
196 个权重 * 3 个专家 = 588 个 expert delta 条目
```

已核对的 full mode 产物：

```text
/tmp/shared-storage/OnPolicy/modes/opvec4-full-bf16-real/mode_manifest.json
format: opvec4_mode_manifest_v1
num_params: 196
basis_entries: 588
dry_run: false
```

delta 以 `torch.save` 存成 `expert_deltas/<expert>/<param>.pt`，manifest 记录 `expert`、`param_name`、`storage_path`、shape、dtype。bfloat16 full 目录约 37G。

## 5. 数据与 Rollout

seed manifest 入口：

- `scripts/data/build_seed_prompt_manifest.py`
- `opvec/data/manifest.py`
- `opvec/data/schema.py`

任务被规整为三类：

```text
tool   -> tool_schema / BFCL AST
memory -> hotpotqa_em_f1 / phase-aware memory reward
code   -> code_tests
```

fold0 原始划分已核对：

| split | code | tool | memory | total |
| --- | ---: | ---: | ---: | ---: |
| train | 38 | 36 | 1401 | 1475 |
| heldout | 6 | 9 | 354 | 369 |

这暴露了一个重要问题：原始池 memory 极度占优，不能直接按原始比例进入更新。当前配置用两层手段修正：

- 采样阶段：`task_balanced_sample` 做任务均衡。
- 更新阶段：`frontier_task_quota` 和 `task_loss_weight` 控制进入梯度的样本。

当前默认：

```yaml
calibration:
  frontier_task_quota:
    code: 64
    tool: 64
    memory: 64
  task_loss_weight:
    code: 2.0
    tool: 1.5
    memory: 0.5
```

Memory 还被重新整理成主校准和保留约束：

```text
/tmp/shared-storage/OnPolicy/data/calibration/memory_focused_main.jsonl
  code: 25, tool: 40, memory: 110, total: 175

/tmp/shared-storage/OnPolicy/data/calibration/memory_focused_retention.jsonl
  长 memory chunk 保留样本

/tmp/shared-storage/OnPolicy/data/calibration/memory_focused_main_bal24_seed9191.jsonl
  code: 8, tool: 8, memory: 8
```

HF rollout 入口：

- `scripts/train/opvec_collect_hf_rollouts.py`

关键流程：

1. 加载 base 模型。
2. 构造 gate manager。
3. 按 mode manifest 把选中的 `torch.nn.Linear` 替换成 `GatedLinear`。
4. 当前系数组成临时合并模型。
5. 对每个 prompt 采样多个 response。
6. `RewardRouter` 打分。
7. 记录 `old_logprob`，后续训练用作 PPO/GRPO ratio 的 old policy。
8. 用 frontier filter 标记是否进入 policy loss。

frontier 规则在 `opvec/train/frontier.py`：

```text
success_rate = 同题内成功样本比例
frontier_weight = 4 * success_rate * (1 - success_rate)
```

保留：

- 混合成功/失败的题：强 policy 信号。
- 连续奖励有方差的题：弱但有方向的信号。

丢弃或转队列：

- 全成功：进入 retention，而不是主 policy loss。
- 全失败且无 contract variance：低信息失败。

## 6. Reward 设计

统一奖励形式：

```text
reward = (1 - w) * task_reward + w * behavior_span_reward
w 默认 0.05
```

`task_reward` 是主信号，`behavior_span_reward` 只做弱 shaping，不能压过正确性。

### Tool

实现位置：

- `opvec/rewards/bfcl.py`
- `opvec/rewards/simple.py`

强路径使用 BFCL AST verifier：

- decode 工具调用。
- 检查函数名、参数名、参数值、调用数量。
- 完全正确给 1.0。
- 部分匹配按 matched calls、name recall、count score shaping。

注意：BFCL adapter 依赖官方 BFCL 环境。裸 `python` 环境缺 `tree_sitter` 时，BFCL reward 测试会失败；`conda run -n BFCL ...` 下 102 项单元测试全通过。

### Memory

实现位置：

- `MemoryRewardAdapter`
- `opvec/data/prompt_filters.py`

分两类：

- `final_answer`：抽取 `\boxed{}` 或 answer marker，计算 exact/sub-exact/F1，压缩冗长答案。
- `memory_update`：检查更新文本与参考 memory/answer 的匹配。

这解决的是 HotpotQA/MemoryAgent 的多阶段交互：chunk update 和 final answer 不应该混成一种奖励。

### Code

实现位置：

- `CodeRewardAdapter`

代码奖励已经从“像不像代码”修正为更硬的执行前检查：

- 能否提取 Python code。
- AST 语法是否正确。
- 是否读取 `input()` 或 `sys.stdin`。
- 是否 `print()` 或写 stdout。
- 若题面含公开样例，运行公开样例并比较输出。
- 如果只是硬编码样例输出但不读输入，成功被禁止，奖励封顶到 0.55。

这仍不是完整 hidden-test reward，但比纯格式奖励更能形成组内差异。

## 7. 训练目标

主入口：

- `scripts/train/opvec_update_gates_from_rollouts.py`
- `scripts/train/opvec_train_task_vector_coefficients_grpo.py`
- `scripts/train/opvec_train_task_vector_global_residual_grpo.py`

训练时：

- base 模型参数冻结。
- expert delta buffer 冻结。
- 只有 gate manager 参数参与梯度。
- 对 rollout 中保存的 response，重新计算当前模型下的 differentiable `log p(response | prompt)`。
- 与采样时记录的 `old_logprob` 构造 ratio。

主 policy loss：

```text
adv_i = frontier_weight * (reward_i - mean(rewards)) / std(rewards)
ratio_i = exp(current_logp_i - old_logp_i)
L_policy = -mean(min(ratio_i * adv_i, clip(ratio_i) * adv_i))
```

额外项：

- KL：`sequence_kl_penalty(current_logps, old/reference_logps)`
- prior：约束 gate 或 coefficient 偏离初始值。
- retention：全成功样本只做 KL 保持，防止已经会的题被拉坏。
- optional best-response/pairwise loss：可把清晰正负样本差异放大，但当前阶段一训练仍是 `ppo=1.0`、pairwise/best-response 为 0。

工程上有一个关键优化：

- `opvec/modeling/logprob.py` 只取需要打分的 response token logits，不保留整段 full-vocab logits，避免长上下文下显存爆炸。

## 8. Bake 与评测闭环

bake 入口：

- `scripts/eval/opvec_bake_checkpoint.py`
- `opvec/modeling/bake.py`

功能：

- 读取 mode manifest 和 gate values。
- 生成 `bake_plan.json`、`gate_values.json`。
- streaming 读取 base safetensors shard。
- 对选中参数执行：

```text
W_baked[p] = W_base[p] + sum_e alpha[p,e] * Delta[p,e]
```

- 复制 tokenizer/config sidecar。
- 如果 tokenizer config 缺 Qwen chat template，补一个最小模板。

full eval 入口：

- `scripts/eval/opvec_run_full_eval.py`

会生成或启动三类正式评测命令：

- Tool/BFCL：`skill/Evaluation_all/scripts/run_bfcl_tool_harness.py`
- Memory/HotpotQA：`run_hotpotqa_memory_harness.py`
- Code/CURE：`run_cure_full_harness.sh`

轻量趋势监控：

- `scripts/eval/opvec_subset_trend_monitor.py`

它只用于阶段性防回退，不替代完整评测。

## 9. 已跑通的关键实验

### 9.1 Full basis 与 full 多卡链路

已存在：

```text
/tmp/shared-storage/OnPolicy/modes/opvec4-full-bf16-real/mode_manifest.json
```

核对：

```text
可合并权重: 196
expert delta entries: 588
gate parameterization 支持 parameter / global-parameter
```

full rollout 曾跑通过 `parameter` 与 `global-parameter` 模式，说明 7B base + 588 delta + 多卡分片链路不是 toy。

### 9.2 Memory-focused stage1

rollout：

```text
/tmp/shared-storage/OnPolicy/runs/opvec4/hf_rollouts_global_residual_memory_focused_bal24_s6_tok768_seed9191.jsonl
```

summary：

```text
rows: 24
tasks: code=8, memory=8, tool=8
samples_per_prompt: 6
kept_frontiers: 7
gate_parameterization: global-parameter
parameter_coefficients: 588
reward: task_reward 主导，behavior_span_weight=0.05
```

训练：

```text
/tmp/shared-storage/OnPolicy/runs/opvec4/gate_update_global_residual_memory_focused_bal24_s6_lr4e3_steps2_retention.summary.json
```

核对：

```text
frontier_task_counts: code=5, memory=1, tool=1
retention_rows: 12
updates: 38
gate_grad_nonzero: true
loss: ppo=1.0, pairwise=0.0, best_response=0.0
```

全局强度变化：

| expert | before | after |
| --- | ---: | ---: |
| code | 0.5328 | 0.5645 |
| memory | 0.4835 | 0.5166 |
| tool | 0.4903 | 0.4906 |

196 个权重上的有效系数均值变化：

| expert | before | after |
| --- | ---: | ---: |
| code | 0.5474 | 0.6057 |
| memory | 0.4818 | 0.5213 |
| tool | 0.4887 | 0.4850 |

解释：

- code 被当前模型自己的成功/失败样本继续推高。
- memory 从 code-only 阶段的压低状态恢复并上移。
- tool 基本没有上移，原因不是 tool expert 一定无效，而是本轮只有 1 条 tool frontier；大部分 tool 样本全对或没有有效差异，不能给 GRPO 提供方向。

已 bake checkpoint：

```text
/tmp/shared-storage/OnPolicy/checkpoints/opvec-global-residual-memory-focused-bal24-s6-lr4e3-steps2-retention
checkpoint_frozen: true
gate_parameterization: global-parameter
num_params: 196
num_delta_entries: 588
```

### 9.3 轻量趋势评测

subset summary：

```text
/tmp/shared-storage/OnPolicy/evaluation/subset_trend/20260508_global_residual_bal24_stage1/subset_trend_summary.md
```

| 模型 | Tool 成功率 | Tool 奖励 | Memory 成功率 | Memory 奖励 | Code 候选选择成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7500 | 0.9121 | 0.3750 | 0.7185 | 0.0000 |
| fixed075 | 0.8500 | 0.9458 | 1.0000 | 0.9985 | 0.0000 |
| learned | 0.8500 | 0.9333 | 1.0000 | 0.9985 | 0.0000 |

读法：

- Tool 小集合上 learned 高于 baseline，成功率追平 fixed075，但奖励略低于 fixed075。
- Memory 小集合上 learned 追平 fixed075，明显高于 baseline。
- Code 候选选择指标没有区分度，不能作为代码生成能力结论。

### 9.4 Code 生成执行小评测

随后补了 CURE 单模型生成执行小评测。流程是模型自己生成代码并执行，不是候选 logprob 选择。

| 模型 | LiveBench pass | LiveBench unit pass | LiveCodeBench pass | LiveCodeBench unit pass |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.3500 | 0.4241 | 0.2500 | 0.4967 |
| fixed075 | 0.3500 | 0.4367 | 0.3500 | 0.4901 |
| learned | 0.4000 | 0.4873 | 0.4000 | 0.5629 |

这是当前最有价值的正信号：第一阶段学出的系数在两个代码小集合上都高于 baseline 和 fixed075。但每个集合只有 8 题，只能说明方向值得继续，不足以支撑最终结论。

## 10. 代码框架评价

### 优点

1. 模块边界清晰。`opvec/data`、`opvec/modes`、`opvec/modeling`、`opvec/rewards`、`opvec/train` 分工明确。
2. 合并权重和训练参数解耦。delta 是 frozen buffer，gate manager 是唯一 trainable state，便于审计。
3. 产物可追踪。rollout、replay buffer、gate summary、bake plan、eval plan 都是 JSON/JSONL。
4. 支持从 toy 到 full。测试覆盖 toy gated linear、layer-band、parameter、global-parameter、bake plan、frontier、reward、replay buffer。
5. bake 后是标准 checkpoint，避免推理时引入额外 router。

### 主要风险

1. `README.md` 仍是旧 ExpertMerging 说明，没有反映 OP-VEC 主线。
2. `scripts/train/opvec_train.py` 的 real LLM training 仍只保留 debug-small；真实训练实际拆在 collect/update 两个脚本里，尚未形成一个长程调度器。
3. Tool 的 BFCL reward 强依赖正确 conda 环境；裸 base 环境会因缺 `tree_sitter` 失败。
4. 当前 Tool frontier 太少，stage1 tool global 没被推起来，下一轮必须构造更难、更有部分错误的工具校准题。
5. Code reward 已改进但仍主要是公开样例/局部执行信号，不能替代完整 hidden-test 评测。
6. Tool/Memory 轻量趋势集合样本少，Memory 还复用了 calibration，不能当泛化证据。
7. `docs/opvec_repo_map.md` 的“当前缺口”中同时写了 gated Qwen wrapper 未接入和已接入，文档状态需要清理。

## 11. 测试状态

已运行：

```bash
conda run -n BFCL python -m unittest discover -s tests -v
```

结果：

```text
Ran 102 tests in 8.178s
OK
```

同时验证了裸环境风险：

```bash
python -m unittest discover -s tests -v
```

结果是 100 项通过、2 项 BFCL reward 测试失败。失败原因是当前 base Python 缺 `tree_sitter`，导致 BFCL 官方 AST verifier 无法 import；这不是 gate/training 链路失败，但说明 BFCL reward 和 Tool 相关测试必须固定在 `BFCL` 环境运行。

## 12. 下一步建议

优先级最高的不是扫固定系数，而是提高有效 on-policy frontier 的质量：

1. Tool：从 BFCL 中抽更容易出现部分错误的多函数、多参数题，避免全对样本占满；目标是让 tool frontier 数明显超过 1。
2. Code：继续用生成执行 reward，至少纳入公开样例和小规模 dev tests；不要再用无区分度的候选选择小集合判断代码能力。
3. Memory：短 final/chunk 进主校准，长 chunk 主要做 retention；继续防止 memory 数量压过 code/tool。
4. Training：在保留 PPO/GRPO 主损失的基础上，小权重尝试 pairwise loss，用于放大同题内正负样本排序。
5. Evaluation：固定 fold0 heldout，对比 `fixed0.5`、`fixed0.75`、`learned`；只有 heldout 稳定后，再跑完整 BFCL/HotpotQA/CURE。

当前最准确的研究判断是：OP-VEC 的代码框架和真实模型训练链路已经成立，第一阶段学习系数在小代码生成执行评测上有正信号；但方法是否优于固定经验合并点，还需要更强 frontier 数据和严格 heldout/full eval 才能下结论。
