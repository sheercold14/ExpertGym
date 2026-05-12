# ExpertMerging 核心方法与 eval6 对照

评估报告：
`/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/aggregate_report_zh.md`

代码目录：
`/mnt/cache/wuruixiao/users/lsc/ExpertMerging`

## 1. 核心方法

当前 Qwen 主线是静态多专家 task-vector 合并，不是 test-time router。

核心公式：

```text
theta_merged = theta_base + sum_i alpha_{p,i} * (theta_expert_i - theta_base)
```

- `theta_base`：`Qwen2.5-7b-instruct`
- 专家：ToolRL、RL-MemoryAgent、ReasonFlux-Coder；部分失败尝试还加入 DeepSeek-R1-Distill-Qwen-7B。
- `theta_expert_i - theta_base` 是冻结 task vector。
- 只学习每个参数/参数块上的合并系数 `alpha`，base 和 expert 权重都冻结。
- `alpha` 由 `sigmoid`、`relu` 或 `clipped_relu` 激活得到；支持每个参数一个系数，也支持 `coeffs_size_dict` 分块系数。
- 默认不合并 `lm_head`、`norm`、`embed_tokens`、`bias`。

主要实现位置：

- `Qwen/model_merging.py`：CLI 入口，加载 base/expert，调用 `expert_merging_method`，保存静态 checkpoint。
- `Qwen/parametric_task_vector_model.py`：Qwen 文本模型适配，chat template 输入、token/span mask、teacher/student forward。
- `global_utils/expert_merging_base.py`：通用 ParametricTaskVectorModel 和 Trainer。
- `global_utils/merging_utils.py`：task vector、参数过滤、分块缩放、激活函数。
- `Qwen/dataset.py`：任务到 teacher 的默认映射。`ToolCall/ToolPlan -> ToolRL`，`Memory/MemoryQA -> RL-MemoryAgent`，`Code/Math -> ReasonFlux-Coder`。
- `extract_behavior_spans.py`：抽取 Tool/Memory/Code 的输入 span 和输出行为 span，用于 token-level 加权蒸馏。

训练目标：

```text
L = L_logits + L_hidden(optional) + L_reg
```

- `L_logits`：teacher logits 到 merged student logits 的 soft CE/KL 风格蒸馏，支持 `top_k_logits`。
- `L_hidden`：可选 hidden states MSE，按指定层计算。
- `L_reg`：把实际合并系数约束在 init 附近的 L1 正则。
- token mask 来源优先级：`prefix_loss_tokens_by_task` > response behavior span/input span > 全 token。
- dataloader 会按 teacher/task 做均衡采样。

重要工程限制：

- 当前 student forward 用 `functional_call` 临时构建完整 merged state，会额外占用大量显存。
- `skill/skill_md/expert_merging_optimization.md` 已记录优化方向：改成“原地合并 + 手动计算系数梯度”，避免每 step 复制一份完整 7B 参数。

## 2. 相关辅助尝试

### 2.1 专家 rollout 与 routed 数据

`run_inference.py` 会让不同 expert 跑 ToolRL、HotpotQA、CodeContests 数据，形成 `expert_on_task` 矩阵；`evaluate_results.py` 用规则/verifier 评估：

- ToolRL：格式和 tool_call correctness。
- HotpotQA：boxed answer、EM/sub-EM/F1。
- CodeContests：提取代码后执行样例/测试。

后续 `datasets_1/correct_samples/*`、`datasets_2/correct_samples/*` 里的 `routed`、`routed_nospan`、`routed_bespan`、`routed_1_*` 都是在利用“不同 expert 在不同任务上也可能答对”的样本池。

### 2.2 行为 span

span 目标是把蒸馏压力集中在关键行为区：

- ToolCall：`<tool_call>` 标签、JSON object、tool name、arguments；system schema 和 user content 也可作为 input span。
- Memory：`Updated memory:`、`\boxed{}`、memory/problem/section 边界。
- Code：代码块、`input/print/main`、题面开头和 IO 关键词。

span 类实验通常能稳住 Tool live 或格式行为，但会和 Memory/Code 指标产生 trade-off。

### 2.3 层替换 probe

`Qwen/layer_replacement_probe.py` 逐层替换 host 模型的 `self_attn`/`mlp`，用小样本 verifier 看 donor 层是否有正贡献。

`Qwen/results/layer_probe_2/visualizations/summary.txt` 的要点：

- ToolRL host 上，少数 Memory/Code/Base 层有正增益，例如 RL-MemoryAgent layer 20 self-attn、ReasonFlux-Coder layer 1/5 mlp。
- DeepSeek-R1-Distill-Qwen-7B 在 ToolRL host 上大面积负增益，多个层替换后 primary score 变 0。

这支持后面实验里“不要直接把 DeepSeek 作为普通第四专家全量混入”的结论。

## 3. eval6 代表性尝试对照

下面只列和当前目录方法最相关的几组。指标来自 eval6 中文聚合报告，表中 `Tool` 是四个 BFCL 子集均值，`ToolLive` 是 live 子集均值，`MemF1` 是 HotpotQA 四集合平均 F1。

| run/model | 核心配置 | Tool | ToolLive | MemF1 | CodeAcc | BoN | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `expert-merging-0425-232232` | 3 expert，原始 `datasets`，`max_length=2048`，15/task，init 0.5，sigmoid，reg 0.1 | 0.7927 | 0.6979 | 0.7117 | 0.3418 | 0.4251 | 早期基线；Tool 稳，Memory/Code 中等。重测 CodeAcc 到 0.3543、BoN 0.4310，但 MemF1 降到 0.7024。 |
| `0428-151150` | 3 expert，原始 `datasets`，`max_length=32768`，behavior spans，init 0.5，reg 0.1 | 0.7823 | 0.6771 | 0.7578 | 0.3360 | 0.3831 | 长上下文 + span 明显提高 Memory，但 Tool live 和 Code/BoN 下降。 |
| `0428-203833` | 与 `0428-151150` 基本同配置的复跑 | 0.7927 | 0.6979 | 0.7197 | 0.3555 | 0.4017 | 同类配置波动较大；Code 提高但 Memory 不如 `0428-151150`。 |
| `0503-212409` | 3 expert，`datasets_1/correct_samples/routed`，2048，init 0.5，sigmoid，reg 0 | 0.7952 | 0.6979 | 0.6992 | 0.3367 | 0.4076 | 新增模型里 Tool 均值并列最高，但 Memory/Code 不强。 |
| `0503-231003` | routed，2048，init 0.5，relu，reg 0 | 0.7915 | 0.6979 | 0.7072 | 0.3545 | 0.4086 | relu + no-reg 对 CodeAcc 有帮助，Memory 仍一般。 |
| `0503-220915` | `datasets_back`，32768，behavior spans，sigmoid，reg 0.1 | 0.7810 | 0.6771 | 0.7353 | 0.3389 | 0.3949 | span/长上下文提高 Memory，但 Tool/Code 回落。 |
| `0504-113412` | routed_nospan，2048，init 0.05，`clipped_relu`，lr 0.005，reg 0 | 0.7629 | 0.6458 | 0.5747 | 0.3030 | 0.3441 | 低 init + 大 lr + clipped_relu 明显失败。 |
| `0504-113525` | routed_bespan，32768，span，relu，reg 0 | 0.7823 | 0.6771 | 0.6893 | 0.3365 | 0.4144 | span + relu/no-reg 没有带来稳定收益。 |
| `0505-104944` | routed_nospan，2048，sigmoid，reg 0.1，hidden layers 20-27，hidden weight 1 | 0.7927 | 0.6979 | 0.6921 | 0.3367 | 0.3841 | late hidden-state 对齐没有改善，反而压低 Memory/BoN。 |
| `0505-105335` | routed_nospan，2048，relu，reg 0，hidden layers 20-27，hidden weight 1 | 0.7927 | 0.6979 | 0.7036 | 0.3289 | 0.3812 | hidden-state 对齐同样不理想，CodeAcc 低。 |
| `0505-152950` | 4 expert，加入 DeepSeek，routed，2048，sigmoid，reg 0.1，DeepSeek 未降权 | 0.0000 | 0.0000 | 0.0006 | 0.0000 | 0.0000 | 崩溃型失败。 |
| `0505-153020` | 4 expert，加入 DeepSeek，relu，reg 0，DeepSeek 未降权 | 0.0000 | 0.0000 | 0.0032 | 0.0000 | 0.0000 | 崩溃型失败。 |
| `0505-155232` | 4 expert，DeepSeek init 降到 0.05，relu，reg 0 | 0.7806 | 0.6562 | 0.6956 | 0.3282 | 0.3831 | DeepSeek 降权后能恢复，但整体仍弱于 3 expert 主线。 |
| `0505-171400` | 3 expert，`datasets_2/correct_samples/routed_1`，2048，init 0.5，sigmoid，reg 0.1 | 0.7927 | 0.6979 | 0.7086 | 0.3611 | 0.4164 | 当前 ExpertMerging 尝试里 CodeAcc 较高。 |
| `0505-233556` | 3 expert，routed_1，2048，init 0.75，sigmoid，reg 0.1 | 0.7810 | 0.6771 | 0.7662 | 0.3582 | 0.4242 | 本组里较均衡：Memory 强，CodeAcc/BoN 也高；Tool live 偏低。 |
| `0505-234546` | 3 expert，routed_1_bespan，32768，span，init 0.75，sigmoid，reg 0.1 | 0.7942 | 0.7083 | 0.7575 | 0.3492 | 0.4203 | Tool live 和 BoN 好，Memory 略低于 `0505-233556`。 |
| `0505-234638` | 3 expert，routed_1_span，32768，span，init 0.75，sigmoid，reg 0.1 | 0.7810 | 0.6771 | 0.7627 | 0.3477 | 0.3920 | Memory 尚可，但 Tool/BoN 不如 bespan。 |
| `0506-185713` | 3 expert，routed_1_merge，32768，span，init 0.75，Code prefix loss 2048 | 0.7837 | 0.6875 | 0.7660 | 0.3543 | 0.4076 | Memory 接近 `0505-233556`，CodeAcc 尚可；prefix loss 没有超过 `0505-233556` 的 CodeAcc/BoN。 |

## 4. 和报告整体结果的关系

eval6 报告里，非本目录主线/外部方法的 VGEC 系列更强：

- Tool/BFCL 最高：`vgec-bfclheldout0503-toolbeta-up-down-no-lower-traj` 和 `vgec-bfclheldout0503-toolbeta-up-down-traj`，Tool 0.7954。
- Memory F1 最高：`vgec-nosota-recoverable-traj`，MemF1 0.7750。
- 粗略三类简单平均最高：报告认为是 `vgec-tool-no-lower-codesignsoft-midedge-traj`。

本目录 ExpertMerging 主线的较好点位：

- 偏 Tool：`0503-212409` / `0503-152926`，Tool 0.7952，但 Memory/Code 不够好。
- 偏 Memory + Code 平衡：`0505-233556`，MemF1 0.7662，CodeAcc 0.3582，BoN 0.4242。
- 偏 Tool live + BoN：`0505-234546`，ToolLive 0.7083，BoN 0.4203，MemF1 0.7575。
- 加 Code prefix 的 `0506-185713` 没有明显超过 `0505-233556`，但 Memory 保持较好。

## 5. 结论

1. 当前可复用的核心方法是“学习少量静态 task-vector 合并系数”，最可解释、最稳定的主线仍是 3 expert：ToolRL + RL-MemoryAgent + ReasonFlux-Coder。
2. routed correct samples 比原始 teacher-only calibration 更有价值；init 从 0.5 提到 0.75 后，`0505-233556`、`0505-234546`、`0506-185713` 整体更接近报告中的强结果。
3. behavior span/长上下文对 Memory 和 Tool live 有帮助，但会牺牲部分 Code 或 BoN，需要按目标选择。
4. late hidden-state 对齐没有显示收益；它不像 logits/token span 蒸馏那样稳定。
5. DeepSeek 不能作为普通第四 expert 直接混入。满权重混入导致 `0505-152950`、`0505-153020` 全指标近零；降权到 0.05 后仍弱于 3 expert。
6. 如果继续沿本目录路线，优先从 `0505-233556` 和 `0505-234546` 两类配置出发：前者更均衡，后者更偏 Tool live/BoN。

## 6. 未完全落地但已记录的下一步想法

`skill.md` 里把后续主线概括为“双对齐”：

- 输入侧：在小 calibration 集上对 base、reasoning expert、instruction expert 的关键层 hidden states 做表征对齐，学习静态 merge 系数。
- 输出侧：复用类似 RAIN 的特殊 token 边界建模，用优质轨迹约束 `<think>`、`</think>`、answer 区切换、tool/memory/code 格式行为。
- 最终仍写成静态权重更新，不引入 test-time router。

`skill/skill_md/multi_expert_grpo_strategy.md` 还记录了 RL 方向：expert rollout 只能作为 off-policy replay/SFT/DPO 辅助，真正 GRPO 必须从 merged student 自己采样。
