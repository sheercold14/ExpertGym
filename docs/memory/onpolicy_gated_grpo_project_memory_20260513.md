# OnPolicy / Gated-GRPO 项目记忆

状态日期：2026-05-13  
工作目录主线：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym`  
关联旧工作树：`/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo`  
关联评测目录：`/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation`

本文是当前对话和项目推进状态的压缩记忆，用于之后恢复上下文。原始结论仍以代码、训练日志、评测 summary 和报告文件为准。

## 1. 项目目标

最终目标是做一个能够自动学习 task vector 组合方式的 on-policy merging 框架。当前关注的是三类能力：

- Tool：BFCL / ToolRL 风格工具调用能力。
- Memory：MemAgent / HotpotQA 风格多轮 memory update + final answer 能力。
- Code：CURE / LiveBench / LiveCodeBench 风格代码能力。

核心思想不是直接微调整个模型，而是在 base model 与多个 expert task vector 之间学习 gate / coefficient，让模型自动发现每类能力、每层、每个模块、甚至每个参数应该使用多少 expert delta。

当前主要路线：

- 初始点使用 Task Arithmetic baseline，常见起点是三专家平均，即每个 expert delta 初始系数约 `1/3`。
- 用 on-policy rollout 产生候选输出。
- 用官方 reward 或尽量贴近官方训练流程的 reward 评估 Tool / Memory / Code。
- 用 GRPO / PPO-style surrogate 更新 gate coefficient。
- 监控 reward、task vector coefficient、不同任务之间的能力竞争。

研究假设：

- 少量 calibration data 约 100 条，如果构造得足够有信号，应该可以让 gate 自动找到比手工 sweep 更好的 task vector 组合。
- 关键不是无限挖 on-policy 数据，而是构造能代表能力边界、能产生梯度方差方向的数据。
- 如果 raw reward 饱和，需要 self-compare reward：`reward(candidate_gate) - reward(reference_gate)`。

## 2. 代码与管线主结构

主要训练仓库现在以 `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym` 为当前工作目录。旧版和原型在：

- `/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge`
- `/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo`

关键训练脚本和概念：

- `scripts/train/opvec_gated_grpo_loop.py`：原始 native loop，总控每轮 rollout、reward、update、bake。
- `scripts/train/opvec_collect_hf_rollouts.py`：HF rollout 版本，慢但实现直接。
- `scripts/train/opvec_update_gates_from_rollouts.py`：gate update 主逻辑，包含 GRPO/PPO-style loss、prior loss、KL 相关项、coefficient projection。
- `scripts/train/opvec_bake_policy.py` 或等价 bake 流程：根据当前 gate 把 base + task vector 合成为可 rollout / eval 的 policy checkpoint。
- `scripts/data/build_self_compare_advantage_calibration.py`：已有 self-compare 方向代码，可用于把绝对 reward 转成相对参考 gate 的 advantage。
- `skill/command/run_qbank_c033333_gate_strategy.sh`：当前多策略训练入口之一，支持 global / layer-band / global-parameter / parameter 等 gate strategy。

当前 native 优化方向：

- 不急着完全迁移 VeRL。
- 保留 native 框架，但将 rollout / reward / update 更 batch 化。
- 当前已经探索过 token-level loss、sequence-level loss、batch update、多卡拆分 rollout、前端监控。
- 常驻 vLLM server 被认为是长期优化方向，但短期工程复杂度较高，先不作为今晚主线。

## 3. Gate 参数化方式

已讨论和训练的 gate strategy：

- `global`：每个 expert 一个全局系数，外加 common / residual 相关项。可学习参数很少。
- `global-parameter`：按参数类型或参数组共享系数，粒度比 global 更细。
- `layer-band`：按层段共享系数。
- `parameter`：最细粒度，对 588 个模块或更多位置学习独立 gate。

关于参数数量和含义的历史判断：

- 模型模块约 196 个位置；每个位置可能涉及 Tool / Memory / Code 三个 task vector，所以通常会看到 `196 * 3 = 588` 个主要 expert gate。
- 早期疑问是“每个位置应该四个参数吗”。后来明确：实际 meaningful expert direction 主要是三个任务向量；common / residual / base 处理方式取决于具体 parameterization，不一定每个位置都是四个独立专家系数。
- 如果 gate 数量不够，会通过共享策略分配：例如 global 让所有模块共享同一组任务系数，layer-band 在层段内共享。

重要约束：

- `MAX_COEFF_DELTA` 会把 coefficient 限制在初始值附近。
- 从 `1/3` 起步时，如果 `MAX_COEFF_DELTA=0.2`，全局系数理论上大致只能到 `0.533` 附近。
- 如果 `MAX_COEFF_DELTA=0.1`，global-parameter 的均值更难接近 `0.75`。
- 因此，若目标是验证 memory factor 是否应该到 `0.75-1.0`，这个约束会直接限制实验结论。

## 4. 数据与 Reward 记忆

### Tool

Tool 数据最早存在两个问题：

- 有些 prompt 没有 system prompt，导致 tool call 格式能力不能充分触发。
- 初始 `0.75` task vector 模型可能破坏 Tool 输出格式，后来怀疑应该从标准 TA baseline 而不是 `0.75` checkpoint 开始筛选。

Tool reward 希望尽量使用官方 BFCL / ToolRL 风格逻辑，而不是只看文本是否匹配。

### Memory

Memory 是最关键的问题之一。

之前确认：

- 不能只看 final answer。
- 应该使用完整轨迹，尤其是 memory update turns + final turn。
- MemAgent 官方 repo 需要参考其 reward / trajectory credit assignment。
- MemAgent 论文和代码使用 HotpotQA 风格数据；本机有完整 raw 数据：
  - `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/hotpotqa`
- 另一路 high-quality 数据在：
  - `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1`

Memory reward 相关判断：

- 只用 final answer reward 会丢掉 memory update 轨迹上的 credit assignment。
- 如果 update turns 很长，logprob 求和会让 memory 的梯度尺度偏大或偏不稳定。
- native 旧实现把 update turns + final turn 的 logprob 加起来；VeRL 对齐 MemAgent 时也应关注这个 credit assignment。
- 当前需要明确 length normalization、task-normalized advantage、token-level vs sequence-level loss 对 memory 梯度的影响。

### Code

Code 当前可先保持现有 CURE / LiveBench / LiveCodeBench reward。

已知风险：

- Code 和 Memory / Tool 在 gate 上竞争。
- 在不同 coefficient 组合下，样本可解性会变化。
- 全错样本不能完全丢弃，但应该和“偏低但至少有 rollout 能做对”的样本区分使用。

## 5. Calibration Data 构造思想

当前最重要的数据筛选思路：

1. 找原始数据集和官方 raw 数据。
2. 找标准 TA baseline checkpoint，不要只用 `0.75` checkpoint。
3. 用 vLLM rollout 评测 baseline。
4. 用官方 reward 统计每条样本的 reward 分布。
5. 不只取全错样本，而是按 reward 区间分桶：
   - 全对样本信号弱，应少取。
   - 全错样本保留少量，用于判断能力下限和可行性。
   - 重点取 reward 偏低但有某些 rollout 能做对的样本。
   - 对不同任务维持比例平衡，避免 Tool / Code / Memory 某一类 dominate。
6. 构建可回溯、可审查的题库。
7. 之后在题库上构造 calibration data。

训练阶段可使用 self-compare reward：

```text
delta_reward = reward(candidate_gate) - reward(reference_gate)
```

它回答的问题不是“这个输出是否绝对正确”，而是“candidate gate 是否比 reference gate 更好”。这可以让 raw reward 已经饱和的样本重新产生优化信号。

## 6. VeRL 迁移状态

曾经尝试将流程迁移到 VeRL，目标是利用其 rollout / logprob / GRPO 框架加速训练。

结论：

- VeRL 理论上适合这类 RL pipeline。
- 但我们的 gate training 不是普通 LoRA/full-parameter RL，需要把 gate coefficient 翻译成 baked model 或动态权重组合。
- 如果每轮都 bake，再交给 VeRL rollout，速度收益会被 bake / load 抵消。
- 如果想严谨高效，需要类似“常驻 vLLM server + 每轮同步 gate checkpoint / 权重 delta”的机制，工程量较大。
- 也探索过 vLLM 单独占卡、训练单独占卡的方案，但显存和通信设计复杂。

当前结论：

- 短期先优化 native。
- 长期可以参考 VeRL 的 batch rollout、batch reward、batch logprob/update 组织方式。
- 如果未来迁移，应写清楚：
  - rollout 轨迹格式；
  - reward 如何计算；
  - old_logprob/current_logprob 来源；
  - token-level / sequence-level loss 对齐；
  - memory update turns + final turn 的 credit assignment。

## 7. Native 训练关键设置

常见环境：

- BFCL 环境：`/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python`
- easyrl 环境也曾用于 VeRL 调试，但 BFCL 更接近当前 native 依赖。

常见训练变量：

```bash
UPDATE_EPOCHS=1
UPDATE_BATCH_SIZE=4
BATCH_LOSS_REDUCTION=mean
LOSS_GRANULARITY=token
FRONTIER_ORDER=as-is
FRONTIER_SHUFFLE_SEED=
STORE_TOKEN_LOGPROBS=auto
TASK_NORMALIZE_ADVANTAGES=1
LENGTH_NORMALIZE_POLICY_LOGPROB=1
GRADIENT_CHECKPOINTING=1
MAX_MEMORY_PER_GPU=70GiB
CPU_MAX_MEMORY=180GiB
DRY_RUN=0
```

历史判断：

- `LOSS_GRANULARITY=token` 更接近 token-level PPO / GRPO，但梯度 norm 可能比 sequence-level 小。
- sequence-level 会让长答案或长轨迹对梯度影响更大。
- token-level 与 sequence-level 最终 gate 更新可能相近，因为 optimizer、clipping、projection 会改变表观 grad norm 和实际 step 的关系。
- `TASK_NORMALIZE_ADVANTAGES=1` 很重要，用于降低 Tool / Memory / Code reward 尺度差异对 gate 的偏置。
- `PRIOR_LOSS_WEIGHT=0.01` 控制 gate 不要过度偏离初始或 prior 分布，太大可能抑制 task vector 增长。
- `beta_kl=0.02` 属于温和 KL loss；如果不开，模型可能更自由地追 reward，但 gate 可能更不稳定。
- surrogate ratio / clipping 不等价于真正 KL 控制；clip 只是限制 policy ratio 的局部目标，不保证分布整体不漂。

## 8. 多卡与速度

已知性能瓶颈：

- 原始 smoke 慢，主要因为 HF rollout / 每轮重新加载 / 串行流程。
- vLLM 可以明显加速生成，但 gate 模型需要 bake / load，无法像普通固定模型那样长期常驻。
- 目前 rollout 阶段曾出现只占一张卡的问题，因此提出多进程 vLLM rollout：
  - 根据 GPU 数量和样本数量自动切 shard。
  - 每张卡独立 rollout。
  - 最终 merge 成一个 JSONL。
  - update 阶段只读取合并后的主文件，不读取子文件。

用户偏好：

- 代码要可维护。
- 自动根据 GPU 数量和 sample 数量分配进程。
- 不要留下临时 shard 文件或死文件；如果必须保留，必须组织清楚、可回溯。

## 9. 前端监控

已有前端 monitor：

- URL：`http://10.119.31.17:8765`
- 本地 tunnel 时：`http://127.0.0.1:8765`

用途：

- 同时展示多个 run。
- 用 `run_id` 区分页面。
- 展示每条样本 reward。
- 展示整体 task vector coefficient。
- 展示训练曲线，最好接近论文可用图。

已知问题：

- 早期前端看不到 gated 系数变化，原因通常是 summary / gate checkpoint 解析不完整或没有读取正确 run_dir。
- 多实验同时跑时，需要确保 monitor 指向所有 run 的真实目录。

## 10. 已完成的四组 20-Iter 实验

四组实验已完成并审查：

- `global`：`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_i20_20260513_010649`
- `layer-band`：`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_layer_band_i20_20260513_010649`
- `global-parameter`：`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_parameter_i20_20260513_010649`
- `parameter`：`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_parameter_i20_20260513_010649`

审查结论：

- 每个 run 有 `strategy_summary.json`。
- 每个 run 有 `gated_grpo_bake_vllm_loop_manifest.json`。
- 每个 run 有 20 个 rollout summaries 和 20 个 gate update summaries。
- 每个 run 有最终 `iter_020/gate_updates.gates.json`。
- 日志中未发现 `Traceback`、`CUDA out of memory`、`No kept frontier`、`Killed`、`Exception`、`Error` 等关键错误。

`global` 最终有效系数大致：

- tool：早期从 `0.3667` 到峰值约 `0.4226`，后降到 `0.3154`。
- memory：升到 `0.4826`。
- code：升到 `0.4184`。
- common：约 `0.4055`。

`global-parameter` iter20 均值大致：

- tool mean：`0.3254`
- memory mean：`0.4348`
- code mean：`0.4823`
- max：tool `0.4154`、memory `0.5141`、code `0.5255`。

重要解释：

- task vector 没有持续学到 `0.75` 区间，不一定是训练失败本身，因为当前 objective 并没有显式把 `0.75` 当目标。
- 但结合 TA scale sweep，Memory 的最优 factor 明显高于当前 learned gate，因此当前训练确实没有把 Memory vector 推到合适幅度。

## 11. Eval6 评测结果

标准评测报告已写入：

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/eval6_gated_grpo_global_vs_global_parameter_20260513.md`

### Global

模型：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_i20_20260513_010649/iter_020/baked_policy`

Tool / BFCL：

- live_parallel：`0.6875`
- live_parallel_multiple：`0.7083`
- parallel：`0.9100`
- parallel_multiple：`0.8750`
- mean：`0.7952`

Memory / HotpotQA：

- eval_50 F1：`0.6588`
- eval_100 F1：`0.7173`
- eval_qa_1_32768 F1：`0.6708`
- eval_qa_1_65536 F1：`0.7119`
- mean F1：`0.6897`
- mean EM：`0.5566`

Code / CURE：

- LiveBench Code Acc：`0.4004`
- LiveCodeBench Code Acc：`0.3068`
- Code mean Acc：`0.3536`
- BoN mean：`0.3978`

### Global-Parameter

模型：

`/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_parameter_i20_20260513_010649/iter_020/baked_policy`

Tool / BFCL：

- live_parallel：`0.6875`
- live_parallel_multiple：`0.7083`
- parallel：`0.9150`
- parallel_multiple：`0.8750`
- mean：`0.7965`

Memory / HotpotQA：

- eval_50 F1：`0.6973`
- eval_100 F1：`0.7327`
- eval_qa_1_32768 F1：`0.6615`
- eval_qa_1_65536 F1：`0.7201`
- mean F1：`0.7029`
- mean EM：`0.5781`

Code / CURE：

- full summary 当时缺失。
- LiveBench 部分存在：
  - Code Acc：`0.3770`
  - Code Accumulate：`0.4851`
  - UnitTest Acc：`0.3660`
  - UnitTest Accumulate：`0.4024`
  - BoN Acc：`0.3828`
  - BoN Accumulate：`0.5122`
- LiveCodeBench 未完整完成，因此不能和 Global 做完整 Code 对比。

## 12. TA Scale Sweep 关键结论

报告路径：

`/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/ta-scale-sweep-20260502-232830/ta_scale_sweep_report_zh.md`

Task Arithmetic 公式：

```text
theta_c[p] = theta_base[p] + c * sum_i(theta_expert_i[p] - theta_base[p])
```

关键总表：

| c | Tool mean | Memory F1 | Code Acc |
|---:|---:|---:|---:|
| 0.10 | 0.7512 | 0.5459 | 0.2891 |
| 0.50 | 0.7810 | 0.7244 | 0.3533 |
| 0.75 | 0.7850 | 0.7588 | 0.3585 |
| 1.00 | 0.7644 | 0.7667 | 0.3423 |
| 2.00 | 0.2144 | 0.6166 | 0.0000 |

Memory 明细：

- `c=0.50`：Memory F1 `0.7244`
- `c=0.75`：Memory F1 `0.7588`
- `c=1.00`：Memory F1 `0.7667`
- `c=2.00`：Memory F1 `0.6166`，明显过冲。

当前重要修正：

- 不能说 “0.5 scale memory 也还行，所以 factor 不是问题”。
- 更准确的说法是：`0.5` 已经可用，但 Memory 的最优区间在 `0.75-1.0`，当前 learned gate 的 memory 系数没有推到这个区间。
- 当前 learned global / global-parameter 的 Memory F1 还低于 `c=0.5` TA baseline，因此不是只差一点，而是训练目标没有把 memory 的有效幅值学出来。

## 13. Memory F1 低的具体原因

之前检查 memory 结果：

- `boxed_found=128/128`，说明不是答案提取格式失败。
- 错误多为语义/entity selection 错误，而不是没有输出 boxed answer。

典型错误：

- Gold `Chief of Protocol`，pred `United States ambassador`。
- Gold `Pedro Rodríguez`，pred `Sergio Pérez`。
- Gold `keyboard function keys`，pred `universal remote`。
- Gold `Fujioka, Gunma`，pred `Japan`。
- Gold `Yellowcraig`，pred `Yellowcraigs`。
- Gold `Marion, South Australia`，pred `Adelaide` 或 `Melbourne`。

这说明：

- 模型具备基本 HotpotQA 回答格式。
- 主要问题是检索 / 多跳实体绑定 / memory update 后的信息选择精度不足。
- 如果 Memory task vector 幅值不足，模型会偏 base/general QA，而不是充分使用 memory expert 的 trajectory 行为。

## 14. 为什么当前 learned gate 没有学到 Memory 最优区间

综合判断：

1. `MAX_COEFF_DELTA` 约束太紧，直接阻止 coefficient 到 `0.75-1.0`。
2. raw reward 对很多样本已经饱和，不能提供继续增大 memory factor 的梯度。
3. calibration data 中有效 memory frontier 比例不足。
4. Tool / Code / Memory 多任务竞争，后期 code / memory / tool 的 frontier 数量不平衡。
5. 当前 objective 是 on-policy 局部提升，不是匹配 TA scale sweep 的全局最优 factor。
6. 没有用 `c=0.5/0.75/1.0` 作为 reference / candidate 来显式构造“memory factor 增大是否更好”的比较信号。

## 15. 下一步最关键实验

优先级最高的不是继续盲跑更多 iter，而是构造能够把 Memory factor 推到正确区间的实验。

建议实验：

1. 固定参考 gate：
   - reference 可以是 `c=0.5` TA baseline。
   - 或当前 learned global/global-parameter。
2. candidate 允许 memory gate 探索 `0.75-1.0`。
3. 使用 self-compare reward：

```text
advantage = reward(candidate_gate) - reward(reference_gate)
```

4. Memory calibration 样本优先选：
   - `c=0.5` 不稳或错；
   - `c=0.75/1.0` 明显更好；
   - 当前 learned gate 不好；
   - Memory expert 能恢复。

5. 按任务平衡：
   - Memory-targeted 实验中可以临时提高 Memory 样本权重。
   - Multi-task 实验中需要防止 Tool 或 Code 被牺牲。

6. 放宽或关闭 coefficient delta 约束：
   - 至少允许 Memory 从 `1/3` 到 `0.75-1.0`。
   - 可以对 Tool / Code 保持更小约束，对 Memory 单独放宽。

## 16. 方法故事可能怎么讲

如果未来实验成功：

- 手工 TA sweep 能找到每个任务粗略最优 factor，但成本高，而且不同任务之间最优点不一致。
- 我们的方法用少量 calibration data 和 on-policy self-compare reward，自动学习 task vector 的组合。
- gate 可以从粗粒度 global 到细粒度 parameter，逐渐提高表达能力。
- calibration data 不是普通训练集，而是用于发现 task vector 组合方向的能力探针。
- self-compare reward 解决 raw reward 饱和问题，让“已经能做对”的任务仍然有区分不同 gate 的信号。

如果实验只接近 TA baseline：

- 仍可讲 automatic coefficient discovery / tradeoff preservation。
- 但必须承认当前不是强 RL improvement，而是 sample-efficient task-vector calibration。

当前不能过度声称：

- 不能说已经超过所有 hand-tuned TA sweep。
- 不能说 learned gate 已自动找到 Memory 最优，因为现有结果没有支持。
- 不能把 coefficient 不到 `0.75` 单独解释为失败，必须结合 reward curve、约束、calibration 信号分析。

## 17. Repo Hygiene 约束

用户明确要求：

- 不许在代码库里留垃圾。
- 没有临时文件。
- 没有死代码。
- 没有死文件。
- 没有无意义文件夹 / 子文件夹。
- 数据结构要组织化、可回溯、可审查。
- 筛选过程要能用命令清晰复现。

当前记忆文件本身是有意保留的长期文档，路径：

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/memory/onpolicy_gated_grpo_project_memory_20260513.md`

## 18. 已知重要报告

- GRPO gate 梯度动力学：
  - `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/grpo_gate_gradient_dynamics.md`
- Reward 与梯度流报告：
  - `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/reward_gradient_flow_report.md`
- Native batch training contract：
  - `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/native_batch_training_contract.md`
- Native batch progress audit：
  - `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/native_batch_progress_audit.md`
- VeRL memory reward / rollout report：
  - `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/verl_memagent_reward_rollout_report.md`
- Eval6 learned gate 报告：
  - `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/eval6_gated_grpo_global_vs_global_parameter_20260513.md`
- TA scale sweep 报告：
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/ta-scale-sweep-20260502-232830/ta_scale_sweep_report_zh.md`

## 19. 当前最短行动建议

下一轮不要直接大规模跑四个 strategy。建议先做一个 Memory-targeted 小实验：

1. 从 `c=1/3` 或 `c=0.5` reference 出发。
2. 只选 HotpotQA 中 `c=0.5` 与 `c=0.75/1.0` reward 有明显差异的样本。
3. 开 self-compare reward。
4. 放宽 Memory coefficient 上限到至少 `1.0`。
5. 暂时固定或弱更新 Tool / Code，避免任务竞争干扰判断。
6. 观察：
   - Memory coefficient 是否能从 `1/3` 或 `0.5` 推向 `0.75-1.0`。
   - Memory validation F1 是否接近 TA `c=0.75/1.0`。
   - Tool / Code 是否显著下降。

如果这个小实验都推不上去，问题大概率在 reward / logprob / advantage / projection 的梯度链路，而不是数据规模。

