# OnPolicy / ExpertGym 项目记忆 2026-05-17

更新时间：`2026-05-17 11:42 CST`

## 项目目标

本项目研究 **task vector / expert delta 如何通过少量 calibration data 自动组合**。当前实现路线是：冻结 base model 与 expert delta，只学习 gate/task-vector 系数；用 on-policy rollout 得到当前 policy 在 tool / memory / code 三类任务上的 reward，再通过 GRPO、OPD、retention 等信号更新 gate。

核心论文问题不是“能否 sweep 到一个好系数”，而是：

> 给定多个 expert task vectors 和约 100 条代表性 calibration prompts，能否用 on-policy RL / distillation 信号自动找到一个泛化更好的 task-vector 组合。

## 代码与主要路径

主仓库：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
```

核心训练入口：

```text
scripts/train/opvec_gated_grpo_bake_vllm_loop.py
scripts/train/opvec_update_gates_from_rollouts.py
scripts/train/opvec_collect_vllm_rollouts.py
scripts/data/build_opd_distill_from_expert_rollouts.py
```

主要配置/命令：

```text
configs/gated_grpo.yaml
skill/command/run_qbank_c033333_gate_strategy.sh
skill/command/run_20260517_expH_global_coeff_grpo_opd_ret.sh
```

正式配置记录：

```text
docs/config/
```

正式评测记录：

```text
docs/evaluation/
```

关键报告：

```text
docs/report/0515_zh.md
docs/report/opvec_best_gp_opd_reproduction_20260514.md
docs/report/opd_only_pcguard_abcd_20260515.md
docs/report/20260516_tonight_abc.md
docs/evaluation/20260517_defg_eval6.md
```

## 数据与 reward

当前 calibration 主体是 `paper96`，三任务均衡：

```text
/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
```

每轮 rollout：

- 每个 prompt 采样多个 responses。
- Tool / Memory / Code 都走 `RewardRouter`。
- 训练侧统一使用 `reward_train` 做 GRPO advantage、frontier stats、OPD 正负样本判定。

reward 口径：

- Tool：ToolRL / BFCL 风格 reward，raw 大致在 `[-3, 4]`，训练中映射到 `[0, 1]`。
- Memory：HotpotQA / MemAgent 风格 boxed final answer exact/F1 相关 reward；目前训练 proxy 主要看 boxed exact。
- Code：CURE 相关 reward，测试通过率、public example、syntax fallback；正式评测用 CURE harness 的 LiveBench / LiveCodeBench。

重要审计结论：

- code `success >= 0.95` 与 OPD positive threshold `1.0` 曾被怀疑不一致。
- 对 paper96 的实际样本检查后，`0.95` 阈值不会明显增加 code OPD positive；当前限制 code 的主因更像是 code expert positive 稀疏、calibration proxy 与 CURE heldout 不完全对齐。

## Gate 参数化

当前主要用过：

1. `global-coefficient`：只学 3 个直接 task-vector 系数，tool / memory / code。
2. `global-parameter`：每个 expert 有一个 global + 196 个 residual，共约 `3 * (1 + 196)` 个可学习 gate 标量。
3. `parameter`：每个 module/expert 直接独立学习，不走 common+residual。

重要发现：

- `global-parameter` 可表达力更强，但 common/global 与 residual 结构可能让整体移动更平滑。
- 直接 `parameter` 没有立刻解决问题；首轮 gate 位移很弱，说明瓶颈更多在训练信号方向，而不是参数化本身。
- `global-coefficient` 更适合诊断，因为只有 3 个系数，能直观看 tool/memory/code 的方向。

## Loss 设计当前状态

当前训练目标由三类信号组成：

1. GRPO/frontier：同一 prompt 多个 rollout 之间 reward 有差异时，按 group advantage 更新 gate。
2. Dynamic OPD：当前 policy 对某 prompt 全错、expert 有 positive 轨迹时，对 expert response 做 NLL/distillation，引导 gate 往可恢复能力移动。
3. Retention：对 all-success prompt 做 NLL preservation，防止已经会的行为被破坏。

实现上仍叫 `ppo_loss_weight`，但实际是 GRPO-style group advantage surrogate；不是完整 verl PPO。

关键实现原则：

- `optimizer-step-scope=epoch` 更符合当前 gate 优化目的：一轮 rollout 后累计整体梯度再更新 gate。
- 小 batch 立即 step 容易让早期 batch 方向主导，且会改变后续 batch 的 policy reference。
- OPD / retention 的 NLL 长度差异很大，必须做 length normalization 或 task-balanced scaling，否则 memory/code 长轨迹主导。
- 只做 length normalization 会让总体梯度变小，因此需要 dynamic scale 或 LR 控制整体步长。

## 5 月 14 日最佳实验

参考报告：

```text
docs/report/opvec_best_gp_opd_reproduction_20260514.md
```

它是当前最重要的 positive evidence：OPD 可以强力推动 gate，并在 eval6 上获得较好的综合能力。其意义不只是 reward 高，而是证明了：

- gate 确实可学，不是梯度完全传不过去；
- OPD signal 可以快速把 task vector 系数推离 `1/3`；
- 好的 checkpoint 往往有更合理的 code/memory/tool gate geometry。

但该实验也暴露出风险：如果只是靠 OPD 推，可能会把某个任务方向过推，尤其是 memory-heavy。

## 5 月 15 日实验结论

中文总结在：

```text
docs/report/0515_zh.md
```

核心结论：

- dynamic OPD scale 能解决 loss 幅度问题，但不能解决方向问题。
- 多个实验都能推动 gate，失败不是“学不动”，而是后期净梯度方向偏 memory-heavy。
- 当 memory gate 进入约 `0.60+` 区间后，Tool 常出现崩溃风险。
- retention 是 preservation，不是 task-vector trust region；它不能告诉优化器应该保持哪个 gate geometry。
- PCGuard/PCGrad 第一版可作为多任务冲突 baseline，但不能指望它替代数据与 reward 设计。

最重要的科学判断：

> LR、momentum、OPD target、max delta 只能控制步幅，不能改变方向。论文主线必须围绕“如何构造有效 on-policy training signal”，而不是围绕“如何把系数调到期望区间”。

## 5 月 16 日 ABC / DE / FG 实验

ABC：

- A：从 `1/3` global-parameter 开始，GRPO + OPD。
- B：OPD-only + retention，code OPD augmentation。
- C：B + reasoning expert task vector，reasoning 初始系数 0。

DE：

- D：B 配置 + code augmentation + 只给 memory 开 GRPO。
- E：B 配置 + code augmentation + tool/memory/code 全开 GRPO。

FG：

- F：init=1，GRPO + retention，关闭 OPD。
- G：init=1，GRPO + OPD + retention。

当前 D/E/F/G 正式 eval6 记录：

```text
docs/evaluation/20260517_defg_eval6.md
```

已完成 Tool / Memory：

| 模型 | Tool 均值 | Tool Live | Memory EM | Memory F1 |
| --- | ---: | ---: | ---: | ---: |
| D i20 | 0.7810 | 0.6771 | 0.6016 | 0.7165 |
| E i20 | 0.7848 | 0.6771 | 0.6055 | 0.7325 |
| F i07 | 0.7788 | 0.6875 | 0.6328 | 0.7580 |
| G i08 | 0.7812 | 0.6875 | 0.6367 | 0.7605 |

Code/CURE 正在补齐；旧 D/E/F/G code 评测已过 GPU 生成阶段，仍在 CPU/判题阶段。F/G final iter20 评测已启动，用于和 best/proxy checkpoint 对比。

## Final gate 系数快照

D/E/F/G 的 final iter20 gate 已写入：

```text
docs/evaluation/20260517_defg_eval6.md
```

核心系数：

| 实验 | tool global | memory global | code global | 备注 |
| --- | ---: | ---: | ---: | --- |
| D final i20 | 0.3290 | 0.5460 | 0.3876 | init=1/3，memory 被推高，code 小幅高于初始 |
| E final i20 | 0.3270 | 0.5305 | 0.3872 | 全任务 GRPO，仍 memory/code 上升有限 |
| F final i20 | 0.9948 | 1.0044 | 1.0001 | init=1，GRPO+retention 基本维持强 task-vector 起点 |
| G final i20 | 0.8772 | 0.7287 | 1.0615 | init=1 + OPD 后 tool/memory 被拉低，code 被推高 |

解释：

- F/G 主评测目前选的是 best/proxy checkpoint：F i07、G i08。
- final iter20 系数用于观察训练最终状态，不可直接对应 i07/i08 评测分数。

## 当前运行状态

截至 `2026-05-17 11:42 CST`：

- 旧 D/E/F/G Code/CURE 仍在跑 CPU/判题，summary 未生成。
- F/G final iter20 已启动评测：
  - tmux：`eval_defg_FG20_now_20260517`
  - log：`/tmp/shared-storage/OnPolicy/eval/logs/eval_defg_FG20_20260517.log`
  - F20 用 GPU 0 / port 8110。
  - G20 用 GPU 1 / port 8111。
- H 实验在 GPU 6/7 上跑：
  - run dir：`/tmp/shared-storage/OnPolicy/runs/gated_grpo/expH_gc_c033333_grpo_opd_ret_20260517`
  - 当前约到 iter2 update。
  - 配置：global-coefficient，init=1/3，GRPO + OPD + retention，code augmentation OPD pool。

## 当前主要 failure modes

1. **Proxy 与正式评测不完全一致**
   - paper96 proxy reward 能上升，但 CURE / HotpotQA / BFCL heldout 不一定同步。
   - 论文中必须区分 training proxy 与 official eval。

2. **OPD 后期信号稀疏**
   - 随着 all-fail 变少，OPD 可恢复样本减少。
   - 如果剩下的 all-fail 主要来自 memory，梯度会偏 memory-heavy。

3. **Code 信号最难构造**
   - code prompt 的 expert positive 稀疏，CURE heldout 与训练 proxy 不完全对齐。
   - code augmentation 提供了更多 expert rollouts，但不保证 code reward 显著提升。

4. **Retention 不等价于 trust region**
   - all-success NLL 可以保留行为，但不能保证 gate geometry 保持在多任务最优区域。

5. **调参不能替代训练信号设计**
   - 加大 LR / OPD scale 可以让 gate 快速移动，但如果方向错，会更快崩。

## 论文实验应收束的方向

建议论文主线围绕：

> On-policy expert-vector composition under sparse calibration signals.

也就是：在少量 calibration prompts 下，如何把三类样本转化成有效 gate 学习信号。

必须做清楚的实验组：

1. **Baseline：静态系数 / sweep**
   - equal average / `1/3`
   - TA scale sweep / TIES 等已有合并基线
   - 说明传统方法需要离线 sweep，不能利用 on-policy failure signals。

2. **GRPO-only**
   - 只用 frontier / partial success 的 on-policy reward。
   - 预期：对全对/全错样本利用不足，gate 推动弱。

3. **OPD-only + retention**
   - 只用 all-fail expert recovery + all-success preservation。
   - 预期：能快速推动 gate，是最强的“启动信号”。

4. **GRPO + OPD + retention**
   - 论文主方法。
   - 需要证明：OPD 负责修复 all-fail，GRPO 负责细化 partial-success frontier，retention 负责保住 all-success。

5. **Task-balanced signal ablation**
   - 无 task balance vs task-balanced OPD/retention。
   - 证明不是简单被 memory 长轨迹或某任务样本数支配。

6. **Gate parameterization ablation**
   - global-coefficient
   - global-parameter
   - parameter
   - 目标不是追求最高分，而是解释低维/高维 gate 的稳定性与可解释性。

7. **Initial point ablation**
   - init=`1/3`
   - init=`1`
   - 说明方法既能从弱组合出发学习，也能从强 task-vector 起点做 on-policy repair / retention。

8. **Official heldout eval**
   - BFCL / Tool
   - HotpotQA / Memory
   - CURE / Code
   - 所有 claim 以 heldout eval 为准，proxy 只作为训练动态分析。

## 当前最值得推进的论文叙事

最有希望的故事不是“GRPO 神奇地直接找到 0.75 系数”，而是：

1. task-vector 合并的难点是多任务能力在系数空间里有冲突；
2. 少量 calibration data 里大量样本 reward 饱和或全错，plain GRPO 信号不足；
3. 把样本按 on-policy 状态分成三类后，可以把无梯度样本重新转化为有用信号：
   - all-fail：expert recovery OPD；
   - partial-success：GRPO frontier；
   - all-success：NLL retention；
4. 这种信号比静态 sweep 更像“自动修复当前组合的失败模式”；
5. 但必须加入 task balance / length normalization / heldout eval，否则会出现 memory-heavy proxy overfitting。

## 下一步优先级

1. 等 D/E/F/G 和 F20/G20 official eval 补齐，形成完整表。
2. 对比 F/G：init=1 时 GRPO-only 是否已经足够保持强能力，OPD 是否额外帮助或伤害。
3. 看 H：global-coefficient + GRPO/OPD/retention 是否能得到更稳定的三系数解释。
4. 若 H 不理想，不再盲调 LR；优先改 OPD/GRPO 信号构造：
   - 每任务 OPD after-filter 保持均衡；
   - OPD 只从 expert positive 严格可审查数据来；
   - GRPO frontier 按任务分别统计 advantage scale；
   - retention 记录 per-task contribution。
5. 论文实验表按“方法组件”组织，而不是按日期堆 run。

