# OnPolicy / OP-VEC Gated-GRPO Project Memory

记录时间：2026-05-14  
工作目录：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym`  
相关旧仓库：

- 原始/早期主线：`/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge`
- gated_grpo 原型：`/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo`
- 当前整理后工作目录：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym`

## 项目目标

核心目标是用大约 100 条左右的 calibration prompts，通过 on-policy GRPO / OPD 训练少量 OP-VEC gate 系数，让 base model 自动发现 Tool、Memory、Code 三个 expert task vectors 的有效组合。

希望最终故事成立：

1. 不是 sweep 手工找 scaling factor，而是用少量代表性 calibration data 学出 task-vector 组合。
2. 训练不直接改模型全量权重，只学习 task-vector merge gate。
3. Gate 能从 `1/3` task arithmetic 起点，被 reward 信号推到更高能力区间。
4. Tool、Memory、Code 三种能力都不被严重牺牲。
5. 用 formal eval harness 验证，而不只看 qbank proxy reward。

## 关键模型与 task vectors

配置文件：`configs/gated_grpo.yaml`

Base:

- `/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct`

Experts:

- Tool: `/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold`
- Memory: `/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B`
- Code: `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B`

OP-VEC mode manifest:

- `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`

OP-VEC 覆盖模块：

- Qwen2.5 7B 的 28 层
- 每层 7 个线性权重：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
- 共 196 个 gated modules
- 三个 expert task vectors 对应每个位置，因此 full parameter 视角为 196 x 3 = 588 个专家系数位置

重要参数化：

- `global`：`common + tool_residual/memory_residual/code_residual`，历史形态。
- `global-coefficient`：只学习三个直接系数 `tool/memory/code`，没有 common/residual 分解，最可解释。
- `global-parameter`：588 位置的 common+residual 参数化；报告中也称 588 common+residual。
- `parameter`：更细粒度参数化，未作为当前主线。

初始点：

- 当前主线坚持从 `1/3 = 0.3333333333333333` task arithmetic 起步。
- 早期曾讨论 `0.75` 起点，但后来认为应从标准 TA baseline 出发。
- `MAX_COEFF_DELTA` 当前实现是相对每轮输入 checkpoint 的位移限制，不是相对原始 `1/3` 的累计上限。

## 数据和 prompt

### 原始与中间数据

Question bank / qbank root:

- `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1`

100 条旧 calibration:

- `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.prompts.jsonl`

当前 paper96 balanced prompts:

- `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl`
- 构造脚本：`scripts/data/build_paper96_balanced_inputs.py`
- 组成：`tool=32, memory=32, code=32`

固定 OPD compact 数据：

- 原始 compact：`/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill_balanced21_compact_pos1_neg2_seed20260514.jsonl`
- 修正版：`/tmp/shared-storage/OnPolicy/data/calibration/high_info_v1_seed20260511.distill_balanced21_paperfix_rewardtrain_len_seed20260514.jsonl`
- 组成：每任务 7 rows，共 21 rows；每 row 1 expert positive + 2 current negatives。
- 修正点：显式写入 `reward_train=1/0`，重算 `length`，避免 tool OPD 因 raw reward 尺度被跳过，也避免 length-normalized loss 尺度异常。

动态 OPD expert cache：

- Tool: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl`
- Memory: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl`
- Code: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl`
- 每个都是同一 paper96 prompt 子集上的对应 expert rollout，32 rows/task，`samples_per_prompt=2`。

### 数据设计结论

过去问题：

- 有效信号比例低，很多 prompt 全对，不产生 GRPO advantage。
- 全错样本如果没有 expert positive 轨迹，也难以产生直接推动。
- Tool 往往容易饱和，进入 all-success 后 GRPO frontier 减少，tool gate 会被其他任务间接压低。
- Memory trajectory 长，容易在 sequence loss 下主导梯度。
- Code qbank proxy 与 CURE formal eval 并不完全对齐。

当前更合理的数据思路：

1. 先用标准 `1/3` TA baseline 对原始题库 rollout。
2. 按 reward 区间分桶，不只取全错，也取低分但可由 rollout 或 expert 做对的题。
3. calibration data 要覆盖三个能力并且保持 task count 均衡。
4. OPD 数据优先来自同一 prompt 的 expert positive 和当前 policy negative。
5. 对 policy 当前全错但 expert 能做对的 prompt，动态构造 OPD 比固定历史 OPD 更贴近 on-policy 修复目标。

## Reward / rollout / gradient 逻辑

### Rollout

当前 native 主线不是 VeRL，而是本仓库自维护的 bake-vLLM loop：

- 每轮先将当前 gate bake 成普通 HF checkpoint。
- vLLM 从 baked checkpoint rollout。
- rollout 结果写 `rollouts.jsonl`。
- update 阶段用 HF 模型重新计算 old/current logprob 并反传 gate 参数。

入口脚本：

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
- `scripts/train/opvec_collect_vllm_rollouts.py`
- `scripts/train/opvec_update_gates_from_rollouts.py`

为什么不直接用 vLLM old logprob：

- 旧版 old/current logprob 都由 HF update 侧同一 tokenizer/model 计算。
- 新版若 old 来自 vLLM、current 来自 HF，会引入实现差异。
- 当前推荐 `STORE_TOKEN_LOGPROBS=0`，让 update 阶段 `--fill-missing-old-logprob` 保持 HF/HF 一致。

### Reward

原则：rollout reward 尽量对齐官方 evaluator，不使用行为 span 混合，`behavior_span_weight=0.0`。

Tool:

- 使用 ToolRL/BFCL 风格 reward。
- raw reward 可能在 `[-3, 4]`，训练侧映射到 `[0,1]`：`(raw + 3) / 7`。
- Tool expert 当前格式能输出 pythonish tool calls；评测没有大规模空输出/解析失败。

Memory:

- 不能只看 final answer。
- Memory update 需要完整 trajectory，update turns + final turn 都要纳入 logprob / credit assignment。
- MemAgent 论文也使用 trajectory 方式，目标是与 MemAgent credit assignment 对齐。
- 现有 collector 对 memory trajectory 有 `memory_update` 与 `final_answer` 阶段。

Code:

- 当前保持官方 CURE / code reward 体系。
- rollout token 长度要给够：当前 paper96 训练用 `CODE_MAX_NEW_TOKENS=2048`，formal CURE eval 用 `max_generation_token=10000`。
- qbank code proxy 与 CURE heldout 不完全对齐，code proxy 升不必然代表 formal code 增益。

### GRPO / PPO 差异

代码里命名有 `ppo_loss_weight`，但训练逻辑本质是 group-relative reward 的 policy-gradient surrogate，接近 GRPO；并不是完整 PPO trainer。

差异点：

- 一轮 rollout 固定当前 policy 后，在 update 内计算 old/current ratio。
- 当前 native loop 是每 outer iteration rollout 一批，然后 update 一次，不是每个 mini-batch 都重新 rollout。
- group advantage 来自同 prompt 多 samples 的 reward 相对差异。
- all-success / all-failure 本身没有正常 GRPO advantage；当前通过 frontier filtering、retention、OPD 辅助利用。

### Loss granularity

早期 token-level 和 sequence-level 做过对比：

- token-level 梯度更小，长答案不会天然占更大权重，但可能推不动 gate。
- sequence-level 直接 sum logprob，长 trajectory 会主导，尤其 Memory。
- 当前 paper96 实验使用 `LOSS_GRANULARITY=sequence`，同时开启：
  - `LENGTH_NORMALIZE_LOGPROB=1`
  - `LENGTH_NORMALIZE_POLICY_LOGPROB=1`
- 这样保留 sequence-level 的稳定性和可推力，同时降低长回答主导问题。

### Advantage / normalization

当前推荐：

- `TASK_NORMALIZE_ADVANTAGES=0`
- `ADVANTAGE_NORMALIZATION=centered`
- `USE_FRONTIER_WEIGHT=0`

原因：

- 单 prompt 内 centered advantage 保留 GRPO 的相对信号。
- 跨 task 再 normalize 会改变 task 间真实 reward / gradient 比例，早期怀疑它影响了三任务可比性。
- 当前要求每个样本 reward 尺度先统一到 `[0,1]`，而不是 update 后再做 task normalization。

### Retention / KL

Retention 当前实现：

- all-success rows 不参与常规 GRPO frontier，但可作为 retention rows。
- 计算当前 gate 相对 rollout gate 的局部 reverse-KL surrogate。
- 不是固定对 base 或 `1/3` 的全局 KL，也不是严格 VeRL PPO KL 控制。
- 作用是保护当前已经做对的行为，尤其 Tool 饱和后防止被 Memory/Code 梯度压坏。

当前 paper96 设置：

- `USE_RETENTION=1`
- `RETENTION_LOSS_WEIGHT=0.03`
- `MAX_RETENTION_ROWS_PER_TASK=8`
- `MAX_RETENTION_ROWS=24`

### OPD

OPD loss 需要：

- 当前 policy negative samples：同 prompt 下 reward 低于正阈值。
- Expert positive samples：同 prompt 下 expert reward 达到阈值。
- 对 positive 做 best-response NLL / logprob 最大化。
- 可选 pairwise：expert positive logprob 应高于 current negative。

固定 OPD：

- `OPD_LOSS_WEIGHT=0.12`
- `OPD_PAIRWISE_LOSS_WEIGHT=0.06`
- `MAX_OPD_PAIRWISE_PAIRS_PER_ROW=2`
- `OPD_POSITIVE_REWARD_THRESHOLD=1.0`

动态 OPD：

- 每轮 rollout 后，用当前 policy all-fail prompt 匹配离线 expert rollouts。
- 生成 `iter_xxx/opd_distill_from_allfail.jsonl`。
- 当前 D run 中动态 OPD rows 分布明显偏 Tool：
  - iter2: `code=1, memory=3, tool=13`
  - iter3: `code=1, memory=5, tool=10`
  - iter4: `code=4, memory=4, tool=9`
- 这说明 dynamic OPD 确实 on-policy，但需要关注 task imbalance。

## 关键代码改动

重要文件：

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
- `scripts/train/opvec_update_gates_from_rollouts.py`
- `scripts/train/opvec_collect_vllm_rollouts.py`
- `scripts/data/build_opd_distill_from_expert_rollouts.py`
- `scripts/data/build_paper96_balanced_inputs.py`
- `scripts/monitor/opvec_run_monitor.py`
- `skill/command/run_qbank_c033333_gate_strategy.sh`
- `skill/command/run_paper96_threeway_20260514.sh`
- `skill/command/run_paper96_dynamic_opd_gc_20260514.sh`

已做过的主改动：

1. bake-vLLM loop 支持多 GPU rollout shards，并自动 merge，update 只读取合并后的 `rollouts.jsonl`。
2. update 支持 `optimizer_step_scope=epoch`，一轮累计梯度后只 step 一次。
3. update 支持 SGD + momentum、persistent optimizer state 相关参数。
4. update 支持 retention rows 与 per-task retention cap。
5. update 支持 OPD distill rows、best-response loss、pairwise loss。
6. bake-vLLM loop 支持每轮 dynamic OPD builder。
7. launcher 支持 `global-coefficient`。
8. monitor 支持读取 direct `tool/memory/code` global coefficients。
9. per-task token length 已显式暴露：
   - `TOOL_MAX_NEW_TOKENS`
   - `MEMORY_UPDATE_MAX_NEW_TOKENS`
   - `MEMORY_FINAL_MAX_NEW_TOKENS`
   - `CODE_MAX_NEW_TOKENS`
10. 固定 OPD 数据修正了 `reward_train` 与 `length`。

代码变更日志：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/code_change_log.md`

## VeRL 迁移状态

曾 clone / 研究 VeRL，并在本项目中有部分适配：

- `opvec/frameworks/verl_gated_actor.py`
- `scripts/frameworks/opvec_verl_external_lib.py`
- `scripts/frameworks/opvec_verl_gate_actor_smoke.py`
- 文档：
  - `docs/verl_gated_grpo_integration.md`
  - `docs/verl_memagent_reward_rollout_report.md`
  - `docs/verl_migration_feasibility_report.md`

结论：

- VeRL 能提供更规范的 batched rollout/update 框架，但对当前 gate-bake/vLLM/HF-logprob 混合结构迁移成本较高。
- 主要瓶颈不是 vLLM 单次生成本身，而是每轮 bake + rollout + HF update 串行与 update 未完全 batch 化。
- 当前优先优化 native 框架：sharded vLLM rollout、epoch-scope update、动态 OPD。
- VeRL 不是当前晚间主线，后续可以作为工程优化方向继续迁移。

## 重要实验记录

### 早期结论

1. 只用 raw GRPO frontier，从 `1/3` 起点很难快速把 gate 推到 `0.6-0.8` 区间。
2. 小 batch 每 4 条更新一次时，梯度方向常被归一化成接近固定步幅，表现为多个系数一起按 lr 上涨或下降。
3. 改成 epoch-scope 后，一轮内所有 rows 累计梯度，能减少 batch 顺序噪声。
4. 但 no-OPD 条件下，即使 epoch-scope，gate 仍几乎不动。
5. OPD 是当前真正提供非零方向的核心信号。

### 2026-05-14 四路 epoch + OPD 实验

报告：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opvec_fourway_epoch_opd_20260514.md`

主矩阵：

- `gc_noopd`: global-coefficient, OPD off
- `gp_noopd`: global-parameter, OPD off
- `gc_opd`: global-coefficient, OPD on
- `gp_opd`: global-parameter, OPD on

统一设置：

- `NUM_PROMPTS=48`
- `SAMPLES_PER_PROMPT=4`
- `NUM_ITERS=10`
- `OPTIMIZER_STEP_SCOPE=epoch`
- `LOSS_GRANULARITY=sequence`
- `OPTIMIZER=sgd`
- `SGD_MOMENTUM=0.8`
- `PPO_LOSS_WEIGHT=6.0`
- `TASK_NORMALIZE_ADVANTAGES=0`
- `ADVANTAGE_NORMALIZATION=centered`

最终 best:

| run | best rollout | overall | tool | memory | code | best gate | gate tool | gate memory | gate code |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `gp_opd` | iter9 | 0.7189 | 0.9809 | 0.7188 | 0.4570 | `iter_008/gate_updates.gates.json` | 0.2906 | 0.5428 | 0.7117 |
| `gc_opd` | iter8 | 0.7051 | 0.9689 | 0.6406 | 0.5059 | `iter_007/gate_updates.gates.json` | 0.3028 | 0.5244 | 0.6311 |
| `gp_noopd` | iter2 | 0.4082 | 0.5156 | 0.2812 | 0.4277 | `iter_001/gate_updates.gates.json` | 0.3330 | 0.3333 | 0.3333 |
| `gc_noopd` | iter2 | 0.3738 | 0.4008 | 0.2969 | 0.4238 | `iter_001/gate_updates.gates.json` | 0.3334 | 0.3332 | 0.3335 |

关键结论：

- epoch-scope 解决 batch step 噪声，但 no-OPD 不足以推动 gate。
- OPD 让 `gc_opd` 从 overall `0.3741` 连续推到 `0.7051`。
- `gp_opd` proxy 上限更高，但 code 更容易过推，tool gate 低于 `0.30`。
- `gc_opd` 只有 3 个 direct coefficients，可解释性强，更适合讲“少参数自动找组合”的故事。
- formal eval 后发现 `gp_opd` 综合能力更强，尤其 Memory。

### Formal Eval6 结果

已写入：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opvec_fourway_epoch_opd_20260514.md`

评测模型：

- `opvec-gp-opd-best-iter9`
- `opvec-gc-opd-best-iter8`

Tool / BFCL:

| model | live_parallel | live_parallel_multiple | parallel | parallel_multiple | weighted |
|---|---:|---:|---:|---:|---:|
| `opvec-gp-opd-best-iter9` | 0.6875 | 0.6667 | 0.9050 | 0.8750 | 0.8705 |
| `opvec-gc-opd-best-iter8` | 0.6875 | 0.6667 | 0.9050 | 0.8750 | 0.8705 |

Memory / HotpotQA:

| model | avg F1 | avg EM | avg sub-EM |
|---|---:|---:|---:|
| `opvec-gp-opd-best-iter9` | 0.7649 | 0.6387 | 0.7969 |
| `opvec-gc-opd-best-iter8` | 0.7361 | 0.6016 | 0.7637 |

Code / CURE:

| model | avg code_acc | avg code_accumulate_acc | avg unit-test acc | avg unit-test accumulate |
|---|---:|---:|---:|---:|
| `opvec-gp-opd-best-iter9` | 0.3487 | 0.4821 | 0.4730 | 0.5115 |
| `opvec-gc-opd-best-iter8` | 0.3448 | 0.4775 | 0.4309 | 0.4668 |

Formal eval 结论：

- Tool 打平，且格式没有崩。
- `gp_opd` Memory 明显更强。
- `gp_opd` Code 略强，尤其 LiveBench unit-test proxy。
- 当前正式评测主候选是 `opvec-gp-opd-best-iter9`。
- `opvec-gc-opd-best-iter8` 作为低参数三系数 baseline。

Eval artifacts:

- Tool:
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/opvec-gp-opd-best-iter9/tool/summary.json`
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/opvec-gc-opd-best-iter8/tool/summary.json`
- Memory:
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/opvec-gp-opd-best-iter9/eval6-20260502-125748/summary.json`
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/opvec-gc-opd-best-iter8/eval6-20260502-125748/summary.json`
- Code:
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/opvec-gp-opd-best-iter9/eval6-20260502-125748/summary.json`
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/opvec-gc-opd-best-iter8/eval6-20260502-125748/summary.json`

注意：aggregate 文件没有自动 append，这次结果是从各任务 summary 手动汇总进报告。

### Paper96 三路/四路实验

报告：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opvec_paper96_threeway_20260514.md`

启动脚本：

- `skill/command/run_paper96_threeway_20260514.sh`
- `skill/command/run_paper96_dynamic_opd_gc_20260514.sh`

固定三路：

| run | GPU | Gate | OPD | Retention | run dir |
|---|---|---|---|---|---|
| A | 0,1 | `global-coefficient` | fixed 21 rows | on | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_A_gc_opd_i8_20260514_paper96_i8` |
| B | 4,5 | `global-coefficient` | off | on | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_B_gc_noopd_i8_20260514_paper96_i8` |
| C | 6,7 | `global-parameter` | fixed 21 rows | on | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_C_gp_opd_i8_20260514_paper96_i8` |

D 追加：

| run | GPU | Gate | OPD | run dir |
|---|---|---|---|---|
| D | 2,3 | `global-coefficient` | dynamic same-96 all-fail OPD | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/paper96_D_gc_dynamic_opd_i8_20260514_paper96_dynopd_i8` |

共享 paper96 设置：

- `NUM_ITERS=8`
- `NUM_PROMPTS=96`
- `SAMPLES_PER_PROMPT=4`
- `INIT_VALUE=1/3`
- `OPTIMIZER_STEP_SCOPE=epoch`
- `LOSS_GRANULARITY=sequence`
- `STORE_TOKEN_LOGPROBS=0`
- `TASK_NORMALIZE_ADVANTAGES=0`
- `ADVANTAGE_NORMALIZATION=centered`
- `USE_FRONTIER_WEIGHT=0`
- `FRONTIER_ORDER=task-interleaved`
- frontier quotas: 32/32/32
- retention: on, `0.03`, cap 8/task and 24 total
- `PPO_LOSS_WEIGHT=6.0`
- `PRIOR_LOSS_WEIGHT=0.005`
- `MAX_COEFF_DELTA=0.40`
- `LR=0.04`
- `OPTIMIZER=sgd`
- `SGD_MOMENTUM=0.8`

Token limits:

- `MAX_MODEL_LEN=16384`
- `MAX_LOGPROB_TOKENS=12288`
- `TOOL_MAX_NEW_TOKENS=768`
- `MEMORY_UPDATE_MAX_NEW_TOKENS=1536`
- `MEMORY_FINAL_MAX_NEW_TOKENS=768`
- `CODE_MAX_NEW_TOKENS=2048`

Current observed progress at 2026-05-14:

- A manifest has 6 completed updates; iter7 rollout summary exists.
- B manifest has 7 completed updates; iter7 update exists.
- C manifest has 6 completed updates; iter7 rollout summary exists.
- D manifest has 4 completed updates; iter5 rollout summary exists.
- D dynamic OPD builder time is negligible, about `0.4s/iter`; update remains dominant.

Recent D dynamic OPD row counts:

| iter | total OPD rows | task counts |
|---:|---:|---|
| 2 | 17 | `code=1, memory=3, tool=13` |
| 3 | 16 | `code=1, memory=5, tool=10` |
| 4 | 17 | `code=4, memory=4, tool=9` |

Timing pattern:

- 96-prompt sharded rollout on 2 GPUs: usually 360-400s.
- update often 1500-1800s.
- update is still bottleneck.

## Active tmux / monitor state

Current monitor sessions:

- `opvec_monitor_i20_20260513_view`
- `opvec_monitor_epoch_opd_8767_view`
- `opvec_monitor_epoch_opd_0440_view`
- `opvec_monitor_paper96_20260514_paper96_i8`
- `opvec_monitor_paper96_dynamic_20260514_paper96_dynopd_i8`

Current paper96 training sessions:

- `paper96_A_gc_opd_20260514_paper96_i8`
- `paper96_B_gc_noopd_20260514_paper96_i8`
- `paper96_C_gp_opd_20260514_paper96_i8`
- `paper96_D_gc_dynopd_20260514_paper96_dynopd_i8`

Important frontend ports:

- `8765`: older 20iter monitor, likely stale but useful for historical curves.
- `8766`: epoch/OPD four-way monitor.
- `8767`: another epoch/OPD monitor.
- `8768`: paper96 A/B/C monitor.
- `8769`: paper96 A/B/C/D monitor with dynamic OPD D.

SSH tunnel example:

```bash
ssh -L 8769:127.0.0.1:8769 <server>
```

Then open:

```text
http://127.0.0.1:8769
```

## Useful commands

Launch paper96 A/B/C:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
RUN_TAG=20260514_paper96_i8 MONITOR_PORT=8768 \
  bash skill/command/run_paper96_threeway_20260514.sh
```

Launch dynamic OPD D:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
RUN_TAG=20260514_paper96_dynopd_i8 \
BASE_RUN_TAG=20260514_paper96_i8 \
MONITOR_PORT=8769 \
  bash skill/command/run_paper96_dynamic_opd_gc_20260514.sh
```

Attach D:

```bash
tmux attach -t paper96_D_gc_dynopd_20260514_paper96_dynopd_i8
```

Check GPU:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Check run summaries:

```bash
find /tmp/shared-storage/OnPolicy/runs/gated_grpo/<run_name> -maxdepth 2 -name '*summary.json' | sort
```

## Main scientific interpretation so far

The cleanest current story:

1. 标准 TA `1/3` 起点在 qbank 上有能力，但仅靠 raw GRPO frontier，梯度信号很弱。
2. epoch-scope update 是必要工程修复：它避免小 batch 顺序带来的方向抖动。
3. 但是 epoch-scope 单独不够；no-OPD 实验显示 gate 基本不动。
4. OPD 提供“当前 policy 不会、expert 会”的可学习方向，显著推动 coefficients 离开 `1/3`。
5. global-coefficient 只学 3 个系数，能证明低维 gate 可被自动学习；global-parameter formal eval 更强，说明更高容量能提高能力但需要更强约束防过推。
6. Tool 往往在 reward 上很快饱和，因此 tool coefficient 低并不必然表示 tool 能力差；formal BFCL 中 gp/gc Tool 都保持 0.8705 weighted。
7. Memory formal eval 是区分 gp/gc 的主要指标；gp 更强。
8. Code proxy 和 CURE eval 不完全一致，需要在论文中避免把 qbank code reward 直接当泛化能力。

## Risks / open problems

1. Dynamic OPD task imbalance:
   - D run 的 all-fail OPD rows 偏 Tool，可能与“增强 Memory/Code”初衷冲突。
   - 后续可加入 per-task OPD quota 或 all-fail 后再按 task 均衡采样。

2. Cumulative cap:
   - `MAX_COEFF_DELTA` 当前是每轮相对上一 checkpoint，不是相对原始 `1/3`。
   - 如果要避免多轮累计过推，需要实现 global anchor cap。

3. Update bottleneck:
   - vLLM sharded rollout 已较快，96 prompts 两卡约 6 分钟。
   - HF update 仍 25-30 分钟左右，是主耗时。
   - 真正提速需要 batch forward/backward 或更彻底的 VeRL/actor update 改造。

4. OPD strength:
   - `OPD_LOSS_WEIGHT=0.12`、`PAIRWISE=0.06` 能推动 gate，但可能推 code 过高、压 tool。
   - 如果 formal eval 工具能力掉，应降低 pairwise 或增强 retention。

5. Reward proxy:
   - qbank reward 与 BFCL/HotpotQA/CURE formal eval 的一致性有限。
   - checkpoint selection 不能只看 qbank overall。

6. Memory trajectory:
   - Memory update/final trajectory 长度和 credit assignment 仍是最容易出错的部分。
   - 必须继续保持与 MemAgent 官方 trajectory reward 逻辑对齐。

7. Aggregate report:
   - Eval6 本次两个 OPD best 的 aggregate 没自动 append。
   - 已手动写进 `opvec_fourway_epoch_opd_20260514.md`，但若需要正式批处理 aggregate，要再补脚本。

## Cleanliness / repo hygiene

用户强要求：

- 不留垃圾文件。
- 不留临时文件。
- 不留死代码/死文件。
- 数据目录要可回溯、可审查。
- 筛选过程要能通过命令复现。

当前新增重要文档：

- `docs/code_change_log.md`
- `docs/report/opvec_fourway_epoch_opd_20260514.md`
- `docs/report/opvec_paper96_threeway_20260514.md`
- `docs/memory/onpolicy_gated_grpo_project_memory_20260514.md`

当前新增重要脚本：

- `scripts/data/build_opd_distill_from_expert_rollouts.py`
- `scripts/data/build_paper96_balanced_inputs.py`
- `skill/command/run_paper96_threeway_20260514.sh`
- `skill/command/run_paper96_dynamic_opd_gc_20260514.sh`

注意：工作树还有较多历史修改和新增文件，不应擅自 revert。提交前需要用户确认 commit scope。

## Recommended next actions

短期：

1. 等 paper96 A/B/C/D 全部跑完。
2. 对比 A/B/C/D 的 reward 曲线、gate 曲线、OPD row 分布。
3. 若 D 动态 OPD 提升不如固定 OPD，先检查 task imbalance，而不是直接否定 dynamic OPD。
4. 选择 paper96 best checkpoints 送 formal eval。

中期：

1. 实现 cumulative coefficient cap。
2. 实现 dynamic OPD per-task quota / balanced all-fail sampling。
3. 优化 update batch forward/backward。
4. 将 CURE / BFCL / Memory formal eval summary 自动写入统一 aggregate。

论文故事：

1. Baseline sweep/TIES/TA 需要人工找 scaling factor。
2. 我们从标准 TA `1/3` 出发，用少量 calibration + on-policy GRPO/OPD 自动学习 gate。
3. OPD 解决全错/弱信号样本无梯度的问题。
4. Epoch-scope 解决小 batch gate 更新不稳定问题。
5. Formal eval 显示 learned OP-VEC gate 能保持 Tool，并提升 Memory/Code 的综合表现。
