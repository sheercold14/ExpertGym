# RCRF 方法蓝图：从残差证据到能力 Pareto Frontier

## 1. 当前问题

原始目标不是找到一个能在 calibration 上涨 reward 的 gate，而是建立一套能泛化的能力归因框架：当 Tool / Memory / Code 三类 RL expert 的 task vector 同时合并时，框架需要解释哪些 residual 在表达能力，哪些 residual 在制造冲突，以及如何在模式组合中选择合理 operating point。

之前的 scalar gate / GRPO 更新有两个核心问题：

1. gate 系数和真实能力不是稳定单调关系，尤其 Code 能力不只由 code expert 系数决定。
2. calibration reward 容易饱和或偏移，无法稳定区分“能力 residual”和“行为破坏 residual”。

因此当前 RCRF 线转向训练外的 residual-level evidence：直接分析每个 OP-VEC residual entry 对不同任务关键 span 的 utility / harm，而不是只学习全局 task coefficient。

## 2. 基本单位

RCRF 的最小决策单位是：

```text
(param_name, expert)
```

当前三专家 OP-VEC 有：

```text
28 layers x 7 linear modules x 3 experts = 588 residual entries
```

每个 entry 都可以被独立判断：

- 是否支持 Code pass trajectory；
- 是否更像 Code fail trajectory；
- 是否支持 Tool call behavior；
- 是否伤害 Tool call behavior；
- 是否支持 Memory full trajectory；
- 是否伤害 Memory full trajectory。

这个单位比 task scalar 更细，也比 full parameter training 更可审查。

## 3. 证据构造

### 3.1 Code：same-prompt pass/fail contrast

Code 不能只模仿 expert positive response。更有效的信号是同一个 prompt 下：

```text
pass trajectory residual utility - fail trajectory residual utility
```

这对应 Code 的 improvement direction：

- pass/fail contrast 为正：可以提高该 residual；
- pass/fail contrast 为负：应该抑制该 residual；
- 多个 Code source 冲突：保守处理。

使用数据：

- LiveBench hurt16
- LiveCodeBench hurt16
- prompt / reasoning / code span contrast

### 3.2 Tool：tool-call behavior span

Tool 能力主要是结构化调用行为，不是长推理轨迹。Tool harm 更像格式/调用行为约束。

当前证据显示 Tool call-span protection 能稳定保持：

```text
parallel / parallel_multiple / live_parallel / live_parallel_multiple
```

因此 Tool harm 更适合 hard 或 near-hard constraint。

### 3.3 Memory：full trajectory span

Memory 不能只看 final answer。HotpotQA / MemAgent 类任务的关键行为在：

```text
update turns + final answer
```

v8 证明这一点：

- 用 full trajectory protection 后，Memory eval_50 F1 从 v7 的 `0.7425` 恢复到 `0.7720`。
- 但 hard veto 会压制 Code repair，说明 Memory trajectory 与 Code reasoning 共享一部分 residual。

因此 Memory harm 更适合 soft constraint。

## 4. Routing 规则

RCRF 不再把 residual 二值化为“保留 / 删除”，而是做多目标 routing：

```text
new_gate = base_gate + code_improvement_delta
```

其中：

```text
code_improvement_delta = f(Code pass/fail contrast)
```

然后用 Tool / Memory 证据调节这个 delta：

```text
if residual harms Tool behavior:
    positive Code delta -> hard or near-hard veto

if residual harms Memory trajectory:
    positive Code delta -> soft scale

if residual supports Tool/Memory behavior:
    negative Code delta -> floor
```

当前实现中有三个 operating point：

| point | rule | 作用 |
|---|---|---|
| v8 | Tool+Memory harm 全 hard | 最强保护，Code 被压制 |
| v9 | Tool+Memory harm 全 soft 0.5 | 当前最均衡 |
| v11 | Tool hard, Memory soft | 行为保护更强，Code 较弱 |

## 5. 当前 Pareto Frontier

| candidate | rule | Tool quick | Memory eval_50 F1 | LiveBench hurt BoN | LiveCodeBench hurt BoN |
|---|---|---|---:|---:|---:|
| v8 | all hard | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7720 | `0.2500 / 0.4688` | `0.3125 / 0.5378` |
| v9 | all soft 0.5 | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v10 | naive ratio | `0.880 / 0.860 / 0.7500 / 0.625` | 0.7495 | `0.2500 / 0.6016` | `0.3125 / 0.5378` |
| v11 | Tool hard, Memory soft | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7701 | `0.2500 / 0.5469` | `0.3750 / 0.5126` |
| v14 | v9, code expert x0.5 | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7774 | `0.2031 / 0.2500` | `0.2500 / 0.1875` |
| v15 | v9, code expert =0 | `0.880 / 0.865 / 0.7500 / 0.625` | 0.7841 | `0.0781 / 0.1250` | `0.1719 / 0.1875` |
| v16 | source-conflict negative dominant suppress | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7660 | `0.1094 / 0.2500` | `0.3281 / 0.4375` |
| v17 | source-conflict dominant route | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7654 | `0.1250 / 0.3125` | `0.2188 / 0.4375` |

结论：

- v9 是当前最均衡 operating point。
- v8 / v11 更像 behavior-preserving operating point。
- v10 是反例：naive evidence ratio 过强压制 Code，也没有保护好 Tool live_parallel。
- v14 / v15 是关键负例：全局压低 Code expert 可以释放 Memory，但会直接摧毁 Code hurt 能力；因此 Code 不能按 task scalar 置零或整体缩小，必须做 residual-level 结构化选择。
- v16 / v17 是 role-routing 负例：source-conflict 的离散 dominant-source rule 能保护 Tool/Memory，但仍不能恢复 v9 的 Code；当前最合理主线是连续 Code pass/fail overlay + Tool/Memory behavior constraints，而不是把 atlas 离散成少数 hard roles。

## 6. 论文 Claim

当前最稳的 claim 不是“某个 gate 单点 SOTA”，而是：

> RL expert residuals are not task-pure. Their utility and harm are outcome-aware and span-conditioned. A residual-level attribution framework can expose a Pareto frontier between capability repair and behavior preservation, enabling principled expert task-vector composition beyond scalar coefficient search.

中文表述：

> RL expert 的 task vector 不是纯任务能力包。能力 residual 的价值依赖任务、输出 span 和成功/失败结果。RCRF 通过 residual-level utility/harm attribution，把模式冲突从黑箱 gate 调参转化为可审查的 Pareto routing 问题。

## 6.1 当前方法收束

v13/v16/v17 证明：把 atlas role 离散成 hard routing rule 足够保护 Tool/Memory，但不能恢复 v9 的 Code。对比脚本显示：

- v9 改 `205` 个 residual row，正负 delta 近似均衡。
- v17 虽然改 `146` 个 row，但仍丢掉 v9 的 `110` 个 delta，并出现 `11` 个 sign mismatch。
- 丢失最多的是 `code_source_conflict*`、`uninformative`、`code_negative_noise`。

因此当前主方法应收束为：

```text
Residual Capability Field with Behavior Constraints
```

而不是：

```text
atlas role -> hard action
```

具体地：

1. Code 用 continuous pass/fail overlay 建立能力证据场。
2. Tool 用 tool-call behavior span 作为 hard / near-hard constraint。
3. Memory 用 full trajectory span 作为 soft constraint。
4. Atlas 的作用是解释、审计和做消融，不是替代连续证据场。

已将主方法语义化为 `v18_rcf_bc`：

```bash
PHASE=generate CANDIDATES=v18_rcf_bc \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

`v18_rcf_bc` 与原 v9 的 588 个 gate 数值完全一致：

```text
max_abs_diff = 0.0
different_count = 0
```

因此论文中应使用 `RCF-BC` 作为方法名，`v9` 只作为历史实验编号。

## 6.2 Attribution Ledger：从解释到可执行协议

为了避免把 `clean_code_repair`、`code_negative_noise` 这类 proxy label 当作因果结论，当前框架新增逐 residual row ledger：

```text
docs/report/RCRF/20260522_rcrf_attribution_ledger.md
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_attribution_ledger_20260522/
```

ledger 对 588 个 row 统一记录：

- `decision`：当前证据下的机制判断；
- `routing_action`：该 row 在组合时应该如何使用；
- `validation_priority`：下一轮是否优先做 isolated shrink/restore 或 behavior audit；
- `next_validation`：下一步需要通过的验证口径。

当前 action 分布：

| action | rows | interpretation |
|---|---:|---|
| `retain_capability_delta` | 77 | 干净 Code 能力 residual |
| `retain_continuous_field` | 279 | mixed source/span evidence，不能 hard route |
| `retain_with_behavior_constraint` | 28 | Code 能力与 Tool/Memory 行为冲突，需要软约束 |
| `protect_behavior_anchor` | 66 | Tool/Memory behavior anchor |
| `do_not_prune_without_counterfactual` | 60 | 看似 weak/noisy 的 Code row，剪之前必须反事实验证 |
| `behavior_guard` | 31 | 行为负约束或 veto |
| `keep_small_until_validated` | 18 | 低置信小 delta |
| `hold_base` | 29 | 暂时保持 base |

这一步把 RCF-BC 从一个 gate 公式扩展为 protocol：

```text
residual evidence
-> row-level decision
-> routing action
-> counterfactual validation
-> paper-facing claim
```

同时新增 validation planner：

```text
docs/report/RCRF/20260522_rcrf_validation_plan.md
```

它把 ledger 聚合成 48 张验证卡片，并给出 P0 队列。每张卡明确一个机制假设、成功判据、失败解释和代表 residual keys。后续实验应该优先从这些卡片生成最小 isolated gate intervention，而不是再做全局 expert 系数调参。

当前已经把 P0 card 物化为可 bake 的 gate probes：

```text
docs/report/RCRF/20260522_rcrf_validation_interventions.md
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/validation_card_interventions_20260522/
```

这些 probes 的定位是机制验证，不是新主方法。它们用于回答：

- early-layer `code_source_conflict` 的连续 delta 是否真有用；
- Memory expert 中 `code_repair_with_behavior_harm` 行是否构成 Pareto boundary；
- behavior anchor 被压低时 Tool/Memory 是否会掉。

因此下一阶段不应继续问“code expert 整体该多大”，而应问：

1. 哪些 `retain_capability_delta` row 是稳定 Code 能力？
2. 哪些 `retain_with_behavior_constraint` row 是 Pareto frontier 的关键冲突点？
3. 哪些 `do_not_prune_without_counterfactual` row 虽然 proxy 低置信，但实际承载 behavior support？

这也解释了 v14/v15 的负例：全局压低 Code 能释放 Memory，但它跨过了 row-level attribution，直接破坏 Code 必需 residual。

## 7. 可复现命令骨架

### 7.1 Memory full trajectory manifest

```bash
PYTHONDONTWRITEBYTECODE=1 $PY scripts/analysis/build_behavior_span_manifest.py \
  --source memory /tmp/shared-storage/ExpertGym/eval/loss_qp_equal_weight/hotpotqa/hotpotqa_inference_results.jsonl \
  --summary-json /tmp/shared-storage/ExpertGym/eval/loss_qp_equal_weight/hotpotqa/evaluation_summary.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/memory_fulltraj_20260521 \
  --max-positive-per-task 32 \
  --max-negative-per-task 32 \
  --memory-response-mode full-trajectory
```

### 7.2 Memory full trajectory signed utility

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 $PY scripts/attention_pauh/probe_signed_utility.py \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --trajectory-jsonl /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/memory_fulltraj_20260521/behavior_positive.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/memory_fulltraj_positive_s32_20260521 \
  --tasks memory \
  --experts tool,memory,code \
  --scope all-linear \
  --span response \
  --samples-per-task 32 \
  --max-seq-length 8192 \
  --response-tail-tokens 0 \
  --write-row-details
```

### 7.3 v9 balanced operating point

```bash
PYTHONDONTWRITEBYTECODE=1 $PY scripts/attention_pauh/build_contrast_aware_residual_gates.py \
  --base-gates /tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json \
  --contrast-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livebench_prompt_alllayers_s16_20260521/contrast_module_summary.jsonl \
  --contrast-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livebench_reasoning_alllayers_s16_20260521/contrast_module_summary.jsonl \
  --contrast-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livecodebench_code_alllayers_s16_20260521/contrast_module_summary.jsonl \
  --contrast-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livecodebench_prompt_alllayers_s16_20260521/contrast_module_summary.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9 \
  --normalization per-file \
  --scale-quantile 0.9 \
  --max-delta 0.05 \
  --min-abs-score 0.1 \
  --aggregation conservative \
  --conflict-penalty 0.35 \
  --min-coeff 0.55 \
  --max-coeff 1.12 \
  --preserve-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s32_20260521/signed_utility_summary.json \
  --preserve-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/memory_fulltraj_positive_s32_20260521/signed_utility_summary.json \
  --preserve-task tool \
  --preserve-task memory \
  --preserve-min-normalized-utility 0.4 \
  --preserve-min-positive-fraction 0.5 \
  --preserve-negative-scale 0.0 \
  --harm-veto-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s32_20260521/signed_utility_summary.json \
  --harm-veto-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/memory_fulltraj_positive_s32_20260521/signed_utility_summary.json \
  --harm-veto-task tool \
  --harm-veto-task memory \
  --harm-veto-min-normalized-harm 0.4 \
  --harm-veto-positive-scale 0.5
```

### 7.4 v11 behavior-preserving operating point

```bash
PYTHONDONTWRITEBYTECODE=1 $PY scripts/attention_pauh/build_contrast_aware_residual_gates.py \
  <same inputs as v9> \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_tasktyped_v11 \
  --harm-veto-positive-scale 0.0 \
  --harm-veto-task-positive-scale tool=0.0 \
  --harm-veto-task-positive-scale memory=0.5
```

## 8. 下一步实验原则

1. 不再用 gate 系数作为目标。
2. 不再做无解释的 scalar sweep。
3. 每个新规则必须回答一个机制问题。
4. 评测闭环固定为 Tool quick + Memory eval_50 + Code hurt16；只有 Pareto 上有意义的点再进完整评测。
5. 如果要进一步提升 Code，不应放松 Tool hard constraint，而应改进 Code pass/fail span 的覆盖，例如更强的 LiveBench reasoning span 或 hidden-test guard span。

## 9. 一键复现 Harness

当前 RCRF Pareto frontier 的规范入口：

```bash
skill/command/run_20260522_rcrf_pareto_frontier.sh
```

它把关键流程拆成可审查 phase：

| phase | 作用 |
|---|---|
| `manifest` | 构造 Memory full-trajectory behavior manifest |
| `probe_memory` | 计算 Memory full-trajectory signed utility |
| `atlas` | 构建 residual conflict atlas，诊断 Code repair / Tool-Memory behavior 冲突 |
| `role_route` | 从 atlas role 直接生成 v12 role-routed gate |
| `generate` | 生成 v8/v9/v10/v11 gate |
| `bake` | 将 gate bake 成 HF checkpoint |
| `quick_eval` | 跑 Tool quick + Memory eval_50 |
| `code_hurt_eval` | 跑 Code hurt16 regression |
| `all` | `generate + bake + quick_eval` |

常用命令：

```bash
DRY_RUN=1 PHASE=generate CANDIDATES=v9 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

```bash
PHASE=all CANDIDATES=v9 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

```bash
PHASE=code_hurt_eval CANDIDATES=v9 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

说明：

- `DRY_RUN=1` 只打印命令，不执行。
- `CANDIDATES=v8,v9,v10,v11` 可选择要复现的 operating point。
- GPU / 端口 / 评测数据可通过环境变量覆盖，例如 `TOOL_GPU=0 CODE_GPU=2 MEMORY_DATASETS=eval_50`。
- 默认 `PHASE=all` 不重建 Memory probe；如果上游证据文件变化，需要先显式跑 `manifest` 和 `probe_memory`。

## 10. Residual Conflict Atlas

新增固定诊断：

```bash
PHASE=atlas bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_summary.md`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_rows.jsonl`

完整说明见：

- `docs/report/RCRF/20260522_residual_conflict_atlas.md`

核心统计：

| role | count |
|---|---:|
| `code_source_conflict_with_behavior` | 167 |
| `code_source_conflict` | 112 |
| `code_repair_only` | 60 |
| `code_negative_but_protected_support` | 58 |
| `shared_positive` | 17 |
| `code_repair_vs_protected_harm` | 16 |

这个 atlas 把当前 RCRF 的方法边界讲清楚：

- 只有一小部分 residual 是干净 Code repair。
- 大量 residual 的 Code span 证据互相冲突。
- Memory expert 是 Code/behavior conflict 的主体。
- Tool residual 中存在少量 shared-positive，但更多是需要保护的 behavior span。

因此下一步不应继续全局系数搜索，而应使用 atlas 中的 residual role 做规则化 routing 或报告 Pareto frontier。

## 11. v12 Role-Routed Gate

v12 是把 atlas 变成方法的第一版实现：

```bash
PHASE=role_route bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/gates.json`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/role_routing_summary.md`
- `docs/report/RCRF/20260522_role_routed_gate_v12.md`

规则：

- 提高 `code_repair_only` 与 `shared_positive`。
- Tool harm 上的 Code positive delta 不提高。
- Memory harm 上的 Code positive delta 软提高。
- 压低 `code_negative_noise` 和 `protected_harm_only`。
- 保持 `code_negative_but_protected_support` 和所有 `code_source_conflict*`。

v12 统计：

| metric | value |
|---|---:|
| changed | `133/588` |
| positive delta | 73 |
| negative delta | 60 |
| mean abs delta | 0.004696 |

v12 的意义：

- 它不是 sweep，也不是按指标倒推。
- 它把 RCRF 固化为 `attribution -> role -> routing`。
- 即使后续评测不优，它也能给出明确负结果：哪些 role 规则破坏了能力，反过来指导下一版归因粒度。

v12 最小闭环评测已完成：

| candidate | Tool quick | Memory eval_50 F1 | LiveBench hurt BoN | LiveCodeBench hurt BoN |
|---|---|---:|---|---|
| v9 soft | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v11 task-typed | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7701 | `0.2500 / 0.5469` | `0.3750 / 0.5126` |
| v12 role-routed | `0.880 / 0.865 / 0.8125 / 0.625` | 0.7627 | `0.2500 / 0.4453` | `0.4375 / 0.5294` |

结论：

- v12 保住了 Tool / Memory，说明 atlas role routing 对 behavior preservation 是有效的。
- v12 Code 弱于 v9，说明 naive `code_negative_noise` suppression 会伤 Code 泛化。
- 下一版应保留 positive role routing，但把负向 Code contrast 改成 source/span-conditioned 处理，而不是直接 pruning。

## 12. v13 Positive-Only Routing

v13 用来隔离 v12 失败原因：

```bash
PHASE=generate CANDIDATES=v13 bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

它只做 positive routing：

- raise `code_repair_only`；
- raise `shared_positive`；
- soft raise 少量 Memory-harm Code repair；
- hold `code_negative_noise`；
- hold `protected_harm_only`；
- hold all `code_source_conflict*`。

完整报告：

- `docs/report/RCRF/20260522_role_routed_positive_only_v13.md`

结果：

| candidate | Tool quick | Memory eval_50 F1 | LiveBench hurt BoN | LiveCodeBench hurt BoN |
|---|---|---:|---|---|
| v9 soft | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v12 role-routed | `0.880 / 0.865 / 0.8125 / 0.625` | 0.7627 | `0.2500 / 0.4453` | `0.4375 / 0.5294` |
| v13 positive-only | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7564 | `0.1250 / 0.2422` | `0.3125 / 0.5294` |

v13 结论：

- Tool / Memory 仍稳定，说明 RCRF behavior protection 路线可用。
- Code 进一步下降，说明问题不只是 v12 的 negative suppression。
- 当前最大错误是把所有 `code_source_conflict*` 都 hold。
- v9 的 Code gain 很可能来自一部分 source-conflict residual，因此 source conflict 不能被当成无信息噪声。

下一步方法应升级为：

```text
residual role routing -> source/span-conditioned conflict routing
```

也就是将归因单位从 `(param_name, expert, role)` 扩展到：

```text
(param_name, expert, role, source/span)
```
