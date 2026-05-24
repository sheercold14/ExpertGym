# 2026-05-22 RCF-BC 可复现闭环

## 目标

这份 harness 用来把当前 RCRF 线收束成论文主方法：

```text
RCF-BC = Residual Capability Field with Behavior Constraints
带行为约束的残差能力场
```

核心不是继续 sweep 全局 task coefficient，而是回答：

> 对每个 `(param_name, expert)` residual，能否用成功/失败轨迹和行为 span 证据判断它是在表达能力、制造冲突，还是纯噪声？

## 最小决策单位

当前三专家 OP-VEC 的 gate 空间：

```text
28 layers * 7 linear modules * 3 experts = 588 residual rows
```

每一行都要有可审查证据：

| 证据 | 来源 | 作用 |
|---|---|---|
| Code pass/fail contrast | 同 prompt 的 pass trajectory vs fail trajectory | 构建连续 Code capability field |
| Tool behavior span | tool-call 格式和调用 span | 保护结构化调用行为 |
| Memory full trajectory | update turns + final answer | 保护多跳记忆轨迹 |
| Atlas role | 上面证据的离散解释 | 只做审计和消融，不替代连续证据场 |
| Operating-point delta | 不同 gate 与 base 的差异 | 解释为什么某些方法掉 Code 或掉 Memory |

## 当前主方法

主候选：

```text
v18_rcf_bc
```

输出 gate：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/residual_capability_field_behavior_constraints_v18/gates.json
```

它与历史最强连续证据 operating point `v9` 数值完全一致：

```text
num_keys = 588
max_abs_diff = 0.0
different_count = 0
```

因此论文中应写 `RCF-BC / v18`，不要把 `v9` 当方法名。`v9` 只是早期实验编号。

## 一键复现入口

统一入口：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
```

生成主方法 gate：

```bash
PHASE=generate CANDIDATES=v18_rcf_bc \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

bake 成 HF checkpoint：

```bash
PHASE=bake CANDIDATES=v18_rcf_bc \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

快速评测 Tool + Memory：

```bash
PHASE=quick_eval CANDIDATES=v18_rcf_bc TOOL_GPU=0 MEMORY_GPU_IDS=0 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

评测 Code hurt subset：

```bash
PHASE=code_hurt_eval CANDIDATES=v18_rcf_bc CODE_GPU=2 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

重建机制诊断：

```bash
PHASE=diagnose \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

完整论文主线闭环：

```bash
PHASE=paper_main CANDIDATES=v18_rcf_bc TOOL_GPU=0 MEMORY_GPU_IDS=0 CODE_GPU=2 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

先检查命令而不执行：

```bash
DRY_RUN=1 PHASE=paper_main CANDIDATES=v18_rcf_bc \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

生成静态诊断面板：

```bash
PHASE=dashboard \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_diagnostic_dashboard_20260522/index.html
```

该面板用于按 candidate / expert / layer / module / role 查看 residual delta，不依赖前端服务，直接打开 HTML 即可。

生成 residual 冲突簇报告：

```bash
PHASE=clusters \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_clusters_20260522/conflict_cluster_report.md
```

该报告把 588 个 residual row 归纳为 `clean_code_repair`、`code_repair_with_behavior_harm`、`code_source_conflict`、`code_negative_with_behavior_support`、`code_negative_noise` 等机制簇，用于决定下一步方法改动。

生成论文证据总表：

```bash
PHASE=paper_table \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_paper_evidence_table_20260522/rcrf_paper_evidence_table.md
docs/report/RCRF/20260522_rcf_bc_paper_evidence_table.md
```

该表统一汇总 `v8-v23` 的 gate provenance、Tool/Memory 快评和 Code hurt 结果。Code 指标口径为 CURE 输出中的 `code_acc / BoN acc`，不是 accumulate acc。

生成反事实 residual 效果表：

```bash
PHASE=effect_table \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_counterfactual_effects_20260522/counterfactual_effect_report.md
docs/report/RCRF/20260522_counterfactual_residual_effects.md
```

该表把 `v14-v23` 的机械干预转成相对 `v18_rcf_bc` 的指标 delta，并显式计算 `v20` 相对 `v22+v23` 的非加性项，用来判断 residual group 是否能独立解释。

生成逐 residual row attribution ledger：

```bash
PHASE=ledger \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_attribution_ledger_20260522/rcrf_attribution_ledger_report.md
docs/report/RCRF/20260522_rcrf_attribution_ledger.md
```

该 ledger 把 588 个 residual row 的机制簇、`v18_rcf_bc` 系数、counterfactual group effect 和建议处理规则放在同一张表里。重点字段包括 `decision`、`routing_action`、`validation_priority` 和 `next_validation`。它的作用是把“code 系数是否太大”这类全局问题落到 row-level 证据：低置信 code 行是否能剪、剪了是否伤 Tool/Memory、是否需要继续作为 behavior-support audit 对象。

生成下一轮验证卡片：

```bash
PHASE=validation_plan \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_validation_plan_20260522/validation_plan.md
docs/report/RCRF/20260522_rcrf_validation_plan.md
```

该计划器读取 ledger，按 `routing_action / validation_priority / archetype / expert / layer_band / module_family` 聚合成验证卡片。它不生成新 gate，也不运行评测；作用是把下一轮反事实实验从“凭经验选几个 row”变成可审查的 P0 card queue。

物化 P0 验证卡片为最小 gate 干预：

```bash
PHASE=validation_interventions \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/validation_card_interventions_20260522/
docs/report/RCRF/20260522_rcrf_validation_interventions.md
```

该步骤默认只处理 validation plan 的前 6 张 P0 card。每个候选 gate 仍有 588 个 OP-VEC key，但只改对应 card 的代表 residual rows，用于后续 bake + Tool/Memory quick + Code hurt 的最小反事实验证。它不是新主方法，也不自动启动 GPU 评测。

生成簇语义一致性消融 `v19`：

```bash
PHASE=generate CANDIDATES=v19_archetype_consistency \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/gates.json
```

`v19` 从 `v18_rcf_bc` 出发，只把违反机制簇语义的 delta 投回 base。它用于和 `v18` 做机制消融，不是参数 sweep。

## 需要保留的负例

这些负例支撑论文 claim，比单点指标更重要：

| 负例 | 结论 |
|---|---|
| `v14_code_half` | 全局压低 code 可以释放 Memory，但 Code 直接崩，说明 task scalar 太粗 |
| `v15_code_zero` | code 置零进一步提高 Memory，但几乎摧毁 Code，证明不是“code 系数太大” |
| `v20_code_noise_half` | 只压低 code 中 `code_negative_noise + weak_or_uninformative` 的 60 行，验证 task scalar 能否局部化 |
| `v21_code_noise_zero` | 上述 60 行直接置零，是 residual-level zero 的强负例/压力测试 |
| `v13` | positive-only role routing 保护 Tool/Memory，但丢掉大量 v9 连续 delta |
| `v16/v17` | source-conflict hard routing 仍恢复不了 Code，说明 hard role rule 不够 |
| `v19` | archetype-consistency projection，检验无证据 drift 和簇语义冲突是否必要 |

当前关键数字：

| candidate | Tool quick | Memory F1 | LiveBench hurt acc / BoN | LiveCodeBench hurt acc / BoN |
|---|---:|---:|---:|---:|
| `v18_rcf_bc / v9` | 0.880 / 0.855 / 0.8125 / 0.625 | 0.7575 | 0.2500 / 0.6250 | 0.6250 / 0.6555 |
| `v14_code_half` | 0.880 / 0.860 / 0.8125 / 0.625 | 0.7774 | 0.2031 / 0.2500 | 0.2500 / 0.1875 |
| `v15_code_zero` | 0.880 / 0.865 / 0.7500 / 0.625 | 0.7841 | 0.0781 / 0.1250 | 0.1719 / 0.1875 |
| `v16_source_suppress` | 0.880 / 0.860 / 0.8125 / 0.625 | 0.7660 | 0.1094 / 0.2500 | 0.3281 / 0.4375 |
| `v17_source_route` | 0.880 / 0.855 / 0.8125 / 0.625 | 0.7654 | 0.1250 / 0.3125 | 0.2188 / 0.4375 |

## 机制诊断读法

主诊断输出：

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/
```

重点看：

| 文件 | 用途 |
|---|---|
| `operating_point_comparison_summary.json` | 每个候选 changed/positive/negative/delta 统计 |
| `operating_point_rows.jsonl` | 588 行逐 residual 对齐结果 |
| `delta_by_role.csv` | 哪类 atlas role 被改动 |
| `reference_lost_by_role.csv` | 相对 v9 丢掉了哪些连续 delta |
| `reference_lost_by_source_pattern.csv` | source-conflict 具体丢失模式 |

当前最重要发现：

```text
v9 changed 205 rows，其中 positive 106，negative 99。
v13 丢掉 145 个 v9 delta。
v16 丢掉 124 个 v9 delta。
v17 丢掉 110 个 v9 delta，并有 11 个 sign mismatch。
```

这支持一个简单结论：

> 能力不是少数 clean positive row；Code 尤其依赖低置信、连续、小幅度的 residual field。离散 role routing 足够保守，但会丢能力。

## 论文主 claim

推荐写法：

> RL expert task vectors are not task-pure modules. They are mixtures of capability residuals and behavior-changing residuals. RCF-BC estimates a continuous residual capability field from outcome contrast, then constrains it with behavior utility and harm evidence from other tasks.

中文解释：

> RL expert 的 task vector 不是纯能力包，而是能力残差和行为改变残差的混合。RCF-BC 用成功/失败轨迹建立连续能力场，再用 Tool/Memory 的行为 utility/harm 约束这个能力场，从 residual 级别处理模式冲突。

## 下一步实验优先级

1. **主方法复验**：用 `v18_rcf_bc` 跑完整 paper_main，确认 bake 和快评链路没有断。
2. **独立 heldout**：把 Code hurt subset 扩到一个不参与构造的 heldout hurt subset，验证不是过拟合 16 题。
3. **ToolRL 80**：Tool live 有波动时，用 ToolRL test 作为更稳定的行为能力指标。
4. **Atlas 可视化**：已补静态 dashboard；下一步把最关键的 heatmap 转成论文图。
5. **最小 ablation**：报告 `v14/v15/v16/v17`，证明 scalar shrinkage 和 hard routing 都不足。
6. **局部 code shrink 验证**：跑 `v20/v21` 的 Tool/Memory 快评；如果不崩，再跑 Code hurt subset，与 `v14/v15` 比较。
7. **反事实归因表**：跑 `effect_table`，把每组 row intervention 的真实指标影响写入论文分析，避免只凭 row label 解释。

## 停止继续调参的规则

以下方向先不再投入：

- 全局 code coefficient 继续扫。
- 只按 atlas role hard route。
- 只看 final answer 的 Memory protection。
- 继续扩大 calibration 但不增加 residual 级别解释。

如果后续要改方法，必须能回答：

```text
它改变了哪一类 residual evidence？
它保护了哪一类 behavior span？
它相对 v18 多保留或少保留了哪些 v9-style continuous delta？
```
