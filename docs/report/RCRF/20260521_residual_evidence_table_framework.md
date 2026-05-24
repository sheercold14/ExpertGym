# 2026-05-21 Residual Evidence Table：泛化能力归因框架的执行基座

## 0. 目的

上一份报告已经说明：Memory-Code 冲突不是简单的任务级 gate 冲突，而是 **residual key 级别、source/span 条件化的 utility / harm / conflict**。

本次新增一个统一证据表，将散落的机制信号对齐到同一个坐标：

`(param_name, expert)`

每一行对应一个 expert residual 在一个具体线性模块上的行为证据。这个表不是为了立刻调出最高分，而是为了让后续方法具备可复查的第一性基础：**我们到底为什么增强、抑制、保护或保持某个 residual。**

## 1. 代码与产物

脚本：

`scripts/analysis/build_residual_evidence_table.py`

行为 span manifest 脚本：

`scripts/analysis/build_behavior_span_manifest.py`

候选 gate 生成脚本：

`scripts/attention_pauh/build_evidence_routed_residual_gates.py`

输出目录：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/residual_evidence_table_20260521`

输出文件：

- `residual_evidence_rows.csv`
- `residual_evidence_rows.jsonl`
- `residual_evidence_summary.json`
- `residual_evidence_summary.md`

Tool/Memory 行为 span manifest：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/tool_memory_20260521`

当前 manifest 从已有评测/rollout 产物构建：

- Tool：`ta_c100_init1_toolrl_all80_20260521` 与 `rcrf_v1_rcrf_toolrl_all80_20260521` 的 ToolRL80 rollout。
- Memory：`loss_qp_equal_weight/hotpotqa` inference rows + `evaluation_summary.json` 的 `correct_indices`。

选中数量：

| task | positive | negative |
|---|---:|---:|
| memory | 32 | 10 |
| tool | 32 | 32 |

该 manifest 已通过 `probe_signed_utility.py --plan-only` 验证：`tool/memory` 各可读取 4 条样本，目标参数为 `196` 个模块、`588` 个 expert residual entry。

第一版 evidence-routed gate：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_evidence_routed_v1/gates.json`

它只 materialize `keep_or_raise` 和 `suppress`，默认不动 `hold_conflict / preserve / no_decision`。当前改动 `56/588` 个 residual key，其中 `35` 个增强、`21` 个抑制；这不是最终模型结论，只是把证据表转成可评测候选。

加入 Tool/Memory behavior-positive utility 后的 gate：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_evidence_routed_tmpos_s8_v1/gates.json`

它合入：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s8_20260521/signed_utility_summary.json`

改动从 `56/588` 收缩到 `25/588`，其中 `20` 个增强、`5` 个抑制。Memory expert 改动从 `25` 收缩到 `5`，Tool expert 改动从 `12` 收缩到 `1`，说明 behavior utility 起到了能力保护作用。

当前覆盖：

- 28 层。
- 7 类线性模块：`q/k/v/o/gate/up/down`。
- 3 个 expert：`tool/memory/code`。
- 总计 `588 = 28 * 7 * 3` 行。

## 2. 输入信号

### 2.1 Code pass/fail source/span contrast

当前接入 5 个 Code source/span：

| source | 含义 | source mean abs |
|---|---|---:|
| `LB_code` | LiveBench final code span | `2.072e-07` |
| `LB_prompt` | LiveBench prompt span | `2.255e-07` |
| `LB_reasoning` | LiveBench reasoning span | `2.751e-06` |
| `LCB_code` | LiveCodeBench final code span | `3.396e-05` |
| `LCB_prompt` | LiveCodeBench prompt span | `1.506e-05` |

每个 source 内先按该 source 的 `mean_abs` 做归一化。这样不会让 LiveCodeBench 因绝对效应量更大而完全压过 LiveBench，也不会让极小数值的符号噪声直接控制决策。

脚本同时记录两类符号：

- raw sign：原始正负号，只用于审计。
- informative sign：`abs(effect / source_mean_abs) >= 0.25` 才计入推荐动作。

这个阈值不是性能调参，而是为了避免把接近 0 的 probe 噪声当成机制证据；原始值仍完整保留在表中。

### 2.2 Tool / Memory / Code behavior utility

当前默认接入：

`/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521/signed_utility_summary.json`

每个 residual key 对每个任务记录：

- `signed_effect_mean`
- `harm_mean`
- `positive_fraction`
- `expression_mean`
- `count`

当前这个 utility signature 仍然偏小，只能作为第一版诊断信号。现在 `build_residual_evidence_table.py` 已支持重复传入 `--signed-utility-summary`，多个 summary 会按 `count` 加权合并。最终应替换或补充为：

- Memory：完整 update turns + final turn 的行为 span。
- Tool：tool-call span，区分 BFCL live / non-live。
- Code：prompt / reasoning / final code span 的 pass/fail contrast。

### 2.3 已有 gate delta

表中也对齐了 v1-v4 的 gate delta，便于复盘某个方法为什么改变某个 key：

- `v1_code_only`
- `v2_spanaware`
- `v3_memory_hard_floor`
- `v4_memory_utility_floor`

## 3. 推荐动作定义

证据表为每个 residual key 输出一个诊断动作：

| recommendation | 含义 |
|---|---|
| `keep_or_raise` | Code source/span 证据一致正向，且没有观测到任务 utility harm。 |
| `suppress` | Code source/span 证据一致负向，且没有观测到正向任务 utility。 |
| `preserve` | 当前没有 Code contrast 信号，但某个任务有正向 behavior utility。 |
| `hold_conflict` | Code source/span 之间冲突，或 Code 方向与 Tool/Memory/Code utility 冲突。 |
| `no_decision` | 现有 probe 不足以做决定。 |

注意：这不是最终 gate 值，而是 residual-level attribution。它回答的是“这个 key 是否有足够证据被动”，不是“系数应该是多少”。

## 4. 当前统计结果

在 informative sign 规则下，588 个 residual key 的分布为：

| recommendation | count |
|---|---:|
| `hold_conflict` | 400 |
| `keep_or_raise` | 76 |
| `suppress` | 43 |
| `preserve` | 17 |
| `no_decision` | 52 |

Code source state：

| code_state | count |
|---|---:|
| `source_conflict` | 352 |
| `positive` | 100 |
| `negative` | 67 |
| `no_signal` | 69 |

按 expert 看：

| expert | hold_conflict | keep_or_raise | suppress | preserve | no_decision |
|---|---:|---:|---:|---:|---:|
| code | 118 | 29 | 9 | 10 | 30 |
| memory | 159 | 20 | 16 | 0 | 1 |
| tool | 123 | 27 | 18 | 7 | 21 |

## 5. 机制解释

### 5.1 为什么 scalar gate 难学

如果大多数 residual key 都是 `keep_or_raise` 或 `suppress`，那么 task-level scalar 可能足够；但现在 `400/588` 是 `hold_conflict`，其中 `352/588` 在 Code source/span 内部已经冲突。

这说明问题不是“学习率不够”或“GRPO 推不动”，而是目标本身不适合用一个全局光滑方向描述：

- 同一 residual 在 LiveBench prompt 上可能正向，在 LiveCodeBench prompt 上可能负向。
- 同一 expert residual 中既有能力成分，也有分布特异 harm。
- Memory expert 里的 residual 不只服务 Memory，也可能影响 Code 的 prompt/constraint behavior。

### 5.2 为什么 v2 有效但不完美

v2 span-aware conservative routing 的成功说明：**用 pass/fail span 证据来选择 residual 是有效的。**

但 v2 仍然使 Memory 小幅下降，说明它只看 Code pass/fail contrast，不知道哪些 Code 修复动作会压到 Memory behavior span。

证据表的作用就是补上这个缺口：让 Code repair 不是孤立地决定 residual，而是和 Memory/Tool utility 放在一起判断。

### 5.3 为什么 v3/v4 是重要反例

v3 把 memory expert 的负向 overlay 全保护，结果 LiveCodeBench 下降。v4 用小规模 Memory utility floor，也没有解决 LCB 损伤。

这证明：

- 不能把 “memory expert residual” 等同于 “Memory 能力 residual”。
- 不能用少量 final/signature 近似替代完整 behavior utility。
- 保护必须发生在 residual key + behavior span 层面。

## 6. 推荐的下一版方法

下一版不要继续做阈值调参，而是做一个固定原则的 conservative evidence routing：

1. 先构建完整 evidence table。
2. 对 `keep_or_raise`：允许小幅增强。
3. 对 `suppress`：允许小幅抑制。
4. 对 `preserve`：保持或轻微保护。
5. 对 `hold_conflict`：默认不动，除非有更强任务条件化机制。
6. 对 `no_decision`：不动。

这个方法的论文价值在于：它不是在 validation 上 sweep 系数，而是基于 trajectory outcome 和 behavior utility 判断每个 residual 的角色。

## 7. 当前缺口

当前 evidence table 已经能支持机制分析，但还不能作为最终方法直接宣称 SOTA，原因是：

1. Memory/Tool utility signature 规模太小。
2. Memory 还没有使用完整轨迹 span。
3. Tool 还没有系统接入 BFCL live / non-live tool-call span。
4. Code 只在 hurt subset 上做了强诊断，还需要和 train-only generated tests / hidden-like guard tests 对齐。

这些缺口不是调参问题，而是 attribution evidence 的覆盖问题。

## 7.1 已补的接口

本轮补齐了两个接口：

1. `build_behavior_span_manifest.py`：把 rollout / inference 输出变成 `probe_signed_utility.py` 可直接读取的 `behavior_positive.jsonl` / `behavior_negative.jsonl`。
2. `build_residual_evidence_table.py --signed-utility-summary ...`：支持多个 behavior utility summary 合并到同一张 residual evidence table。

因此下一步不需要重写框架，只需要跑：

```bash
python scripts/attention_pauh/probe_signed_utility.py \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --trajectory-jsonl /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/tool_memory_20260521/behavior_positive.jsonl \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_<run_id> \
  --tasks tool,memory \
  --scope all-linear \
  --span signature \
  --samples-per-task 32 \
  --write-row-details
```

然后把生成的 `signed_utility_summary.json` 作为额外输入：

```bash
python scripts/analysis/build_residual_evidence_table.py \
  --signed-utility-summary /tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521/signed_utility_summary.json \
  --signed-utility-summary /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_<run_id>/signed_utility_summary.json \
  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/residual_evidence_table_<run_id>
```

## 8. 最小闭环实验

建议下一步做一个最小闭环，不跑大规模 RL：

1. 用完整 Memory trajectory 构建 Memory behavior utility。
2. 用 BFCL tool-call span 构建 Tool behavior utility。
3. 复用当前 Code source/span contrast。
4. 生成新的 evidence table。
5. 按固定 conservative routing rule bake 一个 gate。
6. 先测 ToolRL 80 + HotpotQA eval50。
7. 若 Tool / Memory 过门槛，再测 Code hurt subset 和正式 Code。

当前已经完成第 1 版离线候选生成：

| candidate | base | changed | raise | suppress | 说明 |
|---|---|---:|---:|---:|---|
| `rcrf_evidence_routed_v1` | `rcrf_v1_20260521/rcrf` | 56 | 35 | 21 | 只动非冲突的 `keep_or_raise/suppress` 行；冲突行保持 base。 |
| `rcrf_evidence_routed_tmpos_s8_v1` | `rcrf_v1_20260521/rcrf` | 25 | 20 | 5 | 合入 Tool/Memory behavior-positive utility 后，更保守地保护 Memory/Tool residual。 |
| `rcrf_evidence_routed_tmpos_s32_v1` | `rcrf_v1_20260521/rcrf` | 29 | 24 | 5 | Tool/Memory behavior-positive utility 扩到每任务 32 条后，保持保守但比 s8 多恢复少量 Memory/Tool 正向 residual。 |

如果这个闭环有效，就可以形成论文主线：

> Expert residuals are not task-pure. We attribute each residual by outcome-aware utility and cross-task conflict, then route only the residuals with reliable evidence.

## 9. s8 行为 utility 快速闭环结果

### 9.1 Probe

命令使用：

- trajectory: `behavior_positive.jsonl`
- tasks: `tool,memory`
- samples-per-task: `8`
- scope: `all-linear`
- span: `signature`
- max sequence length: `3072`

产物：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s8_20260521/signed_utility_summary.json`

关键层：

| task/expert | top layers | 解释 |
|---|---|---|
| Tool owner utility | 17, 14, 16, 5, 20, 9, 18, 15 | Tool-call 行为 utility 集中在中后层与部分早层，且 protected harm 可观。 |
| Memory owner utility | 23, 26, 25, 6, 22, 14, 24, 5 | Memory final/trajectory behavior 对后层有强 utility，和之前 Memory 容易被 Code 修复误压的现象一致。 |
| Code on Tool/Memory behavior | owner utility 约 0，存在 protected harm | Code residual 在 Tool/Memory behavior span 上主要表现为潜在 harm，而不是 owner utility。 |

合表变化：

| evidence table | hold_conflict | keep_or_raise | suppress | preserve | no_decision |
|---|---:|---:|---:|---:|---:|
| small signature only | 400 | 76 | 43 | 17 | 52 |
| + Tool/Memory positive s8 | 466 | 41 | 12 | 39 | 30 |

解释：加入真实 Tool/Memory behavior utility 后，许多原本会被 Code contrast 判成可动的 residual 被改判为冲突或 preserve。这是期望行为，因为它避免为了修 Code 而压掉 Tool/Memory 关键行为 span。

### 9.2 Quick Eval

Baked checkpoint：

`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_evidence_routed_tmpos_s8_v1`

评测目录：

`/tmp/shared-storage/ExpertGym/rcrf/eval/evidence_routed_tmpos_s8_v1/quick_tool_memory`

Tool BFCL quick：

| category | accuracy |
|---|---:|
| parallel | 0.8800 |
| parallel_multiple | 0.8550 |
| live_parallel | 0.6875 |
| live_parallel_multiple | 0.6250 |

Memory HotpotQA eval_50：

| metric | value |
|---|---:|
| avg_f1 | 0.7648 |
| exact_match_rate | 0.6016 |
| sub_exact_match_rate | 0.7969 |
| boxed_rate | 1.0000 |

结论：`rcrf_evidence_routed_tmpos_s8_v1` 在 Tool 上保持 RCRF/v2 级别，在 Memory eval_50 上达到 `0.7648`，基本没有出现 v2 小幅 Memory side-effect 的恶化。这支持“behavior utility 作为保护证据”的方向。

## 10. s32 行为 utility 复核

### 10.1 Probe 与合表

产物：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s32_20260521/signed_utility_summary.json`

配置与 s8 相同，但 `samples-per-task=32`。这一步的目的不是调参，而是检查 Tool/Memory behavior utility 在更多样本上是否稳定。

关键现象：

| task/expert | top layers | 解释 |
|---|---|---|
| Tool owner utility | 17, 18, 20, 9, 16, 5, 8, 22 | Tool-call 行为仍集中在中后层，并保留少量早层入口。 |
| Memory owner utility | 19, 18, 17, 16, 20, 15, 21, 22 | 相比 s8，Memory utility 从后层 23-26 转为更稳定的中后层 15-22，说明小样本后层峰值可能偏噪声。 |
| Code on Tool/Memory behavior | owner utility 约 0，存在 protected harm | 继续支持 Code residual 不能无约束覆盖 Tool/Memory behavior span。 |

合表变化：

| evidence table | hold_conflict | keep_or_raise | suppress | preserve | no_decision |
|---|---:|---:|---:|---:|---:|
| small signature only | 400 | 76 | 43 | 17 | 52 |
| + Tool/Memory positive s8 | 466 | 41 | 12 | 39 | 30 |
| + Tool/Memory positive s32 | 463 | 44 | 12 | 39 | 30 |

解释：s32 和 s8 的 `suppress/preserve/no_decision` 基本一致，说明 behavior utility 不是偶然把大量 residual 改判，而是在稳定识别需要保护的区域。s32 多释放了 3 个 `keep_or_raise`，更像是降低小样本噪声后的保守恢复。

### 10.2 Gate 与快速评测

Gate：

`/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_evidence_routed_tmpos_s32_v1/gates.json`

Baked checkpoint：

`/tmp/shared-storage/OnPolicy/checkpoints/rcrf_evidence_routed_tmpos_s32_v1`

评测目录：

`/tmp/shared-storage/ExpertGym/rcrf/eval/evidence_routed_tmpos_s32_v1/quick_tool_memory`

Gate 改动：

| expert | changed | raise | suppress | max_abs_delta |
|---|---:|---:|---:|---:|
| code | 19 | 16 | 3 | 0.0400 |
| memory | 7 | 5 | 2 | 0.0400 |
| tool | 3 | 3 | 0 | 0.0198 |
| overall | 29 | 24 | 5 | 0.0400 |

Tool BFCL quick：

| category | accuracy |
|---|---:|
| parallel | 0.8800 |
| parallel_multiple | 0.8600 |
| live_parallel | 0.8125 |
| live_parallel_multiple | 0.6250 |

Memory HotpotQA eval_50：

| metric | value |
|---|---:|
| avg_f1 | 0.7802 |
| exact_match_rate | 0.6172 |
| sub_exact_match_rate | 0.8125 |

### 10.3 判断

s32 是当前更有希望的 RCRF 离线候选：

- Tool non-live 与 s8 持平或微升，live_parallel 从 `0.6875` 提到 `0.8125`。
- Memory eval_50 F1 从 s8 的 `0.7648` 提到 `0.7802`。
- Gate 只改 `29/588`，因此结果更能支持“少量证据充分的 residual routing 足以保护能力”的机制故事。

下一步不应该马上继续调阈值，而应先补 Code 侧的闭环：用同一套 evidence table 解释哪些 Code hurt 样本被修复、哪些仍失败；如果 Tool/Memory 继续稳定，再进入正式 Code 评测。

## 11. Code Repair 与行为保护闭环

### 11.1 为什么 s32 不够

s32 的结论是“保护能力强，但 Code repair 弱”。它只改 `29/588` 个 residual key，因此能保持 Tool/Memory，却没有保留 v2 那种大范围 Code pass/fail span 修复。

Code hurt subset 快评：

| candidate | changed | LiveBench BoN acc | LiveBench BoN accum | LiveCodeBench BoN acc | LiveCodeBench BoN accum | Tool quick | Memory eval50 F1 |
|---|---:|---:|---:|---:|---:|---|---:|
| `v2_code_spanaware` | 562 | 0.1875 | 0.3672 | 0.7500 | 0.7899 | mean 0.7931 | 0.7650 |
| `s32_evidence_routed` | 29 | 0.1875 | 0.4219 | 0.4375 | 0.6303 | 0.8800 / 0.8600 / 0.8125 / 0.6250 | 0.7802 |

解释：

- v2 是强 Code repair，但 Memory 有小幅副作用。
- s32 是强保护，但 Code repair 过弱。
- 因此正确方向不是继续调 s32 阈值，而是把 v2 的 Code repair 与 s32 的行为证据组合。

### 11.2 v6：Code repair + Tool/Memory preserve floor

v6 使用同一套 Code pass/fail span contrast，保留 v2 的 repair 逻辑；同时接入 s32 Tool/Memory behavior-positive utility：

- 若某个 residual 对 Tool/Memory behavior span 有正 utility，则不允许 Code overlay 降低它。
- 仍保持每个 expert 的均值，避免退化成 task weight sweep。

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_preserve_v6/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_preserve_v6`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/rcrf_code_spanaware_tmpos_s32_preserve_v6/quick_tool_memory`

Gate 改动：

| group | changed | raise | suppress | mean_abs_delta |
|---|---:|---:|---:|---:|
| overall | 310 | 102 | 208 | 0.0040 |
| code | 120 | 39 | 81 | 0.0051 |
| memory | 97 | 43 | 54 | 0.0047 |
| tool | 93 | 20 | 73 | 0.0022 |

评测：

| dataset | BoN acc | BoN accum |
|---|---:|---:|
| LiveBench hurt16 | 0.3125 | 0.4922 |
| LiveCodeBench hurt16 | 0.6250 | 0.7395 |

Tool BFCL quick：

| parallel | parallel_multiple | live_parallel | live_parallel_multiple |
|---:|---:|---:|---:|
| 0.8800 | 0.8600 | 0.7500 | 0.6250 |

Memory eval_50 F1：`0.7528`。

判断：v6 证明“Code repair + behavior preserve”方向有效，Code 明显强于 s32；但 Memory 低于 s32/v2，说明只防止降低 positive utility 还不够。

### 11.3 v7：再加入 behavior harm veto

v7 在 v6 基础上新增一个默认关闭的离线规则：

- 如果某个 residual 在 Tool/Memory behavior span 上有 harm，则 Code overlay 不允许升高它。
- preserve 约束负责“不要降低有用 residual”；harm-veto 负责“不要升高有害 residual”。
- 该规则只在 `build_contrast_aware_residual_gates.py` 显式传入 `--harm-veto-summary` 时启用，默认关闭不影响旧实验。

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_preserve_harmveto_v7/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_preserve_harmveto_v7`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/rcrf_code_spanaware_tmpos_s32_preserve_harmveto_v7/quick_tool_memory`

Gate 改动：

| group | changed | raise | suppress | mean_abs_delta |
|---|---:|---:|---:|---:|
| overall | 201 | 89 | 112 | 0.0030 |
| code | 66 | 27 | 39 | 0.0027 |
| memory | 73 | 29 | 44 | 0.0045 |
| tool | 62 | 33 | 29 | 0.0019 |

评测：

| dataset | BoN acc | BoN accum |
|---|---:|---:|
| LiveBench hurt16 | 0.2500 | 0.5156 |
| LiveCodeBench hurt16 | 0.7500 | 0.7731 |

Tool BFCL quick：

| parallel | parallel_multiple | live_parallel | live_parallel_multiple |
|---:|---:|---:|---:|
| 0.8800 | 0.8650 | 0.8125 | 0.6250 |

Memory eval_50 F1：`0.7425`。

判断：

- v7 恢复了 LiveCodeBench hurt 的 Code 修复能力，几乎回到 v2，同时 Tool 保持很好。
- Memory 继续下降，说明当前 Memory behavior-positive signature 还不能充分刻画 Memory 能力保护；尤其是 HotpotQA 需要 update turns + final answer 的完整轨迹 span，而不只是 signature/final-like 行为。
- 这给出一个清晰机制结论：Tool 的 behavior span 比较容易被 tool-call signature 捕获；Memory 的能力 span 更长、更分散，必须用完整轨迹或更强的 memory-specific residual utility 来约束。

### 11.4 当前研究判断

当前最合理的论文主线不是“一个静态规则已经解决全部任务”，而是：

> Expert residual 的 utility/harm 是 outcome-aware 且 span-conditioned 的。Code repair 需要 pass/fail contrast；Tool 可以由 tool-call span 保护；Memory 需要完整轨迹 span 保护。粗粒度 scalar gate 或 expert-level preserve 都无法表达这个结构。

下一步优先级：

1. 重新构建 Memory utility：必须使用完整或近似完整的 update turns + final turn，而不是只用短 signature。
2. 在同一个 v7 框架里替换 Memory utility summary，再看 Memory 是否回到 `0.77+`。
3. 如果 Memory 能恢复，v7 是最有希望的三任务候选；如果仍不行，说明静态单 gate 不够，需要 task-conditioned residual routing。

## 12. Memory-Code 冲突的当前计划

我认为这条线仍有希望，但必须保持简单，不再靠调参。当前假设是：

> Code repair 方向提升的是“局部执行 / 分支枚举 / 自校验 / 代码块稳定输出”等行为；Memory 需要的是“检索证据保真 / 实体绑定 / 多跳关系约束 / update turns 到 final answer 的一致性”。二者的冲突未必出现在最终答案 token，而更可能出现在 Memory 的中间更新轨迹。

因此，当前只看 Memory final answer 或短 signature 的保护信号是不够的。Tool 可以被 tool-call span 捕获，但 HotpotQA / MemAgent 式 Memory 能力需要完整轨迹 span。

### 12.1 正在做的验证

新增一个默认关闭的 manifest 选项：

```bash
--memory-response-mode full-trajectory
```

它把 Memory inference row 中的 `chunk_rounds[*].response` 与 final answer 串成 teacher-forced response。这样 signed utility probe 衡量的是 residual 对完整记忆轨迹的支持或伤害，而不是只衡量 boxed final answer。

当前产物：

- manifest: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/memory_fulltraj_20260521`
- probe: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/memory_fulltraj_positive_s32_20260521`

### 12.2 这个验证能回答什么

如果 full trajectory utility 能恢复 Memory，则说明 v6/v7 的 Memory 下降主要来自 span 选择错误：保护信号没有覆盖关键 update 行为。此时最干净的方法是：

1. Code 仍用 same-prompt pass/fail span contrast。
2. Tool 仍用 tool-call behavior span 保护。
3. Memory 改用 update turns + final answer 的 full-trajectory utility/harm。
4. 在 residual key 级别做 utility/harm routing，而不是 expert 级别保护。

如果 full trajectory 仍不能恢复 Memory，则说明单一全局 residual gate 可能不足以同时表达 Code repair 和 Memory evidence fidelity；下一步应只做一个很窄的 task-conditional anchor 验证，而不是继续扩大规则复杂度。

### 12.3 当前原则

- 不再根据期望指标调阈值。
- 不把 gate 系数本身当作决策目标。
- 只接受能解释冲突机制、并且能被 quick Tool/Memory + Code hurt subset 闭环验证的改动。
- 优先寻找一个小而可证伪的发现：不同任务能力的关键 span 不同，因此 residual utility 必须是 outcome-aware 且 span-conditioned 的。

## 13. v8：加入 Memory 完整轨迹保护后的结果

v8 在 v7 基础上只做一个机制性修改：额外加入 Memory full trajectory signed utility summary，同时仍保留原 Tool/Memory signature summary。规则不变：

- preserve：不降低 Tool/Memory 有正 utility 的 residual。
- harm-veto：不升高 Tool/Memory 有 harm 的 residual。
- Code repair 仍来自 same-prompt pass/fail contrast。

产物：

- manifest: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/memory_fulltraj_20260521`
- probe summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/memory_fulltraj_positive_s32_20260521/signed_utility_summary.json`
- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_v8/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_v8`

完整轨迹 probe 的 top owner utility 显示：

- Memory expert 的正 utility 集中在 layer `13-18` 附近，layer 14/16/17 最强。
- Tool/code expert 对 Memory full trajectory 基本是 0 或 harm。
- 这支持“Memory 能力 span 是中间轨迹级的，不是 final answer 级的”这一判断。

Gate 改动：

| candidate | changed | + | - | mean_abs_delta |
|---|---:|---:|---:|---:|
| v7 | 201 | 89 | 112 | 0.003021 |
| v8 | 135 | 61 | 74 | 0.001983 |

Protection 证据：

| candidate | preserved keys | memory preserve keys | harm-veto keys | memory veto keys |
|---|---:|---:|---:|---:|
| v7 | 314 | 117 | 402 | 231 |
| v8 | 366 | 191 | 481 | 365 |

Quick Tool/Memory：

| candidate | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Memory eval_50 F1 |
|---|---:|---:|---:|---:|---:|
| v6 | 0.8800 | 0.8600 | 0.7500 | 0.6250 | 0.7528 |
| v7 | 0.8800 | 0.8650 | 0.8125 | 0.6250 | 0.7425 |
| v8 | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7720 |

Code hurt subset：

| candidate | LiveBench BoN acc | LiveBench BoN accum | LiveCodeBench BoN acc | LiveCodeBench BoN accum |
|---|---:|---:|---:|---:|
| v6 | 0.3125 | 0.4922 | 0.6250 | 0.7395 |
| v7 | 0.2500 | 0.5156 | 0.7500 | 0.7731 |
| v8 | 0.2500 | 0.4688 | 0.3125 | 0.5378 |

判断：

- v8 验证了 full-trajectory Memory protection 的必要性：Memory 从 v7 的 `0.7425` 恢复到 `0.7720`。
- Tool 没有下降，说明 Tool call-span + Memory trajectory protection 可以同时存在。
- 但 Code repair 被压弱，尤其 LiveCodeBench hurt 从 `0.7500` 降到 `0.3125`。这说明 v8 不是最终模型，而是一个重要机制证据：Memory full trajectory 能识别更多需要保护的 residual，但当前“硬 veto”过强，会误杀 Code 所需的共享 residual。

下一步的最小改动不是调一堆阈值，而是把 hard veto 改成 Pareto-aware soft routing：

1. 对 Memory full-trajectory harm 使用 soft scale，而不是全部置零。
2. 只对 high-confidence Memory owner utility 做 hard floor。
3. 对 Code pass/fail contrast 保留强信号，避免把 v7 的 Code 修复完全压掉。
4. 评估仍使用同一闭环：Tool quick + Memory eval_50 + Code hurt16。

## 14. v9：Soft Pareto Routing 验证

v9 不引入新的证据，也不改变 Code contrast / Memory full-trajectory probe。唯一变化是将 harm-veto 从 hard veto 改为 soft routing：

```bash
--harm-veto-positive-scale 0.5
```

含义：

- v8：如果 residual 对 Tool/Memory trajectory 有 harm，则 Code positive delta 直接置零。
- v9：同样的 harm residual 仍被惩罚，但只把 Code positive delta 缩小一半。

这不是为了调一个最优常数，而是验证一个更基本的机制：**Memory full trajectory harm 是有效约束，但不能以 hard veto 形式完全覆盖 Code pass/fail utility。**

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/quick_tool_memory`

Gate 改动：

| candidate | harm scale | changed | + | - | mean_abs_delta |
|---|---:|---:|---:|---:|---:|
| v7 | hard short-span | 201 | 89 | 112 | 0.003021 |
| v8 | 0.0 | 135 | 61 | 74 | 0.001983 |
| v9 | 0.5 | 205 | 106 | 99 | 0.002639 |

Quick Tool/Memory：

| candidate | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Memory eval_50 F1 |
|---|---:|---:|---:|---:|---:|
| v7 | 0.8800 | 0.8650 | 0.8125 | 0.6250 | 0.7425 |
| v8 | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7720 |
| v9 | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7575 |

Code hurt subset：

| candidate | LiveBench BoN acc | LiveBench BoN accum | LiveCodeBench BoN acc | LiveCodeBench BoN accum |
|---|---:|---:|---:|---:|
| v7 | 0.2500 | 0.5156 | 0.7500 | 0.7731 |
| v8 | 0.2500 | 0.4688 | 0.3125 | 0.5378 |
| v9 | 0.2500 | 0.6250 | 0.6250 | 0.6555 |

判断：

- v8 证明 full-trajectory Memory protection 能救 Memory，但 hard veto 过度压制 Code。
- v9 证明 soft routing 能恢复大部分 Code repair，尤其 LiveCodeBench hurt 从 `0.3125` 回升到 `0.6250`，同时 Memory 仍高于 v7。
- Tool 基本稳定，说明 Tool call span 保护没有被 softening 破坏。

这给当前方法一个更清晰的第一性形式：

> residual 的正负作用应被视作多目标 utility，而不是二值保留/删除。Code pass/fail utility 给出“可提升方向”，Tool call-span 与 Memory trajectory utility/harm 给出“能力保持约束”；最终 gate 应在 residual key 级做 Pareto-aware soft routing。

下一步如果继续推进，不应继续手扫 `0.0/0.5`，而应把 soft scale 改成由 normalized Code utility 与 normalized Tool/Memory harm 自动决定，例如：

```text
positive_delta *= code_utility / (code_utility + protected_harm + eps)
```

这样 LR/阈值不再决定任务间 trade-off，trade-off 来自同一个 residual key 上的相对证据强度。

## 15. v10：Naive Evidence-Ratio 反例

v10 将 v9 的固定 `0.5` soft scale 改成自动 ratio：

```text
effective_scale = code_utility / (code_utility + protected_harm + eps)
```

实现为默认关闭参数：

```bash
--harm-veto-positive-scale-mode evidence-ratio
```

默认仍为 `constant`，所以旧命令不受影响。

v10 的有效 scale 分布：

| count | min | median | mean | max |
|---:|---:|---:|---:|---:|
| 78 | 0.0156 | 0.1837 | 0.1973 | 0.5192 |

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10/quick_tool_memory`

Quick Tool/Memory：

| candidate | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Memory eval_50 F1 |
|---|---:|---:|---:|---:|---:|
| v8 hard fulltraj | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7720 |
| v9 fixed soft 0.5 | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7575 |
| v10 ratio | 0.8800 | 0.8600 | 0.7500 | 0.6250 | 0.7495 |

Code hurt subset：

| candidate | LiveBench BoN acc | LiveBench BoN accum | LiveCodeBench BoN acc | LiveCodeBench BoN accum |
|---|---:|---:|---:|---:|
| v8 hard fulltraj | 0.2500 | 0.4688 | 0.3125 | 0.5378 |
| v9 fixed soft 0.5 | 0.2500 | 0.6250 | 0.6250 | 0.6555 |
| v10 ratio | 0.2500 | 0.6016 | 0.3125 | 0.5378 |

判断：

- Naive ratio 不是更好解。它的 median scale 只有 `0.1837`，比 v9 的固定 `0.5` 保守得多，导致 LiveCodeBench repair 又退回 v8 水平。
- Tool live_parallel 也下降到 `0.75`，说明该公式没有自动带来更好的保护。
- 反例给出更精确的机制结论：Tool 和 Memory 的保护约束不是同一种东西。Tool call span 是格式/调用行为，可能需要更硬的保护；Memory trajectory 与 Code reasoning 共享更多 residual，适合软约束。把所有 protected harm 放进同一个 ratio 是过强假设。

因此当前最有希望的下一步不是继续 ratio 调参，而是做 **task-typed Pareto routing**：

1. Tool call-span harm：作为 hard constraint 或近似 hard constraint。
2. Memory full-trajectory harm：作为 soft constraint。
3. Code pass/fail utility：作为 improvement direction。
4. 对同一个 residual key，按 harm 来源任务类型决定 soft/hard，而不是按一个全局公式处理。

## 16. v11：Task-Typed Pareto Routing

v11 不改证据，只把 harm 约束按来源任务拆开：

```bash
--harm-veto-task-positive-scale tool=0.0
--harm-veto-task-positive-scale memory=0.5
```

含义：

- Tool call-span harm 是格式/调用行为约束，近似 hard constraint。
- Memory full-trajectory harm 和 Code reasoning 共享更多 residual，只做 soft constraint。
- Code pass/fail contrast 仍然是 improvement direction。

代码实现为默认关闭参数：

```bash
--harm-veto-task-positive-scale TASK=SCALE
```

不传该参数时，仍使用全局 `--harm-veto-positive-scale` / `--harm-veto-positive-scale-mode`，旧实验路径不变。

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_tasktyped_v11/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_tasktyped_v11`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_code_spanaware_tmpos_s32_tasktyped_v11/quick_tool_memory`

Gate 改动：

| candidate | harm rule | changed | + | - | mean_abs_delta |
|---|---|---:|---:|---:|---:|
| v8 | all hard | 135 | 61 | 74 | 0.001983 |
| v9 | all soft 0.5 | 205 | 106 | 99 | 0.002639 |
| v11 | tool hard, memory soft | 186 | 96 | 90 | 0.002360 |

v11 的 harm scale 审计：

| harm task | count | scale |
|---|---:|---:|
| tool | 19 | 0.0 |
| memory | 59 | 0.5 |

Quick Tool/Memory：

| candidate | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Memory eval_50 F1 |
|---|---:|---:|---:|---:|---:|
| v8 hard fulltraj | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7720 |
| v9 all soft 0.5 | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7575 |
| v10 ratio | 0.8800 | 0.8600 | 0.7500 | 0.6250 | 0.7495 |
| v11 task-typed | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7701 |

Code hurt subset：

| candidate | LiveBench BoN acc | LiveBench BoN accum | LiveCodeBench BoN acc | LiveCodeBench BoN accum |
|---|---:|---:|---:|---:|
| v8 hard fulltraj | 0.2500 | 0.4688 | 0.3125 | 0.5378 |
| v9 all soft 0.5 | 0.2500 | 0.6250 | 0.6250 | 0.6555 |
| v10 ratio | 0.2500 | 0.6016 | 0.3125 | 0.5378 |
| v11 task-typed | 0.2500 | 0.5469 | 0.3750 | 0.5126 |

判断：

- v11 证明 task-typed protection 能恢复 v8 级别的 Tool/Memory：Tool 全部保持，Memory `0.7701` 接近 v8 的 `0.7720`。
- 但 v11 的 Code repair 明显弱于 v9，说明 Tool-hard 保护会压制一部分 Code 修复 residual。
- 当前性能 Pareto 上，v9 更均衡；机制解释上，v11 更清楚地说明“保护源任务类型”会改变冲突处理。

因此，当前最稳妥的论文表述应该是：

> RCRF 的核心不是某个固定 gate，而是 residual-level evidence can expose a Pareto frontier. Tool behavior spans create near-hard constraints; Memory trajectories create softer constraints; Code pass/fail spans create improvement directions. The framework can instantiate different operating points depending on whether the target favors capability repair or behavior preservation.

后续若继续推进，应该避免再手调 `memory=0.5` 这类常数，而是基于验证集选择 Pareto operating point，或者把 Tool hard / Memory soft 的权重学习为小规模 meta-parameters。

## 17. Residual Conflict Atlas：把证据表升级为归因框架

新增固定诊断脚本：

```bash
PHASE=atlas bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

核心产物：

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_summary.md`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_rows.jsonl`
- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/RCRF/20260522_residual_conflict_atlas.md`

这个 atlas 把每个 `(param_name, expert)` 同时标注为：

- Code repair；
- Tool/Memory protected support；
- Tool/Memory protected harm；
- Code source/span conflict；
- shared positive；
- negative noise。

关键统计：

| role | count |
|---|---:|
| `code_source_conflict_with_behavior` | 167 |
| `code_source_conflict` | 112 |
| `code_repair_only` | 60 |
| `code_negative_but_protected_support` | 58 |
| `shared_positive` | 17 |
| `code_repair_vs_protected_harm` | 16 |

这补上了 RCRF 的机制闭环：

1. 为什么 scalar gate / GRPO gate 很难稳定：因为大部分 residual 不是 task-pure。
2. 为什么 v9 比 v8 更修 Code：它允许更多 `code_repair_only` 和 `code_repair_vs_protected_harm` residual 被推高。
3. 为什么 v11 更保 Tool/Memory：它把 Tool harm 当 hard constraint，把 Memory harm 当 soft constraint。
4. 为什么不能只看 Code expert 系数：Code expert 自身也有大量 source conflict，而 Memory/Tool residual 中也有 Code 相关 span 信号。

因此当前论文主线应从“学习一个 gate”改成：

> residual-level conflict atlas + conservative Pareto routing。

## 18. v12：Atlas Role-Routed Gate

在 atlas 基础上，新增第一版直接 routing 实现：

```bash
PHASE=role_route bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/gates.json`
- summary: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/role_routing_summary.md`
- report: `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/RCRF/20260522_role_routed_gate_v12.md`

v12 不是新的 sweep，而是固定规则：

| role | 操作 |
|---|---|
| `code_repair_only` | raise |
| `shared_positive` | raise |
| `code_repair_vs_protected_harm` | Tool harm 不 raise；Memory harm soft raise |
| `code_negative_noise` | suppress |
| `protected_harm_only` | suppress |
| `code_negative_but_protected_support` | hold |
| `code_source_conflict*` | hold |

统计：

| metric | value |
|---|---:|
| changed | `133/588` |
| positive delta | 73 |
| negative delta | 60 |
| mean abs delta | 0.004696 |

按 role：

| role | changed | + | - |
|---|---:|---:|---:|
| `code_repair_only` | 60 | 60 | 0 |
| `shared_positive` | 9 | 9 | 0 |
| `code_negative_noise` | 56 | 0 | 56 |
| `protected_harm_only` | 4 | 0 | 4 |
| `code_negative_but_protected_support` | 0 | 0 | 0 |
| `code_source_conflict*` | 0 | 0 | 0 |

结构上，v12 的 expert mean coefficient：

| expert | mean |
|---|---:|
| code | 0.9026 |
| memory | 0.9831 |
| tool | 1.0049 |

这看起来像 code 系数较低，但它不是简单压 Code：它压的是 atlas 认为的 Code-negative noise，同时提高 late-layer Code repair residual。后续评测如果失败，最有价值的诊断点是：`code_negative_noise` 是否被过度压低，说明 Code negative role 需要按 source/span 再拆。

### 18.1 v12 evaluation

已完成 v12 bake 和最小闭环评测。

| candidate | Tool quick | Memory eval_50 F1 | LiveBench hurt BoN | LiveCodeBench hurt BoN |
|---|---|---:|---|---|
| v9 soft | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v11 task-typed | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7701 | `0.2500 / 0.5469` | `0.3750 / 0.5126` |
| v12 role-routed | `0.880 / 0.865 / 0.8125 / 0.625` | 0.7627 | `0.2500 / 0.4453` | `0.4375 / 0.5294` |

结论：

- v12 成功保住 Tool / Memory，说明 residual role routing 对 behavior preservation 有效。
- v12 Code 明显弱于 v9，说明“负向 Code contrast -> suppress”这条规则不成立或至少太粗。
- 这反而强化了 atlas 的必要性：正向 Code repair residual 比负向 Code noise 更可靠，负向证据必须继续拆 source/span。

下一步规则应改成：

1. Positive-only routing 作为下一个主 candidate。
2. `code_negative_noise` 默认 hold。
3. 只有当某个 negative residual 在多个 Code source/span 上稳定负向、且没有 protected support 时，才考虑 suppress。

### 18.2 v13 positive-only evaluation

v13 按上面的第 1-2 条做了隔离实验：

```bash
PHASE=generate CANDIDATES=v13 bash skill/command/run_20260522_rcrf_pareto_frontier.sh
PHASE=bake CANDIDATES=v13 bash skill/command/run_20260522_rcrf_pareto_frontier.sh
PHASE=quick_eval CANDIDATES=v13 TOOL_GPU=0 TOOL_PORT=8153 MEMORY_GPU_IDS=1 MEMORY_DATASETS=eval_50 bash skill/command/run_20260522_rcrf_pareto_frontier.sh
PHASE=code_hurt_eval CANDIDATES=v13 CODE_GPU=2 bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

v13 gate：

- changed `73/588`
- positive delta `73`
- negative delta `0`
- `code_negative_noise` 全部 hold
- `protected_harm_only` 全部 hold
- `code_source_conflict*` 全部 hold

结果：

| candidate | Tool quick | Memory eval_50 F1 | LiveBench hurt BoN | LiveCodeBench hurt BoN |
|---|---|---:|---|---|
| v9 soft | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v12 role-routed | `0.880 / 0.865 / 0.8125 / 0.625` | 0.7627 | `0.2500 / 0.4453` | `0.4375 / 0.5294` |
| v13 positive-only | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7564 | `0.1250 / 0.2422` | `0.3125 / 0.5294` |

v13 的负结果更关键：

- Positive-only 可以保持 Tool/Memory。
- 但 Code 没有恢复，LiveBench hurt 明显下降。
- 因此 v12 的问题不只是 negative suppression。
- 更核心的问题是：`code_source_conflict*` 不能全部 hold。v9 的 Code gain 很可能来自其中一部分 source-conflict residual。

方法更新：

RCRF 下一步应从 role routing 升级为 source/span-conditioned conflict routing：

```text
(param_name, expert, role)
  -> (param_name, expert, role, source/span)
```

这比继续调 `max_delta` 更符合第一性原理：Code 的不同评测 source 需要不同 residual，冲突本身不是噪声，而是需要被分解的能力信号。
