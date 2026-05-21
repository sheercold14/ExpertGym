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
