# 2026-05-22 RCF-BC Validation Plan

## 目的

这个文件把 attribution ledger 转成下一轮可执行验证卡片。它不改 gate、不跑评测，只规定哪些 residual group 最值得做反事实实验，以及通过什么指标判断该机制是否成立。

## 闭环协议

- select one validation card
- build the minimal isolated gate intervention for its representative rows
- bake checkpoint
- run Tool/Memory quick first
- run Code hurt only if Tool/Memory do not fail the agreed guardrail
- write the counterfactual result back into the ledger/effect table

## 优先级概览

| priority | rows | changed |
|---|---:|---:|
| `high` | 172 | 91 |
| `low` | 29 | 0 |
| `medium` | 387 | 114 |

## P0 验证卡片

| rank | card | action | rows | changed | score | validation |
|---:|---|---|---:|---:|---:|---|
| 1 | `high-retain-continuous-field-code-source-conflict-code-early-00-09-attention` | `retain_continuous_field` | 6 | 6 | 291.95 | source_span_counterfactual |
| 2 | `high-retain-continuous-field-code-source-conflict-code-early-00-09-mlp` | `retain_continuous_field` | 7 | 7 | 282.68 | source_span_counterfactual |
| 3 | `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-middle-10-19-mlp` | `retain_with_behavior_constraint` | 1 | 1 | 245.76 | pareto_boundary_scale_check |
| 4 | `high-retain-continuous-field-code-source-conflict-memory-early-00-09-attention` | `retain_continuous_field` | 5 | 5 | 241.55 | source_span_counterfactual |
| 5 | `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-late-20-27-attention` | `retain_with_behavior_constraint` | 6 | 6 | 235.22 | pareto_boundary_scale_check |
| 6 | `high-retain-continuous-field-code-source-conflict-memory-early-00-09-mlp` | `retain_continuous_field` | 4 | 4 | 234.39 | source_span_counterfactual |
| 7 | `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-late-20-27-mlp` | `retain_with_behavior_constraint` | 4 | 4 | 231.78 | pareto_boundary_scale_check |
| 8 | `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-early-00-09-mlp` | `retain_with_behavior_constraint` | 2 | 2 | 225.86 | pareto_boundary_scale_check |
| 9 | `high-retain-capability-delta-clean-code-repair-tool-late-20-27-attention` | `retain_capability_delta` | 1 | 1 | 220.09 | capability_drop_or_restore |
| 10 | `high-protect-behavior-anchor-code-negative-with-behavior-support-memory-late-20-27-mlp` | `protect_behavior_anchor` | 7 | 1 | 218.44 | behavior_anchor_drop_test |
| 11 | `high-behavior-guard-code-negative-noise-memory-early-00-09-mlp` | `behavior_guard` | 6 | 3 | 217.28 | veto_release_test |
| 12 | `high-protect-behavior-anchor-code-negative-with-behavior-support-memory-middle-10-19-mlp` | `protect_behavior_anchor` | 2 | 0 | 214.63 | behavior_anchor_drop_test |

## 卡片详情

### 1. `high-retain-continuous-field-code-source-conflict-code-early-00-09-attention`

- action: `retain_continuous_field`
- priority: `high`
- group: `code_source_conflict / code / early_00_09 / attention`
- rows: `6`, changed: `6`
- score: `291.95`
- hypothesis: mixed source/span row 需要连续场；离散 source dominant routing 会丢失能力。
- success: 按 source/span 分组干预能解释 v18 优于 hard routing 的 Code 差异。
- failure read: 若 source/span 分组没有指标差异，continuous field 的证据需要回到 probe 设计。
- representative keys:
  - `model.layers.0.self_attn.q_proj.weight::code`
  - `model.layers.0.self_attn.k_proj.weight::code`
  - `model.layers.1.self_attn.v_proj.weight::code`
  - `model.layers.1.self_attn.o_proj.weight::code`
  - `model.layers.2.self_attn.o_proj.weight::code`
  - `model.layers.3.self_attn.o_proj.weight::code`

### 2. `high-retain-continuous-field-code-source-conflict-code-early-00-09-mlp`

- action: `retain_continuous_field`
- priority: `high`
- group: `code_source_conflict / code / early_00_09 / mlp`
- rows: `7`, changed: `7`
- score: `282.68`
- hypothesis: mixed source/span row 需要连续场；离散 source dominant routing 会丢失能力。
- success: 按 source/span 分组干预能解释 v18 优于 hard routing 的 Code 差异。
- failure read: 若 source/span 分组没有指标差异，continuous field 的证据需要回到 probe 设计。
- representative keys:
  - `model.layers.0.mlp.up_proj.weight::code`
  - `model.layers.0.mlp.gate_proj.weight::code`
  - `model.layers.0.mlp.down_proj.weight::code`
  - `model.layers.1.mlp.down_proj.weight::code`
  - `model.layers.1.mlp.gate_proj.weight::code`
  - `model.layers.2.mlp.up_proj.weight::code`
  - `model.layers.2.mlp.down_proj.weight::code`

### 3. `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-middle-10-19-mlp`

- action: `retain_with_behavior_constraint`
- priority: `high`
- group: `code_repair_with_behavior_harm / memory / middle_10_19 / mlp`
- rows: `1`, changed: `1`
- score: `245.76`
- hypothesis: 这些 row 同时有能力和行为风险，是 Pareto 边界而非单任务能力。
- success: 连续缩放能沿 Code 与 Tool/Memory trade-off 移动；hard drop 或 hard keep 都更差。
- failure read: 若缩放不影响任何正式指标，说明 behavior harm probe 只是相关性。
- representative keys:
  - `model.layers.10.mlp.up_proj.weight::memory`

### 4. `high-retain-continuous-field-code-source-conflict-memory-early-00-09-attention`

- action: `retain_continuous_field`
- priority: `high`
- group: `code_source_conflict / memory / early_00_09 / attention`
- rows: `5`, changed: `5`
- score: `241.55`
- hypothesis: mixed source/span row 需要连续场；离散 source dominant routing 会丢失能力。
- success: 按 source/span 分组干预能解释 v18 优于 hard routing 的 Code 差异。
- failure read: 若 source/span 分组没有指标差异，continuous field 的证据需要回到 probe 设计。
- representative keys:
  - `model.layers.0.self_attn.k_proj.weight::memory`
  - `model.layers.1.self_attn.v_proj.weight::memory`
  - `model.layers.1.self_attn.o_proj.weight::memory`
  - `model.layers.9.self_attn.o_proj.weight::memory`
  - `model.layers.3.self_attn.k_proj.weight::memory`

### 5. `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-late-20-27-attention`

- action: `retain_with_behavior_constraint`
- priority: `high`
- group: `code_repair_with_behavior_harm / memory / late_20_27 / attention`
- rows: `6`, changed: `6`
- score: `235.22`
- hypothesis: 这些 row 同时有能力和行为风险，是 Pareto 边界而非单任务能力。
- success: 连续缩放能沿 Code 与 Tool/Memory trade-off 移动；hard drop 或 hard keep 都更差。
- failure read: 若缩放不影响任何正式指标，说明 behavior harm probe 只是相关性。
- representative keys:
  - `model.layers.24.self_attn.q_proj.weight::memory`
  - `model.layers.20.self_attn.q_proj.weight::memory`
  - `model.layers.27.self_attn.q_proj.weight::memory`
  - `model.layers.20.self_attn.o_proj.weight::memory`
  - `model.layers.21.self_attn.q_proj.weight::memory`
  - `model.layers.27.self_attn.v_proj.weight::memory`

### 6. `high-retain-continuous-field-code-source-conflict-memory-early-00-09-mlp`

- action: `retain_continuous_field`
- priority: `high`
- group: `code_source_conflict / memory / early_00_09 / mlp`
- rows: `4`, changed: `4`
- score: `234.39`
- hypothesis: mixed source/span row 需要连续场；离散 source dominant routing 会丢失能力。
- success: 按 source/span 分组干预能解释 v18 优于 hard routing 的 Code 差异。
- failure read: 若 source/span 分组没有指标差异，continuous field 的证据需要回到 probe 设计。
- representative keys:
  - `model.layers.6.mlp.down_proj.weight::memory`
  - `model.layers.5.mlp.up_proj.weight::memory`
  - `model.layers.1.mlp.down_proj.weight::memory`
  - `model.layers.1.mlp.gate_proj.weight::memory`

### 7. `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-late-20-27-mlp`

- action: `retain_with_behavior_constraint`
- priority: `high`
- group: `code_repair_with_behavior_harm / memory / late_20_27 / mlp`
- rows: `4`, changed: `4`
- score: `231.78`
- hypothesis: 这些 row 同时有能力和行为风险，是 Pareto 边界而非单任务能力。
- success: 连续缩放能沿 Code 与 Tool/Memory trade-off 移动；hard drop 或 hard keep 都更差。
- failure read: 若缩放不影响任何正式指标，说明 behavior harm probe 只是相关性。
- representative keys:
  - `model.layers.27.mlp.up_proj.weight::memory`
  - `model.layers.26.mlp.gate_proj.weight::memory`
  - `model.layers.26.mlp.up_proj.weight::memory`
  - `model.layers.24.mlp.gate_proj.weight::memory`

### 8. `high-retain-with-behavior-constraint-code-repair-with-behavior-harm-memory-early-00-09-mlp`

- action: `retain_with_behavior_constraint`
- priority: `high`
- group: `code_repair_with_behavior_harm / memory / early_00_09 / mlp`
- rows: `2`, changed: `2`
- score: `225.86`
- hypothesis: 这些 row 同时有能力和行为风险，是 Pareto 边界而非单任务能力。
- success: 连续缩放能沿 Code 与 Tool/Memory trade-off 移动；hard drop 或 hard keep 都更差。
- failure read: 若缩放不影响任何正式指标，说明 behavior harm probe 只是相关性。
- representative keys:
  - `model.layers.1.mlp.up_proj.weight::memory`
  - `model.layers.3.mlp.gate_proj.weight::memory`

### 9. `high-retain-capability-delta-clean-code-repair-tool-late-20-27-attention`

- action: `retain_capability_delta`
- priority: `high`
- group: `clean_code_repair / tool / late_20_27 / attention`
- rows: `1`, changed: `1`
- score: `220.09`
- hypothesis: 这些 row 是最干净的能力 residual；如果把它们投回 base，Code hurt 应该下降。
- success: drop/restore 对照中 Code hurt BoN 下降，同时 Tool/Memory 不出现足以解释该下降的反向收益。
- failure read: 若 drop 后 Code 不降，说明当前 pass/fail contrast 不是因果能力证据。
- representative keys:
  - `model.layers.27.self_attn.k_proj.weight::tool`

### 10. `high-protect-behavior-anchor-code-negative-with-behavior-support-memory-late-20-27-mlp`

- action: `protect_behavior_anchor`
- priority: `high`
- group: `code_negative_with_behavior_support / memory / late_20_27 / mlp`
- rows: `7`, changed: `1`
- score: `218.44`
- hypothesis: 这些 row 承载 Tool/Memory behavior，不能因为 Code 负证据就压低。
- success: drop 或 shrink 后 Tool/Memory quick 下降，证明它们是行为 anchor。
- failure read: 若 drop 后行为不降，应从 behavior constraint 中移除该组。
- representative keys:
  - `model.layers.27.mlp.down_proj.weight::memory`
  - `model.layers.21.mlp.down_proj.weight::memory`
  - `model.layers.23.mlp.up_proj.weight::memory`
  - `model.layers.20.mlp.down_proj.weight::memory`
  - `model.layers.21.mlp.up_proj.weight::memory`
  - `model.layers.20.mlp.up_proj.weight::memory`
  - `model.layers.20.mlp.gate_proj.weight::memory`

### 11. `high-behavior-guard-code-negative-noise-memory-early-00-09-mlp`

- action: `behavior_guard`
- priority: `high`
- group: `code_negative_noise / memory / early_00_09 / mlp`
- rows: `6`, changed: `3`
- score: `217.28`
- hypothesis: 这些 row 的主要作用是行为 guard，不应该为了 Code 单侧收益放大。
- success: 释放 veto 后 Tool/Memory 下降，或 Code 收益不足以补偿行为损失。
- failure read: 若释放后三项都升，当前 veto 过强。
- representative keys:
  - `model.layers.6.mlp.up_proj.weight::memory`
  - `model.layers.8.mlp.gate_proj.weight::memory`
  - `model.layers.5.mlp.down_proj.weight::memory`
  - `model.layers.6.mlp.gate_proj.weight::memory`
  - `model.layers.4.mlp.down_proj.weight::memory`
  - `model.layers.4.mlp.gate_proj.weight::memory`

### 12. `high-protect-behavior-anchor-code-negative-with-behavior-support-memory-middle-10-19-mlp`

- action: `protect_behavior_anchor`
- priority: `high`
- group: `code_negative_with_behavior_support / memory / middle_10_19 / mlp`
- rows: `2`, changed: `0`
- score: `214.63`
- hypothesis: 这些 row 承载 Tool/Memory behavior，不能因为 Code 负证据就压低。
- success: drop 或 shrink 后 Tool/Memory quick 下降，证明它们是行为 anchor。
- failure read: 若 drop 后行为不降，应从 behavior constraint 中移除该组。
- representative keys:
  - `model.layers.12.mlp.down_proj.weight::memory`
  - `model.layers.12.mlp.up_proj.weight::memory`

## 产物

- cards CSV: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_validation_plan_20260522/validation_cards.csv`
- cards JSONL: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_validation_plan_20260522/validation_cards.jsonl`
- summary JSON: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_validation_plan_20260522/validation_plan_summary.json`
- ledger: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_attribution_ledger_20260522/rcrf_attribution_ledger.csv`
