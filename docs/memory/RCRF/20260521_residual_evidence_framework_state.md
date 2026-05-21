# 2026-05-21 RCRF Residual Evidence Framework State

## 当前主线

目标从“调 gate 系数”收束为：

> 对每个 `(param_name, expert)` residual key 建立 outcome-aware utility / harm / conflict 证据，再只对证据充分且非冲突的位置做保守 routing。

这比 task-level scalar 更符合当前观察：Code source/span 内部大量冲突，Memory/Tool 也需要 behavior-span 保护。

## 已有核心产物

- Code source/span conflict 可视化：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521`
- Residual evidence table：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/residual_evidence_table_20260521`
- Tool/Memory behavior span manifest：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/tool_memory_20260521`
- Evidence-routed first candidate gate：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_evidence_routed_v1/gates.json`
- Tool/Memory positive s8 probe：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s8_20260521/signed_utility_summary.json`
- Behavior-protected evidence-routed gate：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_evidence_routed_tmpos_s8_v1/gates.json`
- Baked checkpoint：
  `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_evidence_routed_tmpos_s8_v1`
- Quick eval：
  `/tmp/shared-storage/ExpertGym/rcrf/eval/evidence_routed_tmpos_s8_v1/quick_tool_memory`
- Tool/Memory positive s32 probe：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s32_20260521/signed_utility_summary.json`
- Behavior-protected evidence-routed s32 gate：
  `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_evidence_routed_tmpos_s32_v1/gates.json`
- s32 baked checkpoint：
  `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_evidence_routed_tmpos_s32_v1`
- s32 quick eval：
  `/tmp/shared-storage/ExpertGym/rcrf/eval/evidence_routed_tmpos_s32_v1/quick_tool_memory`

## 当前统计

Residual evidence table 覆盖 `588 = 28 layers * 7 modules * 3 experts`。

- `hold_conflict`: 400
- `keep_or_raise`: 76
- `suppress`: 43
- `preserve`: 17
- `no_decision`: 52

第一版 evidence-routed gate 只改 `56/588` 个 key：

- 增强：35
- 抑制：21
- code expert changed：19
- memory expert changed：25
- tool expert changed：12

合入 Tool/Memory positive s8 后：

- `hold_conflict`: 466
- `keep_or_raise`: 41
- `suppress`: 12
- `preserve`: 39
- `no_decision`: 30

新版 gate `rcrf_evidence_routed_tmpos_s8_v1` 只改 `25/588`：

- 增强：20
- 抑制：5
- code expert changed：19
- memory expert changed：5
- tool expert changed：1

Quick eval：

- Tool BFCL quick: parallel `0.8800`, parallel_multiple `0.8550`, live_parallel `0.6875`, live_parallel_multiple `0.6250`
- Memory eval_50 avg_f1 `0.7648`, EM `0.6016`, sub-EM `0.7969`

解释：Tool/Memory behavior utility 把一批原本由 Code contrast 推动的 Memory/Tool residual 改判为冲突或 preserve，实际 quick eval 没有发现 Tool/Memory 崩坏。

合入 Tool/Memory positive s32 后：

- `hold_conflict`: 463
- `keep_or_raise`: 44
- `suppress`: 12
- `preserve`: 39
- `no_decision`: 30

s32 gate `rcrf_evidence_routed_tmpos_s32_v1` 只改 `29/588`：

- 增强：24
- 抑制：5
- code expert changed：19
- memory expert changed：7
- tool expert changed：3

s32 Quick eval：

- Tool BFCL quick: parallel `0.8800`, parallel_multiple `0.8600`, live_parallel `0.8125`, live_parallel_multiple `0.6250`
- Memory eval_50 avg_f1 `0.7802`, EM `0.6172`, sub-EM `0.8125`

解释：s32 比 s8 更稳。Tool live_parallel 明显回升，Memory F1 也提升；同时 gate 仍只动 29 个 residual key。这是当前 RCRF 主线最有价值的正结果：更多行为 span 证据可以减少误抑制，并保持 residual-level routing 的简洁性。

## 新增脚本

- `scripts/analysis/build_behavior_span_manifest.py`
  - 从 rollout / inference 输出构建 `behavior_positive.jsonl`、`behavior_negative.jsonl`。
  - 输出可直接喂给 `probe_signed_utility.py`。

- `scripts/analysis/build_residual_evidence_table.py`
  - 对齐 Code source/span contrast、behavior utility、gate delta。
  - 支持多个 `--signed-utility-summary`，按 count 加权合并。

- `scripts/attention_pauh/build_evidence_routed_residual_gates.py`
  - 从 evidence table 生成保守 gate。
  - 默认只 materialize `keep_or_raise/suppress`。
  - `hold_conflict/preserve/no_decision` 默认保持 base。

## 下一步

1. 已完成 Tool/Memory behavior-positive signed utility 的 s8 和 s32 版本。后续若扩展，应优先增加 Code 侧 pass/fail span，而不是继续调 s8/s32 阈值。

2. 已把 s32 `signed_utility_summary.json` 加进 evidence table，并生成 / bake / quick-eval 了 `rcrf_evidence_routed_tmpos_s32_v1`。
3. 下一步：在 Code hurt subset 上验证 s32 是否修复 Code hurt；若 Tool/Memory 仍稳定，再跑正式 Code。

## 方法判断

当前最有论文价值的发现：

- Expert task vector 不是 task-pure。
- 大量 residual key 的效用依赖 source/span。
- 粗粒度保护某个 expert 会误保留有害 residual。
- 可行方法应是 residual-level attribution + conservative routing，而不是 sweep 或直接 RL scalar gate。
