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
3. 已在 Code hurt subset 上验证 s32：Tool/Memory 强，但 Code repair 弱。随后构建 v6/v7，保留 Code pass/fail repair，并加入 Tool/Memory behavior preserve / harm-veto。

## 2026-05-21 v6/v7 闭环

### s32 Code hurt

- `LiveBenchCodeHurtRcrfVsTa16`: BoN acc `0.1875`, BoN accum `0.4219`
- `LiveCodeBenchCodeHurtRcrfVsTa16`: BoN acc `0.4375`, BoN accum `0.6303`

解释：s32 保护 Tool/Memory，但只改 `29/588`，不足以修 Code。

### v6: Code repair + Tool/Memory preserve floor

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_preserve_v6/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_preserve_v6`

结果：

- changed `310/588`
- Tool BFCL quick: `0.8800 / 0.8600 / 0.7500 / 0.6250`
- Memory eval_50 F1: `0.7528`
- LiveBench hurt16: BoN acc `0.3125`, accum `0.4922`
- LiveCodeBench hurt16: BoN acc `0.6250`, accum `0.7395`

判断：保留 v2 Code repair 是必要的；只加 preserve floor 能增强 Code，但 Memory 仍掉。

### v7: Code repair + preserve floor + harm veto

代码变更：

- `scripts/attention_pauh/build_contrast_aware_residual_gates.py`
- 新增默认关闭参数：
  - `--harm-veto-summary`
  - `--harm-veto-task`
  - `--harm-veto-expert`
  - `--harm-veto-min-normalized-harm`
  - `--harm-veto-positive-scale`

语义：

- preserve: 不降低 Tool/Memory 有用 residual。
- harm-veto: 不升高 Tool/Memory 有害 residual。
- 默认关闭，旧实验路径不受影响。

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_preserve_harmveto_v7/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_preserve_harmveto_v7`

结果：

- changed `201/588`
- Tool BFCL quick: `0.8800 / 0.8650 / 0.8125 / 0.6250`
- Memory eval_50 F1: `0.7425`
- LiveBench hurt16: BoN acc `0.2500`, accum `0.5156`
- LiveCodeBench hurt16: BoN acc `0.7500`, accum `0.7731`

判断：

- v7 能保持 Tool，并几乎恢复 v2 的 LiveCodeBench hurt 修复能力。
- Memory 下降更明显，说明当前 Memory behavior-positive signature 不足以保护 HotpotQA 能力。
- 下一步应重做 Memory utility，必须覆盖 update turns + final turn 的完整轨迹 span；否则静态 gate 会继续在 Code repair 和 Memory 之间冲突。

## 2026-05-21 Memory full-trajectory diagnostic

当前判断：

- Memory-Code 冲突最可能发生在 Memory 的中间 update turns，而不是 boxed final answer。
- Code repair 方向可能鼓励更 aggressive 的局部推理、自校验和重写；这些对 Code 有利，但会伤害 Memory 的证据保真、实体绑定和多跳关系约束。
- 因此 Tool 可以用 tool-call span 保护，但 Memory 不能只用 final/signature span。

新增默认关闭能力：

- `scripts/analysis/build_behavior_span_manifest.py`
  - `--memory-response-mode final`：旧行为。
  - `--memory-response-mode memory-plus-final`：用 memory 字段加 final answer。
  - `--memory-response-mode full-trajectory`：用 `chunk_rounds[*].response` 加 final answer。
- `scripts/attention_pauh/build_contrast_aware_residual_gates.py`
  - `--preserve-summary` / `--harm-veto-summary` 支持重复传入，以便同时使用 Tool span 和 Memory full trajectory span。

正在运行的诊断：

- manifest: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/behavior_span_manifests/memory_fulltraj_20260521`
- probe: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/memory_fulltraj_positive_s32_20260521`

预期分叉：

- 如果 full trajectory utility 能恢复 Memory，RCRF 主线成立：Code pass/fail contrast + Tool call-span protection + Memory trajectory protection。
- 如果不能恢复 Memory，说明单一全局 residual gate 对 Memory/Code 冲突表达不足，应转向最小 task-conditional anchor，而不是继续调静态阈值。

## 2026-05-21 v8 result

v8 = v7 + Memory full-trajectory preserve/harm-veto summary.

Artifacts:

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_v8/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_v8`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_code_spanaware_tmpos_s32_memoryfull_v8/quick_tool_memory`
- CURE hurt result txt:
  - `/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_code_spanaware_tmpos_s32_memoryfull_v8-LiveBenchCodeHurtRcrfVsTa16.txt`
  - `/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_code_spanaware_tmpos_s32_memoryfull_v8-LiveCodeBenchCodeHurtRcrfVsTa16.txt`

Results:

- changed `135/588`
- Tool BFCL quick: `0.8800 / 0.8600 / 0.8125 / 0.6250`
- Memory eval_50 F1: `0.7720`
- LiveBench hurt16: BoN acc `0.2500`, accum `0.4688`
- LiveCodeBench hurt16: BoN acc `0.3125`, accum `0.5378`

Interpretation:

- Full Memory trajectory protection works for Memory: v7 `0.7425` -> v8 `0.7720`.
- Tool remains stable.
- Code repair is over-suppressed, especially LiveCodeBench. Hard harm-veto on full Memory trajectory is too conservative.
- Best next step is not a threshold sweep: replace full-trajectory hard veto with soft Pareto routing, preserving high-confidence Memory owner utility while allowing strong Code pass/fail contrast to survive.

## 2026-05-22 v9 soft Pareto result

v9 = v8 evidence, but `--harm-veto-positive-scale 0.5`.

Artifacts:

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/quick_tool_memory`
- CURE hurt result txt:
  - `/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9-LiveBenchCodeHurtRcrfVsTa16.txt`
  - `/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.checkpoints.rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9-LiveCodeBenchCodeHurtRcrfVsTa16.txt`

Results:

- changed `205/588`
- Tool BFCL quick: `0.8800 / 0.8550 / 0.8125 / 0.6250`
- Memory eval_50 F1: `0.7575`
- LiveBench hurt16: BoN acc `0.2500`, accum `0.6250`
- LiveCodeBench hurt16: BoN acc `0.6250`, accum `0.6555`

Interpretation:

- Soft routing recovers Code relative to v8 while retaining some Memory benefit over v7.
- This supports the actual method insight: do not make protected-task harm a binary veto; use residual-level multi-objective evidence.
- Next implementation should replace fixed soft scale with an evidence-ratio rule, e.g. `code_utility / (code_utility + protected_harm + eps)`, so the trade-off is determined by residual-level evidence rather than a tuned constant.

## 2026-05-22 v10 evidence-ratio counterexample

v10 = v8 evidence, but `--harm-veto-positive-scale-mode evidence-ratio`.

Artifacts:

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_code_spanaware_tmpos_s32_memoryfull_ratio_v10/quick_tool_memory`

Results:

- effective harm scale count `78`, min `0.0156`, median `0.1837`, mean `0.1973`, max `0.5192`
- Tool BFCL quick: `0.8800 / 0.8600 / 0.7500 / 0.6250`
- Memory eval_50 F1: `0.7495`
- LiveBench hurt16: BoN acc `0.2500`, accum `0.6016`
- LiveCodeBench hurt16: BoN acc `0.3125`, accum `0.5378`

Interpretation:

- Naive evidence ratio is too conservative and collapses LiveCodeBench repair back to v8.
- It also hurts Tool live_parallel, so it is not a better protection rule.
- The more precise next hypothesis is task-typed protection: Tool call-span harm should be hard or near-hard; Memory full-trajectory harm should be soft because it shares reasoning residuals with Code.

## 2026-05-22 v11 task-typed Pareto result

v11 = v8/v9/v10 evidence, but task-specific harm-veto scale:

- tool harm: `0.0`
- memory harm: `0.5`

Artifacts:

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_tasktyped_v11/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_tasktyped_v11`
- quick eval: `/tmp/shared-storage/ExpertGym/rcrf/eval/full_suite/rcrf_code_spanaware_tmpos_s32_tasktyped_v11/quick_tool_memory`

Results:

- changed `186/588`
- harm scale audit: `tool=19 entries @0.0`, `memory=59 entries @0.5`
- Tool BFCL quick: `0.8800 / 0.8600 / 0.8125 / 0.6250`
- Memory eval_50 F1: `0.7701`
- LiveBench hurt16: BoN acc `0.2500`, accum `0.5469`
- LiveCodeBench hurt16: BoN acc `0.3750`, accum `0.5126`

Interpretation:

- Task-typed protection restores Tool/Memory nearly to v8 while keeping a little more Code than hard v8.
- It is worse than v9 on Code, so v9 remains the better performance Pareto point.
- v11 is still valuable mechanistically: it confirms Tool call-span harm behaves like a harder constraint, while Memory trajectory harm can be softened.
- Current RCRF story should present a Pareto frontier, not a single always-best static gate.

## 2026-05-22 method consolidation

Paper-oriented method blueprint:

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/RCRF/20260522_rcrf_method_blueprint.md`

This document is the current canonical RCRF story:

1. Unit: `(param_name, expert)` residual entry, 588 entries total.
2. Code evidence: same-prompt pass/fail contrast over Code hurt subsets.
3. Tool evidence: tool-call behavior span, near-hard protection.
4. Memory evidence: full update trajectory + final answer, soft protection.
5. Method form: residual-level Pareto routing, not scalar gate optimization.
6. Current operating points:
   - v9: best balanced point.
   - v11: stronger behavior-preserving point.
   - v8: hard-protection endpoint.
   - v10: negative result for naive evidence ratio.

Validation added:

- `tests/test_attention_pauh.py` now covers task-specific harm scale parsing and evidence-ratio scale math.
- Targeted test command passed with stdlib unittest: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python tests/test_attention_pauh.py`, `20 tests OK`.

Important next research rule:

- Do not keep searching for a single magic gate.
- Report RCRF as a framework that exposes and controls an ability-preservation/repair Pareto frontier.

## 方法判断

当前最有论文价值的发现：

- Expert task vector 不是 task-pure。
- 大量 residual key 的效用依赖 source/span。
- 粗粒度保护某个 expert 会误保留有害 residual。
- 可行方法应是 residual-level attribution + conservative routing，而不是 sweep 或直接 RL scalar gate。
- 当前更精确的结论：Tool 可以用 tool-call span 保护；Code 需要 pass/fail span contrast；Memory 必须用完整轨迹 span，否则会被 Code repair 牺牲。

## 2026-05-22 reproducibility harness

新增规范入口：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/skill/command/run_20260522_rcrf_pareto_frontier.sh`

用途：

- 复现 v8/v9/v10/v11 gate generation。
- bake selected gates。
- 跑 Tool quick + Memory eval_50。
- 跑 Code hurt16 regression。

关键用法：

```bash
DRY_RUN=1 PHASE=generate CANDIDATES=v9 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

```bash
PHASE=all CANDIDATES=v9 \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

验证：

- `bash -n` 通过。
- `DRY_RUN=1` 的 `generate`、`quick_eval`、`code_hurt_eval` 均可打印完整命令。

研究定位：

- 这不是调参脚本，而是固定当前 Pareto-frontier 证据链的复现入口。
- 后续新方法必须能在这个 harness 里替换一个明确 phase 或新增一个 candidate，而不是散落成临时命令。

## 2026-05-22 residual conflict atlas

新增脚本：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/analysis/build_rcrf_conflict_atlas.py`

复现入口：

```bash
PHASE=atlas bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_summary.md`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_conflict_atlas_20260522/residual_conflict_atlas_rows.jsonl`
- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/RCRF/20260522_residual_conflict_atlas.md`

核心统计：

- `row_count=588`
- `code_source_conflict_with_behavior=167`
- `code_source_conflict=112`
- `code_repair_only=60`
- `code_negative_but_protected_support=58`
- `shared_positive=17`
- `code_repair_vs_protected_harm=16`

当前最重要的机制解释：

1. 干净 Code repair residual 很少，只有 `60/588`。
2. 真正协同 residual 更少，`shared_positive=17/588`。
3. 大量 residual 的 Code source/span 符号冲突，说明 Code 不能用单一 scalar gate 表达。
4. Memory expert 是主要冲突来源：`code_source_conflict_with_behavior=104`。
5. Tool residual 里有少量协同能力，但更多是需要保护的 behavior span。

方法含义：

- RCRF 应被表述为 residual-level conflict atlas + Pareto routing。
- 论文故事不应该是“找到最优 gate 系数”，而是“识别能力 residual、行为保护 residual、冲突 residual，并用简单 routing 暴露能力修复/行为保留的 Pareto frontier”。

## 2026-05-22 v12 role-routed gate

新增脚本：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/analysis/build_rcrf_role_routed_gates.py`

复现：

```bash
PHASE=role_route bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

产物：

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/gates.json`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_v12/role_routing_summary.md`
- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/RCRF/20260522_role_routed_gate_v12.md`

规则：

- `code_repair_only` / `shared_positive`: 提高。
- Tool harm: 不提高 Code positive delta。
- Memory harm: 软提高。
- `code_negative_noise` / `protected_harm_only`: 压低。
- `code_negative_but_protected_support`: 保持。
- `code_source_conflict*`: 保持。

统计：

- changed `133/588`
- positive delta `73`
- negative delta `60`
- mean abs delta `0.004696`
- expert mean coefficient:
  - code `0.9026`
  - memory `0.9831`
  - tool `1.0049`

判断：

- v12 是第一版真正的 `attribution -> role -> routing` 实现。
- 它不依赖评测指标倒推，但也可能过度压 `code_negative_noise`；如果评测不佳，这将成为机制性负结果，而不是无信息失败。
- 下一步若有 GPU，应先 bake v12，再跑 Tool quick + Memory eval_50；如果不崩，再跑 Code hurt16。

## 2026-05-22 v12 evaluation result

已 bake：

- `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_role_routed_v12`

Tool quick：

- parallel `0.8800`
- parallel_multiple `0.8650`
- live_parallel `0.8125`
- live_parallel_multiple `0.6250`

Memory：

- eval_50 F1 `0.7626831501831502`

Code hurt16：

- LiveBenchCodeHurtRcrfVsTa16 BoN `0.2500 / 0.4453125`
- LiveCodeBenchCodeHurtRcrfVsTa16 BoN `0.4375 / 0.5294117647058824`

对比判断：

- Tool 与 v9/v11 一样稳，parallel_multiple 甚至略高。
- Memory 在 v9 和 v11 之间。
- Code 明显弱于 v9，LiveCodeBench 只略高于 v11。

机制解释：

- v12 的 positive routing 是有意义的：protected behavior 没有崩。
- v12 的 negative routing 过强：`code_negative_noise` 全部被压，说明负向 Code contrast 不能直接当 pruning 信号。
- 下一步应该做 positive-only / source-conditioned negative routing：
  - 继续提高 `code_repair_only` 和 `shared_positive`。
  - 不默认压 `code_negative_noise`。
  - 对 Code negative 必须区分 LiveBench prompt / reasoning / LiveCodeBench code / prompt 来源。

当前论文观点更新：

- RCRF atlas 的价值成立；v12 证明 role-routing 能保 Tool/Memory。
- 真正困难在 Code negative evidence：正向 pass/fail contrast 更可靠，负向 contrast 需要更细粒度 span-conditioned 处理。

## 2026-05-22 v13 positive-only result

动机：

- v12 保 Tool/Memory 但 Code 弱。
- 怀疑原因是 `code_negative_noise` suppression 过强。
- v13 只做 positive routing，不压任何 negative residual。

配置：

- candidate: `v13`
- variant: `rcrf_role_routed_positive_only_v13`
- `--code-negative-action hold`
- `--protected-harm-action hold`

产物：

- gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_role_routed_positive_only_v13/gates.json`
- checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_role_routed_positive_only_v13`
- report: `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/RCRF/20260522_role_routed_positive_only_v13.md`

Gate：

- changed `73/588`
- all positive
- `code_negative_noise` unchanged
- `protected_harm_only` unchanged
- all `code_source_conflict*` unchanged

Evaluation：

- Tool quick `0.8800 / 0.8550 / 0.8125 / 0.6250`
- Memory eval_50 F1 `0.7564083485958486`
- LiveBenchCodeHurtRcrfVsTa16 BoN `0.1250 / 0.2421875`
- LiveCodeBenchCodeHurtRcrfVsTa16 BoN `0.3125 / 0.5294117647058824`

结论：

- Tool/Memory 稳，说明 positive-only 不破坏 protected behavior。
- Code 更弱，说明 v12 的问题不只是 negative suppression。
- 核心问题变成：role-routing 把所有 `code_source_conflict*` hold，导致丢掉 v9 中真正修 Code 的冲突 residual。

下一步方法：

- 不继续调步幅。
- 做 source/span-conditioned conflict routing。
- 归因单位从 `(param_name, expert, role)` 升级到 `(param_name, expert, role, source/span)`。
- 对 Code conflict rows，不能一律 hold；要看 LiveBench prompt/reasoning 或 LiveCodeBench code/prompt 哪个 source 是目标能力。

## 2026-05-22 Code expert scalar ablation v14/v15

用户怀疑：Code 系数整体太大，可能压制 Memory/Tool。实验只机械修改 v9：

- v14 `code_half`: 所有 `::code` 系数乘 `0.5`。
- v15 `code_zero`: 所有 `::code` 系数置 `0`。
- Tool / Memory 系数不动。

产物：

- script: `scripts/analysis/build_expert_scaled_gate_ablation.py`
- harness candidates: `v14_code_half`, `v15_code_zero`
- gate v14: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_v9_code_half_v14/gates.json`
- gate v15: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_v9_code_zero_v15/gates.json`
- checkpoint v14: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_half_v14`
- checkpoint v15: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_zero_v15`
- report: `docs/report/RCRF/20260522_code_expert_ablation.md`

结果：

| model | Tool quick | Memory eval_50 F1 | LiveBench hurt acc / BoN | LiveCodeBench hurt acc / BoN |
|---|---|---:|---:|---:|
| v9 | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v14 code_half | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7774 | `0.2031 / 0.2500` | `0.2500 / 0.1875` |
| v15 code_zero | `0.880 / 0.865 / 0.7500 / 0.625` | 0.7841 | `0.0781 / 0.1250` | `0.1719 / 0.1875` |

结论：

- Code expert 确实干扰 Memory：越降低 Code，Memory F1 越高。
- 但 Code expert 对 Code hurt recovery 是必要的：减半已经让 Code BoN 大幅塌，置零更严重。
- 因此不能用 task scalar 全局缩小 Code。下一步必须分离 Code-critical residual rows 和 Code-harmful rows，尤其是 `code_source_conflict*` 内部的 source/span 差异。

## 2026-05-22 Source-conflict routing v16/v17

动机：

- v13 positive-only 保 Tool/Memory，但 Code 弱。
- Atlas 中 `code_source_conflict*` 有 `279/588` 行，是最大未处理区域。
- v9 实际改变了 `83` 个 source-conflict 行，其中 `67` 个是负向 suppression。

代码变化：

- `scripts/analysis/build_rcrf_role_routed_gates.py`
  - 新增默认关闭参数：
    - `--source-conflict-action {hold,suppress-dominant,route-dominant}`
    - `--source-conflict-min-strength`
    - `--source-conflict-dominance-ratio`
    - `--source-conflict-protected-support-action {hold,allow}`
  - 默认 `hold`，旧 v12/v13 路径不变。
- `skill/command/run_20260522_rcrf_pareto_frontier.sh`
  - 新增 `v16_source_suppress`
  - 新增 `v17_source_route`

规则：

- v16: v13 + negative-dominant source-conflict suppression。
- v17: v16 + positive-dominant source-conflict raise。
- dominance: `dominant >= 1.25 * opposing` 且 `dominant >= 1.0`。
- 如果 residual 同时 support Tool/Memory，默认不做 negative suppression。

结果：

| model | Tool quick | Memory eval_50 F1 | LiveBench hurt acc / BoN | LiveCodeBench hurt acc / BoN |
|---|---|---:|---:|---:|
| v9 | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7575 | `0.2500 / 0.6250` | `0.6250 / 0.6555` |
| v13 | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7564 | `0.1250 / 0.2422` | `0.3125 / 0.5294` |
| v16 | `0.880 / 0.860 / 0.8125 / 0.625` | 0.7660 | `0.1094 / 0.2500` | `0.3281 / 0.4375` |
| v17 | `0.880 / 0.855 / 0.8125 / 0.625` | 0.7654 | `0.1250 / 0.3125` | `0.2188 / 0.4375` |

结论：

- Source-conflict routing 是 behavior-safe：Tool 稳，Memory 比 v13 高。
- 但离散 dominant-source rule 不能恢复 v9 的 Code，尤其 LiveBench hurt 仍低。
- v9 的强点来自连续 pass/fail overlay，而不是把 atlas 离散成少数 role。
- 下一版方法应把 v9-style continuous Code overlay 作为主干，只用 Tool hard / Memory soft behavior evidence 做约束，而不是用 role-routing 完全替代 overlay。

## 2026-05-22 Operating point delta diagnosis

新增脚本：

- `scripts/analysis/compare_rcrf_operating_points.py`

默认对比：

- reference: v9 continuous overlay
- candidates: v13 positive-only role, v16 source suppress, v17 source route

输出：

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/operating_point_comparison_summary.json`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/operating_point_rows.jsonl`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/delta_by_role.csv`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_operating_point_compare_20260522/reference_lost_by_source_pattern.csv`
- report: `docs/report/RCRF/20260522_operating_point_delta_diagnosis.md`

关键发现：

- v9: changed `205`, positive `106`, negative `99`, mean abs delta `0.002639`。
- v13: changed `73`, 全 positive，丢 v9 的 `145` 个 delta。
- v16: changed `112`，丢 v9 的 `124` 个 delta。
- v17: changed `146`，丢 v9 的 `110` 个 delta，且 `11` 个 sign mismatch。

v17 仍丢掉：

- `code_source_conflict`: 30
- `code_source_conflict_with_behavior`: 18
- `uninformative`: 22
- `code_negative_noise`: 16
- `code_repair_vs_protected_harm`: 10
- `code_repair_shared_and_harm`: 7

科研结论：

- v9 的有效性来自连续残差证据场，不是少数离散 role。
- “可解释”不应等于“hard rule”。更合理的论文方法是 continuous Code capability field + behavior constraints + atlas audit。
- 后续主方法命名建议：`Residual Capability Field with Behavior Constraints`。

## 2026-05-22 RCF-BC main method alias v18

将当前主方法从历史实验编号 `v9` 语义化为：

```text
RCF-BC = Residual Capability Field with Behavior Constraints
```

新增 harness candidate：

- `v18_rcf_bc`

复现：

```bash
PHASE=generate CANDIDATES=v18_rcf_bc \
  bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

输出：

- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/residual_capability_field_behavior_constraints_v18/gates.json`

数值等价：

- `v18_rcf_bc` 与 `v9` 的 588 个 gate 完全一致。
- `max_abs_diff = 0.0`
- `different_count = 0`

主方法步骤：

1. Code same-prompt pass/fail contrast 构建 continuous capability field。
2. Tool call-span 和 Memory full trajectory positive utility 做 preserve floor。
3. Tool/Memory harm evidence 做 positive delta soft veto。
4. expert-mean recenter 防止退化成全局 task scalar。
5. Atlas 作为 audit / ablation，而不是 hard decision engine。

报告：

- `docs/report/RCRF/20260522_rcf_bc_main_method.md`
