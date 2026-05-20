# 20260520 TRC 长期方法记忆

本文档用于下一轮 ExpertGym/TRC 决策，不是论文正文。结论来自 2026-05-19 到 2026-05-20 的 TRC hidden-state 实验、Round3-13 配置与评测记录。不要把 eval-leak 诊断结果混入 paper-main 非泄漏主结果。

## 0. 当前总判断

TRC 的有效部分不是“把某个 task coefficient 推大”，而是用 expert 成功轨迹提供 dense hidden-state target，再用 Tool/Memory 快评和 Code/CURE 闭环筛选。当前最稳主线：

```text
directional hidden residual
+ task/tool/code span
+ memory trajectory-turn loss
+ coefficient-level retention/floor
+ Tool/Memory quick gate before Code
```

核心经验：

1. MSE/relative-MSE 目标会把非目标 expert residual 当误差，容易压低 Tool/Memory；directional objective 是主线。
2. Memory final-answer-only span 是错的；必须对齐 MemAgent update turns + final answer，但 full trajectory 会 OOM，late3/uniform4 是可行预算。
3. Directional loss 不保证 expert residual 幅度；需要 coefficient floor，特别是 task/expert coefficient floor 或 global floor weight 约 50。
4. 直接放大 task loss multiplier 风险高。R4B code x1.4、R4F memory x2.0、R10E tool x1.5 都造成 Tool 或门槛问题。
5. Code gate/code residual/code projection 变强不等于 CURE 变强；Code 主要瓶颈是 calibration coverage、ability span 与 execution correctness 不匹配。
6. 当前非泄漏 best known 仍在 R3/R5/R8/R11 附近，没有被后续 loss push 明确超过；下一轮应优先改 Code 数据/ability span，而不是继续增大 gate。

关键实现/启动路径：

- Trainer: `scripts/trc/train_trc_layer_gates.py`
- 通用 runner: `skill/command/run_20260519_trc_round_train_one.sh`
- 选择/bake: `scripts/trc/select_trc_gate_checkpoint.py`, `scripts/eval/opvec_bake_checkpoint.py`
- 主 config: `configs/gated_grpo_layer28_wide.yaml`
- Stage-1 harness: `docs/config/20260519_trc_stage1_harness.md`
- Round ledger: `docs/harness/20260520_experiment_ledger.md`
- 评测 harness 文档: `docs/harness/20260519_stage1_training_control.md`, `docs/harness/20260520_overnight_35_attempt_plan.md`

## 1. 数据构造记忆

### 1.1 TRC96 clean baseline

路径：

- Data: `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl`
- Builder: `scripts/trc/build_trc_calibration_v1.py`
- Plan/audit: `docs/report/20260519_trc_method_plan.md`, `docs/report/trc_rounds/data_loss_audit_20260519.md`

构成：Tool 32 / Memory 32 / Code 32，均为 expert success rows；unique prompt 约 Tool 27、Memory 28、Code 27。Code 主要 ReasonFlux，少量 fallback。论文泄漏风险低，但 Code formal eval 覆盖不足。

结论：适合作 clean anchor 和方法验证，不足以解释 formal Code。

### 1.2 Memory trajectory-turn 数据

路径：

- Config: `docs/config/20260520_trc_round3_memorytraj.md`
- Memory: `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round3_memorytraj`
- 记忆: `docs/memory/expertgym_72h/20260520_trc_memory_trajectory_round3.md`

变体：

- `mtr_uniform4_toolaug_code_rf`: 4 memory_update + final，主力折中。
- `mtr_late3_toolaug_code_rf`: last 3 memory_update + final，更省且对 evidence integration 有效。
- `mtr_full_toolaug_code_rf`: all turns，诊断上合理，但 7B merged forward 下 OOM，非今晚主线。

结论：trajectory-turn 解决“Memory span 覆盖”问题，但不解决“memory coefficient 幅度被 directional loss 放掉”问题；必须配 coefficient floor。

### 1.3 CodeP0 / taxonomy / hybrid 数据

路径：

- Round8 CodeP0: `docs/config/20260520_trc_round8_codep0_v3.md`
- Round10 tag quota: `docs/config/20260520_trc_round10_code_taxonomy.md`, builder `scripts/trc/build_trc_codep0_tag_calibration.py`
- Round11 hybrid: `docs/config/20260520_trc_round11_codepush.md`, builder `scripts/trc/build_trc_round11_hybrid_calibration.py`
- Round12 RF-only: `docs/config/20260520_trc_round12_codepush.md`, builder `scripts/trc/build_trc_round12_rfonly_tagquota_calibration.py`

重要数据结论：

- R8 RF-only CodeP0: ReasonFlux 29 unique prompt / 161 success samples；expert-vector 一致性好，但覆盖不足。
- R8 RF+DeepSeek: 32 unique prompt，覆盖好但异质 teacher 未改善 Code。
- R10 tag quota: 32 unique Code prompts，RF 28 + DS 4，按 string/math/graph/DP/greedy/format 标签配额；改善覆盖但 Code 仍弱。
- R11 hybrid: stable Tool/Memory + 24 R10 + 8 RF-only supplement，0 duplicate；是数据工程上最干净的折中之一。
- R12 RF-only tag-quota: 32 RF-only Code rows 但只有 29 unique prompts，3 duplicate 用于补 graph/DP/greedy；R12D quick gate 失败，说明 RF purity 不足以抵消重复/Tool 风险。

下一轮 Code 数据优先级：非泄漏、当前 merged fail + teacher success、显式覆盖 IO/edge-case/算法标签/repair rationale，并记录 ability span 是否实际进入 `max_response_tokens`。

### 1.4 Eval-leak 诊断数据

路径：

- Round6 config/eval: `docs/config/20260520_trc_round6_cure_diagnostic.md`, `docs/evaluation/20260520_trc_round6_diagnostic_eval.md`
- Round13 config: `docs/config/hiddenstate/20260520_round13_evalleak_code16.md`
- Builder: `scripts/trc/build_trc_round13_evalleak_code16_calibration.py`

结论：

- R6B 用 16 LiveBench + 16 LiveCodeBench hidden-test-passing response trajectories，Tool 0.7944、Memory 0.7604、Code mean 0.3211；没有明显超过非泄漏 best。
- R6A 只拟合 LiveBench code-block，Tool 0.7800，被拒。
- Round13 构造 `rfmem_only` 和 `all_with_r1`，用于判断 formal Code ability-span + scaled R1 是否能被 hidden loss 学到；这是诊断，不进主结果。

## 2. Loss / span / coefficient 变量结论

### 2.1 Hidden-state loss

实现：`scripts/trc/train_trc_layer_gates.py::compute_trc_row_loss()`, `hidden_residual_loss()`。

v1 MSE:

- `r_merge ≈ r_expert`，loss 可降，gate 分化明显。
- 初始结果：epoch3 residual 0.3300，Tool 1.1426 / Memory 0.7116 / Code 1.2871。
- 失败原因：把其他 expert residual 当误差，压 Memory。

v2 normalized/span-aware MSE:

- Config: `docs/config/20260519_trc_stage1_harness.md`
- epoch2 仍为 Code 1.2249、Memory 0.7671、Tool 0.8378。
- 失败原因不是尺度，而是目标语义仍错。

v3 directional:

```text
L_dir = 1 - cos(r_merge, r_expert)
projection_ratio = dot(r_merge, r_target) / ||r_target||^2
```

- 允许 merged residual 包含其他 expert 正交能力方向。
- Stage-1 `dir_i8` Tool/Memory 最强，Code 未跟随 residual loss。
- 主线保留 directional + projection，但不要只按 residual loss 选点。

relative-MSE:

- R2C/R8C 均不稳，loss 尺度大，早期压 Tool/Memory；不作为主线。

### 2.2 Ability span / response span

当前推荐：

```text
Tool: tool-call span
Memory: response with trajectory-turn loss
Code: code-block 或经过压缩的 ability response span
```

结论：

- Tool 必须优先 `<tool_call>...</tool_call>`；fallback 到 full response 会制造噪声。
- Memory final answer span 过短，必须使用 memory_update turns。
- Code `code-block` 安全，但只学最终代码形态，LiveCodeBench/edge-case 转化弱。
- Code full `response` 有时可提升 Memory/primary Code 一点，但也可能造成 Tool collapse；必须配 global/task floor 和快评门控。
- Response topK 不是越大越好：R5D response384 weak，R9 topK128 Tool fail；R5C response256 是相对稳点。
- Round13 的 `critical_reasoning_span + final_code_span` 是正确方向：不是 full response，而是压缩 ability span。

### 2.3 Memory trajectory-turn

实现改动记录在 `docs/memory/expertgym_72h/20260520_trc_memory_trajectory_round3.md`：

- `scripts/trc/build_trc_calibration_v1.py`: `--memory-response-source trajectory-turns`
- `scripts/trc/train_trc_layer_gates.py`: `--trajectory-turn-loss-task memory`
- runner env: `TRAJECTORY_TURN_LOSS_TASKS=memory`

结论：

- uniform4/late3 + coefficient floor 能让 Memory F1 回到 0.76+。
- full trajectory OOM，当前不值得作为主线。
- late3 通常比 uniform4 更省，Memory 有时更好；但 Code/Tool 要看配套 span/floor。

### 2.4 Task/expert coefficient floor

关键原因：directional loss 在 init=1 附近可能认为 memory 方向已经对齐，允许降低 memory 幅度服务 Tool/Code。

有效设置：

```text
TASK_EXPERT_COEFFICIENT_FLOOR=1.0
TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0
```

或 global coefficient floor weight 50。R3D/R3F 证明 floor=20 只减缓下降，floor=50 才能把 Memory 稳在约 1.0。

注意：

- task-aware floor 语义更干净，但部分 run Tool 略弱。
- global floor 有时更稳 Tool/Memory，但可能统一拉 unrelated expert。
- 不要用大 task loss multiplier 代替 coefficient floor。

## 3. R1 scaled delta 记忆

路径：

- Old scaled plan: `docs/report/20260518_deepseek_r1_scaled_task_vector_plan.md`
- Correct-R1 config: `docs/config/20260519_r1math_L_experiments.md`
- Config: `configs/gated_grpo_4expert_r1math_layer28.yaml`
- Raw/Scaled modes: `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519`, `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519`

结论：

- 旧 R1 delta 语义错：`DeepSeek-R1-Distill-Qwen-7B - Qwen2.5-7B-Instruct`；正确应为 `DeepSeek-R1-Distill-Qwen-7B - Qwen2.5-Math-7B`。
- correct-R1 scaled factor 为 `0.006910525387901668`。
- R1 原始 delta 范数远大于 agent experts；即使 scaled 后也必须用 micro coefficient / bounds。
- L1/L2/L3 OPD+NLL retention 最终都进入 tool near-zero over-push：iter20/19/17 tool reward 约 0.03-0.04，memory/code proxy 上升但 Tool 崩。
- 冻结 R1 的 L2 也 Tool 崩，说明问题不仅是 R1，而是 OPD/retention/gate over-push 与 Tool protection 不足。
- Round13 才是 R1 与 TRC 结合的合理诊断：formal Code ability span + scaled correct-R1 reasoning expert，对比 no-R1 `rfmem_only`。

下一轮 R1 不应全局大系数混入。只做：

- scaled correct-R1；
- `reasoning` bounded micro coefficient；
- formal/non-leak Code ability span 证明有效后再联合；
- Tool quick gate 必须先过。

## 4. Round3-13 ledger

### Stage-1 / Round1-2

路径：

- Stage config/eval: `docs/config/20260519_trc_stage1_harness.md`, `docs/evaluation/20260519_stage1_candidates_eval.md`
- Supervisor: `docs/report/trc_rounds/experiment_supervisor_20260519.md`

结果：

- `dir_i8`: Tool 0.7981、Memory F1 0.7663、Code mean 0.3223；Tool/Memory 强，Code 不随 residual loss。
- `anchor_i8`: Code mean 0.3313，但 Tool 0.7800 未达 0.79；保守 Code fallback。
- R2 A/B/D loss 继续下降但 gate 过推；e8 较安全，e18 Memory collapse。

失败原因：Code calibration 和 hidden proxy 不等价于 CURE correctness；不能按 residual min 选点。

### Round3 memory trajectory

Config/eval: `docs/config/20260520_trc_round3_memorytraj.md`, `docs/evaluation/20260520_trc_round3_eval.md`

代表结果：

- R3D globalfloor50 uniform4: Tool 0.7944、Memory 0.7636、Code mean 0.3289、BoN 0.4154。
- R3J late3 taskfloor50: Tool 0.7944、Memory 0.7673、Code mean 0.3137。
- R3K code full-response + global floor: Tool 0.7944、Memory 0.7715、Code mean 0.3223。
- R3M/R3N Tool collapse；R3O Memory 0.7585 被拒。

结论：memory trajectory + floor 是有效修复；R3D 是早期非泄漏 Code strong anchor。

### Round4 code push

Config/eval: `docs/config/20260520_trc_round4_codepush.md`, `docs/evaluation/20260520_trc_round4_eval.md`

结果：

- R4A: Tool 0.7944、Memory 0.7638、Code 0.3243。
- R4D code projection: Tool 0.8048、Memory 0.7669、Code 0.3076。
- R4B code multiplier 1.4、R4F memory multiplier 2.0: Tool collapse。

结论：projection 比 loss scale 安全，但 Code 仍不涨。

### Round5 SOTA/recovery calibration

Config/eval: `docs/config/20260520_trc_round5_sota_calib.md`, `docs/evaluation/20260520_trc_round5_eval.md`

结果：

- R5A code-block384: Tool 0.8035、Memory 0.7638、Code 0.3194、BoN 0.4310。
- R5C response256: Tool 0.7944、Memory 0.7690、Code 0.3257。
- R5D response384: Code 0.3042。

结论：v2 calibration 保 Tool/Memory，并提高 BoN/部分 response primary，但没超过 R3D primary。

### Round6 eval-leak diagnostic

Config/eval: `docs/config/20260520_trc_round6_cure_diagnostic.md`, `docs/evaluation/20260520_trc_round6_diagnostic_eval.md`

结果：

- R6A LiveBench code-block leak: Tool 0.7800 reject。
- R6B balanced leak response: Tool 0.7944、Memory 0.7604、Code 0.3211。

结论：即使用 formal hidden-test-passing trajectories，TRC hidden alignment 也未显著突破非泄漏 best；说明 Code objective/ability-span 仍不足。

### Round7 response follow-up

Config/eval: `docs/config/20260520_trc_round7_response_followup.md`, `docs/evaluation/20260520_trc_round7_eval.md`

结果：

- R7B projection 1.05: Tool 0.8048、Memory 0.7821、Code 0.3010。
- R7A 16 epochs: Memory 0.7553 reject。

结论：Tool/Memory 越强不代表 Code 转化；response-span 继续训练会过推 Memory。

### Round8 CodeP0-v3

Config/eval: `docs/config/20260520_trc_round8_codep0_v3.md`, `docs/evaluation/20260520_trc_round8_eval.md`

结果：

- R8B RF+DS response: Tool 0.7944、Memory 0.7687、Code 0.3059。
- R8D RF-only code-block384: Tool 0.7944、Memory 0.7668、Code 0.3203；LiveBench 0.3730 强，LiveCodeBench 弱。
- R8A-e08 RF-only response early: Tool 0.7931、Memory 0.7716、Code 0.3218。
- R8C relative-MSE reject。

结论：早停有意义；RF-only/CodeP0 对 LiveBench 有信号，但整体 Code 仍未过 R3D/R5C。

### Round9 focused topK128

Config: `docs/config/20260520_trc_round9_focus_span.md`

结果：

- R9A/R9B response topK128 Tool mean 0.7788，Memory stopped，Code skipped。

结论：topK128 过窄会伤 Tool/live_parallel，不是默认方向。

### Round10 taxonomy

Config/eval: `docs/config/20260520_trc_round10_code_taxonomy.md`, `docs/evaluation/20260520_trc_round10_eval.md`

结果：

- R10A tag response128: Tool 0.7944、Memory 0.7625、Code 0.3096。
- R10D tag response256 + Memory x1.8: Tool 0.7944、Memory 0.7679、Code 0.3162。
- R10B Memory 0.7572 reject；R10C Tool 0.7788 reject；R10E Tool 0.7788 reject。

结论：taxonomy coverage 帮助稳定但不解决 Code；Memory multiplier 1.8 是可用小修，Tool multiplier 1.5 不安全。

### Round11 hybrid / early-stop

Config/eval: `docs/config/20260520_trc_round11_codepush.md`, `docs/evaluation/20260520_trc_round11_eval.md`

已知到 2026-05-20 11:44：

- R11B R8D-e08 alias: Tool 0.7944、Memory 0.7715，LiveBench 0.3750/BoN 0.5000，LiveCodeBench still pending in doc。
- R11F response320: Tool 0.8048、Memory 0.7726，LiveBench 0.3555。
- R11G hybrid response256: Tool 0.7944、Memory 0.7600，LiveBench 0.3516。
- R11H hybrid code-block384: Tool 0.8048、Memory 0.7619，LiveBench 0.3477。

结论：R11B 是最值得等完整 Code 的候选；但截至文档快照，LiveCodeBench 未完成，不能下最终结论。

### Round12 RF-only tag-quota

Config/eval: `docs/config/20260520_trc_round12_codepush.md`, `docs/evaluation/20260520_trc_round12_eval.md`

结果：

- R12D RF-only code-block384 mem16: Tool 0.7788、Memory 0.7590，reject before Code。
- R12E/F/G 是后续建议/部分运行状态，需以后续 eval 更新。

结论：RF-only + code-block384 + duplicates + mem16 不稳；若继续，应优先 R12E response256 mem18 或 R12F e08/e12 quick gate，不直接跑 Code。

### Round13 eval-leak hidden-state diagnostic

Config: `docs/config/hiddenstate/20260520_round13_evalleak_code16.md`

设计：

- R13A no-R1: `rfmem_only`，Tool32/Memory32/Code10，no DeepSeek/R1。
- R13B all+R1: Tool32/Memory32/Code23，DeepSeek/R1 rows mapped to `reasoning`，mode `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json`。
- ability span 压缩为 600-char reasoning context + final code block，适配 seq1536/resp512。

判定：如果 R13B 明显优于 R13A，说明 scaled correct-R1 + formal ability span 可提供新能力；若都不涨，当前 TRC hidden objective 需要 contrastive / execution-aware loss。

## 5. 下一轮决策规则

### 5.1 评测门控

保持快速门槛：

```text
Tool BFCL mean >= 0.79
Memory HotpotQA mean F1 >= 0.76
```

不过门槛不跑 Code/CURE。边界候选只作为 ablation，不占主评测资源。

### 5.2 不要重复的方向

- 不复活 v2/MSE 或 relative-MSE 主线。
- 不用直接 task loss multiplier 做 Code/Tool/Memory 保护。
- 不把 Code gate/code residual 最低作为选点标准。
- 不把 full memory trajectory 作为主线，除非工程上先解决 OOM/cache。
- 不把 eval-leak R6/R13 报为主结果。
- 不把 DeepSeek/R1 当普通第四 expert 大系数混入。

### 5.3 优先实验

1. 等 R11B/R11F/R11G/R11H 完整 LiveCodeBench 完成，先更新 Code mean 再开新训练。
2. 若 R11B 完整 Code 最好，围绕 R8D-e08/RF-only/code-block early-stop 做数据扩展，而不是继续 e12 over-push。
3. 若 R11F/R11G/H 更好，围绕 response256/320 + Memory x1.8 + hybrid bank 做能力 span 精修。
4. 构造非泄漏 Code ability-span bank：当前 merged fail、teacher success、覆盖 IO/edge-case/algorithm/repair；对每条记录 code-block、reasoning span、tests/format tags。
5. 做小型 execution-aware diagnostic：hidden directional + contrastive pass/fail 或 test-driven repair target，避免只拟合成功文本 hidden state。

### 5.4 当前参考 anchors

- 非泄漏 Code primary anchor: R3D Code mean 0.3289，Tool 0.7944，Memory 0.7636。
- 非泄漏 Memory anchor: R3K Memory 0.7715，Code 0.3223。
- Strong Tool/Memory but weak Code negative: R7B Tool 0.8048，Memory 0.7821，Code 0.3010。
- CodeP0 early-stop anchor: R8A-e08 Code 0.3218，Memory 0.7716。
- Pending high-signal: R11B LiveBench 0.3750 / BoN 0.5000，需完整 LiveCodeBench。

