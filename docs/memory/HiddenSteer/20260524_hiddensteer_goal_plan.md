# 2026-05-24 HiddenSteer Goal and Plan

本文档记录当前 HiddenSteer 目标、合理性判断和最小执行计划。HiddenSteer 暂定为一个推理时 activation projection 框架：离线用成功轨迹建立能力局部几何 atlas，在线只在少数层做低秩投影，解耦 merged agent 中的能力干扰。

## 1. 当前目标

目标不是再训练一个新的 merge，也不是增加 router。当前合理目标应收束为：

> 在不改权重、不运行专家模型、不使用 benchmark label 的前提下，利用离线预计算的成功轨迹几何方向，在推理时通过低秩 activation 投影减少 Tool 子能力被 Memory/Code residual 侵入的问题。

第一阶段只做 Tool，不同时修 Code/Memory。原因是 Tool 干扰的诊断最清楚，且在线触发条件最自然：输入中出现 tool schema 或生成进入 function-call span 时才启用保护。

## 1.1 Active Execution Goal

当前执行目标：

> 先验证 HiddenSteer 在 Tool 上是否同时满足有效性和推理计算可行性。用 HF generation hook 做最小原型，测 BFCL Tool 子项和端到端耗时；如果 Tool 的关键子能力确实被修复且开销可接受，再扩展到 Memory 和 Code。暂不改 vLLM，除非 HF 原型证明该路线值得工程化。

这个目标分三阶段：

1. Tool-first validation：
   - 只实现 Tool projection hook。
   - 只使用离线预计算 basis。
   - 只在 tool affordance 出现时启用。
   - 同时记录 accuracy、error type 和 wall-clock/token throughput。

2. Engineering feasibility gate：
   - 若 HF hook 已经开销过大，先不进入 Memory/Code。
   - 若 hook 开销主要来自 Python 调度，但矩阵计算很轻，再评估是否值得做 vLLM forward patch。
   - 若只有全程开启 hook 才有效，而 affordance trigger 无效，则停止方法化。

3. Conditional expansion：
   - Tool 过线后，再处理 Memory。
   - Memory 过线后，再处理 Code。
   - Memory/Code 仍遵守同一框架：离线 basis，在线低秩投影，不引入新 router，不在线跑专家。

## 2. 为什么这个目标合理

当前现象说明 Tool 不是整体能力不足，而是子能力没有对齐：

| model | parallel | parallel_multiple | live_parallel | live_parallel_multiple |
| --- | ---: | ---: | ---: | ---: |
| RAM / `arm-r-v2` | 0.905 | 0.855 | 0.750 | 0.667 |
| init1 | 0.880 | 0.860 | 0.750 | 0.625 |
| `rcrf_code_spanaware_conservative_v2` | 0.880 | 0.855 | 0.8125 | 0.625 |
| `cg-tool-extra020` rerun | 0.900 | 0.855 | 0.750 | 0.625 |

结论：

- 平均分可以接近 RAM，但不同方案修的是不同 Tool 子能力。
- `parallel` non-live 有 200 题，RAM 比部分 ours 高，说明函数选择和参数匹配仍有真实干扰。
- `live_parallel_multiple` 只有 24 题，但持续低于 RAM，说明真实 schema 下的多调用 count / matching 还没对齐。
- 主要错误是 `cannot_find_match` 和 `wrong_count`，不是大面积 AST 格式崩坏。因此不应该做全局 tool boost，而应保护 function selection、argument matching、multi-call count 这些局部 activation anchor。

已有结构诊断也支持这个目标：

- Tool 在 `tool_signature_s32` 上是强正向锚点，utility 约 `3.288`，positive fraction 约 `0.954`。
- Memory 在 `tool_signature_s32` 上有负向 spillover，utility 约 `-0.360`，negative fraction 约 `0.587`。
- Pairwise-zero 诊断表明 Tool 更像格式行为保护，而不是所有任务的能力主干。

因此，HiddenSteer 第一阶段 claim 应该是：

> Merged agent 的 Tool 退化来自 Tool 子能力 anchor 被其他 capability residual 局部侵入；离线成功轨迹几何可以定义干扰子空间，推理时低秩投影可以降低这种侵入。

## 3. 非目标

当前阶段不要做以下事情：

- 不做 task-label router。
- 不按 benchmark category 直接切换策略。
- 不在线运行 Tool/Memory/Code expert。
- 不用 failure 轨迹作为必要输入。
- 不把 Code/Memory 同时作为第一版优化目标。
- 不用全局 task vector shrink / boost 解释结果。

这些都会让方法实体变多，claim 变混，且容易过拟合小评测。

## 4. 离线预计算产物

第一阶段只构建 Tool atlas。

输入：

- BFCL 成功输出。
- merged model 与 expert/task-vector 的 activation/update 诊断。
- 已有 signed-effect / structure utility 产物。

建议复用或对齐的已有产物：

- `/tmp/shared-storage/ExpertGym/structure_utility_maps/opvec4_rcrf_calibration_20260522/README.md`
- `/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/tool_memory_signature_s2_20260523`
- `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523/pairwise_zero_diagnostic_report.md`
- `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/arm-r-v2/tool/summary.json`

需要新生成的 atlas：

```text
tool_parallel_anchor_basis
tool_parallel_multiple_anchor_basis
tool_live_parallel_anchor_basis
tool_live_parallel_multiple_anchor_basis
memory_code_to_tool_interference_basis
selected_layers_and_modules
projection_strength_defaults
```

每个 basis 应至少包含：

```json
{
  "capability": "tool_parallel",
  "span": "function_name|argument|multi_call_count|delimiter",
  "layer": 0,
  "module": "residual_stream",
  "rank": 4,
  "basis_dtype": "bf16_or_fp16",
  "source_samples": 0,
  "selection_score": 0.0,
  "orthogonalized_against_anchor": true
}
```

## 5. 在线推理框架

在线推理只做低秩投影 hook。对选中的层，取 residual activation `h_l`：

```text
h_l' = h_l - alpha * V_l (V_l^T h_l)
```

其中 `V_l` 是离线得到的干扰 basis。第一版不做 anchor boost，只做 project-out interference。这样最容易判断效果是否来自解耦，而不是手动增强 Tool。

触发条件使用 affordance，不使用 benchmark label：

- prompt 中包含 tool schema / function definitions：启用 Tool protection。
- 解码进入 function-call span：启用 function-name / argument protection。
- 输出进入多调用结构：启用 parallel-multiple protection。
- 普通自然语言、代码块或 Memory answer span：不启用 Tool hook。

工程实现顺序：

1. HF generation hook 原型，验证投影逻辑和离线 atlas。
2. 只 hook selected layers，不全层 hook。
3. 验证有效后再考虑 vLLM forward patch，把 basis 注册为 buffer。

计算预算预期：

```text
extra FLOPs per token ~= selected_layers * hidden_dim * rank * 2
```

若选 12-20 层、rank 4-8，矩阵计算开销应远小于模型 forward。实际工程瓶颈更可能是 Python hook 调度，而不是投影矩阵乘法。

## 6. 最小实验设计

第一轮只比较以下模型/设置：

1. baseline: init1 或当前主要 merged checkpoint。
2. HiddenSteer-tool-proj: 同一 checkpoint + Tool projection hook。
3. optional negative control: projection strength `alpha=0`。
4. optional over-control: 对非 tool prompt 也启用 hook，验证不应提升。

第一轮评测：

```text
BFCL: parallel, parallel_multiple, live_parallel, live_parallel_multiple
Memory: HotpotQA eval_50
Code: CURE small / code-hurt subset
```

判定标准：

- 必须提升或至少追回 `parallel` 与 `live_parallel_multiple`，不能只提升 `live_parallel`。
- `parallel_multiple` 不低于 baseline。
- Memory eval_50 F1 不明显下降。
- Code small BoN 不明显下降。
- 错误类型中 `cannot_find_match` 和 `wrong_count` 应减少，不能只是 AST parse 误差波动。

如果只提高平均 Tool 分，但 `parallel` 或 `live_parallel_multiple` 没改善，则不能声称 Tool 子能力对齐。

运行可行性标准：

- 第一版 HF hook 允许慢，但必须记录：
  - tokens/s 或 samples/min。
  - hook enabled vs disabled 的 wall-clock ratio。
  - selected layers、rank、alpha、触发 token span 占比。
- 若端到端耗时小于 baseline 的 `1.3x`，可以认为原型层面可接受。
- 若端到端耗时在 `1.3x-2.0x`，需要判断开销是否主要来自 Python hook；只有矩阵计算占比低时才考虑 vLLM patch。
- 若端到端耗时超过 `2.0x`，且不能通过减少 layers/rank/span trigger 降到 `2.0x` 内，则停止扩展到 Memory/Code。
- 若 Tool accuracy 有效但开销不可接受，结果只作为 feasibility analysis，不作为主方法。

进入 Memory/Code 的条件：

- Tool 至少满足以下之一：
  - `parallel` 追回到接近 RAM：目标 `>= 0.900`。
  - `live_parallel_multiple` 追回一题：目标 `>= 0.667`。
  - `cannot_find_match` 或 `wrong_count` 相对 baseline 明确减少。
- 同时不能出现：
  - `parallel_multiple` 明显下降。
  - Memory eval_50 F1 明显下降。
  - Code small BoN 明显下降。
  - 推理耗时超过可接受门槛。

只有 Tool 同时满足有效性和耗时条件，才进入 Memory。只有 Memory 不破坏 Tool 且自身有效，才进入 Code。

## 7. 论文 claim 边界

合理 claim：

- Agent merging 的 Tool 退化包含子能力拓扑错位，而不是单一 task vector 权重不足。
- 成功轨迹 activation geometry 可以定位 Tool anchor 与跨能力干扰子空间。
- 推理时低秩投影是一种不改权重的 local decoupling 方法，可减少 Tool 子能力干扰。

暂时不 claim：

- 完整解决 Memory-Code 冲突。
- 超过所有 RAM/ReasonFlux 综合性能。
- 不需要任何触发条件即可通用提升。
- 该方法本质上优于训练时 merge，只能说提供 inference-time correction。

## 8. 风险和停止条件

主要风险：

- Tool 子能力 span 过细，basis 估计不稳。
- Python hook 导致吞吐过低，原型结果难以扩展到正式 vLLM。
- 投影删掉了 Tool anchor 本身，而不是删干扰。
- `live_parallel_multiple` 样本太少，单次提升可能是噪声。

停止条件：

- 两轮不同 alpha / rank 设置都不能同时改善 `parallel` 和 `live_parallel_multiple`。
- BFCL 提升伴随 Memory 或 Code 明显下降。
- 只有 category-label 触发才有效，affordance 触发无效。

若触发停止条件，应把 HiddenSteer 降级为分析/诊断贡献，不作为主方法。

## 9. 下一步行动

1. 写 atlas builder：从 BFCL 成功样本和已有 activation-update 诊断中抽取 Tool 子能力 basis。
2. 写 HF hook 原型：支持 selected layers、rank、alpha、affordance trigger。
3. 跑小规模 BFCL 四项，先确认 `parallel` 与 `live_parallel_multiple` 是否朝 RAM 对齐。
4. 补 Memory eval_50 和 Code small，检查副作用。
5. 如果有效，再整理成 `HiddenSteer: offline geometry, online projection` 方法图和主表。

## 10. 2026-05-24 Tool HF Smoke Update

第一轮 Tool-only HF hook 已完成，记录见：

- `docs/memory/HiddenSteer/20260524_tool_hf_projection_smoke.md`

结论：

- HF hook 和 BFCL partial scorer 已跑通。
- rank8 / 8-module 低秩 basis 的 full `live_parallel_multiple` overhead 约 `1.28x`，工程上接近可接受。
- 但输出完全不变，Tool 分数不变；即使用 `projection_strength=100`，correction 也只占 selected module output norm 约 `0.22%`。
- 当前 module-output low-rank delta projection 不足以作为方法，不应扩展到 Memory/Code，也不值得改 vLLM。

下一步应转向 residual-stream activation basis，而不是继续放大 task-vector module-output correction。

## 11. 2026-05-25 Tool Residual Smoke Update

第二轮 residual-stream projection 已完成，记录见：

- `docs/memory/HiddenSteer/20260525_tool_residual_projection_smoke.md`

结论：

- residual-stream HF hook、success/failure basis builder、BFCL partial scoring 都已跑通。
- rank-4、4-layer basis 很小：contrast basis `456K`。
- 满集 `live_parallel_multiple` 的保守 `remove_failure_orthogonal` 设置开销约 `1.13x`，工程上可接受。
- 但分数没有提升：baseline `16/24`，`remove_failure_orthogonal` 仍是 `16/24`，没有 gain/loss。
- full `remove_failure` 会伤能力：`16/24 -> 13/24`，没有修复原失败，只损伤 3 个原正确样本。

当前 gate 判定：

- 不进入 Memory/Code。
- 不改 vLLM。
- 不继续做单纯 alpha/rank sweep。

原因是 Tool 的当前错误更像 span/decision-level 问题：argument canonicalization、optional parameter preservation、multi-call count separation、schema-value matching。它不是一个稳定的 late-layer low-rank failure residual，可以靠推理时全局 residual projection 后处理掉。

## 12. 2026-05-25 Memory Immunization Smoke Update

第一轮 Memory-Conditioned Residual Immunization 小试已完成，记录见：

- `docs/memory/HiddenSteer/20260525_memory_residual_immunization_smoke.md`

结论：

- 新脚本已跑通：从 Memory expert rollout 抽 teacher trajectory，顺序加载 merged / memory expert / code expert，构建 `code_bad` basis，并训练零输出低秩 residual corrector。
- `final_answer` turn 太容易，merged teacher NLL 已接近 0，没有诊断价值。
- `memory_update` turn 有训练信号，但 20/8 prompt-disjoint ablation 不泛化：
  - `code_bad_weight=0.00`: heldout NLL `0.6210 -> 0.7158`
  - `code_bad_weight=0.05`: heldout NLL `0.6210 -> 0.6763`
  - `code_bad_weight=0.10`: heldout NLL `0.6210 -> 0.7112`
- Tool/Code retention NLL 也小幅上升；`w=0.05` 损伤最小，但仍不是正收益。
- 因此目前正信号主要来自 Memory teacher fitting，不是 code residual immunization 本身。

当前 gate 判定：

- 停止当前 trained residual-corrector 变体。
- 暂不 claim 解决 Memory-Code conflict。
- 暂不进入 HotpotQA eval_50 / BFCL / CURE retention。
- 暂不改 vLLM；单层 rank-2 hook 的推理开销可接受，但方法收益没有过 gate。

后续如果继续做推理时 Memory 修复，不应再做同一 corrector sweep。候选方向必须满足：

- Memory span 条件触发，而不是全局启用。
- 以预计算几何投影或闭式小修正为主，不用 prompt teacher fitting 作为主信号。
- 先过 heldout Memory + Tool/Code retention NLL，再考虑 answer-level benchmark。

## 13. 2026-05-25 Direct Memory Projection Update

直接闭式投影的小试已完成，记录见：

- `docs/memory/HiddenSteer/20260525_memory_direct_projection_smoke.md`

方法：

```text
B_bad = orthogonalize(PCA(h_code_expert - h_merged), PCA(h_memory_expert - h_merged))
h' = h - alpha * B_bad B_bad^T (h - merged_center)
```

关键区别：

- 不训练 adapter / corrector。
- geometry 只用 train Memory prompts，heldout 不进 basis。
- 只投影 train Memory 上异常高的 code-bad energy token。

20/8 prompt-disjoint `memory_update` 初步结果：

| setting | heldout NLL | Tool delta | Code delta |
|---|---:|---:|---:|
| baseline | 0.6210 | 0.0000 | 0.0000 |
| layers 16-27, rank4, p90, alpha 0.35 | 0.6090 | -0.0028 | +0.0010 |
| layers 16-27, rank4, p90, alpha 0.50 | 0.6096 | -0.0026 | +0.0005 |
| layers 16-27, rank4, p90, alpha 0.65 | 0.6171 | -0.0028 | -0.0015 |

当前判断：

- 这是第一个符合方法直觉的正信号：heldout Memory NLL 下降，Tool 不受损，Code 近似不变。
- 过强投影会明显伤 Memory；`p75` 或 `alpha=1.0` 不应作为默认。
- 下一步可以固定 `p90`、`alpha=0.35/0.5`，扩大 Memory heldout 与 Tool/Code retention。

2026-05-25 answer-level update:

- init1 local recurrent HotpotQA first16: F1 `0.7560 -> 0.7143`, no wins, one loss.
- init1 official-wrong16: F1 `0.2130 -> 0.2963`, two wins, one loss.
- init1 full eval_50 paired HF: F1 `0.7754 -> 0.7628`, EM `0.6094 -> 0.5938`, wins/losses/ties `3 / 7 / 118`.
- 结论：projection 有 answer-level 恢复能力，但不是单调安全；full paired F1 不过 gate，不能作为默认 Memory correction。后续若继续，必须做 recoverable-state trigger 或更保守的投影判据。

2026-05-25 recoverability-gated update:

- 对 full128 paired 输出做离线 accept/reject：只接受 answer changed 且 projected/token `< 0.09` 的 corrected answer。
- 该 policy 使用预测文本和 projection runtime stats，不用 gold 做决策。
- 结果从 baseline F1 `0.7754` 到 gated F1 `0.7833`，wins/losses/ties `1 / 0 / 127`。
- 但阈值来自当前 run 的事后分析，必须冻结后在 disjoint split 上验证，不能直接作为最终方法结果。

2026-05-25 eval_100 validation update:

- 冻结的 low-rate gate 在 disjoint eval_100 上没有泛化：F1 `0.7201 -> 0.7196`，wins/losses/ties `1 / 1 / 126`。
- 但无条件 projection 在 eval_100 上是正的：F1 `0.7201 -> 0.7520`，EM `0.5703 -> 0.6016`，wins/losses/ties `8 / 4 / 116`。
- 新增 answer-shape guard：拒绝 baseline answer 扩写、纯数字/年份翻转、短答案变成长且无关短语。
- eval_100 上 projection + guard 达到 F1 `0.7606`，EM `0.6094`，wins/losses/ties `7 / 0 / 121`。
- eval_50 上 projection + guard 从 raw projection F1 `0.7628` 提升到 `0.7717`，但仍低于 baseline `0.7754`。
- 当前判断：projection 有真实 recoverability 信号，side effect 可以被降低，但 answer string guard 还不是完整 selector；下一步需要 no-gold candidate verification 或 residual-consistency selector，而不是继续调 projection_rate 阈值。
