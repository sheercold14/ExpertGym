# TRC 数据与 Loss 审计报告 2026-05-19

## 审计结论

当前 TRC 数据链路可以分成两类：`20260519_trc_v1` 是论文风险最低的三任务成功轨迹基线；`L5/L6` 直接引入 formal eval distribution，诊断价值最高但不能直接作为严格 heldout 主结果；`eval_targeted96` 没拷贝正式评测 prompt/tests，是更适合论文主线的 eval-style calibration，但 Code 正样本覆盖率只有约 47%-53%，说明 Code teacher 轨迹仍稀缺。

当前 TRC loss 实现是 hidden residual alignment，不是 reward optimization。`directional` objective 比 MSE 更适合多 expert 合并，因为它允许 merged residual 同时包含多个 expert 的正交能力方向；但 Code 的 formal correctness 是离散 unit-test 结果，hidden loss 只能提供“像 expert 解题轨迹”的 proxy，不能单独作为收敛标准。下一轮实验必须用 Tool/Memory 快评门槛和 Code mini-CURE/formal CURE 来闭环。

## 审计路径

| 数据目录 | 角色 | 关键文件 |
|---|---|---|
| `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1` | TRC96 成功轨迹基线 | `trc96_expert_trajectories.jsonl`, `trc96_summary.json` |
| `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16` | formal CURE Code16 诊断锚点 | `cure_eval_code16_expert_success_rollouts_seed20260519.jsonl` |
| `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16` | formal BFCL Tool16 + CURE Code16 诊断锚点 | `bfcl_tool16_cure_code16_extra_expert_rollouts_seed20260519.jsonl` |
| `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517` | 无正式题拷贝的 eval-style 训练候选 | `eval_targeted96.prompts.jsonl`, `expert_rollouts/*.jsonl` |
| `scripts/trc/train_trc_layer_gates.py` | TRC loss 实现 | `compute_trc_row_loss()`, `hidden_residual_loss()` |
| `skill/command/run_20260519_trc_layer_init1_v3_directional.sh` | 当前 v3 directional 启动参数 | 默认 `directional`, `auto span`, `floor=0.8` |

## Calibration 数据统计

### 1. TRC96: `20260519_trc_v1`

路径：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl
```

| task | rows | success rows | unique prompts | 数据集 | expert/来源 |
|---|---:|---:|---:|---|---|
| tool | 32 | 32 | 27 | `ToolRL/rlla_4k` | `tool_paper96_s2` |
| memory | 32 | 32 | 28 | `MemAgent/HotpotQA train parquet` | `memory_paper96_s2` |
| code | 32 | 32 | 27 | `CURE/CodeContests_train` | `code_source_00` 30, `code_source_01` 1, `code_source_02` 1 |

DeepSeek 混入：有，但极少。Code 选中的 32 条里 `code_source_02` 只有 1 条；summary 中标注来源策略为 ReasonFlux 优先，DeepSeek-R1 和 old code expert fallback。

Formal eval distribution：无。Tool/Memory/Code 都来自训练或源数据分布，不直接拷贝 BFCL/LiveBench/LiveCodeBench 正式题。

论文风险：低。问题是 Tool/Memory/Code 都存在重复 prompt 补样本：tool 重复 5 行，memory 重复 4 行，code 重复 5 行。作为 96 条小 calibration 可以接受，但不能声称 96 unique prompts。

诊断价值：适合做 clean anchor。它能回答“expert 成功轨迹 hidden residual 是否能学习出稳定 gate”，但不能充分解释 formal BFCL live 和 formal CURE code。

### 2. L5: `20260519_l5_cure_eval_code16`

路径：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16
```

| 项 | 数值 |
|---|---:|
| 新增 code prompts | 16 |
| LiveBench / LiveCodeBench | 8 / 8 |
| expert rollout rows | 16 |
| expert positive samples | 23 |
| merged prompt rows | 112 |
| base fail 要求 | 是，`require_base_fail=true` |
| reward 对齐 | 使用 CURE 官方题面与前 8 个 tests，本地 `CodeRewardAdapter` 复验 reward=1.0 |

正样本 expert 分布来自 selection blueprint：

| expert | positive prompt rows |
|---|---:|
| `deepseek_r1_distill` | 13 |
| `memory_agent` | 6 |
| `reasonflux` | 4 |

注意：一个 prompt 可以有多个 positive experts，所以计数和超过 16。JSONL 样本里 `details.expert_name` 记录真实来源；顶层 source 被统一为 `l5_cure_eval_code16_expert_success`。

DeepSeek 混入：强。16 个 formal CURE prompt 中 13 个有 DeepSeek 正轨迹。

Formal eval distribution：有。直接来自 `LiveBench` 与 `LiveCodeBench` 正式题、正式 tests、正式 temp outputs。

论文风险：高。如果用于主方法训练，会被审稿人质疑 eval leakage。可用于诊断、上限验证、ablation 的 “eval-distribution calibration”，但主表必须单独标注，不应和 leak-safe 结果混合。

诊断价值：最高。它直接回答当前 Code 问题是不是 calibration/eval mismatch；如果 L5 也不能提升 Code，说明 hidden residual proxy 或 code delta 空间本身有问题。

### 3. L6: `20260519_l6_bfcl_tool16_cure_code16`

路径：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16
```

| task | prompt rows | extra expert rows | positive samples | 来源 |
|---|---:|---:|---:|---|
| tool | 16 | 16 | 16 | BFCL official possible_answer anchor |
| code | 16 | 16 | 23 | L5 formal CURE Code16 verified positives |
| memory | 0 extra | 0 | 0 | 保留 paper96 32 rows |

合并后任务分布：

| task | merged rows |
|---|---:|
| tool | 48 |
| memory | 32 |
| code | 48 |

Tool16 结构：

| group | rows |
|---|---:|
| non_live | 8 |
| live | 8 |
| `parallel` | 4 |
| `parallel_multiple` | 4 |
| `live_parallel` | 4 |
| `live_parallel_multiple` | 4 |

Tool recoverability 审计显示：这 16 条 BFCL prompt 上历史模型真实做对很少，`rows_with_any_model_success=4`，ToolRL 只成功 2/16；L6 的 tool OPD 文件用的是 canonical possible_answer，不是模型 rollout。

DeepSeek 混入：Code 部分同 L5，强混入；Tool 部分无 DeepSeek 模型轨迹，使用 official answer。

Formal eval distribution：有。Tool 来自 BFCL official eval，Code 来自 LiveBench/LiveCodeBench formal CURE。

论文风险：很高。Tool official answer anchor 等价于把正式答案形态放入训练；Code 也直接引入正式题。适合压力测试“如果能力点被精确给到，TRC 是否能学”，不适合 leak-free 主结果。

诊断价值：高。可以用来查 Tool/Code formal ability 是否因为数据分布缺失而失败；尤其 Tool live/non-live 结构可以检验 tool-call span loss 是否对 BFCL schema 有用。

### 4. Eval-targeted96: `eval_targeted96_cure_aligned_20260517`

路径：

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517
```

Prompt 分布：

| task | rows | 来源 |
|---|---:|---|
| tool | 32 | 16 paper96 ToolRL/RLLA anchors + 16 synthetic BFCL-style prompts |
| memory | 32 | HotpotQA train parquet |
| code | 32 | 16 paper96 CodeContests anchors + 16 CodeContests-train eval-tag selected rows |

泄漏控制：README 和 summary 均声明没有拷贝正式 BFCL prompt/possible_answer/model output，也没有拷贝 LiveBench/LiveCodeBench prompt/tests/generated code/output。它使用 formal case study 反推 failure tags，再从训练源构造 eval-style 分布。

Expert rollout 成功统计：

| task/expert rollout | rows | samples | success samples | covered prompts | coverage |
|---|---:|---:|---:|---:|---:|
| tool ToolRL seed20260517 | 32 | 128 | 89 | 24 | 0.75 |
| tool ToolRL seed20260518 | 32 | 128 | 88 | 24 | 0.75 |
| memory RL-MemoryAgent seed20260517 | 32 | 128 | 106 | 29 | 0.906 |
| memory RL-MemoryAgent seed20260518 | 32 | 128 | 110 | 28 | 0.875 |
| code ReasonFlux seed20260517 | 32 | 256 | 69 | 15 | 0.469 |
| code ReasonFlux seed20260519 | 32 | 256 | 67 | 16 | 0.500 |
| code ReasonFlux seed20260518 | 16 | 128 | 41 | 未见 coverage file | 约 16 rows 内统计 |
| code DeepSeek-R1 seed20260517 | 32 | 256 | 92 | 17 | 0.531 |
| code DeepSeek-R1 seed20260518 | 32 | 256 | 90 | 16 | 0.500 |
| code DeepSeek-R1 seed20260519 | 2 | 16 | 5 | 2 | 部分文件 |

DeepSeek 混入：可选且有价值。DeepSeek 在 eval-targeted code 上成功样本数高于 ReasonFlux，但仍只覆盖约一半 prompt，说明它能补一部分 reasoning/code 轨迹，不是全能 teacher。

Formal eval distribution：没有直接混入正式题；有 eval-derived tags 和 synthetic BFCL-style prompts。论文风险中等偏低，远低于 L5/L6。需要在论文中明确这是 “case-study-informed calibration”，不能把它描述成完全随机训练集。

诊断价值：适合主线优先尝试。它和 formal eval 能力点更接近，又避免直接泄漏。

## TRC Loss 实现审计

### 当前真实流程

入口：`scripts/trc/train_trc_layer_gates.py::compute_trc_row_loss()`。

每条 row 使用同一个 `prompt + expert response` 做三次 forward：

```text
h_base   = gate 全 0
h_expert = 只打开该 row 对应 expert，系数 1
h_merge  = 当前可学习 gate
```

目标 residual：

```text
r_target = h_expert - h_base
r_merge  = h_merge  - h_base
```

默认 v3 启动脚本：

```text
residual_objective = directional
hidden_layers = 8,16,24,28
topk_tokens = 128
residual_weight_power = 0.5
response_span_mode = auto
directional_projection_floor = 0.8
directional_projection_weight = 0.1
coefficient_floor = 0.9
coefficient_floor_weight = 0.05
beta_base = 0.02
gamma_gate = 0.001
lr = 0.03
accumulation_steps = 96
```

总 loss：

```text
L_total = L_residual
        + beta_base * L_base_drift
        + gamma_gate * L_gate_anchor
        + coefficient_floor_weight * L_coefficient_floor
```

### Span 选择

当前 `response_span_mode=auto`：

| task | span |
|---|---|
| tool | 优先 `<tool_call>...</tool_call>`；找不到则 full response fallback |
| code | 优先最长 fenced code block；找不到则 full response fallback |
| memory | full response |

这个设计合理：Tool 的核心行为是函数调用 JSON/schema；Code 的核心行为是可执行代码块；Memory 的能力通常分布在检索/证据组织/最终回答整段轨迹里。

风险：Code 如果 teacher 输出不是 fenced code block，会 fallback 到 full response，解释文本会污染 hidden target。下一轮 Code 数据应强制记录是否命中 `code-block`，低命中数据不要优先使用。

### Hidden layers 与 top-k

当前只观察 hidden layers `8,16,24,28`。注意这里的 layer index 是 `outputs.hidden_states` index，不是可学习 gate 的全部 28 层；可学习参数仍是 28 层 × expert 的 direct coefficient。

每层在 span 内按 `||r_target||^0.5` 取 top-128 token 并归一化权重。它能聚焦 expert task vector 真正改变表示的位置，也避免长 Code/Memory response 直接按长度支配。

风险：Code 的关键能力可能出现在 prompt schema/题面理解、I/O 格式、算法规划 token，而不仅是 final code block token。只看 response code block 可能错过题面约束和生成前规划阶段。

### Directional / Projection / Regularizer

Directional loss：

```text
L_dir = 1 - cos(r_merge, r_target)
```

合理性：多 expert merged residual 不应该等于单个 expert residual；只要求包含目标方向，可以保留其他 expert 正交能力。

Projection floor：

```text
projection_ratio = dot(r_merge, r_target) / ||r_target||^2
L_proj = relu(floor - projection_ratio)^2
```

合理性：避免只有方向相近但幅度太小。风险是 floor 太高时会把 code/tool gate 过推，尤其 Code hidden proxy 和 unit-test pass-rate 不单调。

Base drift：

```text
L_base_drift = mean ||h_merge_prompt - h_base_prompt||^2
```

只在 prompt tail tokens 上算，当前权重较轻。它是 prompt 表示保持项，不是正式 KL 或 NLL retention。

Gate anchor / coefficient floor：都是软约束。当前 floor 防止 expert 被压太低，但不能保证 Tool/Memory formal 能力，也不能替代评测门槛。

### Task-specific override

代码已支持：

```text
--task-hidden-layers
--task-response-span-mode
--task-topk-tokens
--task-residual-weight-power
--task-directional-projection-floor
--task-directional-projection-weight
--task-loss-multiplier
```

当前没有实现真正 attention loss。代码中只有 `attention_mask` 用于 forward；没有读取 `outputs.attentions`，也没有 schema-code/token alignment loss。

判断：task-specific override 是合理的，应该继续用显式开关做实验，不要改默认路径。优先对 Code 调 span/layers/top-k/floor；对 Tool 可以小心探索 prompt schema token 与 tool-call token 的 attention/hidden 对齐，但这属于新增方法，不能在当前报告中视为已实现。

## 为什么 Code 不能只看 hidden loss

Code 与 Tool/Memory 的差异是：formal Code reward 是离散 unit-test pass-rate，hidden residual 更像“专家轨迹相似度”。

当前证据已经显示 proxy mismatch：`dir_i8` 的 Code gate 被推高、TRC Code residual loss 降低，但 LiveBench Acc 没有超过 `anchor_i8`。这说明 Code hidden direction 可以变得更像 expert，却不必然提升算法正确性、I/O 格式、边界条件、复杂度和代码抽取稳定性。

Code hidden loss 最适合作为 dense signal，用来决定“往哪个 expert residual 方向走”；但收敛点必须靠 CURE-style validation 或 formal CURE 评测来选择。否则容易出现：

1. gate 过高，放大 code delta 噪声；
2. 学到代码风格而不是 test-passing algorithm；
3. teacher 轨迹本身来自 weak/misaligned expert；
4. fenced code span 命中但题面约束理解没有被监督；
5. projection floor 推动幅度，但幅度和 pass-rate 非单调。

## 下一轮实验优先级

### P0：leak-safe 主线

优先组合：

```text
数据：eval_targeted96 prompts + verified expert positives
Code teacher：ReasonFlux + DeepSeek-R1 只用 trajectory，不加 R1 delta
Tool teacher：ToolRL generated positives，优先保留 synthetic BFCL-style 成功轨迹
Memory teacher：RL-MemoryAgent positives
Loss：directional + projection floor，小心控制 Code floor
Gate：layer-band-coefficient，init=1.0
```

理由：它针对 formal failure tags 设计，但没有直接拷贝 formal eval prompt/tests。它最适合写进论文主线。

建议筛选：

- 每 task 保持 32 条左右；
- Code 只选 verified reward=1.0 的 trajectory，且优先覆盖 `stdin_stdout`、`format_sensitive`、`math`、`greedy`、`array/string/simulation`；
- 对每个 Code prompt 最多保留 1-2 条 teacher trajectory，避免少数 prompt 多样本主导；
- 记录 DeepSeek/ReasonFlux 来源，做 “RF only vs RF+DeepSeek trajectory” 对照。

### P1：formal diagnostic 上限

优先组合：

```text
L5 Code16: 只诊断 Code formal alignment
L6 Tool16+Code16: 诊断 Tool/Code formal anchors 是否能被 TRC 学到
```

用途：如果 L5/L6 能显著提升 formal Code/Tool，说明主要问题是 calibration mismatch；如果 L5/L6 也不能提升，说明 hidden residual loss 或 delta 空间存在根本 proxy gap。

论文写法：只能作为 diagnostic / oracle-style calibration，不进入 leak-free 主结果。

### P2：loss 消融

最值得做：

1. `directional` vs `directional + projection floor`：检查 floor 是否导致 Code 过推。
2. Code `hidden_layers=4,8,12,16,20,24,28` vs 默认 `8,16,24,28`：看早中层题面/规划表征是否更关键。
3. Code `topk=64/128/256`：64 更稀疏，256 更稳；需要看是否过拟合长代码。
4. Code `response_span=code-block` 且丢弃 fallback row：减少解释文本污染。
5. Tool `tool-call` span 保持，不建议 full response。

暂不优先做：直接上 MSE。MSE 会把其他 expert 的正交 residual 当误差，已被 v2 早停现象证明容易压低多能力合并。

### P3：高风险方向

不建议今晚优先：

- 用 L5/L6 作为主训练数据直接报 SOTA；
- 加未缩放 R1 delta；
- 只按 TRC loss 选择 final checkpoint；
- 只追 Code gate 变高；
- 对 Code formal eval 数据做大量 prompt-level训练。

## 推荐四路实验模板

| run | 数据 | teacher | loss override | 目的 |
|---|---|---|---|---|
| A clean anchor | `20260519_trc_v1` | TRC96 原选择 | 当前 v3 directional | 维持可复现 baseline |
| B evaltarget RF | `eval_targeted96` | ToolRL + MemAgent + ReasonFlux positives | Code code-block, directional floor 0.6-0.8 | leak-safe Code 改善 |
| C evaltarget RF+DS | `eval_targeted96` | B + DeepSeek positives | 同 B | 判断 DeepSeek trajectory 是否补 reasoning |
| D formal diagnostic | L5 或 L6 | formal verified positives | 同 B，显式标 leak-diagnostic | 判断 eval-aligned 数据上限 |

评测门槛建议：

```text
先测 Tool/Memory。
Memory F1 >= 0.76 且 Tool mean >= 0.79，再测 Code。
不过线不测 Code，避免浪费 GPU。
```

## 最终建议

今晚要冲最高性能，应把 P0/P1 分开：P0 用 `eval_targeted96` 走 leak-safe 主线，P1 用 L5/L6 做 oracle diagnostic。Code 的核心不是继续推高 code gate，而是让 selected trajectory 和 CURE pass-rate 更一致；DeepSeek 适合先作为 trajectory teacher，不适合此阶段直接加入 delta。
