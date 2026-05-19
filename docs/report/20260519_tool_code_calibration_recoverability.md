# 20260519 Tool/Code Calibration 可恢复性审计

## 结论

1. 已构建 `paper96 + BFCL Tool16 + CURE Code16` 合并版 calibration，独立于 L4/L5，不覆盖旧数据。
2. Tool16 覆盖 BFCL non-live 8 条、live 8 条，但当前 Tool OPD 正样本是 BFCL 官方 `possible_answer`，不是 ToolRL/R1 模型真实 rollout。
3. 可恢复性统计显示：ToolRL 在这 16 条 Tool16 上真实做对 `2/16`，本机可查模型中任意一个做对的只有 `4/16`。如果坚持只用模型 rollout 作为 Tool OPD positive，这批 Tool16 的可恢复信号偏少。
4. Code16 的正轨迹质量明显更好：16 条都有至少一个真实专家成功，DeepSeek-R1-Distill-Qwen-7B 做对 `13/16`，总 positive samples 为 `23`。
5. 本机没有找到纯 DeepSeek-R1-Distill-Qwen-7B 的 BFCL result；只有 R1-injected merged model 的 BFCL result，因此不能证明纯 R1 在 Tool16 上有正确 tool trajectory。

## 合并数据

路径：

- prompt：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16/qbank_c033333_paper96_plus_bfcl_tool16_cure_code16_seed20260519.prompts.jsonl`
- extra expert rollout：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16/bfcl_tool16_cure_code16_extra_expert_rollouts_seed20260519.jsonl`
- summary：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16/qbank_c033333_paper96_plus_bfcl_tool16_cure_code16_seed20260519.summary.json`
- config：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260519_l6_tool_code_eval16.md`

计数：

| task | rows |
|---|---:|
| tool | 48 |
| memory | 32 |
| code | 48 |
| total | 128 |

额外 expert rollout：

| task | rows | positive samples | 来源 |
|---|---:|---:|---|
| tool | 16 | 16 | BFCL official possible_answer |
| code | 16 | 23 | ReasonFlux / DeepSeek-R1-Distill / RL-MemoryAgent verified outputs |

## Tool16 审计

数据分布：

| split/category | rows |
|---|---:|
| non_live | 8 |
| live | 8 |
| parallel | 4 |
| parallel_multiple | 4 |
| live_parallel | 4 |
| live_parallel_multiple | 4 |

模型真实轨迹可恢复性：

| model | success | mean reward | missing |
|---|---:|---:|---:|
| qwen25_instruct | 1/16 | 0.5596 | 0 |
| toolrl | 2/16 | 0.6036 | 0 |
| reasonflux | 1/16 | 0.5596 | 0 |
| r1_injected_alpha001 | 2/16 | 0.6276 | 0 |

解释：

- Tool16 的官方答案 anchor 是满分 `16/16`，所以作为 OPD 正样本一定可用。
- 但 ToolRL/R1-injected 等真实模型 rollout 成功率很低，说明这 16 条主要是 hard eval anchors，不是高密度 model-positive recovery bank。
- 如果训练目标是“用专家真实轨迹 distill”，Tool16 需要重新生成更多 ToolRL samples 或降低选择难度；如果目标是“强行对齐 BFCL parser/canonical answer”，官方答案 anchor 可以保留，但论文中要说明它是 oracle-style calibration。

## Code16 审计

Code16 来自正式 CURE `LiveBench` / `LiveCodeBench` 各 8 条。选择条件是 base Qwen2.5-7B-Instruct 无全通过样本，至少一个 code-capable expert 有全通过样本。

统计：

| item | count |
|---|---:|
| prompt rows | 16 |
| expert rollout rows | 16 |
| positive samples | 23 |
| base success rows | 0 |
| rows with any expert success | 16/16 |
| R1 positive rows | 13/16 |
| R1 positive samples | 13 |

按专家：

| expert | positive rows | positive samples |
|---|---:|---:|
| deepseek_r1_distill | 13 | 13 |
| memory_agent | 6 | 6 |
| reasonflux | 4 | 4 |

解释：

- Code16 的 OPD positive 是真实模型输出，并且已由本地 `CodeRewardAdapter` 用 CURE 前 8 个官方测试重新验证。
- R1 在 Code16 上是主要 positive 来源，因此这批数据适合测试 R1/code delta 是否能被 gate 学出来。
- 这批 Code16 可恢复性充足，不属于“专家也做不对”的坏 calibration。

## 训练建议

下一步如果要跑：

```bash
PHASE=L6 GPU_LIST=6,7 NUM_ITERS=20 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

建议观察：

- Tool proxy 是否比 L5 稳定；
- Code all-fail 是否下降；
- Code/R1 gate 是否因 Code16 positive 增强而上涨；
- Tool gate 是否仍被 task weight `0.5` 压制。

如果 Tool 仍崩，优先问题不是“有没有 Tool16 prompt”，而是 Tool16 的 model-positive recovery 过少，当前 Tool OPD 更像 oracle answer forcing，不一定能提供自然模型分布下的 distillation 信号。
