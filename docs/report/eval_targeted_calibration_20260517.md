# Eval-Targeted 96 Calibration Data 设计与产物

## 最终推荐版本：CURE-Aligned Mixed 96

为避免影响已经跑过的 `paper96` 和第一版 `eval_targeted96_20260517`，最终推荐数据单独写入新目录：

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl
```

配套审计文件：

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/summary.json
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/tool_synthetic_blueprints.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/code_train_blueprints.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/README.md
```

这版相较第一版的关键变化：Code 从 `32` 条全 targeted CodeContests 改为 `16` 条旧 `paper96` Code frontier anchor + `16` 条 CURE-style targeted CodeContests。这样既保留旧 96 中已经验证过的 Code 分布，又补正式 CURE 评测中暴露出的 `stdin/stdout`、hidden-test、format、math/greedy/array/string/simulation/graph 缺口。

最终结构：

| task | rows | 组成 | 目的 |
|---|---:|---|---|
| tool | 32 | 16 paper96 ToolRL/RLLA anchor + 16 BFCL-style synthetic | 保留 ToolRL/RLLA 原分布，同时补 BFCL live/parallel/schema-following |
| memory | 32 | 32 paper96 HotpotQA-train anchor | memory 当前不是主要短板，先保持稳定 |
| code | 32 | 16 paper96 Code anchor + 16 CURE-style targeted CodeContests | 保留旧 code frontier，同时补 CURE formal eval 能力轴 |

计数审计：

| item | count |
|---|---:|
| total prompts | 96 |
| unique prompt ids | 96 |
| tool source anchor | 16 |
| tool BFCL-style synthetic | 16 |
| memory paper96 anchor | 32 |
| code source anchor | 16 |
| code targeted probe | 16 |

Code targeted tags：

| tag | count |
|---|---:|
| paper96_source_anchor | 16 |
| stdin_stdout / math / greedy / format_sensitive / string | 16 each |
| array / graph / simulation | 14 each |
| dynamic_programming | 9 |

复现命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_eval_targeted_calibration.py \
  --output-dir /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517 \
  --tool-source-count 16 \
  --tool-synthetic-count 16 \
  --memory-count 32 \
  --code-count 32 \
  --code-source-count 16 \
  --code-targeted-count 16 \
  --seed 20260517
```

校验结果：

- `python -m py_compile scripts/data/build_eval_targeted_calibration.py` 通过。
- 新 manifest 为 `tool=32 / memory=32 / code=32`，`96` 个 prompt id 全部唯一。
- Code 32 条全部能解析到 CodeContests source tests，当前 reward 侧可执行 ground-truth tests。
- 16 条 synthetic Tool reference response 经当前 `RewardRouter` / BFCL adapter 检查，全部 `success=True` 且 `reward_train=1.0`。
- 旧 `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/` 未覆盖；新旧重叠为 80 条，其中 Tool 32、Memory 32、Code 16，符合“保留旧锚点 + 替换 16 条 Code targeted”的设计。

推荐训练时的 Code reward 对齐参数：

```bash
ADVANTAGE_NORMALIZATION=zscore
TASK_NORMALIZE_ADVANTAGES=0
CODE_MAX_NEW_TOKENS=10000
OPVEC_CODE_REWARD_MAX_TESTS=8
```

如果算力允许，Code 的 `samples_per_prompt` 应向 `8/16` 靠近；当前统一 samples-per-prompt 若仍用 `4`，这版数据依然可用，但 Code 方差会弱于 CURE/ReasonFlux 的 `k_code=16` 设置。

## 目标

基于 eval case browser 的 Tool/BFCL 与 Code/CURE case study，构造一版类比 `paper96` 的 96 条 calibration manifest，但让它更像 formal eval 真正考的能力，同时避免明显 benchmark leakage。

产物：

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/eval_targeted96.prompts.jsonl
```

配套审计文件：

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/summary.json
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/tool_synthetic_blueprints.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/code_train_blueprints.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/README.md
```

构建脚本：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/data/build_eval_targeted_calibration.py
```

复现命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_eval_targeted_calibration.py
```

## Case Study 结论

输入 case queue：

```text
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/bfcl_live_calibration_candidates.jsonl
```

候选规模：

| benchmark | rows |
|---|---:|
| BFCL | 58 |
| CURE | 405 |

Tool/BFCL 主要失败：

| failure tag | count | calibration 含义 |
|---|---:|---|
| parallel_alignment | 45 | 多个 intent 必须输出正确数量、正确对齐的 calls |
| wrong_function | 36 | distractor schemas 下函数选择不稳 |
| enum_exactness | 35 | enum 必须精确，不能 plural/synonym |
| parameter_value_error | 35 | 参数值规范化和 schema 对齐不足 |
| canonicalization | 13 | 中文/西语/韩语实体要转 canonical English |
| wrong_count | 12 | 漏 call / 多 call |
| default_value | 6 | 未指定 optional 参数时不能乱填非默认值 |

Code/CURE 主要失败：

| failure/tag | count | calibration 含义 |
|---|---:|---|
| stdin_stdout | 405 | 评测强调脚本式 stdin/stdout，不是函数式回答 |
| multi_test_hidden_eval | 402 | 需要多 hidden tests，不能只过 public examples |
| math | 360 | 数学/整数/公式题占主导 |
| no_correct_code_sample | 358 | 生成阶段本身不足，OPD positive 需要 expert verified code |
| partial_unit_test_pass | 269 | 易过样例但挂边界 |
| format_sensitive | 214 | 输出格式、大小写、空格敏感 |
| greedy/array/string/simulation/graph | 188/148/125/124/35 | 需要覆盖常见算法技能轴 |

## 数据设计

保持 `paper96` 的总量和任务平衡：

| task | rows | 来源 | 目的 |
|---|---:|---|---|
| tool | 32 | 16 条 paper96 ToolRL/RLLA + 16 条 fresh BFCL-style synthetic | 保住原 Tool 分布，同时补 BFCL live / parallel / schema-following 能力 |
| memory | 32 | 复用 paper96 HotpotQA-train memory | memory 当前不是主要短板，保留旧锚点 |
| code | 32 | CodeContests train targeted resample | 用非 eval 数据覆盖 CURE-like stdin/stdout + hidden-test 能力 |

最终 manifest 已 round-robin interleave：`tool -> memory -> code` 循环，避免 rollout/update 阶段连续任务块造成局部统计偏置。

## Tool 构造

Tool 现在是混合设计，不再用纯 synthetic 替换原 Tool 数据：

| Tool source | rows | 作用 |
|---|---:|---|
| paper96 ToolRL/RLLA source anchor | 16 | 保住原始 `<tool_call>` / ToolRL source reward 分布 |
| fresh BFCL-style synthetic | 16 | 暴露 BFCL live / parallel / default / enum / canonicalization 缺口 |

synthetic 部分不使用官方 BFCL prompt/answer/result，只按 failure taxonomy 合成 fresh schemas/entities/prompts，并用当前 `BFCLToolRewardAdapter` 可验证的 `reference.bfcl` 作为 reward。

synthetic Tool 子类：

| BFCL-style category | rows |
|---|---:|
| live_parallel | 7 |
| live_parallel_multiple | 6 |
| parallel | 1 |
| parallel_multiple | 2 |

覆盖能力：

- 多调用对齐：2-4 个 reference calls；
- enum exactness：`piece` vs `pieces`、`snack/lunch/dinner`、`on/off/eco/boost`；
- default discipline：未指定 `unit/meal_type/channel/priority/mode` 时允许省略或默认值；
- canonicalization：中文/西语/韩语 query 映射到 canonical English 参数；
- wrong-function resistance：混合 `set/get` device、message、reminder 等 distractor schemas。

校验：16 条 Tool synthetic 的 `reference.response` 用当前 `RewardRouter` 全部 `success=True, reward=1.0`。

## Code 构造

Code 不使用 LiveBench/LiveCodeBench 官方 prompt/tests/outputs。脚本从 `CodeContests_train` 中按 CURE case-study tag 重采样，保留可执行 `test_input/test_output` 到 `reference.metadata`，让现有 `CodeRewardAdapter` 直接用 ground-truth tests 给 reward。

选择 bucket：

| bucket | rows |
|---|---:|
| math | 7 |
| greedy | 5 |
| array | 5 |
| format_sensitive | 5 |
| string | 4 |
| simulation | 3 |
| graph | 2 |
| dynamic_programming | 1 |

每条 Code row 至少 10 个 source tests。训练时 reward 仍是当前 CURE-style pass rate，不是文本相似度。

## Code Reward 与 ReasonFlux/CURE 对齐

用户判断是对的：如果目标是提升 LiveCodeBench/CURE formal eval，Code reward 不应只“近似 pass rate”，而应尽量对齐 ReasonFlux/CURE 的训练信号。

CURE 官方 code branch 的核心逻辑在：

```text
/mnt/cache/wuruixiao/users/lsc/CURE/optimization/reward.py
```

对每个 prompt，CURE 先采样多份 code，再执行 ground-truth tests：

```python
code_reward = np.mean(all_test_table_i, 1)
code_reward = normalize_reward(code_reward)
```

其中：

```python
normalize_reward(r) = (r - mean(r)) / std(r)
```

如果所有 sample reward 没有方差，则该 prompt 不产生 code RL 数据。这一点和 GRPO 的“同组样本相对优势”是一致的。

当前 ExpertGym 与 CURE 已对齐的部分：

- `CodeRewardAdapter` 对有 `test_input/test_output` 的 prompt 执行代码；
- raw reward 是 source tests pass rate；
- all-correct / all-wrong 且无方差的 row 不会进入 GRPO frontier；
- OPD 可以用 verified passing code 做 positive。

当前仍不完全对齐的部分：

| 项 | CURE/ReasonFlux | 当前 ExpertGym 默认 | 影响 |
|---|---|---|---|
| code advantage | per-prompt z-score | 默认 `centered`，只减均值 | 梯度尺度弱，和 CURE 不一致 |
| samples per prompt | `k_code=16` | 常用 2/4 | 更难形成 pass-rate 方差 |
| max generation token | `10000` | 常用 1536/4096 | 长代码题可能被截断 |
| max ground-truth tests | `8` | 当前默认 env 也是 8 | 基本一致 |
| unit test branch | 同时训练 case/test generator | 当前没有训练 unit-test generator | BoN/select 能力难复制 |
| training target | 全模型 code RL | gate-only task vector 学习 | 可塑性明显更弱 |

因此最小可执行对齐方案：

```text
Code raw reward: pass_rate over source tests
Code advantage: per-prompt z-score
Code no-variance row: skip GRPO
Code samples_per_prompt: 尽量 8/16
Code max_new_tokens: 接近 10000，至少不能低于 formal eval 常用上限
```

在当前 updater 中对应设置：

```bash
ADVANTAGE_NORMALIZATION=zscore
TASK_NORMALIZE_ADVANTAGES=0
CODE_MAX_NEW_TOKENS=10000
OPVEC_CODE_REWARD_MAX_TESTS=8
```

如果算力受限，`samples_per_prompt=8` 是折中；若要更接近 CURE，Code 应使用 `16`。

更完整的 ReasonFlux/CURE 对齐还需要第二阶段：

- rollout 时同时生成 code samples 和 unit-test/case samples；
- 用 code samples 的 ground-truth pass table 判断 correct code；
- 用 correct/wrong code 对生成 tests 打分，形成 CURE 的 `case_reward`；
- 训练/蒸馏 unit-test generation 或 selector，使 BoN 能选中正确代码。

这解释了为什么 ReasonFlux-Coder 的 LiveCodeBench 能力强：它不是只靠 pass-rate reward，而是 code generator 和 unit tester 的 co-evolution；formal eval 的 BoN `(4,4)` 正好利用了它学到的 tester/selector 能力。我们的 gate-only 方法若只优化 code pass_rate，不训练 case/test branch，最多对 code generation 有帮助，对 CURE BoN/selection 的帮助会有限。

## 泄漏边界

这版数据明确只用 eval case study 的统计结构，不复制正式评测内容：

- Tool：原 ToolRL/RLLA 只来自旧 paper96 source rows；synthetic BFCL-style 不复制 BFCL 官方 prompt、function schema、entities、possible_answer、模型输出，只复用“失败类型”。
- Code：不复制 LiveBench/LiveCodeBench prompt、hidden tests、generated code、outputs；只从 CodeContests train 取非 eval 题。
- Memory：沿用旧 paper96 的 HotpotQA train/source rows，不引入 formal eval memory。

审计字段：

- 每条 row 有 `eval_targeted_calibration`；
- Tool 有 `tool_synthetic_blueprints.jsonl`；
- Code 有 `code_train_blueprints.jsonl`；
- `summary.json` 记录输入、输出、任务计数、case-study profile 和 leakage policy。

## 和旧 paper96 的差异

旧 `paper96`：

- Tool 主要来自 ToolRL/RLLA，偏 `<tool_call>` 格式和 source reward；
- Code 从原 question bank 采样，未显式对齐 CURE case-study failure tags；
- Memory/Tool/Code 各 32，结构简单但 eval proxy 对 Tool live / CURE 不够敏感。

新 `eval_targeted96`：

- Tool 改为 16 条原 ToolRL/RLLA anchor + 16 条 BFCL-style scorer-compatible synthetic；
- Code 改为 CURE-like tag-targeted CodeContests train；
- Memory 保持旧锚点；
- 仍保持 96 条、三任务均衡，便于和旧实验对照。

## 使用建议

第一步不要直接替代所有正式训练，先跑 smoke：

```bash
SEED=/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/eval_targeted96.prompts.jsonl
```

建议对比：

| 实验 | seed manifest | 目的 |
|---|---|---|
| old-paper96 | `qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` | 原 proxy baseline |
| eval-targeted96 | `eval_targeted96.prompts.jsonl` | 看 Tool live-style / Code CURE-style reward 是否更有信号 |

监控重点：

1. Tool synthetic 的 parseable / reward 是否非饱和；
2. Tool coefficient 是否比旧 paper96 更能维持 BFCL live；
3. Code rows 是否出现 frontier，而不是 all-fail 全靠 OPD；
4. Memory reward 是否因新 Tool/Code 分布被牺牲；
5. formal eval 是否比 proxy 更同步，尤其 BFCL live 与 CURE Code。

## 当前判断

这版 calibration 比 paper96 更适合做下一轮 on-policy data：它不解决所有问题，但能把训练信号从“旧 source reward 能做对”转向“formal eval 真实短板可见”。如果这版仍然不能推动 Code/Tool formal eval，问题就更可能在 gate 参数化、OPD/GRPO loss routing、或者 task vector 本身的能力上限，而不是 calibration distribution 完全不对。
