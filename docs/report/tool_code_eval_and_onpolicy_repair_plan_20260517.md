# Tool/Code 评测差异与 On-policy 修复计划

日期：`2026-05-17`

## 产物

前端：

```text
http://127.0.0.1:8791
```

数据：

```text
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/
├── app_data.json                         # 28M, 前端直接加载
├── cases.jsonl                           # 22M, BFCL + CURE case-level
├── pairwise_diffs.jsonl
├── bfcl_live_calibration_candidates.jsonl
└── model_metrics.json
```

模型：

| model | role |
|---|---|
| `best-ever-tame-cg-r1calib-global-v2` | 当前最好综合模型 |
| `expertgym-B-codeaug-opd-i18` | 当前 B 设置模型 |
| `ta-c075` | Task Arithmetic `0.75` baseline |

case 数量：

| benchmark | cases |
|---|---:|
| BFCL | 440 |
| CURE | 639 |
| total | 1079 |

## 总体指标

| model | Tool mean | Tool live | Code BoN mean | Code sample acc |
|---|---:|---:|---:|---:|
| best-ever | 0.7954 | 0.7083 | 0.4408 | 0.3597 |
| B | 0.7823 | 0.6771 | 0.3871 | 0.3370 |
| TA-0.75 | 0.7850 | 0.6875 | 0.4222 | 0.3585 |

结论：B 的 Tool 比 TA-0.75 略低，主要亏在 live；B 的 Code 明显低于 TA-0.75 和 best-ever，说明当前 OPD code augmentation 没有转化成正式 CURE 能力。

## Tool 差异

best-ever vs B：

| BFCL subset | both correct | both wrong | best wins | B wins |
|---|---:|---:|---:|---:|
| live_parallel | 11 | 4 | 1 | 0 |
| live_parallel_multiple | 16 | 8 | 0 | 0 |
| parallel | 177 | 15 | 4 | 4 |
| parallel_multiple | 170 | 24 | 2 | 4 |
| total | 374 | 51 | 7 | 8 |

Tool 不是单向碾压：B 也有 8 题赢 best-ever。真正短板集中在：

- `parallel_alignment`: 多意图、多调用时 call 数量和意图绑定不稳；
- `wrong_function`: 相似函数或 namespace 下选错；
- `enum_exactness`: enum/string value exact match 不稳；
- `parameter_value_error`: 参数值规范化不稳；
- `default_value`: schema 默认值/optional 参数处理不稳；
- `canonicalization`: 中文/西语/韩语实体到 scorer 接受值的规范化不稳。

TA-0.75 vs B：

- TA 与 B Tool mean 接近：`0.7850` vs `0.7823`。
- TA 的 `live_parallel=0.7500` 明显高于 B 的 `0.6875`。
- B 的 `live_parallel_multiple=0.6667` 高于 TA 的 `0.6250`。

解释：Tool 能力不是“系数越大越好”，而是 live 子分布和 parallel-multiple 子分布存在不同偏好。on-policy 修复不应只推 tool gate，而应构造能覆盖 live/default/canonical 与 multi-call alignment 的 calibration。

## Code 差异

Code primary success 使用 CURE BoN(4,4)：由模型生成的测试选择一个 code sample，再看该 sample 是否通过 hidden tests。

| model | LiveBench BoN | LiveBench any-pass | LiveBench sample | LCB BoN | LCB any-pass | LCB sample |
|---|---:|---:|---:|---:|---:|---:|
| best-ever | 0.5000 | 0.5391 | 0.4062 | 0.3816 | 0.4560 | 0.3131 |
| B | 0.4219 | 0.5156 | 0.3672 | 0.3523 | 0.4618 | 0.3068 |
| TA-0.75 | 0.4688 | 0.5391 | 0.4004 | 0.3757 | 0.4599 | 0.3165 |

关键观察：

1. B 的 LiveBench code generation 明显弱：sample acc `0.3672`，低于 TA `0.4004` 和 best `0.4062`。
2. B 的 LiveCodeBench any-pass 不低：`0.4618`，略高于 TA `0.4599` 和 best `0.4560`；但 BoN 低，说明很多题“有正确 sample”，但 generated-test selection 没选中。
3. best-ever 的优势不只是生成正确代码，也包括更好的 test-guided selection：LiveBench BoN `0.5000`，LCB BoN `0.3816`。

best-ever vs B pairwise：

| CURE subset | both correct | both wrong | best wins | B wins |
|---|---:|---:|---:|---:|
| LiveBench | 51 | 61 | 13 | 3 |
| LiveCodeBench | 149 | 285 | 46 | 31 |
| total | 200 | 346 | 59 | 34 |

TA-0.75 vs B pairwise：

| CURE subset | both correct | both wrong | TA wins | B wins |
|---|---:|---:|---:|---:|
| LiveBench | 48 | 62 | 12 | 6 |
| LiveCodeBench | 143 | 282 | 49 | 37 |
| total | 191 | 344 | 61 | 43 |

Code 失败 tag 高频：

- `no_correct_code_sample`: 生成侧没有正确代码；
- `unit_test_selection_failure`: 有正确代码但 BoN 选择错；
- `partial_unit_test_pass`: 代码过部分 hidden tests，缺边界条件；
- `format_sensitive` / `stdin_stdout`: I/O 格式敏感；
- `math` / `array` / `string` / `simulation`: 当前 CURE 题型主要分布。

## Tool 应该怎么评测

正式评测继续用 BFCL 四个 subset，但训练诊断必须拆开看：

| 维度 | 必看指标 | 用途 |
|---|---|---|
| non-live parallel | exact success | 保持基础多调用能力 |
| live_parallel | exact success | 检测 default/canonical/live entity |
| live_parallel_multiple | exact success | 检测 live + 多函数/多步骤 |
| failure tags | enum/default/canonical/wrong_function/alignment | 指导 calibration 构造 |
| pairwise diff | best/TA/B 谁赢谁输 | 避免只看均值误判 |

on-policy proxy 不应只用 ToolRL raw prompt。下一批 Tool calibration 应该按 tag 均衡采样：

- `parallel_alignment`: 多意图、多 call；
- `wrong_function`: schema distractors；
- `enum_exactness`: close enum values；
- `default_value`: optional/default 参数；
- `canonicalization`: 多语言、别名、实体规范化；
- `format_validity`: AST/tool-call parse。

## Code 应该怎么评测

正式评测继续报告 CURE 两组指标：

- `Code sample acc`: 4 个 code sample 中平均有多少能过 hidden tests；
- `Code BoN(4,4)`: generated tests 选择出的 code 是否过 hidden tests；
- `any-pass acc`: 诊断上限，表示当前 rollout 是否已经包含正确代码；
- `BoN hidden-test acc`: 被选中代码的 hidden test 通过比例。

必须区分两类短板：

| 短板 | 现象 | 修复信号 |
|---|---|---|
| generation 不够 | sample acc / any-pass 低 | expert OPD + verified code NLL |
| selection 不够 | any-pass 高但 BoN 低 | generated-test OPD / counterexample tests |

B 在 LiveBench 两者都弱；在 LiveCodeBench 主要是 selection 弱。

## On-policy 修复方案

### Tool

1. 从前端 `Calibration Candidate Queue` 读取 BFCL failures，但只把 tag/schema pattern 当模板，不复制官方题。
2. 合成 BFCL-live-style calibration：
   - fresh function schema；
   - fresh entities；
   - fresh enum/default/canonical cases；
   - scorer-compatible possible answers。
3. 每轮 rollout 后分三类：
   - all-success：NLL preservation，保持 tool 格式；
   - frontier：GRPO，优化当前可分辨题；
   - all-fail + expert-success：OPD，把模型推到可探索区。
4. Tool reward report 必须按 tag 和 subset 分解，不只看均值。

### Code

1. calibration 数据按 CURE 诊断拆两池：
   - `generation pool`: 当前 policy 全错，但 code expert / TA / best 能给 verified passing code；
   - `selection pool`: policy 有至少一个 passing code，但 generated tests 选错。
2. generation pool 用 OPD：
   - 正样本必须是执行验证通过的 code；
   - 对同一 prompt 保留多专家 BoN positive；
   - 目标是提高 sample acc / any-pass。
3. selection pool 用 test-guided loss：
   - 对 generated test 能区分正确/错误代码的轨迹做 OPD；
   - 或直接把 “正确 code + 能杀错解的 tests” 作为双轨迹蒸馏；
   - 目标是提高 BoN acc，而不是只提高 any-pass。
4. GRPO 只在同一 prompt 的 rollout 有 reward 方差时使用：
   - reward 用 hidden/public executable tests；
   - advantage 按 prompt 分组；
   - 单任务内样本平均，再和 Tool/Memory 做任务级均衡。
5. 对 Code 的训练监控必须同时看：
   - sample acc；
   - any-pass；
   - BoN acc；
   - unit-test selection failure count；
   - response extraction failure。

## 下一步实验建议

最小可执行实验：

| 实验 | 改动 | 判据 |
|---|---|---|
| T-code-gen | 加 verified passing code OPD，只针对 `no_correct_code_sample` | sample acc / any-pass 上升 |
| T-code-select | 加 generated-test/counterexample OPD，只针对 `unit_test_selection_failure` | BoN acc 上升，any-pass 不必大变 |
| T-tool-live | 加 BFCL-live-style tag-balanced synthetic tool data | Tool live mean 上升 |
| T-mixed | Tool-live + Code-gen/select，保持 Memory 数据不变 | Tool/Code 上升且 Memory F1 不回退 |

论文叙事上，最好不要只说“on-policy 学 gate”。更强的故事是：

> on-policy gate learning 需要把 evaluation failure 分解成可优化能力单元；Tool 用 schema/canonical/alignment tags，Code 用 generation-vs-selection tags。GRPO 负责利用 frontier 方差，OPD 负责把全错样本推入可探索区域，retention 负责守住已正确行为。
