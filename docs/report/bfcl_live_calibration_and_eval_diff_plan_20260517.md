# BFCL Live Calibration 与 Eval 差异可视化任务计划

更新时间：`2026-05-17 12:00 CST`

## 背景

当前 best-ever 模型信息见：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/best_ever_model.md
```

best-ever：`tame-cg-r1calib-global-v2`

- Tool mean `0.7954`
- Tool live mean `0.7083`
- Memory F1 `0.7720`
- Code Acc `0.3597`
- Code BoN `0.4408`

当前 B 设置模型：

```text
expertgym-abcB-gp-codeaug-opd-i18-20260516
```

正式 eval6：

- Tool mean `0.7823`
- Tool live mean `0.6771`
- Memory F1 `0.7118`
- Code Acc `0.3370`
- Code BoN `0.3871`

用户判断：当前 B 的 memory 已经接近训练目标，但 Tool live 与 Code 仍有欠缺。Tool 的主要问题是 calibration data 没有覆盖 BFCL live 类分布。

## 已定位的真实评测文件

### best-ever / TAME

```text
/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/diagnostics/calibration_global_r1_v2/bfcl/result/tame-cg-r1calib-global-v2-20260504/
/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/diagnostics/calibration_global_r1_v2/bfcl/score/tame-cg-r1calib-global-v2-20260504/
/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/diagnostics/calibration_global_r1_v2/summary/evaluation_summary.json
```

### Current B

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/expertgym-abcB-gp-codeaug-opd-i18-20260516/tool/result/
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/expertgym-abcB-gp-codeaug-opd-i18-20260516/tool/score/
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260516_tonight_abc_eval6.md
```

### BFCL 官方本地数据

```text
/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_parallel.json
/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_parallel_multiple.json
/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_parallel.json
/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_parallel_multiple.json
```

## 初步诊断：BFCL Live 与当前 ToolRL calibration 的分布差异

当前 `paper96` 中 tool 样本主要来自 ToolRL/RLLA 风格，reward 关注 `<tool_call>` 格式与函数参数正确性。它对“能不能产生工具调用”有用，但不足以覆盖 BFCL live 的细粒度约束。

BFCL live 的真实失败样例显示以下能力缺口：

| 失败类型 | 例子 | 训练含义 |
|---|---|---|
| multilingual canonicalization | 用户中文问北京/上海天气，答案要求 `Beijing, China` / `Shanghai, China`，模型输出 `北京, China` / `上海, China` | 需要从自然语言实体映射到 schema 允许的英文 canonical value |
| default value discipline | weather function 默认 `fahrenheit`，模型显式给 `celsius` 被判错 | 需要学习“未指定参数时不要乱填非默认值” |
| enum exactness | `piece` vs `pieces`，`lunch` vs 默认/期望 `snack` | 需要精确遵守 enum 与默认参数 |
| parallel call count | 多个城市/食物/订单必须输出对应数量工具调用，不能漏/多/错配 | 需要训练多调用对齐，不只是单工具格式 |
| namespace / mixed tool set | `ChaFod` + `ChaDri.change_drink`，或天气 + 家电控制混合 | 需要多函数选择与 namespace 保持 |
| cross-lingual instruction | Spanish / Korean / Chinese 用户输入，schema 英文 | 需要跨语言理解 + 英文参数归一 |

结论：Tool live 不只是“工具调用能力”，而是 **schema-grounded entity normalization + default/enum discipline + parallel call alignment**。

## Calibration 构造原则

### 原则 1：官方 eval live 只做诊断，不直接训练

`BFCL_v4_live_parallel*.json` 是当前 official eval 的直接题目。为避免 benchmark leakage，不应把这些 exact prompts 直接加入 calibration。可以做：

- case-level error analysis；
- failure type taxonomy；
- prompt/template 抽象；
- 生成同分布但不同实体/函数/问法的新 calibration。

### 原则 2：Tool calibration 应拆成 BFCL 子能力配额

建议下一版 96/120 prompt calibration 中，Tool 不再只放 ToolRL/RLLA。Tool 任务内部应分层：

| 子类 | 建议条数 | 目的 |
|---|---:|---|
| ToolRL/RLLA strict | 8-12 | 保留 `<tool_call>` 格式与基础 function correctness |
| BFCL non-live parallel | 6-8 | 学习稳定多调用输出 |
| BFCL non-live parallel_multiple | 6-8 | 学习多函数、多 namespace 选择 |
| BFCL live-style parallel | 8-12 | 学习 multilingual canonicalization / default discipline |
| BFCL live-style parallel_multiple | 8-12 | 学习 live 多调用 + 多函数组合 |

如果总 calibration 仍限制在约 100 条，建议总配比：

```text
tool 36-40
memory 28-32
code 28-32
```

原因：当前 memory 已达到预期，Tool live 和 Code 是短板；Tool 需要从原来的单一 ToolRL 分布扩展到 BFCL live-style。

### 原则 3：优先选“模型间有差异”的高信息量样本

构造 question bank 时，不要随机采样。应对候选 tool prompts 做多模型 rollout/score，然后按以下优先级选：

1. best-ever 正确、当前 B 错误；
2. tool expert 正确、当前 B 错误；
3. 当前 B 有部分正确但参数/enum/default 失败；
4. 多模型都错但 expert 或 reference 可提供 clean positive trajectory；
5. 多模型全对样本只少量保留，用于 retention。

这样 calibration 的目标不是覆盖题目数量，而是最大化 gate 梯度可用性。

### 原则 4：每条 calibration 必须记录可审查 metadata

每条 tool calibration 至少需要：

```json
{
  "prompt_id": "...",
  "task": "tool",
  "bfcl_category": "live_parallel | live_parallel_multiple | parallel | parallel_multiple | toolrl",
  "source": "synthetic_from_bfcl_live_template | bfcl_non_live_heldout | toolrl",
  "function_names": ["..."],
  "num_calls": 2,
  "language": "zh | en | es | ko | mixed",
  "failure_tags": ["canonicalization", "default_value", "enum_exactness"],
  "leakage_safe": true,
  "reference_calls": [...]
}
```

这会让后续前端能按 failure type / category / language / call count 聚合。

## best-ever vs B 差异分析目标

对两个模型做 case-level 对比，不只看 summary：

| 对比状态 | 意义 |
|---|---|
| best 正确，B 错 | 当前方法缺失的能力，优先转化为 calibration family |
| best 错，B 正确 | B 已有优势，避免新训练破坏 |
| 两者都错 | 判断上限：是否 expert 也能做对；如果 expert 能做对，可加入 OPD |
| 两者都对 | retention 候选，少量保留 |

重点分析 BFCL live：

- `live_parallel`: 16 条，小而高价值；
- `live_parallel_multiple`: 24 条，覆盖更多混合函数、跨语言、enum/default。

非 live parallel / parallel_multiple 各 200 条，可用于统计更稳定的工具调用模式，但论文中的 Tool live 短板主要由 live 两类解释。

## 可视化界面设计

目标：不是做 dashboard 装饰，而是让研究者能从真实 eval case 中发现下一轮 calibration 应该补什么。

### 数据输入

建议建立一个可复用 registry：

```text
docs/evaluation/eval_case_browser/models.json
```

每个模型注册：

```json
{
  "model_id": "best-ever-tame-cg-r1calib-global-v2",
  "display_name": "best-ever TAME",
  "tool_score_root": ".../bfcl/score/tame-cg-r1calib-global-v2-20260504",
  "tool_result_root": ".../bfcl/result/tame-cg-r1calib-global-v2-20260504",
  "memory_summary": ".../memory_eval/.../summary.json",
  "code_summary": ".../code_eval/.../summary.json",
  "tags": ["best-ever", "tame", "cg+r1"]
}
```

第二个模型：

```json
{
  "model_id": "expertgym-B-codeaug-opd-i18",
  "display_name": "ExpertGym B",
  "tool_score_root": ".../runs/expertgym-abcB-gp-codeaug-opd-i18-20260516/tool/score/expertgym-abcB-gp-codeaug-opd-i18-20260516",
  "tool_result_root": ".../runs/expertgym-abcB-gp-codeaug-opd-i18-20260516/tool/result/expertgym-abcB-gp-codeaug-opd-i18-20260516",
  "memory_summary": ".../eval6-memory-hotpotqa/.../summary.json",
  "code_summary": ".../eval6-code-cure-full/.../summary.json",
  "tags": ["expertgym", "B", "opd-only", "codeaug"]
}
```

后续新增模型只改 registry，不改前端逻辑。

### 数据处理产物

不要让前端直接读散落的 BFCL result/score 文件。先生成统一 case database：

```text
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/cases.jsonl
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/model_metrics.json
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/pairwise_diffs.jsonl
```

每条 case：

```json
{
  "benchmark": "BFCL",
  "category": "live_parallel",
  "case_id": "live_parallel_0-0-0",
  "language": "zh",
  "function_names": ["get_current_weather"],
  "num_reference_calls": 2,
  "prompt": "...",
  "reference": [...],
  "models": {
    "best-ever": {
      "valid": false,
      "error_type": "parallel_function_checker_no_order:cannot_find_match",
      "raw_output": "...",
      "decoded_calls": [...]
    },
    "B": {
      "valid": false,
      "error_type": "parallel_function_checker_no_order:cannot_find_match",
      "raw_output": "...",
      "decoded_calls": [...]
    }
  },
  "failure_tags": ["canonicalization", "default_value"]
}
```

### 页面功能

建议做一个静态本地网页 + 小型 Python HTTP server，优先可维护：

```text
scripts/analysis/build_eval_case_browser.py
scripts/analysis/serve_eval_case_browser.py
docs/evaluation/eval_case_browser/
```

前端页面模块：

1. **Overview**
   - 模型总分表：Tool / Tool live / Memory F1 / Code Acc / Code BoN。
   - Tool 子类柱状图：parallel / parallel_multiple / live_parallel / live_parallel_multiple。

2. **Pairwise Diff**
   - best vs B：四象限计数。
   - filter：category、language、function name、error_type、failure_tag、call count。
   - 一键筛选 `best_correct && B_wrong`。

3. **Case Browser**
   - 左侧 case 列表；
   - 中间 prompt + function schema + reference；
   - 右侧两个模型 raw output / decoded calls / error。
   - 高亮参数差异：missing call、wrong function、wrong enum、wrong default、wrong canonical value。

4. **Calibration Builder View**
   - 将候选 case family 标成：
     - `calib_candidate`
     - `retention_candidate`
     - `do_not_train_eval_leakage`
   - 输出一个 `candidate_manifest.jsonl`，后续由单独脚本生成非泄漏 calibration。

5. **Trend / Registry**
   - 后续加入更多模型后，能按 case 看哪些模型做对；
   - 统计每个 failure tag 的 success rate；
   - 找“上限”：有多少题 best-ever 也错、是否需要更强 tool expert 或数据生成。

## 前端工程原则

- 用静态 HTML/CSS/JS 即可，避免重型框架。
- 数据由 Python 脚本离线汇总成 JSON，前端只读一个目录。
- 页面不写训练逻辑，不改评测原始文件。
- 所有衍生数据放到 `/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/`，代码与文档留在 repo。
- 原始 result/score 路径必须保留在 case record 中，保证可追溯。

## 任务计划

### Phase 0：修正实验状态记录

- H 实验继续跑，当前作为正式监控对象，不停止。
- H 当前设置是 `init=1/3` 的 `global-coefficient + GRPO + OPD + retention`，不是 init=1 ablation。
- 后续如果需要 init=1 的 H-like 实验，另起新 run 名，不覆盖当前 H。

### Phase 1：BFCL case-level 数据抽取

1. 解析 best-ever 的 BFCL score/result：
   - live_parallel
   - live_parallel_multiple
   - parallel
   - parallel_multiple
2. 解析 B 的 BFCL score/result。
3. join 官方 BFCL prompt/schema 数据：
   - `/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/*.json`
4. 生成统一 `cases.jsonl` 与 `pairwise_diffs.jsonl`。
5. 验证总数与 summary 对齐：
   - live_parallel 16
   - live_parallel_multiple 24
   - parallel 200
   - parallel_multiple 200

### Phase 2：失败类型自动打标

第一版规则打标即可：

| tag | 判断规则 |
|---|---|
| `ast_decode_failed` | score error_type 包含 `ast_decoder` |
| `wrong_count` | error_type 包含 `wrong_count` |
| `wrong_function` | decoded function name 与 reference 不匹配 |
| `canonicalization` | value_error 且输出值为非英文/别名，reference 为 canonical enum/string |
| `default_value` | 参数非 required，reference 接受空/default，模型填了不被接受的值 |
| `enum_exactness` | value_error 且参数有 enum |
| `missing_call` | reference call 数多于 decoded call 数 |
| `extra_call` | decoded call 数多于 reference call 数 |
| `parallel_alignment` | no-order checker 无法匹配多个 calls |

这些 tag 会直接指导 calibration 应该补什么能力。

### Phase 3：可视化页面

实现：

```text
scripts/analysis/build_eval_case_browser.py
scripts/analysis/serve_eval_case_browser.py
docs/evaluation/eval_case_browser/index.html
docs/evaluation/eval_case_browser/README.md
```

页面必须支持：

- 添加模型 registry；
- best vs B pairwise diff；
- case-level prompt/schema/output/error 展示；
- failure tag 聚合；
- live vs non-live 分布；
- candidate calibration 标注导出。

### Phase 4：Calibration 设计报告与候选集

基于 Phase 1/2 的真实差异，输出：

```text
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/bfcl_live_calibration_candidates.jsonl
docs/report/bfcl_live_calibration_candidates_20260517.md
```

候选集只记录“应该生成什么类型的 calibration”，不直接复制官方 eval prompt。

### Phase 5：构造 leakage-safe BFCL-live-style calibration

构造方式：

1. 从 failure tags 抽象模板；
2. 替换实体、语言、函数 schema、enum 值；
3. 保留 BFCL live 的难点结构；
4. 用 tool expert / best-ever / stronger model rollout 生成 positive；
5. 用 RewardRouter 或 BFCL scorer 做验证；
6. 只保留：
   - 当前 policy 错；
   - expert/best 能对；
   - 与 official eval exact prompt 不重复；
   - metadata 完整。

### Phase 6：训练实验

下一轮训练不应只说“加 tool 数据”，而应做受控 ablation：

| 实验 | tool calibration | 目的 |
|---|---|---|
| T0 | old ToolRL-only | 旧 baseline |
| T1 | ToolRL + BFCL non-live | 看 parallel 多调用是否提升 |
| T2 | ToolRL + BFCL live-style | 看 live canonical/default 是否提升 |
| T3 | ToolRL + non-live + live-style | 全量 tool calibration |

保持 memory/code 数据不变，避免混淆。

## 成功判据

1. 前端能清楚显示 best-ever 与 B 在 BFCL live 上具体差在哪些题。
2. 能统计 failure tag，并将 live 低分归因到可训练能力。
3. 生成的 calibration 候选不是泄漏 official eval，而是同分布能力补丁。
4. 下一轮训练后至少观察：
   - Tool live mean 提升；
   - Tool non-live 不明显下降；
   - Memory F1 不回退；
   - Code Acc/BoN 不明显回退。

## 立即行动顺序

1. 写 `build_eval_case_browser.py`，先支持 best-ever 与 B 两个模型。
2. 生成 `cases.jsonl` / `pairwise_diffs.jsonl`。
3. 做静态前端，先能筛选 `best_correct && B_wrong`。
4. 人工检查 top failure tags，确认 calibration 生成方向。
5. 再构造 BFCL-live-style calibration，不先动训练。

## 2026-05-17 实现状态

已完成第一版 case-level browser，当前只接入两个模型：

- `best-ever-tame-cg-r1calib-global-v2`
- `expertgym-B-codeaug-opd-i18`

生成命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/build_eval_case_browser.py \
  --registry docs/evaluation/eval_case_browser/models.json \
  --output-dir /tmp/shared-storage/OnPolicy/analysis/eval_case_browser
```

启动命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/serve_eval_case_browser.py \
  --host 127.0.0.1 \
  --port 8791 \
  --site-dir docs/evaluation/eval_case_browser \
  --data-dir /tmp/shared-storage/OnPolicy/analysis/eval_case_browser
```

当前数据位置：

```text
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/
├── app_data.json
├── bfcl_live_calibration_candidates.jsonl
├── cases.jsonl
├── model_metrics.json
└── pairwise_diffs.jsonl
```

已验证统计：

| 指标 | best-ever | ExpertGym B |
|---|---:|---:|
| Tool mean | 0.7954 | 0.7823 |
| Tool live mean | 0.7083 | 0.6771 |
| BFCL parallel | 0.9050 | 0.9050 |
| BFCL parallel_multiple | 0.8600 | 0.8700 |
| BFCL live_parallel | 0.7500 | 0.6875 |
| BFCL live_parallel_multiple | 0.6667 | 0.6667 |

pairwise case 统计：

| 状态 | 数量 |
|---|---:|
| both_correct | 374 |
| both_wrong | 51 |
| best-ever correct / B wrong | 7 |
| best-ever wrong / B correct | 8 |

当前自动 tag 的主要失败类型：

| tag | count |
|---|---:|
| parallel_alignment | 50 |
| wrong_function | 42 |
| enum_exactness | 40 |
| parameter_value_error | 40 |
| canonicalization | 15 |
| wrong_count | 15 |
| missing_call | 10 |
| default_value | 6 |

calibration candidate 已按 B 模型失败样本导出，共 `58` 条：

| 维度 | 分布 |
|---|---|
| priority | high 1 / medium 18 / low 39 |
| category | live_parallel 5 / live_parallel_multiple 8 / parallel 19 / parallel_multiple 26 |
| status | best-ever correct/B wrong 7 / both wrong 51 |

candidate 文件只保存 source id、category、failure tags、函数数量、目标能力和 synthetic requirements，不保存官方 prompt、ground truth 或模型输出。后续构造训练数据时必须按 candidate 中的 `leakage_policy` 生成新 schema、新实体和新答案。

前端已加入 `Calibration Candidate Queue` 区域，可按 category / status / tag / language / search 联动过滤候选项；点击 candidate 会跳到对应 case detail 便于人工确认失败原因。

直接访问：

```text
http://127.0.0.1:8791
```
