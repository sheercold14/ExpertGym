# 2026-05-21 Structured Capability Gates v1

## 目的

在已有 attention / MLP 机制分析基础上，设计一个简单、第一性原理驱动的第二阶段方法，目标是在 Tool / Memory / Code 三任务上寻找更合理的能力表达，而不是继续做无解释 coefficient sweep。

核心思想：

> 从 `init=1` 的强能力 prior 出发，保留已证明有用的能力通道，只抑制机制诊断中明确有 harm 或冲突的 residual 子空间。

## 机制依据

| 任务 | 结构证据 | 方法决策 |
| --- | --- | --- |
| Memory | MLP raw exposure 最大；memory-attn-calm 只压 Memory attention 后 F1 从 `0.7761` 掉到 `0.7479` | Memory attention + MLP 全保留为 `1.0` |
| Tool | ToolRL all80 基本稳定；BFCL 错误主要是 schema/call-count，不是不会 tool call | Tool 全保留为 `1.0`，先不做无证据压缩 |
| Code | pass/fail contrast 在 layer `8-20` 有弱正方向；layer `27` 强反向；`mlp_down/q/k` 相对弱 | 只开放 Code 中层正向 family，抑制 conflict layers |

## 方法：Structured Capability Gates

可训练/可搜索对象不是全局 3 个系数，也不是 588 个完全自由参数，而是一个极小的结构化候选族。

固定规则：

- `memory:* = 1.0`
- `tool:* = 1.0`
- `code` 按 layer / family 分组：
  - 中层 positive family：layers `8-20` 的 `mlp_gate / mlp_up / attn_o / attn_v`
  - 中层 weak family：layers `8-20` 的 `mlp_down / attn_q / attn_k`
  - conflict layers：`24,27`
  - 其他 Code residual：background prior

候选只在不确定的 Code 强度上分三档：

| candidate | code background | code mid positive | code mid weak | code conflict | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| `balanced` | 0.75 | 1.00 | 0.85 | 0.50 | 默认结构 prior |
| `code_mid_push` | 0.75 | 1.15 | 0.90 | 0.35 | 尝试增强中层 Code 正向方向 |
| `code_safe` | 0.70 | 0.95 | 0.75 | 0.25 | 更保护 Tool/Memory，保留少量 Code 表达 |

这不是 sweep：三个候选分别对应机制假设的三种强度，不改变 Memory/Tool 的能力通道。

## 当前产物

生成命令已运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/attention_pauh/build_structured_capability_gates.py \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --output-dir /tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521
```

产物：

| artifact | path |
| --- | --- |
| candidate manifest | `/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521/candidate_manifest.json` |
| README | `/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521/README.md` |
| balanced gate | `/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521/balanced/gates.json` |
| code_mid_push gate | `/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521/code_mid_push/gates.json` |
| code_safe gate | `/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521/code_safe/gates.json` |

当前 coefficient summary：

| candidate | memory mean | tool mean | code mean | code min | code max |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced | 1.0000 | 1.0000 | 0.8184 | 0.5000 | 0.9357 |
| code_mid_push | 1.0000 | 1.0000 | 0.8574 | 0.3500 | 1.0429 |
| code_safe | 1.0000 | 1.0000 | 0.7441 | 0.2500 | 0.8643 |

## 复现脚本

统一入口：

```bash
bash skill/command/run_20260521_structured_capability_gates_v1.sh
```

分阶段：

```bash
# 只生成 gate
PHASE=generate bash skill/command/run_20260521_structured_capability_gates_v1.sh

# bake 所有 candidate
PHASE=bake bash skill/command/run_20260521_structured_capability_gates_v1.sh

# Tool + Memory quick eval
PHASE=quick_eval GPU_LIST=1 bash skill/command/run_20260521_structured_capability_gates_v1.sh

# 单独评测一个 candidate，避免 BFCL harness 全局配置并发冲突
PHASE=quick_eval CANDIDATES=code_safe GPU_LIST=3 TOOL_PORT=8103 \
  bash skill/command/run_20260521_structured_capability_gates_v1.sh
```

默认路径：

```text
ROOT=/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521
CHECKPOINT_ROOT=/tmp/shared-storage/OnPolicy/checkpoints/structured_capability_gates_v1_20260521
EVAL_ROOT=/tmp/shared-storage/ExpertGym/structured_capability_gates/eval/scg_v1_20260521
```

## 评测与选择原则

第一阶段 quick gate：

1. 先跑 Tool + Memory。
2. Tool 用 BFCL quick mean；Memory 用 HotpotQA `eval_50/eval_100` mean F1。
3. 只有满足：
   - Tool quick mean 不明显低于 SPRE-v2 / PAUH layer-all；
   - Memory F1 接近 `0.76+`；
   才送 Code official。

第二阶段 Code：

- 不看 Code gate 是否高；
- 看 CURE / LiveBench / LiveCodeBench official acc；
- 如果 `code_mid_push` 过不了 Tool/Memory，优先 `balanced`；
- 如果三者 Code 都不提升，说明当前 Code residual 子空间仍不足，应回到 pass/fail contrast 轨迹筛选，而不是继续加大 Code gate。

## 2026-05-21 Quick 结果

Tool 使用 BFCL quick 四类平均：`parallel / parallel_multiple / live_parallel / live_parallel_multiple`。
Memory 使用 HotpotQA `eval_50 / eval_100` 的 `avg_f1`。

| candidate | Tool quick mean | parallel | parallel_multiple | live_parallel | live_parallel_multiple | Memory eval_50 F1 | Memory eval_100 F1 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced` | 0.7656 | 0.8800 | 0.8700 | 0.6875 | 0.6250 | 0.7611 | 0.7271 | Memory 可用，Tool live 偏低 |
| `code_mid_push` | 0.7644 | 0.8800 | 0.8650 | 0.6875 | 0.6250 | 0.7492 | 0.7431 | Code 中层更强推没有带来 Tool/Memory 收益 |
| `code_safe` | 0.7813 | 0.8850 | 0.8650 | 0.7500 | 0.6250 | 0.7714 | 0.7427 | 当前最优折中，推荐进入 Code 快评 |

关键观察：

- `code_mid_push` 的 Code mean 最高，但 Tool/Memory 没有变好，说明简单增强 Code residual 不是稳定路径。
- `code_safe` 降低 Code background、weak family 和 conflict layers 后，Tool live_parallel 从 `0.6875` 提升到 `0.7500`，Memory eval_50 也最高。
- 当前证据支持“保护 RL agent 行为 span + 只在低冲突 Code 子空间补能力”的第二阶段原则。

下一步：

1. 将 `code_safe` 送入 Code quick / official，确认 Code 是否被保留或改善。
2. 如果 Code 仍低，不继续推高 Code gate；改为增加 pass/fail contrast 和 hidden-state residual 对齐，尤其是 LiveBench / LiveCodeBench 的 reasoning span + final code span。
3. Tool 的 live_parallel_multiple 仍卡在 `0.6250`，后续要针对 call-count / multi-call composition 构造小规模诊断集，而不是增加普通 ToolRL 样本。

## 论文叙事

这版方法可以支撑的 claim：

> Mechanistic diagnostics identify which parts of each task vector express useful capability and which parts cause interference. A small structure-constrained gate family can preserve Tool and Memory while testing whether Code ability is recoverable from the identified middle-layer residual subspace.

它的价值在于：

- 有明确机制依据；
- 候选数极小；
- Memory/Tool 保护不是靠调参，而是来自 ablation；
- Code 失败也可解释为 residual 子空间不可恢复，而不是优化器没调好。
