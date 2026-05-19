# 2026-05-18 DeepSeek-R1 Scaled Task Vector Plan

## 结论

DeepSeek-R1-Distill-Qwen-7B 可以作为 ExpertGym 的一个关键应用点，但不能作为普通第四 expert 直接加入。它的 delta 幅值和 Tool/Memory/Code task vector 不在一个量级；合理用法是：

```text
tool/memory/code 提供 agent capability prior
DeepSeek-R1 提供小幅 reasoning/code prior
executable feedback 学习每层/每模块需要注入多少 R1 delta
```

这比单纯 “code expert + memory expert + tool expert” 更能体现论文 claim：

```text
task vectors provide structured priors;
executable feedback learns their composition.
```

## 范数审计

mode manifest:

```text
/tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json
```

diagnostics:

```text
/tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/diagnostics.json
```

总 L2 范数：

| expert | total L2 | relative to code | R1 等效到该 expert 的系数 |
|---|---:|---:|---:|
| code / ReasonFlux-Coder | 0.6203 | 1.0 | 0.000328 |
| tool / ToolRL | 1.2549 | 2.02 | 0.000664 |
| memory / RL-MemoryAgent | 5.3621 | 8.64 | 0.002839 |
| reasoning / DeepSeek-R1-Distill-Qwen | 1888.5855 | 3044.81 | 1.000000 |

含义：

- `reasoning=0.001` 的有效扰动约等于 `1.89` L2，已经大于 Tool full delta，约为 Code full delta 的 `3.0x`。
- `reasoning=0.003` 的有效扰动约等于 Memory full delta。
- `reasoning=0.01` 的有效扰动约等于 `18.9` L2，是 Memory full delta 的 `3.5x`，只能作为上界压力测试。
- 因此 R1 的合理初始/学习区间是 `3e-4 ~ 3e-3`；若放到 `0.01`，必须配强 retention/monitor 防崩。

## 过去尝试的教训

已有记录：

- 直接把 DeepSeek 作为普通第四 expert 满权混入，曾出现 near-zero 崩溃。
- 2026-05-16 Experiment C 用 `global-parameter`，初始 `tool=memory=code=1/3, reasoning=0`，20 iter 后：
  - `tool ~= 0.3248`
  - `memory ~= 0.5107`
  - `code ~= 0.3656`
  - `reasoning ~= 0.0453`
- 这个 `reasoning=0.045` 从范数看非常大，不应解释为“R1 只学了一点”；它等效扰动约 `85.5` L2，远超 memory/tool/code。

因此上一版 C 的问题不是“R1 没学到”，而是 **R1 的系数量纲不受控**。必须把 R1 gate 的学习边界压到 `0.001~0.01`，最好先用 `0.003` 作为硬上限。

## 推荐方法

### 参数化

首选：

```text
layer-band / layer28 + four experts
```

每层学习：

```text
tool_l, memory_l, code_l, reasoning_l
```

初始建议：

```text
tool = 1.0
memory = 1.0
code = 1.0
reasoning = 0.0005 或 0.001
```

边界建议：

```text
tool/memory/code coefficient: [0.6, 1.3]
reasoning coefficient: [0.0, 0.003]  # 主实验
reasoning coefficient: [0.0, 0.01]   # 压力测试
```

为什么用 per-layer：

- R1 delta 是异质 reasoning prior，不同层作用差异很大；
- 全局一个 R1 系数太粗，会同时影响格式、推理、代码长度、tool-call 风格；
- per-layer 允许 executable reward 自动选择 R1 应该进入中层 MLP、后层 attention，还是完全不进入。

### 数据与 loss

R1 这条线应该和 Code P0 bank 结合，而不是继续用旧 paper96 作为主依据。

训练信号：

| 样本状态 | loss | 作用 |
|---|---|---|
| code frontier | GRPO z-score | 判断 R1 是否提升 pass-rate |
| code all-fail + R1/ReasonFlux verified pass | OPD | 让 R1/code prior 给出恢复方向 |
| memory stable/frontier | retention + GRPO | 目标 memory gate 超过 0.55 且不崩 |
| tool stable | retention / small GRPO | 防止 R1 改坏 tool-call 格式 |

关键不是让 R1 imitation 主导，而是让 executable code reward 决定哪些层吸收 R1。

## 实验设计

### R1-A：安全小扰动

目的：验证 R1 作为 micro prior 是否能提高 Code monitor，不破坏 Tool/Memory。

```text
init: tool=memory=code=1.0, reasoning=0.0005
parameterization: layer28 four-expert
reasoning upper bound: 0.003
train: Code P0 v3 + Tool/Memory retention
loss: GRPO + OPD + retention
selection: monitor code reward + memory gate >= 0.55
```

通过标准：

- Code train reward 上升；
- Code monitor reward 上升；
- formal CURE quick eval 上升；
- memory effective coefficient/gate 能到 `>0.55` 或保持强 memory eval；
- Tool BFCL/ToolRL 不明显掉。

### R1-B：上限压力测试

目的：判断更大 R1 注入是否能显著提升 Code。

```text
init reasoning=0.001
reasoning upper bound=0.01
其他同 R1-A
```

只跑短程，若 Tool/Memory 或 monitor 崩，立即停。

### R1-C：Code-only 诊断

目的：排除 Tool/Memory 信号干扰，确认 R1 是否真的能让 Code reward 上升。

```text
tasks=code
init: code=1.0, reasoning=0.0005/0.001
tool/memory gates 固定或不 bake
samples_per_prompt=8/16
advantage=zscore
```

如果 Code-only 都不涨，说明当前 Code bank/reward 仍不够；如果 Code-only 涨但 joint 不涨，说明是跨任务 retention/gradient conflict 问题。

## 工程状态

已修复：

- `opvec/modeling/bake.py` 的 layer-band bake 现在按 manifest `expert_names` 投影，不再丢第四个 `reasoning` expert。
- 新增 bake 单测覆盖四专家 layer-band。

验证：

```text
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_bake_global_coefficients.py
Ran 2 tests in 0.002s
OK
```

待做：

1. 生成 layer28 四专家 init gate checkpoint：
   - all1 T/M/C + R1 `0.0005`
   - all1 T/M/C + R1 `0.001`
2. 给 R1 设置 expert-specific bound 或 post-step clamp，不能复用 T/M/C 的大范围 coefficient bound。
3. 用 Code P0 v3 bank 先跑 Code-only sanity。
4. 只有 Code monitor 和 CURE quick eval 同步上涨，才进入 joint 主实验。

## 论文价值

如果成立，这条线比“再加一个 code expert”更有论文味：

```text
ExpertGym can merge heterogeneous experts whose task vectors are not commensurate.
The executable feedback does not only choose between same-scale agents;
it learns scale-aware, layer-wise composition of structured priors.
```

关键实验图：

- R1 coefficient per layer 曲线；
- Code reward / CURE quick eval 对比；
- Memory gate 是否保持或超过 `0.55`；
- R1 scaled vs unscaled / no-R1 ablation。

