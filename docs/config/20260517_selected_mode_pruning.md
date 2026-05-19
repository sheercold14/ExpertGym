# 2026-05-17 Selected-Mode Pruning Smoke

## 目标

验证两类 init=1 结构化 mode selection 是否值得进入正式训练：

1. **expert-specific structured pruning**：tool/memory/code 各自只保留对应 expert 的 selected modes，其余同 expert delta 系数置 0。
2. **reasoning micro-addition**：保持 `tool=memory=code=1` 全量能力，再额外加入 reasoning expert 的 TAME selected 64 modes，reasoning 系数 `0.001`。
3. **tool/memory top64 structured pruning**：只裁剪 Tool/Memory，各自保留 64 个 mode，Code 保持全量 `1.0`，用于排查“保留 mode 太少”是否是 hard pruning 失败主因。

结论先行：当前 smoke 不支持直接用 hard pruning 替代 init=1；Tool 即使保留 64 个 mode 仍然 zero-call，说明 Tool 的 tool-call 格式能力不是简单 top-k hard mask 能保住的。Memory 从 30-entry pruning 的 `0.1667` 提升到 top64 的 `0.3333`，但仍低于 all1 baseline 的 `0.6667`。

## 产物

生成脚本：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/analysis/build_selected_mode_gate_checkpoints.py
```

checkpoint 目录：

```text
/tmp/shared-storage/OnPolicy/data/init_gates/selected_mode_pruning_20260517
```

生成命令：

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python

$PY scripts/analysis/build_selected_mode_gate_checkpoints.py \
  --output-dir /tmp/shared-storage/OnPolicy/data/init_gates/selected_mode_pruning_20260517 \
  --expert-selected-modes tool=/tmp/shared-storage/TAME/experiments/tame-tool-alllayers-gradient-20260504/sparse_solution_bfcl_live40_alllayers_top16_z1/selected_modes.json \
  --expert-selected-modes memory=/tmp/shared-storage/TAME/experiments/tame-calibration-global-dryrun-20260504b/selected_modes.json \
  --expert-selected-modes code=/tmp/shared-storage/TAME/experiments/tame-calibration-global-dryrun-20260504b/selected_modes.json \
  --reasoning-selected-modes /tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/calibration_global_r1_v2_solution/selected_modes.json
```

## Mode 来源

| expert | selected modes | 来源 | 备注 |
|---|---:|---|---|
| tool | 16 | `/tmp/shared-storage/TAME/experiments/tame-tool-alllayers-gradient-20260504/sparse_solution_bfcl_live40_alllayers_top16_z1/selected_modes.json` | BFCL all-layer tool-specific selection |
| memory | 7 | `/tmp/shared-storage/TAME/experiments/tame-calibration-global-dryrun-20260504b/selected_modes.json` | 只有早期 global dryrun 的 memory selection，可用性弱 |
| code | 7 | `/tmp/shared-storage/TAME/experiments/tame-calibration-global-dryrun-20260504b/selected_modes.json` | 只有早期 global dryrun 的 code selection，可用性弱 |
| reasoning | 64 | `/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/calibration_global_r1_v2_solution/selected_modes.json` | 只适用于 DeepSeek-R1-Distill-Qwen-7B reasoning expert |

关键修正：reasoning 的 64 modes 不应用来裁剪 tool/memory/code。它只描述 reasoning expert 的可用结构。

### Tool/Memory top64 来源

这版只裁剪 Tool/Memory，Code 保持全 1：

| expert | top64 构造 | 来源 | 备注 |
|---|---:|---|---|
| tool | 56 ranked + 8 delta-L2 supplement | `/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/free_space/tool_boundary_repair/boundary_sensitivity_last8_attn_mlp_v1/boundary_sensitivity_summary.json` | 使用 boundary sensitivity 的 ranked rows；不足 64 时用 OP-VEC diagnostics 里该 expert delta L2 最大的 param 补齐 |
| memory | 7 selected + 57 delta-L2 supplement | `/tmp/shared-storage/TAME/experiments/tame-calibration-global-dryrun-20260504b/selected_modes.json` | 早期 selected modes 太少，补齐部分主要是范数选择，不是 task reward selection |
| code | 196 full | 无裁剪 | 排除 Code 被裁剪带来的混淆 |

## Checkpoint 语义

### expert-specific pruning

```text
/tmp/shared-storage/OnPolicy/data/init_gates/selected_mode_pruning_20260517/expert_selected_prune_init1.parameter.json
```

参数化：`parameter`，共 `588 = 196 * 3` 个系数。

规则：

- 对每个 expert 自己的 selected params，系数 `1.0`。
- 对该 expert 未 selected params，系数 `0.0`。
- 实际 bake delta entries：`30 = 16(tool) + 7(memory) + 7(code)`。

不用 `global-parameter` 的原因：它有 residual clamp，不能表达真正的 0/1 hard pruning。

### init1 + reasoning64@0.001

```text
/tmp/shared-storage/OnPolicy/data/init_gates/selected_mode_pruning_20260517/init1_plus_reasoning64_z0001.parameter.json
```

参数化：`parameter`，共 `784 = 196 * 4` 个系数。

规则：

- tool/memory/code 全部 param 系数 `1.0`。
- reasoning selected 64 个 param 系数 `0.001`。
- reasoning 非 selected param 系数 `0.0`。
- 实际 bake delta entries：`652 = 588 + 64`。

### tool/memory top64 pruning + code full

```text
/tmp/shared-storage/OnPolicy/data/init_gates/selected_mode_pruning_20260517/tool_memory_top64_prune_codefull_init1.parameter.json
```

参数化：`parameter`，共 `588 = 196 * 3` 个系数。

规则：

- Tool：64 个 selected/supplement param 系数 `1.0`，其余 132 个系数 `0.0`。
- Memory：64 个 selected/supplement param 系数 `1.0`，其余 132 个系数 `0.0`。
- Code：未提供 source，因此全部 196 个系数保持 `1.0`。
- 实际 bake delta entries：`324 = 64(tool) + 64(memory) + 196(code)`。

生成命令：

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python

$PY scripts/analysis/build_selected_mode_gate_checkpoints.py \
  --output-dir /tmp/shared-storage/OnPolicy/data/init_gates/selected_mode_pruning_20260517 \
  --expert-prune-name tool_memory_top64_prune_codefull_init1.parameter.json \
  --top-k-per-expert 64 \
  --expert-selected-modes tool=/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/free_space/tool_boundary_repair/boundary_sensitivity_last8_attn_mlp_v1/boundary_sensitivity_summary.json \
  --expert-selected-modes memory=/tmp/shared-storage/TAME/experiments/tame-calibration-global-dryrun-20260504b/selected_modes.json
```

## Smoke 设置

对照基线：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/smoke_all1_gc_20260517
```

本轮 smoke：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/smoke_expert_selected_prune_init1_20260517
/tmp/shared-storage/OnPolicy/runs/gated_grpo/smoke_init1_plus_reasoning64_z0001_20260517
/tmp/shared-storage/OnPolicy/runs/gated_grpo/smoke_tool_memory_top64_prune_codefull_init1_20260517
```

共同设置：

- prompt：`qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl`
- `num_prompts=9`，按 manifest order，tool/memory/code 各 3 条
- `samples_per_prompt=2`
- vLLM 单卡，`max_model_len=12288`
- `tool_max_new_tokens=512`
- `code_max_new_tokens=1536`
- `memory_update_max_new_tokens=1024`
- `memory_final_max_new_tokens=1024`

## Smoke 结果

| checkpoint | Tool mean | Tool parse | Tool zero-call | Memory mean | Code mean | 结论 |
|---|---:|---:|---:|---:|---:|---|
| all1 baseline | 1.0000 | 1.0000 | 0.0000 | 0.6667 | 0.5000 | init=1 小样本稳定 |
| expert-selected pruning | 0.0000 | 0.0000 | 1.0000 | 0.1667 | 1.0000 | hard pruning 让 Tool 断崖，不能直接用 |
| init1 + reasoning64@0.001 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 | Tool 无害，Memory/Code 小样本无收益 |
| tool/memory top64 + code full | 0.0000 | 0.0000 | 1.0000 | 0.3333 | 0.5000 | Memory 比 30-entry pruning 好，但 Tool 仍 zero-call |

## 判断

- `expert-selected pruning` 过于稀疏，尤其 tool selected 16 个 mode 不足以保留 BFCL 输出格式能力；tool zero-call 到 1.0 是硬失败信号。
- `tool/memory top64 + code full` 说明“裁得太少”确实影响 Memory，但不是 Tool 失败的唯一原因；Tool 从 16 扩到 64 后仍然完全不输出 tool call，说明该 selection/ranking 没有捕获 BFCL live/tool-call generation 所需的关键分布式结构。
- memory/code 当前 selected modes 来源只是早期 global dryrun，不是可靠的 task-specific pruning 解；不能据此否定“专家内 mode selection”，只能否定这版 hard pruning。
- `init1 + reasoning64@0.001` 没有破坏 Tool，但 Memory 3/3 prompt 全错，说明 reasoning 微扰仍可能影响长轨迹生成；后续若要正式试，需要至少用 96-prompt proxy，而不是只看 9-prompt smoke。

## 下一步建议

1. 不把 hard pruning 作为主训练起点；主线继续用 `tool=memory=code=1`。
2. 若要研究结构选择，优先做 **soft mask / low floor**，例如 non-selected coefficient 保留 `0.25` 或 `0.5`，不要直接置 0。
3. Tool 需要用 BFCL live/tool-call calibration 重新做 mode selection，目标不是只提升 reward，而是保留 `<tool_call>` 格式、函数名选择和参数生成的可执行结构。
4. 对 memory/code 重新构造 task-specific mode selection，而不是复用早期 global dryrun。
5. reasoning expert 可以作为“小扰动能力补丁”单独做 96-prompt proxy 和 eval6，不应混入三专家 pruning 结论。
