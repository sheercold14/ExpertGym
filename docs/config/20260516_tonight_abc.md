# 2026-05-16 晚间 A/B/C 受控实验配置

## 目标

在同一批 paper96 calibration prompts 上比较三条路径：

| 实验 | 变量 | 目的 |
|---|---|---|
| A | 在上一轮 B 的 `global-parameter` 设置上，从初始 `1/3` gate 直接加入 `GRPO + OPD`，两者 loss weight 都为 `1.0` | 验证 partial-success GRPO 是否能从第一轮开始约束 overall reward，而不是只靠 all-fail OPD |
| B | 仍用上一轮 B 的 OPD-only + retention 设置，只扩充 code expert positive pool | 验证 code 不涨是否主要来自 expert success 轨迹太稀疏 |
| C | 在 B 基础上加入 DeepSeek-R1-Distill-Qwen-7B reasoning task vector，初始 `reasoning=0` 但可学习；同时使用 B 的 code 扩充 pool | 验证 reasoning delta 是否能作为 code/reasoning 补充方向，而不是直接满权混入 |

## 固定公共设置

| 项 | 值 |
|---|---|
| repo | `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym` |
| launcher | `skill/command/run_20260516_tonight_abc.sh` |
| base train script | `skill/command/run_qbank_c033333_gate_strategy.sh` |
| train entry | `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` |
| seed manifest | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| prompts | `96 = tool 32 + memory 32 + code 32` |
| samples per prompt | `4` |
| gate strategy | `global-parameter` |
| init | A/B: `tool=memory=code=1/3`; C: `tool=memory=code=1/3, reasoning=0` |
| optimizer | `sgd`, momentum `0.2`, epoch-scope step, persisted state within run |
| lr | `0.1876` unless explicitly overridden |
| loss granularity | `sequence` |
| update batch size | `4`; optimizer step scope is `epoch`, so batch只影响loss累积切片，不在batch后立刻更新gate |
| retention | enabled, `nll`, positive threshold `1.0`, task-balanced row scale |
| OPD | dynamic all-fail current rows + offline expert positives |
| OPD positive threshold | `1.0` |
| OPD per task | `32` |
| OPD max positive/negative per row | `1 / 2` |
| OPD length norm | enabled |
| policy logprob length norm | enabled |
| task normalize advantages | disabled |
| advantage normalization | `centered` |
| prior loss | `0.0` |
| max coefficient delta from init | `1.0` |

## 数据隔离

旧 expert rollout 只读不改：

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
```

今晚新增 code expert rollout 单独放在：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/
```

计划生成：

| 文件 | 模型 | 作用 |
|---|---|---|
| `code_expert_reasonflux_coder7b_s8_seed20260516.jsonl` | `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B` | 增加 code expert BoN 覆盖率 |
| `code_expert_reasonflux_coder7b_s8_seed20260517.jsonl` | `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B` | seed20260516 覆盖不足时的额外 BoN；文件存在时自动加入 B/C |
| `code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl` | `/mnt/cache/wuruixiao/models/DeepSeek-R1-Distill-Qwen-7B` | 作为 reasoning/code 外部正轨迹 |
| `code_expert_rl_memoryagent7b_s8_seed20260516.jsonl` | `/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B` | 检查 memory expert 对 code prompt 是否能提供额外 success |

每个 rollout 生成后写同名 `.coverage.json`，记录 `covered_prompts` 与正样本数直方图。旧 `code_expert_paper96_s2` 当前覆盖率是 `17/32`，失败 prompt 数 `15`。

## C 的四专家适配

C 需要单独 mode manifest：

```text
/tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json
```

配置文件：

```text
configs/gated_grpo_reasoning.yaml
```

代码路径已做最小适配：`global-parameter` / `parameter` / `global-coefficient` gate 的 expert 列表可以从 config/manifest 读取。三专家默认配置仍保持 `tool,memory,code`。

## 启动命令

生成 code expert rollout：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s code_aug_reasonflux_20260516 \
  'GPU_LIST=6 POLICY=code bash skill/command/run_20260516_code_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/reasonflux.log'

tmux new -d -s code_aug_reasonflux2_20260516 \
  'GPU_LIST=6 POLICY=code SEED_VALUE=20260517 bash skill/command/run_20260516_code_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/reasonflux_seed20260517.log'

tmux new -d -s code_aug_deepseek_20260516 \
  'GPU_LIST=7 POLICY=deepseek bash skill/command/run_20260516_code_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/deepseek.log'

tmux new -d -s code_aug_memory_20260516 \
  'GPU_LIST=0 POLICY=memory bash skill/command/run_20260516_code_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/memory.log'
```

训练 A：

```bash
tmux new -d -s train_expA_20260516 \
  'GPU_LIST=0,1 PHASE=train_a bash skill/command/run_20260516_tonight_abc.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expA_gp_grpo_opd_from_init_20260516/train.log'
```

训练 B：

```bash
tmux new -d -s train_expB_20260516 \
  'GPU_LIST=2,3 PHASE=train_b bash skill/command/run_20260516_tonight_abc.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expB_gp_code_opd_aug_20260516/train.log'
```

训练 C：

```bash
tmux new -d -s train_expC_20260516 \
  'GPU_LIST=4,5 PHASE=train_c bash skill/command/run_20260516_tonight_abc.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expC_gp_reasoning_code_opd_aug_20260516/train.log'
```

注意：B/C 必须等新增 code expert rollout 都完成后再启动。`run_20260516_tonight_abc.sh` 会默认使用三路核心文件；如果 `code_expert_reasonflux_coder7b_s8_seed20260517.jsonl` 已存在，会自动作为额外 code expert rollout 加入。C 第一次启动会构建 reasoning mode manifest；如果需要提前构建：

```bash
PHASE=build_reasoning_modes bash skill/command/run_20260516_tonight_abc.sh
PHASE=build_c_init bash skill/command/run_20260516_tonight_abc.sh
```

## Run Dir

| 实验 | run dir |
|---|---|
| A | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expA_gp_grpo_opd_from_init_20260516` |
| B | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expB_gp_code_opd_aug_20260516` |
| C | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expC_gp_reasoning_code_opd_aug_20260516` |

## 监控

每轮必须检查：

```text
<run_dir>/iter_*/rollouts.summary.json
<run_dir>/iter_*/opd_distill_from_allfail.summary.json
<run_dir>/iter_*/gate_updates.summary.json
<run_dir>/iter_*/gate_updates.gates.json
```

核心指标：

| 指标 | 解释 |
|---|---|
| `rollouts.summary.json` task reward | 选 best checkpoint 的 proxy 主依据 |
| `opd_distill_from_allfail.summary.json` selected_task_counts | B/C 的 code 补偿是否真的生效 |
| `gate_updates.summary.json` loss_weights | 确认 A 唯一多了 `ppo=1.0`；B/C 是 `ppo=0.0` |
| `epoch_summaries[].gate_delta_max` | 判断 gate 是否被推得动 |
| `__global__::*` | 看 tool/memory/code/reasoning 的全局趋势 |
| `frontier_task_counts` 和 `retention_rows` | 判断 GRPO/retention 是否淹没 OPD |

## Best Checkpoint 选择

每组训练完成后，不默认取最后一轮。按同一规则选择：

1. 从 `iter_*/rollouts.summary.json` 读取当前 policy rollout 的 overall proxy reward。
2. 选 overall reward 最高的 iteration 对应的 baked policy。
3. 送入 `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/roadmap.md` 正式评测。
4. 汇总到 `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/20260516_tonight_abc.md`。

如果 proxy reward 最高点伴随某个任务严重崩溃，以正式评测为准；proxy 只用于候选 checkpoint 排序，不作为最终结论。
