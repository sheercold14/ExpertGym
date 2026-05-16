# 2026-05-16 晚间 D/E 受控实验配置

## 目标

在 A/B/C 正式评测同时追加两个 ablation，专门回答：`code OPD augmentation` 已经加入后，GRPO frontier 信号应该只给 memory，还是三任务全开。

| 实验 | 变量 | 目的 |
|---|---|---|
| D | 上一轮 B 的 `global-parameter + OPD + retention + code augmentation`，额外加入 GRPO，但 frontier 只保留 `memory=32` | 检查 memory-only GRPO 能否补强 memory，同时避免 tool/code frontier 干扰 |
| E | 与 D 相同，但 frontier 保留 `tool=32, memory=32, code=32` | 对照三任务全开 GRPO 是否更稳地提升 overall reward |

## 公共配置

| 项 | 值 |
|---|---|
| repo | `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym` |
| launcher | `skill/command/run_20260516_tonight_de.sh` |
| base train script | `skill/command/run_qbank_c033333_gate_strategy.sh` |
| train entry | `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` |
| config | `configs/gated_grpo.yaml` |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` |
| seed manifest | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| prompts | `96 = tool 32 + memory 32 + code 32` |
| samples per prompt | `4` |
| gate strategy | `global-parameter` |
| init | `tool=memory=code=1/3` |
| optimizer | `sgd`, momentum `0.2`, epoch-scope step, persisted state within run |
| lr | `0.1876` |
| loss granularity | `sequence` |
| update batch size | `4`; optimizer step scope is `epoch` |
| GRPO | enabled, `PPO_LOSS_WEIGHT=1.0`; D/E 只通过 frontier quota 区分任务 |
| OPD | dynamic all-fail current rows + offline expert positives |
| OPD positive threshold | `1.0` |
| OPD per task | `32` |
| OPD max positive/negative per row | `1 / 2` |
| OPD length norm | enabled |
| OPD task-balanced row scale | enabled |
| retention | enabled, `nll`, positive threshold `1.0`, task-balanced row scale |
| policy logprob length norm | enabled |
| task normalize advantages | disabled |
| advantage normalization | `centered` |
| prior loss | `0.0` |
| max coefficient delta from init | `1.0` |

## OPD Positive Pool

沿用 2026-05-16 B/C 的 code augmentation，全部只读：

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl
```

旧 code expert 覆盖 `17/32` prompts；加入四个 code augmentation 后 union 覆盖 `27/32` prompts。

## 实验矩阵

| 实验 | run dir | GPU | frontier quota | GRPO | OPD | retention | 解释 |
|---|---|---|---|---:|---:|---:|---|
| D | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expD_gp_code_aug_memory_grpo_20260516` | `0,1` | `tool=0, memory=32, code=0` | `1.0` | `1.0` | enabled | 只让 memory partial-success rows 进入 GRPO |
| E | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expE_gp_code_aug_all_grpo_20260516` | `4,5` | `tool=32, memory=32, code=32` | `1.0` | `1.0` | enabled | 三任务 frontier/GRPO 全开 |

说明：D 的 `tool/code` 仍可通过 OPD all-fail 与 retention all-success 产生梯度；只是 frontier GRPO 不使用 tool/code partial-success 样本。

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expD_20260516 \
  'GPU_LIST=0,1 PHASE=train_d bash skill/command/run_20260516_tonight_de.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expD_gp_code_aug_memory_grpo_20260516/train.log'

tmux new -d -s train_expE_20260516 \
  'GPU_LIST=4,5 PHASE=train_e bash skill/command/run_20260516_tonight_de.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expE_gp_code_aug_all_grpo_20260516/train.log'
```

Dry run：

```bash
DRY_RUN=1 PHASE=train_d bash skill/command/run_20260516_tonight_de.sh
DRY_RUN=1 PHASE=train_e bash skill/command/run_20260516_tonight_de.sh
```

## 监控指标

每轮检查：

```text
<run_dir>/iter_*/rollouts.summary.json
<run_dir>/iter_*/opd_distill_from_allfail.summary.json
<run_dir>/iter_*/gate_updates.summary.json
<run_dir>/iter_*/gate_updates.gates.json
```

重点字段：

| 字段 | 用途 |
|---|---|
| `rollouts.summary.json` task reward | 选择 best checkpoint |
| `frontier_task_counts` | D 应只有 memory；E 应三任务均有 |
| `raw_frontier_task_counts` | 确认不是数据没产生 frontier，而是 quota 控制 |
| `opd_distill_task_counts` | 确认 code augmentation 仍进入 OPD |
| `retention_rows` / `retention_task_counts` | 检查 all-success preservation 是否保留三任务能力 |
| `epoch_summaries[].grad_norm` | 与 A/B/C 比较训练动力 |
| `__global__::*` | 观察 tool/memory/code gate 演化 |

## 评测规则

训练完成后不默认取最后一轮：

1. 读取 `iter_*/rollouts.summary.json` 的 three-task overall proxy reward。
2. 每组选择 overall reward 最高 iteration 的 `baked_policy`。
3. 送入 `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/roadmap.md` 正式 eval6。
4. 结果写入 `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/` 主表。
