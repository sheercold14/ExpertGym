# 2026-05-16 F 实验配置：code augmentation only OPD

## 目标

F 是新一轮 B 的最小改动 ablation：保留 B 的 `global-parameter + OPD-only + retention` 主设置，但把 dynamic OPD 的轨迹来源收窄为 **只使用 code augmentation expert rollouts**，且 `DYNAMIC_OPD_TASKS=code`。目的在于确认：之前 B 的 code 信号不稳定，是否来自旧三专家 OPD / tool-memory OPD 混入，还是 code augmentation 本身也不足以稳定提升 code reward。

## 核心差异

| 实验 | 差异 |
|---|---|
| B `expB_gp_code_opd_aug_20260516` | dynamic OPD tasks 为 `tool,memory,code`；expert pool 包含旧 tool、旧 memory、旧 code、四个 code augmentation |
| F `expF_gp_code_aug_only_code_opd_20260516` | dynamic OPD tasks 仅 `code`；expert pool 仅四个 code augmentation；不启用 GRPO |

## 公共设置

| 项 | 值 |
|---|---|
| repo | `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym` |
| launcher | `skill/command/run_20260516_expF_code_aug_opd_only.sh` |
| base train script | `skill/command/run_qbank_c033333_gate_strategy.sh` |
| train entry | `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` |
| config | `configs/gated_grpo.yaml` |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` |
| seed manifest | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| prompts | `96 = tool 32 + memory 32 + code 32` |
| samples per prompt | `4` |
| gate strategy | `global-parameter` |
| init | `tool=memory=code=1/3` |
| optimizer | `sgd`, momentum `0.2`, epoch-scope step, persisted state |
| lr | `0.1876` |
| loss granularity | `sequence` |
| GRPO | disabled, `PPO_LOSS_WEIGHT=0.0` |
| frontier quota | `tool=0, memory=0, code=0` |
| OPD | enabled, dynamic all-fail current rows + offline code augmentation positives |
| OPD task | `code` only |
| OPD positive threshold | `1.0` |
| OPD per task | `32` |
| OPD max positive/negative per row | `1 / 2` |
| OPD length norm | enabled |
| OPD task-balanced row scale | enabled |
| retention | enabled, `nll`, positive threshold `1.0`, task-balanced row scale |
| policy logprob length norm | enabled |
| task normalize advantages | disabled |
| prior loss | `0.0` |
| max coefficient delta from init | `1.0` |

## Code OPD Positive Pool

只读以下四个 code augmentation rollout，不混入旧 code expert、不混入 tool/memory expert：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl
```

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_expF_20260516 \
  'GPU_LIST=2,3 bash skill/command/run_20260516_expF_code_aug_opd_only.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_gp_code_aug_only_code_opd_20260516/train.log'
```

Dry run：

```bash
DRY_RUN=1 GPU_LIST=2,3 bash skill/command/run_20260516_expF_code_aug_opd_only.sh
```

## 监控与判据

重点看：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_gp_code_aug_only_code_opd_20260516/iter_*/rollouts.summary.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_gp_code_aug_only_code_opd_20260516/iter_*/opd_distill_from_allfail.summary.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_gp_code_aug_only_code_opd_20260516/iter_*/gate_updates.summary.json
/tmp/shared-storage/OnPolicy/runs/gated_grpo/expF_gp_code_aug_only_code_opd_20260516/iter_*/gate_updates.gates.json
```

判断逻辑：

- `opd_distill_task_counts` 应只有 `code`。
- `frontier_task_counts` 应为空或全 0。
- 如果 code gate 上涨但 code reward 仍不涨，说明 code augmentation OPD 本身不足以稳定优化 pass-rate proxy。
- 如果 code reward 明显涨而 tool/memory 不崩，说明旧 B 的干扰主要来自多任务 OPD 混合。
