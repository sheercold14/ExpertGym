# 2026-05-18 no-R1 Direct Layer-Band Equal-Weight OPD

## 目的

对照刚才失败的 memory-push 版本，检查问题是否来自：

1. `TASK_WEIGHT_MEMORY=5.0` 把错误方向的 memory NLL 梯度放大；
2. `layer-band` 的 `common + residual` 参数化导致专家系数耦合；
3. layer-band 粒度本身过粗。

本实验只改两个点：

- task weight 改回 `tool=1, memory=1, code=1`；
- gate 改为 direct `layer-band-coefficient`，每个 band 每个 expert 一个独立系数，不使用 common/residual。

## Run

```text
run_name: expE2_noR1_lbc_equalw_opd_ret_nogrpo_20260518
run_dir: /tmp/shared-storage/OnPolicy/runs/gated_grpo/expE2_noR1_lbc_equalw_opd_ret_nogrpo_20260518
tmux: train_expE2_noR1_lbc_equalw_20260518
GPUs: 0,1
```

## 核心配置

```text
config: configs/gated_grpo.yaml
mode: /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
experts: tool, memory, code
strategy: layer-band-coefficient
bands: early / mid / late
init: every band has tool=memory=code=1/3
num_iters: 10
num_prompts: 96
samples_per_prompt: 4
calibration: /tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
```

## Loss

```text
GRPO/PPO: off, PPO_LOSS_WEIGHT=0
OPD: on, OPD_LOSS_WEIGHT=1.0
retention: on, objective=NLL, RETENTION_LOSS_WEIGHT=0.5
prior: off
task weights: tool=1.0, memory=1.0, code=1.0
OPD length norm: on
retention length norm: on
OPD task-balanced scale: on
retention task-balanced scale: on
optimizer: SGD, lr=0.1876, momentum=0.2
optimizer_step_scope: epoch
max coefficient delta from init: 1.0
```

LR 选 `0.1876`，对齐 2026-05-15 B-step012 的 global-parameter OPD-only 成功设置，而不是沿用刚才 memory-push 的 `0.4`。

## Expert Rollouts

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
```

## 判据

重点观察：

- memory gate 是否从 `1/3` 正向移动；
- 三个 band 的方向是否一致，还是只在某些层段上升；
- memory reward 是否跟随 gate 正向移动；
- tool/code 是否被 direct layer-band 过粗更新破坏。

若 iter1/2 仍明显把 memory 压低，说明失败主要不是 task weight 或 common/residual，而是当前 layer-band 粒度下 OPD NLL 局部方向与 memory coefficient 不对齐。

## Follow-up: no-R1 hierarchical layer-band

用户要求停止 direct `layer-band-coefficient` run，改用新加入的 hierarchical layer-band 参数化。旧 run 保留在原目录，不继续训练。

```text
run_name: expE3_noR1_layerband_hier_equalw_opd_ret_nogrpo_20260518
run_dir: /tmp/shared-storage/OnPolicy/runs/gated_grpo/expE3_noR1_layerband_hier_equalw_opd_ret_nogrpo_20260518
tmux: train_expE3_noR1_lbp_equalw_20260518
GPUs: 0,1
strategy: layer-band-parameter
formula: coefficient[band, expert] = global[expert] + residual[band, expert]
```

除 `STRATEGY` 外，核心设置沿用上面的 equal-weight OPD+retention noR1 对照：

```text
task weights: tool=1.0, memory=1.0, code=1.0
GRPO/PPO: off
OPD loss: 1.0
retention NLL: 0.5
LR: 0.1876
optimizer: SGD momentum=0.2
num_iters: 10
num_prompts: 96
samples_per_prompt: 4
expert rollouts: paper96 tool/memory/code expert rollouts only
```
