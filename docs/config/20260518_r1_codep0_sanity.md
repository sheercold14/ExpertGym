# 2026-05-18 R1 Code P0 Sanity

## 目的

在新的 Code P0 v3 bank 上做短程 sanity，验证 DeepSeek-R1-Distill-Qwen-7B 作为小幅 reasoning/code prior 时，Code reward 是否能被 executable feedback 推动。

该实验不是主结果，只回答：

```text
Code P0 bank + ReasonFlux/DeepSeek positives + R1 scaled layer gate
是否能让 train code reward / gate 方向出现正信号？
```

## 配置

```text
run_name: r1_codep0_layer28_z001_codeonly_sanity_20260518
run_dir: /tmp/shared-storage/OnPolicy/runs/gated_grpo/r1_codep0_layer28_z001_codeonly_sanity_20260518
config: configs/gated_grpo_reasoning_layer28.yaml
mode: /tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json
calibration: /tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/train_code64.prompts.jsonl
task filter: code
strategy: layer-band
init: tool=memory=code=1.0, reasoning=0.001
reasoning trust region: max delta from init = 0.002
```

注意：本次已完成的 sanity 使用的是逐轮 trust region。后续正式 R1 实验必须增加全程绝对 effective coefficient bound：

```text
COEFF_BOUND_BY_EXPERT=reasoning=0.0:0.003  # safe
COEFF_BOUND_BY_EXPERT=reasoning=0.0:0.01   # stress
```

Loss:

```text
GRPO weight = 1.0
dynamic OPD weight = 1.0
retention NLL weight = 0.05
advantage normalization = zscore
optimizer = SGD, lr=0.05, momentum=0.2
step scope = epoch
loss granularity = sequence
```

Rollout:

```text
num_iters = 4
num_prompts = 64
samples_per_prompt = 4
code_max_new_tokens = 10000
max_model_len = 24576
rollout_gpus = 2,3
rollout_shards = auto
```

Expert positives:

```text
/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_code_p0_v3_train64_s8_seed20260518.merged.jsonl
```

## 启动

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
GPU_LIST=2,3 ROLLOUT_GPUS=2,3 \
  bash skill/command/run_20260518_r1_codep0_sanity.sh
```

tmux:

```bash
tmux new -d -s train_r1_codep0_sanity_20260518 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && \
   GPU_LIST=2,3 ROLLOUT_GPUS=2,3 \
   bash skill/command/run_20260518_r1_codep0_sanity.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/r1_codep0_layer28_z001_codeonly_sanity_20260518/train.log'
```

## 判据

继续条件：

- train code reward 上升；
- frontier / recoverable OPD rows 非空；
- code effective gate 上升或 R1 layer gate 在 `[0, 0.003]` 内出现有结构的分层变化；
- 不出现 gate clipping 全部顶满。

停止条件：

- Code reward 不上升且 OPD rows 快速耗尽；
- reasoning gate 被 trust region 持续夹住但 reward 不涨；
- loss/grad norm 明显接近 0。
