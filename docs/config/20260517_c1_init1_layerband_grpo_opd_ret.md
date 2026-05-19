# 2026-05-17 c1：init1 + per-layer layer-band + GRPO/OPD/Retention

## 目的

c1 用来检查：从 `init=1.0` 的强 task-vector 初始点出发，把 gate 从全局参数换成 per-layer layer-band 后，是否能在 `GRPO + dynamic OPD + NLL retention` 的共同约束下，找到比全局系数更细的层级组合。

注意：默认 `layer-band` 只有 `early/mid/late` 三段。c1 不使用默认配置，而是使用 `configs/gated_grpo_layer28.yaml`，每一层一个 band。

## 核心设置

| 项 | 值 |
|---|---|
| launcher | `skill/command/run_20260517_c1_init1_layerband_grpo_opd_ret.sh` |
| run dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/c1_init1_layerband_grpo_opd_ret_20260517` |
| config | `configs/gated_grpo_layer28.yaml` |
| gate strategy | `layer-band` |
| layer bands | `28` 个，`layer0` 到 `layer27` |
| effective trainable coefficients | 每层 tool / memory / code 三个有效专家系数；实现内部为 per-layer common + zero-mean residual |
| init | 所有有效 expert coefficient 从 `1.0` 开始 |
| prompts | `qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| prompt count | `96 = tool 32 + memory 32 + code 32` |
| samples per prompt | `4` |
| iterations | `20` |
| optimizer | SGD, momentum `0.2`, persisted state |
| lr | `0.1876` |
| optimizer step | epoch-scope |
| loss granularity | sequence |
| GRPO | `PPO_LOSS_WEIGHT=1.0` |
| OPD | dynamic all-fail expert-positive NLL, `OPD_LOSS_WEIGHT=1.0` |
| retention | all-success NLL, task-balanced, `RETENTION_LOSS_WEIGHT=0.5` |
| prior | `0.0` |
| max coeff delta from init | `1.0` |
| GPUs | default `2,3` |
| monitor | `http://127.0.0.1:8795` |

## 数据

Calibration:

```text
/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
```

Dynamic OPD expert rollouts：沿用 old-B 的三专家 paper96 rollout，不混入 code augmentation，保持与 b1 可对照。

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
```

Init gate:

```text
/tmp/shared-storage/OnPolicy/data/init_gates/c1_20260517/init_layer_band_layer28_init1.json
```

## 启动

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

DRY_RUN=1 GPU_LIST=2,3 bash skill/command/run_20260517_c1_init1_layerband_grpo_opd_ret.sh

tmux new -d -s train_c1_init1_layerband_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && OVERWRITE=1 GPU_LIST=2,3 bash skill/command/run_20260517_c1_init1_layerband_grpo_opd_ret.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/c1_init1_layerband_grpo_opd_ret_20260517/train.log'
```

## 监控

```bash
tmux new -d -s opvec_monitor_c1_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/monitor/opvec_run_monitor.py --run-dir c1_init1_layerband=/tmp/shared-storage/OnPolicy/runs/gated_grpo/c1_init1_layerband_grpo_opd_ret_20260517 --init-value 1.0 --host 127.0.0.1 --port 8795 --quiet 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/c1_init1_layerband_grpo_opd_ret_20260517/monitor_8795.log'
```

## 重点观察

| 指标 | 判断 |
|---|---|
| per-layer gate dispersion | 是否只有少数层偏移，还是全层一起漂移 |
| task reward | Tool/Memory/Code 是否同时维持，尤其 code 是否比 global 设置更易推动 |
| OPD rows | all-fail expert-positive 是否在三任务间仍均衡 |
| retention rows | init1 是否导致大量 all-success，retention 是否阻尼过强 |
| grad norm / gate delta | 与 b1、H 对照判断 layer-band 是否因为参数变多而单参数梯度变弱 |
