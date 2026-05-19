# 20260519 L6: L1 设置 + BFCL Tool16 + CURE Code16

## 目标

L6 是 L4/L5 的合并版数据入口：在 paper96 96 条基础上，同时加入正式 BFCL Tool16 和正式 CURE Code16，用来测试 Tool/Code eval-distribution anchors 是否能同时保护 tool 并推动 code。

## 数据产物

- 合并 prompt：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16/qbank_c033333_paper96_plus_bfcl_tool16_cure_code16_seed20260519.prompts.jsonl`
- 额外 expert rollout：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16/bfcl_tool16_cure_code16_extra_expert_rollouts_seed20260519.jsonl`
- summary：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16/qbank_c033333_paper96_plus_bfcl_tool16_cure_code16_seed20260519.summary.json`

计数：

| task | rows |
|---|---:|
| tool | 48 |
| memory | 32 |
| code | 48 |
| total | 128 |

额外 expert rollout：

| task | rows | positive samples | 来源 |
|---|---:|---:|---|
| tool | 16 | 16 | BFCL official possible_answer anchor |
| code | 16 | 23 | ReasonFlux / DeepSeek-R1-Distill / RL-MemoryAgent formal CURE verified outputs |

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_l6_tool_code_eval_calibration.py

PHASE=L6 GPU_LIST=6,7 NUM_ITERS=20 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

## 与 L1 保持一致的设置

- mode：`/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json`
- strategy：`layer-band-parameter`
- init：tool/memory/code = `1/3`，reasoning = `0`
- optimizer：SGD, `lr=0.25`, momentum `0.2`
- OPD：`OPD_LOSS_WEIGHT=1.0`
- GRPO：`PPO_LOSS_WEIGHT=0.0`
- retention：NLL enabled
- task weights：tool `1.0`, memory `1.0`, code `1.0`

## 注意

Tool16 的 OPD 正样本目前是 BFCL 官方答案 anchor，不是 ToolRL/R1 真实模型 rollout；Code16 的正样本是真实模型 rollout 并已用本地 reward 复验。这个差异会影响 OPD 信号解释。
