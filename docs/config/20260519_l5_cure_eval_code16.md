# 20260519 L5: L1 设置 + 正式 CURE Code16 Calibration

## 目标

L5 只验证一个变量：在 L1 的 R1Math 4-expert layer-band-parameter 设置不变的情况下，把正式 `LiveBench` / `LiveCodeBench` 各 8 条 hard-vs-base code 题加入 calibration，并使用与 formal CURE eval 一致的 reward 切片，观察 code reward / gate / final eval 是否改善。

## 数据

- 基础 prompt：`/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl`
- 新增 code prompt：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_livebench8_livecodebench8_seed20260519.prompts.jsonl`
- 合并 prompt：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/qbank_c033333_paper96_plus_cure_code16_seed20260519.prompts.jsonl`
- 专家正轨迹：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_expert_success_rollouts_seed20260519.jsonl`
- 数据摘要：`/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/qbank_c033333_paper96_plus_cure_code16_seed20260519.summary.json`

新增 16 条来自正式 CURE eval：

- `LiveBench`: 8 条
- `LiveCodeBench`: 8 条

默认选取策略：base Qwen2.5-7B-Instruct 在已有 formal eval temp result 中没有全通过样本，至少一个 code-capable expert 有全通过轨迹。专家池：

- `ReasonFlux-Coder-7B`
- `DeepSeek-R1-Distill-Qwen-7B`
- `RL-MemoryAgent-7B`

## Reward 对齐

- prompt 使用 CURE formal eval 的原始 prompt 模板：`CURE/evaluation/evaluation_config.py::system_prompts`。
- seed record 的 `reference.metadata.test_input/test_output/test_time_limit` 直接来自 `CURE/data/LiveBench.json` 和 `CURE/data/LiveCodeBench.json`。
- 本项目 `CodeRewardAdapter` 会读取上述 metadata，执行前 8 个官方测试并返回 pass-rate。
- CURE formal eval config 中 `max_test=8`，因此训练 reward 的测试切片与 formal eval 一致。
- expert positive 写入 OPD 文件前，会重新调用本地 `CodeRewardAdapter` 验证 `reward_train >= 1.0`。

## 训练设置

保持 L1 主设置不变：

- config: `configs/gated_grpo_4expert_r1math_layer28.yaml`
- mode: `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_scaled_20260519/mode_manifest.json`
- gate strategy: `layer-band-parameter`
- init gate: tool/memory/code = `1/3`, reasoning = `0`
- trainable coefficients: all experts trainable
- optimizer: `sgd`, `lr=0.25`, `momentum=0.2`
- OPD: `OPD_LOSS_WEIGHT=1.0`
- GRPO: `PPO_LOSS_WEIGHT=0.0`
- retention: enabled, `RETENTION_OBJECTIVE=nll`, target scale `0.5`
- task weights: tool `0.5`, memory `2.0`, code `1.5`
- prompts: `NUM_PROMPTS=112`
- samples per prompt: `4`
- iterations: `20`

只相对 L1 改动：

1. `CALIBRATION` 换成 paper96 + CURE Code16 的合并 prompt。
2. `EXTRA_DYNAMIC_OPD_EXPERT_ROLLOUT` 追加 CURE Code16 expert success 文件。

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_cure_eval_code16_calibration.py

PHASE=L5 GPU_LIST=4,5 NUM_ITERS=20 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

## 审计风险

这不是 leakage-safe 的训练集设计；它是按当前研究需求显式引入 formal eval 分布，用来验证 code reward / OPD 目标能否带来 code 能力。所有 source row、expert temp file、positive sample index 都在 summary / blueprints 中记录，后续论文写作需要把它定位为 eval-distribution calibration 或用于诊断，不应和 leakage-safe calibration 主结果混淆。
