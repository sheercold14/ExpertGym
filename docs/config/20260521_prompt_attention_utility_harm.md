# 2026-05-21 PromptAttention-UtilityHarm 配置说明

## 目标

PAUH 是一条 prompt-only、training-free 的 OP-VEC 合并线：只用 base model 在 calibration prompts 上的 attention projection 输入 activation，估计每个 expert/layer 对 owner task 与 non-owner tasks 的 exposure ratio，然后生成 parameter-level gate checkpoint。

它不读取 reward、不读取 expert rollout、不训练 gate。生成的 `pauh_gates.json` 可直接交给现有 `scripts/eval/opvec_bake_checkpoint.py` 烘焙。

当前实现的 layer score 是：

`raw_score = log(utility(owner task)+eps) - log(harm(non-owner tasks)+eps)`，随后在 expert 内做 z-score。

`--beta` 不是 harm 权重，而是 score 到 layer weight 的指数映射温度：

`weight_l = exp(beta * score_l)`。

## 推荐命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
PROMPTS=/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl
OUT=/tmp/shared-storage/ExpertGym/pauh/pauh_energy_layer_all_20260521

$PY scripts/attention_pauh/build_prompt_attention_utility_harm_gates.py \
  --mode-manifest "$MODE" \
  --prompt-jsonl "$PROMPTS" \
  --output-dir "$OUT" \
  --samples-per-task 32 \
  --prompt-tail-tokens 256 \
  --scope layer-all \
  --beta 0.7 \
  --default-alpha 0.75 \
  --min-coeff 0.25 \
  --max-coeff 1.25

$PY scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest "$MODE" \
  --gate-checkpoint "$OUT/pauh_gates.json" \
  --output "$OUT/baked_policy"
```

## 对照实验

1. `TA0.75`：所有 expert 固定 0.75。
2. `PAUH layer-all`：attention activation score 控制整层所有 OP-VEC modules。
3. `PAUH attn-only`：仅注入 attention q/k/v/o modes，检查 attention-only 是否足够。

先跑 Tool + Memory quick eval；若不低于 TA0.75 约 0.005，再送 Code 正式评测。
