# 20260520 TRC Round16 Non-Leak Code Contrast

## Goal

Round15 证明 R5A 训练配置可复现，但 formal contrast8 只能作为诊断：contrast 行太少，后期 active rate 很低，而且包含正式 Code eval anchor，不适合作为论文主结果。

Round16 将 Code 数据换成严格非泄漏的 CodeP0-v3 train pass/fail pairs，目标是验证：

```text
same prompt: pass code block direction > fail code block direction
```

是否能把 R5A/R11B 的高 BoN 转成更高 single-sample Acc，同时保持 Tool/Memory。

## Calibration

Builder:

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
$PY scripts/trc/build_trc_round16_nonleak_code_contrast_calibration.py
```

Output:

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl
```

Summary:

- Tool: 32 rows, reuse R5A stable Tool bank.
- Memory: 32 rows, reuse R5A stable Memory late3 trajectory bank.
- Code: 32 rows from CodeP0-v3 CodeContests train only.
- Code contrast: 22 unique prompts with same-prompt pass/fail trajectories.
- Code positive fill: 10 unique train prompts.
- ReasonFlux rows: 26.
- DeepSeek fallback rows: 6.
- Negative reward mean: 0.1864, min 0.0, max 0.8333.
- Leakage policy: no LiveBench, LiveCodeBench, or formal CURE eval prompt/output/test is used.

Rationale: strict non-leak RF+DS merged rollouts only provide 22 unique pass/fail prompts. We keep this number honest and fill the remaining 10 Code rows from CodeP0-v3 positive train trajectories instead of importing formal eval anchors.

## Experiments

| ID | run id | GPUs | data | contrast weight | main delta |
|---|---|---|---|---:|---|
| R16A | `trc_r16a_nonleak_contrast22_w15_e12_20260520` | 0,1 | non-leak Tool32/Memory32/Code22+10 | 1.5 | Clean replacement of Round15 formal contrast8 with CodeP0 train contrast22. |
| R16B | `trc_r16b_nonleak_contrast22_w30_e12_20260520` | 2,3 | same as R16A | 3.0 | Tests whether pass/fail signal strength is the bottleneck. |
| R16C | `trc_r16c_rfonly_contrast16_w15_e12_20260520` | 4,5 | Tool32/Memory32 + RF-only Code16+16 | 1.5 | Tests whether DeepSeek fallback trajectories hurt the existing ReasonFlux code task vector. |
| R16D | `trc_r16d_nonleak_contrast22_response256_w15_e12_20260520` | 6,7 | same as R16A | 1.5 | Tests whether Code needs response-span reasoning alignment instead of code-block-only alignment. |

## Shared Training Settings

- Config: `configs/gated_grpo_layer28_wide.yaml`
- Init: `1.0`
- Gate parameterization: `layer-band-coefficient`
- Epochs: `12`
- LR: `0.02`
- Optimizer: `adamw`
- Gradient checkpointing: on
- Hidden layers:
  - Code: `4,8,12,16,20,24,28`
  - Tool/Memory: `8,16,24,28`
- TopK:
  - Code: `384`
  - Tool/Memory: `96`
- Span:
  - Tool: `tool-call`
  - Memory: `response`, with late3 trajectory turns
  - Code: `code-block`
- Loss multipliers:
  - Code: `1.0`
  - Memory: `1.6`
  - Tool: `1.2`
- Directional projection:
  - Code floor: `0.95`
  - Code weight: `0.25`
- Gate regularization:
  - `beta_base=0.05`
  - `gamma_gate=0.03`
  - `task_expert_coefficient_floor=1.0`
  - `task_expert_coefficient_floor_weight=50.0`

## Promotion Rule

Do not run expensive Code formal eval until quick gate passes:

- BFCL Tool quick mean `>= 0.79`
- Memory quick mean F1 `>= 0.76`

If both pass, run CURE Code. Main success criterion is not just BoN increase, but mean Acc improvement and smaller BoN-to-Acc gap relative to R5A/R11B.

## Added Parallel Controls

R16C data:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/trc/build_trc_round16_nonleak_code_contrast_calibration.py \
  --output-dir /tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_rfonly_code_contrast_v1 \
  --code-rollout /tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl \
  --code-contrast-count 16 \
  --code-positive-fill-count 16
```

R16C uses 32 Code rows from ReasonFlux only: 16 pass/fail contrast + 16 positive fill, 29 unique prompts. The small duplicate count is intentional and mirrors earlier RF-only banks; it avoids mixing DeepSeek hidden trajectories into a ReasonFlux code task vector.

R16D keeps R16A data but changes only Code span:

- `TASK_RESPONSE_SPAN_MODE`: `code=response`
- `TASK_TOPK_TOKENS`: `code=256`

This tests whether final-code-only alignment is too narrow for Code Acc.
