# RAIN vs ExpertGym Task Vector Diagnosis

## Question

比较 RAIN-Merging 中的 instruction / reasoning 角色与 ExpertGym 中 Tool / Memory / Code / R1 task vector 的差异，判断下一步是否应继续沿 gate learning 推进，还是把 RAIN 的 behavior-preserving idea 迁移到 ExpertGym。

## Key Conclusion

RAIN 和 ExpertGym 当前使用的 `task vector` 不是同一种对象。RAIN 的 reasoning model 不是 additive vector，而是被保护的 anchor；ExpertGym 的 Tool/Memory/Code 是相对同一个 instruction base 的 RL expert delta。直接把 RAIN 的成功解释成“多 expert gate 会自动学好”是不成立的。

## Source Artifacts

- rain_stage1_config: `/tmp/shared-storage/RAIN/rain_paper_eq_formula_fp64_full_05003cc_20260522_203804/stage1/projected_task_vectors_config.json`
- rain_stage3_stats: `/tmp/shared-storage/RAIN/rain_fp64_lambda_sweep_05003cc_20260522_223000/runs/lambda_0.9/stage3/unified_merge_stats.json`
- rain_lambda_grid: `/tmp/shared-storage/RAIN/rain_fp64_lambda_sweep_05003cc_20260522_223000/lambda_grid_summary.json`
- opvec_manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4-full-bf16-real/mode_manifest.json`
- opvec_diagnostics: `/tmp/shared-storage/OnPolicy/modes/opvec4-full-bf16-real/diagnostics.json`
- opvec_r1_manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519/mode_manifest.json`
- opvec_r1_diagnostics: `/tmp/shared-storage/OnPolicy/modes/opvec4_r1math_raw_20260519/diagnostics.json`

## Semantic Comparison

| item | RAIN-Merging | ExpertGym OP-VEC |
|---|---|---|
| anchor | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-7B-Instruct |
| additive vector | Qwen2.5-Instruct - Qwen2.5-Base | Tool / Memory / Code expert - Qwen2.5-Instruct |
| reasoning role | protected target behavior, not a vector | optional fourth heterogeneous R1 delta |
| primary control | null-space projection + attention utility/harm alpha + lambda grid | reward/OPD/TRC/hand-designed gates over expert deltas |
| risk | instruction delta may break reasoning rollout | expert deltas are unbalanced and can conflict across behaviors |

## Norm / Structure Evidence

### RAIN lambda=0.9 projected instruction perturbation

- params modified: `7580418048`
- slices: `3105`
- sum slice norm: `605.6711`
- sqrt-sum slice norm: `18.8662`
- max slice norm: `2.2188`

| module | n | sum norm | sqrt-sum norm | mean | max |
|---|---:|---:|---:|---:|---:|
| q | 756 | 116.8498 | 4.3802 | 0.1546 | 0.2119 |
| k | 756 | 115.3989 | 4.3215 | 0.1526 | 0.1914 |
| v | 756 | 111.8617 | 4.1838 | 0.1480 | 0.1885 |
| o | 756 | 111.2560 | 4.1582 | 0.1472 | 0.2275 |
| ffn | 81 | 150.3047 | 16.8309 | 1.8556 | 2.2188 |

### ExpertGym Tool / Memory / Code deltas

| expert | total L2 | sum param L2 | max abs | MLP energy | attn energy |
|---|---:|---:|---:|---:|---:|
| code | 0.6203 | 7.4708 | 0.000450 | 0.8546 | 0.1454 |
| memory | 5.3621 | 64.3707 | 0.000488 | 0.8713 | 0.1287 |
| tool | 1.2549 | 15.0563 | 0.000196 | 0.8694 | 0.1306 |

### ExpertGym raw R1/Math delta diagnostic

| expert | total L2 | sum param L2 | max abs |
|---|---:|---:|---:|
| code | 0.6203 | 7.4708 | 0.000451 |
| memory | 5.3621 | 64.3707 | 0.000488 |
| reasoning | 347.0678 | 4186.9895 | 0.500000 |
| tool | 1.2549 | 15.0563 | 0.000196 |

## Findings

- RAIN does not learn or add a reasoning task vector; the reasoning model is the anchor being protected.
- ExpertGym adds multiple RL expert deltas relative to the instruction anchor, so its deltas are capability priors rather than behavior anchors.
- RAIN's alpha/grid controls a projected instruction delta after behavior protection; ExpertGym gates directly scale unprojected expert deltas unless an explicit behavior constraint is added.
- ExpertGym delta norms are highly unbalanced: code=0.620, tool=1.255, memory=5.362.
- The raw R1/Math reasoning delta norm is 347.068, about 559.5x code and 64.7x memory.
- RAIN lambda=0.9 applies a projected instruction perturbation with slice sqrt-norm 18.866; this is not directly comparable to OP-VEC total L2, but it shows RAIN operates after projection and alpha selection, not on raw expert deltas.

## Research Implication

下一步不应直接把 RAIN 的 alpha 公式照搬到 ExpertGym，也不应继续只调全局 gate。更合理的主线是：

1. 用 RAIN 的第一性原则定义 ExpertGym 的 behavior anchors：Tool call span、Memory full trajectory、Code pass/fail execution span。
2. 对每个 candidate expert delta 先做 behavior-preserving projection 或 soft constraint，再估计 utility/harm。
3. 把 calibration 从“训练答案”降级为“诊断 residual 是否支持/伤害某个行为”的 probe；算法输出是 residual-level routing，而不是 reward-fitting gate。

这条线能把 RAIN/RAM/ExpertGym 统一为一个更简洁的论文叙事：能力向量必须在行为子空间约束下组合。
