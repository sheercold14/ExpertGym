# 20260520 TRC Round9 Focused Code Span

## Goal

R8A/B show that CodeP0-v3 response-span training is stable, but it is unclear whether Code needs broad response coverage or a smaller set of high-energy residual tokens. Round9 tests a focused-span variant while keeping calibration and loss otherwise unchanged.

## Experiments

| ID | run_id | GPUs | calibration | change | purpose |
|---|---|---|---|---|---|
| R9A | `trc_r9a_codep0_rf_response128_e12_20260520` | `2,3` | R8A `rf_only_late3` | Code response topK `256 -> 128` | Test whether cleaner high-energy Code response alignment improves primary Code without overfitting long explanations. |
| R9B | `trc_r9b_rolequota_response128_e12_20260520` | `6,7` | R8E `rf_rolequota_late3` | role-quota Code rows + Code response topK `128` | Combine role-balanced Code prompt coverage with focused response residual; controls against R8E broad-span and R9A RF-only data. |

## Shared Settings

Same as R8A:

- Config: `configs/gated_grpo_layer28_wide.yaml`
- Init: `1.0`
- Gate parameterization: `layer-band-coefficient`
- Objective: directional residual
- Tool span: `tool-call`
- Memory: late3 trajectory turns + final answer
- Code span: `response`
- Loss multipliers: Code `1.0`, Memory `1.6`, Tool `1.2`
- Task expert coefficient floor: `1.0`, weight `50.0`

## Decision

Use the same gate as R8: Tool mean `>=0.79` and Memory mean F1 `>=0.76` before CURE Code.

## Live Status

- R9A completed and baked: selected epoch `12`, gate means code `1.2410`, memory `0.9935`, tool `1.2307`. Tool eval failed gate: mean `0.7788` (`live_parallel=0.75`, `live_parallel_multiple=0.625`, `parallel=0.885`, `parallel_multiple=0.855`), so Memory was stopped and no Code eval is planned.
- R9B completed and baked: selected epoch `12`, gate means code `1.2409`, memory `0.9934`, tool `1.2286`. Tool eval failed gate: mean `0.7788`, same failure pattern as R9A, so Memory was stopped and no Code eval is planned.
