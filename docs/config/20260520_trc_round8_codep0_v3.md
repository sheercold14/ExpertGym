# 20260520 TRC Round8 CodeP0-v3 Non-Leak Calibration

## Goal

Round5/7 showed that changing span and projection can keep Tool/Memory stable, but Code primary accuracy remains below R3D. Round8 changes the non-leak Code trajectory source rather than only the optimizer: use `code_p0_v3` verified recoverable Code prompts, which have more execution-style successful ReasonFlux trajectories than the previous `sota_v2` code subset.

## Calibration

Common Tool/Memory:

- Tool: `sota_calib_v2_20260518` ToolRL rollout first, paper96 fallback.
- Memory: `sota_calib_v2_20260518` RL-MemoryAgent rollout first, paper96 fallback.
- Memory response: late 3 memory update turns + final answer.
- Per-task rows: 32, total 96.

Code variants:

| calibration | path | Code source | selected Code prompts | purpose |
|---|---|---|---:|---|
| `rf_only_late3` | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round8_codep0_v3/rf_only_late3/trc96_expert_trajectories.jsonl` | ReasonFlux `code_p0_v3_train64_s8` only | 29 unique + 3 duplicate successful trajectories | Cleanest alignment to the existing `code` task vector. |
| `rf_then_ds_late3` | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round8_codep0_v3/rf_then_ds_late3/trc96_expert_trajectories.jsonl` | ReasonFlux first, DeepSeek-R1 fallback | 32 unique | Tests whether more Code prompt coverage beats strict expert-vector consistency. |
| `rf_rolequota_late3` | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round8_codep0_v3/rf_rolequota_late3/trc96_expert_trajectories.jsonl` | ReasonFlux only, role quotas | 28 unique + 4 duplicate successful trajectories | Forces Code rows to cover `frontier=12 / partial_edge=10 / generation=8 / stable=2`. |

## Experiments

| ID | run_id | GPUs | calibration | objective/span | key change |
|---|---|---|---|---|---|
| R8A | `trc_r8a_codep0_rf_response_e12_20260520` | `0,1` | `rf_only_late3` | directional, Code response topK 256 | R5C training objective with better Code source. |
| R8B | `trc_r8b_codep0_rfds_response_e12_20260520` | `6,7` | `rf_then_ds_late3` | directional, Code response topK 256 | Tests prompt coverage vs expert-vector purity. |
| R8C | `trc_r8c_codep0_rf_relmse_e10_20260520` | pending | `rf_only_late3` | relative-MSE, Code response topK 256 | Tests whether Code needs magnitude matching rather than directional matching. |
| R8D | `trc_r8d_codep0_rf_codeblock_e12_20260520` | pending | `rf_only_late3` | directional, Code block topK 384 | Checks whether execution-style CodeP0 trajectories prefer code span again. |
| R8E | `trc_r8e_codep0_rolequota_response_e12_20260520` | pending | `rf_rolequota_late3` | directional, Code response topK 256 | Tests whether structured Code role balance beats source-order selection. |

## Shared Training Settings

- Config: `configs/gated_grpo_layer28_wide.yaml`
- Init: `1.0`
- Gate parameterization: `layer-band-coefficient`
- Epochs: 12 for R8A/B/D, 10 for R8C probe.
- LR: `0.02`
- Hidden layers: Code `4,8,12,16,20,24,28`; Tool/Memory `8,16,24,28`.
- Span: Tool `tool-call`; Memory `response` with `trajectory-turn-loss-task=memory`; Code as above.
- Loss multipliers: Code `1.0`, Memory `1.6`, Tool `1.2`.
- Task expert coefficient floor: `1.0`, weight `50.0`.
- Base drift: `beta_base=0.05`; gate anchor `gamma_gate=0.005`.

## Promote Rule

Same as Round7. First run Tool/Memory. Promote to CURE Code only if Tool mean `>=0.79` and Memory mean F1 `>=0.76`. R8C is an objective probe; if Tool collapses it is useful evidence but not a main candidate.
