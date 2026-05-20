# 20260520 TRC Round7 Response-Span Followup

## Goal

R5C is the best non-leak Round5 primary Code candidate so far: Tool `0.7944`, Memory `0.7690`, Code mean `0.3257`. It still does not beat R3D primary `0.3289`, but response span helped more than code-block in primary accuracy. Round7 tests two direct followups without changing calibration data.

## Experiments

| ID | run_id | GPU | base setting | change | purpose |
|---|---|---|---|---|---|
| R7A | `trc_r7a_v2_response_e16_20260520` | 0,2 | R5C | epochs `12 -> 16` | Check whether response-span primary acc continues improving with longer optimization. |
| R7B | `trc_r7b_v2_response_proj105_e12_20260520` | 3,5 | R5C | code projection floor `0.95 -> 1.05` | Test whether stronger code projection is harmful only for code-block or also for response span. |

## Shared Settings

- Calibration: `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round5_sota_v2/mtr_late3_toolv2_codev2/trc96_expert_trajectories.jsonl`
- Init: `1.0`
- Gate parameterization: `layer-band-coefficient`
- Code span: `response`
- Code topk: `256`
- Memory trajectory: `late3 + final`
- Task floor: `1.0`, weight `50`
- Loss multipliers: `code=1.0 memory=1.6 tool=1.2`

## Decision

Use Tool/Memory gate before Code. If R7A/R7B pass Tool `>=0.79` and Memory `>=0.76`, run CURE. Otherwise delete baked checkpoint and keep logs.
