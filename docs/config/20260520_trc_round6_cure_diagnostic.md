# 20260520 TRC Round6 CURE Diagnostic Config

## Warning

Round6 is an eval-leak diagnostic experiment. It must not be reported as a paper main result. The purpose is to test whether TRC can learn Code when the code calibration trajectory is directly aligned to CURE hidden-test-passing behavior.

## Data

Shared Tool/Memory rows are copied from Round5 v2 calibration:

- Tool: 32 v2/paper ToolRL successful trajectories.
- Memory: 32 v2 RL-MemoryAgent late3 update-turn trajectories.

Code rows:

| data id | path | code composition |
|---|---|---|
| `cure_success32` | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round6_cure_diagnostic/cure_success32/trc96_expert_trajectories.jsonl` | 32 R3D LiveBench hidden-test-passing trajectories |
| `cure_success16lb16lcb` | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round6_cure_diagnostic/cure_success16lb16lcb/trc96_expert_trajectories.jsonl` | 16 R3D LiveBench + 16 R3D LiveCodeBench hidden-test-passing trajectories |

## Experiments

| ID | run_id | GPU | calibration | code span | epochs | purpose |
|---|---|---|---|---|---:|---|
| R6A | `trc_r6a_curediag_lb_codeblock_e8_20260520` | 3,5 | `cure_success32` | `code-block`, topk384 | 8 | Upper-bound test for LiveBench-aligned code-block trajectories. |
| R6B | `trc_r6b_curediag_bal_response_e8_20260520` | 0,2 | `cure_success16lb16lcb` | `response`, topk256 | 8 | Test whether balanced CURE-aligned algorithm/response span helps both LiveBench and LiveCodeBench. |

## Shared Training Settings

- Init: `1.0`
- Gate parameterization: `layer-band-coefficient`
- Task expert coefficient floor: `1.0`, weight `50`
- Loss multipliers: `code=1.0 memory=1.6 tool=1.2`
- Memory trajectory loss: enabled
- Code directional projection: `floor=0.95`, `weight=0.25`
- Selection: `loss-plateau`

## Decision

If R6 improves Code sharply while preserving Tool/Memory, the bottleneck is calibration trajectory quality. If R6 still fails, TRC hidden alignment itself is insufficient for Code and needs a different objective, such as test-driven repair / preference loss / execution-aware distillation.
