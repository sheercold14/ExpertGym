# 20260520 TRC Round12 RF-only Code-Push Calibration

## Goal

Construct R12D RF-only tag-quota calibration data without launching training.
The purpose is to isolate whether R8D's stronger LiveBench opening signal came
from ReasonFlux/RF-only CodeP0 purity rather than DeepSeek fallback coverage.

## Output

`/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3`

Files:

- `trc96_expert_trajectories.jsonl`
- `trc96_summary.json`
- `trc96_summary.md`
- `README.md`

## Inputs

- Tool/Memory stable late3 rows:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round11_hybrid_v1/r11b_r11g_r10tag24_rf8_stablelate3/trc96_expert_trajectories.jsonl`
- Code RF-only raw positives:
  `/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl`

No DeepSeek fallback is used. No CURE hidden, LiveBench, or LiveCodeBench row is
used.

## Reproduction

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python

$PY scripts/trc/build_trc_round12_rfonly_tagquota_calibration.py
```

Builder:

`scripts/trc/build_trc_round12_rfonly_tagquota_calibration.py`

## Construction Rule

- Total rows: 96
- Tool: 32 rows copied from the R11 hybrid stable late3 Tool rows.
- Memory: 32 rows copied from the R11 hybrid stable late3 Memory trajectory-turn rows.
- Code: 32 rows selected only from ReasonFlux CodeP0-v3 successful trajectories.
- Code quota type: primary code tag.
- Code quotas: `string=11`, `math=7`, `graph=5`, `dynamic_programming=4`, `greedy=3`, `format_sensitive=2`.

RF-only capacity before selection:

- Positive samples: 161
- Positive unique prompts: 29
- Unique primary-tag capacity:
  `string=11`, `math=7`, `graph=4`, `dynamic_programming=3`, `greedy=2`, `format_sensitive=2`

Because RF-only has only 29 unique positive prompts, 32 Code rows require 3
duplicate prompt rows. The duplicates are assigned to the low-capacity algorithm
tags needed for coverage:

- `code_p0v3__cbe4ee0799da565e`: graph, 2 rows
- `code_p0v3__e9f843a1eb08e58e`: dynamic_programming, 2 rows
- `code_p0v3__63718bf5a80920f3`: greedy, 2 rows

## Final Statistics

Final task balance:

- Rows: Tool 32 / Memory 32 / Code 32
- Unique prompts: Tool 32 / Memory 32 / Code 29
- Duplicate prompt rows: Tool 0 / Memory 0 / Code 3
- Reward train: min/mean/max all `1.0`

Final expert/source distribution:

- Tool: 27 `tool_expert_toolrl_qwen25_7b_sota_v2_train128_s4_seed20260518` + 5 paper96 ToolRL
- Memory: 32 `memory_expert_rl_memoryagent7b_sota_v2_train128_s4_seed20260518`
- Code: 32 `code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518`

Final Code roles:

```json
{"frontier": 13, "generation": 6, "partial_edge": 11, "stable": 2}
```

Final Code primary tags:

```json
{"dynamic_programming": 4, "format_sensitive": 2, "graph": 5, "greedy": 3, "math": 7, "string": 11}
```

Final Code all-tag counts:

```json
{"array": 6, "dynamic_programming": 4, "format_sensitive": 32, "graph": 7, "greedy": 9, "math": 13, "simulation": 5, "stdin_stdout": 32, "string": 17}
```

## Quality Checks

From `trc96_summary.json`:

- `row_count_ok=true`
- `task_balance_ok=true`
- `trajectory_id_unique_ok=true`
- `blank_response_count=0`
- `nonpositive_reward_train_count=0`
- `code_rf_only_ok=true`
- `code_deepseek_marker_hits=[]`
- `code_leakage_marker_hits=[]`
- `code_present_false_count=0`
- `syntax_false_count=0`

## Interpretation

R12D is intentionally narrower than R11: it sacrifices 3 Code unique prompts to
remove DeepSeek fallback entirely and preserve RF-only purity. If this improves
or matches R8D-style Code transfer while Tool/Memory stay stable, the likely
signal is expert-source purity rather than mixed-teacher coverage. If it loses
Code, then the duplicate rows and reduced unique prompt coverage are likely
hurting more than RF-only purity helps.

## Training Attempt

### R12D RF-only Tag-Quota Code-Block

Status: running since 2026-05-20 11:02 CST. Latest observed epoch 9 at
2026-05-20 11:19 CST: code gate `1.1801`, memory gate `0.9964`, tool gate
`1.1665`, mean total loss `0.7538`.

```bash
EXP_ID=trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520
GPU_LIST=1,4
CONFIG=configs/gated_grpo_layer28_wide.yaml
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl
EPOCHS=12
LR=0.02
TASK_EXPERT_COEFFICIENT_FLOOR=1.0
TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0
TASK_HIDDEN_LAYERS=code=4,8,12,16,20,24,28
TASK_TOPK_TOKENS=code=384
TASK_DIRECTIONAL_PROJECTION_FLOOR=code=0.95
TASK_DIRECTIONAL_PROJECTION_WEIGHT=code=0.25
TASK_RESPONSE_SPAN_MODE="tool=tool-call code=code-block memory=response"
TASK_LOSS_MULTIPLIER="code=1.0 memory=1.6 tool=1.2"
TRAJECTORY_TURN_LOSS_TASKS=memory
BAKED_DIR=/tmp/shared-storage/OnPolicy/checkpoints/trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520-selected
```

Run artifacts:

- `/tmp/shared-storage/OnPolicy/runs/trc/trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520`
- `/tmp/shared-storage/OnPolicy/runs/trc/trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520.launch.log`
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520-selected`

Evaluation rule: after bake, run Tool+Memory quick gate first. Promote to Code
only if Tool mean is at least `0.79` and Memory mean F1 is at least `0.76`.

Result status: selected/baked at epoch 12 on 2026-05-20 11:23 CST. Gate means
are approximately Code `1.2403`, Memory `0.9945`, Tool `1.2110`. Tool/Memory
quick gate is running as `eval_r12d_tm_20260520`.

### R12B Mixed Tag-Quota Code-Block

Status: running since 2026-05-20 11:04 CST. Latest observed epoch 8 at
2026-05-20 11:19 CST: code gate `1.1600`, memory gate `0.9991`, tool gate
`1.1434`, mean total loss `0.7332`.

This is the direct contrast to R12D. It keeps the same Code `code-block` topK384
alignment, but uses the Round10 tag-quota calibration with mixed RF+DeepSeek
coverage and Memory multiplier `1.8`.

```bash
EXP_ID=trc_r12b_tag_codeblock384_mem18_e12_20260520
GPU_LIST=2,3
CONFIG=configs/gated_grpo_layer28_wide.yaml
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round10_codep0_tag_v1/tag_quota_default_late3/trc96_expert_trajectories.jsonl
EPOCHS=12
LR=0.02
TASK_EXPERT_COEFFICIENT_FLOOR=1.0
TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0
TASK_HIDDEN_LAYERS=code=4,8,12,16,20,24,28
TASK_TOPK_TOKENS=code=384
TASK_DIRECTIONAL_PROJECTION_FLOOR=code=0.95
TASK_DIRECTIONAL_PROJECTION_WEIGHT=code=0.25
TASK_RESPONSE_SPAN_MODE="tool=tool-call code=code-block memory=response"
TASK_LOSS_MULTIPLIER="code=1.0 memory=1.8 tool=1.2"
TRAJECTORY_TURN_LOSS_TASKS=memory
BAKED_DIR=/tmp/shared-storage/OnPolicy/checkpoints/trc_r12b_tag_codeblock384_mem18_e12_20260520-selected
```

Run artifacts:

- `/tmp/shared-storage/OnPolicy/runs/trc/trc_r12b_tag_codeblock384_mem18_e12_20260520`
- `/tmp/shared-storage/OnPolicy/runs/trc/trc_r12b_tag_codeblock384_mem18_e12_20260520.launch.log`
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r12b_tag_codeblock384_mem18_e12_20260520-selected`

### R12E RF-only Tag-Quota Response256

Status: running since 2026-05-20 11:15 CST.

This is the response-span counterpart of R12D. It uses the same RF-only
tag-quota data as R12D, but aligns Code on `response` topK256 instead of
`code-block` topK384, with Memory multiplier `1.8`.

```bash
EXP_ID=trc_r12e_rfonly_tagquota_response256_mem18_e12_20260520
GPU_LIST=0,7
CONFIG=configs/gated_grpo_layer28_wide.yaml
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl
EPOCHS=12
LR=0.02
TASK_EXPERT_COEFFICIENT_FLOOR=1.0
TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0
TASK_HIDDEN_LAYERS=code=4,8,12,16,20,24,28
TASK_TOPK_TOKENS=code=256
TASK_DIRECTIONAL_PROJECTION_FLOOR=code=0.95
TASK_DIRECTIONAL_PROJECTION_WEIGHT=code=0.25
TASK_RESPONSE_SPAN_MODE="tool=tool-call code=response memory=response"
TASK_LOSS_MULTIPLIER="code=1.0 memory=1.8 tool=1.2"
TRAJECTORY_TURN_LOSS_TASKS=memory
BAKED_DIR=/tmp/shared-storage/OnPolicy/checkpoints/trc_r12e_rfonly_tagquota_response256_mem18_e12_20260520-selected
```

Evaluation rule: after bake, run Tool+Memory quick gate first. Promote to Code
only if Tool mean is at least `0.79` and Memory mean F1 is at least `0.76`.

## Live Monitor Snapshot

2026-05-20 11:19 CST:

- `train_r12d_rfonly_codeblock384_mem16_20260520` is training, not baked.
  Epoch gates: e1 C/M/T `1.0200/0.9800/1.0186`, e2
  `1.0400/0.9800/1.0383`, e3 `1.0600/0.9854/1.0580`, e4
  `1.0800/0.9913/1.0773`, e5 `1.1000/0.9956/1.0964`, e6
  `1.1200/0.9978/1.1149`, e7 `1.1401/0.9982/1.1328`, e8
  `1.1601/0.9974/1.1500`, e9 `1.1801/0.9964/1.1665`.
- `train_r12b_tag_codeblock384_mem18_20260520` is training, not baked. Epoch
  gates: e1 C/M/T `1.0200/0.9800/1.0186`, e2
  `1.0400/0.9813/1.0373`, e3 `1.0600/0.9878/1.0557`, e4
  `1.0800/0.9941/1.0736`, e5 `1.1000/0.9984/1.0920`, e6
  `1.1200/1.0003/1.1099`, e7 `1.1400/1.0003/1.1270`, e8
  `1.1600/0.9991/1.1434`.
- No selected checkpoint directory is available yet for either R12B or R12D, so
  neither is ready for quick gate.
