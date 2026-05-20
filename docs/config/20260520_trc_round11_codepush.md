# 20260520 TRC Round11 Code-Push Hybrid Calibration

## Goal

Construct the next Code-improving calibration bank without pushing gates blindly.
The bank is intended for R11B/R11G-style follow-up runs after R10 tag-quota
improved Tool but R10B missed Memory (`0.7572`), while R8D CodeP0 RF-only with
code-block384 had the strongest LiveBench start (`0.3730`).

No training was launched for this task.

## Input Banks Checked

| bank | path | rows | unique prompts | expert/source distribution | Code tag/role summary |
|---|---|---:|---:|---|---|
| R10 tag-quota | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round10_codep0_tag_v1/tag_quota_default_late3/trc96_expert_trajectories.jsonl` | Tool 32 / Memory 32 / Code 32 | Tool 27 / Memory 28 / Code 32 | Code RF 28 + DS 4; Tool/Memory paper96 only | primary tags: string 11, math 7, graph 5, DP 4, greedy 4, format 1; roles: frontier 13, partial_edge 11, generation 7, stable 1 |
| R8D RF-only late3 | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round8_codep0_v3/rf_only_late3/trc96_expert_trajectories.jsonl` | Tool 32 / Memory 32 / Code 32 | Tool 32 / Memory 32 / Code 29 | Code RF 32; Tool 27 v2 ToolRL + 5 paper96; Memory 32 v2 RL-MemoryAgent | primary tags: string 14, math 7, graph 4, DP 3, greedy 2, format 2; roles: partial_edge 14, frontier 11, generation 5, stable 2 |
| R8B RF+DS late3 | `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round8_codep0_v3/rf_then_ds_late3/trc96_expert_trajectories.jsonl` | Tool 32 / Memory 32 / Code 32 | Tool 32 / Memory 32 / Code 32 | Code RF 29 + DS 3; same stable Tool/Memory as R8D | primary tags: string 11, math 7, graph 5, DP 3, greedy 3, format 3; roles: frontier 12, partial_edge 11, generation 6, stable 3 |
| stable late3 Tool/Memory | same rows as R8D/R8B/R5 stable late3 | Tool 32 / Memory 32 | Tool 32 / Memory 32 | Tool 27 v2 ToolRL + 5 paper96; Memory 32 v2 RL-MemoryAgent | not applicable |

The R8B/R8D/R5 Tool and Memory row hashes are identical. Round11 uses these
stable late3 rows instead of R10's paper96-only Tool/Memory rows.

## Construction Rule

Output directory:

`/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round11_hybrid_v1/r11b_r11g_r10tag24_rf8_stablelate3`

Rows:

- Tool: 32 stable late3 rows from R8D/R8B/R5 shared Tool bank.
- Memory: 32 stable late3 trajectory-turn rows from R8D/R8B/R5 shared Memory bank.
- Code: 24 R10 tag-quota core rows plus 8 RF-only CodeP0-v3 supplement rows.

Code selection details:

- Keep 24 R10 tag-quota rows.
- Exclude the 7 R10 rows whose prompts are replaced by RF-only supplement rows:
  `e08aa7ed80a06eb9`, `e9f843a1eb08e58e`, `a5e6e2381fb1732e`,
  `cbe4ee0799da565e`, `702801cee1f4495b`, `63718bf5a80920f3`,
  `d0fd7fc20a7cda49`.
- Drop one extra long DS greedy row:
  `code_p0v3__3175ad9d3d49acec`.
- Add 8 RF-only supplement rows from CodeP0-v3 ReasonFlux raw positives:
  2 dynamic-programming generation rows, 3 graph rows, 2 greedy rows, and
  1 stable format-sensitive row absent from R10.
- For 7 overlapping prompts, the supplement uses an alternate successful RF
  sample instead of the R10-selected sample when available.

All Code rows are `CodeContests_train` CodeP0-v3 expert-success rows. No
LiveBench, LiveCodeBench, CURE hidden prompt, hidden test, generated output, or
formal CURE diagnostic row is used.

## Reproduction

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python

$PY scripts/trc/build_trc_round11_hybrid_calibration.py
```

Generated files:

- `trc96_expert_trajectories.jsonl`
- `trc96_summary.json`
- `trc96_summary.md`
- `README.md`

Builder:

`scripts/trc/build_trc_round11_hybrid_calibration.py`

## Final Statistics

Final task balance:

- Rows: Tool 32 / Memory 32 / Code 32
- Unique prompts: Tool 32 / Memory 32 / Code 32
- Duplicate prompt rows: 0 for all tasks
- Reward train: min/mean/max all `1.0`

Final expert/source distribution:

- Tool: 27 `tool_expert_toolrl_qwen25_7b_sota_v2_train128_s4_seed20260518` + 5 paper96 ToolRL
- Memory: 32 `memory_expert_rl_memoryagent7b_sota_v2_train128_s4_seed20260518`
- Code: 29 ReasonFlux CodeP0-v3 RF + 3 DeepSeek-R1 CodeP0-v3 fallback

Final Code roles:

```json
{"frontier": 12, "generation": 7, "partial_edge": 11, "stable": 2}
```

Final Code primary tags:

```json
{"dynamic_programming": 4, "format_sensitive": 2, "graph": 5, "greedy": 3, "math": 7, "string": 11}
```

Final Code all-tag counts:

```json
{"array": 7, "dynamic_programming": 4, "format_sensitive": 32, "graph": 8, "greedy": 10, "math": 14, "simulation": 5, "stdin_stdout": 32, "string": 18}
```

Quality checks from `trc96_summary.json`:

- `row_count_ok=true`
- `task_balance_ok=true`
- `trajectory_id_unique_ok=true`
- `blank_response_count=0`
- `nonpositive_reward_train_count=0`
- `code_leakage_marker_hits=[]`
- `code_present_false_count=0`
- `syntax_false_count=0`

## Why This Might Improve Code

This bank keeps R10's useful algorithm-tag coverage while avoiding R10B's weak
paper96-only Memory base. It also shifts Code slightly toward the R8D/RF-only
code-block-friendly signal that produced the best Round8 LiveBench opening
score, without sacrificing Code prompt uniqueness.

The final Code mix preserves the R10 tag spread for DP/graph/math/string, keeps
most DS coverage from R8B/R10, removes one very long DS greedy response, and
adds RF-only alternate successful samples for DP/graph/greedy prompts. This is
a data change, not a gate-magnitude push.

## Training Runs

| ID | run_id | calibration | key setting | status |
|---|---|---|---|---|
| R11G | `trc_r11g_hybrid_response256_mem18_e12_20260520` | `r11b_r11g_r10tag24_rf8_stablelate3` | Code response topK256, Memory loss multiplier `1.8`, Tool `1.2`, Code `1.0` | running on GPUs `0,7` |
| R11H | `trc_r11h_hybrid_codeblock384_mem18_e12_20260520` | `r11b_r11g_r10tag24_rf8_stablelate3` | Code code-block topK384, Memory loss multiplier `1.8`, Tool `1.2`, Code `1.0` | running on GPUs `1,6` |

R11G intentionally reuses the R10D loss scale because R10D repaired R10B's
Memory miss (`0.7572 -> 0.7679`) without damaging Tool (`0.7944`). R11H uses
the same data and loss scale but swaps Code span to code-block384 to test the
R8D LiveBench signal under the cleaner hybrid bank.
