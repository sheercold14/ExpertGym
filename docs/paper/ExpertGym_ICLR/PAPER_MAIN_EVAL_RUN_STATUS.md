# Paper-Main Eval6 Run Status

Last updated: 2026-05-23 15:50:29 +0800

This file tracks the paper-main Eval6 execution state.  The selected minimum queue is complete.

## Queue Status

| candidate | Tool/BFCL | Memory/HotpotQA | Code/CURE | status |
| --- | --- | --- | --- | --- |
| `bcrc_v18_alias_v9` | done | done | done | ready |
| `no_behavior_v1_code_only` | done | done | done | ready |
| `hard_behavior_v8` | done | done | done | ready |

Aggregate:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md
```

Regeneration command:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/aggregate_iclr_paper_main_eval.py
```

## Completed Scores

| candidate | Tool | Tool live | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN | Avg(T/M/C) | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bcrc_v18_alias_v9` | 0.7931 | 0.7188 | 0.6270 | 0.7570 | 0.3301 | 0.4373 | 0.3939 | 0.6267 | 0.3301 |
| `no_behavior_v1_code_only` | 0.7956 | 0.7188 | 0.6387 | 0.7650 | 0.3260 | 0.4355 | 0.4076 | 0.6289 | 0.3260 |
| `hard_behavior_v8` | 0.7919 | 0.7188 | 0.6289 | 0.7568 | 0.3274 | 0.4381 | 0.4047 | 0.6254 | 0.3274 |

## Launch Commands Used

Tool/Memory:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export CANDIDATES=bcrc_v18_alias_v9,no_behavior_v1_code_only,hard_behavior_v8
export ROOT=/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523
export RUN_ID=iclr_main_eval6_20260523
export EXPERIMENT_NAME=expertgym-iclr-main-eval6
DRY_RUN=0 PHASE=tool_memory TOOL_GPU=0 TOOL_PORT=8160 MEMORY_GPU_IDS=0 \
  bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

Code, main method:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export CANDIDATES=bcrc_v18_alias_v9
export ROOT=/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523
export RUN_ID=iclr_main_eval6_20260523
export EXPERIMENT_NAME=expertgym-iclr-main-eval6
DRY_RUN=0 PHASE=code CODE_GPU_GROUPS="[[2,3]]" \
  bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

Code, no-behavior ablation:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export CANDIDATES=no_behavior_v1_code_only
export ROOT=/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523
export RUN_ID=iclr_main_eval6_20260523
export EXPERIMENT_NAME=expertgym-iclr-main-eval6
DRY_RUN=0 PHASE=code CODE_GPU_GROUPS="[[2,3]]" \
  bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

Code, hard-behavior ablation:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
export CANDIDATES=hard_behavior_v8
export ROOT=/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523
export RUN_ID=iclr_main_eval6_20260523
export EXPERIMENT_NAME=expertgym-iclr-main-eval6
DRY_RUN=0 PHASE=code CODE_GPU_GROUPS="[[2,3]]" \
  bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

## Artifact Roots

| candidate | Tool/Memory logs | Code logs |
| --- | --- | --- |
| `bcrc_v18_alias_v9` | `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/bcrc_v18_alias_v9/iclr_main_eval6_20260523/tool_memory/logs` | `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/bcrc_v18_alias_v9/iclr_main_eval6_20260523/code/logs` |
| `no_behavior_v1_code_only` | `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/no_behavior_v1_code_only/iclr_main_eval6_20260523/tool_memory/logs` | `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/no_behavior_v1_code_only/iclr_main_eval6_20260523/code/logs` |
| `hard_behavior_v8` | `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/hard_behavior_v8/iclr_main_eval6_20260523/tool_memory/logs` | `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/hard_behavior_v8/iclr_main_eval6_20260523/code/logs` |

Queue logs:

```text
/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/tool_memory_queue.log
/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/code_bcrc_queue.log
/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/code_no_behavior_queue.log
/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/code_hard_behavior_queue.log
```

## Paper Interpretation

The selected queue is complete.  It supports an ablation claim inside the BCRC-family rows:

```text
soft behavior constraints have the highest Code pass@1 and worst-task score;
no-behavior has the highest simple average.
```

It does not support a broad SOTA claim.
