# Paper-Main Full Eval6 Queue

Date: 2026-05-23

Purpose: freeze the minimum full-evaluation queue needed to turn the current RCF-BC mechanism evidence into a comparable ICLR main benchmark. Existing baseline rows in `MAIN_BENCHMARK_TABLE_DRAFT.md` are full Eval6; current RCF-BC rows are mechanism/quick rows and must not be mixed into the final main table until this queue is completed.

## Candidate Set

| candidate id | paper role | baked checkpoint | gate checkpoint | status |
|---|---|---|---|---|
| `bcrc_v18_alias_v9` | main method: soft behavior-constrained residual field | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/residual_capability_field_behavior_constraints_v18/gates.json` | checkpoint ready |
| `no_behavior_v1_code_only` | no behavior constraint ablation | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_contrast_v1` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_contrast_v1/gates.json` | checkpoint ready |
| `hard_behavior_v8` | hard behavior constraint ablation | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_v8` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_v8/gates.json` | checkpoint ready |
| `strict_cleanup_v19` | optional strict residual-archetype cleanup | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_archetype_consistency_v19` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_archetype_consistency_v19/gates.json` | optional |
| `scalar_code_half_v14` | optional scalar negative control | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_half_v14` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_v9_code_half_v14/gates.json` | optional |
| `scalar_code_zero_v15` | optional scalar negative control | `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v9_code_zero_v15` | `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_v9_code_zero_v15/gates.json` | optional |

Important alias note:

- The named `v18` gate and `v9` gate have numerically identical 588 gate coefficients.
- The `v18` baked checkpoint path `/tmp/shared-storage/OnPolicy/checkpoints/residual_capability_field_behavior_constraints_v18` does not currently exist.
- The queue therefore evaluates the existing `v9` checkpoint under the explicit id `bcrc_v18_alias_v9`. This must be stated in any paper table footnote or artifact README.

## Required Metrics

Each paper-main candidate needs the same metrics as the baseline rows:

| metric | source |
|---|---|
| Tool BFCL mean | `run_full_eval_suite.sh` with `RUN_TOOL=1` |
| Memory F1 | `run_full_eval_suite.sh` with `RUN_MEMORY=1` |
| Code pass@1 | `run_full_eval_suite.sh` with `RUN_CODE=1` |
| Code BoN | Code/CURE harness output |
| Average / worst-task regression | computed after table aggregation |

ToolRL-80 can be reported as a secondary robustness metric, but it should not replace BFCL in the main Eval6 table unless the baseline rows are also recomputed under ToolRL-80.

## Reproducible Entry Point

Script:

```bash
skill/command/run_20260523_iclr_paper_main_eval.sh
```

Aggregate finished logs:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/aggregate_iclr_paper_main_eval.py
```

Aggregate outputs:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md
docs/paper/ExpertGym_ICLR/paper_main_eval6_aggregate.csv
docs/paper/ExpertGym_ICLR/paper_main_eval6_aggregate.json
```

Default behavior is dry-run:

```bash
DRY_RUN=1 PHASE=list bash skill/command/run_20260523_iclr_paper_main_eval.sh
DRY_RUN=1 PHASE=tool_memory bash skill/command/run_20260523_iclr_paper_main_eval.sh
DRY_RUN=1 PHASE=code bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

Run the minimum queue:

```bash
export CANDIDATES=bcrc_v18_alias_v9,no_behavior_v1_code_only,hard_behavior_v8
export ROOT=/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523
export RUN_ID=iclr_main_eval6_20260523
export EXPERIMENT_NAME=expertgym-iclr-main-eval6
```

Tool + Memory:

```bash
DRY_RUN=0 \
PHASE=tool_memory \
TOOL_GPU=0 \
TOOL_PORT=8160 \
MEMORY_GPU_IDS=0 \
bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

Code:

```bash
DRY_RUN=0 \
PHASE=code \
CODE_GPU_GROUPS="[[0,1]]" \
bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

Optional candidates:

```bash
export CANDIDATES=strict_cleanup_v19,scalar_code_half_v14,scalar_code_zero_v15
DRY_RUN=0 PHASE=tool_memory TOOL_GPU=0 MEMORY_GPU_IDS=0 bash skill/command/run_20260523_iclr_paper_main_eval.sh
DRY_RUN=0 PHASE=code CODE_GPU_GROUPS="[[0,1]]" bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

## Scheduling Notes

- Run BFCL jobs sequentially unless the harness is confirmed safe for concurrent model configs and ports.
- Tool + Memory are relatively fast and should be used as the first gate before spending time on Code.
- Code is the expensive leg; only run it for candidates whose Tool and Memory do not regress below the paper threshold.
- Store outputs under `/tmp/shared-storage/ExpertGym/iclr_main_eval6_20260523/<candidate>/<run_id>/...`.

## Paper Decision Rule

Use the full Eval6 table only after the three minimum candidates finish:

```text
main method = bcrc_v18_alias_v9
no behavior constraint = no_behavior_v1_code_only
hard behavior constraint = hard_behavior_v8
```

If the main method is not better than the ablations on the full suite, keep RCF-BC as a mechanism finding and do not claim SOTA. The paper claim should then be narrowed to residual-level diagnosis and behavior-constrained trade-off control.
