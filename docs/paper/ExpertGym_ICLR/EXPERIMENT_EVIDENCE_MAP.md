# Experiment Evidence Map

This file maps the ICLR draft claims to current artifacts. A claim should not be promoted to paper-main unless the evidence status is `ready`.

## Claims

| claim | current evidence | status | paper use |
|---|---|---|---|
| RAIN and ExpertGym use different task-vector semantics | `docs/report/RCRF/20260523_rain_expertgym_task_vector_diagnosis.md` | ready | motivation / diagnostics |
| RAIN full protocol improves over local R1 after lambda grid | `/mnt/cache/wuruixiao/users/lsc/Agent/RAIN-merging/skill/RAIN-Paper-Full-Reproduction-20260522.md` | ready | related mechanism, not main benchmark |
| ExpertGym deltas have unequal norms | `docs/report/RCRF/20260523_rain_expertgym_task_vector_diagnosis.md` and `docs/report/task_vector_norm_diagnostics_20260517.md` | ready | explains why scalar coefficient is misleading |
| R1/Math delta is not commensurate with Tool/Memory/Code deltas | `docs/report/RCRF/20260523_rain_expertgym_task_vector_diagnosis.md` | ready | supports excluding raw R1 as ordinary expert |
| 8765 frontend materializes the residual diagnostic protocol | `docs/report/RCRF/20260523_8765_frontend_diagnostic_method_and_findings.md`, `docs/paper/ExpertGym_ICLR/DIAGNOSTIC_PROTOCOL_8765.md` | ready | method / diagnostic protocol |
| 8765 diagnostics imply the BCRC design rule | `docs/paper/ExpertGym_ICLR/DIAGNOSIS_TO_METHOD_BRIDGE.md` | ready | method derivation |
| 8765 diagnostic claims are mapped to paper-safe boundaries | `docs/paper/ExpertGym_ICLR/DIAGNOSTIC_CLAIMS_TABLE.md` | ready | prevents overclaiming mechanism evidence as benchmark evidence |
| Code source/span evidence is not a single smooth direction | `docs/report/RCRF/20260523_8765_frontend_diagnostic_method_and_findings.md`, `source_conflict_pairs.csv` | ready | motivates span-conditioned Code utility |
| Memory-Code conflict is residual-key level, not expert-scalar level | `docs/report/RCRF/20260523_8765_frontend_diagnostic_method_and_findings.md`, `docs/report/RCRF/20260522_residual_conflict_atlas.md` | ready | motivates behavior constraints |
| Main residual diagnostic figure exists | `docs/paper/ExpertGym_ICLR/figures/diagnostic_residual_field.pdf`, `docs/report/RCRF/20260523_iclr_main_diagnostic_figure.md` | ready | Figure 1 / mechanism evidence |
| Pairwise-zero diagnostics show expert deletion removes mixed residual roles | `docs/paper/ExpertGym_ICLR/figures/pairwise_zero_diagnostics.pdf`, `docs/report/RCRF/20260523_pairwise_zero_diagnostics.md`, `docs/report/RCRF/20260523_iclr_pairwise_zero_figure.md` | ready | diagnostic figure / scalar negative control |
| Heldout/probe protocol is specified | `docs/paper/ExpertGym_ICLR/HELDOUT_PROTOCOL.md`, heldout paragraph in `main.tex` | ready | methods / experimental protocol |
| Paper-main method config is frozen | `docs/paper/ExpertGym_ICLR/PAPER_MAIN_METHOD_CONFIG.md` | ready | reproducibility / method details |
| Full Eval6 static baseline table exists | `docs/evaluation/20260518_baselines_eval6.md`, `docs/evaluation/20260517_p0_static_baselines_eval6.md`, `MAIN_BENCHMARK_TABLE_DRAFT.md` | ready | baseline rows for main table |
| Paper-main RCF-BC full Eval6 queue is frozen and complete | `docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL_QUEUE.md`, `docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL_RUN_STATUS.md`, `skill/command/run_20260523_iclr_paper_main_eval.sh` | ready | reproducibility / selected ablation queue |
| Paper-main full Eval6 aggregation path exists | `scripts/analysis/aggregate_iclr_paper_main_eval.py`, `docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md` | ready | converts finished queue logs into table rows |
| RCF-BC full Eval6 result exists | `PAPER_MAIN_EVAL6_AGGREGATE.md` has ready BCRC, no-behavior, and hard-behavior rows; `PAPER_MAIN_EVAL6_INTERIM_ANALYSIS.md` records that it is not a SOTA row | ready | ablation table / benchmark sanity check |
| Scalar code shrinkage improves Memory but damages Code | `docs/report/RCRF/20260522_rcf_bc_paper_evidence_table.md`, v14/v15 | ready | mechanism table |
| Hard routing protects behavior but loses continuous Code signal | `docs/report/RCRF/20260522_rcf_bc_paper_evidence_table.md`, v12/v13/v16/v17/v19 | ready | mechanism table |
| Continuous residual field + behavior constraints gives a better Pareto operating point | `docs/report/RCRF/20260522_rcf_bc_paper_evidence_table.md`, v18; `PAPER_MAIN_EVAL6_AGGREGATE.md` | ready, narrow | paper-main as trade-off-control claim, not SOTA |
| Residual row effects are non-additive | `docs/report/RCRF/20260522_counterfactual_residual_effects.md` | ready | analysis section |
| BCRC generalizes RAIN/RAM principle to multi-agent deltas | current RAIN diagnosis + RCF-BC ledger; RAM artifact still missing | partial | discussion, not main empirical claim yet |

## Required Before Submission

### E1 Main Benchmark

Need one consolidated table with:

- TA / average;
- TIES / DARE or available static baselines;
- RAM / RAIN-inspired preservation if local artifacts exist;
- `v18_rcf_bc`;
- two principled ablations: no behavior constraint and hard behavior constraint.

Required metrics:

```text
Tool BFCL mean
ToolRL-80 accuracy
Memory eval F1
Code pass@1
Code BoN
Average
Worst-task regression
```

Current draft artifact:

```text
docs/paper/ExpertGym_ICLR/MAIN_BENCHMARK_TABLE_DRAFT.md
```

Frozen execution queue:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL_QUEUE.md
```

Current queue aggregate:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md
```

Minimum candidates:

```text
bcrc_v18_alias_v9
no_behavior_v1_code_only
hard_behavior_v8
```

Status: `ready` for the selected BCRC-family queue.  The rows do not support a SOTA claim: soft behavior constraints have the best Code pass@1 and worst-task score within the queue, while no-behavior has the best simple average.

### E2 Mechanism Table

Can be written now from existing evidence:

```text
v8 hard constraints
v18 soft constraints
v19 strict cleanup
v14/v15 scalar code shrink
v20-v23 row-level interventions
```

Status: `ready`.

### E3 Vector Semantics / Norm Audit

Can be written now:

```text
RAIN vs ExpertGym anchor/vector semantics
Tool/Memory/Code/R1 norm mismatch
```

Status: `ready`.

### E4 Generalization / Heldout

Protocol artifact:

```text
docs/paper/ExpertGym_ICLR/HELDOUT_PROTOCOL.md
```

Status: protocol `ready`; selected full Eval6 queue `ready`; broader heldout generalization beyond the selected queue remains `not ready`.

## Decision Rule

The current paper should be framed as:

```text
Mechanism-first agent task-vector composition with strong diagnostic evidence and a candidate operating point.
```

It should not yet claim:

```text
SOTA across all Tool/Memory/Code benchmarks.
```
