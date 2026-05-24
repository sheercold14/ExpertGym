# ICLR Submission Readiness Audit

This audit tracks whether the current mechanism-first ExpertGym draft satisfies the real paper goal:

```text
diagnose first -> derive a simple first-principles algorithm -> validate with reproducible experiments -> write an ICLR-style paper
```

## Current Status

| requirement | status | evidence | remaining work |
|---|---|---|---|
| ICLR template draft exists | ready | `docs/paper/ExpertGym_ICLR/main.tex`, `main.pdf` | none |
| Method is simple and first-principles | ready | BCRC / RCF-BC algorithm box in `main.tex`; exact config in `PAPER_MAIN_METHOD_CONFIG.md` | keep wording aligned with mechanism claim |
| Diagnostic experiments precede algorithm | ready | `docs/report/RCRF/20260523_8765_frontend_diagnostic_method_and_findings.md`, `DIAGNOSIS_TO_METHOD_BRIDGE.md`, `figures/diagnostic_residual_field.pdf`, `figures/pairwise_zero_diagnostics.pdf` | none |
| Main diagnostic figures are paper-ready | ready | `figures/diagnostic_residual_field.pdf`, `figures/pairwise_zero_diagnostics.pdf`, `docs/report/RCRF/20260523_iclr_main_diagnostic_figure.md`, `docs/report/RCRF/20260523_iclr_pairwise_zero_figure.md` | none |
| RAIN vs ExpertGym task-vector semantics are audited | ready | `docs/report/RCRF/20260523_rain_expertgym_task_vector_diagnosis.md` | add RAM artifact if available |
| Residual-level evidence ledger exists | ready | `docs/report/RCRF/20260522_rcrf_attribution_ledger.md`, conflict atlas reports | ensure final paper uses one consistent ledger version |
| Code source/span conflict is documented | ready | 8765 report, `source_conflict_pairs.csv` | add figure/table to main paper if space permits |
| Main method has interpretable ablations | ready for mechanism claim | `docs/report/RCRF/20260522_rcf_bc_paper_evidence_table.md` | select only 3-5 ablations for paper |
| Broad benchmark/SOTA claim is supported | not ready | `PAPER_MAIN_EVAL6_AGGREGATE.md` has ready BCRC, no-behavior, and hard-behavior rows; the completed rows do not support SOTA | keep claim narrowed to mechanism and trade-off control |
| Heldout protocol is specified | ready | `HELDOUT_PROTOCOL.md`, heldout paragraph in `main.tex` | none for selected queue |
| Heldout generalization is supported | partial | protocol exists; selected paper-main Eval6 queue is complete, but broad benchmark superiority is not supported | separate selected-queue evidence from broad generalization |
| Paper avoids overclaiming | ready | `EXPERIMENT_EVIDENCE_MAP.md` | keep abstract/conclusion aligned after final results |
| Diagnostic claims have benchmark-safe boundaries | ready | `DIAGNOSTIC_CLAIMS_TABLE.md` | update only if final Eval6 changes the claim boundary |
| Paper-main full Eval6 queue is frozen and complete | ready | `PAPER_MAIN_EVAL_QUEUE.md`, `PAPER_MAIN_EVAL_RUN_STATUS.md`, `skill/command/run_20260523_iclr_paper_main_eval.sh` | none for selected queue |
| Paper-main Eval6 aggregator exists | ready | `scripts/analysis/aggregate_iclr_paper_main_eval.py`, `PAPER_MAIN_EVAL6_AGGREGATE.md` | rerun only if more candidates are added |

## Submission-Critical Gaps

### G1. Consolidated Main Benchmark Table

The current paper has a generated draft table, but not a final main benchmark table. The draft is:

```text
docs/paper/ExpertGym_ICLR/MAIN_BENCHMARK_TABLE_DRAFT.md
```

It currently proves:

- static baselines and prior best models have comparable full Eval6 rows;
- RCF-BC has strong mechanism rows;
- the paper-main RCF-BC queue now has complete BCRC, no-behavior, and hard-behavior rows.

Before submission, finalize one table with:

```text
TA / average
TIES / DARE or available static baselines
RCF-BC / BCRC main candidate
no behavior constraint ablation
hard behavior constraint ablation
```

Required metrics:

```text
Tool BFCL mean
ToolRL-80
Memory F1
Code pass@1
Code BoN
Average
Worst-task regression
```

The minimum queue is now frozen in:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL_QUEUE.md
```

Reproducible entry point:

```bash
DRY_RUN=1 PHASE=list bash skill/command/run_20260523_iclr_paper_main_eval.sh
```

Aggregation entry point:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/aggregate_iclr_paper_main_eval.py
```

Current aggregate artifact:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md
```

It currently marks the minimum candidates as `ready`: BCRC, no-behavior, and hard-behavior all have Tool, Memory, and Code legs.

Minimum candidates:

```text
bcrc_v18_alias_v9
no_behavior_v1_code_only
hard_behavior_v8
```

Important artifact note: `bcrc_v18_alias_v9` evaluates the existing baked checkpoint `rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9` because the named `v18` baked checkpoint is absent; the `v18` and `v9` gate coefficients are numerically identical.

### G2. Exact Paper-Main Algorithm Config

The paper-main algorithm config is now frozen in:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_METHOD_CONFIG.md
```

It specifies:

```text
residual granularity
behavior probe set
utility score
harm/support threshold or continuous scale
base gate
changed-row count
checkpoint path
```

It is tied to one semantic gate file, one evaluated alias gate file, one baked checkpoint path, and the exact bake commands.  The selected full Eval6 queue has now been executed; the remaining submission issue is claim framing and final table curation, not method specification.

### G3. Heldout Protocol

A mechanism paper still needs evidence that the diagnostic probes are not merely answer memorization. The heldout protocol should state:

```text
calibration/probe size
probe source
heldout source
which eval samples, if any, are diagnostic-only
```

The protocol is now specified in:

```text
docs/paper/ExpertGym_ICLR/HELDOUT_PROTOCOL.md
```

It separates `probe`, `monitor`, `guard`, `formal_eval`, and `diagnostic_eval_leak`.  The selected full Eval6 queue has been executed under the frozen harness; broader heldout generalization beyond this selected queue remains a future empirical item.

### G4. RAM Connection

The RAIN comparison is now concrete. RAM is still mostly conceptual. If local RAM artifacts are available, add:

```text
anchor model
delta definition
behavior-preservation signal
whether it uses protected behavior as anchor or additive vector
```

If RAM artifacts are not available, keep RAM in related work/discussion only.

## Next Best Work Order

1. Keep the paper claim narrowed: mechanism-first residual diagnosis plus trade-off control, not SOTA.
2. Decide whether to add ToolRL-80 to the final main table or keep it as an auxiliary stability check.
3. Add RAM artifacts to the vector-semantics audit if we want the RAIN/RAM discussion to become an empirical comparison rather than related-work framing.
4. Add RAM artifacts if available; otherwise keep RAM in related work/discussion only.
5. Prune older exploratory variants from the main paper and move them to appendix/report references.

## Current Paper Framing

Allowed claim:

```text
Agent task vectors are not task-pure. Their useful and harmful components are residual-level and span-conditioned. BCRC uses behavior probes to compose residuals under executable-behavior constraints.
```

Not yet allowed:

```text
BCRC is SOTA across all Tool/Memory/Code benchmarks.
```
