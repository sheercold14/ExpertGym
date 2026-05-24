# 2026-05-22 RCF-BC Counterfactual Residual Effects

## Purpose

This report converts RCF-BC ablations into counterfactual effect estimates. The goal is to separate two quantities that are easy to conflate:

- how many residual rows were directly intervened on;
- how Tool, Memory, and Code metrics moved relative to `v18_rcf_bc`.

This is the evidence layer needed for a general capability-attribution framework: row labels are hypotheses, while counterfactual metric deltas are the behavioral evidence.

## Main Effects

| candidate | direct rows | intervention | dTool | dTool live | dMemory F1 | dLB BoN | dLCB BoN | read |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `v14` | 196 | all code expert coefficients * 0.5 | 0.0013 | 0.0000 | 0.0198 | 0.0000 | -0.4375 | diagnostic trade-off |
| `v15` | 196 | all code expert coefficients = 0 | -0.0131 | -0.0625 | 0.0266 | -0.1250 | -0.4375 | memory/code trade-off from coarse code suppression |
| `v20` | 60 | code_negative_noise + weak_or_uninformative code rows * 0.5 | 0.0013 | 0.0000 | 0.0197 | -0.1250 | -0.1250 | localized memory gain but code evidence is mixed |
| `v21` | 60 | code_negative_noise + weak_or_uninformative code rows = 0 | 0.0000 | 0.0000 | 0.0089 | 0.1250 | -0.3750 | diagnostic trade-off |
| `v22` | 15 | only code_negative_noise code rows * 0.5 | -0.0144 | -0.0625 | -0.0075 | 0.0625 | -0.1875 | code-capable rows also support behavior; unsafe to prune |
| `v23` | 45 | only weak_or_uninformative code rows * 0.5 | -0.0144 | -0.0625 | -0.0134 | -0.0625 | -0.0625 | diagnostic trade-off |

## Row-Normalized Effects

These values are metric delta per 10 directly changed rows. They are not causal constants; they flag which interventions are disproportionately risky.

| candidate | rows | dMemory/10 | dTool live/10 | dLB BoN/10 | dLCB BoN/10 |
|---|---:|---:|---:|---:|---:|
| `v14` | 196 | 0.0010 | 0.0000 | 0.0000 | -0.0223 |
| `v15` | 196 | 0.0014 | -0.0032 | -0.0064 | -0.0223 |
| `v20` | 60 | 0.0033 | 0.0000 | -0.0208 | -0.0208 |
| `v21` | 60 | 0.0015 | 0.0000 | 0.0208 | -0.0625 |
| `v22` | 15 | -0.0050 | -0.0417 | 0.0417 | -0.1250 |
| `v23` | 45 | -0.0030 | -0.0139 | -0.0139 | -0.0139 |

## Non-Additivity

| interaction | combined | parts | dTool live interaction | dMemory interaction | dLB BoN interaction | dLCB BoN interaction | read |
|---|---|---|---:|---:|---:|---:|---|
| `v20_minus_v22_plus_v23` | `v20` | `v22+v23` | 0.1250 | 0.0407 | -0.1250 | 0.1250 | combined intervention has strong positive behavior interaction; row effects are not additive |
| `v21_minus_v22_plus_v23` | `v21` | `v22+v23` | 0.1250 | 0.0299 | 0.1250 | -0.1250 | combined intervention has strong positive behavior interaction; row effects are not additive |

## Takeaways

- v20 directly changes only 60 code rows but recovers nearly the same Memory F1 as global code-half, so Memory harm is localized but not safely separable yet.
- v22 and v23 individually reduce Tool live_parallel and Memory F1 while preserving or improving parts of Code, so these rows are not disposable noise; they are capability/behavior coupling points.
- The v20-v22-v23 non-additivity shows residual rows interact: a row group's effect cannot be inferred by a hard archetype label alone.
- The next method should keep a continuous residual field and add behavior-support constraints for low-evidence rows, instead of hard pruning.

## Artifacts

- CSV: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_counterfactual_effects_20260522/counterfactual_effect_rows.csv`
- JSON: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_counterfactual_effects_20260522/counterfactual_effect_summary.json`
- Source paper table: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/rcrf_paper_evidence_table_20260522/rcrf_paper_evidence_table.csv`
