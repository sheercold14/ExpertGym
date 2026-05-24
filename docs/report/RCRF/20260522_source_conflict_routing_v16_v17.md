# 2026-05-22 Source-Conditioned Conflict Routing v16/v17

## Motivation

v13 positive-only role routing preserved Tool/Memory but lost Code. The atlas showed the largest unresolved region is:

```text
code_source_conflict + code_source_conflict_with_behavior = 279 / 588 residual rows
```

v9, the current best balanced point, did not hold these rows. It changed 83 source-conflict rows:

- `code_source_conflict`: 50 changed, 6 positive, 44 negative.
- `code_source_conflict_with_behavior`: 33 changed, 10 positive, 23 negative.

This suggests Code repair is not only “raise clean positive residuals”. It also needs to suppress some source-conflict residuals whose pass/fail evidence points in the wrong direction.

## Method

Added optional source-conflict routing to `build_rcrf_role_routed_gates.py`. Defaults remain unchanged:

```text
--source-conflict-action hold
```

New actions:

- `suppress-dominant`: suppress a `code_source_conflict*` row only when negative source evidence dominates positive evidence.
- `route-dominant`: additionally raise rows where positive source evidence dominates negative evidence.

Dominance rule:

```text
dominant_strength >= 1.25 * opposing_strength
dominant_strength >= 1.0
```

Behavior protection:

```text
--source-conflict-protected-support-action hold
```

So negative suppression does not lower rows that also support Tool/Memory behavior.

## Variants

| variant | rule | changed | positive | negative |
|---|---|---:|---:|---:|
| v16 | v13 + negative-dominant source-conflict suppression | 112 | 73 | 39 |
| v17 | v16 + positive-dominant source-conflict raise | 146 | 107 | 39 |

Artifacts:

- v16 gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_source_conflict_suppress_v16/gates.json`
- v17 gate: `/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_source_conflict_route_v17/gates.json`
- v16 checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_source_conflict_suppress_v16`
- v17 checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_source_conflict_route_v17`

## Results

| model | Tool parallel | Tool parallel_multi | Tool live_parallel | Tool live_parallel_multi | Memory eval_50 F1 | LiveBench hurt acc / BoN | LiveCodeBench hurt acc / BoN |
|---|---:|---:|---:|---:|---:|---:|---:|
| v9 | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7575 | 0.2500 / 0.6250 | 0.6250 / 0.6555 |
| v13 positive-only | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7564 | 0.1250 / 0.2422 | 0.3125 / 0.5294 |
| v16 source suppress | 0.8800 | 0.8600 | 0.8125 | 0.6250 | 0.7660 | 0.1094 / 0.2500 | 0.3281 / 0.4375 |
| v17 source route | 0.8800 | 0.8550 | 0.8125 | 0.6250 | 0.7654 | 0.1250 / 0.3125 | 0.2188 / 0.4375 |

## Interpretation

1. Source-conflict routing is behavior-safe. Tool remains stable and Memory improves over v13.
2. Negative-dominant suppression helps some LiveCodeBench acc but does not recover v9.
3. Positive-dominant conflict raise slightly helps LiveBench BoN but hurts LiveCodeBench acc.
4. The discrete role/dominance rule is still too coarse. v9's strength likely comes from the continuous pass/fail overlay over many row types, including `code_negative_noise`, `uninformative`, and source-conflict rows, plus expert-mean recentering.

## Mechanistic Update

The current evidence supports this refined claim:

> Source conflict rows are not all harmful, but they cannot be routed by a single dominant-source rule. Code repair needs a continuous residual evidence field, while Tool/Memory behavior protection should act as a constraint or veto.

This pushes RCRF away from hard role routing and toward:

```text
continuous Code pass/fail overlay
+ Tool hard behavior constraint
+ Memory soft trajectory constraint
+ source/span audit table for interpretability
```

In paper terms, v16/v17 are useful ablations: they show that a readable but over-discretized atlas rule preserves behavior, but loses Code. The best current method remains v9-style continuous evidence routing.
