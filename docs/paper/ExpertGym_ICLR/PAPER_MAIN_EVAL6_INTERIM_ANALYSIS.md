# Paper-Main Eval6 Analysis

Last updated: 2026-05-23 15:50:29 +0800

This note records the completed paper-main Eval6 rows for the selected BCRC queue.  It is intentionally conservative: the full rows support a residual-level diagnostic / trade-off-control claim, not a broad SOTA claim.

## Completed Queue

The minimum paper-main queue is complete:

```text
bcrc_v18_alias_v9
no_behavior_v1_code_only
hard_behavior_v8
```

Aggregate artifact:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md
```

## Full Eval6 Rows

| candidate | role | Tool | Tool live | Memory F1 | Code Acc | Code TP | Code BoN | Avg(T/M/C) | Worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bcrc_v18_alias_v9` | soft behavior constraint | 0.7931 | 0.7188 | 0.7570 | 0.3301 | 0.4373 | 0.3939 | 0.6267 | 0.3301 |
| `no_behavior_v1_code_only` | no behavior constraint | 0.7956 | 0.7188 | 0.7650 | 0.3260 | 0.4355 | 0.4076 | 0.6289 | 0.3260 |
| `hard_behavior_v8` | hard behavior constraint | 0.7919 | 0.7188 | 0.7568 | 0.3274 | 0.4381 | 0.4047 | 0.6254 | 0.3274 |

## Code Details

| candidate | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | mean Acc | mean TP | mean BoN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bcrc_v18_alias_v9` | 0.3730 | 0.4640 | 0.4453 | 0.2872 | 0.4106 | 0.3425 | 0.3301 | 0.4373 | 0.3939 |
| `no_behavior_v1_code_only` | 0.3594 | 0.4537 | 0.4531 | 0.2926 | 0.4173 | 0.3620 | 0.3260 | 0.4355 | 0.4076 |
| `hard_behavior_v8` | 0.3574 | 0.4535 | 0.4375 | 0.2975 | 0.4227 | 0.3718 | 0.3274 | 0.4381 | 0.4047 |

## Baseline Context

From `MAIN_BENCHMARK_TABLE_DRAFT.md`:

| model | Tool | Memory F1 | Code Acc | Code BoN | Avg | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TA-0.75 | 0.7850 | 0.7587 | 0.3494 | 0.4173 | 0.6310 | 0.3494 |
| Historical best / TAME-style | 0.7954 | 0.7720 | 0.3597 | 0.4408 | 0.6424 | 0.3597 |
| RAM-Merged ARM-R-v2 | 0.7942 | 0.7361 | 0.3441 | 0.3812 | 0.6248 | 0.3441 |
| BCRC v18/v9 | 0.7931 | 0.7570 | 0.3301 | 0.3939 | 0.6267 | 0.3301 |

## Interpretation

The completed rows do not support a broad SOTA claim.  BCRC is below TA-0.75 and the historical TAME-style model on Code Acc, average score, and worst-task score.

The rows do support a narrower mechanism claim:

- Soft behavior constraints give the best Code pass@1 among the three BCRC-family rows (`0.3301` vs `0.3260` / `0.3274`).
- Soft behavior constraints also give the best worst-task score within the BCRC-family rows (`0.3301` vs `0.3260` / `0.3274`).
- No-behavior has the best average score and stronger Tool / Memory / Code-BoN in this small family, so the soft constraint is not a uniformly dominant operating point.
- Hard behavior is competitive on Code TP and BoN, but slightly worse than soft behavior on Code pass@1 and worst-task score.

This makes the paper story sharper:

```text
BCRC is not currently a SOTA benchmark row.
BCRC is an interpretable residual-level operating point derived from diagnostics.
The value is mechanism and trade-off control, not a tuned highest-score checkpoint.
```

## Claim Boundary

Allowed:

```text
Agent task vectors are not task-pure. Their useful and harmful components are residual-level and span-conditioned. BCRC uses behavior probes to compose residuals under executable-behavior constraints and yields an auditable operating point.
```

Allowed, with numbers:

```text
Within the selected BCRC-family Eval6 queue, the soft behavior-constrained field has the highest Code pass@1 and worst-task score, while no-behavior has the highest simple average.
```

Not allowed:

```text
BCRC is SOTA across Tool/Memory/Code.
Soft behavior constraints uniformly improve all metrics.
```

## Paper Use

Use this row as:

- a full benchmark sanity check for the mechanism-derived candidate;
- an ablation block comparing no behavior constraint, hard behavior constraint, and soft behavior constraint;
- evidence that the method should be framed as residual-level diagnosis plus trade-off control.

Do not use it as:

- a final SOTA table claim;
- proof that the current soft-veto hyperparameters are optimal;
- proof that behavior constraints improve every metric.
