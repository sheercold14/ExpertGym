# ICLR Main Diagnostic Figure

PDF: `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/paper/ExpertGym_ICLR/figures/diagnostic_residual_field.pdf`
PNG: `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/paper/ExpertGym_ICLR/figures/diagnostic_residual_field.png`

## What The Figure Shows

1. Code evidence is span/source-conditioned rather than one smooth direction.
2. Most residual rows are not clean task rows; they are conflict, behavior-related, or weak-evidence rows.
3. RCF-BC variants expose a Memory-Code trade-off that scalar coefficients hide.

## Top Code Span Conflicts

| left | right | pearson | conflict rate |
| --- | --- | ---: | ---: |
| LB_prompt | LCB_prompt | -0.9948 | 0.6395 |
| LB_prompt | LCB_code | -0.0145 | 0.5646 |
| LB_reasoning | LCB_prompt | -0.0543 | 0.5646 |
| LB_code | LCB_prompt | -0.0144 | 0.5272 |
| LB_code | LCB_code | -0.0382 | 0.5102 |

## Residual Role Counts

| role | rows | fraction |
| --- | ---: | ---: |
| code_source_conflict | 279 | 0.4745 |
| weak_or_uninformative | 78 | 0.1327 |
| clean_code_repair | 77 | 0.1310 |
| code_negative_with_behavior_support | 58 | 0.0986 |
| code_negative_noise | 56 | 0.0952 |
| code_repair_with_behavior_harm | 28 | 0.0476 |
| behavior_only | 12 | 0.0204 |

## Trade-off Rows

| variant | Tool | Memory F1 | Code BoN | Code Acc |
| --- | ---: | ---: | ---: | ---: |
| v8 | 0.7944 | 0.7720 | 0.2812 | 0.1953 |
| v18 | 0.7931 | 0.7575 | 0.4375 | 0.2344 |
| v19 | 0.7956 | 0.7793 | 0.3125 | 0.2266 |
| v14 | 0.7944 | 0.7774 | 0.2188 | 0.2266 |
| v15 | 0.7800 | 0.7841 | 0.1562 | 0.1250 |

## Paper Claim Supported

This figure supports the mechanism claim that agent task vectors should be composed at residual-entry granularity under behavior constraints, not by a single expert-level scalar.
