# ICLR Main Benchmark Table Draft

This draft is generated from existing artifacts only. It does not run new evaluation.

## Comparable Full Eval6 Rows

These rows share the full Eval6-style Tool / Memory / Code protocol recorded in `docs/evaluation/20260518_baselines_eval6.md` and `docs/evaluation/20260517_p0_static_baselines_eval6.md`.

| model | type | Tool | Memory F1 | Code Acc | Code BoN | Avg(T/M/C) | Worst | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | Base | 0.7500 | 0.5288 | 0.2800 | 0.3304 | 0.5196 | 0.2800 | docs/evaluation/20260518_baselines_eval6.md |
| task-arithmetic-c033333 | TA-1/3 | 0.7848 | 0.6465 | 0.3409 | 0.4173 | 0.5907 | 0.3409 | docs/evaluation/20260518_baselines_eval6.md |
| ta-c075-global-20260517 | TA-0.75 | 0.7850 | 0.7587 | 0.3494 | 0.4173 | 0.6310 | 0.3494 | docs/evaluation/20260517_p0_static_baselines_eval6.md |
| RAM-Merged ARM-R-v2 | TA-0.75 | 0.7942 | 0.7361 | 0.3441 | 0.3812 | 0.6248 | 0.3441 | docs/evaluation/20260518_baselines_eval6.md |
| wudi-qwen7b-3expert | WUDI | 0.7823 | 0.6591 | 0.3304 | 0.4095 | 0.5906 | 0.3304 | docs/evaluation/20260518_baselines_eval6.md |
| ties-c033333-k02 | TIES | 0.7642 | 0.6359 | 0.3355 | 0.3880 | 0.5785 | 0.3355 | docs/evaluation/20260518_baselines_eval6.md |
| dare-ta-c033333-d08 | DARE-TA | 0.7952 | 0.6901 | 0.3365 | 0.3900 | 0.6073 | 0.3365 | docs/evaluation/20260518_baselines_eval6.md |
| dare-ties-c033333-k02-d08 | DARE-TIES | 0.7952 | 0.6891 | 0.3426 | 0.4007 | 0.6090 | 0.3426 | docs/evaluation/20260518_baselines_eval6.md |
| adamerging-taskwise-len1024 | AdaMerging | 0.7835 | 0.6678 | 0.3406 | 0.4242 | 0.5973 | 0.3406 | docs/evaluation/20260518_baselines_eval6.md |
| mixture-grpo-ta13-l96-step1 | Mixture GRPO | 0.7823 | 0.6643 | 0.3384 | 0.3782 | 0.5950 | 0.3384 | docs/evaluation/20260518_baselines_eval6.md |
| tame-cg-r1calib-global-v2 | Historical best / TAME-style | 0.7954 | 0.7720 | 0.3597 | 0.4408 | 0.6424 | 0.3597 | docs/evaluation/best_ever_model.md |

## Paper-Main BCRC-Family Full Eval6 Rows

The selected BCRC-family queue is complete.  These rows are comparable within the paper-main ablation block and bound the claim.

| model | type | Tool | Memory F1 | Code Acc | Code BoN | Avg(T/M/C) | Worst | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bcrc_v18_alias_v9 | main method: soft behavior-constrained residual field | 0.7931 | 0.7570 | 0.3301 | 0.3939 | 0.6267 | 0.3301 | docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md |
| no_behavior_v1_code_only | ablation: no behavior constraint | 0.7956 | 0.7650 | 0.3260 | 0.4076 | 0.6289 | 0.3260 | docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md |
| hard_behavior_v8 | ablation: hard behavior constraint | 0.7919 | 0.7568 | 0.3274 | 0.4047 | 0.6254 | 0.3274 | docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md |

## RCF-BC Mechanism Rows

These rows are mechanism evidence, not full Eval6 rows. Tool/Memory are quick metrics and Code is the code-hurt diagnostic subset.

| candidate | rule | changed | Tool quick | Mem eval50 | LB hurt acc | LB hurt BoN | LCB hurt acc | LCB hurt BoN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v18 | Main method name for v9: RCF-BC | 205.0000 | 0.7931 | 0.7575 | 0.1406 | 0.2500 | 0.3281 | 0.6250 |
| v8 | Code field + Tool/Memory hard behavior veto | 135.0000 | 0.7944 | 0.7720 | 0.0938 | 0.2500 | 0.2969 | 0.3125 |
| v19 | Ablation: strict archetype-consistency projection | 173.0000 | 0.7956 | 0.7793 | 0.1250 | 0.1875 | 0.3281 | 0.4375 |
| v14 | Negative control: v9 with code expert coefficients halved | 320.0000 | 0.7944 | 0.7774 | 0.2031 | 0.2500 | 0.2500 | 0.1875 |
| v15 | Negative control: v9 with code expert coefficients set to zero | 320.0000 | 0.7800 | 0.7841 | 0.0781 | 0.1250 | 0.1719 | 0.1875 |

## Claim Boundary / Open Items

The paper-main queue has the required BCRC, no-behavior, and hard-behavior rows.  The remaining issue is claim framing, not missing selected Eval6 rows.

| item | status | action |
| --- | --- | --- |
| Broad SOTA claim | not supported | Keep the claim narrowed to mechanism and trade-off control; BCRC is below TA-0.75 / TAME-style on Code and average score. |
| ToolRL-80 in main table | optional | Either omit ToolRL from the main table or report it as an auxiliary source-distribution stability check. |
| RAM artifacts | discussion-only unless added | Add RAM artifact rows only if the RAIN/RAM comparison becomes an empirical claim. |

## Recommended Paper-Main Evaluation Set

| role | candidate |
| --- | --- |
| main method | v18 / RCF-BC soft behavior constraints |
| no behavior constraint | v1 or explicit code-field-only gate |
| hard behavior constraint | v8 memoryfull hard veto or v19 strict cleanup |
| scalar negative control | v14 code-half and/or v15 code-zero |
| static baseline anchor | TA-0.75, DARE-TIES, AdaMerging taskwise, RAM-Merged ARM-R-v2 |
