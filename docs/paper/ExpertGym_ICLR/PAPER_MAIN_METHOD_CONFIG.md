# Paper-Main Method Config: BCRC / RCF-BC

Date: 2026-05-23

This file freezes the exact paper-main method configuration for the current ICLR draft.  It is intended to make the main method reproducible without relying on conversation memory.

## 1. Method Identity

Paper name:

```text
BCRC: Behavior-Constrained Residual Composition
```

Internal lineage:

```text
RCF-BC / v18
```

Evaluated checkpoint id:

```text
bcrc_v18_alias_v9
```

Important artifact note:

```text
v18 is the semantic paper-main name.
v9 is the existing baked checkpoint lineage.
The 588 gate coefficients in v18 and v9 are numerically identical.
```

Verified equality:

```text
v18_v9_gates_equal = True
num_gate_keys = 588
```

## 2. Reproducibility Pointers

Repository root:

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
```

Mode manifest:

```text
/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
```

Main method gate file:

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/residual_capability_field_behavior_constraints_v18/gates.json
```

Gate SHA256:

```text
fadf4ebf1bb6d8335285b00e26fef1982b141faece785a2fea58dd48710f2f0b
```

Evaluated alias gate file:

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/gates.json
```

Alias gate SHA256:

```text
57d05513e8e1c4cfba2f9b0c3f66ee06e2ee8b31e19e6fb745f33fd49d6595c6
```

Evaluated baked checkpoint:

```text
/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9
```

Named v18 baked checkpoint status:

```text
/tmp/shared-storage/OnPolicy/checkpoints/residual_capability_field_behavior_constraints_v18
```

This named checkpoint is currently absent.  Because the v18 and v9 gate coefficients are identical, the paper-main Eval6 queue evaluates the existing v9 baked checkpoint under the explicit id `bcrc_v18_alias_v9`.

## 3. Residual Granularity

BCRC acts on residual entries:

```text
(expert, layer, linear module)
```

Experts:

```text
tool, memory, code
```

Layers:

```text
28 transformer layers
```

Linear modules per layer:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

Total train-free gate entries:

```text
3 experts * 28 layers * 7 modules = 588 residual coefficients
```

The method does not learn a global expert scalar.  It edits a residual coefficient field with behavior constraints.

## 4. Evidence Inputs

Base gate:

```text
/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json
```

Base gate SHA256:

```text
c892653ecb791d1d1a45cbf76a8d88d1b9f5a4a47ba63ccbcab16533e9bd1a59
```

Code contrast summaries:

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livebench_prompt_alllayers_s16_20260521/contrast_module_summary.jsonl
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livebench_reasoning_alllayers_s16_20260521/contrast_module_summary.jsonl
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livecodebench_code_alllayers_s16_20260521/contrast_module_summary.jsonl
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/livecodebench_prompt_alllayers_s16_20260521/contrast_module_summary.jsonl
```

Tool / Memory behavior summaries:

```text
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/tool_memory_positive_signature_s32_20260521/signed_utility_summary.json
/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/memory_fulltraj_positive_s32_20260521/signed_utility_summary.json
```

Interpretation:

- Code supplies pass/fail residual contrast over prompt, reasoning, and final-code spans.
- Tool and Memory supply protected behavior spans, not ordinary scalar reward targets.
- The method uses behavior support and harm-veto evidence to constrain where Code contrast is allowed to move the residual field.

## 4.1 Paper Scoring Primitive

The ICLR draft now states the concrete scoring primitive used by the evidence ledger.  For each residual entry `(expert, layer, module)` and labeled behavior span:

```text
u = DeltaW h
expression = mean ||u||^2
signed_effect = mean[-<grad, u>]
```

Code capability is computed as same-prompt outcome contrast:

```text
code_utility = signed_effect(pass trajectory) - signed_effect(fail trajectory)
```

Tool and Memory use the same signed-effect score on protected spans:

```text
behavior_harm = max(-signed_effect, 0)
behavior_support = max(signed_effect, 0)
```

The gate update is a bounded signed utility update, followed by behavior constraints:

```text
positive update + high protected-behavior harm -> shrink
negative update + high protected-behavior support -> floor at base gate
weak / source-conflicted evidence -> no update
```

The local harmful-mask ablations are part of the paper evidence:

| variant | operation | read |
| --- | --- | --- |
| `v20` | halve 60 `code_negative_noise + weak_or_uninformative` code rows | local shrink recovers most Memory gain but still hurts LiveBench and some Code BoN |
| `v21` | zero the same 60 rows | hard local mask gives smaller Memory gain and still hurts LiveCodeBench |
| `v22` | halve only 15 `code_negative_noise` rows | not a safe pruning target by itself |
| `v23` | halve only 45 weak rows | weak evidence can still carry source-specific Code ability |

Conclusion: the main method should not delete every row that looks harmful under one diagnostic source.  Hard masking is reserved only for high-confidence harm: source-consistent negative evidence, active target-span expression, and no protected Tool/Memory support.  Current BCRC/v18 therefore keeps soft behavior constraints as the main operating point.

## 5. Frozen Gate-Building Parameters

The exact gate builder command is defined in:

```text
skill/command/run_20260522_rcrf_pareto_frontier.sh
```

Candidate:

```bash
PHASE=generate CANDIDATES=v18_rcf_bc bash skill/command/run_20260522_rcrf_pareto_frontier.sh
```

Core parameters:

```text
normalization = per-file
scale_quantile = 0.9
max_delta = 0.05
min_abs_score = 0.1
aggregation = conservative
conflict_penalty = 0.35
min_coeff = 0.55
max_coeff = 1.12
preserve_task = tool,memory
preserve_min_normalized_utility = 0.4
preserve_min_positive_fraction = 0.5
preserve_negative_scale = 0.0
harm_veto_task = tool,memory
harm_veto_min_normalized_harm = 0.4
harm_veto_positive_scale = 0.5
harm_veto_positive_scale_mode = constant
recenter_passes = 3
```

## 6. Frozen Gate Statistics

Coefficient summary:

| expert | count | mean | min | max | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| tool | 196 | 1.0042 | 0.6582 | 1.1200 | 0.1240 |
| memory | 196 | 0.9874 | 0.6680 | 1.1200 | 0.1065 |
| code | 196 | 0.9007 | 0.6577 | 1.1117 | 0.1281 |

Delta summary relative to the base gate:

| expert | changed | positive | negative | mean abs delta | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| tool | 60 | 33 | 27 | 0.001725 | 0.018750 |
| memory | 64 | 34 | 30 | 0.002936 | 0.029167 |
| code | 81 | 39 | 42 | 0.003255 | 0.020748 |
| overall | 205 | 106 | 99 | 0.002639 | 0.029167 |

Decision counts:

| reason | rows |
| --- | ---: |
| below_min_abs_score | 204 |
| preserve_utility_floor | 200 |
| pass_fail_negative | 92 |
| behavior_harm_veto | 78 |
| pass_fail_positive | 14 |

Expert-specific decision counts:

| expert | behavior harm veto | below min abs score | pass/fail negative | pass/fail positive | preserve utility floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| tool | 25 | 80 | 27 | 3 | 61 |
| memory | 20 | 41 | 30 | 5 | 100 |
| code | 33 | 83 | 35 | 6 | 39 |

## 7. Bake Commands

Semantic v18 bake command:

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --gate-checkpoint /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/residual_capability_field_behavior_constraints_v18/gates.json \
  --output /tmp/shared-storage/OnPolicy/checkpoints/residual_capability_field_behavior_constraints_v18
```

Current evaluated alias bake command:

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --gate-checkpoint /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/gates.json \
  --output /tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9
```

## 8. Paper-Main Eval6 Queue

Frozen queue:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL_QUEUE.md
skill/command/run_20260523_iclr_paper_main_eval.sh
```

Minimum candidates:

```text
bcrc_v18_alias_v9
no_behavior_v1_code_only
hard_behavior_v8
```

Current full Eval6 aggregate:

```text
docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md
```

Aggregation command:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/aggregate_iclr_paper_main_eval.py
```

## 9. Paper Use Rule

Use this configuration as the paper-main method only under the following wording:

```text
BCRC/RCF-BC is a mechanism-derived residual composition rule.  It is not a tuned GRPO gate checkpoint and not a global coefficient sweep.
```

Allowed claim with the completed selected Eval6 queue:

```text
BCRC exposes and operationalizes residual-level behavior constraints.  Within the selected BCRC-family Eval6 queue, the soft behavior-constrained field gives the highest Code pass@1 and worst-task score, while the no-behavior ablation gives the highest simple average.
```

Not allowed from the current evidence:

```text
BCRC is SOTA across Tool / Memory / Code.
```
