# From 8765 Diagnostics to Success-Conditioned TRC

Date: 2026-05-23

Status: method-construction note. This is not the frozen paper-main BCRC
checkpoint. It records the cleaner geometry route suggested by the 8765
diagnostics and the earlier TRC hidden-residual experiments.

## 1. Goal

The method should not introduce hand-written capability atoms and should not
optimize against task priors such as "protect Tool" or "protect Memory".

The only units are:

- a successful behavior span `z`;
- a model site `s`, such as a linear-module output or residual-stream hidden
  state;
- a task-vector residual entry `r = (expert, layer, module)`.

Task names and expert names remain as data provenance. They are not the
optimization object.

## 2. Why BCRC Is Not the Final Geometry

The current BCRC bridge correctly moves the unit from expert-level scalars to
residual entries, but its rule is still typed by task:

```text
Code contrast may move a residual, unless Tool/Memory behavior support vetoes it.
```

That operating point was useful because 8765 showed that pairwise zeroing and
global expert scaling are unsafe. It is not the most fundamental method. The
more basic statement is:

```text
Successful trajectories define oriented hidden-residual directions. Residual
updates should be composed by their projection onto those directions.
```

This removes the need for protected-task labels in the rule. A residual is
kept, moved, or projected because of its geometry with respect to successful
behavior spans, not because it belongs to a named task.

## 3. Success-Conditioned Direction

Let `M0` be the init1 operating point. For a successful span `z` and site `s`,
define a success direction:

```text
b_{z,s} = phi_s(M_success(z), z) - phi_s(M0, z)
```

In the OP-VEC implementation, a first practical version uses the module-output
activation update:

```text
b_{z,s} = DeltaW_{source(z),s} h_{M0,z,s}
```

where `source(z)` is the task-vector delta that produced or witnessed the
successful span. This source label is only provenance. The optimizer receives
the vector `b_{z,s}`, not a capability atom.

For any residual entry `r`:

```text
u_{r,z,s} = DeltaW_{r,s} h_{M0,z,s}
```

The core measurement is the oriented projection:

```text
proj(r,z,s) = <u_{r,z,s}, b_{z,s}> / (||b_{z,s}||^2 + eps)
```

The signed decomposition is:

```text
support  = positive projection onto successful directions
conflict = negative projection onto successful directions
null     = residual update outside the observed success span
```

`null` must not be treated as noise. It is just unsupported by the current
success bank.

## 4. Relation to Existing Evidence

This frame unifies the two existing lines:

1. The 8765 workbench already measures residual responses on behavior spans:

```text
u_r = DeltaW_r h
expression(r) = mean ||DeltaW_r h||^2
signed_effect(r) = mean[-<grad, DeltaW_r h>]
```

2. TRC already optimizes hidden residual direction rather than scalar reward:

```text
r_target = h_expert - h_base
r_merge  = h_merge - h_base
loss     = 1 - cos(r_merge, r_target)
```

The failure mode of direct TRC is that `r_target` is a full expert residual.
The full expert residual can mix support, conflict, and null components. The
success-conditioned version keeps TRC's directional idea but replaces "imitate
the full expert residual" with "preserve projection onto successful behavior
directions and avoid stable anti-projection."

## 5. Conservative Composition Rule

Start from init1. Build a ledger over success spans:

```text
support_score(r)  = average_z,s positive projection mass
conflict_score(r) = average_z,s negative projection mass
null_score(r)     = average_z,s orthogonal mass
agreement(r)      = bootstrap or source agreement of the projection sign
```

Only the following decisions are allowed:

```text
stable support      -> preserve or allow small positive movement
stable conflict     -> project out the anti-aligned component with a small cap
mixed support/conflict -> keep unchanged unless a later span-conditioned router is used
mostly null / weak  -> keep unchanged
```

The coefficient-level objective can be written as a small trust-region problem:

```text
min_alpha
  sum_z [rho - sum_r alpha_r support_{r,z}]_+^2
  + lambda sum_z [sum_r alpha_r conflict_{r,z}]^2
  + eta ||alpha - alpha_init1||^2

subject to |alpha_r - alpha_init1_r| <= cap
```

The tensor-level version uses the existing activation projection idea. For a
candidate residual field `R_s = sum_r c_r u_{r,z,s}`, if:

```text
alpha_R(z,s) = <R_s, b_{z,s}> / (||b_{z,s}||^2 + eps) < 0
```

then remove only the stable negative component:

```text
R'_s = R_s - gamma * min(0, alpha_R) * b_{z,s}
```

and distribute the correction back to editable OP-VEC entries with a least-norm
split and a strict edit cap. This is a projection, not a shrink rule.

## 6. First Minimal Experiment

Use the existing activation projection probes as a no-training diagnostic pass:

- `/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/livebench_code_s8_20260523/activation_update_projection_summary.csv`
- `/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/livecodebench_code_s8_20260523/activation_update_projection_summary.csv`
- `/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/memory_fulltraj_s16_20260523/activation_update_projection_summary.csv`
- `/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/tool_memory_signature_s2_20260523/activation_update_projection_summary.csv`

Aggregate them with:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/build_success_projection_ledger.py \
  --projection-csv livebench_code=/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/livebench_code_s8_20260523/activation_update_projection_summary.csv \
  --projection-csv livecodebench_code=/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/livecodebench_code_s8_20260523/activation_update_projection_summary.csv \
  --projection-csv memory_fulltraj=/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/memory_fulltraj_s16_20260523/activation_update_projection_summary.csv \
  --projection-csv tool_memory_signature=/tmp/shared-storage/ExpertGym/activation_update_geometry/opvec4_rcrf_calibration_20260522/projection_probes/tool_memory_signature_s2_20260523/activation_update_projection_summary.csv \
  --output-dir /tmp/shared-storage/ExpertGym/activation_update_geometry/success_conditioned_ledger_20260523
```

This produces a success-conditioned ledger before any checkpoint edit. The
ledger should be used to answer three questions:

1. Do the previous `hold_conflict` rows become mixed support/conflict rows
   rather than simple negative rows?
2. Do pairwise-zero failures correspond to many stable-support rows being
   deleted?
3. Are there enough stable-conflict rows to justify a small projection edit
   from init1?

Only if those checks pass should the tensor projection script be adapted to use
success directions instead of owner/protected task filters.

Pilot result on 2026-05-23:

```text
num_input_rows = 2940
num_ledger_rows = 588
stable_support = 295
mixed_success_geometry = 55
mostly_null = 103
weak_or_unstable = 135
stable_conflict = 0
```

The important read is that no residual entry is a clean scalar-suppression
target once any strong support from a successful span is respected. The useful
set is the `mixed_success_geometry` set: rows that support one successful
direction while anti-aligning with another. This pushes the method toward
module-level projection decomposition, not shrink or pruning.

Current ledger artifact:

```text
/tmp/shared-storage/ExpertGym/activation_update_geometry/success_conditioned_ledger_20260523/success_projection_ledger.md
```

## 7. Paper-Safe Claims

Supported by the current evidence plus this geometry:

```text
Agent task vectors are not task-pure directions. Their useful composition unit
is a behavior-span-conditioned residual update.
```

```text
Successful trajectories induce local hidden-residual support directions. Expert
updates can be decomposed into support, conflict, and null components under the
activation metric.
```

```text
Projection should only edit stable anti-aligned components. Orthogonal and weak
components are not disposable.
```

Not supported yet:

```text
The success subspace is the true capability subspace.
Stable conflict is equivalent to failure causality.
This method is SOTA on the full benchmark.
```

## 8. Immediate Implementation Path

The repo already contains most primitives:

- `scripts/attention_pauh/probe_signed_utility.py` computes `DeltaW h` and
  can write activation projection summaries.
- `scripts/trc/train_trc_layer_gates.py` computes hidden residual directional
  alignment and projection floors.
- `scripts/analysis/build_activation_residual_projected_mode.py` already
  materializes tensor-level activation projection, but its selection logic is
  task-prior based.

The minimal implementation should therefore be:

1. build the success-conditioned ledger from existing projection probes;
2. compare the ledger against 8765 role tables and pairwise-zero diagnostics;
3. adapt `build_activation_residual_projected_mode.py` so candidate selection
   uses ledger roles, not owner/protected task filters;
4. generate a capped projection candidate from init1;
5. evaluate the subset loop first: Tool quick, Memory eval50, Code hurt subset;
6. only then promote to Eval6.

The first candidate should be projection-only:

```text
base = init1
editable rows = mixed_success_geometry only
no-op rows = stable_support, mostly_null, weak_or_unstable
edit = remove source-specific negative projection component
cap = small, e.g. |alpha| <= 0.05 or 0.10
```

This is deliberately not an enhancement objective. It is a conflict-removal
test. If the source-conditioned projection is correct, it should be more stable
than scalar shrink on the same mixed rows.

Minimal candidate set:

```text
success_project_init1_tm_guard_v0
  Project only components that anti-align with Tool/Memory successful spans.
  Replay the success ledger and reject if Tool/Memory support drops.

success_project_init1_mixed_all_v0
  Also allow Code-source projection, but only where LiveBench and LiveCodeBench
  agree in sign and the Tool/Memory replay guard passes.
```

Required negative controls:

```text
same_rows_scalar_shrink
shuffled_success_directions
random_mixed_rows_projection
```

The method claim is supported only if projection beats scalar shrink under the
same row set and cap. A higher benchmark number without this control would be a
tuned-checkpoint story, not mechanism evidence.
