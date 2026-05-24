# Diagnostic Claims Table

This table distills the `http://127.0.0.1:8765` diagnostic workbench into paper-safe claims.  It is intentionally conservative: a claim is paper-main only when the evidence identifies a mechanism, not merely a tuned checkpoint.

## Core Diagnostic Unit

The analysis unit is:

```text
residual entry r = (expert, layer, module)
```

For each entry, the diagnostic primitive is:

```text
u_r = Delta W_r h
expression(r) = mean ||Delta W_r h||^2
signed_effect(r) = mean[-<grad, Delta W_r h>]
```

This asks whether a task vector component is expressed on a behavior span and whether that expression helps or hurts a teacher-forced behavior.

## Paper-Safe Claims

| claim | 8765 evidence | method implication | paper status |
| --- | --- | --- | --- |
| Expert task vectors are not task-pure directions | useful and harmful entries coexist inside the same expert delta | avoid expert-level scalar claims | ready |
| The actionable unit is residual-level, not expert-level | records are stable at `(expert, layer, module)` and expose mixed utility/harm | learn or route a residual coefficient field | ready |
| Ability is span-conditioned | Tool, Memory, and Code differ when measured on tool-call, memory-trajectory, prompt/reasoning/code spans | define behavior spans before assigning utility | ready |
| Code is not a smooth scalar direction | LiveBench and LiveCodeBench source/span evidence frequently have opposite signs | use outcome contrast and avoid uniform Code scaling | ready |
| Memory-Code conflict is residual-key level | memory residuals are strong on Memory spans but mixed on Code spans | protect Memory behavior at residual level | ready |
| Tool is primarily a behavior-format constraint | Tool scores are sensitive to call-format span, while broad scalar reward is noisy | protect tool-call behavior spans | ready |
| Hard routing is too coarse | strict cleanup protects some behavior but can erase low-confidence Code signal | prefer soft behavior constraints over all-or-nothing masks | ready |
| Scalar shrinkage is a useful negative control, not a method | shrinking or removing Code trades Code for Memory | use scalar baselines to show why residual routing is needed | ready |
| Pairwise-zero diagnosis exposes mixed residual roles | `TM(code=0)`, `TC(memory=0)`, and `MC(tool=0)` remove capability, behavior-support, and conflict rows at the same time | use zeroing only as a diagnostic negative control, not as the composition rule | ready |
| BCRC is a simple consequence of diagnostics | increase positive capability rows, soften rows that harm protected behavior, leave weak evidence unchanged | present as behavior-constrained residual composition | ready |
| BCRC is SOTA on the full benchmark | current BCRC full row is below TA-0.75 / historical TAME-style on Code and average | do not claim until full Eval6 supports it | not ready |

## What the Frontend Proves

The 8765 frontend supports this statement:

```text
Agent task-vector merging is governed by residual-level, span-conditioned utility and harm.
```

It does not by itself support this statement:

```text
The selected BCRC checkpoint is the best model across Tool, Memory, and Code.
```

The first statement is a mechanism claim and is already supported by the diagnostic records.  The second is a benchmark claim and depends on the full Eval6 table.

## Minimal Method Derived From the Diagnosis

The method should remain deliberately small:

1. compute residual response on task behavior spans;
2. identify positive capability entries with outcome contrast;
3. identify protected Tool/Memory behavior support and harm;
4. increase clean positive entries;
5. soften positive entries that hurt protected behavior;
6. keep weak or contradictory entries unchanged.

The pairwise-zero diagnostic figure adds an explicit sanity check: if zeroing an expert removes mixed residual roles, then a paper method should not be framed as deleting or scaling whole experts.  It should instead route residual entries with task-span evidence.

This gives a first-principles method:

```text
Behavior-Constrained Residual Composition
```

The important point for the paper is not that the rule has many knobs.  The important point is that the rule follows directly from the measured role of each residual entry.

## Current Benchmark Boundary

As of the current paper-main Eval6 status:

| candidate | status | consequence |
| --- | --- | --- |
| `bcrc_v18_alias_v9` | full Eval6 ready | usable as main method operating point, but not SOTA by itself |
| `no_behavior_v1_code_only` | full Eval6 ready | has the best simple average in the selected BCRC-family queue |
| `hard_behavior_v8` | full Eval6 ready | is competitive on Code TP/BoN but below soft constraints on Code pass@1 and worst-task score |

The completed queue narrows the paper claim.  Soft behavior constraints have the highest Code pass@1 and worst-task score inside the selected BCRC-family rows, but no-behavior has the highest simple average.  Therefore the paper should frame BCRC as an interpretable diagnostic-derived operating point, not as a final state-of-the-art benchmark result.
