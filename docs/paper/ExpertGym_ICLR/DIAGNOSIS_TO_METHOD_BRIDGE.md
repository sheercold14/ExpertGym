# From 8765 Diagnostics to BCRC

This note records the paper-facing bridge from the `http://127.0.0.1:8765` diagnostic workbench to the BCRC method.  It is meant to prevent the paper from reading like a tuned checkpoint story.  The method should be presented as a simple consequence of measured residual behavior.

## 1. First-Principles Question

The target problem is not:

```text
Which global coefficient should each expert receive?
```

The target problem is:

```text
For each expert residual, does it express useful behavior, harmful behavior, or no reliable signal on the behavior span where a task is evaluated?
```

This distinction matters because the ExpertGym deltas are not task-pure.  Tool, Memory, and Code residuals contain capability directions, formatting behavior, trajectory behavior, and cross-task interference in the same task vector.

## 2. Diagnostic Primitive

For a residual entry `r = (expert, layer, module)`, the workbench evaluates the response:

```text
u_r = Delta W_r h
```

and records:

```text
expression(r) = mean ||Delta W_r h||^2
signed_effect(r) = mean[-<grad, Delta W_r h>]
```

The first quantity asks whether a residual is active on the current span.  The second asks whether the active residual helps or hurts a teacher-forced behavior.  This separates large inactive deltas, useful active deltas, and harmful active deltas.

## 3. Span Choice

BCRC uses task-specific behavior spans:

| task | span | reason |
| --- | --- | --- |
| Tool | function-call / tool-call span | the observable behavior is structured tool invocation |
| Memory | update turns plus final answer | the task is a trajectory behavior, not just final QA |
| Code | prompt constraints, reasoning span, final code block | pass/fail depends on parsing, reasoning, and executable code |

Tool and Memory spans are treated as protected behavior evidence.  Code spans are treated as capability contrast evidence.

## 4. Outcome Contrast

For Code, the cleanest signal is same-prompt pass/fail contrast:

```text
code_utility(r) = signed_effect_r(pass trajectory) - signed_effect_r(fail trajectory)
```

This is more reliable than imitating an expert trajectory because it asks which residuals distinguish executable solutions from failed ones.  It also explains why a global Code coefficient can fail: only a small subset of Code residuals are clean repair directions.

## 5. Diagnostic Findings That Motivate the Method

Current 8765 evidence supports these findings:

| finding | implication |
| --- | --- |
| Clean Code repair residuals are sparse (`60/588`) | do not scale the entire Code expert uniformly |
| Shared positive residuals are rare (`17/588`) | synergy exists but should be selected, not assumed |
| Code source/span conflict is common (`167/588`) | Code needs span-conditioned evidence |
| Memory-Code is the dominant conflict pair | Memory should be protected at the residual level |
| MLP carries more conflict than attention | residual role matters more than expert identity alone |

The strongest Code warning is that LiveBench prompt evidence and LiveCodeBench prompt evidence can be almost oppositely signed.  Treating Code as one smooth scalar direction is therefore not justified.

## 6. BCRC Rule

BCRC is deliberately small:

1. propose a positive residual update when capability contrast is positive;
2. soften the update if the same residual harms protected Tool or Memory behavior;
3. avoid pruning residuals that support protected Tool or Memory behavior;
4. leave weak or contradictory evidence unchanged.

This yields a behavior-constrained residual coefficient field over:

```text
3 experts * 28 layers * 7 modules = 588 residual entries
```

## 7. Claim Boundary

The diagnostic evidence supports:

```text
Agent task vectors are residual-level, span-conditioned, and outcome-dependent.
BCRC operationalizes this structure with a simple behavior-constrained residual composition rule.
```

The diagnostic evidence alone does not support:

```text
BCRC is SOTA across Tool, Memory, and Code.
```

The selected full Eval6 rows in `PAPER_MAIN_EVAL6_AGGREGATE.md` are now ready, and they still do not support that stronger claim.  They support a narrower statement: BCRC is an interpretable residual-level operating point, while no-behavior and hard-behavior ablations expose the trade-off.

## 8. Paper-Writing Guidance

The main paper should emphasize:

- the unit shift from expert-level scalars to residual-level behavior evidence;
- the span shift from whole-response rewards to behavior spans;
- the objective shift from tuning coefficients to preserving useful behavior while adding capability repair residuals;
- the conservative design: if evidence is weak, do not move the residual.

This keeps the method simple enough to be credible and makes the experimental story auditable.
