# 8765 Diagnostic Protocol: From Residual Evidence to Paper Claims

This note distills the `http://127.0.0.1:8765` diagnostic workbench into a paper-facing protocol. The workbench should be cited as mechanism evidence, not as a training dashboard.

## What 8765 Proves

The central finding is:

```text
Agent task vectors are not task-pure capability directions.
Their useful and harmful components are residual-level, span-conditioned, and outcome-dependent.
```

The correct diagnostic unit is therefore:

```text
(expert, layer, module)
```

not a global expert coefficient such as `tool = 0.7`, `memory = 0.8`, or `code = 0.6`.

## Diagnostic Quantities

For each expert residual entry, 8765 computes the residual response:

```text
u = DeltaW h
```

and records two complementary scores:

| score | meaning | why it matters |
|---|---|---|
| `expression = mean ||DeltaW h||^2` | whether a delta is active on the current behavior span | parameter norm alone is misleading |
| `signed_effect = mean[-<grad, DeltaW h>]` | whether the active delta helps or hurts teacher-forced behavior | active residuals can have the wrong sign |

This separates three cases that scalar merging cannot distinguish:

1. large but inactive residuals;
2. active and useful residuals;
3. active but harmful residuals.

## Span Definitions

Each task must be diagnosed on the span where its behavior is expressed:

| task | diagnostic span | role in the method |
|---|---|---|
| Tool | tool-call / function-call span | protected behavior constraint |
| Memory | update turns + final answer trajectory | protected behavior constraint |
| Code | prompt constraints + reasoning + final code block | capability contrast signal |

The important design choice is that Tool and Memory are not reduced to ordinary scalar rewards. They define behavior anchors that should not be destroyed when Code repair residuals are added.

## Outcome Contrast

For Code, the clean signal is same-prompt pass/fail contrast:

```text
utility(entry) = signed_effect(pass trajectory) - signed_effect(fail trajectory)
```

This is stronger than imitating an expert trajectory because it asks which residuals separate executable solutions from non-executable ones. It also explains why Code cannot be fixed by simply increasing the code expert coefficient.

## Empirical Facts From 8765

Current workbench evidence:

| table | count | content |
|---|---:|---|
| residual records | 2016 | residual response / expression / signed effect |
| gate records | 2352 | how candidate gates move each residual entry |
| interference records | 504 | pairwise expert conflict and cross-task harm |
| eval records | 73 | Tool / Memory / Code quick and diagnostic results |

Key residual atlas facts over `588 = 3 experts * 28 layers * 7 modules` entries:

| finding | evidence |
|---|---|
| clean Code repair residuals are sparse | only `60/588` entries are `code_repair_only` |
| true cross-task synergy is rare | only `17/588` entries are `shared_positive` |
| Code is source/span-conditioned | `167/588` entries are `code_source_conflict_with_behavior` |
| Memory-Code is the dominant conflict | code-memory conflict is much stronger than memory-tool conflict |
| MLP carries more conflict than attention | MLP has more `code_source_conflict_with_behavior`; attention has more weak/noisy evidence |

The strongest Code diagnostic result is that LiveBench prompt-span evidence and LiveCodeBench prompt-span evidence have Pearson correlation about `-0.995` with a sign-conflict rate of `63.95%`. This rules out treating Code as one smooth direction.

## Method Implication

The method should be framed as:

```text
behavior-constrained residual composition
```

with a deliberately small rule:

1. if capability evidence is positive, propose a positive residual update;
2. if the row harms protected Tool/Memory behavior, soften the update;
3. if the row supports protected behavior, avoid pruning it below the base gate;
4. if evidence is weak or contradictory, leave the row unchanged.

This converts black-box coefficient search into an auditable residual evidence ledger.

## Paper Claim Boundary

Supported claim:

```text
ExpertGym exposes a residual-level, span-conditioned structure in reinforced-agent task vectors, and BCRC operationalizes this structure through behavior-constrained residual composition.
```

Not supported by the completed selected Eval6 queue:

```text
BCRC is SOTA across Tool, Memory, and Code.
```

The current paper should therefore emphasize mechanism, diagnostics, and an interpretable Pareto operating point. `PAPER_MAIN_EVAL6_AGGREGATE.md` now has ready rows for the main method and ablations; those rows still do not justify a broad SOTA claim.

## Artifact Pointers

- Mechanism report: `docs/report/RCRF/20260523_8765_frontend_diagnostic_method_and_findings.md`
- Conflict atlas: `docs/report/RCRF/20260522_residual_conflict_atlas.md`
- Method config: `docs/paper/ExpertGym_ICLR/PAPER_MAIN_METHOD_CONFIG.md`
- Eval aggregate: `docs/paper/ExpertGym_ICLR/PAPER_MAIN_EVAL6_AGGREGATE.md`
