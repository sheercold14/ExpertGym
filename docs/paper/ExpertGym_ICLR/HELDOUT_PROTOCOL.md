# Heldout And Probe Protocol

Date: 2026-05-23

Purpose: make the mechanism-first ExpertGym story auditable.  BCRC/RCF-BC uses behavior probes to diagnose residual utility and harm, but the paper must not imply that those probes are ordinary training data or that final evaluation prompts were used for method selection.

## Core Principle

The calibration/probe set is not a small training set.  It is a measurement instrument:

```text
probe prompt -> behavior span -> residual evidence row
```

A prompt is useful only if it reveals one of:

- capability utility: a residual helps a verified successful trajectory more than a failed trajectory;
- behavior support: a residual preserves tool-call or memory-trajectory behavior;
- behavior harm: a residual damages a protected executable behavior;
- low confidence: the residual has no stable evidence and should remain unchanged.

Final claims must be based on heldout evaluation, not on probe reward.

## Data Splits

| split | gradient / gate decision | checkpoint selection | paper metric | allowed use |
|---|---:|---:|---:|---|
| `probe` | yes | no | no | build residual evidence: utility, support, harm |
| `monitor` | no | yes | no | choose between predeclared variants / early stop |
| `guard` | no | no except fail-stop | no | non-regression sanity before full eval |
| `formal_eval` | no | no | yes | final paper table |
| `diagnostic_eval_leak` | no paper-main use | no | no | failure analysis only; must be labeled |

The paper-main method should be frozen before looking at `formal_eval` results.  If a formal result motivates a new method variant, that variant must restart as a new experiment with a new heldout claim.

## Task-Specific Split Rules

### Tool

Probe sources:

- ToolRL/RLLA train-style prompts for source expert behavior.
- BFCL-style synthetic probes for schema, canonicalization, default values, enum arguments, and parallel calls.

Disjointness:

- Probe/monitor/guard must differ in function names, namespaces, entities, enum values, and schema templates.
- BFCL official Eval6 categories are formal evaluation only.
- ToolRL test all80 may be used as a secondary robustness metric, but only if clearly reported separately from BFCL.

Behavior span:

```text
tool-call JSON / Pythonish call span
```

Evidence:

- parseability;
- function-name correctness;
- argument exactness;
- call-count / parallel alignment;
- default-value discipline.

### Memory

Probe sources:

- HotpotQA train/dev questions disjoint from Eval6.
- Prefer full MemAgent-style trajectories: update turns plus final answer.

Disjointness:

- Split by question id and article/entity id.
- Long-context monitor/guard should include article sets not used in probe.

Behavior span:

```text
memory update turns + final answer span
```

Evidence:

- final-answer verifier is necessary but not sufficient;
- update-turn behavior should be included when available;
- final-only memory OPD must be labeled as weak evidence.

### Code

Probe sources:

- CodeContests-train / train-side generated tasks;
- CURE-style task format with public-like and hidden-like tests;
- same-prompt pass/fail trajectories when available.

Disjointness:

- No official LiveBench/LiveCodeBench prompt or hidden test should be used for paper-main probes.
- Diagnostic eval-leak experiments are allowed only for debugging and must be excluded from paper-main evidence.
- Split by task id, algorithm tag, solution template, and test-generator seed.

Behavior span:

```text
prompt constraints + reasoning span + final code block
```

Evidence:

- pass/fail contrast is stronger than expert-positive-only imitation;
- BoN gap is diagnostic: high any-pass but low pass@1 means selection/test-generation signal is missing;
- public-example pass is not a full positive unless hidden-like tests pass.

## RCF-BC Evidence Status

Current RCF-BC evidence uses a code-hurt diagnostic subset to reveal residual conflict.  This is valid as mechanism evidence, but not sufficient for a broad benchmark claim.

Paper-safe status:

| evidence | paper use | caveat |
|---|---|---|
| 8765 residual workbench | mechanism / diagnosis | not a final benchmark |
| Code source/span conflict | motivates method | subset diagnostic |
| v8/v18/v19/v14/v15 mechanism table | ablation intuition | quick Tool/Memory + code-hurt subset |
| full Eval6 queue | selected benchmark sanity check | complete for BCRC / no-behavior / hard-behavior |
| eval-leak TRC experiments | internal debugging only | exclude from paper-main table |

## Minimum Paper-Main Protocol

For the selected paper-main benchmark sanity check:

1. Freeze candidate set:

```text
bcrc_v18_alias_v9
no_behavior_v1_code_only
hard_behavior_v8
```

2. Run the same full Eval6 harness for all candidates:

```text
Tool BFCL mean
Memory Eval6 mean F1
Code CURE pass@1
Code CURE BoN(4,4)
```

3. Aggregate with:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/aggregate_iclr_paper_main_eval.py
```

4. Interpret:

- The completed queue makes BCRC usable as the main mechanism-derived operating point.
- Because BCRC does not beat TA-0.75 or the historical TAME-style model on Code and average score, the paper must remain a mechanism paper: task vectors are residual-level, span-conditioned, and behavior-constrained; current method is an interpretable operating point, not SOTA.

## Reporting Rule

Every result table should identify one of:

```text
probe result
monitor result
guard result
formal eval result
diagnostic eval-leak result
```

Do not mix these categories in one table without a column stating the evidence type.
