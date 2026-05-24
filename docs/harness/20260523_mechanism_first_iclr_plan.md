# 2026-05-23 Mechanism-First ICLR Plan

## Objective

目标不是继续堆实验或调 gate，而是把 ExpertGym 收束成一个可投稿 ICLR 的机制型故事：

```text
Task vectors are capability priors, but capability priors must be composed under behavior constraints.
```

中文：

```text
任务向量提供能力先验；真正困难的是在行为约束下组合这些能力先验。
```

这条主线必须满足三个条件：

1. **第一性**：从 residual 对行为的作用出发，而不是从某个指标倒推超参。
2. **简单**：算法只保留必要的 utility / harm / preserve 三类信号。
3. **可泛化**：同一个判断协议能解释 Tool、Memory、Code、RAIN、RAM，而不是只服务某个数据集。

## Current Diagnostic Starting Point

最新诊断报告：

```text
docs/report/RCRF/20260523_rain_expertgym_task_vector_diagnosis.md
```

可复现脚本：

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/compare_rain_expertgym_task_vectors.py \
  --output-json /tmp/shared-storage/ExpertGym/rain_expertgym_task_vector_diagnosis_20260523/summary.json \
  --output-md docs/report/RCRF/20260523_rain_expertgym_task_vector_diagnosis.md
```

关键事实：

| fact | implication |
|---|---|
| RAIN 没有 additive reasoning vector；R1 是 protected anchor | 不能把 RAIN 解释成“学 reasoning gate” |
| RAIN 加的是 `Qwen2.5-Instruct - Qwen2.5-Base` | 它是 instruction injection under reasoning constraints |
| ExpertGym 加的是 Tool / Memory / Code RL expert deltas | 它是多能力 prior composition，不是单一 instruction transfer |
| ExpertGym delta norm: code `0.620`, tool `1.255`, memory `5.362` | `1/3` 系数不是等能力注入 |
| R1/Math raw delta norm `347.068` | R1 不能作为普通第四 expert 满量级学习 |

## ICLR-Grade Claim

不要主张“我们找到一个神奇 gate”。应主张：

> RL-trained expert deltas are not task-pure vectors. Their useful and harmful components are span-conditioned and behavior-dependent. ExpertGym turns task-vector merging into residual-level capability attribution under behavior constraints.

更短版本：

> ExpertGym composes agent task vectors by attributing residual-level utility and harm on executable behavior spans.

## Phase 0: Artifact Hygiene

必须先把当前证据链变成可审查资产。

| deliverable | path | status |
|---|---|---|
| RAIN vs ExpertGym vector semantics | `docs/report/RCRF/20260523_rain_expertgym_task_vector_diagnosis.md` | done |
| RAIN strict reproduction | `/mnt/cache/wuruixiao/users/lsc/Agent/RAIN-merging/skill/RAIN-Paper-Full-Reproduction-20260522.md` | done |
| RCF-BC reproducible loop | `docs/harness/20260522_rcf_bc_reproducible_loop.md` | exists |
| residual evidence ledger | `docs/report/RCRF/20260522_rcrf_attribution_ledger.md` | exists |
| paper evidence table | `docs/report/RCRF/20260522_rcf_bc_paper_evidence_table.md` | exists |

Rule: every claim in the paper must cite one of these artifacts or a generated result table.

## Phase 1: Diagnostic Experiments

These experiments answer *why* scalar merging fails and *where* ability lives. They should precede any new method claim.

### D1. Vector Semantics Audit

Question:

```text
Are RAIN, RAM, and ExpertGym vectors commensurate objects?
```

Required evidence:

- anchor / additive vector definition;
- selected modules;
- total norm / block energy;
- whether behavior anchor is additive or protected;
- whether coefficient controls raw delta or projected delta.

Decision:

```text
If vectors are not commensurate, do not compare methods only by scalar coefficient.
Compare by behavior-preserved residual effect.
```

### D2. Behavior Span Sensitivity

Question:

```text
Which residual rows support or harm behavior spans?
```

Spans:

| task | positive behavior span | failure span |
|---|---|---|
| Tool | tool-call JSON / function arguments | malformed call, wrong schema, missing call |
| Memory | update turns + final retrieval | final-only answer without update consistency |
| Code | pass trajectory reasoning + final code | same-prompt fail code / unstable output |
| RAIN | reasoning trajectory / boxed answer | instruction leakage into reasoning |

Evidence:

- pass/fail contrast for Code;
- tool-call preservation score;
- memory full-trajectory preservation score;
- RAIN attention utility/harm alpha.

### D3. Counterfactual Residual Intervention

Question:

```text
Does changing a residual group produce the predicted metric movement?
```

Only three intervention types are allowed:

1. shrink predicted-harm rows;
2. restore predicted-behavior rows;
3. amplify predicted-clean-capability rows.

No metric-targeted sweep. A valid intervention must be justified before eval.

## Phase 2: Simple General Algorithm

Working name:

```text
Behavior-Constrained Residual Composition (BCRC)
```

This should be presented as a simplification / generalization of the current RCF-BC line.

### Inputs

```text
base model theta_0
expert deltas Delta_e
small behavior calibration probes
verifiers or span labels
```

### Step 1: Define Behavior Anchors

For each behavior `b`, collect spans whose representation must not be destroyed:

```text
A_b = hidden/activation/attention features on successful behavior spans
```

Examples:

- RAIN: reasoning/thinking trajectory;
- Tool: tool-call span;
- Memory: update + final retrieval trajectory;
- Code: pass trajectory spans.

### Step 2: Score Residual Utility and Harm

For each residual row `(param, expert)`:

```text
utility = support(success span) - support(failure span)
harm    = interference with protected behavior spans
```

The exact score can be attention-, activation-, or logprob-based, but the paper should keep the abstraction:

```text
residual row is useful if it moves successful behavior more than failed behavior;
residual row is harmful if it moves protected behavior away from its anchor.
```

### Step 3: Compose with a Minimal Rule

Use one rule for all tasks:

```text
gate = base_gate + positive_utility_delta
if harm is high:
    shrink positive delta
if behavior_support is high:
    floor negative delta
if evidence is weak:
    leave unchanged
```

This is the current RCF-BC rule in simpler language. Do not introduce RL, PCGrad, dynamic OPD, or extra losses into the main method unless a diagnostic proves they are necessary.

## Phase 3: Core Experiments for Paper

### E1. Method Main Table

Compare:

- TA / average;
- TIES / DARE / other static baselines;
- RAIN-style behavior projection where applicable;
- RAM-style preservation where available;
- ExpertGym BCRC / RCF-BC.

Tasks:

- Tool: BFCL plus ToolRL-80 as stability metric;
- Memory: HotpotQA/MemAgent-compatible F1;
- Code: CURE/LiveCodeBench diagnostic, report pass@1 and BoN separately.

### E2. Mechanism Table

Must show:

- scalar coefficient shrinkage can improve one task but destroys another;
- hard routing protects behavior but loses Code;
- continuous residual field + behavior constraints gives better Pareto behavior.

This is more important than a single SOTA number.

### E3. Generalization / Heldout

Calibration must be small and documented. The paper should report:

```text
calibration size
source distribution
heldout distribution
whether eval leakage is diagnostic-only or paper-main
```

### E4. RAIN/RAM Connection

Use RAIN and RAM as conceptual baselines:

- RAIN: behavior-preserving instruction injection;
- RAM: behavior/knowledge preservation for RL agents;
- ExpertGym: residual-level utility/harm attribution for multiple agent deltas.

We do not need to beat RAIN on its own R1+instruction task. We need to show the same principle generalizes to multi-agent Tool/Memory/Code composition.

## Paper Structure: ICLR Style

Use a mechanism-first paper structure:

1. Introduction: task vectors are priors; agent merging needs executable behavior constraints.
2. Background: task arithmetic, TIES/DARE, RAIN, RAM, ExpertMerging.
3. Diagnostic Study: scalar gates fail because residuals are not task-pure.
4. Method: Behavior-Constrained Residual Composition.
5. Experiments: main table, mechanism table, heldout, ablations.
6. Analysis: residual atlas, counterfactual interventions, failure cases.
7. Limitations: calibration quality, verifier dependence, code eval cost.

Avoid writing the paper as “we tried many training losses”. The story should be:

```text
diagnosis -> simple rule -> predicted interventions -> heldout behavior
```

## Stop Rules

Do not add a new algorithmic component unless it passes one of these tests:

| proposed component | required diagnostic proof |
|---|---|
| GRPO gate learning | frontier samples have non-saturated reward and gate-sensitive logprob |
| OPD | expert-positive same-prompt trajectory exists and target task lacks success |
| PCGrad | task-specific gradients have measured negative cosine |
| R1 delta | coefficient is norm-controlled and improves Code without Tool/Memory regression |
| extra calibration | it covers a documented eval failure mode not already represented |

If no proof exists, keep the main method static and residual-evidence-based.

## Immediate Next Actions

1. Extend the vector diagnosis script to include RAM artifacts once local RAM/RLLA model directories are identified.
2. Build a single residual evidence table that contains RAIN alpha, ExpertGym utility/harm, and RAM preservation labels where possible.
3. Select one paper-main candidate gate and two principled ablations:
   - no behavior constraint;
   - hard behavior constraint;
   - continuous soft behavior constraint.
4. Rebuild the paper evidence table from these candidates only.
5. Rewrite `docs/paper/ExpertGym/main.tex` around BCRC/RCF-BC instead of generic GRPO+OPD training.

