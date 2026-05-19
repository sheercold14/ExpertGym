# Agent Group V2 Synthesis

## Strategist
Main story: task vectors provide structured priors, executable feedback learns the composition. Do not sell the paper as a new loss mixture. Sell it as the missing learning layer between geometric merging and agent execution.

## Community Positioning
- Geometric merging (TA/TIES/DARE/WUDI/TSV) reduces parameter-side interference, but it does not observe rollout success.
- RAM/ARM show that agent deltas and role behaviors are sparse and heterogeneous; ExpertGym adds verifier-side coefficient learning.
- Expert Merging learns coefficients by hidden/logit alignment. ExpertGym differs by using same-prompt recovery only after current merge failure and by accepting updates through executable reward and non-regression guards.

## Method Architect
Core method is an evidence-to-objective map:
- frontier -> direction credit -> GRPO
- recoverable -> recovery credit -> same-prompt Recovery-OPD
- stable -> boundary credit -> Retention
- unsolved -> no local coefficient credit -> skip or data acquisition

## Experiment Designer
Minimum closure:
1. Equal-prior state distribution.
2. Geometry vs executable-feedback main table.
3. Routed objective vs unrouted loss mixture.
4. Recovery-OPD vs offline/logit/hidden imitation.
5. Code reachability/selection diagnostic.
6. Coefficient granularity ladder.
7. Optional mixed-agent generalization.

## Skeptical Reviewer
Current risk: if OPD drives most coefficient movement, the method may look like imitation. Defense: present OPD as early-phase recovery, report gradient share, test same-prompt recovery against imitation-only baselines, and show verifier-guarded non-regression.

## Claim Ladder
Safe: ExpertGym formalizes executable-feedback-driven coefficient learning for agent task-vector composition.
Medium: state routing is necessary because prompt states supply different credit types.
Strong: ExpertGym improves over geometric and imitation-only baselines under non-regression.
Do not claim yet: SOTA agent merging.
