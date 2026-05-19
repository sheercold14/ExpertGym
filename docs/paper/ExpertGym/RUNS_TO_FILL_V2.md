# Runs to Fill for ExpertGym V2

## P0: State Distribution at 1/3
- Use 100-150 balanced calibration prompts.
- K=8 or K=16 rollouts per prompt.
- Report frontier / recoverable / stable / unsolved and reward variance by task.
- This supports the Executable Credit Collapse observation.

## P1: Global Coefficient Closure
Train only global 3 or common+residual 4 coefficients first.
Compare:
- equal-prior 1/3
- GRPO only
- OPD only
- retention only
- unrouted weighted sum
- random routing
- state-routed ExpertGym

## P2: Recovery Is Not Imitation
Compare:
- offline expert trace distillation
- logit imitation
- hidden/logit Expert-Merging-style alignment if available
- same-prompt OPD
- same-prompt OPD + verifier guard
- full ExpertGym

Metrics: average, worst-task drop, collapse count, and coefficient movement.

## P3: Code Planning Diagnostic
Report:
- any-pass@K reachability
- BoN selection accuracy
- Pass@1
- failure type: syntax, runtime, unit-test fail, hidden-test fail, wrong selection

## P4: Capacity Ladder
Compare:
- global 3
- common+residual 4
- layer-wise
- module-wise 196 x 3
- module-wise + guard/sparsity

Do not make module-wise a main claim unless it improves held-out non-regressive metrics.

## P5: Mixed-Agent Generalization
Create small held-out mixed prompts:
- Tool + Memory
- Memory + Code
- Tool + Code
- Tool + Memory + Code

This is the strongest optional experiment for showing composition rather than single-task recovery.
