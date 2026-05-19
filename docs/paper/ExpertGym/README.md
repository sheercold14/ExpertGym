# ExpertGym V2 Paper Framework

This package contains a second-version EMNLP-style LaTeX framework for ExpertGym.

Main framing:
> Task vectors provide structured priors; executable feedback learns their composition.

Key changes from the previous draft:
- The paper no longer centers on a generic GRPO+OPD+Retention recipe.
- Distillation is reframed as verifier-grounded same-prompt recovery, not imitation.
- Expert Merging is positioned as hidden/logit alignment; ExpertGym is positioned as executable-feedback coefficient learning.
- Experiments are organized to close the story before claiming SOTA.

Files:
- main.tex: paper source
- references.bib: bibliography
- main.pdf: compiled PDF
- AGENT_GROUP_V2.md: internal review synthesis
- RUNS_TO_FILL_V2.md: prioritized experiment plan
