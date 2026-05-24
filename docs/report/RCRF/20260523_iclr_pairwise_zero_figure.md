# ICLR Pairwise-Zero Diagnostic Figure

PDF: `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/paper/ExpertGym_ICLR/figures/pairwise_zero_diagnostics.pdf`
PNG: `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/paper/ExpertGym_ICLR/figures/pairwise_zero_diagnostics.png`

## What The Figure Shows

1. Pairwise conflict is task-span specific; missing behavior probes are explicitly shown as `N/A`, not as zero conflict.
2. Zeroing one expert removes mixed residual roles, so an expert cannot be interpreted as pure ability or pure noise.
3. The existing `code=0` ablation increases Memory but damages Code, making scalar zeroing a negative control rather than a method.

## Paper-Safe Claim

Pairwise-zero diagnostics show that each expert removal deletes a mixture of capability, behavior-support, and conflict residuals; scalar zeroing is therefore useful as a negative control but not as a composition method.

## Pairwise Conflict Rows

| pair | task | comparable modules | opposite-sign rate | both-positive rate |
| --- | --- | ---: | ---: | ---: |
| tool_memory__code_zero | tool | 196 | 0.5000 | 0.4745 |
| tool_memory__code_zero | memory | 196 | 0.6480 | 0.3163 |
| tool_memory__code_zero | code | 156 | 0.4551 | 0.1667 |
| tool_code__memory_zero | tool | 0 | N/A | N/A |
| tool_code__memory_zero | memory | 0 | N/A | N/A |
| tool_code__memory_zero | code | 136 | 0.5294 | 0.2941 |
| memory_code__tool_zero | tool | 0 | N/A | N/A |
| memory_code__tool_zero | memory | 0 | N/A | N/A |
| memory_code__tool_zero | code | 150 | 0.6067 | 0.1867 |

## Zeroed Expert Role Rows

| pair | zero expert | code repair | shared positive | memory support | tool support |
| --- | --- | ---: | ---: | ---: | ---: |
| tool_memory__code_zero | code | 43 | 0 | 0 | 0 |
| tool_code__memory_zero | memory | 4 | 0 | 84 | 20 |
| memory_code__tool_zero | tool | 13 | 17 | 0 | 126 |
