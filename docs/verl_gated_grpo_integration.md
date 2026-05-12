# VeRL integration plan for OP-VEC Gated-GRPO

## Why VeRL, but with a gate-only patch

VeRL already provides GRPO-style rollout/reward/policy-update infrastructure and
a custom reward-function interface.  OP-VEC differs from a normal VeRL run in one
place: the trainable module is not the whole actor LM, but the small task-vector
coefficient manager.

For first experiments, the recommended path is still the native OP-VEC loop:

```bash
python scripts/train/opvec_gated_grpo_loop.py ...
```

It directly composes the existing collector, reward router, logprob calculator,
and gate updater.  Use VeRL after the small loop proves the learning signal,
especially if generation throughput becomes the bottleneck.

## VeRL-compatible pieces included

```text
scripts/frameworks/opvec_prepare_verl_data.py       # OP-VEC manifest -> VeRL JSONL/Parquet
scripts/frameworks/opvec_verl_reward_fn.py          # custom_reward_function.compute_score
opvec/frameworks/verl_gated_actor.py                # actor patch: install gates + freeze base
scripts/frameworks/opvec_verl_gate_actor_smoke.py   # hook smoke test without VeRL
recipes/verl/opvec_gated_grpo_template.yaml         # OP-VEC-specific VeRL config template
```

## Gate-only actor patch

Inside a VeRL actor worker/model-construction patch, after the HuggingFace model
is loaded:

```python
from opvec.frameworks.verl_gated_actor import install_opvec_gate_actor

gate_manager, audit = install_opvec_gate_actor(
    torch,
    model,
    config_path="configs/gated_grpo.yaml",
    mode_manifest_path="/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json",
    gate_parameterization="global",
    init_gate_checkpoint=None,
    max_gated_modules=None,
    device="cuda",
)
optimizer = torch.optim.AdamW(gate_manager.parameters(), lr=1e-2)
```

The patch freezes all original LM parameters and exposes `model.opvec_gate_manager`.
The optimizer must use `gate_manager.parameters()` only.

## Reward adapter

Use:

```text
scripts/frameworks/opvec_verl_reward_fn.py:compute_score
```

The function accepts the standard fields `data_source`, `solution_str`,
`ground_truth`, and `extra_info`, then calls the existing OP-VEC `RewardRouter`.

## Research default

For the current project state, do not start with parameter-level gates.  Run:

```text
global coefficients -> layer-band -> global-parameter residual
```

The first successful baseline should train only three expert strengths from a
`fixed0.75` or `cg-tool-extra020` initialization, with task-wise reward
normalization and a trust-region prior.
