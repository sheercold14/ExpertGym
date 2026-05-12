# OP-VEC adapters for VeRL / slime-style experiments

These files are adapters, not a full fork of VeRL/slime.  Native VeRL/slime actor
training updates full LM weights; OP-VEC Gated-GRPO updates only the task-vector
coefficient module.  Use `scripts/train/opvec_gated_grpo_loop.py` for gate-only
training, and use these adapters when you want the same prompt/reward semantics
inside a framework run.

## Build prompt data

```bash
python scripts/frameworks/opvec_prepare_verl_data.py \
  --seed-manifest /tmp/shared-storage/OnPolicy/data/seed_prompt_manifest.jsonl \
  --output /tmp/shared-storage/OnPolicy/data/verl_opvec_prompts.jsonl \
  --parquet /tmp/shared-storage/OnPolicy/data/verl_opvec_prompts.parquet \
  --tasks tool,code,memory \
  --limit 150
```

## Reward function

Point the framework's reward hook to:

```text
scripts/frameworks/opvec_verl_reward_fn.py:compute_score
```

For native Gated-GRPO, do not use this adapter directly; the collector already
uses `opvec.rewards.router.RewardRouter` and records old policy log-probabilities
needed by the coefficient update.


## Experimental VeRL actor patch

For a VeRL run that trains only OP-VEC gates, use the external lib hook:

```bash
export OPVEC_ENABLE_VERL_PATCH=1
export OPVEC_CONFIG=$PWD/configs/gated_grpo.yaml
export OPVEC_MODE_MANIFEST=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
export OPVEC_GATE_PARAMETERIZATION=global
export OPVEC_MAX_GATED_MODULES=0
export OPVEC_FREEZE_BASE=1
```

Then start from `configs/verl_gated_grpo_experimental.yaml`.  The hook attaches
`model.opvec_gate_manager` and freezes all non-gate parameters.  This is a
research shim; if a VeRL release changes loader order, fall back to the native
`opvec_gated_grpo_loop.py` and use VeRL only for reward/prompt compatibility.
