# OP-VEC Gated-GRPO

This worktree contains the cleaned Gated-GRPO trunk for OP-VEC task-vector
coefficient training.

The base model and expert task-vector deltas are frozen.  Training updates only
the gate / coefficient module:

```text
W(alpha) = W_base + sum_e alpha_e * (W_expert_e - W_base)
```

Native training entry:

```bash
python scripts/train/opvec_gated_grpo_loop.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --seed-manifest /tmp/shared-storage/OnPolicy/data/source_reward/source_reward_t80_m80_c80_seed20260508.jsonl \
  --run-dir /tmp/shared-storage/OnPolicy/runs/gated_grpo/global_smoke \
  --num-iters 1 \
  --num-prompts 16 \
  --samples-per-prompt 4 \
  --gate-parameterization global
```

Main docs:

- `skill/command/README.md`: Chinese runbook for data preparation and training.
- `docs/gated_grpo_project.md`: design notes for the native loop.
- `docs/verl_gated_grpo_integration.md`: optional VeRL adapter notes.

Keep the first experiments on the native loop.  The VeRL path is an experimental
adapter for framework integration after the native loop shows a useful reward
signal.
