# ExpertGym Baseline Reproduction

This folder contains clean wrappers for paper baselines. Large checkpoints are
written under `/tmp/shared-storage/ExpertGym/baselines/qwen7b`.

## Static Task-Vector Baselines

`build_static_merge_baseline.py` bakes static baselines from the OP-VEC
manifest without loading all expert checkpoints into GPU memory.

Supported methods:

- `task_arithmetic`
- `ties`
- `dare_ta`
- `dare_ties`

Default command:

```bash
DRY_RUN=1 bash scripts/baselines/run_qwen_static_baselines.sh
```

Run after current experiments release resources:

```bash
bash scripts/baselines/run_qwen_static_baselines.sh
```

Important defaults:

- `MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`
- `SCALING=0.3333333333333333`
- `TIES_KEEP_RATIO=0.2`
- `DARE_DROP_RATE=0.8`
- `SEED=20260518`

Existing TA sweep checkpoints are also available under
`/tmp/shared-storage/AgentMerging_plan/experiments/task_arithmetic`.

## AdaMerging

`run_qwen_adamerging_baseline.sh` calls the existing local Qwen AdaMerging
adapter in `/mnt/cache/wuruixiao/users/lsc/era-2026/ExpertGym/worktrees/worktree`.
It supports:

- `task_wise_adamerging`
- `layer_wise_adamerging`
- `task_wise_adamergingpp`
- `layer_wise_adamergingpp`

Dry-run example:

```bash
DRY_RUN=1 METHOD=task_wise_adamerging GPU=0 bash scripts/baselines/run_qwen_adamerging_baseline.sh
```

## WUDI / ExpertMerging

Per current experiment policy, WUDI and ExpertMerging are not relaunched here.
Their local model paths are tracked in
`docs/config/20260518_local_wudi_expertmerging_model_dirs.md`.

## Fisher / Mixture Training

No clean local Qwen Fisher implementation was found in the existing
ExpertMerging worktree. Do not report a Fisher number until a separate
diagonal-Fisher statistics pipeline is implemented and audited.

A MergeBench Fisher implementation exists at:

```text
/mnt/cache/wuruixiao/users/lsc/era-2026/MergeBench/merging/merging_methods/fisher.py
```

It is a useful reference only. It depends on MergeBench `TaskLoader`,
DeepSpeed patched backward, and last-token pseudo-target Fisher estimation, so
it should not be used as an ExpertGym number without adaptation.

Mixture Training is a training baseline, not a static merge. The clean launcher
is:

```bash
DRY_RUN=1 bash scripts/baselines/run_qwen_mixture_grpo_baseline.sh
```

It starts from the TA-1/3 static checkpoint by default and trains all actor
parameters with VeRL GRPO plus the same OP-VEC RewardRouter. It does not install
OP-VEC gates and does not use OPD or retention losses. Store outputs under:

```text
/tmp/shared-storage/ExpertGym/baselines/mixture_grpo/
```
