# 2026-05-18 P0 State Distribution

## Purpose

本报告补论文 P0 地基：在同一批 `eval-targeted96` prompt 上，对比
`1/3` 初始化与 `init=1` 初始化的 executable state 分布，验证为什么
ExpertGym 需要把 prompt 拆成 `frontier / recoverable / stable / unsolved`，
而不是只做普通 GRPO 或普通 imitation。

## Inputs

| item | path |
|---|---|
| `1/3` current rollout | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518/iter_001/rollouts.jsonl` |
| `init=1` current rollout | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_001/rollouts.jsonl` |
| expert positives | `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/*.jsonl` |
| script | `scripts/analysis/build_rollout_state_distribution.py` |

Classification rule:

- `stable`: current policy all-success.
- `frontier`: current policy has mixed success or reward variance.
- `recoverable`: current policy all-fail, but same-prompt expert rollout has verified positive.
- `unsolved`: current policy all-fail and no expert positive found.

## Overall

| init | frontier | recoverable | stable | unsolved | key reading |
|---|---:|---:|---:|---:|---|
| `1/3` K=8 | 54 / 96 | 10 / 96 | 12 / 96 | 20 / 96 | high learning surface: GRPO-dominant, OPD still has all-fail positives |
| `init=1` K=8 | 23 / 96 | 5 / 96 | 48 / 96 | 20 / 96 | strong prior saturates Tool/Memory; fewer recoverable rows |
| `1/3` K=4 | 44 / 96 | 18 / 96 | 12 / 96 | 22 / 96 | interim sanity check |
| `init=1` K=4 | 15 / 96 | 11 / 96 | 50 / 96 | 20 / 96 | interim sanity check |

## By Task

### `1/3` Start, K=8

| task | frontier | recoverable | stable | unsolved | mean reward | mean success |
|---|---:|---:|---:|---:|---:|---:|
| tool | 14 | 1 | 10 | 7 | 0.6448 | 0.5000 |
| memory | 28 | 3 | 0 | 1 | 0.3633 | 0.3633 |
| code | 12 | 6 | 2 | 12 | 0.2810 | 0.1992 |

### `init=1` Start, K=8

| task | frontier | recoverable | stable | unsolved | mean reward | mean success |
|---|---:|---:|---:|---:|---:|---:|
| tool | 3 | 1 | 23 | 5 | 0.9184 | 0.7695 |
| memory | 4 | 0 | 25 | 3 | 0.8594 | 0.8594 |
| code | 16 | 4 | 0 | 12 | 0.3301 | 0.2227 |

## Interpretation

K=8 后结论更稳定：`1/3` 起点仍更适合做论文里的 learnable-composition
主设定，54 个 frontier prompt 支持 GRPO，10 个 recoverable prompt 支持
same-prompt OPD，只有 12 个 stable prompt 需要 retention 保护。它不是 reward
饱和状态，因此能支撑
“executable feedback learns composition”。

`init=1` 是 strong-prior upper-init ablation：Tool 和 Memory 大量变成 stable，
但 Code 仍然弱。K=8 下它的 frontier 有 23 个，比 K=4 的 15 个更高，但
recoverable 只有 5 个，说明“直接全量注入专家”会明显减少 OPD 可利用的
all-fail expert-positive surface；如果后续训练变慢，并不一定是 optimizer
问题，而是可学习 recoverable 样本变少。

最关键的失败面仍是 Code：两个起点都有 12 个 unsolved Code prompt。也就是说，
当前 expert positive 和 reward 构造仍不能充分覆盖 CURE-style Code 修复，这解释了
为什么 Code formal eval 往往无法随 proxy 同步上涨。

## Artifacts

| artifact | path |
|---|---|
| `1/3` K=8 markdown | `docs/report/expertgym_72h/state_distribution_20260518/c033_k8_state_distribution.md` |
| `1/3` K=8 json | `docs/report/expertgym_72h/state_distribution_20260518/c033_k8_state_distribution.json` |
| `init=1` K=8 markdown | `docs/report/expertgym_72h/state_distribution_20260518/init1_k8_state_distribution.md` |
| `init=1` K=8 json | `docs/report/expertgym_72h/state_distribution_20260518/init1_k8_state_distribution.json` |
| `1/3` markdown | `docs/report/expertgym_72h/state_distribution_20260518/c033_iter001_state_distribution.md` |
| `1/3` json | `docs/report/expertgym_72h/state_distribution_20260518/c033_iter001_state_distribution.json` |
| `init=1` markdown | `docs/report/expertgym_72h/state_distribution_20260518/init1_iter001_state_distribution.md` |
| `init=1` json | `docs/report/expertgym_72h/state_distribution_20260518/init1_iter001_state_distribution.json` |
