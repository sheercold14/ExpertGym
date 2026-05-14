# OP-VEC Retention Push Experiments 2026-05-13

## Objective

用 GPU 0-3 跑两组 2-GPU 对照实验，验证在从 `1/3` task-vector 初始点重新训练时，增大 LR / PPO loss 后能否更快推动 global task-vector 系数，同时通过 KL retention 保持已经全对的 Tool 行为。

## Runs

| run_id | GPU | path | key difference |
|---|---:|---|---|
| `retention_push_centered_i8` | `0,1` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_retention_push_centered_i8_20260513_175332` | `ADVANTAGE_NORMALIZATION=centered` |
| `retention_push_zscore_i8` | `2,3` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_retention_push_zscore_i8_20260513_175332` | `ADVANTAGE_NORMALIZATION=zscore` |
| `retention_fast_centered_n30_i4` | `0,1` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_retention_fast_centered_n30_i4_20260513_181903` | fast diagnostic, `ADVANTAGE_NORMALIZATION=centered` |
| `retention_fast_zscore_n30_i4` | `2,3` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/qbank_c033333_global_retention_fast_zscore_n30_i4_20260513_181903` | fast diagnostic, `ADVANTAGE_NORMALIZATION=zscore` |

## Shared Parameters

```bash
STRATEGY=global
INIT_VALUE=0.3333333333333333
NUM_ITERS=8
NUM_PROMPTS=100
SAMPLES_PER_PROMPT=4
LR=0.006
PPO_LOSS_WEIGHT=2.0
PRIOR_LOSS_WEIGHT=0.003
MAX_COEFF_DELTA=0.5
UPDATE_EPOCHS=2
UPDATE_BATCH_SIZE=2
BATCH_LOSS_REDUCTION=mean
LOSS_GRANULARITY=token
TASK_NORMALIZE_ADVANTAGES=0
USE_FRONTIER_WEIGHT=0
STORE_TOKEN_LOGPROBS=0
LENGTH_NORMALIZE_POLICY_LOGPROB=1
FRONTIER_ORDER=task-interleaved
TASK_WEIGHT_TOOL=1.0
TASK_WEIGHT_MEMORY=1.0
TASK_WEIGHT_CODE=1.0
USE_RETENTION=1
RETENTION_LOSS_WEIGHT=0.05
MAX_RETENTION_ROWS=64
```

## Monitoring Criteria

- `effective coefficient`: `common + *_residual`，观察 Tool / Memory / Code 是否从 `1/3` 明显移动。
- `reward`: 观察 Tool 是否保持，Memory / Code 是否有持续推进。
- `frontier_task_counts`: 判断有效信号是否足够，特别是 Tool all-success 后是否进入 retention。
- `retention_rows` / `retention_loss`: 判断保护项是否真的生效。
- `clip_frac` / `approx_kl`: 判断更新是否过猛。
- `grad_norm`: 判断梯度链路是否有效。

## Live Notes

- 17:53:32 started both runs via tmux:
  - `opvec_push_centered_20260513_175332`
  - `opvec_push_zscore_20260513_175332`
- Initial `nvidia-smi` showed GPU 0-3 free before launch.
- 18:18 stopped the 100-prompt token-level attempt before first update summary.
  - Reason: both runs spent more than 16 minutes in the first HF update after rollout, with no `gate_updates.summary.json` yet.
  - Diagnosis: `NUM_PROMPTS=100`, `UPDATE_EPOCHS=2`, token-level loss, long Code/Memory responses, and retention made first-iteration feedback too slow for same-evening search.
  - The partial directories are retained as audit evidence, but they are not used for conclusions.
- Next action: run a smaller fast diagnostic with `NUM_PROMPTS=30`, `NUM_ITERS=4`, `UPDATE_EPOCHS=1`, and stronger `LR/PPO_LOSS_WEIGHT` to obtain immediate task-vector movement signals.

## Epoch Metrics

### Sampling Balance

Fast diagnostic uses `NUM_PROMPTS=30`, `SAMPLES_PER_PROMPT=4`, `FRONTIER_ORDER=task-interleaved`.

| run_id | iter | prompt balance | sample balance | frontier rows | all-success prompts | all-failure prompts |
|---|---:|---|---|---|---|---|
| `retention_fast_centered_n30_i4` | 1 | code 10 / memory 10 / tool 10 | code 40 / memory 40 / tool 40 | code 6 / memory 6 / tool 7 | code 1 / memory 1 / tool 1 | code 4 / memory 3 / tool 6 |
| `retention_fast_zscore_n30_i4` | 1 | code 10 / memory 10 / tool 10 | code 40 / memory 40 / tool 40 | code 7 / memory 6 / tool 9 | code 1 / memory 0 / tool 0 | code 3 / memory 4 / tool 3 |

Interpretation: input sampling is balanced. Effective PPO/GRPO update rows are mildly unbalanced because frontier filtering keeps only prompts with reward variance.

### Iteration 1

| run_id | adv norm | common | code eff | memory eff | tool eff | code delta | memory delta | tool delta | approx_kl | clip_frac | grad_norm_max | retention_rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `retention_fast_centered_n30_i4` | centered | 0.344325 | 0.335929 | 0.339240 | 0.357806 | +0.002596 | +0.005906 | +0.024472 | 0.000641 | 0.003260 | 53.867535 | 3 |
| `retention_fast_zscore_n30_i4` | zscore | 0.341446 | 0.327625 | 0.355311 | 0.341401 | -0.005708 | +0.021978 | +0.008068 | 0.001458 | 0.005106 | 0.196714 | 1 |

Notes:

- `eff = common + task_residual`; deltas are relative to `1/3`.
- Both first iterations move at least one task vector upward. Centered mainly pushes Tool; zscore mainly pushes Memory.
- `clip_frac` and `approx_kl` are low, so the larger movement is not currently caused by obvious PPO clipping saturation.
- `filled_missing_old_logprobs` is nonzero because `STORE_TOKEN_LOGPROBS=0`; old logprobs are recomputed in update instead of read from vLLM rollout.

### Iteration 2

| run_id | adv norm | common | code eff | memory eff | tool eff | code delta | memory delta | tool delta | approx_kl | clip_frac | grad_norm_max | retention_rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `retention_fast_centered_n30_i4` | centered | 0.343311 | 0.323641 | 0.336591 | 0.369702 | -0.009692 | +0.003257 | +0.036369 | 0.000842 | 0.004489 | 0.122573 | 4 |
| `retention_fast_zscore_n30_i4` | zscore | 0.346761 | 0.320526 | 0.360113 | 0.359645 | -0.012808 | +0.026780 | +0.026312 | 0.000353 | 0.004097 | 0.587922 | 6 |

### Reward Trajectory

| run_id | iter | code reward | memory reward | tool reward | code success | memory success | tool success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `retention_fast_centered_n30_i4` | 1 | 0.3719 | 0.3500 | 0.4386 | 0.350 | 0.350 | 0.250 |
| `retention_fast_centered_n30_i4` | 2 | 0.3312 | 0.2000 | 0.7405 | 0.325 | 0.200 | 0.525 |
| `retention_fast_zscore_n30_i4` | 1 | 0.4156 | 0.2750 | 0.4400 | 0.400 | 0.275 | 0.300 |
| `retention_fast_zscore_n30_i4` | 2 | 0.4513 | 0.4250 | 0.7195 | 0.425 | 0.425 | 0.500 |

Interim judgment after two iterations:

- `zscore` is healthier for the current fast setting: all three raw reward means improve from iteration 1 to 2, while Memory and Tool coefficients both move upward.
- `centered` strongly pushes Tool but sacrifices Code/Memory reward in iteration 2. This is consistent with unnormalized centered advantages letting the current frontier composition dominate the update.
- Neither run pushes all three task coefficients monotonically upward; Code is suppressed in both. If Code preservation is required, the next diagnostic should add Code retention/replay or reduce the common/residual coupling pressure instead of only increasing LR.

### Iteration 3

| run_id | adv norm | common | code eff | memory eff | tool eff | code delta | memory delta | tool delta | approx_kl | clip_frac | grad_norm_max | retention_rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `retention_fast_centered_n30_i4` | centered | 0.344744 | 0.328665 | 0.326625 | 0.378941 | -0.004668 | -0.006708 | +0.045607 | 0.000265 | 0.003302 | 2.687576 | 7 |
| `retention_fast_zscore_n30_i4` | zscore | 0.346881 | 0.324886 | 0.359143 | 0.356615 | -0.008448 | +0.025810 | +0.023282 | 0.000771 | 0.003221 | 0.580724 | 6 |

Updated reward trajectory including rollout before the third update:

| run_id | iter | code reward | memory reward | tool reward | code success | memory success | tool success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `retention_fast_centered_n30_i4` | 3 | 0.4094 | 0.4750 | 0.7746 | 0.400 | 0.475 | 0.600 |
| `retention_fast_zscore_n30_i4` | 3 | 0.3375 | 0.3250 | 0.8506 | 0.325 | 0.325 | 0.625 |

Interim judgment after three iterations:

- `centered` keeps pushing Tool monotonically (`0.3578 -> 0.3697 -> 0.3789`) but Code/Memory coefficients are not stable. The third rollout reward recovers Code/Memory, so the coefficient drop is not yet fatal, but this setting is visibly Tool-biased.
- `zscore` stabilizes Memory and Tool above `1/3` while Code stays below `1/3`. Its second-iteration all-task reward improvement did not persist on the third rollout.
- Retention rows grow as Tool all-success prompts appear, but retention is currently sparse and mostly protects already-correct behavior; it does not solve Code suppression by itself.

### Iteration 4

| run_id | adv norm | code eff | memory eff | tool eff | code delta | memory delta | tool delta | approx_kl | clip_frac | grad_norm_max | retention_rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `retention_fast_centered_n30_i4` | centered | 0.326408 | 0.322010 | 0.417542 | -0.006926 | -0.011323 | +0.084209 | -0.000009 | 0.001865 | 1.559268 | 6 |
| `retention_fast_zscore_n30_i4` | zscore | 0.313062 | 0.332499 | 0.377790 | -0.020271 | -0.000835 | +0.044457 | 0.000347 | 0.002262 | 28.524134 | 8 |

Four-iteration conclusion:

- The current conservative setting can move coefficients, but it does not reach the target `0.6-0.8` region. Best observed coefficient is Tool `0.4175`.
- More iterations alone are unlikely to solve this quickly: Code/Memory oscillate or fall, while Tool absorbs most of the upward movement.
- The next experiments should be reachability tests, not final training runs. Success criterion: an effective task scaler crosses `0.5` within 2-3 iterations, then we test whether `0.6-0.8` improves held-out evaluation.

## Strong-Push Diagnosis

Current behavior is conservative calibration around `1/3`, not rapid discovery of large task-vector scaling. Main causes:

- Frontier signal is sparse: each 30-prompt iteration keeps only about 18-24 update rows.
- Raw reward is saturated/sparse: all-success rows mainly enter retention, all-failure rows carry little direction, and only mixed-success prompts produce GRPO policy signal.
- Global `common + residual` parameterization couples task movement; one task can rise while another is pushed down through residual compensation.
- Token-level length-normalized loss spreads long Code/Memory trajectories over many tokens, reducing effective scalar movement relative to short Tool behavior.

Recommended next diagnostic for fast scaler movement:

```bash
NUM_PROMPTS=30
SAMPLES_PER_PROMPT=4
LR=0.03
PPO_LOSS_WEIGHT=6.0
PRIOR_LOSS_WEIGHT=0.0
MAX_COEFF_DELTA=1.0
UPDATE_EPOCHS=2
LOSS_GRANULARITY=sequence
TASK_NORMALIZE_ADVANTAGES=0
ADVANTAGE_NORMALIZATION=zscore
LENGTH_NORMALIZE_POLICY_LOGPROB=0
BATCH_LOSS_REDUCTION=sum
USE_RETENTION=0
```

More important than hyperparameters: construct scale-preference calibration rows. For the same prompt, evaluate `scale=1/3,0.5,0.75,1.0`, keep prompts where `1/3` is low but a larger scale succeeds, and train with `reward(candidate_gate) - reward(reference_gate)` or a pairwise/margin objective. This turns the objective from local reward fitting into an explicit signal that larger task-vector scaling is useful.

Status: scale-preference / self-compare signal is deferred as a backup method. The immediate reachability experiments below do not add this new signal; they only test whether stronger optimization and less-coupled parameterization can push scalers away from `1/3`.

For interpretability, treat the next experiment as a scaler reachability test: success means any target task scaler crosses `0.5` within 2-3 iterations, and ideally reaches the `0.6-0.8` band. If this fails under the strong setting above, the blocker is not LR but insufficient comparative signal.

The current `global` gate already represents three effective expert scalers through `common + zero-mean residual`; the weak movement is therefore more likely objective/data related than parameter-count related. After proving the objective can push global effective scalers, reintroduce retention/KL and then test layer-band or 588-parameter gates.

## Reachability Runs

Launched 2026-05-13 19:10.

| run_id | GPU | strategy | purpose |
|---|---:|---|---|
| `qbank_c033333_global_reach_seq_strong_n30_i3_20260513_191010` | `0,1` | `global` | test whether strong sequence-level PPO can move effective global scalers past `0.5` |
| `qbank_c033333_global_parameter_reach_seq_strong_n30_i3_20260513_191010` | `2,3` | `global-parameter` | approximate less-coupled global expert scalers while still allowing parameter residuals |

Shared reachability parameters:

```bash
NUM_ITERS=3
NUM_PROMPTS=30
SAMPLES_PER_PROMPT=4
LR=0.03
PPO_LOSS_WEIGHT=6.0
MAX_COEFF_DELTA=1.0
UPDATE_EPOCHS=2
UPDATE_BATCH_SIZE=2
BATCH_LOSS_REDUCTION=sum
LOSS_GRANULARITY=sequence
TASK_NORMALIZE_ADVANTAGES=0
ADVANTAGE_NORMALIZATION=zscore
LENGTH_NORMALIZE_POLICY_LOGPROB=0
USE_RETENTION=0
FRONTIER_ORDER=task-interleaved
TASK_WEIGHT_TOOL=1.0
TASK_WEIGHT_MEMORY=1.0
TASK_WEIGHT_CODE=1.0
```

Differences:

- `global`: `PRIOR_LOSS_WEIGHT=0.0`
- `global-parameter`: `PRIOR_LOSS_WEIGHT=0.001`

Success criterion: any effective expert scaler crosses `0.5` within 2-3 iterations; stronger success if a scaler enters `0.6-0.8`.

### Reachability Iteration 1 Result

| run_id | strategy | tool | memory | code | max coefficient | clip_frac | grad_norm | judgment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `qbank_c033333_global_reach_seq_strong_n30_i3_20260513_191010` | `global` | 0.335475 | 0.311670 | 0.275140 | 0.335475 | 0.804348 | 132460.609375 | failed: high clipping, no upward movement |
| `qbank_c033333_global_parameter_reach_seq_strong_n30_i3_20260513_191010` | `global-parameter` | 0.354867 | 0.339490 | 0.368548 | 0.468548 | 0.675000 | 388.522339 | partial: parameter max near 0.5, global scalers still low |

Notes:

- These two runs were stopped after iteration 1 because they already answered the reachability question for this setting.
- `global-parameter` has `parameter_residual=[-0.10,0.10]`; with global Code `0.368548`, the maximum effective Code coefficient is bounded near `0.468548`. It is therefore not a clean test for reaching `0.6-0.8` unless residual bounds are widened.
- Strong hyperparameters alone are not sufficient. The update is already highly clipped, so simply raising LR/PPO further is unlikely to produce a useful global scaler jump.

### Additional Reachability Run

Launched 2026-05-13 19:29:

| run_id | GPU | strategy | purpose |
|---|---:|---|---|
| `qbank_c033333_parameter_reach_seq_strong_n30_i2_20260513_192949` | `0,1` | `parameter` | test whether fully independent per-parameter coefficients can cross `0.5` without adding scale-preference/self-compare signal |

Parameters: same strong sequence-level setting as above, except `NUM_ITERS=2`, `UPDATE_EPOCHS=1`, `PRIOR_LOSS_WEIGHT=0.0`.

### Parameter Iteration 1 Result

| expert | mean | min | p50 | p90 | max | `>0.5` | `>0.6` |
|---|---:|---:|---:|---:|---:|---:|---:|
| tool | 0.370408 | 0.053002 | 0.387327 | 0.502332 | 0.557173 | 22/196 | 0/196 |
| memory | 0.347412 | 0.133889 | 0.350196 | 0.433482 | 0.597031 | 2/196 | 0/196 |
| code | 0.365551 | 0.109328 | 0.371601 | 0.432456 | 0.559411 | 4/196 | 0/196 |

Other metrics: `frontier={code:8,memory:7,tool:7}`, `clip_frac=0.784091`, `approx_kl=0.470106`, `grad_norm_max=734.486206`.

Interpretation:

- Fully independent parameter coefficients can cross `0.5` locally without scale-preference/self-compare signal.
- The movement is not yet a useful global scaler: means remain near `0.35-0.37`, no coefficient crosses `0.6`, and clipping is very high.
- This supports the hypothesis that raw GRPO has enough local gradient to identify some helpful modules, but not enough coherent signal to push whole expert task-vector scalers into `0.6-0.8`.

### Parameter Iteration 2 Result

Rollout reward before the second update:

| iter | code reward | memory reward | tool reward | code success | memory success | tool success |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3875 | 0.2750 | 0.3743 | 0.375 | 0.275 | 0.250 |
| 2 | 0.4469 | 0.3250 | 0.9577 | 0.425 | 0.325 | 0.725 |

Coefficient distribution after the second update:

| expert | mean | min | p50 | p90 | max | `>0.5` | `>0.6` | `>0.75` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 0.419643 | 0.008964 | 0.431744 | 0.576089 | 0.688214 | 65/196 | 12/196 | 0/196 |
| memory | 0.345699 | -0.010842 | 0.355857 | 0.484741 | 0.639633 | 15/196 | 4/196 | 0/196 |
| code | 0.383190 | 0.009597 | 0.391836 | 0.527496 | 0.628492 | 36/196 | 3/196 | 0/196 |

Other metrics: `frontier={code:8,memory:6,tool:3}`, `clip_frac=0.705882`, `approx_kl=0.234198`, `grad_norm_max=437.935974`.

Conclusion:

- Without scale-preference/self-compare, the current raw-GRPO objective can push some local parameter coefficients into the `0.6-0.7` region after two iterations.
- It still does not push the mean/global expert scaler into `0.6-0.8`; no coefficient reaches `0.75`.
- The movement is strongly Tool-biased in rollout reward and remains highly clipped. This is a reachability success for local coefficients, but not yet a stable merge strategy.

## Global Direct Coefficient Run

Launched 2026-05-13 23:30.

| run_id | GPU | strategy | purpose |
|---|---:|---|---|
| `qbank_c033333_global_coefficient_reach_seq_n30_i2_20260513_233029` | `0,1` | `global-coefficient` | learn exactly three direct coefficients `tool/memory/code`, without `common + residual` |

Parameters:

```bash
NUM_ITERS=2
NUM_PROMPTS=30
SAMPLES_PER_PROMPT=4
LR=0.03
PPO_LOSS_WEIGHT=6.0
PRIOR_LOSS_WEIGHT=0.0
MAX_COEFF_DELTA=1.0
UPDATE_EPOCHS=1
UPDATE_BATCH_SIZE=2
BATCH_LOSS_REDUCTION=sum
LOSS_GRANULARITY=sequence
TASK_NORMALIZE_ADVANTAGES=0
ADVANTAGE_NORMALIZATION=zscore
LENGTH_NORMALIZE_POLICY_LOGPROB=0
USE_RETENTION=0
FRONTIER_ORDER=task-interleaved
```

Success criterion: any direct global coefficient crosses `0.5`; stronger success if it approaches `0.6-0.8`.

Result after two iterations:

| iter | optimizer scope | optimizer steps | tool | memory | code | code reward | memory reward | tool reward |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | batch | 10 | 0.316332 | 0.292581 | 0.247340 | 0.5000 | 0.2750 | 0.3248 |
| 2 | batch | 11 | 0.330062 | 0.238955 | 0.245832 | 0.3969 | 0.2500 | 0.3962 |

Interpretation:

- Direct three-coefficient parameterization is not the blocker; the first epoch had nonzero, high-variance gradients.
- With `UPDATE_BATCH_SIZE=2`, early batches briefly pushed `tool/code` upward, but later batches pulled the already-updated gate back down.
- This suggests an order-sensitive optimizer dynamic, so the next controlled run uses epoch-scope gradient accumulation: mini-batches only split backward work, and `optimizer.step()` runs once after all selected rows.

## Global Direct Epoch-Step Diagnostic

Launched 2026-05-13 23:52.

| run_id | GPU | strategy | changed variable |
|---|---:|---|---|
| `qbank_c033333_global_coeff_epochstep_seq_n30_i2_20260513_235223` | `2,3` | `global-coefficient` | `OPTIMIZER_STEP_SCOPE=epoch` |

All other major reachability settings match the previous direct global run: `NUM_PROMPTS=30`, `SAMPLES_PER_PROMPT=4`, `LR=0.03`, `PPO_LOSS_WEIGHT=6.0`, `PRIOR_LOSS_WEIGHT=0.0`, `BATCH_LOSS_REDUCTION=sum`, `LOSS_GRANULARITY=sequence`, `ADVANTAGE_NORMALIZATION=zscore`.

Expected diagnostic: if the gate moves upward or at least stops collapsing, the main issue is mini-batch step order. If it still moves down, the problem is the aggregate reward/logprob objective direction, not just update ordering.

Iteration 1 result:

| iter | optimizer scope | optimizer steps | updates | tool | memory | code | code reward | memory reward | tool reward |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | epoch | 1 | 22 | 0.363333 | 0.363333 | 0.363333 | 0.4313 | 0.3750 | 0.3658 |

Metrics: `frontier={code:9,memory:5,tool:8}`, `grad_norm_max=405.793488`, `gate_delta_max=0.030000`, `loss_normalizer=22`.

Immediate interpretation:

- This is a positive diagnostic. Changing only the optimizer step scope flips the first update direction from coefficient collapse to coefficient growth.
- In epoch-scope mode, all rows are evaluated under the same pre-update gate, so `old/current` ratios are not distorted by earlier mini-batch steps in the same rollout batch.
- The first step is small because there is only one Adam step per rollout iteration. If the second rollout reward is stable, the next useful variants are higher `LR`, multiple update epochs, or `BATCH_LOSS_REDUCTION=mean` for a more normalized full-batch objective.

Iteration 2 result:

| iter | optimizer scope | optimizer steps | updates | tool | memory | code | code reward | memory reward | tool reward |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | epoch | 1 | 18 | 0.393333 | 0.333333 | 0.333333 | 0.3594 | 0.4250 | 0.8936 |

Metrics: `frontier={code:7,memory:7,tool:4}`, `grad_norm_max=633.613098`, `gate_delta_max=0.030000`, `loss_normalizer=18`.

Interpretation update:

- Epoch-scope update can separate coefficients: after the second iteration, `tool` continues upward while `memory/code` move back to the initialization point.
- The current signal is therefore not a clean three-expert preference signal. It is first learning shared task-vector strength, then a Tool-biased correction.
- Tool reward became near-saturated in the second rollout, which reduces future Tool frontier rows; Memory/Code still need stronger expert-specific calibration or self-compare signal to separate.

### Fast Diagnostic Overrides

```bash
NUM_ITERS=4
NUM_PROMPTS=30
LR=0.01
PPO_LOSS_WEIGHT=3.0
PRIOR_LOSS_WEIGHT=0.001
UPDATE_EPOCHS=1
UPDATE_BATCH_SIZE=2
MAX_COEFF_DELTA=0.5
```

These runs keep task weights at `1.0/1.0/1.0` and use the same retention defaults: `USE_RETENTION=1`, `RETENTION_LOSS_WEIGHT=0.05`, `MAX_RETENTION_ROWS=64`.
