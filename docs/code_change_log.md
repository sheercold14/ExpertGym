# Code Change Log

## 2026-05-12 Native Batch Update

### Goal

- Keep the native OP-VEC training path transparent and low-memory.
- Start replacing row-by-row gate updates with auditable mini-batch updates.
- Preserve legacy behavior by default: `--update-batch-size 1`.

### Changes

- `scripts/train/opvec_update_gates_from_rollouts.py`
  - Added `--update-batch-size`.
  - Added `--batch-loss-reduction {mean,sum}`.
  - Added `--loss-granularity {sequence,token}`.
  - Added `_UpdateBatcher` to accumulate row-level gradients and flush one `optimizer.step()` per mini-batch.
  - Frontier and retention rows now log `optimizer_step_index`, `update_batch_size`, `batch_loss_reduction`, and `batch_loss_scale`.
  - Summary now records `optimizer_steps` and `skipped_optimizer_steps`.
  - Reworked the pure GRPO incremental helper so it does not leave partial gradients behind when fewer than two samples have valid current logprob.
  - Token loss mode uses stored `old_logprobs` and `response_mask` with newly computed current per-token logprobs for PPO/GRPO and KL terms.
  - `--fill-missing-old-logprob` now also fills token-level `old_logprobs` and `response_mask` when `--loss-granularity token` is enabled, including Memory trajectory turns.
  - Best-response and pairwise auxiliary losses still use summed sequence logprob.
  - Update rows now log `clip_frac` and `approx_kl`; epoch summaries log their means.

- `tests/test_update_gates_objectives.py`
  - Added a unit test proving `_UpdateBatcher` delays stepping until the configured batch size and uses mean scaling.
  - Added coverage for token-level updater entries and response-mask application.
  - Added coverage for token-level old-logprob filling on Memory trajectories.
  - Added coverage for token-level policy monitoring metrics.

- `opvec/rewards/router.py`
  - Added `RewardRouter.batch_score()` as the stable batch reward entry point.
  - The first implementation preserves official reward semantics by dispatching each item through the existing per-task adapters.

- `opvec/data/schema.py`
  - Added rollout row/sample validation, token-level field validation, and stable `make_gate_id()`.
  - Token-level validation enforces aligned `response_token_ids`, `old_logprobs`, and `response_mask` lengths.

- `opvec/modeling/logprob.py`
  - Added token-level old-logprob helpers that return `response_token_ids`, per-token logprobs, `response_mask`, and summed logprob from the same scoring path.
  - Added current-logprob scoring from explicit response token ids for vLLM-generated samples to avoid detokenize/re-tokenize drift during update.
  - Kept the existing summed logprob APIs intact by implementing them on top of the token-level helper.

- `opvec/train/gated_grpo.py`
  - Added pure token-level clipped GRPO loss, token-level reverse-KL penalty, and masked token mean helpers.
  - These functions fix the intended VeRL-style loss semantics before the main updater is rewired.

- `scripts/train/opvec_collect_vllm_rollouts.py`
  - Routed batched non-Memory responses and Memory final answers through `RewardRouter.batch_score()`.
  - Added `--store-token-logprobs` so vLLM rollouts can persist exact sampled `response_token_ids`, per-token `old_logprobs`, and `response_mask`.
  - Memory trajectory rollouts now store token logprobs on each update/final turn and aggregate them on the sample.

- `scripts/train/opvec_collect_hf_rollouts.py`
- `scripts/train/opvec_collect_vllm_rollouts.py`
  - Validate each rollout row before writing it.
  - New rollout rows now include `group_id`, stable `gate_id`, and row-level `seed`.

- `scripts/train/opvec_collect_hf_rollouts.py`
  - Added `--store-token-logprobs` to optionally write token-level old logprobs and masks.
  - Memory trajectory samples aggregate update-turn and final-turn token fields while each turn also keeps its own token fields.

- `scripts/train/opvec_gated_grpo_loop.py`
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Added passthrough for `--update-batch-size`, `--batch-loss-reduction`, and `--loss-granularity`.
  - HF loop auto-enables token logprob storage when `--loss-granularity token` is selected.
  - Bake+vLLM loop now passes `--store-token-logprobs` to vLLM collection when requested.

- `tests/test_reward_router.py`
  - Added tests proving batch reward matches single-sample reward, supports one-record broadcasting, and rejects length mismatches.

- `tests/test_rollout_schema.py`
  - Added tests for current rollout rows, token-level length checks, Memory trajectory shape, and stable gate ids.

- `tests/test_logprob.py`
  - Added coverage that token-level old logprobs sum to the legacy sequence logprob.
  - Added coverage for current-logprob scoring from explicit response token ids.

- `tests/test_vllm_rollout_logprobs.py`
  - Added coverage for parsing vLLM sampled token logprobs and Memory trajectory token aggregation.

- `tests/test_gated_grpo_utils.py`
  - Added token-level loss tests for one-token equivalence, response-mask behavior, and zero KL when old/new logprobs match.

- `docs/native_batch_training_contract.md`
  - Added the Chinese training contract for behavior policy binding, target rollout schema, reward semantics, batch update semantics, and monitoring.

- `docs/native_batch_progress_audit.md`
  - Added an explicit checklist mapping the 10 objective items to current evidence and remaining gaps.

- `docs/native_sequence_token_smoke_report.md`
  - Added the partial sequence/token smoke result from `sequence_vs_token_smoke10_native_20260512_153139`.

- `skill/command/*.sh`
  - Replaced old hardcoded `OnPolicyMerge_gated_grpo` worktree `cd` paths with repo-root resolution from the script location.

- `skill/command/run_smoke_sequence_vs_token.sh`
  - Added a 10-prompt A/B launcher that collects one token-aware HF rollout file, then updates the same rollouts with sequence and token losses.

- `skill/command/README.md`
  - Updated the working directory to the standalone `Agent/ExpertGym` repo.

### Current Semantics

- Default behavior remains row-by-row because `--update-batch-size` defaults to `1`.
- Batch mode still uses the current native sequence-level logprob objective.
- `--batch-loss-reduction mean` uses a fixed `1/update_batch_size` loss scale; a final partial flush is intentionally conservative until the token-level batch loss lands.
- `RewardRouter.batch_score()` is an API boundary first, not a parallel reward executor yet.
- Default production updater still uses sequence-level logprob ratio.
- Token-level PPO/GRPO can be enabled with `--loss-granularity token` when rollout samples include `old_logprobs` and `response_mask`.

### Next Items

- Run sequence vs token loss comparison on the same 10-prompt smoke before changing the default.
- Run vLLM baked rollout -> token-level fill -> update smoke on 10 prompts.
- Parallelize or vectorize expensive reward adapters behind `RewardRouter.batch_score()` where useful.
- Add vLLM batch rollout after the update semantics are stable.

## 2026-05-12 Frontier Row Ordering

- `scripts/train/opvec_update_gates_from_rollouts.py`
  - Added `--frontier-order {as-is,shuffle,task-interleaved}` and `--frontier-shuffle-seed`.
  - `as-is` preserves the previous rollout/frontier order exactly.
  - `shuffle` applies deterministic global random shuffle over all selected frontier rows.
  - `task-interleaved` shuffles rows within each task, then round-robins `tool -> memory -> code`; this is the recommended mode for small update batches because each optimizer window is less likely to be dominated by one task.
  - Update summaries now record the selected order and seed.

- `scripts/train/opvec_gated_grpo_loop.py`
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Added passthrough for frontier ordering.
  - If no explicit shuffle seed is supplied, the outer loop uses `seed + iteration - 1` so each iteration has a reproducible but different ordering.

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added `FRONTIER_ORDER` and `FRONTIER_SHUFFLE_SEED` environment overrides.

## 2026-05-13 Epoch-Scope Gate Optimizer Step

Context:

- Direct global coefficient experiments showed that small mini-batch optimizer steps can be order-sensitive: early frontier rows pushed coefficients up, while later rows pulled them back after the gate had already moved.
- We need a controlled mode where mini-batches are only gradient-accumulation chunks, and the gate is updated once from the whole selected frontier set.

Changes:

- `scripts/train/opvec_update_gates_from_rollouts.py`
  - Added `--optimizer-step-scope {batch,epoch}`.
  - `batch` preserves previous behavior: `optimizer.step()` after every `--update-batch-size` processed rows.
  - `epoch` defers `optimizer.step()` until the end of each update epoch, so all frontier/retention/OPD rows contribute to one accumulated gradient.
  - With `--batch-loss-reduction mean`, `epoch` scales each row by the planned epoch row count rather than by `update_batch_size`, so the result is independent of chunk size.
  - Update logs and summaries now record `optimizer_step_scope` and `loss_normalizer`.

- `scripts/train/opvec_gated_grpo_loop.py`
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Added passthrough for `--optimizer-step-scope`.

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added `OPTIMIZER_STEP_SCOPE=batch|epoch`; default is `batch`.

## 2026-05-14 Gate Optimizer Choice And Persistent State

Context:

- Epoch-scope accumulation fixes order-sensitive mini-batch updates, but a freshly initialized AdamW with one optimizer step per rollout iteration behaves like a sign step: each active coefficient moves by roughly `LR`.
- To preserve gradient magnitude information and test more standard optimization dynamics, the gate updater needs optimizer choice and cross-iteration optimizer state.

Changes:

- `scripts/train/opvec_update_gates_from_rollouts.py`
  - Added `--optimizer {adamw,sgd}`.
  - Added `--sgd-momentum` and `--sgd-nesterov`.
  - Added `--optimizer-state-in` and `--optimizer-state-out` for loading/saving `torch.optim` state dictionaries.
  - Update summaries now record optimizer name, SGD settings, state in/out paths, and whether a state checkpoint was loaded.

- `scripts/train/opvec_gated_grpo_loop.py`
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Added passthrough for optimizer settings.
  - Added `--persist-optimizer-state` so each outer rollout/update iteration saves `gate_updates.optimizer.pt` and reloads it in the next iteration.
  - Added `--optimizer-state-checkpoint` for resuming from an existing optimizer state.

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added `OPTIMIZER`, `SGD_MOMENTUM`, `SGD_NESTEROV`, `PERSIST_OPTIMIZER_STATE`, and `OPTIMIZER_STATE_CHECKPOINT`.

## 2026-05-13 Per-Task Rollout Tokens And Task-Normalize Default

Context:

- Code/CURE alignment review showed that current Code rollout used the shared `MAX_NEW_TOKENS=1024`, while CURE official optimization/evaluation allows much longer code generations.
- Cross-task `TASK_NORMALIZE_ADVANTAGES=1` can erase real differences in Tool/Memory/Code signal strength after each task's raw reward has already been put on a comparable scale.
- We keep per-prompt GRPO normalization intact, but stop rescaling advantages across tasks by default in the qbank command.

Changes:

- `scripts/train/opvec_collect_vllm_rollouts.py`
  - Added `--tool-max-new-tokens` and `--code-max-new-tokens`.
  - Non-Memory rollout now selects `SamplingParams` by `prompt_record["task"]`.
  - Memory continues to use `--memory-update-max-new-tokens` and `--memory-final-max-new-tokens`.
  - Rollout summaries now record `max_new_tokens`, `tool_max_new_tokens`, and `code_max_new_tokens`.

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Added passthrough for:
    - `--tool-max-new-tokens`
    - `--code-max-new-tokens`
    - `--memory-update-max-new-tokens`
    - `--memory-final-max-new-tokens`

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added environment overrides:
    - `TOOL_MAX_NEW_TOKENS`, default `512`
    - `CODE_MAX_NEW_TOKENS`, default `4096`
    - `MEMORY_UPDATE_MAX_NEW_TOKENS`, default `2048`
    - `MEMORY_FINAL_MAX_NEW_TOKENS`, default `2048`
  - Kept `MAX_NEW_TOKENS=1024` as the generic fallback.
  - Changed `TASK_NORMALIZE_ADVANTAGES` default from `1` to `0`.
  - Added startup logging for per-task token limits.

Recommended command shape:

```bash
STRATEGY=global \
NUM_ITERS=20 \
NUM_PROMPTS=100 \
SAMPLES_PER_PROMPT=4 \
TOOL_MAX_NEW_TOKENS=512 \
CODE_MAX_NEW_TOKENS=4096 \
MEMORY_UPDATE_MAX_NEW_TOKENS=2048 \
MEMORY_FINAL_MAX_NEW_TOKENS=2048 \
TASK_NORMALIZE_ADVANTAGES=0 \
bash skill/command/run_qbank_c033333_gate_strategy.sh
```

Semantics:

- `TASK_NORMALIZE_ADVANTAGES=0` does not disable GRPO's within-prompt advantage normalization.
- It only disables the extra cross-task mean-absolute-advantage rescaling.
- Tool, Memory, and Code reward should therefore be kept on comparable per-sample scales before GRPO.
- If a future self-compare field such as `reward_delta_vs_baseline` is used, prefer:

```bash
ADVANTAGE_FIELD=reward_delta_vs_baseline \
ADVANTAGE_FIELD_APPLY_FRONTIER_WEIGHT=0 \
TASK_NORMALIZE_ADVANTAGES=0
```

Validation:

- `python -m py_compile scripts/train/opvec_collect_vllm_rollouts.py scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
- `bash -n skill/command/run_qbank_c033333_gate_strategy.sh`

## 2026-05-13 KL Retention Control Launcher

Context:

- In qbank Gated-GRPO runs, Tool rows can quickly become all-success and then leave the GRPO frontier.
- All-success rows have no group-relative reward gradient, but they are still useful as behavior-preservation data while Memory/Code frontier rows continue to move the shared gate.
- This change keeps the original qbank launcher defaults unchanged and adds an explicit retention-control launcher for clean A/B comparison against the previous run.

Changes:

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Added passthrough arguments to the updater:
    - `--use-retention`
    - `--retention-loss-weight`
    - `--max-retention-rows`
  - These map directly to the existing retention implementation in `scripts/train/opvec_update_gates_from_rollouts.py`.

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added optional environment variables, all defaulting to no behavior change:
    - `USE_RETENTION=0`
    - `RETENTION_LOSS_WEIGHT=`
    - `MAX_RETENTION_ROWS=`
  - When enabled, the script forwards the corresponding retention arguments to the bake-vLLM loop.
  - Existing task weights, learning rates, update epochs, frontier quotas, and rollout defaults are not changed.

- `skill/command/run_qbank_c033333_gate_strategy_retention.sh`
  - Added a small wrapper for A/B experiments.
  - Defaults:
    - `USE_RETENTION=1`
    - `RETENTION_LOSS_WEIGHT=0.05`
    - `MAX_RETENTION_ROWS=64`
  - It delegates to `run_qbank_c033333_gate_strategy.sh`, so all other training parameters remain inherited from the base launcher unless explicitly overridden by the caller.

Semantics:

- Retention rows are all-success rollout rows.
- They do not contribute GRPO reward-policy gradient.
- They add a KL-to-old-policy preservation term:

```text
loss_retention = task_weight * retention_loss_weight * reverse_kl_surrogate(current_gate, rollout_gate)
```

- This is a local behavior-preservation constraint for the current iteration, not a fixed KL-to-base or KL-to-1/3 reference.

Recommended initial coefficient:

- Use `RETENTION_LOSS_WEIGHT=0.05` first.
- Increase to `0.1` only if all-success Tool behavior still degrades while `retention_loss` remains small.
- Avoid starting much larger than `0.1`, because retention should protect already-correct behavior without overwhelming Memory/Code frontier gradients.

Validation:

- `python -m py_compile scripts/train/opvec_gated_grpo_bake_vllm_loop.py scripts/train/opvec_update_gates_from_rollouts.py`
- `bash -n skill/command/run_qbank_c033333_gate_strategy.sh`
- `bash -n skill/command/run_qbank_c033333_gate_strategy_retention.sh`

## 2026-05-13 Global Direct Coefficient Summary Support

Context:

- The codebase already has `global-coefficient` / `global-direct` support in the gate manager and checkpoint builders.
- This parameterization learns exactly three direct coefficients: `tool`, `memory`, and `code`, without `common + residual` decomposition.
- The qbank launcher accepts `STRATEGY=global-coefficient`.

Change:

- `scripts/eval/summarize_gate_strategy_run.py`
  - Fixed `_effective_coefficients()` so gate checkpoints containing direct keys `tool`, `memory`, `code` are summarized as three global expert coefficients.
  - Without this, summaries would fall back to the legacy `common + *_residual` interpretation and report incorrect coefficient movement for `global-coefficient` runs.

Validation:

- `python -m py_compile scripts/eval/summarize_gate_strategy_run.py`

## 2026-05-14 Dynamic OPD From Current All-Fail Prompts

Context:

- The fixed OPD buffer used by the previous paper96 runs was not restricted to the exact 96 prompts in the current calibration manifest.
- For the next experiment, OPD should target only prompts that the current policy fails on in the current rollout, while reusing same-prompt expert trajectories.

Changes:

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Added per-iteration dynamic OPD construction.
  - New args:
    - `--dynamic-opd-expert-rollout`
    - `--dynamic-opd-tasks`
    - `--dynamic-opd-current-max-success`
    - `--dynamic-opd-positive-threshold`
    - `--dynamic-opd-max-positives-per-row`
    - `--dynamic-opd-max-negatives-per-row`
    - `--dynamic-opd-per-task`
  - Each iteration now optionally builds:

```text
iter_xxx/opd_distill_from_allfail.jsonl
```

  - The update step reads the merged rollout plus this single per-iteration OPD file; it does not read shard files.

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added environment passthrough for dynamic OPD expert rollout paths and selection controls.

- `skill/command/run_paper96_dynamic_opd_gc_20260514.sh`
  - Added a reproducible launcher for the D experiment:
    - 96 balanced prompts.
    - `global-coefficient`.
    - GPUs `2,3`.
    - Offline expert rollouts over the same 96 prompts.
    - Per-iteration OPD rows selected from current policy all-fail prompts.

Validation:

- `python -m py_compile scripts/train/opvec_gated_grpo_bake_vllm_loop.py scripts/data/build_opd_distill_from_expert_rollouts.py`
- `bash -n skill/command/run_qbank_c033333_gate_strategy.sh skill/command/run_paper96_dynamic_opd_gc_20260514.sh`
- `DRY_RUN=1` verified that the loop emits dynamic OPD builder commands before gate update.

## 2026-05-14 Retention NLL Preservation

Context:

- Paper96 dynamic OPD ABCD showed that the legacy KL retention did not prevent Tool collapse.
- Inspection found retention rows had `retention_loss=0.0` and `kl_loss=0.0` because epoch-scope updates compute KL before the optimizer step, when current policy equals the rollout policy.
- For all-success rows, we need a preservation objective with non-zero gradient before the step.

Changes:

- `scripts/train/opvec_update_gates_from_rollouts.py`
  - Added `--retention-objective {kl,nll}`.
  - Legacy default remains `kl` for old run compatibility.
  - `nll` mode applies best-response NLL to retention/all-success rows:

```text
L_retention_nll = - retention_loss_weight * mean log pi_gate(y_success | x)
```

  - Added `--retention-positive-reward-threshold`, default `1.0`.
  - NLL retention no longer fills unused old logprobs for retention rows; KL retention still does.
  - Summary now records `retention_objective` and `retention_positive_reward_threshold`.

- `scripts/train/opvec_gated_grpo_loop.py`
  - Passes `--retention-objective` and `--retention-positive-reward-threshold` through to the updater.

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Same passthrough for the vLLM bake/rollout loop.

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added env controls:
    - `RETENTION_OBJECTIVE=kl|nll`
    - `RETENTION_POSITIVE_REWARD_THRESHOLD=1.0`
  - Logs the active retention objective at launch.

- `skill/command/run_paper96_dynamic_opd_nolen_abcd_20260514.sh`
  - Protected B/D variants now use:

```text
USE_RETENTION=1
RETENTION_OBJECTIVE=nll
RETENTION_LOSS_WEIGHT=0.05
RETENTION_POSITIVE_REWARD_THRESHOLD=1.0
MAX_RETENTION_ROWS_PER_TASK=8
MAX_RETENTION_ROWS=24
```

Validation:

- `python -m py_compile scripts/train/opvec_update_gates_from_rollouts.py scripts/train/opvec_gated_grpo_loop.py scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
- `bash -n skill/command/run_qbank_c033333_gate_strategy.sh skill/command/run_paper96_dynamic_opd_nolen_abcd_20260514.sh`
- `PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_update_gates_objectives.py`
  - 18 tests passed.
- `DRY_RUN=1 RUN_TAG=20260514_nll_dry NUM_ITERS=1 NUM_PROMPTS=2 SAMPLES_PER_PROMPT=2 bash skill/command/run_paper96_dynamic_opd_nolen_abcd_20260514.sh`
  - Verified B/D update commands include `--use-retention --retention-objective nll`; protected variants now use `RETENTION_LOSS_WEIGHT=0.05`.

## 2026-05-15 Dynamic OPD Scale And Component Length Norm Split

Context:

- The balanced Paper96 run with `LENGTH_NORMALIZE_LOGPROB=1` made OPD numerically clean but too weak: gate deltas stayed around `1e-3`.
- The previous successful OPD run used sequence-level OPD and had much larger gate movement, but raw sequence loss can let long Memory/Code trajectories dominate.
- The new control rule separates overall step size from relative objective scale: LR controls total movement, OPD scale controls OPD-vs-GRPO pressure, and OPD rows are balanced by task.

Changes:

- `scripts/train/opvec_update_gates_from_rollouts.py`
  - Added component-specific length normalization:
    - `--opd-length-normalize-logprob` / `--no-opd-length-normalize-logprob`
    - `--retention-length-normalize-logprob` / `--no-retention-length-normalize-logprob`
    - Both default to legacy `--length-normalize-logprob` when unspecified.
  - Added `--opd-dynamic-scale`.
    - Before OPD backward, the updater runs a no-grad OPD scoring pass per task.
    - It estimates mean absolute OPD loss per task and computes a component scale:

```text
scale_task = clamp(
  ppo_loss_weight * target_ratio(recoverable_all_fail_rate_task)
  / (mean_abs_opd_loss_task + eps),
  scale_min,
  scale_max
)
```

  - Added `--opd-task-balanced-loss-scale`.
    - OPD row reduction becomes `1 / (3 * opd_rows_in_task)`.
    - Missing-task OPD is not redistributed to other tasks.
  - Summary and row logs now include:
    - `opd_scale_plan`
    - `opd_dynamic_scale`
    - `opd_row_loss_scale`
    - `opd_scale_target_ratio`
    - `opd_recoverable_all_fail_rate`
    - component-specific length norm settings.

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - Passes the new OPD/retention length-norm and OPD scale arguments to the updater.

- `scripts/train/opvec_gated_grpo_loop.py`
  - Same passthrough for the non-bake native loop.

- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - Added env controls:
    - `OPD_LENGTH_NORMALIZE_LOGPROB=inherit|0|1`
    - `RETENTION_LENGTH_NORMALIZE_LOGPROB=inherit|0|1`
    - `RETENTION_DYNAMIC_SCALE=0|1`
    - `RETENTION_TASK_BALANCED_LOSS_SCALE=0|1`
    - `RETENTION_SCALE_TARGET`
    - `OPD_DYNAMIC_SCALE=0|1`
    - `OPD_TASK_BALANCED_LOSS_SCALE=0|1`
    - `OPD_SCALE_TARGET_HIGH/MID/LOW/TAIL`
  - Launch logs now print active component length norm, retention scale, and OPD scale settings.

Follow-up refinement:

- Added dynamic scaling for retention NLL, matching the OPD scale mechanism.
  - `--retention-dynamic-scale` performs a no-grad retention NLL scoring pass.
  - `--retention-task-balanced-loss-scale` reduces retention as `1 / (3 * retention_rows_in_task)`.
  - `--retention-scale-target` sets the GRPO-relative target, default `0.5`.
  - Row logs now include `retention_dynamic_scale`, `retention_row_loss_scale`, and `retention_scale_target_ratio`.
  - Summary now includes `retention_scale_plan`.
- `skill/command/run_dynamic_opd_scale_20260515.sh` enables retention dynamic scaling by default for the four-run matrix.

Validation:

- `python -m py_compile scripts/train/opvec_update_gates_from_rollouts.py scripts/train/opvec_gated_grpo_bake_vllm_loop.py scripts/train/opvec_gated_grpo_loop.py`
- `bash -n skill/command/run_qbank_c033333_gate_strategy.sh`
- Dry-run verified that the update command includes:
  - `--no-opd-length-normalize-logprob`
  - `--retention-length-normalize-logprob`
  - `--retention-dynamic-scale`
  - `--retention-task-balanced-loss-scale`
  - `--opd-dynamic-scale`
  - `--opd-task-balanced-loss-scale`

## 2026-05-17 Norm-aware Init And CURE Code Calibration Blueprints

Context:

- Code reward and gate movement were weak even when code coefficients increased.
- Diagnostics showed the code task vector covers the same 196 modules as tool/memory but has much smaller total L2 norm.
- Formal CURE failures split into generation failures and generated-test selection failures, which the previous all-fail OPD pipeline did not distinguish.

Changes:

- Added `scripts/analysis/analyze_task_vector_norms.py`.
  - Reads OP-VEC `diagnostics.json` and `mode_manifest.json`.
  - Reports total/layer/block/module L2, code effective sparsity, and top code modules.
  - Writes deterministic norm-aware gate checkpoints:
    - `all_ones`
    - `all1_sqrt_weak_compensation`
    - `all1_linear_weak_compensation`
    - `sum1_equal_effective_l2`
    - `baseline_mean_effective_l2`
    - `global-coefficient` and `global-parameter` formats.
- Added `scripts/data/build_cure_code_calibration_pools.py`.
  - Reads case-level CURE results from the eval case browser.
  - Builds separate `generation`, `selection`, and `partial_edge` blueprint pools for Code calibration.
  - Defaults to blueprint/audit outputs and explicitly marks official CURE data as non-training leakage-risk material.
- Added config/report docs:
  - `docs/report/task_vector_norm_diagnostics_20260517.md`
  - `docs/config/20260517_norm_aware_code_calibration.md`
- Added smoke records showing `all_ones` is stable on 9 prompts while sqrt weak compensation hurts Code reward in the same tiny sample.

Validation:

- `python -m py_compile scripts/analysis/analyze_task_vector_norms.py scripts/data/build_cure_code_calibration_pools.py`

## 2026-05-17 Selected-Mode Gate Checkpoints

Context:

- We need to test whether the strongest model's mode-selection evidence can support structured pruning from `init=1`.
- The reasoning model's TAME selected modes are expert-specific and must not be reused to prune tool/memory/code deltas.

Changes:

- Added `scripts/analysis/build_selected_mode_gate_checkpoints.py`.
  - Converts expert-specific `selected_modes.json` files into OP-VEC `parameter` gate checkpoints.
  - Supports hard expert pruning: selected expert-param coefficients stay at `1.0`, non-selected coefficients become `0.0`.
  - Supports `--top-k-per-expert` for controlled top-k retention; if a source has fewer mapped modes than requested, the script supplements with per-expert delta-L2 ranked params from OP-VEC diagnostics.
  - Supports selected/ranked source formats including `selected`, `top_rows`, `plans`, `coefficients`, and `rows_path` JSONL summaries.
  - Supports 4-expert reasoning micro-addition: `tool=memory=code=1.0`, reasoning selected modes at `0.001`, other reasoning modes at `0.0`.
  - Writes audit Markdown beside each generated gate checkpoint.
- Added config doc:
  - `docs/config/20260517_selected_mode_pruning.md`
- Updated evaluation report:
  - Filled F/G final iter20 Code/CURE results in `docs/evaluation/20260517_defg_eval6.md`.

Validation:

- `PYTHONPATH=. python -m py_compile scripts/analysis/build_selected_mode_gate_checkpoints.py`
- Built selected-mode checkpoints under `/tmp/shared-storage/OnPolicy/data/init_gates/selected_mode_pruning_20260517`, including Tool/Memory top64 pruning with Code kept full.
- Ran 9-prompt vLLM smoke for expert-specific pruning, reasoning selected64@0.001, and Tool/Memory top64 + Code full.

## 2026-05-17 Eval-Targeted Calibration Builder

Context:

- The eval case browser showed that the old `paper96` calibration is too weakly aligned with BFCL-live Tool failures and CURE Code failures.
- We need a reproducible calibration manifest that reflects formal-eval ability without copying official eval prompts, answers, tests, or model outputs.

Changes:

- Added `scripts/data/build_eval_targeted_calibration.py`.
  - Reads case-study candidates from `/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/bfcl_live_calibration_candidates.jsonl`.
  - Builds a 96-row task-balanced manifest under `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517`.
  - Tool: mixes 16 paper96 ToolRL/RLLA source-anchor rows with 16 fresh BFCL-style synthetic rows using `reference.bfcl`, covering live/non-live parallel calls, default values, enum exactness, canonicalization, and distractor function selection.
  - Memory: reuses 32 paper96 HotpotQA-train memory rows as a stable anchor.
  - Code: selects 32 CodeContests-train rows by CURE-derived tags, storing executable source tests in `reference.metadata`.
  - Writes `summary.json`, `tool_synthetic_blueprints.jsonl`, `code_train_blueprints.jsonl`, and README for audit.
- Added report:
  - `docs/report/eval_targeted_calibration_20260517.md`

Validation:

- `PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile scripts/data/build_eval_targeted_calibration.py`
- Built `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517/eval_targeted96.prompts.jsonl`.
- Verified row counts are `tool=32`, `memory=32`, `code=32` with 96 unique prompt ids.
- Verified all 16 synthetic Tool reference responses score `success=True, reward=1.0` through the current `RewardRouter` / BFCL adapter.

Follow-up:

- Extended `scripts/data/build_eval_targeted_calibration.py` with optional Code source-anchor mixing:
  - `--code-source-count`
  - `--code-targeted-count`
- Defaults preserve the original builder behavior (`--code-source-count 0`), so existing reproduction commands and the first `eval_targeted96_20260517` data path are unchanged.
- Built the recommended mixed calibration manifest:
  - `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl`
  - Tool: 16 paper96 source anchors + 16 BFCL-style synthetic probes.
  - Memory: 32 paper96 HotpotQA-train anchors.
  - Code: 16 paper96 Code frontier anchors + 16 CURE-style targeted CodeContests probes.
- Updated `docs/report/eval_targeted_calibration_20260517.md` with the recommended data path, reproduction command, audit counts, and Code reward alignment settings.

Validation:

- `python -m py_compile scripts/data/build_eval_targeted_calibration.py`
- Built `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl`.
- Verified row counts are `tool=32`, `memory=32`, `code=32` with 96 unique prompt ids.
- Verified all 32 Code rows resolve to CodeContests source tests.
- Verified all 16 synthetic Tool reference responses score `success=True, reward_train=1.0` through the current `RewardRouter` / BFCL adapter.
# 2026-05-17 c1 layer-band bake config support

- 修改 `opvec/modeling/bake.py`：`create_bake_plan()` / `bake_checkpoint()` 新增可选 `layer_bands` 参数。默认不传时仍使用原来的 `DEFAULT_LAYER_BANDS`，不影响旧实验。
- 修改 `scripts/eval/opvec_bake_checkpoint.py`：从 config 的顶层 `layer_bands` 或 `modes.layer_bands` 读取自定义 layer-band，并传给 bake。用于 c1 的 `configs/gated_grpo_layer28.yaml`，避免 bake 阶段仍按 `early/mid/late` 查 band。
- 不修改 reward、rollout、update loss、OPD、retention、GRPO 或 gate optimizer 逻辑。

# 2026-05-17 frontier/retention random row sampling

- 修改 `scripts/train/opvec_update_gates_from_rollouts.py`：新增可选 `--sample-frontier-before-limit`、`--sample-retention-before-limit`、`--retention-shuffle-seed`。默认关闭，旧实验仍按原顺序截断。
- 新增 `--ignore-config-frontier-task-quota`：显式忽略 config 里的 `calibration.frontier_task_quota`，从而实现“不设置 frontier quota 就全量”。
- frontier 随机采样发生在 per-task quota 之前，例如每任务随机取 4 条后再进入 `task-interleaved` 顺序。
- retention 随机采样发生在 `max_retention_rows_per_task` / `max_retention_rows` 截断之前；未显式指定 retention seed 时复用 frontier seed。
- 修改 `skill/command/run_qbank_c033333_gate_strategy.sh`：暴露 `FRONTIER_SAMPLE_BEFORE_LIMIT`、`RETENTION_SAMPLE_BEFORE_LIMIT`、`RETENTION_SHUFFLE_SEED`、`FRONTIER_ROWS_PER_TASK`、`IGNORE_CONFIG_FRONTIER_TASK_QUOTA` 环境变量，并在启动日志中记录。wrapper 默认 `IGNORE_CONFIG_FRONTIER_TASK_QUOTA=1`，因此不设置 quota 时默认 full frontier。
# 2026-05-17 P0 State Distribution Utility

## 变更

- 新增 `scripts/analysis/build_rollout_state_distribution.py`。
- 新增 `configs/init_gates/ta_c033333_global.json` 与 `configs/init_gates/ta_init1_global.json`。

## 目的

服务论文 P0：从 rollout JSONL 与 expert rollout JSONL 统计 `frontier / recoverable / stable / unsolved`，把 calibration prompts 明确解释为 executable probes。

## 行为

- 只读 rollout，不改训练主流程。
- `stable`: current rollout 全 success。
- `frontier`: current rollout 有 reward 方差或成功/失败混合。
- `recoverable`: current all-fail 且 same-prompt expert rollout 有 verified positive。
- `unsolved`: current all-fail 且 expert 也没有 positive。
- 输出 JSON 与 Markdown，供 `docs/report/expertgym_72h` 和论文表格使用。

## 验证

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/analysis/build_rollout_state_distribution.py \
  scripts/eval/opvec_bake_checkpoint.py

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/build_rollout_state_distribution.py --help
```

# 2026-05-18 P0 SOTA calibration v2 data harness

## 变更

- 新增 `scripts/data/partition_calibration_bank.py`：
  - 从多个 seed manifest 读取候选池；
  - 按 `prompt_hash` 去重；
  - 按 task 和 `eval_targeted_calibration.role` 分层；
  - 生成 disjoint `train/monitor/guard` split；
  - 只处理数据 manifest，不改 reward、rollout、update 或训练主逻辑。
- 新增 `skill/command/build_20260518_sota_calib_v2.sh`：
  - 构建 Tool/Code 候选池；
  - 构建 HotpotQA memory trajectory 候选池；
  - 产出 `sota_calib_v2_20260518/{train128,monitor64,guard64}`。
- 新增 `skill/command/run_20260518_sota_v2_expert_rollouts.sh`：
  - 对 `train128` 生成 same-prompt expert rollouts；
  - 每个专家 rollout 写 coverage summary。
- 新增 `skill/command/run_20260518_p0_sota_v2.sh`：
  - 管理 P0 主实验 `train_gc` / `train_gp`；
  - 默认使用 `train128`、dynamic OPD、GRPO、retention；
  - 默认只作为新实验入口，不影响旧 paper96 / eval_targeted96 命令。

## 产物

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/
  train128.prompts.jsonl
  monitor64.prompts.jsonl
  guard64.prompts.jsonl
  summary.json
```

计数：

- `train128`: tool=32, memory=48, code=48
- `monitor64`: tool=16, memory=24, code=24
- `guard64`: tool=16, memory=24, code=24

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m py_compile scripts/data/partition_calibration_bank.py

bash -n skill/command/build_20260518_sota_calib_v2.sh
bash -n skill/command/run_20260518_sota_v2_expert_rollouts.sh
bash -n skill/command/run_20260518_p0_sota_v2.sh
```

已实际构建 `sota_calib_v2_20260518`，schema 校验通过。

# 2026-05-18 Recoverable-code calibration filter

## 变更

- 新增 `scripts/data/build_recoverable_code_calibration.py`。
- 功能：从同 prompt expert rollouts 中统计 verified positive，只保留有 expert positive 的 Code 行进入训练 split；Tool/Memory 按给定数量保留。
- 默认不参与任何旧训练命令，必须显式调用。

## 目的

`sota_calib_v2_20260518/train128` 的 Code 部分中，两个 code expert 合并后只有 `21/48` 有 positive。直接训练会让 dynamic OPD 的 Code 信号过稀，导致主实验仍然“Code gate/reward 不涨”。这个 filter 把 hard Code 行留在 monitor/guard，把训练集收束到 verified recoverable directions。

## 产物

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/
  train_recoverable101.prompts.jsonl
  train_recoverable101.prompts.summary.json
```

计数：

- Tool: 32
- Memory: 48
- Code: 21
- Total: 101

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m py_compile scripts/data/build_recoverable_code_calibration.py

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_recoverable_code_calibration.py \
  --input /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/train128.prompts.jsonl \
  --output /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl \
  --expert-rollout /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/code_expert_reasonflux_coder7b_sota_v2_train128_s4_seed20260518.jsonl \
  --expert-rollout /tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_sota_v2_train128_s4_seed20260518.jsonl \
  --tool-count 32 --memory-count 48 --code-count -1 --seed 20260518
```

# 2026-05-18 SOTA monitor64 eval wrapper

## 变更

- 新增 `skill/command/run_20260518_sota_monitor64_eval.sh`。
- 功能：对任意 baked HF policy 在 `sota_calib_v2_20260518/monitor64.prompts.jsonl` 上执行 vLLM rollout，并用 `scripts/eval/summarize_rollouts.py` 生成 summary。
- 默认不影响训练；仅作为 checkpoint 筛选和三任务 proxy sanity check。

## 用法

```bash
MODEL_PATH=/path/to/baked_policy \
RUN_ID=my-model-monitor64 \
GPU_LIST=0 \
bash skill/command/run_20260518_sota_monitor64_eval.sh
```

## 主读数

```text
summary.json:
  task_stats.tool.mean_reward / success_rate
  task_stats.memory.mean_reward / success_rate
  task_stats.code.mean_reward / success_rate
```

## 验证

```bash
bash -n skill/command/run_20260518_sota_monitor64_eval.sh
```

# 2026-05-18 Four-expert layer-band bake compatibility

## 变更

- 修复 `opvec/modeling/bake.py` 中 layer-band bake 的专家列表硬编码。
- 之前 layer-band bake 侧会通过三专家 `project_gates()` 和三专家 `_band_gate_values()` 解析 `tool/memory/code`，当 manifest 包含第四个 `reasoning` expert 时，`reasoning_residual` 会被丢掉。
- 现在 layer-band bake 按 mode manifest 的 `expert_names` 做 common+residual 零均值投影，因此可用于 `tool/memory/code/reasoning` 四专家配置。
- `scripts/train/opvec_update_gates_from_rollouts.py` 的 `--train-coefficient` 投影也改为读取 `gate_manager.expert_names`，因此可冻结/解冻 `*.reasoning` 等非三专家系数。
- 新增可选 expert-specific trust region：
  - updater: `--max-coefficient-delta-from-init-by-expert reasoning=0.002`
  - launcher env: `MAX_COEFF_DELTA_BY_EXPERT=reasoning=0.002`
  - 默认空，不影响既有实验；设置后只覆盖指定 expert 的最大位移。
- `scripts/modes/build_constant_gate_checkpoint.py` 新增 `--expert-value EXPERT=VALUE`，可直接生成 `tool/memory/code=1.0, reasoning=0.001` 这类异尺度 init checkpoint。
- 三专家默认路径语义不变。

## 背景

DeepSeek-R1-Distill-Qwen-7B 的 task vector 范数远大于其他 experts，应作为小幅 reasoning/code prior 学习。要做 per-layer reasoning gate，bake 必须保留第四 expert 的 layer-band 系数。

## 验证

系统环境没有 `pytest`，已用 unittest 直接跑新增/既有 bake 测试：

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  tests/test_bake_global_coefficients.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  tests/test_gated_grpo_trust_region.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python - <<'PY'
import json, tempfile
from pathlib import Path
from scripts.modes.build_constant_gate_checkpoint import build_constant_gate_checkpoint

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "mode_manifest.json"
    p.write_text(json.dumps({
        "expert_names": ["tool", "memory", "code", "reasoning"],
        "basis_entries": [{"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool"}],
    }))
    gates, _ = build_constant_gate_checkpoint(
        mode_manifest=p,
        parameterization="parameter",
        value=1.0,
        expert_names=("tool", "memory", "code", "reasoning"),
        expert_values={"reasoning": 0.001},
    )
    assert gates["model.layers.0.mlp.down_proj.weight::reasoning"] == 0.001
print("expert override smoke OK")
PY

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo_reasoning_layer28.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json \
  --gate-checkpoint /tmp/shared-storage/OnPolicy/data/init_gates/r1_scaled_20260518/all1_r1_z001.layer-band.json \
  --output /tmp/shared-storage/OnPolicy/data/init_gates/r1_scaled_20260518/plan_only_bake \
  --plan-only
```

结果：

```text
tests/test_bake_global_coefficients.py: Ran 2 tests, OK
tests/test_gated_grpo_trust_region.py: Ran 3 tests, OK
OK
```

# 2026-05-18 Code P0 v3 calibration bank builder

## 变更

- 新增 `scripts/data/build_code_p0_calibration_bank.py`。
- 新增复现入口 `skill/command/build_20260518_code_p0_v3.sh`。
- 新增 expert rollout 入口 `skill/command/run_20260518_code_p0_v3_expert_rollouts.sh`，专门生成 Code P0 v3 同 prompt 的 ReasonFlux / DeepSeek positive 轨迹。
- 新增 recoverable 子集入口 `skill/command/build_20260518_code_p0_v3_recoverable.sh`，复用现有 `build_recoverable_code_calibration.py` 筛出专家能做对的 Code OPD rows。
- 新增 `scripts/data/merge_rollout_shards.py`，用于按 prompt manifest 顺序合并多卡 rollout shard，并输出去重 summary。
- `scripts/data/build_recoverable_code_calibration.py` 的 role 统计增加 `code_p0_calibration.role` / `reference.metadata.code_bank_role` fallback；旧 eval-targeted 统计不变。
- 新增配置说明 `docs/config/20260518_code_p0_v3.md`。
- 功能：只从 CodeContests train 构建 Code-only `train/monitor/guard` bank，不修改旧 paper96、sota_calib_v2、recoverable101 数据。

## 数据语义

每条 Code row 显式写入：

```text
reference.metadata.test_input/test_output      # 当前训练 reward 实际读取
reference.metadata.reward_test_input/output    # 同上，显式审计字段
reference.metadata.guard_test_input/output     # 同题 held-out test slice
reference.metadata.code_bank_role              # generation/frontier/partial_edge/stable
```

默认输出：

```text
/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/
  train_code64.prompts.jsonl
  monitor_code32.prompts.jsonl
  guard_code32.prompts.jsonl
  code_p0_blueprints.jsonl
  summary.json
  README.md
```

## 验证

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/data/build_code_p0_calibration_bank.py

bash skill/command/build_20260518_code_p0_v3.sh
```

结构校验：

```text
train rows 64 unique_task_ids 64 bad 0
monitor rows 32 unique_task_ids 32 bad 0
guard rows 32 unique_task_ids 32 bad 0
total unique 128
```

当前 expert rollout / recoverable 结果：

```text
ReasonFlux-Coder-7B: covered 29/64, mean reward 0.4089
DeepSeek-R1-Distill-Qwen-7B merged: covered 31/64, mean reward 0.4306
recoverable Code rows: 36/64
recoverable role counts: generation 7, frontier 15, partial_edge 11, stable 3
```

# 2026-05-18 R1 expert coefficient absolute bounds

## 变更

- `scripts/train/opvec_update_gates_from_rollouts.py` 新增可选参数：

```text
--coefficient-bound-by-expert reasoning=0.0:0.003
```

- 该参数对 effective coefficient 生效，即 common/residual 参数化下实际 bake 的专家系数。
- 默认不传时完全不进入新增边界逻辑，不改变旧实验路径。
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` 增加参数透传。
- `skill/command/run_qbank_c033333_gate_strategy.sh` 增加环境变量：

```text
COEFF_BOUND_BY_EXPERT=reasoning=0.0:0.003
```

- 新增 `skill/command/run_20260518_r1_codep0_bounded_sanity.sh`，默认在 R1 Code P0 sanity 上启用 `reasoning=0.0:0.003` 全程绝对边界。
- `scripts/eval/summarize_gate_strategy_run.py` 和 `scripts/monitor/opvec_run_monitor.py` 改为从 gate keys 动态推断 expert 名称，支持 `tool/memory/code/reasoning` 四专家，不再把 R1 实验误解析成三专家。

## 背景

R1 sanity 暴露了一个语义问题：

```text
--max-coefficient-delta-from-init-by-expert reasoning=0.002
```

在 bake-vLLM loop 中每轮 update 的 init checkpoint 是上一轮 checkpoint，因此它是逐轮 trust region，不是全程绝对边界。R1 delta 范数远大于其他 task vector，需要额外的绝对系数边界来保证可复现、可审查。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/train/opvec_update_gates_from_rollouts.py \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  scripts/eval/summarize_gate_strategy_run.py \
  scripts/monitor/opvec_run_monitor.py

bash -n skill/command/run_qbank_c033333_gate_strategy.sh skill/command/run_20260518_r1_codep0_sanity.sh

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_gated_grpo_trust_region.py
```

结果：

```text
tests/test_gated_grpo_trust_region.py: Ran 5 tests, OK
```

# 2026-05-18 Skip old-logprob fill for OPD-only NLL updates

## 变更

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` 不再无条件向 update 阶段传：

```text
--fill-missing-old-logprob
```

- 现在仅在以下场景传入：

```text
ppo_loss_weight != 0
或
use_retention=1 且 retention_objective=kl
```

## 原因

`expD_r1scaled_*` 当前是 OPD + NLL retention-only：

```text
ppo_loss_weight=0
retention_objective=nll
```

这一路不使用 PPO ratio，也不使用 KL retention，因此 old logprob 不参与 loss。旧逻辑仍会先用 HF 模型对 rollout 样本补 `old_logprob`，造成无效前向计算。

## 影响范围

- GRPO/PPO 路径不变：`ppo_loss_weight != 0` 时仍会填 old logprob。
- KL retention 路径不变：`retention_objective=kl` 时仍会填 old logprob。
- OPD-only + NLL retention 后续实验会更快，loss 语义不变。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py

bash -n scripts/train/run_4expert_r1scaled.sh
```

结果：

```text
ok
```
## 2026-05-18: 新增 direct layer-band coefficient gate

目的：支持一个干净对照实验，保留 layer-band 的层段划分，但不使用 `common + residual` 参数化，而是直接学习每个 band 上每个 expert 的 coefficient。

改动范围：

- `opvec/modeling/gate_parameters.py`
  - 新增 `TorchLayerBandCoefficientManager`。
  - 新参数化名：`layer-band-coefficient`。
  - gate 形式：`early.tool`、`early.memory`、`early.code`、`mid.*`、`late.*`。
  - 不改变已有 `global` / `layer-band` / `parameter` / `global-parameter` / `global-coefficient` 行为。
- `scripts/modes/build_constant_gate_checkpoint.py`
  - 支持生成 direct layer-band 初始化 gate。
- `scripts/modes/build_zero_gate_checkpoint.py`
  - 支持 direct layer-band zero gate。
- `scripts/train/opvec_update_gates_from_rollouts.py`
  - CLI `--gate-parameterization` 接受 `layer-band-coefficient`。
  - direct layer-band 使用已有 `raw_coefficients` trust-region / prior 逻辑。
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`、`scripts/train/opvec_collect_hf_rollouts.py`、`scripts/train/opvec_gated_grpo_loop.py`
  - 只扩展 CLI choices，不改训练目标。
- `skill/command/run_qbank_c033333_gate_strategy.sh`
  - 支持 `STRATEGY=layer-band-coefficient`，默认 LR/ prior/ delta 沿用 `layer-band`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  opvec/modeling/gate_parameters.py \
  scripts/train/opvec_update_gates_from_rollouts.py \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  scripts/train/opvec_collect_hf_rollouts.py \
  scripts/modes/build_constant_gate_checkpoint.py \
  scripts/modes/build_zero_gate_checkpoint.py

bash -n skill/command/run_qbank_c033333_gate_strategy.sh

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_gated_linear.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_bake_global_coefficients.py
```

## 2026-05-18: Monitor 支持续训 run 的显示迭代 offset

目的：续训实验的物理目录仍从 `iter_001` 开始，但前端曲线需要接在原 run 后面显示，例如 3band continue10 应显示为 `iter_011..iter_020`。

改动：

- `scripts/monitor/opvec_run_monitor.py`
  - 新增 CLI：`--run-iteration-offset RUN_ID=OFFSET`。
  - offset 只改 API / 前端中的 `iteration` label。
  - 原始目录名保留在 `physical_iteration`，不改训练产物、不影响复现路径。

示例：

```bash
python scripts/monitor/opvec_run_monitor.py \
  --run-dir r1_3band_continue10=/path/to/run \
  --run-iteration-offset r1_3band_continue10=10
```

备注：`BFCL` 环境没有 `pytest`，因此这里用标准库 `unittest` 运行这两个测试文件。

## 2026-05-18: Monitor 支持 reasoning gate 显示

问题：`scripts/monitor/opvec_run_monitor.py` 的前端表格和 coefficient 下拉框写死了 `tool/memory/code`，导致 4-expert R1 run 虽然后端 gate JSON 里有 `reasoning_residual`，前端仍不显示 reasoning gate。

改动：

- 后端 `_infer_experts()` 支持 `__global__::<expert>` 和 direct layer-band key `band.expert`。
- 后端 `_effective_coefficients()` 支持 direct layer-band coefficient，不再把 `band.tool` 误读成 common/residual fallback。
- `gate_stats.expert_delta` 改为基于 loop manifest 首轮 `input_gate_checkpoint` 的真实初始 gate 计算；因此 R1 的 `reasoning` delta 以 `0.0` 为初始点，而不是错误地以默认 `1/3` 为初始点。
- 前端 gate 总览、Gate/Update 表、source 聚合表、coefficient selector 改为动态 expert 列表；R1 run 会显示 `reasoning`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile scripts/monitor/opvec_run_monitor.py
curl -sS 'http://127.0.0.1:8794/api/state?max_prompt_rows=1'
```

当前 API 已包含：

```text
coefficient/code
coefficient/memory
coefficient/reasoning
coefficient/tool
```

## 2026-05-18: 新增 hierarchical layer-band gate

目的：验证 `28layer` 细粒度 gate 训练不动是否来自缺少全局 expert 聚合项。新增 `layer-band-parameter`，形式为：

```text
coefficient[band, expert] = global[expert] + residual[band, expert]
```

行为边界：

- 不覆盖已有 `layer-band`、`layer-band-coefficient`、`global-parameter`。
- `gate_values` 同时写出 `__global__::<expert>` 和 `band.expert` 的有效系数。
- bake 时忽略只有诊断用途的 `__global__::<expert>`，按 `band.expert` 烘焙实际模型；因此不会误判成 588 parameter gate。
- trust-region / prior / coefficient bounds 复用已有 `raw_global_coefficients + raw_residual_coefficients` 路径。

主要改动：

- `opvec/modeling/gate_parameters.py`
  - 新增 `TorchLayerBandParameterCoefficientManager`。
  - 新增 aliases：`layer-band-parameter`、`layer-band-hierarchical`、`hierarchical-layer-band` 等。
- `opvec/modeling/bake.py`
  - `__global__::<expert>` 不再单独触发 parameter bake。
  - 当 gate 同时含 `__global__::<expert>` 和 `band.expert` 时，bake summary 标记为 `layer-band-parameter`。
- 训练入口：
  - `scripts/train/opvec_update_gates_from_rollouts.py`
  - `scripts/train/opvec_collect_hf_rollouts.py`
  - `scripts/train/opvec_gated_grpo_loop.py`
  - `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
  - `skill/command/run_qbank_c033333_gate_strategy.sh`
- 初始化脚本：
  - `scripts/modes/build_constant_gate_checkpoint.py`
  - `scripts/modes/build_zero_gate_checkpoint.py`
- R1 实验脚本：
  - `scripts/train/run_4expert_r1scaled.sh`
  - 新增 `PHASE=layer28_hier`，默认 `LAYER28_STRATEGY=layer-band-parameter`、`LAYER28_LR=0.25`。

验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_gated_linear.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_bake_global_coefficients.py
```
# 2026-05-18: SOTA recovery calibration v3

新增文件：

- `scripts/data/build_sota_recovery_calibration_v3.py`
- `skill/command/build_20260518_sota_recovery_calib_v3.sh`
- `skill/command/run_20260518_sota_recovery_v3.sh`
- `docs/config/20260518_sota_recovery_calib_v3.md`
- `docs/report/20260518_sota_recovery_calib_v3.md`

功能：

- 构建 `sota_recovery_calib_v3_20260518`，把 train 与 monitor/guard 分工拆开。
- `train128` 使用 verified-recoverable rows，提升 OPD/frontier/retention 的训练信号密度。
- `monitor64` / `guard64` 使用 harder audit rows，尤其是 Code P0 hard rows，防止 train proxy 过拟合。
- 训练脚本显式接入 sota_v2 与 code_p0 两套 expert rollouts，避免 v3 train 中 Code P0 rows 没有 OPD positive。
- v3 训练脚本把 init gates 写到 v3 bank 自己的 `init_gates/`，不覆盖公共 question-bank init。

验证：

- `build_sota_recovery_calibration_v3.py` 通过 `py_compile`。
- `build_20260518_sota_recovery_calib_v3.sh` / `run_20260518_sota_recovery_v3.sh` 通过 `bash -n`。
- 已生成：
  - `/tmp/shared-storage/OnPolicy/data/calibration/sota_recovery_calib_v3_20260518/train128.prompts.jsonl`
  - `/tmp/shared-storage/OnPolicy/data/calibration/sota_recovery_calib_v3_20260518/monitor64.prompts.jsonl`
  - `/tmp/shared-storage/OnPolicy/data/calibration/sota_recovery_calib_v3_20260518/guard64.prompts.jsonl`

注意：

- 曾在 dry-run 中发现底层 launcher 会把 init gate 写入公共 question-bank 路径；已恢复公共 `init_layer_band_parameter_c033333.json` 为 `0.3333333333333333`，并在 v3 launcher 中设置 `QB=$BANK` 进行隔离。
## 2026-05-19: Correct DeepSeek-R1 Delta Base Support

Problem:

- `DeepSeek-R1-Distill-Qwen-7B` is distilled from `Qwen2.5-Math-7B`, not `Qwen2.5-7B-Instruct`.
- The old R1 mode artifacts subtracted the Instruct base, so the reasoning delta mixed true reasoning behavior with the Math/Instruct base gap.

Changes:

- `opvec/modes/build_modes.py` now supports optional per-expert delta bases:

```yaml
models:
  base: /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct
  delta_bases:
    reasoning: /mnt/cache/wuruixiao/models/Qwen2.5-Math-7B
```

- Default behavior is unchanged: experts without `delta_bases` still subtract `models.base`.
- Added `configs/gated_grpo_4expert_r1math_layer28.yaml` for correct-R1 layer28 experiments.
- Added reproducible scripts:
  - `skill/command/build_20260519_r1math_modes.sh`
  - `skill/command/run_20260519_r1math_L_experiments.sh`
  - `skill/command/orchestrate_20260519_r1math_L1_L2.sh`
- Added experiment table: `docs/config/20260519_r1math_L_experiments.md`.

Validation:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  opvec/modes/build_modes.py \
  scripts/modes/build_opvec4_modes.py \
  scripts/modes/build_scaled_r1_modes.py

bash -n \
  skill/command/build_20260519_r1math_modes.sh \
  skill/command/run_20260519_r1math_L_experiments.sh \
  skill/command/orchestrate_20260519_r1math_L1_L2.sh

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python scripts/modes/build_opvec4_modes.py \
  --config configs/gated_grpo_4expert_r1math_layer28.yaml \
  --output-dir /tmp/shared-storage/OnPolicy/modes/opvec4_r1math_dryrun_20260519 \
  --dry-run
```

## 2026-05-19: L-series Resume and Layer-Band-Parameter Freeze Fix

Problem:

- L1 needed clean resume after an external GPU process blocked the second rollout shard.
- L2 freezes reasoning by `TRAIN_COEFFICIENTS=*.tool,*.memory,*.code`; the update-side train-coefficient projection only handled old global / common-residual managers and could fall through to `raw_common` on direct `layer-band-parameter`.

Changes:

- `skill/command/run_qbank_c033333_gate_strategy.sh` now exposes `START_ITERATION` and passes it to `opvec_gated_grpo_bake_vllm_loop.py`.
- `skill/command/run_20260519_r1math_L_experiments.sh` forwards `START_ITERATION` and respects an external `INIT_GATE_CHECKPOINT`, enabling deterministic continuation from a completed gate checkpoint.
- `scripts/train/opvec_update_gates_from_rollouts.py` now supports train-coefficient projection for managers with `raw_global_coefficients` + `raw_residual_coefficients`:
  - trainable expert coefficients keep the current effective value;
  - frozen coefficients are restored to the anchor gate value;
  - global + residual tensors are rewritten consistently, then projected.
- `_anchor_expert_coefficients` now also understands direct coefficient checkpoint keys such as `__global__::reasoning` and `layer0.reasoning`; this prevents frozen coefficients from falling back to the old `common=0.5` default when resuming `layer-band-parameter` runs from a gate checkpoint.

Default impact:

- Runs without `TRAIN_COEFFICIENTS` or without resume variables are unchanged.
- Reward, OPD construction, retention, task weights, loss weights, and gate parameterization semantics are unchanged.

Validation:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/train/opvec_update_gates_from_rollouts.py

bash -n \
  skill/command/run_qbank_c033333_gate_strategy.sh \
  skill/command/run_20260519_r1math_L_experiments.sh
```

## 2026-05-19: L4 BFCL Tool Augmentation and Dynamic OPD Require-All Guard

Problem:

- L1/L2/L3 showed a repeated failure mode: when Tool no longer contributed dynamic OPD rows, Memory/Code OPD could keep pushing gates and Tool proxy collapsed.
- L4 needs Tool calibration closer to BFCL live/non-live evaluation while preserving the old launcher behavior for all previous experiments.

Changes:

- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` adds `--dynamic-opd-require-all-tasks`.
  - Default is off, so existing experiments keep the exact previous path.
  - When on, the loop reads `opd_distill_from_allfail.summary.json` after dynamic OPD construction.
  - If any task listed by `--dynamic-opd-tasks` has zero selected OPD rows, the update command receives no dynamic OPD rollout for that iteration.
  - Retention, rollout reward, OPD builder, task weights, and static OPD rollouts are unchanged.
- `skill/command/run_qbank_c033333_gate_strategy.sh` exposes `DYNAMIC_OPD_REQUIRE_ALL_TASKS=0|1`.
- `scripts/data/build_bfcl_tool_calibration.py` builds a reproducible BFCL Tool augmentation:
  - 8 non-live BFCL rows: `parallel=4`, `parallel_multiple=4`;
  - 8 live BFCL rows: `live_parallel=4`, `live_parallel_multiple=4`;
  - live rows are selected from existing model-failure cases, satisfying the requested hard-live requirement;
  - an official-answer expert rollout JSONL is emitted so dynamic OPD can use these BFCL prompts as positive Tool anchors.
- `skill/command/run_20260519_r1math_L_experiments.sh` adds `PHASE=L4`, using L1 settings plus:
  - merged 112-row manifest with BFCL Tool rows;
  - BFCL official-answer expert rollout;
  - `DYNAMIC_OPD_REQUIRE_ALL_TASKS=1`.

Artifacts:

- Prompt-only BFCL Tool rows:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_nonlive8_live8_seed20260519.prompts.jsonl`
- BFCL official-answer expert rollout:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_official_answer_expert_rollouts_seed20260519.jsonl`
- L4 merged manifest:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/qbank_c033333_paper96_plus_bfcl_tool16_seed20260519.prompts.jsonl`

Validation:

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  scripts/data/build_bfcl_tool_calibration.py

bash -n \
  skill/command/run_qbank_c033333_gate_strategy.sh \
  skill/command/run_20260519_r1math_L_experiments.sh

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/data/build_bfcl_tool_calibration.py

PHASE=L4 GPU_LIST=0,1 DRY_RUN=1 \
  bash skill/command/run_20260519_r1math_L_experiments.sh
```

## 2026-05-19: Optional Tool Behavior-Span Null-Space Projection

Problem:

- L1/L2/L3 showed strong Memory/Code gate growth but Tool proxy collapsed in later iterations.
- NLL retention on Tool all-success rows produced much smaller gradients than OPD and did not strongly preserve the tool-call behavior span.
- We need an optional gradient-control baseline that protects Tool behavior without changing the default training path.

Changes:

- `scripts/train/opvec_update_gates_from_rollouts.py` adds optional Tool null-space projection:
  - enabled only by `--tool-nullspace-gate-gradients`;
  - default off, so existing experiments do not enter this branch;
  - collects Tool positive rows from current retention rows plus a fixed replay rollout file;
  - recomputes behavior-span NLL gate gradients on selected Tool trajectories;
  - builds an SVD basis from those gradients;
  - projects the currently accumulated total gate gradient away from the protected span before gradient clipping and optimizer step.
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` forwards the null-space args to update.
- `skill/command/run_qbank_c033333_gate_strategy.sh` exposes:
  - `TOOL_NULLSPACE_GATE_GRADIENTS`
  - `TOOL_NULLSPACE_REPLAY_ROLLOUT`
  - `TOOL_NULLSPACE_ROWS`
  - `TOOL_NULLSPACE_MIN_ROWS`
  - `TOOL_NULLSPACE_RANK`
  - `TOOL_NULLSPACE_EPS`
  - `TOOL_NULLSPACE_POSITIVE_REWARD_THRESHOLD`
- `scripts/data/build_tool_nullspace_calibration_v1.py` creates the v1 calibration bank:
  - Tool: 16 original paper96 rows + 16 BFCL live historical-success rows;
  - Memory: original paper96 32 rows;
  - Code: original paper96 32 rows + 8 CURE eval-aligned hard-vs-TA anchors.
- `skill/command/run_20260519_tool_nullspace_v1.sh` is the dedicated M1 launcher.
- `tests/test_pcgrad_gate_gradients.py` adds unit tests for the null-space projection math and Tool behavior span extraction.

Default impact:

- Default training is unchanged unless `--tool-nullspace-gate-gradients` is passed.
- Reward, dynamic OPD selection, retention selection, task weights, OPD/retention loss formulas, PCGrad, optimizer semantics, and gate parameterization are unchanged.

Artifacts:

- Config/report: `docs/config/20260519_tool_nullspace_v1.md`
- Merged prompt manifest:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/tool32_memory32_code40_toolnullspace_seed20260519.prompts.jsonl`
- Tool replay rollout:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1/toolnullspace_tool_replay_rollouts_seed20260519.jsonl`

Validation:

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/train/opvec_update_gates_from_rollouts.py \
  scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  scripts/data/build_tool_nullspace_calibration_v1.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_pcgrad_gate_gradients.py
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_update_gates_objectives.py
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_gated_grpo_trust_region.py
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_gated_grpo_utils.py
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_bake_global_coefficients.py
```

`pytest` is not installed in the available BFCL/easyrl/system Python environments, so validation used the direct `unittest` script entrypoints.

## 2026-05-20: Round13 Formal-Code Eval-Leak TRC Diagnostic Builder

Purpose:

- Add a clean diagnostic builder for testing whether TRC hidden-state alignment can learn formal LiveBench/LiveCodeBench Code ability when the calibration trajectories are explicitly aligned to the formal eval distribution.
- This is eval-leak diagnostic data only; it must not be reported as a paper-main non-leak result.

Changes:

- Added `scripts/trc/build_trc_round13_evalleak_code16_calibration.py`.
- The builder emits two isolated calibration banks under `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round13_evalleak_code16`:
  - `rfmem_only`: copies stable Tool32/Memory32 rows and adds only ReasonFlux/MemoryAgent positive Code16 trajectories.
  - `all_with_r1`: copies stable Tool32/Memory32 rows and adds all positive Code16 trajectories, mapping DeepSeek/R1 rows to the `reasoning` gate expert.
- Code rows map source trajectory to the matching gate expert:
  - ReasonFlux -> `code`
  - MemoryAgent -> `memory`
  - DeepSeek/R1 -> `reasoning`
- Code responses are compacted to `critical_reasoning_span + final_code_span`; span metadata is written to `sample_metadata.ability_spans`.
- The default reasoning context is 600 chars so that the final code block remains inside the stable 512-token response budget used by the TRC train rerun.

Default impact:

- No existing training, reward, GRPO/OPD, retention, or TRC train path is changed.
- The new builder is standalone and only runs when explicitly invoked.

Config:

- `docs/config/hiddenstate/20260520_round13_evalleak_code16.md`

Validation:

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/trc/build_trc_round13_evalleak_code16_calibration.py
```

## 2026-05-20: Optional Code Negative-Contrastive TRC Diagnostic

Purpose:

- Add a default-off diagnostic path for Code: use same-prompt failed CURE generations as negative trajectories, so TRC can learn to approach expert-positive residuals while not matching failed-response residuals.
- This is meant to test whether Code needs execution-aware contrastive signal rather than positive-only hidden-state alignment.

Changes:

- `scripts/trc/train_trc_layer_gates.py`
  - Added optional CLI args:
    - `--contrastive-negative-loss-weight` (default `0.0`, branch disabled)
    - `--contrastive-negative-margin` (default `0.05`)
    - `--contrastive-negative-response-key` (default `negative_response`)
    - `--contrastive-negative-task` (repeatable allowlist)
  - When enabled and a row has `negative_response`, the trainer computes a hinge loss:
    `relu(margin + positive_residual_loss - negative_residual_loss)`.
  - The negative branch does extra base/expert/merged hidden-state forwards for that failed response, so it is slower by design.
  - Default training commands with weight `0.0` do not enter this branch.
- `skill/command/run_20260519_trc_round_train_one.sh`
  - Exposes the new contrastive args through `CONTRASTIVE_NEGATIVE_*` environment variables.
  - Defaults keep old behavior unchanged.
- `scripts/trc/build_trc_round14_code_contrast_calibration.py`
  - Builds a calibration JSONL by copying a positive TRC bank and adding `negative_response` to Code rows from failed CURE temp outputs.
  - Default output:
    `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_code_contrast_v1/trc_expert_trajectories.jsonl`
- `scripts/trc/build_trc_round14_mixed_train_contrast_calibration.py`
  - Builds a 96-row mixed bank where Code keeps original CodeP0 train prompts and adds a small formal contrast hard-anchor slice.
  - Default output:
    `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_mixed_train_contrast_v1/trc96_expert_trajectories.jsonl`
  - Default Code composition: 24 CodeContests_train / CodeP0 rows + 8 formal contrast rows with `negative_response`.

Default impact:

- Existing reward, GRPO/OPD, retention, Tool/Memory TRC, positive TRC residual loss, task balancing, task weights, and checkpoint selection are unchanged when `CONTRASTIVE_NEGATIVE_LOSS_WEIGHT=0.0`.

Validation:

```bash
PYTHONPATH=. python -m py_compile \
  scripts/trc/train_trc_layer_gates.py \
  scripts/trc/build_trc_round14_code_contrast_calibration.py \
  scripts/trc/build_trc_round14_mixed_train_contrast_calibration.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/trc/build_trc_round14_code_contrast_calibration.py

PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/trc/build_trc_round14_mixed_train_contrast_calibration.py
```

## 2026-05-20: Round16 Non-Leak CodeP0 Pass/Fail Contrast Builder

Purpose:

- Build a paper-main-safe Code contrast calibration bank without formal Code eval anchors.
- Keep Tool/Memory from the stable R5A bank, but make all Code rows come from CodeP0-v3 `CodeContests_train` expert rollouts.
- Support the main Code hypothesis: same-prompt pass code block direction should dominate fail code block direction.

Changes:

- Added `scripts/trc/build_trc_round16_nonleak_code_contrast_calibration.py`.
- Default output:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl`
- The builder emits:
  - Tool32 from stable R5A Tool bank.
  - Memory32 from stable R5A late3 Memory bank.
  - Code22 pass/fail rows from CodeP0-v3 train rollouts, each with `negative_response`.
  - Code10 positive-fill rows from CodeP0-v3 train rollouts.
- Strict RF+DS-merged non-leak data only provides 22 unique pass/fail prompts, so the default is explicitly `22 + 10` rather than silently duplicating prompts or importing formal eval anchors.
- All Code rows keep `expert=code`; DeepSeek fallback rows are marked in metadata for audit.
- The builder now supports explicit `--code-rollout` overrides. This is used by R16C to build a ReasonFlux-only contrast bank.
- Positive-fill selection prefers prompts outside the contrast set, then allows additional successful trajectories if a strict single-source bank does not have enough unique prompts. This keeps RF-only probes possible while recording unique prompt counts in `summary.json`.

Default impact:

- No existing train path changes.
- The new builder is standalone and only affects experiments that explicitly set `CALIB` to its output.
- The optional contrast branch in `train_trc_layer_gates.py` remains default-off.

Config:

- `docs/config/20260520_trc_round16_nonleak_code_contrast.md`

Validation:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/trc/build_trc_round16_nonleak_code_contrast_calibration.py
```

RF-only probe:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/trc/build_trc_round16_nonleak_code_contrast_calibration.py \
  --output-dir /tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_rfonly_code_contrast_v1 \
  --code-rollout /tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl \
  --code-contrast-count 16 \
  --code-positive-fill-count 16
```

## 2026-05-20: Optional TRC Prompt Expert-Residual Loss

Purpose:

- Test the hypothesis that prompt hidden states contain task-vector understanding, so pulling prompt tokens back to base can suppress useful expert behavior.
- Keep the old path unchanged by default while allowing a clean R17 comparison:
  - no prompt-base drift;
  - no prompt-base drift plus expert-residual alignment on prompt tokens.

Changes:

- Updated `scripts/trc/train_trc_layer_gates.py`.
- Added CLI flags:
  - `--prompt-residual-weight`, default `0.0`;
  - `--prompt-residual-tokens`, default `256`.
- When `--prompt-residual-weight > 0`, the trainer computes `prompt_residual_loss` on the prompt-tail token positions using the same `hidden_residual_loss` objective as output span residual alignment:
  `merged_hidden - base_hidden -> expert_hidden - base_hidden`.
- `prompt_residual_loss` is logged at row, task-summary, and epoch-summary levels.
- Existing `base_drift_loss` remains controlled only by `--beta-base`; setting `BETA_BASE=0` cleanly disables prompt-to-base drift.
- Updated `skill/command/run_20260519_trc_round_train_one.sh` to record and pass:
  - `PROMPT_RESIDUAL_WEIGHT`;
  - `PROMPT_RESIDUAL_TOKENS`;
  - `PROMPT_DRIFT_TOKENS`.

Default impact:

- With `PROMPT_RESIDUAL_WEIGHT=0.0`, the new prompt residual branch is not used.
- Existing R16 / earlier TRC experiments keep the same loss path unless explicitly setting the new env vars or `BETA_BASE=0`.

Validation:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m py_compile scripts/trc/train_trc_layer_gates.py

bash -n skill/command/run_20260519_trc_round_train_one.sh
```

## 2026-05-20: Optional TRC Response NLL Auxiliary Loss

Purpose:

- Address the observed Code gap where hidden-residual steering raises BoN but not single-sample Code accuracy.
- Keep TRC default behavior unchanged while allowing an explicit teacher-forcing term on successful expert trajectories.

Changes:

- Updated `scripts/trc/train_trc_layer_gates.py`.
- Added CLI flags:
  - `--response-nll-weight`, default `0.0`;
  - `--task-response-nll-weight`, repeated task override such as `code=0.2`.
- When enabled, the trainer computes language-model NLL only on the selected response span tokens. Prompt tokens and non-selected response tokens are masked with `-100`.
- The NLL is logged as `response_nll_loss` at row, task, and epoch levels.
- Updated `skill/command/run_20260519_trc_round_train_one.sh` to record and pass:
  - `RESPONSE_NLL_WEIGHT`;
  - `TASK_RESPONSE_NLL_WEIGHT`.

Default impact:

- With `RESPONSE_NLL_WEIGHT=0.0`, no extra forward pass is made and the previous hidden-residual loss path is unchanged.
- Reward, calibration data, contrastive residual loss, task filters, trainable expert masking, and selection/bake behavior are unchanged.

Validation:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m py_compile scripts/trc/train_trc_layer_gates.py

bash -n skill/command/run_20260519_trc_round_train_one.sh
```

## 2026-05-20: Optional TRC Code-Only / Expert-Slice Gate Training

Purpose:

- Isolate whether Code hidden-residual steering can improve formal Code evaluation without Memory/Tool loss or floor gradients disturbing the gate update.
- Keep default TRC training unchanged; the new behavior only activates when explicit allowlists are passed.

Changes:

- Updated `scripts/trc/train_trc_layer_gates.py`.
- Added CLI flags:
  - `--train-tasks`, default empty: optional comma/space separated task allowlist for rows used in training.
  - `--trainable-experts`, default empty: optional comma/space separated expert allowlist whose gate gradients are kept.
- When `--trainable-experts code` is set, the trainer masks `raw_coefficients.grad` before clipping and `optimizer.step()`, so non-Code expert gate slices stay fixed.
- Updated `skill/command/run_20260519_trc_round_train_one.sh` to record and pass:
  - `TRAIN_TASKS`;
  - `TRAINABLE_EXPERTS`.

Default impact:

- Empty `TRAIN_TASKS` keeps all calibration rows.
- Empty `TRAINABLE_EXPERTS` keeps the old full gate gradient path.
- Reward, calibration rows, hidden-residual loss, contrastive loss, task-balanced loss, and selection/bake logic are unchanged unless these env vars are set.

Validation:

```bash
PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m py_compile scripts/trc/train_trc_layer_gates.py

bash -n skill/command/run_20260519_trc_round_train_one.sh
```

## 2026-05-21: Attention / MLP Diagnostic Probes

Purpose:

- Diagnose task-specific information flow and OP-VEC residual expression before changing the merge/training method.
- Separate three signals that should not be conflated:
  - attention matrix routing;
  - linear / MLP residual exposure;
  - signed first-order utility from teacher-forced gradients.

Changes:

- Added `scripts/attention_pauh/probe_attention_matrix_patterns.py`.
  - Runs base-model forward with `output_attentions=True`.
  - Reports prompt, prompt-tail, response-local, long-response, marker, sink, entropy, and head-specialization metrics.
- Added `scripts/attention_pauh/probe_linear_module_exposure_patterns.py`.
  - Hooks OP-VEC target linear modules from the mode manifest.
  - Computes prompt/response activation diagonals and scores expert deltas with `E||Delta W x||^2`.
  - Supports attention, MLP, or all-linear scopes and raw or delta-norm normalization.
- Added `scripts/attention_pauh/probe_signed_utility.py`.
  - Uses teacher-forced backward passes to estimate `-grad(loss) · Delta`.
  - Reports owner utility, protected-task harm, and induced residual conflict cosines.
- Added `linear_delta_probe()` to `scripts/attention_pauh/core.py`.
- Added unit coverage in `tests/test_attention_pauh.py`.

Default impact:

- No training, reward, rollout, gate update, or evaluation logic is changed.
- These scripts are standalone diagnostics and only write to explicit output directories.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  -m unittest discover -s tests -p 'test_attention_pauh.py' -v
```

## 2026-05-21: Signature-Preserving Residual Editing Gate Builder

Purpose:

- Build simple structure-aware OP-VEC gate checkpoints from attention / MLP diagnostics.
- Provide a clean baseline for testing whether preserving MLP residuals and only mildly shrinking risky attention families can maintain task abilities without reward sweeps.

Changes:

- Added `scripts/attention_pauh/build_signature_preserving_gates.py`.
  - Starts from `init=1` full expert prior.
  - Keeps MLP residuals at coefficient 1 by default.
  - Supports three explicit methods:
    - `exposure-shrink`: shrink only families with owner/protected exposure evidence.
    - `static-code-attn-shrink`: shrink Code attention q/k/v/o by fixed scales.
    - `mlp-preserve-attn-calm`: keep all MLP at 1 while calming all attention modules.
  - Writes `spre_gates.json`, `spre_config.json`, and `spre_summary.md` under the requested output directory.
- Extended `tests/test_attention_pauh.py` to cover SPRE helper behavior and coefficient summaries.
- Added structural validation report: `docs/report/20260521_spre_structure_validation.md`.

Default impact:

- No training, reward, rollout, update, evaluation, or existing gate behavior is changed.
- The builder only runs when explicitly invoked and writes to the provided output directory.

## 2026-05-21: Task-Signature Spans for Signed Utility Probe

Purpose:

- Make signed utility diagnostics inspect task-relevant behavior spans instead of only coarse prompt / response / all spans.
- Separate Tool tool-call behavior and Code final code behavior when estimating `-grad(loss) · delta`.

Changes:

- Updated `scripts/attention_pauh/probe_signed_utility.py`.
- Added `--span` choices:
  - `tool-call`: spans matching `<tool_call>...</tool_call>`.
  - `code-block`: markdown fenced code blocks.
  - `reasoning`: `<think>...</think>` when present, otherwise text before code/tool-call span.
  - `signature`: maps `tool -> tool-call`, `code -> code-block`, `memory -> response`.
- Response-specific spans use causal-LM alignment: selected response token `k` probes the output position that predicts it, i.e. `prompt_len + k - 1`.
- Added unit coverage for tool-call / code-block interval extraction and causal-shifted signature masks.

Default impact:

- Existing `response`, `prompt`, and `all` behavior is preserved.
- No training, reward, rollout, update, evaluation, or gate building logic is changed.

## 2026-05-21: Gate Structure Summary Utility

Purpose:

- Connect gate coefficients to downstream metrics by summarizing checkpoints at expert, layer, attention, and MLP granularity.
- Support both full OP-VEC gate files and TRC layer-band gate files in one diagnostic format.

Changes:

- Added `scripts/attention_pauh/summarize_gate_structure.py`.
  - Loads one or more gate JSON files.
  - Expands full keys like `model.layers.N.xxx.weight::expert`.
  - Expands TRC keys like `layerN.expert` through the mode manifest so layer-band gates can be summarized as attention / MLP groups.
  - Writes JSON and Markdown summaries when requested.
- Added unit coverage for layer-gate expansion and group summaries.
- Added structure-performance matrix report:
  - `docs/report/20260521_gate_structure_performance_matrix.md`.

Default impact:

- Pure read-only diagnostic utility.
- No training, reward, rollout, update, evaluation, or gate building behavior is changed.

## 2026-05-21: PAUH Shuffle Mechanism Check Materialization

Purpose:

- Add a deterministic PAUH layer-shuffle mechanism check for testing whether PAUH layer ordering matters beyond expert-average scale.

Artifacts:

- Materialized with existing `scripts/attention_pauh/materialize_pauh_variant.py`.
- Gate: `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_shuffle_20260521/pauh_gates.json`.
- Baked checkpoint: `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_shuffle_20260521/baked_policy`.
- Quick Tool/Memory eval: `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_alpha1_shuffle_20260521/quick_tool_memory`.
- Reports updated:
  - `docs/report/20260521_prompt_attention_utility_harm.md`.
  - `docs/report/20260521_gate_structure_performance_matrix.md`.

Default impact:

- No code path was changed for this artifact; it uses the existing PAUH variant script.

## 2026-05-21: Memory-Only Attention Calm SPRE Ablation

Purpose:

- Isolate whether Memory performance depends on its own attention residuals after MLP residuals are preserved.
- Separate Memory attention causality from the broader `mlp-preserve-attn-calm` variant, which also calmed Tool and Code attention.

Changes:

- Updated `scripts/attention_pauh/build_signature_preserving_gates.py`.
- Added SPRE method `memory-attn-calm`:
  - starts from `init=1`;
  - scales only Memory attention families `q/k/v/o` by `--attention-calm-scale`;
  - keeps Memory MLP at `1.0`;
  - keeps Tool and Code at `1.0`.
- Added unit coverage in `tests/test_attention_pauh.py`.

Artifacts:

- Gate: `/tmp/shared-storage/ExpertGym/spre/spre_memory_attn_calm_20260521/spre_gates.json`.
- Baked checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/spre_memory_attn_calm_20260521`.
- Quick Tool/Memory eval: `/tmp/shared-storage/ExpertGym/spre/eval/spre_memory_attn_calm_20260521/quick_tool_memory`.
- Gate structure summary:
  - `/tmp/shared-storage/ExpertGym/attention_matrix/gate_structure_representatives_20260521/spre_memory_attn_calm_gate_summary.json`.
  - `/tmp/shared-storage/ExpertGym/attention_matrix/gate_structure_representatives_20260521/spre_memory_attn_calm_gate_summary.md`.
- Reports updated:
  - `docs/report/20260521_spre_structure_validation.md`.
  - `docs/report/20260521_gate_structure_performance_matrix.md`.

Observed diagnostic result:

- BFCL quick mean stays close to SPRE-v2: `0.7788` vs `0.7825`.
- HotpotQA quick mean F1 drops: `0.7479` vs SPRE-v2 `0.7761`.
- This supports the structural claim that Memory relies on attention routing in addition to MLP residual magnitude.

Default impact:

- The new behavior is only used when explicitly passing `--method memory-attn-calm`.
- No training, reward, rollout, update, evaluation, or existing gate building default behavior is changed.

## 2026-05-21: Structured Capability Gates v1

Purpose:

- Turn the attention/MLP/signed-utility findings into a small first-principles gate family for second-stage capability search.
- Preserve known-useful Tool and Memory channels while testing whether Code can be expressed through middle-layer positive residual families.

Changes:

- Added `scripts/attention_pauh/build_structured_capability_gates.py`.
  - Generates a tiny mechanism-constrained candidate family.
  - Keeps `memory:* = 1.0` and `tool:* = 1.0`.
  - Splits Code into:
    - middle positive families: `mlp_gate`, `mlp_up`, `attn_o`, `attn_v` on layers `8-20`;
    - middle weak families: `mlp_down`, `attn_q`, `attn_k`;
    - conflict layers: default `24,27`;
    - background Code residual.
  - Writes one `gates.json` and `summary.md` per candidate plus a family `candidate_manifest.json`.
- Added `skill/command/run_20260521_structured_capability_gates_v1.sh`.
  - `PHASE=generate`: build gates.
  - `PHASE=bake`: bake all candidates.
  - `PHASE=quick_eval`: run Tool+Memory quick evaluation.
  - `PHASE=all`: run the full sequence.
  - `CANDIDATES=name1,name2`: restrict bake/eval to selected candidates. This is useful because BFCL harness setup mutates shared config files and should not be run concurrently across candidates.
- Added unit coverage in `tests/test_attention_pauh.py`.
- Added config/method document:
  - `docs/config/20260521_structured_capability_gates_v1.md`.

Artifacts:

- Candidate manifest: `/tmp/shared-storage/ExpertGym/structured_capability_gates/scg_v1_20260521/candidate_manifest.json`.
- Candidates:
  - `balanced`;
  - `code_mid_push`;
  - `code_safe`.
- Quick evaluation summary is recorded in `docs/config/20260521_structured_capability_gates_v1.md`.
  - Best current Tool/Memory tradeoff: `code_safe`.
  - Tool quick mean: `0.7813`.
  - Memory F1: `0.7714` on `eval_50`, `0.7427` on `eval_100`.

Default impact:

- Standalone generation/evaluation path only.
- No training, reward, rollout, update, existing SPRE, PAUH, TRC, or evaluation default behavior is changed.

## 2026-05-21: Response-Conditioned Residual Filtering v1

Purpose:

- Test a minimal mechanism hypothesis: the useful merge unit is module-level `DeltaW h` on task response spans, not an expert-level coefficient.
- Produce a training-free gate from signed utility, cross-task agreement, conflict, and low-energy/noise diagnostics.

Changes:

- Added `scripts/attention_pauh/build_response_conditioned_residual_filtering_gates.py`.
  - Builds `rcrf` and `energy_only` candidates from an existing signed-utility summary.
  - `rcrf` keeps stable owner residuals, mildly amplifies agreement, and suppresses conflict/noise.
  - `energy_only` is the single ablation that only uses response-conditioned expression energy.
- Added `skill/command/run_20260521_rcrf_v1.sh`.
  - `PHASE=generate`: build gates.
  - `PHASE=bake`: bake selected candidates.
  - `PHASE=quick_eval`: run Tool+Memory quick evaluation.
  - `CANDIDATES=name1,name2`: restrict bake/eval to selected candidates.
- Added unit coverage in `tests/test_attention_pauh.py`.
- Added config/result document:
  - `docs/config/20260521_rcrf_v1.md`.

Artifacts:

- Candidate manifest: `/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/candidate_manifest.json`.
- Baked checkpoints:
  - `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v1_20260521/rcrf`;
  - `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_v1_20260521/energy_only`.

Observed quick result:

- `rcrf`: BFCL quick mean `0.7956`, HotpotQA F1 `0.7708 / 0.7567`.
- `energy_only`: BFCL quick mean `0.7800`, HotpotQA F1 `0.7720 / 0.7296`.
- The Tool gap between `rcrf` and `energy_only` supports the role of signed utility / conflict beyond raw `DeltaW h` energy.

Default impact:

- Standalone generation/evaluation path only.
- No training, reward, rollout, update, existing SPRE, PAUH, TRC, or evaluation default behavior is changed.
