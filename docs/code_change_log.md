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
