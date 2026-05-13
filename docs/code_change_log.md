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
