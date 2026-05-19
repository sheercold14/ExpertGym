# 2026-05-18 Baseline Reproduction Plan

## Scope

Goal: prepare and run Qwen2.5-7B three-expert baselines for ExpertGym paper
comparison. WUDI / ExpertMerging are indexed only in this round; other
reportable baselines are prioritized for same-protocol Eval6.

Base and experts:

- Base: `/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct`
- Tool: `/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold`
- Memory: `/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B`
- Code: `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B`
- OP-VEC manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`

## Priority

1. TA / Task Arithmetic: use existing TA sweep where available, and keep
   c=1/3 as the canonical no-calibration static baseline.
2. TIES: static OP-VEC implementation, keep top 20% by magnitude, disjoint mean.
3. DARE-TA: random drop 80%, rescale kept deltas, then task arithmetic.
4. DARE-TIES: DARE preprocessing plus TIES conflict filtering.
5. AdaMerging: Qwen adapter from local ExpertGym worktree; entropy minimization
   on calibration prompts.
6. Fisher: local Qwen implementation was not found in
   `/mnt/cache/wuruixiao/users/lsc/ExpertMerging` or the local AdaMerging
   worktree. Treat as a separate baseline gap: a faithful diagonal-Fisher merge
   needs per-expert backward/Fisher statistics on calibration data, not just a
   static delta operation.
   Official reference: Matena & Raffel, "Merging Models with Fisher-Weighted
   Averaging" (`https://arxiv.org/abs/2111.09832`) and
   `https://github.com/mmatena/model_merging`; the released script computes
   diagonal Fisher matrices first and then merges with those statistics.
7. WUDI / ExpertMerging: do not rerun for now; only record existing model dirs.
8. Mixture Training: separate full-parameter GRPO training baseline. A clean
   launcher is now available under
   `scripts/baselines/run_qwen_mixture_grpo_baseline.sh`; run it after current
   GPU-heavy experiments release enough cards.

Local WUDI / ExpertMerging path index:

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260518_local_wudi_expertmerging_model_dirs.md
```

Detailed gap audit for Fisher and Mixture/full-parameter GRPO:

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/harness/20260518_baseline_gap_fisher_mixture.md
```

## Commands

Static baselines:

```bash
DRY_RUN=1 bash scripts/baselines/run_qwen_static_baselines.sh
bash scripts/baselines/run_qwen_static_baselines.sh
```

AdaMerging:

```bash
DRY_RUN=1 METHOD=task_wise_adamerging GPU=0 bash scripts/baselines/run_qwen_adamerging_baseline.sh
METHOD=task_wise_adamerging GPU=0 bash scripts/baselines/run_qwen_adamerging_baseline.sh
```

Recommended AdaMerging first run:

```bash
METHOD=task_wise_adamerging \
RUN_NAME=qwen_task_wise_adamerging_20260518 \
GPU=<free_gpu> \
NUM_EPOCHS=1 \
MAX_BATCHES_PER_EPOCH=16 \
MAX_LENGTH=2048 \
bash scripts/baselines/run_qwen_adamerging_baseline.sh
```

Mixture/full-parameter GRPO baseline:

```bash
DRY_RUN=1 bash scripts/baselines/run_qwen_mixture_grpo_baseline.sh

GPU_LIST=0,1,2,3,4,5,6,7 \
LIMIT=24 \
SAMPLES_PER_PROMPT=2 \
TOTAL_TRAINING_STEPS=1 \
bash scripts/baselines/run_qwen_mixture_grpo_baseline.sh
```

This baseline starts from the static TA-1/3 checkpoint by default:

```text
/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/task_arithmetic_c0p3333333333333333_k0p2_d0p8_seed20260518
```

It trains full actor parameters with executable reward only; no OP-VEC gate,
no OPD, and no retention loss.

Active 96-prompt candidate launched on 2026-05-18:

```text
run_name: mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518
tmux: baseline_mixture_grpo_l96_gpu4_20260518
run_dir: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518
launch_log: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518.launch.log
```

Command:

```bash
GPU_LIST=4 \
LIMIT=96 \
SAMPLES_PER_PROMPT=1 \
TOTAL_TRAINING_STEPS=1 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=96 \
PPO_MINI_BATCH_SIZE=1 \
MAX_RESPONSE_LENGTH=512 \
MAX_MODEL_LEN=4608 \
PPO_MAX_TOKEN_LEN_PER_GPU=5120 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=2048 \
ROLLOUT_MAX_NUM_SEQS=8 \
ROLLOUT_GPU_MEM_UTIL=0.45 \
AGENT_LOOP_NUM_WORKERS=1 \
RUN_NAME=mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518 \
bash scripts/baselines/run_qwen_mixture_grpo_baseline.sh \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  trainer.balance_batch=False
```

2026-05-18 training result:

```text
status: training completed, HF checkpoint exported, and Eval6 completed
verl_actor_checkpoint: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518/checkpoints/global_step_1/actor
hf_checkpoint: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518/hf_merged_global_step_1
rollout_jsonl: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518/rollouts/1.jsonl
```

Training metrics from VeRL step 1:

```text
expanded_rows: 275
critic/score/mean: 0.0828831270
critic/score/min: -3.0
critic/score/max: 4.0
actor/grad_norm: 10.8611555663
actor/lr: 1e-6
actor/pg_loss: -0.08210045
actor/kl_loss: 0.04682548
kl_coef: 0.001
response_length/mean: 381.35
response_length/clip_ratio: 0.68
prompt_length/mean: 2709.96
prompt_length/clip_ratio: 0.5818
timing/gen: 124.45s
timing/old_log_prob: 125.95s
timing/ref: 183.04s
timing/update_actor: 580.04s
timing/save_checkpoint: 32.95s
timing/total_step: 1048.31s
```

HF export command:

```bash
PY=/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python
ACTOR=/tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518/checkpoints/global_step_1/actor
TARGET=/tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518/hf_merged_global_step_1
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/third_party/verl
PYTHONPATH=/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/third_party/verl:/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym:$PYTHONPATH \
  "$PY" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR" \
    --target_dir "$TARGET" \
    --trust-remote-code \
    --use_cpu_initialization
```

Eval6 result:

```text
Tool mean: 0.7823
Tool live mean: 0.6771
Memory mean EM: 0.5313
Memory mean F1: 0.6643
Code mean Acc: 0.3384
Code mean TP: 0.4716
Code mean BoN(4,4) Acc: 0.3782
Eval6 summary: /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260518_baselines_eval6.md
```

Dry-run sanity checked on 2026-05-18: script expands VeRL command and converts
the calibration manifest to parquet without starting GPU training. The dry-run
artifacts were removed after checking.

Actual AdaMerging note:

- `MAX_LENGTH=2048` on one 80G GPU hit CUDA OOM during the first backward
  pass. The process used one masked GPU as `cuda:0`; this is expected under
  `CUDA_VISIBLE_DEVICES`, not a GPU selection bug.
- The active reproducible run is therefore the same task-wise AdaMerging setup
  with `MAX_LENGTH=1024`, `NUM_EPOCHS=1`, `MAX_BATCHES_PER_EPOCH=16`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

```bash
METHOD=task_wise_adamerging \
RUN_NAME=qwen_task_wise_adamerging_len1024_20260518 \
GPU=6 \
NUM_EPOCHS=1 \
MAX_BATCHES_PER_EPOCH=16 \
MAX_LENGTH=1024 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/baselines/run_qwen_adamerging_baseline.sh
```

Additional AdaMerging variant launched after WUDI / ExpertMerging were moved to
path-index-only status:

```text
tmux: baseline_adamerging_layer_len1024_20260518
run_name: qwen_layer_wise_adamerging_len1024_20260518
output_root: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging
launch_log: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/qwen_layer_wise_adamerging_len1024_20260518.launch.log
status: completed
model: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/layer_wise_adamerging/qwen_layer_wise_adamerging_len1024_20260518/model
epoch0_loss: 1.037140
lambda_mean: 0.299807
eval6_tmux: eval_baseline_adamerging_lw_len1024_20260518
eval6_summary_dir: /tmp/shared-storage/ExpertGym/baselines/eval/adamerging_lw_len1024_20260518/20260518_baseline_adamerging_lw_len1024_eval6
eval6_status: completed
eval6_code_mean_acc: 0.3350
eval6_code_mean_tp: 0.4653
eval6_code_mean_bon44_acc: 0.3949
```

Command:

```bash
METHOD=layer_wise_adamerging \
RUN_NAME=qwen_layer_wise_adamerging_len1024_20260518 \
GPU=7 \
NUM_EPOCHS=1 \
MAX_BATCHES_PER_EPOCH=16 \
MAX_LENGTH=1024 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/baselines/run_qwen_adamerging_baseline.sh
```

Second additional variant:

```text
tmux: baseline_adamergingpp_task_len1024_20260518
run_name: qwen_task_wise_adamergingpp_len1024_20260518
output_root: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging
launch_log: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/qwen_task_wise_adamergingpp_len1024_20260518.launch.log
status: completed
model: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/task_wise_adamergingpp/qwen_task_wise_adamergingpp_len1024_20260518/model
epoch0_loss: 1.044321
lambda_mean: 0.300608
eval6_tmux: eval_baseline_adamergingpp_tw_len1024_20260518
eval6_summary_dir: /tmp/shared-storage/ExpertGym/baselines/eval/adamergingpp_tw_len1024_20260518/20260518_baseline_adamergingpp_tw_len1024_eval6
eval6_status: completed
eval6_code_mean_acc: 0.3309
eval6_code_mean_tp: 0.4647
eval6_code_mean_bon44_acc: 0.3773
```

Command:

```bash
METHOD=task_wise_adamergingpp \
RUN_NAME=qwen_task_wise_adamergingpp_len1024_20260518 \
GPU=0 \
NUM_EPOCHS=1 \
MAX_BATCHES_PER_EPOCH=16 \
MAX_LENGTH=1024 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/baselines/run_qwen_adamerging_baseline.sh
```

Third additional variant:

```text
tmux: baseline_adamergingpp_layer_len1024_20260518
run_name: qwen_layer_wise_adamergingpp_len1024_20260518
output_root: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging
launch_log: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/qwen_layer_wise_adamergingpp_len1024_20260518.launch.log
status: completed
model: /tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/layer_wise_adamergingpp/qwen_layer_wise_adamergingpp_len1024_20260518/model
epoch0_loss: 1.044022
lambda_mean: 0.299856
eval6_tmux: eval_baseline_adamergingpp_lw_len1024_20260518
eval6_summary_dir: /tmp/shared-storage/ExpertGym/baselines/eval/adamergingpp_lw_len1024_20260518/20260518_baseline_adamergingpp_lw_len1024_eval6
eval6_status: completed
eval6_code_mean_acc: 0.3287
eval6_code_mean_tp: 0.4593
eval6_code_mean_bon44_acc: 0.3841
```

Command:

```bash
METHOD=layer_wise_adamergingpp \
RUN_NAME=qwen_layer_wise_adamergingpp_len1024_20260518 \
GPU=1 \
NUM_EPOCHS=1 \
MAX_BATCHES_PER_EPOCH=16 \
MAX_LENGTH=1024 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/baselines/run_qwen_adamerging_baseline.sh
```

## Output Layout

Static baseline checkpoints:

```text
/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/
  task_arithmetic_c0p3333333333333333_k0p2_d0p8_seed20260518/
  ties_c0p3333333333333333_k0p2_d0p8_seed20260518/
  dare_ta_c0p3333333333333333_k0p2_d0p8_seed20260518/
  dare_ties_c0p3333333333333333_k0p2_d0p8_seed20260518/
```

AdaMerging checkpoints:

```text
/tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/<method>/<run_name>/model
```

## Existing Model Dirs

WUDI main result:

```text
/tmp/shared-storage/AgentMerging_plan/experiments/wudi_merging/wudi_qwen7b_3expert_diag/model
```

Other WUDI / TC-WUDI / VGEC variants are under:

```text
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/
```

Representative TC-WUDI model directories found locally:

```text
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-iter001/tc-diagwudi-traj-alpha050-codeprotect/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-iter002/tc-diagwudi-traj-alpha025-codeprotect/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-iter003/tc-diagwudi-traj-alpha050-codeprotect-code115/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-ae-stage1-20260506/tc-wudi-a-normmatch-traj/model
```

Qwen ExpertMerging repo-local logs:

- Current repo contains historical Qwen ExpertMerging scripts and lambda-stage
  logs under `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/Qwen/results/logs`.
- No current HF checkpoint was found at
  `/mnt/cache/wuruixiao/users/lsc/ExpertMerging/Qwen/results/logs/expert_merging/*/model`.

Moved / shared-storage ExpertMerging checkpoints found locally:

```text
/tmp/shared-storage/expert_merging/*/model
```

Representative directories:

```text
/tmp/shared-storage/expert_merging/0425-232232/model
/tmp/shared-storage/expert_merging/0428-151150/model
/tmp/shared-storage/expert_merging/0428-203833/model
/tmp/shared-storage/expert_merging/0503-152926/model
/tmp/shared-storage/expert_merging/0505-171400/model
/tmp/shared-storage/expert_merging/0506-185713/model
```

These WUDI / ExpertMerging paths are recorded for later comparison only and are
not launched in the current baseline batch.

RAIN / older ExpertMerging-style merged models:

```text
/tmp/shared-storage/RAIN/ExpertMerging/
```

AdaMerging smoke checkpoint already exists, but it is not a valid trained
baseline because `num_epochs=0`:

```text
/tmp/shared-storage/ExpertGym/adamerging/task_wise_adamerging/smoke_task_wise_init/model
```

## Eval Protocol

After each checkpoint exists, evaluate with:

```bash
RUN_TOOL=1 RUN_MEMORY=1 RUN_CODE=1 \
TOOL_GPU=<gpu> MEMORY_GPU_IDS=<gpu> CODE_GPU_GROUPS="[[<gpu0>,<gpu1>]]" \
RUN_ID=<run_id> SUMMARY_DIR=/tmp/shared-storage/ExpertGym/baselines/eval/<model_name>/<run_id> \
bash skill/command/run_full_eval_suite.sh <model_path> <model_name>
```

Final table should be recorded under `docs/evaluation/`.

## Completed Jobs

Completed on 2026-05-18:

- Static build completed for TA, TIES, DARE-TA, DARE-TIES under
  `/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/`.
- TIES, DARE-TA, DARE-TIES, and AdaMerging Eval6 completed and are recorded in
  `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260518_baselines_eval6.md`.
- AdaMerging task-wise length-1024 checkpoint completed at
  `/tmp/shared-storage/ExpertGym/baselines/qwen7b/adamerging/task_wise_adamerging/qwen_task_wise_adamerging_len1024_20260518/model`.
- WUDI / ExpertMerging are not relaunched in this batch; local model paths are
  indexed in
  `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260518_local_wudi_expertmerging_model_dirs.md`.
- Mixture/full-parameter GRPO smoke is now executable. Multiple early attempts
  on 2026-05-18 exposed VeRL mixed-task agent-loop compatibility issues; the
  successful smoke is recorded in
  `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260518_mixture_grpo_smoke.md`.
  The successful recipe is one-card DP with actor parameter offload,
  `ROLLOUT_GPU_MEM_UTIL=0.45`, `PPO_MAX_TOKEN_LEN_PER_GPU=5120`, and
  `PPO_MINI_BATCH_SIZE=1`. Mixed MemAgent trajectories expanded 8 prompts into
  25 training rows, so multi-card DP or larger mini-batches need audited
  padding/chunking before they are used for formal numbers.

BFCL caveat: concurrent first-time harness startup can race on BFCL's shared
`model_config.py`. For reruns, model names were pre-registered sequentially
before launching concurrent evaluation jobs.

Stronger BFCL caveat: Tool/BFCL evaluation also shares BFCL's `.env`
(`REMOTE_OPENAI_BASE_URL`), so different Tool models cannot be evaluated
concurrently even after model registration. The valid scheduling rule is:

1. Run at most one `RUN_TOOL=1` job at a time.
2. Run Memory/Code-only jobs concurrently with `RUN_TOOL=0`.
3. Merge final reported metrics from the Tool-only/full Tool job plus the
   Memory/Code job when a model was split for scheduling.

The failed concurrent Tool attempts for DARE-TA and AdaMerging are kept only as
failure logs and are not valid metrics. Replacement Memory+Code jobs completed:

- DARE-TA Memory+Code: `baseline_eval_dare_ta_mc_20260518`, GPU group `2/3`.
- AdaMerging Memory+Code: `baseline_eval_adamerging_tw_len1024_mc_20260518`,
  GPU group `6/7`.
- DARE-TA Tool-only: `baseline_eval_dare_ta_tool_20260518`, GPU `1`, port
  `8142`.
- AdaMerging Tool-only: `baseline_eval_adamerging_tw_len1024_tool_20260518`,
  GPU `1`, port `8146`.

TA c=1/3 is already covered by the existing P0 TA-1/3 evaluation chain, so the
new static eval batch does not duplicate it unless a consistency recheck is
needed.

## Reproducibility Notes

- Static builder uses deterministic per-parameter DARE masks with
  `SEED=20260518`.
- TIES/DARE default hyperparameters match the local implementations:
  keep top 20% / drop 80%.
- Static methods use OP-VEC deltas and the same mergeable parameter set as
  current ExpertGym experiments: 196 modules, 588 expert delta entries.
- Baseline scripts do not touch gated-GRPO training code.
