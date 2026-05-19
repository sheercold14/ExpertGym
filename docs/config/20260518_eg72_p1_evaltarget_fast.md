# 2026-05-18 EG72 P1 Eval-Targeted Fast Runs

## Purpose

This batch is the first post-baseline P1 run for the 72h ExpertGym paper loop.
It tests the paper claim:

```text
task vectors provide structured priors; executable feedback learns their composition
```

The run uses `eval_targeted96_cure_aligned_20260517` as the training probe bank,
because P0 state distribution showed that raw `paper96` is a useful anchor but
not a sufficient main calibration bank.

## Data

Training prompts:

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl
```

Composition:

| task | rows | source |
|---|---:|---|
| tool | 32 | 16 paper96 Tool anchors + 16 BFCL-style synthetic probes |
| memory | 32 | HotpotQA train anchors |
| code | 32 | 16 paper96 Code anchors + 16 CURE-style CodeContests probes |

Dynamic OPD expert rollouts:

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/tool_expert_toolrl_qwen25_7b_evaltarget_s4_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/memory_expert_rl_memoryagent7b_evaltarget_s4_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260518.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260519.jsonl
```

Expert positive coverage:

| task/source | positive prompts |
|---|---:|
| tool ToolRL seed20260517 | 24 / 32 |
| memory RL-MemoryAgent seed20260517 | 29 / 32 |
| code DeepSeek seed20260517 | 17 / 32 |
| code DeepSeek seed20260518 | 16 / 32 |
| code ReasonFlux seed20260517 | 15 / 32 |
| code ReasonFlux seed20260519 | 16 / 32 |
| code union | 20 / 32 |

## Shared Fast Settings

These settings follow `docs/harness/runbook_5h_fast_iteration.md` and are
intended to make each run decision-relevant within 5 hours.

| item | value |
|---|---|
| iterations | 12 |
| prompts | 96 |
| samples per prompt | 4 |
| update step scope | epoch |
| update batch size | 8 |
| loss granularity | sequence |
| frontier cap | 4 rows per task per iteration |
| retention cap | 8 rows per task, 24 total |
| retention objective | NLL |
| retention weight | 0.5 |
| OPD weight | 1.0 for full/OPD-only |
| optimizer | SGD |
| momentum | 0.2 |
| lr | 0.1876 |
| prior loss | 0.0 |
| max coefficient delta from init | 1.0 |
| task advantage normalization | off |
| per-prompt advantage normalization | centered |
| OPD / retention length normalization | on |
| OPD / retention task-balanced loss scale | on |
| code reward max tests | 8 |

## Runs

| run | GPU | strategy | init | PPO/GRPO | OPD | retention | purpose |
|---|---|---|---:|---:|---:|---:|---|
| `eg72_main_gc_c033_evaltarget_fast_20260518` | 0,1 | global-coefficient | 1/3 | 1.0 | 1.0 | 0.5 | clean 3-coefficient main method |
| `eg72_main_global_c033_evaltarget_fast_20260518` | 2,3 | global common+residual | 1/3 | 1.0 | 1.0 | 0.5 | 4-parameter common/residual comparison |
| `eg72_main_gc_init1_evaltarget_fast_20260518` | 4,5 | global-coefficient | 1.0 | 1.0 | 1.0 | 0.5 | strong-prior executable refinement |
| `eg72_opd_gc_c033_evaltarget_fast_20260518` | 6,7 | global-coefficient | 1/3 | 0.0 | 1.0 | 0.5 | recovery-only ablation |

## Runtime Status

Started at `2026-05-18 02:33 CST`.

| run | tmux | run dir |
|---|---|---|
| `eg72_main_gc_c033_evaltarget_fast_20260518` | `train_eg72_main_gc_c033_fast_20260518` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_c033_evaltarget_fast_20260518` |
| `eg72_main_global_c033_evaltarget_fast_20260518` | `train_eg72_main_global_c033_fast_20260518` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_global_c033_evaltarget_fast_20260518` |
| `eg72_main_gc_init1_evaltarget_fast_20260518` | `train_eg72_main_gc_init1_fast_20260518` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518` |
| `eg72_opd_gc_c033_evaltarget_fast_20260518` | `train_eg72_opd_gc_c033_fast_20260518` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_opd_gc_c033_evaltarget_fast_20260518` |

Monitor:

```text
http://127.0.0.1:8796
tmux: opvec_monitor_eg72_p1_evaltarget_20260518
log: /tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_p1_evaltarget_monitor_8796.log
```

Init-gate note:

- Initial concurrent launch of the three `global-coefficient` runs wrote the
  shared default file
  `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/init_gates/init_global_coefficient_c033333.json`
  concurrently and produced an invalid concatenated JSON.
- The file was rebuilt immediately.
- Active `global-coefficient` runs use explicit, stable init files:

```text
/tmp/shared-storage/OnPolicy/data/init_gates/eg72_p1_evaltarget_20260518/init_gc_c033333.json
/tmp/shared-storage/OnPolicy/data/init_gates/eg72_p1_evaltarget_20260518/init_gc_init1.json
```

## Launch Template

For each run, override only `RUN_NAME`, `RUN_DIR`, `GPU_LIST`, `ROLLOUT_GPUS`,
`STRATEGY`, `INIT_VALUE`, and `PPO_LOSS_WEIGHT` as shown above.

```bash
CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl
DYNAMIC_OPD_EXPERT_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/tool_expert_toolrl_qwen25_7b_evaltarget_s4_seed20260517.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/memory_expert_rl_memoryagent7b_evaltarget_s4_seed20260517.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260517.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260518.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260517.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260519.jsonl

NUM_ITERS=12 NUM_PROMPTS=96 SAMPLES_PER_PROMPT=4 \
UPDATE_BATCH_SIZE=8 OPTIMIZER_STEP_SCOPE=epoch LOSS_GRANULARITY=sequence \
FRONTIER_ROWS_PER_TASK=4 FRONTIER_SAMPLE_BEFORE_LIMIT=1 FRONTIER_ORDER=task-interleaved \
USE_RETENTION=1 RETENTION_OBJECTIVE=nll RETENTION_LOSS_WEIGHT=0.5 \
MAX_RETENTION_ROWS_PER_TASK=8 MAX_RETENTION_ROWS=24 RETENTION_SAMPLE_BEFORE_LIMIT=1 \
OPD_LOSS_WEIGHT=1.0 OPD_POSITIVE_REWARD_THRESHOLD=1.0 \
OPD_TASK_BALANCED_LOSS_SCALE=1 RETENTION_TASK_BALANCED_LOSS_SCALE=1 \
OPD_LENGTH_NORMALIZE_LOGPROB=1 RETENTION_LENGTH_NORMALIZE_LOGPROB=1 \
OPTIMIZER=sgd SGD_MOMENTUM=0.2 PERSIST_OPTIMIZER_STATE=1 \
LR=0.1876 PRIOR_LOSS_WEIGHT=0.0 MAX_COEFF_DELTA=1.0 \
STORE_TOKEN_LOGPROBS=0 TASK_NORMALIZE_ADVANTAGES=0 ADVANTAGE_NORMALIZATION=centered \
USE_FRONTIER_WEIGHT=0 LENGTH_NORMALIZE_POLICY_LOGPROB=1 LENGTH_NORMALIZE_LOGPROB=0 \
DYNAMIC_OPD_TASKS=tool,memory,code DYNAMIC_OPD_KEY=prompt_id \
DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0 DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0 \
DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1 DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2 DYNAMIC_OPD_PER_TASK=32 \
MAX_NEW_TOKENS=1024 TOOL_MAX_NEW_TOKENS=512 CODE_MAX_NEW_TOKENS=4096 \
MEMORY_UPDATE_MAX_NEW_TOKENS=2048 MEMORY_FINAL_MAX_NEW_TOKENS=2048 \
MAX_PROMPT_TOKENS=8192 MAX_MODEL_LEN=12288 MAX_LOGPROB_TOKENS=12288 \
ROLLOUT_BATCH_SIZE=32 ROLLOUT_SHARDS=auto TENSOR_PARALLEL_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.82 TEMPERATURE=0.7 TOP_P=0.95 SEED_VALUE=20260518 \
OPVEC_CODE_REWARD_MAX_TESTS=8 \
bash skill/command/run_qbank_c033333_gate_strategy.sh
```

## Stop / Promote Rule

Check after iterations 3, 6, 9, and 12.

- Stop if all task rewards are flat and `selected_task_counts` for dynamic OPD
  is near zero.
- Stop if one task collapses for two consecutive iterations while gate movement
  keeps increasing.
- Promote best checkpoint to eval6 only if proxy improves without worsening
  both Tool live and Code proxy simultaneously.
