# 2026-05-17 A1：B Setting + CURE-Aligned Calibration

## 目标

在 `20260515_opd_continue_abcd.md` 的 B 设置基础上，测试新构造的 `eval_targeted96_cure_aligned_20260517` calibration 是否能给 Code/CURE 和 BFCL-style Tool 提供更有效的 on-policy 信号。

核心假设：

- 旧 B 的 `global-parameter + OPD + retention` 能稳定推动 memory，但 Code formal eval 不涨。
- 新 calibration 保留 `paper96` anchor，同时补 CURE/BFCL case-study 暴露的短板。
- 新增 targeted rows 没有完整离线 expert rollout，因此 A1 在 B 设置上额外打开 GRPO，使这些 rows 至少通过 on-policy reward variance 进入梯度。

## Run

| 项 | 值 |
|---|---|
| run name | `a1_gp_curealigned_b_grpo_opd_ret_shorttok_20260517` |
| run dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/a1_gp_curealigned_b_grpo_opd_ret_shorttok_20260517` |
| tmux | `train_a1_shorttok_20260517` |
| GPU | `0,5` |
| gate | `global-parameter` |
| init | `tool=memory=code=1/3` |
| prompts | `96 total = tool 32 + memory 32 + code 32` |
| samples per prompt | `4` |
| iterations | `20` |

## 试跑记录

第一版 `a1_gp_curealigned_b_grpo_opd_ret_20260517` 使用 `CODE_MAX_NEW_TOKENS=10000`、`MAX_MODEL_LEN=24576`、`MAX_LOGPROB_TOKENS=24576`。第 1 轮 rollout 正常完成，但 update 在 `max_logprob_tokens=24576` 下耗时过长，停止该试跑，仅保留 `iter_001/rollouts*.jsonl` 与 `opd_distill_from_allfail.summary.json` 用于诊断。

当前正式 A1-short 使用与 B/expH 一致的训练长度档位：

| 项 | 值 |
|---|---|
| code max new tokens | `4096` |
| max model len | `12288` |
| max logprob tokens | `12288` |

这版仍保留 CURE-aligned calibration 和 `OPVEC_CODE_REWARD_MAX_TESTS=8`，目标是先验证新 calibration 的训练信号，不把训练时间浪费在过长 logprob 回放上。

## 数据

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl
```

组成：

| task | rows | 组成 |
|---|---:|---|
| tool | 32 | 16 paper96 Tool anchor + 16 BFCL-style synthetic |
| memory | 32 | 32 paper96 HotpotQA-train anchor |
| code | 32 | 16 paper96 Code anchor + 16 CURE-style CodeContests targeted |

## Loss 与优化

| 项 | 值 |
|---|---|
| GRPO / PPO loss | `1.0` |
| OPD loss | `1.0` |
| retention | enabled, `nll` |
| retention task-balanced | enabled |
| OPD task-balanced | enabled |
| OPD positive threshold | `1.0` |
| dynamic OPD per task | `32` |
| optimizer | `sgd` |
| momentum | `0.2` |
| lr | `0.1876` |
| step scope | `epoch` |
| update batch size | `4` |
| loss granularity | `sequence` |
| prior loss | `0.0` |
| max coeff delta from init | `1.0` |

与 B 的主要差异：

| 项 | B | A1 |
|---|---|---|
| calibration | paper96 / calib100 family | `eval_targeted96_cure_aligned_20260517` |
| GRPO | `0.0` in B OPD-only | `1.0` |
| advantage normalization | `centered` | `zscore` |
| code max new tokens | `4096` | `4096` |
| max model/logprob tokens | `12288` | `12288` |
| Code reward tests | default env | `OPVEC_CODE_REWARD_MAX_TESTS=8` |

## Expert Rollout

沿用旧 paper96 与 20260516 code augmentation expert pool，只读不改：

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl
```

注意：这些 expert rollout 只覆盖 paper96 overlap prompts；新 synthetic/targeted rows 主要靠 GRPO 信号。

## 启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

tmux new -d -s train_a1_curealigned_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && rm -rf /tmp/shared-storage/OnPolicy/runs/gated_grpo/a1_gp_curealigned_b_grpo_opd_ret_20260517 && mkdir -p /tmp/shared-storage/OnPolicy/runs/gated_grpo/a1_gp_curealigned_b_grpo_opd_ret_20260517 && DRY_RUN=0 GPU_LIST=0,5 ROLLOUT_GPUS=0,5 RUN_NAME=a1_gp_curealigned_b_grpo_opd_ret_20260517 RUN_DIR=/tmp/shared-storage/OnPolicy/runs/gated_grpo/a1_gp_curealigned_b_grpo_opd_ret_20260517 STRATEGY=global-parameter INIT_VALUE=0.3333333333333333 CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl NUM_ITERS=20 NUM_PROMPTS=96 SAMPLES_PER_PROMPT=4 STORE_TOKEN_LOGPROBS=0 OPTIMIZER=sgd SGD_MOMENTUM=0.2 PERSIST_OPTIMIZER_STATE=1 LR=0.1876 PRIOR_LOSS_WEIGHT=0.0 MAX_COEFF_DELTA=1.0 UPDATE_EPOCHS=1 UPDATE_BATCH_SIZE=4 BATCH_LOSS_REDUCTION=mean OPTIMIZER_STEP_SCOPE=epoch LOSS_GRANULARITY=sequence FRONTIER_ORDER=task-interleaved FRONTIER_TOOL_QUOTA=32 FRONTIER_MEMORY_QUOTA=32 FRONTIER_CODE_QUOTA=32 USE_RETENTION=1 RETENTION_OBJECTIVE=nll RETENTION_POSITIVE_REWARD_THRESHOLD=1.0 RETENTION_TASK_BALANCED_LOSS_SCALE=1 RETENTION_SCALE_TARGET=0.5 OPD_LOSS_WEIGHT=1.0 OPD_POSITIVE_REWARD_THRESHOLD=1.0 OPD_LENGTH_NORMALIZE_LOGPROB=1 RETENTION_LENGTH_NORMALIZE_LOGPROB=1 OPD_TASK_BALANCED_LOSS_SCALE=1 LENGTH_NORMALIZE_POLICY_LOGPROB=1 LENGTH_NORMALIZE_LOGPROB=0 TASK_NORMALIZE_ADVANTAGES=0 ADVANTAGE_NORMALIZATION=zscore USE_FRONTIER_WEIGHT=0 PPO_LOSS_WEIGHT=1.0 DYNAMIC_OPD_EXPERT_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl DYNAMIC_OPD_TASKS=tool,memory,code DYNAMIC_OPD_KEY=prompt_id DYNAMIC_OPD_CURRENT_MAX_SUCCESS=0 DYNAMIC_OPD_POSITIVE_THRESHOLD=1.0 DYNAMIC_OPD_MAX_POSITIVES_PER_ROW=1 DYNAMIC_OPD_MAX_NEGATIVES_PER_ROW=2 DYNAMIC_OPD_PER_TASK=32 MAX_NEW_TOKENS=1024 TOOL_MAX_NEW_TOKENS=512 CODE_MAX_NEW_TOKENS=10000 MEMORY_UPDATE_MAX_NEW_TOKENS=2048 MEMORY_FINAL_MAX_NEW_TOKENS=2048 MAX_PROMPT_TOKENS=8192 MAX_MODEL_LEN=24576 MAX_LOGPROB_TOKENS=24576 ROLLOUT_BATCH_SIZE=32 ROLLOUT_SHARDS=auto TENSOR_PARALLEL_SIZE=1 GPU_MEMORY_UTILIZATION=0.82 TEMPERATURE=0.7 TOP_P=0.95 SEED_VALUE=20260517 PROGRESS_EVERY=10 OPVEC_CODE_REWARD_MAX_TESTS=8 bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/a1_gp_curealigned_b_grpo_opd_ret_20260517/train.log'
```

## 每轮判断规则

每个 iteration 完成后检查：

```text
iter_*/rollouts.summary.json
iter_*/opd_distill_from_allfail.summary.json
iter_*/gate_updates.summary.json
iter_*/gate_updates.gates.json
```

判定重点：

| 指标 | 合理区间 / 解释 |
|---|---|
| task reward | code/tool/memory 至少不能单边持续崩；overall 上升优先 |
| frontier_task_counts | targeted rows 是否通过 GRPO 产生方差 |
| OPD selected_task_counts | paper96 overlap 的 all-fail 是否仍能拿到 expert positive |
| retention rows | all-success 是否被 NLL preservation 约束 |
| gate means | code 不应长期卡在 0.33；memory 上涨不能以 tool/code 崩溃为代价 |
| grad_norm / gate delta | 若长期过小，说明新数据没有有效推动；若异常过大且 reward 崩，需停机调 lr/token length |
