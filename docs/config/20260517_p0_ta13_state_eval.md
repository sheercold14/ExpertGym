# 2026-05-17 P0: TA-1/3 Eval6 and State Distribution

## 目的

补齐 ExpertGym 论文 P0 地基：

1. `TA-1/3 formal eval6`：严格 equal-prior task arithmetic checkpoint，不能用 TA-0.75 代替。
2. `state distribution`：在 `1/3` 与 `init1` 两个 prior 上统计 `frontier / recoverable / stable / unsolved`，解释为什么 GRPO 只对 frontier 有效、Recovery-OPD 为什么必要、retention 为什么不能省。

## 固定输入

| 项 | 值 |
|---|---|
| config | `configs/gated_grpo.yaml` |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` |
| calibration | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` |
| expert rollouts | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/*.jsonl` |
| TA-1/3 gate | `configs/init_gates/ta_c033333_global.json` |
| init1 gate | `configs/init_gates/ta_init1_global.json` |

## 产物路径

```text
/tmp/shared-storage/OnPolicy/checkpoints/ta_c033333_global_20260517
/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517
/tmp/shared-storage/OnPolicy/runs/state_distribution/p0_ta13_k8_20260517
/tmp/shared-storage/OnPolicy/runs/state_distribution/p0_init1_k8_20260517
/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c033333-global-20260517/
```

## Rollout 设置

```bash
NUM_PROMPTS=96
SAMPLES_PER_PROMPT=8
TEMPERATURE=0.7
TOP_P=0.95
TOOL_MAX_NEW_TOKENS=512
MAX_NEW_TOKENS=1024
MEMORY_UPDATE_MAX_NEW_TOKENS=2048
MEMORY_FINAL_MAX_NEW_TOKENS=2048
CODE_MAX_NEW_TOKENS=4096
MAX_PROMPT_TOKENS=8192
MAX_MODEL_LEN=12288
```

## State 判定

| state | 判定 |
|---|---|
| stable | current rollout 全部 success |
| frontier | current rollout 有成功也有失败，或 reward 有方差 |
| recoverable | current rollout 全失败，且 same-prompt expert rollout 有 verified positive |
| unsolved | current rollout 全失败，且 expert rollout 也没有 verified positive |

## 运行记录

已启动：

| 项 | 值 |
|---|---|
| TA-1/3 bake | completed, `num_delta_entries=588` |
| init1 bake | completed, `num_delta_entries=588` |
| eval6 tmux | `eval_p0_ta13_eval6_20260517` |
| eval6 summary dir | `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c033333-global-20260517/20260517_p0_ta13_eval6` |
| TA-1/3 state tmux | `p0_state_ta13_k8_20260517` |
| init1 state tmux | `p0_state_init1_k8_20260517` |
| TA-1/3 state dir | `/tmp/shared-storage/OnPolicy/runs/state_distribution/p0_ta13_k8_20260517` |
| init1 state dir | `/tmp/shared-storage/OnPolicy/runs/state_distribution/p0_init1_k8_20260517` |

## 启动命令摘要

TA-1/3 formal eval6：

```bash
RUN_ID=20260517_p0_ta13_eval6 \
EXPERIMENT_NAME=expertgym-p0-ta13 \
RUN_TOOL=1 RUN_MEMORY=1 RUN_CODE=1 \
TOOL_GPU=0 TOOL_PORT=8031 \
MEMORY_GPU_IDS=1 MEMORY_TP=1 \
CODE_GPU_GROUPS="[[2,3]]" \
SUMMARY_DIR=/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c033333-global-20260517/20260517_p0_ta13_eval6 \
bash skill/command/run_full_eval_suite.sh \
  /tmp/shared-storage/OnPolicy/checkpoints/ta_c033333_global_20260517 \
  ta-c033333-global-20260517
```

State rollout 使用 `scripts/train/opvec_collect_vllm_rollouts.py`，`K=8`，`paper96` manifest 顺序，分别在 GPU 4/5 跑 `TA-1/3` 和 `init1`。
