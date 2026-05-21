# R11B Clean Repeat Evaluation - 2026-05-20

## Decision Rule

本轮 R11B 复现不使用 gate 系数作为晋级依据。流程固定为：

1. 先测 Tool 与 Memory。
2. Tool mean acc >= 0.79 且 Memory mean F1 >= 0.76 时，送入 Code/CURE。
3. gate 系数仅用于事后解释，不参与候选筛选。

## Model

| Field | Value |
|---|---|
| Alias | R11B |
| Checkpoint | `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11b_r8d_codeblock_e08_20260520-selected` |
| Source | R8D CodeP0 RF-only code-block span, epoch 8 early stop |
| Code repeat session | `eval_r11b_code_repeat_clean_20260520` |

## Tool / Memory Gate

| Task | Run ID | Main Metric | Details | Status |
|---|---:|---:|---|---|
| Tool BFCL | `r11b_tool_repeat_clean_20260520` | 0.7944 | live_parallel 0.8125, live_parallel_multiple 0.6250, parallel 0.8850, parallel_multiple 0.8550 | pass |
| Memory HotpotQA | `r11b_memory_repeat_clean_20260520` | 0.7668 | eval_50 0.7817, eval_100 0.7704, qa_32768 0.7825, qa_65536 0.7325 | pass |

## Code / CURE

| Run ID | LiveBench Acc | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench BoN | Mean Acc | Mean BoN | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `r11b_code_repeat_clean_20260520` | 0.3691 | 0.4609 | 0.2715 | 0.3444 | 0.3203 | 0.4027 | done |

## Notes

- Earlier `R11B repeat` under `code_repeat_20260520_1350` is invalid because vLLM hit GPU0 OOM during startup. It should not be used as a model score.
- This clean repeat explicitly cleared overlapping R19/R20/R21 eval/training sessions before running Tool/Memory/Code.
- Clean Code repeat confirms R11B is stable enough to pass Tool/Memory, but does not reproduce the previous high Code BoN (`0.4310`). The stronger historical Code number should be treated as stochastic or resource-condition sensitive until another clean repeat confirms it.
