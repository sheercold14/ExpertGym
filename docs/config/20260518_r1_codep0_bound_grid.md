# 2026-05-18 R1 Code P0 Bound Grid

## 目的

在 Code P0 v3 上验证 DeepSeek-R1-Distill-Qwen-7B 小系数 task vector 是否能带来稳定 Code reward 增益，并区分两个因素：

```text
R1 幅值上限是否过紧；
tool/memory/code common 同动是否干扰 Code/R1 学习。
```

## 共同配置

```text
base script: skill/command/run_20260518_r1_codep0_bounded_sanity.sh
mode: /tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json
calibration: /tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/train_code64.prompts.jsonl
strategy: layer-band
task filter: code
init effective: tool=1.0, memory=1.0, code=1.0, reasoning=0.001
num_iters: 4
num_prompts: 64
samples_per_prompt: 4
loss: GRPO 1.0 + dynamic OPD 1.0 + retention NLL 0.05
optimizer: SGD lr=0.05 momentum=0.2
```

Expert positives:

```text
/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_code_p0_v3_train64_s8_seed20260518.merged.jsonl
```

## 四个短程诊断

| id | run_name | GPUs | R1 bound | train coefficients | 目的 |
|---|---|---|---|---|---|
| safe-all | `r1_codep0_safe_all_bound003_20260518` | 0,1 | `reasoning=0.0:0.003` | all | 安全 R1 注入，观察 code reward 是否能涨 |
| stress-all | `r1_codep0_stress_all_bound01_20260518` | 2,3 | `reasoning=0.0:0.01` | all | 检查 safe 不涨是否因为 R1 上限太紧 |
| safe-cr | `r1_codep0_safe_cr_bound003_20260518` | 4,5 | `reasoning=0.0:0.003` | `*.code,*.reasoning` | 冻结 tool/memory，隔离 Code/R1 学习 |
| stress-cr | `r1_codep0_stress_cr_bound01_20260518` | 6,7 | `reasoning=0.0:0.01` | `*.code,*.reasoning` | 检查 Code/R1 专属空间能否更快提升 |

## 判据

优先看：

1. Code train mean reward 是否 first-to-last 上升，且不是单轮噪声。
2. all-fail prompt 数是否下降。
3. R1 gate 是否在边界内形成有结构的 per-layer 分布。
4. `safe-cr` / `stress-cr` 若明显优于 all，说明 common 同动干扰 Code 专家信号。
5. 若 `stress` 优于 `safe`，R1 上限应放宽到 `0.01` 并依靠 monitor/guard 控制风险。
6. 若四个都不涨，下一步应回到 Code reward/OPD 数据设计，而不是继续调系数。
