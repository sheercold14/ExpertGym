# 2026-05-18 R1 Code P0 Bound Grid

## 实验目的

本实验验证 DeepSeek-R1-Distill-Qwen-7B 的小幅 task vector 是否能作为 Code 专家补偿信号，并排查两个关键问题：

1. R1 系数绝对上界过紧，导致 reasoning 能力注入不足；
2. `tool/memory/code` common 同动会干扰 Code/R1 的局部学习。

配置见：

```text
docs/config/20260518_r1_codep0_bound_grid.md
```

## 共同设置

```text
mode: /tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json
calibration: /tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/train_code64.prompts.jsonl
task: code only
strategy: layer-band
init effective: tool=1.0, memory=1.0, code=1.0, reasoning=0.001
loss: GRPO 1.0 + dynamic OPD 1.0 + retention NLL 0.05
optimizer: SGD, lr=0.05, momentum=0.2
num_iters: 4
num_prompts: 64
samples_per_prompt: 4
```

新增安全机制：

```text
--coefficient-bound-by-expert reasoning=0.0:0.003 或 reasoning=0.0:0.01
```

该约束是 effective coefficient 的绝对边界，不是“相对初始点每步 delta”。这是本轮之前 sanity 中 R1 系数漂移为负的主要修正。

## 运行矩阵

| id | run | R1 bound | train coefficients | 完成情况 |
|---|---|---:|---|---|
| safe-all | `r1_codep0_safe_all_bound003_20260518` | [0, 0.003] | all | iter4 done |
| stress-all | `r1_codep0_stress_all_bound01_20260518` | [0, 0.01] | all | iter3 done, iter4 rollout 后按资源调度停止 |
| safe-cr | `r1_codep0_safe_cr_bound003_20260518` | [0, 0.003] | `*.code,*.reasoning` | iter3 done, 按资源调度停止 |
| stress-cr | `r1_codep0_stress_cr_bound01_20260518` | [0, 0.01] | `*.code,*.reasoning` | iter4 done |

## 训练曲线

| run | iter | reward | any success | all fail | all success | frontier | code mean | reasoning mean/min/max |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| safe-all | 1 | 0.4160 | 25 | 39 | 15 | 21 | 1.0038 | 0.0024 / 0.0000 / 0.0030 |
| safe-all | 2 | 0.4049 | 27 | 37 | 14 | 23 | 1.0076 | 0.0024 / 0.0000 / 0.0030 |
| safe-all | 3 | 0.4095 | 27 | 37 | 15 | 23 | 1.0113 | 0.0024 / 0.0000 / 0.0030 |
| safe-all | 4 | 0.4160 | 26 | 38 | 16 | 20 | 1.0151 | 0.0024 / 0.0000 / 0.0030 |
| stress-all | 1 | 0.4175 | 24 | 40 | 19 | 17 | 1.0038 | 0.0023 / 0.0000 / 0.0030 |
| stress-all | 2 | 0.4036 | 27 | 37 | 14 | 21 | 1.0075 | 0.0038 / 0.0000 / 0.0050 |
| stress-all | 3 | 0.3999 | 27 | 37 | 15 | 21 | 1.0113 | 0.0053 / 0.0000 / 0.0070 |
| safe-cr | 1 | 0.3952 | 24 | 40 | 14 | 23 | 1.0038 | 0.0024 / 0.0000 / 0.0030 |
| safe-cr | 2 | 0.3997 | 25 | 39 | 16 | 18 | 1.0076 | 0.0024 / 0.0000 / 0.0030 |
| safe-cr | 3 | 0.4023 | 25 | 39 | 16 | 20 | 1.0114 | 0.0024 / 0.0000 / 0.0030 |
| stress-cr | 1 | 0.4086 | 23 | 41 | 19 | 17 | 1.0038 | 0.0023 / 0.0000 / 0.0030 |
| stress-cr | 2 | 0.4017 | 27 | 37 | 11 | 24 | 1.0076 | 0.0038 / 0.0000 / 0.0050 |
| stress-cr | 3 | 0.4180 | 24 | 40 | 16 | 20 | 1.0114 | 0.0054 / 0.0000 / 0.0070 |
| stress-cr | 4 | 0.4036 | 29 | 35 | 15 | 22 | 1.0151 | 0.0069 / 0.0000 / 0.0090 |

## 结论

1. `stress-cr iter3` 是本轮最高 Code proxy 点，reward=0.4180；但 iter4 回落到 0.4036，说明 R1 小系数补偿没有形成单调收益。
2. `safe-all` 较稳定，iter1 和 iter4 都是 0.4160；但是这更像在 init1 附近小幅移动 Code coefficient，缺少明确的 R1 增益证据。
3. `stress-all` 在 R1 上界放宽后 reward 连续下降，说明让 `tool/memory/code` common 与 R1 一起同动会增加干扰。
4. 冻结 tool/memory 的 `stress-cr` 更合理：tool/memory effective coefficient 保持 1.0，Code/R1 独立移动，避免 Code-only 训练破坏其他任务。
5. 当前结果不足以作为论文主实验；它更适合作为 R1 异质专家接入的诊断：R1 task vector 可以被安全注入并被 gate 推动，但 Code reward 的泛化收益仍受 calibration/reward 信号限制。

## 下一步

短期只建议把 `stress-cr iter3` 与 `safe-all iter4` 做 code-only formal eval 对照。如果 formal CURE 没有超过静态 R1-inject 或 TA-1/3，则不继续在这个 Code P0 grid 上消耗主资源。

长期更应该把 R1 作为异质 expert 的结构化 prior，放回三任务 calibration 中测试是否在保持 memory/tool 的同时提升 Code，而不是只用 Code-only reward 追逐小幅 proxy 波动。
