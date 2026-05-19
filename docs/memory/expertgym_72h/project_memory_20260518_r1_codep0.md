# ExpertGym 72h Memory: 2026-05-18 R1/Code P0 状态

## 当前主线

论文主 claim 仍然是：

```text
task vectors provide structured priors; executable feedback learns their composition
```

当前 P0 瓶颈是 Code：已有静态 R1-inject `arm-r-v2_plus_r1_alpha0.001` 在 eval6 上已经很强，Code mean Acc=0.3592，Memory F1=0.7586，Tool mean=0.7942。要让 ExpertGym 方法成立，需要证明 learned composition 能在 frozen task-vector prior 上接近或超过静态注入，而不是只靠手工 alpha。

## 2026-05-18 新增机制

为防止 DeepSeek-R1-Distill-Qwen-7B task vector 系数漂移过大，训练脚本新增：

```text
--coefficient-bound-by-expert reasoning=low:high
```

已接入：

- `scripts/train/opvec_update_gates_from_rollouts.py`
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`
- `skill/command/run_qbank_c033333_gate_strategy.sh`

含义：对 effective coefficient 做专家级绝对投影，例如 `reasoning=0.0:0.003`。这与旧的 `MAX_COEFF_DELTA_BY_EXPERT` 不同；旧参数主要控制相对初始点的更新幅度，不能保证 R1 effective 系数非负或绝对不超界。

测试：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python tests/test_gated_grpo_trust_region.py
```

结果：5 tests passed。

## R1 Code P0 Sanity

报告：

```text
docs/report/20260518_r1_codep0_sanity.md
docs/config/20260518_r1_codep0_sanity.md
```

run：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/r1_codep0_layer28_z001_codeonly_sanity_20260518
```

结论：R1 系数可被推高，但旧设置下 R1 effective coefficient 会越界甚至为负，必须加绝对边界。

## R1 Code P0 Bound Grid

报告：

```text
docs/report/20260518_r1_codep0_bound_grid.md
docs/config/20260518_r1_codep0_bound_grid.md
```

四个短程诊断：

| run | 状态 | 结论 |
|---|---|---|
| `r1_codep0_safe_all_bound003_20260518` | iter4 done | 稳定但没有明显 Code proxy 增益 |
| `r1_codep0_stress_all_bound01_20260518` | iter3 done | R1 放宽但 all-expert/common 同动导致 reward 下降 |
| `r1_codep0_safe_cr_bound003_20260518` | iter3 done | 冻结 tool/memory 后更干净，但 safe bound 太紧 |
| `r1_codep0_stress_cr_bound01_20260518` | iter4 done | iter3 为本轮最高 proxy 0.4180，iter4 回落 |

最值得 eval 的两个 checkpoint：

```text
stress-cr iter3:
/tmp/shared-storage/OnPolicy/runs/gated_grpo/r1_codep0_stress_cr_bound01_20260518/iter_003/baked_policy

safe-all iter4:
/tmp/shared-storage/OnPolicy/runs/gated_grpo/r1_codep0_safe_all_bound003_20260518/iter_004/baked_policy
```

但当前结论是：Code-only R1 小系数学习还不足以作为主实验，只能作为异质 expert 接入诊断。后续应把 R1 放回三任务 calibration，测试是否在不破坏 tool/memory 的前提下提升 Code。

## 当前 GPU 状态记录

2026-05-18 17:55 CST 附近：

```text
GPU 0,1,6,7: 空闲
GPU 2,3: expD_r1scaled_3band_20260518
GPU 4,5: expD_r1scaled_layer28_20260518
```

当前有价值的 4 卡 run：

```text
expD_r1scaled_3band_20260518
expD_r1scaled_layer28_20260518
```

来源脚本：

```text
scripts/train/run_4expert_r1scaled.sh
```

设置摘要：

```text
4 experts: tool, memory, code, reasoning
init: tool/memory/code=1/3, reasoning=0
loss: OPD + retention only, no GRPO
calibration: qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
3band lr=0.25, layer28 lr=0.10
target: memory >= 0.55 within 10 iters
```

注意：这两路是从 1/3 初始化的 R1-scaled OPD-only 结构诊断，与 Code P0 init1/R1 bound grid 不是同一条实验线。

## 下步优先级

1. 继续监控 `expD_r1scaled_*`：若 memory/code proxy 不上升或 reasoning 系数失控，应早停。
2. 在只占 4 卡的前提下，可对 `stress-cr iter3` 做 code-only formal eval；但这不是最高优先级，除非需要验证 R1 bound grid 是否有意外 heldout 收益。
3. 主线应转向三任务 calibration + R1 异质 expert 学习，目标是接近或超过静态 R1-inject 的 Code/Memory，同时保持 Tool。
