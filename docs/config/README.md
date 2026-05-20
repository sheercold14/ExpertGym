# 实验配置与结果索引

这个目录用于记录每次 OnPolicy / Gated-GRPO 实验的可复现配置、结果路径和分析备注。原则是：所有正式启动过的实验，都要能从这里反查到“当时为什么这样跑、从哪个 checkpoint 接、loss 怎么配、结果在哪里”。

## 记录规范

每个实验批次单独建一个 Markdown：

```text
YYYYMMDD_short_name.md
```

每个批次至少记录：

| 字段 | 内容 |
|---|---|
| 实验目的 | 要验证的假设，例如 OPD-only 后期是否动力不足 |
| run_dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/...` |
| 初始化 | gate checkpoint、optimizer state 是否继承 |
| 参数化 | `global-coefficient` / `global-parameter` / `parameter` 等 |
| 数据 | seed manifest、prompt 数、samples-per-prompt、expert rollout |
| loss | GRPO/OPD/retention/PCGrad/normalization/任务均衡 |
| 优化器 | lr、SGD/AdamW、momentum、step scope、batch size |
| rollout | GPU、vLLM shards、token length、temperature/top-p |
| 结果路径 | loop manifest、per-iter summary、train.log、baked_policy、eval report |
| 观察结论 | proxy reward、gate 变化、all-fail/all-success、异常 |

## 当前活跃索引

- [20260518_code_p0_v3.md](20260518_code_p0_v3.md): Code-first P0 calibration bank，CodeContests train-only，显式 reward/guard test split，输出 `train_code64 / monitor_code32 / guard_code32`。
- [20260518_deepseek_r1_scaled_layer28.md](20260518_deepseek_r1_scaled_layer28.md): DeepSeek-R1-Distill-Qwen-7B 作为小幅 reasoning/code task vector prior 的 layer28 四专家配置，含 `reasoning=0.001` init 和 expert-specific trust region。
- [20260518_r1_codep0_sanity.md](20260518_r1_codep0_sanity.md): Code P0 v3 上的 R1 scaled layer28 code-only sanity，4 iter，验证 train code reward 和 R1/layer gate 是否出现正信号。
  - 结果报告：[../report/20260518_r1_codep0_sanity.md](../report/20260518_r1_codep0_sanity.md)。
  - 后续 R1 实验应显式设置 `COEFF_BOUND_BY_EXPERT=reasoning=0.0:0.003` 或压力测试 `reasoning=0.0:0.01`，因为逐轮 `MAX_COEFF_DELTA_BY_EXPERT` 不是全程绝对边界。
  - 安全重跑入口：`skill/command/run_20260518_r1_codep0_bounded_sanity.sh`。
- [20260518_r1_codep0_bound_grid.md](20260518_r1_codep0_bound_grid.md): R1 Code P0 bound grid，`safe/stress` x `all/code+reasoning only` 四个短程诊断，用于判断 R1 幅值上限与 common 同动是否是 Code reward 不涨的瓶颈。
- [20260519_r1math_L_experiments.md](20260519_r1math_L_experiments.md): 修正 DeepSeek-R1-Distill-Qwen-7B 的 delta base 为 Qwen2.5-Math-7B 后，L1/L2/L3 20-iter 主实验计划与命令。
- [20260519_tool_nullspace_v1.md](20260519_tool_nullspace_v1.md): Tool behavior-span null-space v1，Tool32/Memory32/Code40 calibration，R1 init=0.05，OPD+NLL retention，update 阶段投影 gate 梯度保护 Tool 行为格式。
- [hiddenstate/20260520_round13_evalleak_code16.md](hiddenstate/20260520_round13_evalleak_code16.md): Round13 formal-code eval-leak hidden-state diagnostic，构造 RF/Mem-only 与 all+R1 两个 Code16 ability-span TRC bank。
- [20260520_trc_round3_memorytraj.md](20260520_trc_round3_memorytraj.md): TRC Round3 memory trajectory 与 coefficient-retention 主线，记录 uniform4/late3/full trajectory、global/task-aware floor、Tool/Memory 快评。
- [20260520_trc_round4_codepush.md](20260520_trc_round4_codepush.md): TRC Round4 code-push variants，在 Round3 Tool/Memory 过线基础上测试 longer epochs、code loss boost 与 code full-response/global-floor。
- [20260518_p0_sota_calib_v2.md](20260518_p0_sota_calib_v2.md): P0 SOTA-oriented calibration v2，新增 `train128/monitor64/guard64`、ToolRL all80 test 口径、expert rollout 与主实验入口。
- P0 快速评测脚本：
  - `skill/command/run_20260518_toolrl_rlla4k_eval.sh`: ToolRL `rlla_4k/test` all80 overall correct。
  - `skill/command/run_20260518_sota_monitor64_eval.sh`: `sota_calib_v2_20260518/monitor64` 三任务 proxy rollout。
- [20260515_opd_continue_abcd.md](20260515_opd_continue_abcd.md): A/B OPD-only continuation 与 C/D GRPO+OPD+retention PCGuard continuation。
- [20260517_b1_oldB_lr04.md](20260517_b1_oldB_lr04.md): b1，复现 old B 设置，仅将 `LR` 从 `0.1876` 提高到 `0.4` 的 OPD-only 学习率对照。
- [20260517_c1_init1_layerband_grpo_opd_ret.md](20260517_c1_init1_layerband_grpo_opd_ret.md): c1，从 `init=1.0` 出发，用 28 个 per-layer `layer-band` 训练 `GRPO + dynamic OPD + NLL retention`。
- [20260517_a1_curealigned_b_setting.md](20260517_a1_curealigned_b_setting.md): A1，在 B 的 global-parameter 设置基础上加入 CURE-aligned mixed 96 数据、Code zscore reward 设置和 GRPO+OPD+retention。
- [20260517_norm_aware_code_calibration.md](20260517_norm_aware_code_calibration.md): task-vector 范数诊断、norm-aware init gate 与 CURE-style Code calibration blueprint。
- [20260517_selected_mode_pruning.md](20260517_selected_mode_pruning.md): init=1 下的 expert-specific selected-mode hard pruning、Tool/Memory top64 pruning + Code full、reasoning selected64@0.001 smoke。
- [../report/eval_targeted_calibration_20260517.md](../report/eval_targeted_calibration_20260517.md): 基于 case browser 的 eval-targeted96 calibration manifest。当前推荐使用 `eval_targeted96_cure_aligned_20260517`，即 Tool 16+16、Memory 32 anchor、Code 16 paper96 anchor + 16 CURE-style targeted。

## 结果文件优先级

排查问题时优先看：

1. `gated_grpo_bake_vllm_loop_manifest.json`: 每轮 bake / rollout / update 命令、耗时、输入输出。
2. `iter_*/gate_updates.summary.json`: loss 配置、frontier/retention/OPD 统计、optimizer 配置、gate 输出。
3. `iter_*/opd_distill_from_allfail.summary.json`: dynamic OPD 选择了哪些 all-fail 样本。
4. `iter_*/rollouts.summary.json`: rollout 样本数、任务分布、生成耗时。
5. `train.log`: tmux 内部完整运行日志。
6. `docs/report/*.md`: 实验解释、评测结果、失败分析。

## 快速实验采样约定

默认约定：不设置 frontier / retention 上限时使用全量。之后需要压缩 `GRPO + retention` update 成本时，显式打开随机采样和行数上限：

```bash
FRONTIER_ROWS_PER_TASK=4
FRONTIER_SAMPLE_BEFORE_LIMIT=1
FRONTIER_ORDER=task-interleaved

MAX_RETENTION_ROWS_PER_TASK=8
MAX_RETENTION_ROWS=24
RETENTION_SAMPLE_BEFORE_LIMIT=1
```

`IGNORE_CONFIG_FRONTIER_TASK_QUOTA=1` 是 wrapper 默认值，用来忽略 config 里旧的 `calibration.frontier_task_quota`，因此不设置 `FRONTIER_ROWS_PER_TASK` / `FRONTIER_*_QUOTA` 就是 full frontier。`FRONTIER_SHUFFLE_SEED` 为空时由训练 loop 按 `seed + iter - 1` 传入；`RETENTION_SHUFFLE_SEED` 为空时复用 frontier seed。这样每轮是可复现随机采样，同时避免固定取每个任务文件里的前几条。
