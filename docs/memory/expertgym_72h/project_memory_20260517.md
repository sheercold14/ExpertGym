# ExpertGym 72h Project Memory

日期：2026-05-17

## 论文主线

核心句子：

```text
Task vectors provide structured priors; executable feedback learns their composition.
```

方法不应被写成“GRPO + OPD + retention 的调参配方”。正确表述是：

- task vectors 是 frozen priors；
- calibration prompts 是 probes；
- rollout verifier 把 prompts 路由到 state；
- state 决定 credit operator；
- coefficient update 是小参数空间中的 executable-feedback learning。

## 四类 state

| state | 条件 | credit | operator |
|---|---|---|---|
| frontier | 同 prompt 多 rollout reward 有差异 | direction | GRPO |
| recoverable | current all-fail, same-prompt expert success | recovery | Recovery-OPD |
| stable | current all / mostly success | boundary | Retention |
| unsolved | current all-fail, expert also fails | none | skip / data acquisition |

## 当前可信事实

正式 eval6：

- best-ever TAME: Tool `0.7954`, Tool Live `0.7083`, Memory F1 `0.7720`, Code Acc `0.3597`, Code BoN `0.4408`。
- TA-0.75: Tool `0.7850`, Tool Live `0.6875`, Memory F1 `0.7588`, Code Acc `0.3585`, Code BoN `0.4222`。
- `opvec-gp-opd-best-iter9`: Tool `0.7835`, Memory F1 `0.7649`, Code Acc `0.3487`, Code BoN `0.4144`；只作为历史诊断参考，不作为主线起点或复现实验基础。
- ExpG init1 GRPO+OPD+Ret final: Tool `0.7942`, Tool Live `0.7083`, Memory F1 `0.7548`, Code Acc `0.3382`, Code BoN `0.4252`。
- ExpF init1 GRPO+Ret final: Tool `0.7788`, Tool Live `0.6875`, Memory F1 `0.7612`, Code Acc `0.3460`, Code BoN `0.3998`。
- ABC-A 1/3 GRPO+OPD: Tool `0.7823`, Memory F1 `0.7346`, Code Acc `0.3431`, Code BoN `0.3919`。

训练经验：

- OPD-only + retention 能快速推 proxy，从约 `0.42` 到 `0.66-0.70`。
- full GRPO+OPD+retention 旧设置一轮约 30 分钟，主要慢在 HF update logprob/backward。
- fast setting 应随机限制 frontier 每任务 4 条、retention 每任务 8 条，单实验控制在 5 小时内。
- `UPDATE_BATCH_SIZE=8` 是当前两卡较稳的 batch 上限；`16` 风险高。
- 默认不设置 frontier / retention 上限时应该 full；显式设置才采样。

## 当前代码高价值能力

- `skill/command/run_qbank_c033333_gate_strategy.sh`: 主训练 wrapper。
- `scripts/train/opvec_gated_grpo_bake_vllm_loop.py`: bake + vLLM rollout + dynamic OPD + update loop。
- `scripts/train/opvec_update_gates_from_rollouts.py`: GRPO / OPD / retention update。
- `scripts/data/build_opd_distill_from_expert_rollouts.py`: same-prompt expert recovery rows。
- `scripts/monitor/opvec_run_monitor.py`: 前端监控。
- `scripts/eval/summarize_gate_strategy_run.py`: run 总结。
- `scripts/analysis/build_eval_case_browser.py`: formal eval case browser。

## 新增配置原则

快速采样：

```bash
FRONTIER_ROWS_PER_TASK=4
FRONTIER_SAMPLE_BEFORE_LIMIT=1
MAX_RETENTION_ROWS_PER_TASK=8
MAX_RETENTION_ROWS=24
RETENTION_SAMPLE_BEFORE_LIMIT=1
```

不设置时默认 full frontier / full retention。

## 2026-05-17 P0 当前启动项

- 已烘焙 `TA-1/3` checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/ta_c033333_global_20260517`，`num_delta_entries=588`。
- 已烘焙 `init1` checkpoint：`/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517`，`num_delta_entries=588`。
- `TA-1/3 formal eval6` 正在 `eval_p0_ta13_eval6_20260517` 中运行，summary dir 为 `/tmp/shared-storage/OnPolicy/eval/full_suite/ta-c033333-global-20260517/20260517_p0_ta13_eval6`。
- `TA-1/3 K=8 state rollout` 正在 `p0_state_ta13_k8_20260517` 中运行。
- `init1 K=8 state rollout` 正在 `p0_state_init1_k8_20260517` 中运行。
- Tool formal 早期结果：parallel `0.9150`，parallel_multiple `0.8700`，live_parallel `0.6875`，live_parallel_multiple `0.6667`。

## Calibration 设计原则

用户明确指出 calibration 是突破口。后续不能只沿用 paper96 或追历史 best checkpoint；必须针对 reward 和 distillation target 设计 probe bank：

- 每条样本要标注 state：frontier / recoverable / stable / unsolved。
- Tool 需要 BFCL-style schema/function/default/canonical/parallel probes，但 train/monitor schema/entity 必须 disjoint。
- Memory OPD 必须覆盖 update turns + final turn，不能只蒸馏 final answer。
- Code OPD positive 必须是 verified code solution，不能用自然语言解题过程或只过 public examples 的轨迹。
- 必须区分 train calibration 和 heldout monitor calibration；最终 checkpoint 不能只按 train calibration reward 选择。

详细设计见 `docs/harness/calibration_design.md`。

## 主风险

1. OPD dominant：必须解释为 early recovery，而不是 imitation。
2. GRPO weak/slow：必须通过 state distribution 解释 frontier 稀疏。
3. init1 strong：必须把它写成 stronger static prior，而不是推翻 1/3。
4. Code weak：必须拆 generation vs selection，不能只看 avg reward。
5. Tool live weak：必须用 BFCL live-style calibration 解释，不要只推 tool gate。
6. coefficient learning 不新：必须对照 AdaMerging / Expert Merging / static sweep。
7. GRPO 公式表述风险：ratio 应是 trajectory logprob ratio under old/new coefficient-induced policies。

## 72h 目标

最低闭环：

1. P0 state distribution。
2. TA-1/3 formal eval6。
3. 从原则化 prior 出发做 P1：`TA-1/3` symmetric prior 与 `init1/scale-calibrated` strong prior。
4. P1 global 3 / common+residual 4 main method。
5. OPD-only、GRPO-only、full routed 三个 ablation。
6. 至少 2 个候选完整 eval6。
7. 论文 E1-E5 至少填出可信数值和分析。
