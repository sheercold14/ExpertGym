# 2026-05-16 晚间 A/B/C 受控实验报告

## 实验设置

配置文档：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260516_tonight_abc.md`

| 实验 | run dir | 变量 |
|---|---|---|
| A | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expA_gp_grpo_opd_from_init_20260516` | 从初始 `1/3` global-parameter gate 加入等权 `GRPO + OPD` |
| B | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expB_gp_code_opd_aug_20260516` | 只扩充 code OPD expert positive pool |
| C | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expC_gp_reasoning_code_opd_aug_20260516` | 在 B 基础上加入 init=0、可学习的 DeepSeek reasoning task vector |

公共训练数据：`/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl`

新增 code OPD 数据目录：`/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/`

## 当前状态

更新时间：`2026-05-16 15:25 CST`。

| 项 | 状态 | 证据 |
|---|---|---|
| A 训练 | 已完成 20 iter | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expA_gp_grpo_opd_from_init_20260516/strategy_summary.json` |
| B 训练 | 已完成 20 iter | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expB_gp_code_opd_aug_20260516/strategy_summary.json` |
| C 训练 | 已完成 20 iter | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expC_gp_reasoning_code_opd_aug_20260516/strategy_summary.json` |
| reasoning mode | 已完成 | `/tmp/shared-storage/OnPolicy/modes/opvec4_reasoning_20260516/mode_manifest.json` |
| 正式 eval6 | 已启动 | `tmux attach -t eval_expertgym_abc_20260516` |

训练全部跑满，未发现训练主流程中断。A 的日志末尾出现一次 bash 后处理变量报错，但发生在 `[done] summary` 之后；20 个 `gate_updates.summary.json`、`rollouts.summary.json`、`strategy_summary.json` 均已落盘，不影响训练产物。

## Code Expert Coverage

旧 code expert rollout：`17/32` prompts 至少有一个 `reward_train >= 1.0` 的 positive。

| rollout | covered prompts | 备注 |
|---|---:|---|
| ReasonFlux-Coder-7B s8 seed20260516 | `22/32` | 新增 BoN；coverage `0.6875`，positive histogram 见 `.coverage.json` |
| ReasonFlux-Coder-7B s8 seed20260517 | `25/32` | 额外 BoN；coverage `0.78125` |
| DeepSeek-R1-Distill-Qwen-7B s8 | `19/32` | 新增 reasoning/code source；coverage `0.59375` |
| RL-MemoryAgent-7B s8 | `19/32` | 新增 cross-expert source；coverage `0.59375` |

五个 code positive 源，即旧 `code_s2` 加四个新增 rollout，union 覆盖 `27/32` prompts；仍无 expert positive 的 prompt 为：

```text
code__11a303002e6dbf74
code__1663d948eff01b1a
code__46794ab3e0b79809
code__6df1c33cbaadcc0e
code__9ef2816bdb2e521f
```

## 训练动态

`overall proxy reward` 按 `tool / memory / code` 三任务 mean reward 简单平均。best checkpoint 选对应 iteration 的 `baked_policy`，即该轮 rollout 实际使用并被 proxy 验证过的模型。

| run | best proxy iter | best overall | tool | memory | code | best global tool | best global memory | best global code | best global reasoning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 18 | 0.7048 | 0.9600 | 0.7734 | 0.3809 | 0.3299 | 0.5261 | 0.3604 | n/a |
| B | 18 | 0.6959 | 0.9700 | 0.6797 | 0.4379 | 0.3296 | 0.5020 | 0.3838 | n/a |
| C | 16 | 0.6845 | 0.9472 | 0.6797 | 0.4268 | 0.3262 | 0.4861 | 0.3596 | 0.0529 |

| run | first overall | final overall | delta | final tool | final memory | final code | final mean gate tool | final mean gate memory | final mean gate code | final mean gate reasoning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.4167 | 0.6821 | +0.2654 | 0.9506 | 0.7734 | 0.3223 | 0.3292 | 0.5509 | 0.3642 | n/a |
| B | 0.4256 | 0.6909 | +0.2653 | 0.9732 | 0.7344 | 0.3652 | 0.3285 | 0.5273 | 0.3898 | n/a |
| C | 0.4217 | 0.6604 | +0.2387 | 0.9518 | 0.6641 | 0.3652 | 0.3248 | 0.5116 | 0.3657 | 0.0453 |

## 结论

- 三组都能把 proxy overall 从约 `0.42` 推到 `0.66-0.70`，说明 OPD/retention/GRPO 组合可以继续推动 task vector，训练不是失效状态。
- A 的 best overall 最高，主要来自 memory 和 tool；但 code 从首轮到末轮下降 `-0.1016`，说明加入 GRPO 后没有解决 code proxy 不稳。
- B 的 code OPD 扩充有效：best code reward `0.4379` 是三组最高，final code gate mean `0.3898` 也最高；但 code reward 仍波动，说明只扩充 expert positive 不能保证 CURE 泛化。
- C 加 reasoning vector 后没有超过 B。reasoning gate 从 `0` 被推到正值，但 best 时只有约 `0.0529`，final 约 `0.0453`；它增加了梯度/耗时，却没有带来更高三任务 proxy。
- 当前最值得正式评测的是 A-iter18 和 B-iter18；C-iter16 作为 reasoning-vector ablation 保留。

## 正式评测

已启动 eval6 add-on：

```bash
tmux attach -t eval_expertgym_abc_20260516
```

评测 runner：

`/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_expertgym_abc_20260516_addons.py`

送评模型：

| eval name | model path |
|---|---|
| `expertgym-abcA-gp-grpo-opd-i18-20260516` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expA_gp_grpo_opd_from_init_20260516/iter_018/baked_policy` |
| `expertgym-abcB-gp-codeaug-opd-i18-20260516` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expB_gp_code_opd_aug_20260516/iter_018/baked_policy` |
| `expertgym-abcC-gp-reasoning-codeaug-i16-20260516` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expC_gp_reasoning_code_opd_aug_20260516/iter_016/baked_policy` |

待正式 eval6 完成后，继续补 Tool/BFCL、Memory/HotpotQA、Code/CURE 结果。
