# 2026-05-16 A/B/C 受控实验 eval6 正式评测

完成时间：`2026-05-16 17:10 CST`。

## 一、实验设置

| 实验 | eval name | checkpoint | 变量 |
| --- | --- | --- | --- |
| A | `expertgym-abcA-gp-grpo-opd-i18-20260516` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expA_gp_grpo_opd_from_init_20260516/iter_018/baked_policy` | 从初始 `1/3` global-parameter gate 加入等权 `GRPO + OPD` |
| B | `expertgym-abcB-gp-codeaug-opd-i18-20260516` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expB_gp_code_opd_aug_20260516/iter_018/baked_policy` | OPD-only + retention，仅扩充 code expert positive pool |
| C | `expertgym-abcC-gp-reasoning-codeaug-i16-20260516` | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/expC_gp_reasoning_code_opd_aug_20260516/iter_016/baked_policy` | B + init=0、可学习 DeepSeek-R1-Distill-Qwen-7B reasoning task vector |

公共配置：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/config/20260516_tonight_abc.md`

训练报告：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/20260516_tonight_abc.md`

训练 launcher：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/skill/command/run_20260516_tonight_abc.sh`

eval runner：`/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_expertgym_abc_20260516_addons.py`

## 二、总表

| 模型 | Tool 均值 | Tool Live 均值 | Memory EM | Memory F1 | Code Acc | Code TP | Code BoN(4,4) Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expertgym-abcA-gp-grpo-opd-i18-20260516 | 0.7823 | 0.6771 | 0.6055 | 0.7346 | 0.3431 | 0.4792 | 0.3919 |
| expertgym-abcB-gp-codeaug-opd-i18-20260516 | 0.7823 | 0.6771 | 0.5879 | 0.7118 | 0.3370 | 0.4782 | 0.3871 |
| expertgym-abcC-gp-reasoning-codeaug-i16-20260516 | 0.7898 | 0.6771 | 0.5586 | 0.6862 | 0.3186 | 0.4444 | 0.3675 |

口径：Tool 均值为 `parallel`、`parallel_multiple`、`live_parallel`、`live_parallel_multiple` 简单平均；Tool Live 为两个 live 子集平均；Memory 为四个 HotpotQA 数据集简单平均；Code 为 `LiveBench` 与 `LiveCodeBench` 简单平均，BoN 使用 `(4, 4).acc`。

## 三、明细

### Tool/BFCL

| 模型 | parallel | parallel_multiple | live_parallel | live_parallel_multiple | 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| expertgym-abcA-gp-grpo-opd-i18-20260516 | 0.9050 | 0.8700 | 0.6875 | 0.6667 | 0.7823 |
| expertgym-abcB-gp-codeaug-opd-i18-20260516 | 0.9050 | 0.8700 | 0.6875 | 0.6667 | 0.7823 |
| expertgym-abcC-gp-reasoning-codeaug-i16-20260516 | 0.9200 | 0.8850 | 0.6875 | 0.6667 | 0.7898 |

### Memory/HotpotQA

| 模型 | eval_50 F1 | eval_100 F1 | eval_qa_1_32768 F1 | eval_qa_1_65536 F1 | 平均 EM | 平均 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expertgym-abcA-gp-grpo-opd-i18-20260516 | 0.7345 | 0.7428 | 0.7339 | 0.7271 | 0.6055 | 0.7346 |
| expertgym-abcB-gp-codeaug-opd-i18-20260516 | 0.6963 | 0.7190 | 0.7057 | 0.7260 | 0.5879 | 0.7118 |
| expertgym-abcC-gp-reasoning-codeaug-i16-20260516 | 0.7020 | 0.6935 | 0.6646 | 0.6846 | 0.5586 | 0.6862 |

### Code/CURE

| 模型 | LiveBench Acc | LiveBench TP | LiveBench BoN | LiveCodeBench Acc | LiveCodeBench TP | LiveCodeBench BoN | 平均 Acc | 平均 BoN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expertgym-abcA-gp-grpo-opd-i18-20260516 | 0.3887 | 0.5113 | 0.4453 | 0.2975 | 0.4471 | 0.3386 | 0.3431 | 0.3919 |
| expertgym-abcB-gp-codeaug-opd-i18-20260516 | 0.3672 | 0.4983 | 0.4219 | 0.3068 | 0.4581 | 0.3523 | 0.3370 | 0.3871 |
| expertgym-abcC-gp-reasoning-codeaug-i16-20260516 | 0.3496 | 0.4598 | 0.3984 | 0.2877 | 0.4290 | 0.3366 | 0.3186 | 0.3675 |

## 四、结果文件

| 模型 | Tool summary | Memory summary | Code summary |
| --- | --- | --- | --- |
| expertgym-abcA-gp-grpo-opd-i18-20260516 | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/expertgym-abcA-gp-grpo-opd-i18-20260516/tool/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/expertgym-abcA-gp-grpo-opd-i18-20260516/eval6-20260502-125748/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/expertgym-abcA-gp-grpo-opd-i18-20260516/eval6-20260502-125748/summary.json` |
| expertgym-abcB-gp-codeaug-opd-i18-20260516 | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/expertgym-abcB-gp-codeaug-opd-i18-20260516/tool/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/expertgym-abcB-gp-codeaug-opd-i18-20260516/eval6-20260502-125748/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/expertgym-abcB-gp-codeaug-opd-i18-20260516/eval6-20260502-125748/summary.json` |
| expertgym-abcC-gp-reasoning-codeaug-i16-20260516 | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/runs/expertgym-abcC-gp-reasoning-codeaug-i16-20260516/tool/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-memory-hotpotqa/expertgym-abcC-gp-reasoning-codeaug-i16-20260516/eval6-20260502-125748/summary.json` | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval6-code-cure-full/expertgym-abcC-gp-reasoning-codeaug-i16-20260516/eval6-20260502-125748/summary.json` |

## 五、结论

- A 在 A/B/C 中 Memory 与 Code 综合最好：Memory F1 `0.7346`，Code Acc `0.3431`，Code BoN `0.3919`。
- C 的 Tool 均值最高（`0.7898`），但 Memory 与 Code 均低于 A/B；reasoning vector 这轮没有带来正式 eval6 增益。
- B 的 code expert positive pool 扩充没有转化为超过 A 的 Code/CURE 结果；B 的 LiveCodeBench Acc/TP/BoN 高于 A，但 LiveBench 降幅更大，双数据集平均后仍低于 A。
