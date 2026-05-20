# Evaluation Records

本目录作为 ExpertGym 的主评测记录入口。以后正式评测结果优先写在这里，原始产物仍保留在各 harness 输出目录中，并在本文档或对应报告里链接。

## 现有主表

| 文件 | 内容 | 原始来源 |
|---|---|---|
| `eval6-20260502-125748_zh.md` | eval6 中文总表，保持原大表结构 | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/aggregate_report_zh.md` |

## 单次实验报告

| 文件 | 内容 | 原始来源 |
|---|---|---|
| `20260516_tonight_abc_eval6.md` | 2026-05-16 A/B/C 受控实验正式 eval6 结果 | Tool/Memory/Code 三类 `summary.json`，eval runner `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_expertgym_abc_20260516_addons.py` |
| `20260517_defg_eval6.md` | 2026-05-17 D/E/F/G 正式 eval6 结果，含 F/G final iter20 补充评测 | Tool/Memory/Code `summary.json`，eval runner `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_expertgym_defg_20260517_addons.py` |
| `20260519_r1math_L_eval6.md` | 2026-05-19 Correct-R1 L 系列 `L1 iter009` / `L2 iter009` 正式 Eval6，当前 running | Tool/Memory/Code `summary.json`，eval runner `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_expertgym_r1math_L_20260519_addons.py` |
| `20260519_stage1_candidates_eval.md` | 2026-05-19 TRC directional stage-1 candidate full-suite evaluation | Tool/Memory done，Code CURE running |
| `20260519_trc_round1_eval.md` | 2026-05-19/20 TRC Round1/Round2 candidate monitor table | E1/E3 Code running，R2D/R2E Tool done + Memory running |
| `20260520_trc_round3_eval.md` | 2026-05-20 TRC Round3 memory-trajectory + coefficient-retention evaluation | R3D Tool/Memory done and Code running，R3I/R3J fast eval running |
| `20260520_trc_round4_eval.md` | 2026-05-20 TRC Round4 stronger-gate code-push evaluation | R4A promoted to Code，R4B/R4C rejected |

## 记录格式

每个正式评测报告保持和原 eval6 大表一致的结构：

1. **评测对象与口径**：模型名、checkpoint 路径、评测 batch id、数据集口径。
2. **总表**：列名保持：
   `模型 / 类型 / Tool 均值 / Tool Live 均值 / Memory EM / Memory F1 / Code Acc / Code TP / Code BoN(4,4) Acc`。
3. **Tool/BFCL 明细**：`parallel / parallel_multiple / live_parallel / live_parallel_multiple / 均值`。
4. **Memory/HotpotQA 明细**：`eval_50 F1 / eval_100 F1 / eval_qa_1_32768 F1 / eval_qa_1_65536 F1 / 平均 EM / 平均 F1`。
5. **Code/CURE 明细**：`LiveBench Acc/TP/BoN / LiveCodeBench Acc/TP/BoN / 平均 Acc / 平均 BoN`。
6. **结果文件**：Tool、Memory、Code 的 `summary.json` 路径，以及评测 runner / log 路径。

## 命名约定

- 大批次主表：`eval6-<batch_id>_zh.md`。
- 单次实验评测报告：`YYYYMMDD_<experiment_name>_eval6.md`。
- 未完成的评测可以先写 `pending`，但必须标明已完成和缺失的子项；完成后再替换为正式数值。

## 最近完成

`2026-05-16` A/B/C 受控实验正式 eval6 已完成。训练报告在：

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/20260516_tonight_abc.md`

评测报告：

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260516_tonight_abc_eval6.md`

## 2026-05-17 P0

- [20260517_p0_ta13_eval6.md](20260517_p0_ta13_eval6.md): strict `TA-1/3` formal eval6, used as symmetric-prior baseline for ExpertGym paper.
- [20260517_p0_static_baselines_eval6.md](20260517_p0_static_baselines_eval6.md): P0 static baseline table for `TA-1/3`, `TA-0.75`, and `init1`; Code rows are updated as CURE finishes.
- [20260518_baselines_eval6.md](20260518_baselines_eval6.md): paper baseline batch for TIES, DARE-TA, DARE-TIES, and AdaMerging; WUDI/ExpertMerging are indexed but not rerun.
- [20260518_p1_evaltarget_candidates.md](20260518_p1_evaltarget_candidates.md): P1 eval-targeted training candidate queue; entries are not final eval6 until explicitly marked done.
- [20260518_p1_evaltarget_eval6.md](20260518_p1_evaltarget_eval6.md): P1 eval-targeted candidate formal Eval6 table; Tool results are filled first, Memory/Code rows are completed as harnesses finish.
