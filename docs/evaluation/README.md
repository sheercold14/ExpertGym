# Evaluation Records

本目录作为 ExpertGym 的主评测记录入口。以后正式评测结果优先写在这里，原始产物仍保留在各 harness 输出目录中，并在本文档或对应报告里链接。

## 现有主表

| 文件 | 内容 | 原始来源 |
|---|---|---|
| `eval6-20260502-125748_zh.md` | eval6 中文总表，保持原大表结构 | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/aggregate_report_zh.md` |

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

## 当前进行中

`2026-05-16` A/B/C 受控实验的正式 eval6 正在运行。训练报告在：

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/20260516_tonight_abc.md`

待 CURE 完成后，将新增：

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260516_tonight_abc_eval6.md`
