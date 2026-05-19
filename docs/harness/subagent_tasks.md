# Subagent Tasks

## 当前已启动

### Feynman：论文实验审计

任务：

- 审计 `docs/report`、`docs/evaluation`、`docs/config` 中已有实验。
- 映射到论文 E1-E7。
- 区分可做主 claim 的结果和只能做失败诊断的结果。
- 输出缺失的最小实验列表。

交付物：中文高密度报告给主 agent。

### Hubble：72 小时调度审计

任务：

- 审计训练脚本、历史耗时、GPU 资源。
- 给出 5 小时实验配置、8 卡并发策略、停止规则。
- 重点判断 GRPO / OPD / retention 的实际成本和收益。

交付物：中文调度建议给主 agent。

### Arendt：相关工作与审稿风险

任务：

- 审阅 `main.tex` 和 `references.bib`。
- 梳理 Task Arithmetic / TIES / DARE / WUDI / TSV / RAM / ARM / Expert Merging / OPD / GRPO 的实验化差异。
- 给出“不是 imitation、不是 sweep、不是普通 loss mixing”的防御实验。

交付物：中文科研定位报告给主 agent。

## 后续可派发任务

| 任务 | 触发条件 | 交付 |
|---|---|---|
| eval watcher | 有模型送 eval6 | 自动填 `docs/evaluation/YYYYMMDD_expertgym_72h_eval6.md` |
| figure builder | P0/P1 完成 | state distribution / coefficient trajectory 图 |
| paper editor | 主表有 2-3 个可信数值 | 修改 `docs/paper/ExpertGym/main.tex` |
| code diagnostic | Code 仍弱 | 更新 eval case browser 和 CURE failure taxonomy |

