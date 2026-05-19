# Reviewer Risks and Baseline Defense

本文件记录 ExpertGym 当前最容易被审稿人质疑的点，以及必须用实验而不是文字解决的防线。

## 需要收窄的 claim

| 当前表述风险 | 审稿人可能质疑 | 推荐表述 |
|---|---|---|
| agent task vectors cannot be composed reliably from geometry alone | 太绝对，几何 baseline 可能很强 | geometry-only merging misses rollout demand signals in our executable suite |
| calibration cannot memorize because only coefficients trainable | layer/module/588 参数仍会 overfit small calibration | low-dimensional global setting has limited capacity; fine-grained settings require held-out guard |
| Recovery-OPD is not imitation | loss 形式仍是 expert NLL | it is verifier-triggered same-prompt recovery; prove by imitation baselines |
| 1/3 prior is natural | scale factor 任意，TA-0.75 很强 | 1/3 is symmetric prior; scale-calibrated priors are stronger geometry starts |
| GRPO optimizes coefficient ratio | RL reviewer 会指出 ratio 应是 trajectory probability ratio | ratio is old/new policy logprob under different coefficient-induced policies |

## 必须考虑的 baseline

### Geometry-only

| Baseline | 最低要求 | 作用 |
|---|---|---|
| TA-1/3 | 必跑/已部分有 | symmetric prior |
| TA-0.75 / best static scale | 必跑/已有 | strong static baseline |
| per-expert static grid / sweep | 必跑 global 3 | 防止方法被看成手调系数 |
| TIES / DARE / Fisher / WUDI / TSV | TIES/DARE 已补同 eval6；Fisher 需单独实现对角 Fisher 统计；WUDI/ExpertMerging 先索引已有目录 | 证明 geometry/statistics-only 不看 executable demand |
| AdaMerging-style coefficient learning | 需要补文献和可行 baseline | 最接近“learn coefficients from calibration”的图像/多任务合并工作 |

### Imitation / Alignment

| Baseline | 最低要求 | 作用 |
|---|---|---|
| offline expert trace distill | OPD expert rows 不看 current failure | 证明 generic imitation 不等于 recovery |
| logit imitation | 可选，若工程重可作为 related-work baseline | 对照 GKD / distill |
| Expert Merging hidden/logit alignment | 最强同类对照 | 证明 executable feedback 与 hidden/logit alignment 差异 |
| same-prompt OPD | 当前核心 | current all-fail + expert success |
| same-prompt OPD + retention/guard | 主方法防线 | 证明非退化 recovery |

### Loss Routing

| Baseline | 作用 |
|---|---|
| GRPO-only | frontier direction 是否足够 |
| OPD-only | recoverable recovery 是否足够 |
| retention-only | boundary term 是否只保守 |
| unrouted weighted sum | 防止被看成 loss mixture |
| random routing / state permutation | 证明 state label 有意义 |
| task-only routing | 区分 task label 和 execution state |

## 最强防御实验包

优先级按 72h 内可完成度排序：

1. **State distribution table**：frontier / recoverable / stable / unsolved。没有这个表，GRPO 弱和 OPD 强都解释不清。
2. **Global coefficient sweep defense**：在 global 3/4 空间做静态 scale/sweep 或至少引用已跑 scale sweep，证明 ExpertGym 不是简单选系数。
3. **Recovery vs imitation**：offline OPD vs same-prompt OPD vs same-prompt OPD+retention。
4. **Routed vs unrouted**：full routed vs random/unrouted loss mixture。
5. **Non-regression Pareto**：average score vs worst-task drop，不能只报均值。
6. **Geometry + feedback**：如果有时间，把 TA/TIES/DARE/WUDI/TSV 中一个强 prior 接 ExpertGym feedback，证明互补。
7. **Mixed-agent heldout**：P3，可极大增强“composition”叙事，但不能挤占 P0/P1。

## 论文写作约束

- 不 claim SOTA，除非同套 eval6 覆盖 RAM/ARM/Expert Merging/strong geometry。
- 如果 OPD 主导，就写成 early-stage recovery，不要隐藏。
- 如果 GRPO 弱，就明确它只对 frontier prompts 有效。
- 如果 TA-0.75 很强，就把 ExpertGym 写成 executable refinement of strong priors。
- Code 不应只看 Acc；必须拆 `sample acc / any-pass@K / BoN selection`。
- Tool 不应只看 mean；必须拆 BFCL live、parallel alignment、schema/default/canonicalization。
