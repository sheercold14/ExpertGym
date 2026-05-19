# ExpertGym Experiment Matrix

## 与论文表格的映射

| 论文实验 | 必需程度 | 目标证据 | 对应 run / 数据 |
|---|---:|---|---|
| E1 Task-vector priors | P0 | base/expert/TA prior 有结构能力 | 复用 eval6 主表 + 补 TA-1/3/TA-0.75 |
| E2 Geometry vs executable feedback | P1 | ExpertGym 在非退化指标上优于或补充 static merge | TA-1/3, TA-0.75, best static/TAME, EG-main |
| E3 State distribution | P0 | frontier/recoverable/stable/unsolved 解释 credit collapse | `P0-state-c033-k8`, `P0-state-init1-k8` |
| E4 Routing matters | P1 | state-routed > GRPO-only / OPD-only / retention-only / unroute | fast ablations |
| E5 Recovery not imitation | P1 | same-prompt recoverable OPD 更安全 | OPD-only vs offline imitation |
| E6 Code diagnostic | P2 | Code 失败分解与 CURE mismatch | eval case browser + CURE-aligned calibration |
| E7 Capacity ladder | P2 | global/layer/module trade-off | global 3, common+residual 4, layer28, global-parameter |
| E8 Mixed-agent generalization | P3 | 组合泛化 | 暂不占前 48h GPU |

## 审稿防线矩阵

| 质疑 | 必要实验 | 72h 内策略 |
|---|---|---|
| 这只是 coefficient sweep | global 3/4 static sweep 或 best scale baseline | 先整理已有 TA scale sweep；若缺 global 3 grid，跑小规模 offline sweep |
| 这只是 imitation | offline expert trace distill vs same-prompt recovery OPD | P1 防御实验 |
| 这只是 loss mixing | routed vs unrouted vs random routing | P1/P2 防御实验 |
| geometry baseline 不充分 | TA/TIES/DARE/WUDI/TSV/AdaMerging-style | 先补 TA/best static；AdaMerging 作为 related baseline 写入待补 |
| 只提升平均、破坏单项 | non-regression Pareto and worst-task drop | 所有主表必须报告 |
| GRPO 没贡献 | frontier-only + state distribution | 用 E3 解释 GRPO 只对 frontier 有效 |

## 主实验矩阵

## 已有结果可填表

| 模型 / run | Tool | Tool Live | Memory F1 | Code Acc | Code BoN | 结论 |
|---|---:|---:|---:|---:|---:|---|
| TA-0.75 | 0.7850 | 0.6875 | 0.7588 | 0.3585 | 0.4222 | 强静态 baseline，但不能替代 TA-1/3 |
| best-ever TAME | 0.7954 | 0.7083 | 0.7720 | 0.3597 | 0.4408 | 当前最强 overall reference，不是 ExpertGym 主方法 |
| OP-VEC GP OPD best iter9 | 0.7835 | pending | 0.7649 | 0.3487 | 0.4144 | 历史诊断参考，不作为主线起点 |
| ExpG init1 final | 0.7942 | 0.7083 | 0.7548 | 0.3382 | 0.4252 | init1 strong prior ablation，Tool/BoN 好但 Code Acc 弱 |

## 必补最小行

| 编号 | 实验 | 目的 | 优先级 |
|---|---|---|---:|
| E1/E2-min | TA-1/3 formal eval6 | equal-prior reference，防止 baseline 选择偏差 | P0 |
| E3-min | 1/3 与 init1 的 K=8 state distribution | 解释 frontier / recoverable / stable / unsolved 分布 | P0 |
| E4-min | 同设置下 GRPO-only / OPD-only / Full routed | 证明 routing operator 各自贡献 | P1 |
| E5-min | offline imitation baseline | 区分 Recovery-OPD 与 generic imitation | P1 |

### P0 State Distribution

| run name | init | prompts | K | output | GPU | 预算 |
|---|---|---:|---:|---|---|---:|
| `p0_state_c033_k8_20260518` | 1/3 | 96/150 | 8 | state table | 4,5 | 2h |
| `p0_state_init1_k8_20260518` | 1.0 | 96/150 | 8 | state table | 6,7 | 2h |

统计字段：

- prompt-level success count / reward variance。
- frontier: `0 < success_count < K` 或 reward std 足够。
- stable: all / mostly success。
- recoverable: current all-fail + expert rollout has verified positive。
- unsolved: current all-fail + no expert positive。

### P1 Main Runs

| run name | init | strategy | loss | fast sampling | 目标 |
|---|---|---|---|---|---|
| `eg72_main_gc_c033_fast` | 1/3 | global-coefficient | GRPO+OPD+Ret | yes | 最干净 3 系数主方法 |
| `eg72_main_global_c033_fast` | 1/3 | global common+residual | GRPO+OPD+Ret | yes | 4 参数 common/residual |
| `eg72_main_gc_init1_fast` | init1 | global-coefficient | GRPO+OPD+Ret | yes | strong prior + executable refinement |
| `eg72_opd_gc_c033_fast` | 1/3 | global-coefficient | OPD+Ret | yes | recovery-only ablation |
| `eg72_grpo_gc_c033_fast` | 1/3 | global-coefficient | GRPO+Ret | yes | frontier-only ablation |
| `eg72_offline_opd_gc_c033_fast` | 1/3 | global-coefficient | offline OPD+Ret | yes | imitation baseline |
| `eg72_unrouted_gc_c033_fast` | 1/3 | global-coefficient | mixed all losses without state routing | yes | loss mixing baseline |

P1 默认不再直接使用原始 `paper96` 作为主 bank。`paper96` 是 anchor/baseline；主方法应使用 state-audited `train96`，并用 `monitor48/guard48` 防 calibration overfit。

默认 fast 参数：

```bash
NUM_ITERS=12
NUM_PROMPTS=96
SAMPLES_PER_PROMPT=4
UPDATE_BATCH_SIZE=8
OPTIMIZER_STEP_SCOPE=epoch
LOSS_GRANULARITY=sequence
FRONTIER_ROWS_PER_TASK=4
FRONTIER_SAMPLE_BEFORE_LIMIT=1
MAX_RETENTION_ROWS_PER_TASK=8
MAX_RETENTION_ROWS=24
RETENTION_SAMPLE_BEFORE_LIMIT=1
RETENTION_OBJECTIVE=nll
RETENTION_LOSS_WEIGHT=0.5
OPD_TASK_BALANCED_LOSS_SCALE=1
RETENTION_TASK_BALANCED_LOSS_SCALE=1
```

### P2 Capacity Ladder

| run | strategy | 是否主表 | 说明 |
|---|---|---|---|
| global 3 | `global-coefficient` | yes | 最易解释，应优先 |
| common+residual 4 | `global` | yes | 稳定偏移 symmetric prior |
| layer-wise 28x3 effective | `layer-band` + `configs/gated_grpo_layer28.yaml` | maybe | 只有 proxy 强才送 eval |
| module/global-parameter | `global-parameter` | diagnostic | 参数多，容易 overfit；可引用既有 ABC/D/E/F/G |

## 当前可复用数据

| 数据 | 路径 | 用途 |
|---|---|---|
| paper96 balanced | `/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl` | 主实验，避免换数据导致对照混乱 |
| eval-targeted CURE aligned | `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl` | P2 Tool/Code repair |
| old expert rollouts | `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/` | same-prompt OPD |
| code aug expert rollouts | `/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/` | Code OPD coverage |

## 结果判据

主表优先排序：

1. Formal eval6 average and worst-task drop.
2. Tool live / Memory F1 / Code BoN 分项不崩。
3. Proxy trajectory 与 formal eval 不冲突。
4. Coefficient trajectory 支持方法解释。

不作为主 claim 的情况：

- 只提升 proxy，不提升 formal eval。
- 只提升 Tool，Memory/Code 大幅下降。
- 依赖 code augmentation 但 formal CURE 不涨。
- module/layer 参数多但 held-out gap 变大。
