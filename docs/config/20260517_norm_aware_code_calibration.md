# 2026-05-17 Norm-aware Init 与 Code Calibration Blueprint

## 目的

这批产物服务于下一阶段论文方法验证：不把 scaling sweep 当作方法，而是用确定性 `norm-aware initialization` 加 on-policy gate learning 自动寻找 task-vector 组合。当前先完成三个前置步骤：

1. 统计 tool / memory / code task vector 的范数与层/模块分布。
2. 生成 norm-aware gate 初始化 checkpoint。
3. 从正式 CURE 评测 case-level 结果中抽取 Code generation / selection / partial-edge failure blueprint。

## Task Vector 诊断

命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

python scripts/analysis/analyze_task_vector_norms.py \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --diagnostics /tmp/shared-storage/OnPolicy/modes/opvec4/diagnostics.json \
  --output-json /tmp/shared-storage/OnPolicy/data/init_gates/norm_aware_20260517/task_vector_norm_diagnostics.json \
  --output-report docs/report/task_vector_norm_diagnostics_20260517.md \
  --output-gate-dir /tmp/shared-storage/OnPolicy/data/init_gates/norm_aware_20260517 \
  --gate-parameterization global-coefficient,global-parameter
```

核心结论：

| expert | total L2 | relative to code |
|---|---:|---:|
| tool | `1.254945` | `2.0232` |
| memory | `5.362092` | `8.6449` |
| code | `0.620264` | `1.0000` |

Code 不是 module 覆盖少；三专家都是 `196 = 28 layers * 7 modules`。问题是 code delta 幅度显著小，且 code energy 主要集中在中层 MLP：50% code energy 只需要 39/196 个 modules 覆盖。

详细报告：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/task_vector_norm_diagnostics_20260517.md
```

## Init Checkpoints

输出目录：

```text
/tmp/shared-storage/OnPolicy/data/init_gates/norm_aware_20260517/
```

文件：

| file | gate parameterization | 用途 |
|---|---|---|
| `all_ones.global-coefficient.json` | global-coefficient | 第一阶段推荐基线；三个专家都完整注入 |
| `all_ones.global-parameter.json` | global-parameter | 同上，global-parameter 版本 |
| `all1_sqrt_weak_compensation.global-coefficient.json` | global-coefficient | 第一阶段弱补偿版本；memory 保持 1，tool/code 做 sqrt 补偿 |
| `all1_sqrt_weak_compensation.global-parameter.json` | global-parameter | 同上，global-parameter 版本 |
| `all1_linear_weak_compensation.global-coefficient.json` | global-coefficient | 激进稳定性诊断；memory 保持 1，tool/code 线性补偿到同等 effective L2 |
| `all1_linear_weak_compensation.global-parameter.json` | global-parameter | 同上，global-parameter 版本 |
| `sum1_equal_effective_l2.global-coefficient.json` | global-coefficient | 保守版；三个系数和为 1，并让三专家 effective L2 相同 |
| `sum1_equal_effective_l2.global-parameter.json` | global-parameter | 同上，但兼容 common run 的 196 param residual 框架 |
| `baseline_mean_effective_l2.global-coefficient.json` | global-coefficient | 激进版；保持 `1/3` TA 的平均 perturbation 强度，显著放大 code |
| `baseline_mean_effective_l2.global-parameter.json` | global-parameter | 同上，global-parameter 版本 |

系数：

| init | tool | memory | code | 解释 |
|---|---:|---:|---:|---|
| `all_ones` | `1.0000` | `1.0000` | `1.0000` | 第一阶段保留三专家完整能力 |
| `all1_sqrt_weak_compensation` | `2.0668` | `1.0000` | `2.9402` | 不压 memory，只对弱 delta 做次线性补偿 |
| `all1_linear_weak_compensation` | `4.2728` | `1.0000` | `8.6449` | 强 stress test；容易过冲，只建议短 smoke |
| `sum1_equal_effective_l2` | `0.3070` | `0.0719` | `0.6211` | 系数和为 1；适合保守 ablation |
| `baseline_mean_effective_l2` | `0.6408` | `0.1500` | `1.2965` | 维持 `1/3` TA 的平均扰动量；更适合验证 code 能否被推出来 |

修正：第一阶段目标是先得到保留三专家能力的 strong merged model，所以不应把 memory 压到 `0.15` 或 `0.07`。`sum1_equal_effective_l2` 和 `baseline_mean_effective_l2` 只保留为诊断 ablation。主线建议：

1. 先用 `all_ones` 看三专家完整注入是否稳定；
2. 再用 `all1_sqrt_weak_compensation` 看 weak code/tool delta 的小幅补偿是否有收益；
3. `all1_linear_weak_compensation` 只做极小 smoke，不直接长训。

## CURE Code Calibration Blueprints

命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

python scripts/data/build_cure_code_calibration_pools.py \
  --cases /tmp/shared-storage/OnPolicy/analysis/eval_case_browser/cases.jsonl \
  --output-dir /tmp/shared-storage/OnPolicy/data/calibration/cure_code_blueprints_20260517 \
  --target-model expertgym-B-codeaug-opd-i18 \
  --strong-model best-ever-tame-cg-r1calib-global-v2 \
  --strong-model ta-c075 \
  --prompt-limit 96
```

输出目录：

```text
/tmp/shared-storage/OnPolicy/data/calibration/cure_code_blueprints_20260517/
```

计数：

| pool | rows | 作用 |
|---|---:|---|
| `code_generation_pool.jsonl` | `60` | 当前 B 没有正确 code sample，但强模型/TA 至少有 correct sample |
| `code_selection_pool.jsonl` | `68` | 当前 B 有 correct sample，但 CURE BoN 选错 |
| `code_partial_edge_pool.jsonl` | `237` | 当前 B 过部分 hidden tests，缺边界条件 |
| `code_calibration_blueprints.jsonl` | `96` | generation / selection / partial-edge 交错后的默认候选 |

注意：这些是 blueprint，不是最终训练数据。不能直接把官方 CURE / LiveBench / LiveCodeBench prompt、hidden tests 或官方生成结果放进训练。正确用法是按 `code_tags` / `failure_tags` 生成 fresh programming tasks、fresh tests 和 verified expert solutions。

## 下一步训练建议

第一组实验先做 2 个最小对照：

| 实验 | init | gate | data | loss | 目的 |
|---|---|---|---|---|---|
| NA-A | `all_ones.global-coefficient.json` | global-coefficient | 原 paper96 96 prompts | GRPO + OPD + retention | 验证三专家完整注入是否稳定 |
| NA-B | 同 NA-A | global-coefficient | paper96 + fresh CURE-style code generation/selection synthetic data | GRPO + OPD + retention | 验证 CURE-aligned Code calibration 是否提升正式 Code BoN |
| NA-C | `all1_sqrt_weak_compensation.global-coefficient.json` | global-coefficient | 同 NA-A | GRPO + OPD + retention | 验证弱 delta 次线性补偿是否比 all=1 更好 |

如果 NA-A code proxy 仍不涨，说明仅靠系数初始化不够，需要更强 code/reasoning delta 或 layer-band/mid-MLP gate。如果 NA-A proxy 涨但正式 Code 不涨，说明必须优先做 selection calibration。

## NA-A 启动模板

先 dry-run 检查命令：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

DRY_RUN=1 \
GPU_LIST=2,3 \
RUN_NAME=naA_normaware_baseline_mean_gc_20260517 \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/gated_grpo/naA_normaware_baseline_mean_gc_20260517 \
STRATEGY=global-coefficient \
INIT_GATE_CHECKPOINT=/tmp/shared-storage/OnPolicy/data/init_gates/norm_aware_20260517/all_ones.global-coefficient.json \
CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl \
NUM_PROMPTS=96 \
NUM_ITERS=20 \
SAMPLES_PER_PROMPT=4 \
PPO_LOSS_WEIGHT=1.0 \
OPD_LOSS_WEIGHT=1.0 \
USE_RETENTION=1 \
RETENTION_OBJECTIVE=nll \
RETENTION_LOSS_WEIGHT=0.05 \
OPTIMIZER=sgd \
SGD_MOMENTUM=0.2 \
LR=0.1876 \
OPTIMIZER_STEP_SCOPE=epoch \
LOSS_GRANULARITY=sequence \
STORE_TOKEN_LOGPROBS=0 \
TASK_NORMALIZE_ADVANTAGES=0 \
FRONTIER_ORDER=task-interleaved \
DYNAMIC_OPD_EXPERT_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl \
DYNAMIC_OPD_TASKS=tool,memory,code \
DYNAMIC_OPD_PER_TASK=32 \
bash skill/command/run_qbank_c033333_gate_strategy.sh
```

正式启动：

```bash
tmux new -d -s train_naA_normaware_gc_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && DRY_RUN=0 GPU_LIST=2,3 RUN_NAME=naA_all1_gc_20260517 RUN_DIR=/tmp/shared-storage/OnPolicy/runs/gated_grpo/naA_all1_gc_20260517 STRATEGY=global-coefficient INIT_GATE_CHECKPOINT=/tmp/shared-storage/OnPolicy/data/init_gates/norm_aware_20260517/all_ones.global-coefficient.json CALIBRATION=/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl NUM_PROMPTS=96 NUM_ITERS=20 SAMPLES_PER_PROMPT=4 PPO_LOSS_WEIGHT=1.0 OPD_LOSS_WEIGHT=1.0 USE_RETENTION=1 RETENTION_OBJECTIVE=nll RETENTION_LOSS_WEIGHT=0.05 OPTIMIZER=sgd SGD_MOMENTUM=0.2 LR=0.1876 OPTIMIZER_STEP_SCOPE=epoch LOSS_GRANULARITY=sequence STORE_TOKEN_LOGPROBS=0 TASK_NORMALIZE_ADVANTAGES=0 FRONTIER_ORDER=task-interleaved DYNAMIC_OPD_EXPERT_ROLLOUT=/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl,/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl DYNAMIC_OPD_TASKS=tool,memory,code DYNAMIC_OPD_PER_TASK=32 bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/gated_grpo/naA_all1_gc_20260517/train.log'
```

## Smoke 结果

时间：`2026-05-17 14:31 CST`

口径：只做 bake + vLLM rollout，不做 update；`9 prompts = tool/memory/code 各 3`，每题 `2 samples`。baked checkpoint 已删除，保留 `rollouts.jsonl` 和 `summary.json`。

| init | gates | run dir | Tool mean / parse | Memory mean | Code mean | 结论 |
|---|---|---|---:|---:|---:|---|
| `all_ones` | tool=1, memory=1, code=1 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/smoke_all1_gc_20260517` | `1.0 / 1.0` | `0.6667` | `0.5000` | 不崩；第一阶段合理基线 |
| `all1_sqrt_weak_compensation` | tool=2.067, memory=1, code=2.940 | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/smoke_sqrtcomp_gc_20260517` | `1.0 / 1.0` | `0.6667` | `0.1667` | 系统不崩，但 Code 小样本明显变差；不建议直接长训 |

判断：第一阶段应采用 `all_ones`，而不是压 memory 的 norm-aware equalization，也不是直接大幅补偿弱 delta。后续如果要补 code，应通过训练信号或更细粒度 code modes，而不是一开始把 code coefficient 放到 3 以上。
