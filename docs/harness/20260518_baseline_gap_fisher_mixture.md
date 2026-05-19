# 2026-05-18 Baseline Gap: Fisher and Mixture Training

本文件记录 baseline 审计结论，防止把未复现方法误写为 completed。

## 结论

| baseline | 当前状态 | 是否可报告正式数值 | 下一步 |
|---|---|---:|---|
| Fisher-weighted merging | 本地无干净 Qwen 实现；无 Fisher diag 统计、无 merge script、无 checkpoint、无 Eval6 | 否 | 先实现并审计 diagonal Fisher pipeline |
| Mixture Training / full-parameter GRPO | 已补独立 baseline wrapper；8-prompt smoke 与 96-prompt candidate 均跑通，96-prompt 已导出 HF checkpoint 并完成 Eval6 | 是 | 已填入 `docs/evaluation/20260518_baselines_eval6.md` |
| WUDI / ExpertMerging | 本地有历史模型目录；本轮按策略不重跑 | 仅可索引目录，不能混入本轮新 baseline 表 | 若需要，单独核对同协议 eval6 |

## Fisher 审计

已检查：

- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/baselines/`
- `/mnt/cache/wuruixiao/users/lsc/ExpertMerging`
- `/mnt/cache/wuruixiao/users/lsc/era-2026/MergeBench/merging/merging_methods/fisher.py`
- 官方论文/代码参考：
  - `https://arxiv.org/abs/2111.09832`
  - `https://github.com/mmatena/model_merging`

现有 baseline 脚本只覆盖：

- `task_arithmetic`
- `ties`
- `dare_ta`
- `dare_ties`
- AdaMerging wrapper

未发现：

- per-expert diagonal Fisher 统计脚本；
- Fisher-weighted merge 脚本；
- Fisher baseline run wrapper；
- Fisher checkpoint；
- Fisher Eval6 结果。

补充审计：本机 MergeBench 有 Fisher 代码，但不能直接作为当前
ExpertGym/Qwen baseline 复现：

- 依赖 MergeBench `TaskLoader` / `SFTTrainer` / DeepSpeed patched backward，
  不是当前 OP-VEC prompt/reward harness。
- `fisher_utils.py` 默认只用 last-token hard/soft pseudo target 估计 Fisher，
  与当前 Tool/Memory/Code executable feedback 不对齐。
- 文件中存在硬编码 `sys.path`，需要清理后才能纳入 ExpertGym 脚本。
- 没有已完成的 Qwen three-expert Fisher statistics/checkpoint/Eval6。

因此 Fisher 仍标记为待实现，不向正式表格填入未经审计的数字。

可信最小实现应拆成三步：

1. `scripts/baselines/compute_qwen_fisher_diag.py`
   对每个 expert 在校准数据上计算 NLL backward，保存 mergeable parameters 的 diagonal Fisher。
2. `scripts/baselines/build_fisher_weighted_baseline.py`
   按 `theta = sum_i F_i * theta_i / (sum_i F_i + eps)` 合并，必要时加入 base prior / epsilon。
3. `scripts/baselines/run_qwen_fisher_baseline.sh`
   固定 base、Tool/Memory/Code expert、calibration、seed、输出目录。

注意：Fisher 需要可计算 NLL 的 target。若只用 prompt 没有 target completion，只能做 pseudo-label Fisher，不能直接作为严格 Fisher baseline。

当前决策：WUDI / ExpertMerging 本轮只索引本机目录，不重跑；Fisher 先不报数。
优先把已经具备干净复现链路的 TA / TIES / DARE / AdaMerging / Mixture
Training 跑完同口径 Eval6。

## Mixture / Full-Parameter GRPO 审计

可复用组件：

- VeRL vendored: `third_party/verl/`
- VeRL OP-VEC 数据/奖励适配：
  - `third_party/verl/verl/experimental/opvec/prepare_data.py`
  - `third_party/verl/verl/experimental/opvec/reward_fn.py`
- 原生 gate-only 主入口：
  - `skill/command/run_qbank_c033333_gate_strategy.sh`

已完成：

- 96-prompt one-step VeRL GRPO training；
- HF checkpoint export；
- Eval6 同口径记录，见 `docs/evaluation/20260518_baselines_eval6.md`。

当前入口：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/baselines/run_qwen_mixture_grpo_baseline.sh
```

隔离原则：

- 不写入 `/tmp/shared-storage/OnPolicy/runs/gated_grpo/`。
- 使用 `/tmp/shared-storage/ExpertGym/baselines/mixture_grpo/<run_name>/`。
- 不使用 gate actor、不加载 gate checkpoint、不启用 OPD/retention/frontier routing。
- 只复用 prompt 数据与官方 reward，训练 actor 全参数或明确标注 LoRA/FSDP。

5 小时内优先 smoke：

```text
model_init: TA-1/3 或 base
prompts: 24-48
rollout.n: 2
train_steps: 1-3
max_response_length: 512/1024
output: /tmp/shared-storage/ExpertGym/baselines/mixture_grpo/<run_name>/
```

默认起点：

```text
/tmp/shared-storage/ExpertGym/baselines/qwen7b/static_merges/task_arithmetic_c0p3333333333333333_k0p2_d0p8_seed20260518
```

该入口显式不设置 `OPVEC_ENABLE_VERL_PATCH`，也不设置
`actor_rollout_ref.model.external_lib`。因此它是 full actor 参数训练 baseline，
不是 gated-GRPO 变体。

目标不是最终分数，而是验证：

- rollout 能生成；
- reward 能按 Tool/Memory/Code 路由；
- full actor backward 能跑；
- checkpoint 能保存；

2026-05-18 已验证/失败点：

- 双卡 smoke：与 P1 OPD-only 训练抢占 GPU 6/7，vLLM KV cache 初始化失败。
- 单卡 smoke：数据转换、TA-1/3 checkpoint 加载、FSDP 初始化成功，但未进入
  rollout/checkpoint 阶段；不作为有效 baseline。
- 双卡 + actor param offload：vLLM 可以启动，rollout 可以进入，但混合任务
  `reward_extra_info` 缺键触发 `KeyError: acc`。已在 VeRL agent loop 中做
  最小兼容 patch：用 union keys，缺失值填 `None`，不改变 scalar reward。
- 双卡 + extra-info patch：rollout 后失败于 `25 % 2 != 0`。原因是 MemAgent
  multi-turn 轨迹把 8 prompt 展开为 25 条训练序列，VeRL 的 DP split 要求
  old-logprob batch 能被 worker 数整除。
- 双卡 + `trainer.balance_batch=False`：仍在 old-logprob dispatch 失败，
  因为 worker-group chunk 仍要求 `25 % 2 == 0`。
- 单卡 DP=1 + actor param offload + low vLLM memory + `PPO_MINI_BATCH_SIZE=1`
  已跑通 8-prompt smoke，并保存 checkpoint：
  `/tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l8_n1_step1_gpu4_1dp_minib1_20260518/checkpoints/global_step_1/actor`。
- 沿用成功 smoke recipe 扩展到 96 prompts 后已完成一次 full-parameter
  GRPO step，并导出 HF checkpoint：
  `/tmp/shared-storage/ExpertGym/baselines/mixture_grpo/mixture_grpo_ta13_evaltarget_l96_n1_step1_gpu4_1dp_minib1_20260518/hf_merged_global_step_1`。
- 已完成同口径 Eval6：
  Tool mean 0.7823 / live 0.6771，Memory EM/F1 0.5313/0.6643，
  Code Acc/TP/BoN 0.3384/0.4716/0.3782。
- 结论：Mixture/full-GRPO 目前有可执行 baseline 路线，并已进入正式
  baseline 表。不要在 baseline 里临时 padding/drop trajectory，除非单独审计。

## 当前可正式报告

- TA-1/3、TA-0.75、init1：`docs/evaluation/20260517_p0_static_baselines_eval6.md`
- TIES、DARE-TA、DARE-TIES、AdaMerging task-wise len1024、Mixture/full-GRPO：
  `docs/evaluation/20260518_baselines_eval6.md`

## 推荐优先级

GPU 释放后优先补 P1 candidate 的完整评测与 K=8 状态分布，而不是 Fisher。

理由：Mixture baseline 已有可报告数字；Fisher 还缺统计 pipeline 和 target
completion，贸然跑会得到不可审计数字。
