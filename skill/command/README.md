---
title: OP-VEC Gated-GRPO 启动命令
project: ExpertGym
date: 2026-05-12
tags:
  - gated-grpo
  - opvec
  - command
  - training
---

# OP-VEC Gated-GRPO 启动命令

## 0. 当前清理后的主干

当前独立仓库只保留 gated_grpo 训练主干和必要文档：

```text
configs/gated_grpo.yaml                  # native loop 主配置
opvec/                                   # gate、delta、reward、logprob、GRPO 工具
scripts/train/opvec_gated_grpo_loop.py   # 主训练入口
scripts/train/opvec_collect_hf_rollouts.py
scripts/train/opvec_update_gates_from_rollouts.py
scripts/data/build_source_reward_seed_manifest.py
scripts/modes/build_opvec4_modes.py
scripts/modes/build_zero_gate_checkpoint.py
scripts/eval/opvec_bake_checkpoint.py
scripts/frameworks/                      # 可选 VeRL adapter
recipes/verl/                            # 可选 VeRL template
tests/                                   # gated_grpo 相关最小测试
```

已清理掉的内容包括旧 ExpertMerging 多模态代码、Qwen/InternVL 合并脚本、历史 plan、旧 evaluation harness、analysis 脚本和 inference 产物。

## 1. 环境变量

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

export PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
export ROOT=/tmp/shared-storage/OnPolicy
export ONPOLICY_STORAGE_ROOT=$ROOT

export MODE=$ROOT/modes/opvec4/mode_manifest.json
export SEED=$ROOT/data/source_reward/source_reward_t80_m80_c80_seed20260508.jsonl
```

## 1.1 Sequence vs Token Loss Smoke

先用同一份 10-prompt rollout 对照 legacy sequence loss 和 token-level loss：

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
GPU_LIST=0,1,2,3 \
NUM_PROMPTS=10 \
SAMPLES_PER_PROMPT=4 \
UPDATE_BATCH_SIZE=4 \
skill/command/run_smoke_sequence_vs_token.sh
```

只检查命令拼接：

```bash
DRY_RUN=1 skill/command/run_smoke_sequence_vs_token.sh
```

如果使用多卡加载模型，后续命令里再加：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 2. 数据准备

Gated-GRPO 使用 prompt-only calibration manifest。训练时当前 gated policy 自己采样多条回答，`RewardRouter` 给每条回答打 verifier reward，再用同 prompt 组内 reward 差异做 GRPO。

### 2.1 官方 reward 对齐的 routed correct pool

如果要优先复现专家训练流程，建议先用 routed correct pool 构建 manifest。它会：

```text
Tool:   使用 ToolRL rlla.py 风格 reward
Memory: 按 MemAgent recurrent trajectory 组织 chunk rollout，final answer 用 hotpotqa.py boxed exact reward
Code:   使用 CURE / CodeContests source-test pass rate，GRPO 内部做组内归一化
```

```bash
export SEED=$ROOT/data/source_reward/routed1_correct_official_seed20260510.jsonl

$PY scripts/data/build_routed_correct_seed_manifest.py \
  --input-root /mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1 \
  --output $SEED
```

注意：Memory 在这个 manifest 里是一条 question 级 trajectory 记录，不再是单个 final-answer prompt。

### 2.2 原始 source-reward prompt manifest

构建 source-reward prompt manifest：

```bash
mkdir -p $ROOT/data/source_reward

$PY scripts/data/build_source_reward_seed_manifest.py \
  --output $SEED \
  --tool-limit 80 \
  --memory-final-limit 80 \
  --code-limit 80 \
  --seed 20260508
```

输入来源在脚本默认值里：

```text
Tool:   /tmp/shared-storage/OnPolicy/external_repos/ToolRL/dataset/rlla_4k/train.parquet
Memory: /mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets/Memory.json
Code:   /mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json
```

产物是一行一个 prompt：

```text
prompt_id / task / prompt / messages / reference / verifier / tags
```

## 3. Mode manifest 准备

`mode_manifest.json` 记录 base model、专家 delta 文件和可合并参数列表。没有它就不能安装 gated linears。

构建 OP-VEC delta：

```bash
$PY scripts/modes/build_opvec4_modes.py \
  --config configs/gated_grpo.yaml
```

默认输出：

```text
$ROOT/modes/opvec4/mode_manifest.json
```

注意：

- 真实训练必须用包含 `basis_entries` 和 delta 文件的 real manifest。
- dry-run manifest 只能看结构，不能训练。
- 全量 7B delta 会占较大磁盘和内存，第一次构建较慢。

## 4. 命令级 sanity check

不加载模型、不启动训练，只检查 loop 拼出来的 collect/update 命令：

```bash
$PY scripts/train/opvec_gated_grpo_loop.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest $MODE \
  --seed-manifest $SEED \
  --run-dir $ROOT/runs/gated_grpo/dryrun \
  --num-iters 1 \
  --num-prompts 2 \
  --samples-per-prompt 2 \
  --max-gated-modules 1 \
  --dry-run
```

## 5. 小规模 smoke 训练

只挂 1 个 gated module，确认数据、reward、old_logprob、GRPO update 全链路能跑：

```bash
CUDA_VISIBLE_DEVICES=0 \
$PY scripts/train/opvec_gated_grpo_loop.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest $MODE \
  --seed-manifest $SEED \
  --run-dir $ROOT/runs/gated_grpo/smoke_global_16x4 \
  --run-id smoke-global \
  --num-iters 1 \
  --num-prompts 16 \
  --samples-per-prompt 4 \
  --gate-parameterization global \
  --max-gated-modules 1 \
  --max-new-tokens 512 \
  --max-prompt-tokens 2048 \
  --max-logprob-tokens 3072 \
  --lr 0.01 \
  --prior-loss-weight 0.02 \
  --max-coefficient-delta-from-init 0.20 \
  --frontier-task-quota tool=8 \
  --frontier-task-quota code=8 \
  --frontier-task-quota memory=4 \
  --task-weight tool=1.5 \
  --task-weight code=1.5 \
  --task-weight memory=0.5
```

检查输出：

```text
$ROOT/runs/gated_grpo/smoke_global_16x4/iter_001/rollouts.jsonl
$ROOT/runs/gated_grpo/smoke_global_16x4/iter_001/rollouts.summary.json
$ROOT/runs/gated_grpo/smoke_global_16x4/iter_001/gate_updates.jsonl
$ROOT/runs/gated_grpo/smoke_global_16x4/iter_001/gate_updates.summary.json
$ROOT/runs/gated_grpo/smoke_global_16x4/iter_001/gate_updates.gates.json
$ROOT/runs/gated_grpo/smoke_global_16x4/gated_grpo_loop_manifest.json
```

关键字段：

```text
kept_frontier_rows > 0
updates > 0
gate_grad_nonzero = true
final_gates 发生变化
```

## 6. 当前推荐复现顺序

当前不要再用随机大 batch 直接训。先固定有梯度信号的 balanced calibration，再比较 gate 是否真的提升 heldout。

固定 calibration：

```text
$ROOT/data/calibration/frontier_balanced_7each_seed20260511.jsonl
```

### 6.1 方案一耗时 benchmark

方案一是每轮 `bake -> vLLM rollout -> HF update`：

```bash
bash skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
```

当前 48 prompt x 4 samples 实测：

```text
bake:    52.8s
rollout: 396.7s wall, 373.5s internal generation
update:  ~145s after CPU delta-load fix
total:   ~9.9-11min / iteration
```

推荐显式打开下一轮 objective 开关：

```bash
GPU_LIST=0,1,2,3,4,5 \
MAX_MEMORY_PER_GPU=55GiB \
CPU_MAX_MEMORY=200GiB \
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
LOSS_GRANULARITY=token \
UPDATE_BATCH_SIZE=4 \
STORE_TOKEN_LOGPROBS=auto \
RUN_NAME=plan1_tasknorm_lenavg_one_iter \
bash skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
```

2026-05-11 修复了 update 阶段的 delta 安装显存峰值：`device_map=auto` 时 delta 先 CPU load，再注册到目标 GPU buffer，避免 GPU 上临时双份 delta。

2026-05-12 vLLM 路径支持在 rollout 阶段直接保存 sampled token 的 `old_logprobs`。`LOSS_GRANULARITY=token` 时 launcher 默认打开 `--store-token-logprobs`，update 阶段只需重算 current logprob，不再为完整样本重复 old-policy forward。`MAX_LOGPROB_TOKENS` 默认跟随 `MAX_MODEL_LEN`，避免 old vLLM context 和 current HF scoring context 不一致。

注意：vLLM 路径生成快，但 gate update 仍在 HF/torch 里做；588 个 gate 暂时不能直接在 vLLM 里动态更新。

### 6.2 固定 calibration 的 global baseline

```bash
bash skill/command/run_fixed_frontier7_global.sh
```

从上一轮继续：

```bash
INIT_GATE_CHECKPOINT=$ROOT/runs/gated_grpo/fixed_frontier7_global_e3_20260511/gate_updates.gates.json \
RUN_NAME=fixed_frontier7_global_continue \
MAX_STEPS=6 \
bash skill/command/run_fixed_frontier7_global.sh
```

当前可用 checkpoint：

```text
$ROOT/checkpoints/fixed_frontier7_global_continue_e3b_20260511
```

当前 global 系数：

```text
tool   ~= 0.85143
memory ~= 0.75758
code   ~= 0.73501
```

### 6.3 588 系数细粒度版本

只有在 global 系数没有掉到 0.5 以下时才跑这一版：

```bash
bash skill/command/run_fixed_frontier7_global_parameter.sh
```

默认从 `fixed_frontier7_global_continue_e3b_20260511` 初始化，训练 `global-parameter`：

```text
parameter_coefficients: 588
run_dir:    $ROOT/runs/gated_grpo/fixed_frontier7_global_parameter_e3_from_e3b_20260511
checkpoint: $ROOT/checkpoints/fixed_frontier7_global_parameter_e3_from_e3b_20260511
```

当前 588 统计：

```text
tool   mean=0.87239, min=0.81473, max=0.91473
memory mean=0.74063, min=0.69328, max=0.79328
code   mean=0.72669, min=0.68138, max=0.78138
```

但 48x4 sanity eval 里 588 版本相对 0.75 初始点退化，不能作为最终正结果；完整评测前不要继续盲目加步数。

### 6.4 完整评测入口

当前 cleaned worktree 不恢复旧 `skill/Evaluation_all` 目录；统一从 `skill/command/run_full_eval_suite.sh` 调外部 harness：

```bash
# 先只跑 Tool/BFCL
RUN_TOOL=1 RUN_MEMORY=0 RUN_CODE=0 \
bash skill/command/run_full_eval_suite.sh \
  $ROOT/checkpoints/fixed_frontier7_global_continue_e3b_20260511 \
  fixed-frontier7-global-e3b

# 先只跑 Memory/HotpotQA 小集
RUN_TOOL=0 RUN_MEMORY=1 RUN_CODE=0 \
MEMORY_DATASETS="eval_50 eval_100" \
bash skill/command/run_full_eval_suite.sh \
  $ROOT/checkpoints/fixed_frontier7_global_continue_e3b_20260511 \
  fixed-frontier7-global-e3b

# 先只跑 Code/CURE
RUN_TOOL=0 RUN_MEMORY=0 RUN_CODE=1 \
CODE_GPU_GROUPS="[[0,1]]" \
bash skill/command/run_full_eval_suite.sh \
  $ROOT/checkpoints/fixed_frontier7_global_continue_e3b_20260511 \
  fixed-frontier7-global-e3b
```

结果总入口：

```text
$ROOT/eval/full_suite/<model_name>/<run_id>/
```

注意：完整评测会启动 vLLM/CURE 任务，先确认 GPU 空闲、端口无冲突，再分任务启动。

### 6.5 下一轮 objective 开关

当前 `7/7/7` 已经证明不泛化，下一轮不要只加训练步数。先打开两个保守开关，降低 task/长度导致的梯度不均衡：

```bash
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
RUN_NAME=fixed_frontier7_global_tasknorm_lenavg \
bash skill/command/run_fixed_frontier7_global.sh
```

含义：

```text
TASK_NORMALIZE_ADVANTAGES=1
  对每个 task 的 row-normalized GRPO advantage 再做 task-level mean-abs rescale，
  避免某个 task 的 frontier weight / reward variance 系统性放大。

LENGTH_NORMALIZE_POLICY_LOGPROB=1
  PPO ratio 和 KL 使用 response 平均 token logprob，而不是整段 logprob，
  避免 memory/code/tool 的输出长度差异直接变成梯度尺度差异。
```

这两个开关默认关闭，旧实验仍可严格复现。开启后必须重新跑 heldout，不能和旧 run 直接混作同一设置。

## 7. 参数化升级顺序

推荐顺序：

```text
global -> layer-band -> global-parameter
```

不要一开始就用 `global-parameter`，它会训练 196 * 3 个参数级系数，对 prompt 数量和 reward 质量要求更高。

### layer-band

```bash
--gate-parameterization layer-band
```

### global-parameter

```bash
--gate-parameterization global-parameter
```

使用 `global-parameter` 时建议：

```bash
--prior-loss-weight 0.02
--max-coefficient-delta-from-init 0.10
--num-prompts >= 300
```

## 8. 单独 bake

如果训练时没有传 `--bake-final`：

```bash
$PY scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest $MODE \
  --gate-checkpoint $ROOT/runs/gated_grpo/global_150_seed20260510/iter_002/gate_updates.gates.json \
  --output $ROOT/checkpoints/gated_grpo_global_150_seed20260510
```

## 9. 可选 VeRL 数据与 adapter

先准备 VeRL prompt 数据：

```bash
$PY scripts/frameworks/opvec_prepare_verl_data.py \
  --seed-manifest $SEED \
  --output $ROOT/data/verl_opvec_prompts.jsonl \
  --parquet $ROOT/data/verl_opvec_prompts.parquet \
  --tasks tool,code,memory \
  --limit 150
```

环境变量：

```bash
export OPVEC_ENABLE_VERL_PATCH=1
export OPVEC_CONFIG=$PWD/configs/gated_grpo.yaml
export OPVEC_MODE_MANIFEST=$MODE
export OPVEC_GATE_PARAMETERIZATION=global
export OPVEC_MAX_GATED_MODULES=0
export OPVEC_FREEZE_BASE=1
```

模板：

```text
recipes/verl/opvec_gated_grpo_template.yaml
configs/verl_gated_grpo_experimental.yaml
```

当前建议：先跑 native loop；VeRL adapter 只在需要框架吞吐时再接。

## 10. 最小测试

```bash
$PY -m unittest discover -s tests -p "test_*.py" -v
```

## 11. 读结果的标准

一轮 Gated-GRPO 合理信号应该满足：

```text
1. rollouts 里每个 prompt 有多条当前 policy 采样。
2. old_logprob 存在，且 max_logprob_tokens 与 update 一致。
3. 同 prompt reward 有方差；全成功/全失败组不会贡献有效 policy gradient。
4. update summary 中 gate_grad_nonzero=true。
5. gate 变化被 trust region 限制住，没有单步飘到边界。
6. 下一轮 rollout 的 kept frontiers 和 mean reward 不应系统性变差。
```

核心判断不是 loss 数值，而是 verifier reward、frontier 数量、gate 变化方向和 heldout 趋势是否一致。
