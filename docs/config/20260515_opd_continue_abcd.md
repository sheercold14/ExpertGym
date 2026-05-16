# 2026-05-15 OPD Continuation 与 GRPO+OPD+Retention

## 目的

验证两个问题：

1. A/B 的 OPD-only + retention 在 iter15 之后是否还能继续推动 gate，尤其是 memory gate。
2. C/D 的 PCGuard 组在后期加入 GRPO 后，能否利用 partial-success 样本，缓解 OPD-only 只看 all-fail/all-success 导致的 memory 动力不足。

## 公共配置

| 项 | 值 |
|---|---|
| repo | `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym` |
| 入口脚本 | `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` |
| config | `configs/gated_grpo.yaml` |
| mode manifest | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` |
| seed manifest | `/tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.prompts.jsonl` |
| prompts | `96` |
| samples per prompt | `4` |
| manifest order | enabled |
| start iteration | `16` |
| num iters | `5` |
| task weights | `tool=1.0, memory=1.0, code=1.0` |
| frontier quotas | `tool=32, memory=32, code=32` |
| optimizer | `sgd` |
| momentum | `0.2` |
| optimizer step scope | `epoch` |
| update batch size | `4` |
| loss granularity | `sequence` |
| batch loss reduction | `mean` |
| prior loss | `0.0` |
| max coeff delta | `1.0` |
| retention | enabled, `nll`, positive threshold `1.0`, scale target `0.5` |
| OPD | dynamic from current all-fail + expert positive |
| OPD positive threshold | `1.0` |
| OPD max positives / negatives | `1 / 2` |
| OPD per task | `32` |
| OPD length norm | enabled |
| retention length norm | enabled |
| OPD task-balanced scale | enabled |
| retention task-balanced scale | enabled |
| policy logprob length norm | enabled |
| advantage normalization | `centered` |

### Expert Rollout

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
```

### Rollout 配置

| 项 | 值 |
|---|---|
| rollout engine | vLLM, per-GPU shard |
| rollout shards | `auto` |
| tensor parallel | `1` |
| vLLM batch size | `32` |
| GPU memory utilization | `0.82` |
| max new tokens | `1024` |
| tool max new tokens | `512` |
| code max new tokens | `4096` |
| memory update max new tokens | `2048` |
| memory final max new tokens | `2048` |
| max prompt tokens | `8192` |
| max logprob tokens | `12288` |
| max model len | `12288` |
| temperature / top-p | `0.7 / 0.95` |
| dtype | `bfloat16` |
| gradient checkpointing | enabled |
| HF device map | `auto` |
| HF max memory | `0=70GiB, 1=70GiB, cpu=180GiB` inside each visible GPU group |

## 实验矩阵

| ID | run_dir | 起点 gate | optimizer state | GPU | gate 参数化 | GRPO | OPD | PCGrad | lr | 目的 |
|---|---|---|---|---|---|---:|---:|---|---:|---|
| A | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdA_gc_nopc_step012_continue5_20260515` | `opdA_gc_nopc_step012_20260515/iter_015/gate_updates.gates.json` | 继承 A iter15 | `0,1` | `global-coefficient` | `0.0` | `1.0` | off | `0.1626` | 观察 OPD-only 后续是否还能涨 |
| B | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdB_gp_nopc_step012_continue5_20260515` | `opdB_gp_nopc_step012_20260515/iter_015/gate_updates.gates.json` | 继承 B iter15 | `6,7` | `global-parameter` | `0.0` | `1.0` | off | `0.1876` | 观察 global-parameter 后续 memory 是否继续涨 |
| C | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdC_gc_pcguard_grpo_opd_ret_continue5_20260515` | `opdC_gc_pcguard_step012_20260515/iter_015/gate_updates.gates.json` | 重置 | `2,3` | `global-coefficient` | `1.0` | `1.0` | on: tool/memory/code | `0.2242` | 给 PCGuard 组加入 GRPO partial-success 信号 |
| D | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdD_gp_pcguard_grpo_opd_ret_continue5_20260515` | `opdD_gp_pcguard_step012_20260515/iter_015/gate_updates.gates.json` | 重置 | `4,5` | `global-parameter` | `1.0` | `1.0` | on: tool/memory/code | `0.1483` | global-parameter + PCGuard + GRPO 对照 |

说明：

- A/B 继承 optimizer state，因为它们是 OPD-only 延续实验。
- C/D 只继承 gate，不继承 optimizer state，因为目标函数从 OPD-only 改成 GRPO+OPD+retention，沿用旧 momentum 会污染判断。
- C/D 的 GRPO 和 OPD loss weight 都设为 `1.0`，用于验证“同量级目标”能否补充后期 partial-success 梯度。

## 活跃进程

| 用途 | tmux |
|---|---|
| A training | `train_opdA_continue5_20260515` |
| B training | `train_opdB_continue5_20260515` |
| C training | `train_opdC_grpo_opd_ret_continue5_20260515` |
| D training | `train_opdD_grpo_opd_ret_continue5_20260515` |
| monitor | `opvec_monitor_opd_abcd_continue_20260515` |

前端：

```text
http://127.0.0.1:8787
```

当前前端包含 8 条曲线：

| label | 内容 |
|---|---|
| `A_base15` | A 原始 OPD-only global-coefficient, iter1-15 |
| `B_base15` | B 原始 OPD-only global-parameter, iter1-15 |
| `C_base15` | C 原始 PCGuard global-coefficient, iter1-15 |
| `D_base15` | D 原始 PCGuard global-parameter, iter1-15 |
| `A_continue` | A 从 iter15 继续 OPD-only + retention |
| `B_continue` | B 从 iter15 继续 OPD-only + retention |
| `C_grpo_opd_ret` | C 从 iter15 继续 GRPO+OPD+retention+PCGrad |
| `D_grpo_opd_ret` | D 从 iter15 继续 GRPO+OPD+retention+PCGrad |
 
注意：`*_base15` 和 `*_continue` 在前端是两条 run 曲线，不是物理合并后的单条曲线；continue run 的 iteration 编号从 `16` 开始，所以横轴仍可连着读。

## 结果与日志路径

每个 run 下固定查看：

```text
<run_dir>/train.log
<run_dir>/gated_grpo_bake_vllm_loop_manifest.json
<run_dir>/iter_*/rollouts.summary.json
<run_dir>/iter_*/opd_distill_from_allfail.summary.json
<run_dir>/iter_*/gate_updates.summary.json
<run_dir>/iter_*/gate_updates.gates.json
<run_dir>/iter_*/baked_policy
```

实验解释与评测汇总：

```text
/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opd_only_pcguard_abcd_20260515.md
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/aggregate_report_zh.md
```

## 已安排正式评测

用户指定：等待 B_continue 训练结束后，正式评测 `iter_019` checkpoint。

| 项 | 值 |
|---|---|
| model name | `expertgym-opdB-gp-nopc-step012-continue-i19` |
| model path | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/opdB_gp_nopc_step012_continue5_20260515/iter_019/baked_policy` |
| eval runner | `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_expertgym_opdB_continue_i19_20260515.py` |
| eval work root | `/tmp/shared-storage/AgentMerging_plan/evaluation_workdirs/eval6-20260502-125748-expertgym-opdB-continue-i19-20260515` |
| eval GPU | `6` |
| eval port | `8096` |
| watcher tmux | `eval_expertgym_opdB_continue_i19_20260515` |
| trigger | `train_opdB_continue5_20260515` tmux session exits, then eval starts |
| append target | `aggregate_report_zh.md` / `aggregate_summary.json` via `append_eval6_models.py` |

注意：这里的 `iter_019/baked_policy` 是 iter19 rollout 使用的 policy checkpoint，符合前面 eval step012 时对 `iter_015/baked_policy` 的口径；它不是 `iter_019/gate_updates.gates.json` 再额外 bake 后的模型。

## 已完成基线评测

下面是继续训练之前，A/B/C/D 原 step012 checkpoint 的完整 eval6 结果：

| 实验 | Tool mean | Memory F1 | Code avg Acc | Code avg BoN | 备注 |
|---|---:|---:|---:|---:|---|
| A `opdA_gc_nopc_step012_i15` | `0.7835` | `0.6924` | `0.3570` | `0.4105` | Code 最好，整体更均衡 |
| B `opdB_gp_nopc_step012_i15` | `0.7835` | `0.7171` | `0.3360` | `0.3636` | Memory 最好 |
| C `opdC_gc_pcguard_step012_i14` | `0.7927` | `0.6504` | `0.3511` | `0.4017` | Tool/Code 尚可，Memory 明显低 |
| D `opdD_gp_pcguard_step012_i15` | `0.7915` | `0.6661` | `0.3350` | `0.3880` | 均衡但未超过 A/B |

## 当前中间状态

记录时间：2026-05-15 21:33 CST。

| 实验 | 已有 summary | 最新轮 | ppo | opd | loss normalizer | dynamic OPD rows | retention rows | raw frontier rows |
|---|---|---|---:|---:|---:|---|---:|---|
| A | iter16, iter17 | iter17 | `0.0` | `1.0` | `85` | `code=3, memory=1` | `41` | `code=24, memory=12, tool=4` |
| B | iter16, iter17 | iter17 | `0.0` | `1.0` | `85` | `code=1, memory=1` | `46` | `code=26, memory=8, tool=3` |
| C | iter16 | iter16 | `1.0` | `1.0` | `89` | `code=3, memory=7, tool=1` | `27` | `code=27, memory=18, tool=6` |
| D | iter16 | iter16 | `1.0` | `1.0` | `88` | `code=3, memory=7, tool=1` | `29` | `code=21, memory=21, tool=6` |

初步观察：

- A/B 后期 dynamic OPD 的 memory rows 已经非常少，继续增长动力可能不足。
- C/D 加入 GRPO 后，理论上会让 partial-success 样本重新进入梯度路径；需要观察 iter17-20 的 memory reward 与 gate 是否改善。
- C/D 继续保留 PCGrad，若 memory 仍推不动，需要检查 PCGrad conflict count、pre/post cosine 和 memory 梯度是否被投影削弱。

## 复现模板

C/D 这次不是通过 `.sh` 启动，而是直接在 tmux 中传参。未来如需复现，建议把下列模板整理成正式脚本。

```bash
export PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

CUDA_VISIBLE_DEVICES=<GPU_LIST> $PY scripts/train/opvec_gated_grpo_bake_vllm_loop.py \
  --config configs/gated_grpo.yaml \
  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
  --seed-manifest /tmp/shared-storage/OnPolicy/data/question_bank/ta_avgvec_c033333_hotpotqa_v1/calibration/calib100_seed20260511.prompts.jsonl \
  --run-dir <RUN_DIR> \
  --run-id <RUN_ID> \
  --num-iters 5 \
  --start-iteration 16 \
  --num-prompts 96 \
  --samples-per-prompt 4 \
  --use-manifest-order \
  --rollout-shards auto \
  --rollout-gpus <GPU_LIST> \
  --gate-parameterization <global-coefficient|global-parameter> \
  --init-gate-checkpoint <ITER15_GATE_JSON> \
  --ppo-loss-weight <0.0_or_1.0> \
  --opd-loss-weight 1.0 \
  --use-retention \
  --retention-objective nll \
  --retention-scale-target 0.5 \
  --opd-length-normalize-logprob \
  --retention-length-normalize-logprob \
  --opd-task-balanced-loss-scale \
  --retention-task-balanced-loss-scale \
  --length-normalize-policy-logprob \
  --optimizer sgd \
  --sgd-momentum 0.2 \
  --lr <LR> \
  --optimizer-step-scope epoch \
  --loss-granularity sequence \
  --update-batch-size 4 \
  --frontier-order task-interleaved \
  --frontier-task-quota tool=32 \
  --frontier-task-quota memory=32 \
  --frontier-task-quota code=32 \
  --dynamic-opd-expert-rollout /tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl \
  --dynamic-opd-expert-rollout /tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl \
  --dynamic-opd-expert-rollout /tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
```

PCGrad 组追加：

```bash
--pcgrad-gate-gradients --pcgrad-task tool --pcgrad-task memory --pcgrad-task code
```

A/B OPD-only continuation 追加 optimizer state：

```bash
--persist-optimizer-state --optimizer-state-checkpoint <ITER15_OPTIMIZER_PT>
```

C/D GRPO+OPD continuation 不传 `--optimizer-state-checkpoint`，只保留：

```bash
--persist-optimizer-state
```
