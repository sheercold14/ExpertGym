# 2026-05-18 expD R1-Scaled OPD+Retention

## 目的

对比两种 layer-band 粒度下，4-expert R1-scaled task vector 是否能在不使用 GRPO frontier 的情况下，仅靠动态 OPD + all-success retention 推动三任务能力。

该实验不是 Code P0 bound grid；它使用三任务 paper96 calibration，从 `tool/memory/code=1/3, reasoning=0` 初始化，目的是检验 R1 异质 expert 是否能被 ExpertGym 的结构化 gate 学起来。

## 启动脚本

```bash
bash scripts/train/run_4expert_r1scaled.sh
```

原始两个后台 run 因为 loop 旧逻辑会对 OPD-only/NLL retention 无条件补 old-logprob，iter1 update 超过 15 分钟仍未落 summary，已停止保留诊断产物：

```text
expD_r1scaled_3band_20260518
expD_r1scaled_layer28_20260518
```

修复 `scripts/train/opvec_gated_grpo_bake_vllm_loop.py` 后，已用新 run name 重启：

```text
expD_r1scaled_3band_noold_20260518
expD_r1scaled_layer28_noold_20260518
```

差异仅为：OPD-only + NLL retention 时不再向 update 阶段传 `--fill-missing-old-logprob`。GRPO/PPO 或 KL retention 路径仍会保留该参数。

## 共同配置

```text
mode: /tmp/shared-storage/OnPolicy/modes/opvec4_r1scaled_20260518/mode_manifest.json
calibration: /tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
num_prompts: 96
samples_per_prompt: 4
num_iters: 10
tasks: tool,memory,code
init: tool=1/3, memory=1/3, code=1/3, reasoning=0
loss: OPD 1.0 + retention NLL, PPO/GRPO 0.0
optimizer: SGD, momentum=0.2
frontier quotas: tool=32, memory=32, code=32
task weights: tool=0.5, memory=2.0, code=1.5
dynamic OPD current max success: 0
dynamic OPD positives: expert success threshold 1.0
code max new tokens: 4096
memory update/final max new tokens: 2048/2048
```

Expert rollout pool:

```text
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/tool_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/code_expert_paper96_s2_seed20260514.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_deepseek_r1_distill_qwen7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_rl_memoryagent7b_s8_seed20260516.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/20260516_code_opd_aug/code_expert_reasonflux_coder7b_s8_seed20260517.jsonl
```

## 变体

| run | config | GPU | granularity | lr | init gate |
|---|---|---|---|---:|---|
| `expD_r1scaled_3band_noold_20260518` | `configs/gated_grpo_4expert_r1scaled.yaml` | 2,3 | 3 layer bands | 0.25 | `/tmp/shared-storage/OnPolicy/data/calibration/20260518_r1scaled_init_gates/init_layer_band_3band_tmc033_r0.json` |
| `expD_r1scaled_layer28_noold_20260518` | `configs/gated_grpo_4expert_r1scaled_layer28.yaml` | 4,5 | 28 layers | 0.10 | `/tmp/shared-storage/OnPolicy/data/calibration/20260518_r1scaled_init_gates/init_layer_band_28layer_tmc033_r0.json` |

## 监控判据

继续运行的条件：

1. proxy overall 或 memory/code reward 至少有一个稳定上升；
2. all-fail 数下降或 OPD recoverable 样本减少；
3. reasoning coefficient 没有大幅污染 tool/memory/code；
4. tool reward 不明显坍塌。

早停条件：

```text
连续 2-3 iter overall 不升且 memory/code 不升；
tool reward 明显下跌；
reasoning 系数快速上升但 proxy 没收益；
GPU 资源需要让给更高优先级 formal eval。
```
