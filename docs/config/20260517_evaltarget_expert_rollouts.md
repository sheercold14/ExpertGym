# 2026-05-17 eval-targeted96 expert rollout 配置

## 目的

为 `eval_targeted96_cure_aligned_20260517` 生成独立的 same-prompt expert trajectories，解决 A1 的核心缺陷：新 calibration 中 synthetic Tool / CURE-aligned Code prompt 与旧 paper96 expert rollout 不完全重合，dynamic OPD 对新样本缺少 expert-positive 信号。

这些 rollout 只作为 OPD positive pool，不覆盖 reward、不混入 paper96 旧文件，便于审计和复现。

## 数据与输出

| 项 | 值 |
|---|---|
| prompt manifest | `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl` |
| output dir | `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts` |
| launcher | `skill/command/run_20260517_evaltarget_expert_rollouts.sh` |
| samples | tool/memory `4`；code/deepseek `8` |
| decoding | `temperature=0.7`, `top_p=0.95` |
| code max new tokens | `10000` for code/deepseek, to better align CURE-style long solutions |

## 并行启动

当前 GPU 调度目标是填满空卡，同时不打断正在 CPU-test 阶段的 CURE 评测：

```bash
tmux new -d -s evaltarget_exp_tool_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=0 POLICY=tool SAMPLES_PER_PROMPT=4 bash skill/command/run_20260517_evaltarget_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/tool.log'

tmux new -d -s evaltarget_exp_memory_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=1 POLICY=memory SAMPLES_PER_PROMPT=4 bash skill/command/run_20260517_evaltarget_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/memory.log'

tmux new -d -s evaltarget_exp_code_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=2 POLICY=code SAMPLES_PER_PROMPT=8 CODE_MAX_NEW_TOKENS=10000 MAX_MODEL_LEN=24576 bash skill/command/run_20260517_evaltarget_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_reasonflux.log'

tmux new -d -s evaltarget_exp_deepseek_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=7 POLICY=deepseek SAMPLES_PER_PROMPT=8 CODE_MAX_NEW_TOKENS=10000 MAX_MODEL_LEN=24576 bash skill/command/run_20260517_evaltarget_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_deepseek.log'
```

实际调度时，tool 两个 seed 很快完成并释放 GPU，因此追加了 code expert 第三组 seed：

```bash
tmux new -d -s evaltarget_exp_code_s19_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=0 POLICY=code SAMPLES_PER_PROMPT=8 SEED_VALUE=20260519 CODE_MAX_NEW_TOKENS=10000 MAX_MODEL_LEN=24576 bash skill/command/run_20260517_evaltarget_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_reasonflux_seed20260519.log'

tmux new -d -s evaltarget_exp_deepseek_s19_20260517 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && GPU_LIST=6 POLICY=deepseek SAMPLES_PER_PROMPT=8 SEED_VALUE=20260519 CODE_MAX_NEW_TOKENS=10000 MAX_MODEL_LEN=24576 bash skill/command/run_20260517_evaltarget_expert_rollouts.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_deepseek_seed20260519.log'
```

运行中修正：`eval_p0_expH_code_20260517` 的 LiveBench CPU execution 即将切换到 LiveCodeBench，需要重新占用 GPU `4,6`。为保护正式评测，已中断两个非关键 extra seed：

```text
evaltarget_exp_code_s18_20260517      # GPU 4，中断前未写 coverage
evaltarget_exp_deepseek_s19_20260517  # GPU 6，中断前未写 coverage
```

保留继续运行：

```text
tool seed 20260517/20260518: done
memory seed 20260517/20260518: running on GPU 1/7
code ReasonFlux seed 20260517/20260519: running on GPU 2/0
code DeepSeek seed 20260517/20260518: running on GPU 3/5
```

## 完成后检查

每个 rollout 会写同名 `.summary.json` 与 `.coverage.json`。下一步只在 coverage 足够且 task 分布合理时，把这些文件加入下一版 `calib_bank_v1` 的 dynamic OPD pool。
