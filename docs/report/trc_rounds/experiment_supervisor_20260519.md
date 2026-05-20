# 20260519 TRC Round 实验监督报告

更新时间：2026-05-19 22:38 CST

监督边界：本报告只读训练/评测产物，记录今晚多轮 TRC 实验的合理性与决策规则；不修改训练代码、不清理他人进程、不覆盖已有实验结果。

## 1. 当前已知基线与候选

### 1.1 训练产物路径

| 候选 | 训练 run | gate epoch | baked checkpoint |
|---|---|---:|---|
| anchor_i4 | `/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_anchor_20260519` | 4 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i4_20260519` |
| anchor_i8 | `/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_anchor_20260519` | 8 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i8_20260519` |
| dir_i8 | `/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_20260519` | 8 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_dir_i8_20260519` |

### 1.2 正式评测指标

来源：

- Tool/Memory 汇总：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260519_stage1_candidates_eval.md`
- Code 原始结果：
  - `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/trc_stage1_v3_anchor_i4_20260519/stage1_20260519/results`
  - `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/trc_stage1_v3_anchor_i8_20260519/stage1_20260519_rerun/results`
  - `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/trc_stage1_v3_dir_i8_20260519/stage1_20260519_dir_i8/results`

| 候选 | Tool mean | Tool live | Memory EM | Memory F1 | LiveBench Acc | LiveCodeBench Acc | Code mean Acc | Code BoN mean | 读法 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| anchor_i4 | 0.7788 | 0.6875 | 0.6348 | 0.7603 | 0.3633 | 0.2886 | 0.3260 | 0.3988 | Memory 过线，Tool 未达 0.79，Code 保守 |
| anchor_i8 | 0.7800 | 0.6875 | 0.6406 | 0.7594 | 0.3770 | 0.2857 | 0.3313 | 0.4134 | Code/BoN 最好，Memory 接近阈值，Tool 未达 0.79 |
| dir_i8 | 0.7981 | 0.7188 | 0.6445 | 0.7663 | 0.3594 | 0.2852 | 0.3223 | 0.4076 | Tool/Memory 最好，Code 未跟随 TRC loss 提升 |

阶段性结论：

- 如果只看 Tool/Memory，`dir_i8` 是当前最强 stage-1 候选，已经满足今晚快速筛选阈值 Tool mean >= 0.79、Memory F1 >= 0.76。
- 如果看 Code，`anchor_i8` 更稳：LiveBench Acc 和 Code mean Acc 最高；`dir_i8` 的 Code 没崩盘，但低于 anchor_i8。
- Code 的核心风险已经明确：hidden residual loss 下降、code gate 增大，并不单调对应 CURE/LiveBench 单测正确率。

### 1.3 Gate 与 loss 区间

来源：`epoch_*.gates.json` 与 `trc_metrics.jsonl`。

| 候选 | residual loss | total loss | tool gate | memory gate | code gate | task loss 读法 |
|---|---:|---:|---:|---:|---:|---|
| anchor_i4 | 0.5540 | 0.5939 | 1.0606 | 0.9199 | 1.0801 | 保守，memory 保留高 |
| anchor_i8 | 0.4958 | 0.5320 | 1.1068 | 0.8391 | 1.1609 | Code/Tool 被推高，Memory 刚好在安全边缘 |
| dir_i8 | 0.4423 | 0.4559 | 1.1799 | 0.7583 | 1.2416 | loss 最低，但 memory gate 偏低、code 过推风险已出现 |

细节：

- `dir_i8` task residual：tool 0.3056、memory 0.0202、code 1.0011。
- `anchor_i8` task residual：tool 0.3425、memory 0.0133、code 1.1317。
- `dir_i8` code residual 明显低于 anchor_i8，但 Code formal eval 没提升，说明 Code 需要 validation/评测闭环，不能只追 residual loss。

建议的 gate 安全区间：

| expert | 建议区间 | 理由 |
|---|---:|---|
| tool | 1.05 - 1.20 | Tool 在 dir_i8 提升明显；过高是否破坏 Code 还需验证 |
| memory | >= 0.82 优先，最低警戒 0.76 | dir_i8 memory gate 0.7583 仍能过线，但已是低边界；再压低风险高 |
| code | 1.10 - 1.18 优先，>1.22 警戒 | dir_i8 code 1.2416 未带来 Code 提升，说明过推无收益 |

## 2. 今晚 round 监督规则

### 2.1 每个 run 必看文件

训练：

- `/tmp/shared-storage/OnPolicy/runs/trc/<run_id>/trc_metrics.jsonl`
- `/tmp/shared-storage/OnPolicy/runs/trc/<run_id>/epoch_*.gates.json`
- `/tmp/shared-storage/OnPolicy/runs/trc/<run_id>/trc_run_manifest.json`
- `/tmp/shared-storage/OnPolicy/runs/trc/<run_id>/train.log`

评测：

- `/tmp/shared-storage/OnPolicy/eval/full_suite/<model_name>/<run_id>/logs`
- `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/<model_name>/<run_id>/results`
- `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/evaluation/20260519_stage1_candidates_eval.md`

### 2.2 训练中监控指标

每个 epoch 记录：

| 指标 | 正常趋势 | 风险信号 |
|---|---|---|
| `mean_residual_loss` | 下降，但不强求最低 | 继续下降但 gate 过推，说明 proxy 可能错位 |
| `task_loss.tool.residual_loss` | 下降或平稳 | tool loss 不降且 tool gate 极端漂移 |
| `task_loss.memory.residual_loss` | 不要求继续下降到极低 | memory gate 被压低到 <0.76，且 memory loss 上升 |
| `task_loss.code.residual_loss` | 下降即可 | code loss 下降但 code gate >1.22，优先怀疑过拟合 hidden proxy |
| `gate_means` | 三任务有结构化分化 | code/tool 同步上升、memory 持续下降 |
| layer gate std | tool 有层结构是好信号 | code/memory 只有全局平移，说明 layer 粒度没有被有效利用 |

### 2.3 停止/选点标准

不每 epoch bake 大模型。每个实验固定训练 10-12 epoch，训练后只 bake 1-2 个 gate。

优先选点：

1. `mean_residual_loss` 已下降，但最近两轮改善 <1.5%。
2. `memory gate >=0.82`；如果 Tool/Memory 方向特别强，最低允许参考 `dir_i8` 的 0.758，但必须先测 Memory。
3. `code gate` 优先 1.10-1.18；若 >1.22，除非 Code validation 明确上涨，否则不作为主候选。
4. `tool gate` 优先 1.05-1.20。
5. 如果 final epoch loss 最低但 gate 明显越界，选前一个更稳 epoch。

立即标记为“不建议继续”的方向：

- 复现 v2 MSE：它会惩罚非目标 expert residual，历史上 2 epoch 内 memory/tool 明显坍缩。
- 单纯追 code residual 最低：dir_i8 已证明不能保证 Code Acc。
- 把 memory gate 压到 0.70 以下：即使短期 loss 好，也不符合 Memory F1 >=0.76 的主线目标。
- 每轮都 bake/eval：训练速度优势会被评测拖垮，不适合今晚 4x4 迭代。

## 3. 评测门控规则

每个实验训练完后：

1. 根据 loss/gate 选 1-2 个 gate bake。
2. 先测 Tool + Memory。
3. 只有同时满足：

```text
Tool mean >= 0.79
Memory F1 >= 0.76
```

才跑 Code CURE。

边界情况：

- Tool mean 0.785-0.790 且 Memory/Code validation 很强：可作为备选测 Code，但不能占用主评测卡。
- Memory F1 0.755-0.760 且 Tool 明显强：可保留为 ablation，不作为主模型。
- Code CURE 特别慢，优先测满足门控的 top-2，不测明显不过线模型。

## 4. 当前运行状态

截至 2026-05-19 22:38 CST：

- `/tmp/shared-storage/OnPolicy/runs/trc` 下尚未出现今晚 round harness 的新增正式 run；最新仍是 `trc_layer_init1_v3_directional*`。
- `nvidia-smi` 显示 8 张卡仅 2 MiB 占用，无 compute app，GPU 处于可调度状态。
- 活跃相关 tmux：`stage1_dashboard_20260519`；未发现新的 TRC round 训练 tmux。
- 当前代码树有其他 agent 的未提交修改：
  - `scripts/trc/build_trc_calibration_v1.py`
  - `scripts/trc/train_trc_layer_gates.py`
  本监督报告没有触碰这些文件。

## 5. 下一轮实验建议

### Round 1：先验证 evaluation-aligned Code trajectory 是否有效

四路并行建议：

| 实验 | 数据 | loss 重点 | 预期判断 |
|---|---|---|---|
| R1-E0 | TRC96 原数据 | 当前 directional | 复现 anchor/dir 区间，作为今晚 sanity check |
| R1-E1 | Code 32-48 eval-aligned ReasonFlux verified | code block directional + projection | 看 Code trajectory 替换是否提高 Code proxy 且不过推 gate |
| R1-E2 | E1 + DeepSeek verified trajectory，不加 R1 delta | reasoning/code 轨迹补算法推理 | 判断 DeepSeek 轨迹是否补 LiveBench/LiveCodeBench |
| R1-E3 | E2 + task-specific span weight | tool_call / memory answer / code I/O token 权重 | 看任务能力点定向 loss 是否更均衡 |

训练设置建议：

```text
init tool/memory/code = 1.0
trainable gate = 28 layer × 3 expert
epochs = 10-12
hidden layers 起步 = 8,16,24,28
topk tokens = 128
不加 R1 delta
不每 epoch bake
```

### Round 2：根据 Round 1 结果收缩

如果 Tool/Memory 不过线：

- 增强 Tool span 权重或加入 tool_call attention alignment；
- memory gate floor 提高到接近 anchor_i8 安全区；
- 不优先动 Code 数据。

如果 Tool/Memory 过线但 Code 不涨：

- 不再追 code gate；
- 改 Code trajectory 组成：增加 LiveCodeBench hard、I/O/format-sensitive、partial-pass contrast；
- 加小型 Code validation 选点，不按 residual loss 选最低点。

如果 Code validation 涨但 Tool 掉：

- 单独限制 code gate 上界到 1.18；
- 使用 task-specific loss scale，而不是全局降学习率。

## 6. 当前最有希望 checkpoint

当前排序：

1. `trc_stage1_v3_dir_i8_20260519`：Tool/Memory 最强，适合作为 stage-1 main 候选；Code 弱于 anchor_i8，但没有崩。
2. `trc_stage1_v3_anchor_i8_20260519`：Code 最稳，适合作为 Code-sensitive fallback。
3. `trc_stage1_v3_anchor_i4_20260519`：保守但整体没有领先项，主要用于验证过推边界。

今晚实验目标不是继续复刻 dir_i8，而是在保持 dir_i8 的 Tool/Memory 区间时，把 Code 从 `0.3223` mean Acc / `0.4076` BoN mean 推向或超过 anchor_i8 的 `0.3313` mean Acc / `0.4134` BoN mean。

## 7. 可追溯命令

建议后续监督循环使用：

```bash
find /tmp/shared-storage/OnPolicy/runs/trc -maxdepth 2 -type f -name trc_metrics.jsonl -printf '%T@ %p\n' | sort -nr | head -30
tmux ls
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Code 原始结果核对：

```bash
for f in /tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/trc_stage1_v3_*_20260519/*/results/*.txt; do
  echo "$f"
  tail -40 "$f"
done
```

## 8. Round2 Code Recovery 四路审计

更新时间：2026-05-20 00:19 CST

审计对象：

```text
/tmp/shared-storage/OnPolicy/runs/trc/trc_r2a_cleanrf_wide_dir_20260519
/tmp/shared-storage/OnPolicy/runs/trc/trc_r2b_cleanrf_wide_resp_20260519
/tmp/shared-storage/OnPolicy/runs/trc/trc_r2c_cleanrf_wide_relmse_20260519
/tmp/shared-storage/OnPolicy/runs/trc/trc_r2d_cleanrf_wide_codeheavy_20260519
```

### 8.1 配置差异

四路都使用同一份 Round2 calibration：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_round2/e4_tool_aug_code_rf/trc96_expert_trajectories.jsonl
```

数据计数均为 96 条，三任务各 32 条。主要差异：

| run | residual objective | code span | code hidden layers | code topk | task multiplier | 正常预期 |
|---|---|---|---|---:|---|---|
| R2A `wide_dir` | directional | code-block auto | 4,8,12,16,20,24,28 | 256 | code 1.2 / tool 1.1 / memory 1.2 | 最干净对照，保留 code-block 结构 |
| R2B `wide_resp` | directional | full response | 4,8,12,16,20,24,28 | 256 | code 1.3 / tool 1.1 / memory 1.3 | 检查解释/推理轨迹是否补 Code |
| R2C `wide_relmse` | relative-mse | full response | 4,8,12,16,20,24,28 | 256 | code 1.1 / tool 1.1 / memory 1.2 | 检查 MSE-like 目标，但高风险 |
| R2D `codeheavy` | directional | full response | 4,8,12,16,20,24,28 | 384 | code 1.6 / tool 1.1 / memory 1.2 | 强推 Code |

### 8.2 运行状态

| run | 状态 | summary | 说明 |
|---|---:|---:|---|
| R2A | 18/18 done | yes | 正常完成 |
| R2B | 18/18 done | yes | 正常完成 |
| R2C | 4/18 ended | no | tmux/session 已消失，进程不在；该方向本身也已显示高风险 |
| R2D | 18/18 done | yes | 正常完成 |

额外观察：审计期间发现新增 `trc_r2e_cleanrf_wide_codelayers_20260520` 进程，占用 GPU 4/5；它不属于本节四路审计范围，未纳入结论。

### 8.3 Loss 曲线与 plateau

| run | e1 residual | e8 residual | e12 residual | e18 residual | 判断 |
|---|---:|---:|---:|---:|---|
| R2A | 0.6032 | 0.5165 | 0.4658 | 0.3930 | 单调下降，无 plateau，但后期明显以 gate 过推换 loss |
| R2B | 0.6350 | 0.5419 | 0.4840 | 0.3955 | 单调下降，无 plateau，趋势同 R2A |
| R2C | 7.9942 | n/a | n/a | n/a | 尺度异常大，4 epoch 后无 summary |
| R2D | 0.6313 | 0.5408 | 0.4848 | 0.3995 | 单调下降，无 plateau，code-heavy 没带来更优 loss 形态 |

关键判断：

- A/B/D 的 residual loss 都还在下降，不能按最低 loss 选 final epoch。
- e8 是最接近历史 `anchor_i8` 安全区的点；e12 接近历史 `dir_i8` 激进区；e18 已明显过推。
- R2C 的 relative-MSE 与多专家合并目标冲突仍然明显：loss 尺度高两个数量级，且 tool gate 被压低。

### 8.4 Gate collapse / over-push

| run | e8 gate code/memory/tool | e12 gate code/memory/tool | e18 gate code/memory/tool | 风险 |
|---|---|---|---|---|
| R2A | 1.1605 / 0.8395 / 1.1452 | 1.2411 / 0.7584 / 1.2056 | 1.3594 / 0.6374 / 1.2720 | e12 已到 dir_i8 激进区；e18 memory 明显 collapse |
| R2B | 1.1607 / 0.8395 / 1.1561 | 1.2422 / 0.7583 / 1.2232 | 1.3634 / 0.6361 / 1.2966 | e18 过推最强，tool/code 同涨、memory 被压 |
| R2C | e4 1.0800 / 0.9200 / 0.9316 | n/a | n/a | tool gate 反向下降，不建议评测 |
| R2D | 1.1606 / 0.8395 / 1.1512 | 1.2418 / 0.7584 / 1.2110 | 1.3628 / 0.6363 / 1.2701 | code-heavy 未形成有价值区分，仍压 memory |

结论：

- A/B/D 的 gate 方向高度一致，说明当前 loss 信号主要仍是“推 code/tool、压 memory”的全局方向；code-heavy multiplier 没能产生更细粒度的 code 能力选择。
- e18 不建议作为第一批评测点：memory gate `~0.636` 明显低于历史安全边界，极可能损害 Memory F1。
- e12 可作为激进备选，但它与 `dir_i8` 区间接近，历史上 Code 没跟随变好；不应优先。

### 8.5 Task loss 与三任务失衡

最终 epoch：

| run | code residual | tool residual | memory residual | 读法 |
|---|---:|---:|---:|---|
| R2A e18 | 0.7974 | 0.3407 | 0.0411 | Code 仍主导，Memory loss 变大但幅值仍小 |
| R2B e18 | 0.8062 | 0.3386 | 0.0415 | Full response 没显著改善 code residual |
| R2D e18 | 0.8147 | 0.3425 | 0.0413 | code-heavy 反而不优于 A/B |
| R2C e4 | 15.4647 | 5.6397 | 0.0133 | relative-MSE 尺度不适合直接比较/优化 |

中期 e12：

- R2A code residual `0.9769`，tool `0.3991`，memory `0.0214`。
- R2B code residual `1.0319`，tool `0.3986`，memory `0.0215`。
- R2D code residual `1.0329`，tool `0.4002`，memory `0.0214`。

结论：

- R2A 在相似 gate 区间下 code residual 最低，且使用 code-block span，更符合“可执行代码块”能力目标。
- R2B/D 的 full-response code span 把解释/非代码 token 纳入目标，可能补 reasoning，但当前从 loss 看没有带来更高效的优化。
- R2D 增大 code topk/multiplier 没换来更好 task loss 或更合理 gate，因此不是优先评测对象。

### 8.6 Span token 审计

| run | code span avg | tool span avg | memory span avg | 是否异常 |
|---|---:|---:|---:|---|
| R2A | 195.53 | 63.75 | 9.00 | 正常；code-block auto 生效 |
| R2B | 476.22 | 63.75 | 9.00 | 符合 `code=response`，但 code span 很长 |
| R2C | 476.22 | 63.75 | 9.00 | 平均正常，但有 tool row fallback 到 response 且 residual 极大 |
| R2D | 476.22 | 63.75 | 9.00 | 符合 `code=response` 和 topk 384 |

异常点：

- R2C 有 tool row 使用 `span_mode=response_fallback`、span 91、residual 10.7313；说明某些 tool trajectory 没被识别出 tool-call span，在 relative-MSE 下会制造极大梯度噪声。
- A/B/D 没看到 span token 计数异常；Memory span 短是数据本身 final-answer 风格导致，不是训练脚本错误。

### 8.7 第一批评测优先级

建议先 bake + Tool/Memory 评测：

| 优先级 | gate checkpoint | 原因 |
|---:|---|---|
| 1 | `R2A epoch_008.gates.json` | 最干净、code-block 对齐；gate 与 anchor_i8 安全区接近；loss 比 B/D 更低 |
| 2 | `R2B epoch_008.gates.json` | 检查 full-response code trajectory 是否补 reasoning/I/O；比 R2D 更克制 |
| 3 | `R2A epoch_010.gates.json` | 激进一点，memory gate 0.799；可测 Tool/Memory 判断是否仍过线 |
| 4 | `R2A epoch_012.gates.json` | 对照 dir_i8 激进区；只有前面模型 Tool/Memory 不够或想看上限时再测 |

不建议第一批评测：

- `R2C` 任意 checkpoint：objective/尺度/进程状态都不合格。
- `R2D` final 或 e12+：code-heavy 没提供独立收益，过推风险高。
- A/B/D e18：memory gate `~0.636`，明显偏离当前主目标。

如果只允许选一个：先测 `trc_r2a_cleanrf_wide_dir_20260519/epoch_008.gates.json`。

如果 Tool mean >= 0.79 且 Memory F1 >= 0.76，再跑 Code CURE。若 R2A e8 的 Tool/Memory 低于门槛，则不建议立刻测 Code，先看 R2A e10 或 R2B e8 的 Tool/Memory。
