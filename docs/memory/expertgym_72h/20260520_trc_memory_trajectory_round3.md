# 2026-05-20 TRC Round3: Memory Trajectory 修正与自动实验策略

## 当前核心判断

- R2D/R2E 的 Tool 已恢复到强区间：BFCL 四类均值约 `0.8085`，说明 Tool augmentation + directional TRC 可以保住 tool-call 格式与参数能力。
- R2D/R2E 的 memory gate 仍偏低，主要不是“memory expert 不可用”，而是 TRC calibration 只对齐了 MemAgent 的 final answer，例如 `\boxed{Laceby}`。这类 response span 只有 1-10 token，无法覆盖 memory update / evidence aggregation / long-context reading 等关键能力 span。
- Code 仍是最难项：ReasonFlux 正样本可恢复，但 code delta 稀疏；hidden-state residual 对齐能把 code gate 推高，却不稳定等价于 official eval 的解题能力。因此今晚不能只盯 gate，要用 Tool/Memory 快评筛选，再少量送 Code。

## 已改动的最小实现

- `scripts/trc/build_trc_calibration_v1.py`
  - 新增 `--memory-response-source final|trajectory-turns`，默认 `final`，旧数据路径不变。
  - `trajectory-turns` 会在 memory row 中保存 `trajectory_turns`，每个 turn 保留自己的 `prompt_text` 与 `text`。
  - 新增 `--memory-trajectory-max-update-turns` 与 `--memory-trajectory-turn-policy uniform|late|first-last`。
- `scripts/trc/train_trc_layer_gates.py`
  - 新增 `--trajectory-turn-loss-task memory`，默认不开。
  - 开启后，memory 每条 prompt 内对多个 turn 分别用真实 `prompt_text + response` 计算 TRC loss，再做 prompt 内平均；这样 memory 仍是一票，不因 turn 数天然压制 Tool/Code。
- `skill/command/run_20260519_trc_round_train_one.sh`
  - 支持 `TRAJECTORY_TURN_LOSS_TASKS="memory"`。

## Round3 calibration

根目录：`/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round3_memorytraj`

- `mtr_full_toolaug_code_rf`
  - Memory: all memory_update turns + final_answer，平均约 `6.66` turns / row。
  - 用于验证完整 MemAgent 轨迹是否直接恢复 memory。
- `mtr_uniform4_toolaug_code_rf`
  - Memory: uniform 4 个 memory_update turns + final_answer，固定 `5` turns / row。
  - 用于平衡 memory 覆盖率与训练速度，是今晚主力配置。
- `mtr_late3_toolaug_code_rf`
  - Memory: last 3 个 memory_update turns + final_answer，固定 `4` turns / row。
  - 用于验证 final evidence 聚合 span 是否比全局均匀更有效。

共同数据：

- Tool: 16 条 BFCL targeted + 16 条 paper96 ToolRL successful。
- Memory: paper96 MemAgent successful，28 unique prompts / 32 rows。
- Code: ReasonFlux-only positive，32 unique prompts；DeepSeek/R1 sample 暂不混入，避免大 delta 与异质格式污染。

## 今晚 35-attempt 原则

- Attempt 不等于完整 official eval；分三层：
  - `probe`: 6-10 epochs TRC train，只看 loss/gate/trajectory span，失败立即删 baked checkpoint。
  - `fast-eval`: Tool + Memory official 子评，只有两项达标才进入 Code。
  - `full-eval`: Code/CURE 或 LiveCodeBench，保留到 evaluation 大表。
- 达标线：
  - Tool BFCL mean `>= 0.79`
  - Memory HotpotQA mean F1 `>= 0.76`
  - Code 只给 Tool/Memory 达标者测，目标超过当前 TRC/E1/E3 与 TA0.75。
- 不再用 gate 安全区间筛 checkpoint；checkpoint selection 以 loss plateau / loss min 为主，gate 只用于诊断。

## 下一批关键实验假设

1. `MTR-U4-main`: uniform4 memory trajectory + code response span + ToolAug，应该解决 memory 被压低。
2. `MTR-L3-late`: late3 memory trajectory，若 memory 更高且 Tool 不降，说明 memory 最关键 span 在后半段 evidence integration。
3. `MTR-Full`: full memory trajectory，若 memory 上升但 Tool/Code 下降，说明完整轨迹过强，需要 turn budget。
4. `MTR-U4-codeweak`: 降低 code multiplier，观察 code residual 是否压制 memory。
5. `MTR-U4-memorylayers`: memory 用更多中后层 hidden states，判断 memory 能力是否主要在中后层表示。

## 当前风险

- Memory trajectory prompt 很长，`max_seq_length=1536` 会截断大量 evidence prompt；Round3 至少需要试 `4096`。
- Full trajectory 训练速度会明显慢，不能全部实验都跑 full。
- Code official eval 慢，今晚必须只测筛选后的少量候选。

## 00:55 新诊断：trajectory 进入了，但 memory loss 幅度不够

`trc_r3a3_mtr_u4_ckpt_1536_20260520` 和 `trc_r3b3_mtr_late3_ckpt_1536_20260520` 已验证：

- OOM 已由 `--gradient-checkpointing + seq=1536` 解决。
- Memory row 确实按 turn 训练：
  - uniform4: `trajectory_turns=5.0`，`span_tokens≈857`
  - late3: `trajectory_turns=4.0`，`span_tokens≈728`
- 但 memory residual loss 仍只有 `0.005-0.006`，远低于：
  - code residual `≈1.18-1.20`
  - tool residual `≈0.49-0.50`
- 因此 optimizer 仍被 Code/Tool 主导，memory gate 从 `1.00` 持续降到 `0.98/0.96`。

解释：directional TRC 在 init=1 附近已经认为 merged residual 与 memory expert residual 方向一致；它不强制 memory delta 的投影幅度保持在 expert 级别。因此 trajectory 修正只解决“覆盖 span”，没有解决“memory 幅度被其他任务挤掉”的问题。

下一步不应继续原配置全量训练，而应测试：

- `TASK_DIRECTIONAL_PROJECTION_FLOOR="memory=1.05"` 或 `memory=1.10`
- `TASK_DIRECTIONAL_PROJECTION_WEIGHT="memory=0.5/1.0"`
- 降低 code multiplier，防止 code residual 大幅主导。
- 使用小样本 8-per-task probe 先确认 memory gate 是否能停止下降，再扩展到 96 条。

## 01:30 新诊断：需要 coefficient-level retention，trajectory loss 与系数幅度保护是两个问题

小样本 probe 结果显示，单独放大 memory hidden-state loss 或提高 memory projection floor 都不能阻止 memory gate 下降：

- `memscale50`: memory loss 放大到 50 倍，memory gate 仍到 `0.9368`。
- `memfloor105/110`: directional projection floor 增加了 memory residual loss，但 memory gate 仍到 `0.92` 左右。
- `codeoff_memtool`: 关闭 code 后 memory 仍下降，说明不是简单 code-vs-memory 冲突。

有效机制是 coefficient-level retention：

- `global coefficient floor=1.0, weight=50` 全量 96 条 run（`trc_r3d_globalfloor50_u4_20260520`）到 epoch8：
  - gate mean: Tool `1.1520`, Memory `0.9996`, Code `1.1599`
- `task-aware expert floor=1.0, weight=50` 全量 96 条 run（`trc_r3f_taskfloor50_u4_20260520`）到 epoch8：
  - gate mean: Tool `1.1520`, Memory `1.0019`, Code `1.1599`

解释：

- memory trajectory 修复了“训练 span 是否覆盖 MemAgent 行为”的问题。
- coefficient floor 修复了“directional loss 不约束 task-vector 投影幅度”的问题。
- 这两个约束缺一不可。只用 hidden-state directional loss 时，init=1 附近 memory 方向已经被认为对齐，optimizer 会继续降低 memory 幅度以服务 tool/code residual。

当前进入快评测的候选：

- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3d_globalfloor50_u4_20260520-selected`
- `/tmp/shared-storage/OnPolicy/checkpoints/trc_r3f_taskfloor50_u4_20260520-selected`

下一轮正在并行验证：

- `trc_r3g_full_taskfloor50_20260520`: full memory trajectory + task-aware floor50。
- `trc_r3h_full_globalfloor50_20260520`: full memory trajectory + global floor50。
- `trc_r3i_u4_codefull_taskfloor50_20260520`: uniform4 memory + code full-response span，测试 Code 是否需要对齐完整解题过程而不是只对齐 code block。

## 01:58 快评测反馈与下一轮方向

`trc_r3d_globalfloor50_u4_20260520-selected` 已完成 Tool + Memory：

- Tool/BFCL: live_parallel `0.8125`, live_parallel_multiple `0.6250`, parallel `0.8800`, parallel_multiple `0.8600`，mean `0.7944`。
- Memory/HotpotQA F1: eval_50 `0.7691`, eval_100 `0.7704`, eval_qa_1_32768 `0.7738`, eval_qa_1_65536 `0.7409`，mean `0.7636`。
- 结论：刚过 Tool/Memory 阈值，已送 Code/CURE eval。

`trc_r3f_taskfloor50_u4_20260520-selected` 的 Tool mean 约 `0.7775`，低于 `0.79`，说明 task-aware floor 虽然更“语义干净”，但当前设置下 Tool live_parallel 有轻微回撤。R3F 暂时等待 Memory 结果，不优先送 Code。

Full trajectory 结论：

- `trc_r3g_full_taskfloor50_20260520` 和 `trc_r3h_full_globalfloor50_20260520` 启动后退出且没有 metrics。
- 加日志的小样本 `trc_r3g2_full_probe_taskfloor50_20260520` 在 seq1536、8 rows/task 下仍 OOM。
- 因此 full trajectory 不是今晚主路线；memory 需要 budgeted turn selection（uniform/late/first-last），而不是 all turns。

下一步并行：

- `trc_r3i_u4_codefull_taskfloor50_20260520`: code span 改为 full response，selected e8 gates `T=1.1567/M=1.0018/C=1.1599`，已送 Tool+Memory。
- `trc_r3j_late3_taskfloor50_20260520`: late3 memory，selected e8 gates `T=1.1523/M=1.0017/C=1.1599`，已送 Tool+Memory。
- `trc_r3k_u4_codefull_globalfloor50_20260520`: 在 R3I 的 code-full span 基础上改用 global floor50，测试 Tool 是否比 task-aware 更稳。

## 02:16 Code-span 消融中期结论

R3I（code full-response span + task-aware floor50）Tool 四类全为 `0`，说明直接把 code 的完整 response hidden state 当作能力 span 会破坏 tool-call behavior。该 baked checkpoint 已删除，只保留 run/eval 日志。

这给出一个重要负结果：

- Code 能力不是“越多 response span 越好”。
- full response 中包含解释、格式、非代码 token，和 ToolRL 的函数调用格式 span 冲突很强。
- 后续 code 修复需要更结构化地选 span：优先 code block / tests / error-repair reasoning，而不是整个 response。

正在验证的两个更稳对照：

- `trc_r3l_u4_codeblock384_globalfloor50_20260520`: 保持 `code-block` span，只把 code topk 从 192 扩到 384。
- `trc_r3m_u4_codefull_toolprotect_globalfloor50_20260520`: full response 但 tool multiplier=2.0、code multiplier=0.8，测试强 Tool 保护是否能抵消冲突。

R3D Code 评测的第一次 `code_20260520_0200` 在 `481/512` 因 GPU 资源冲突 OOM，不计模型结论；已用 GPU 4/7 独占重跑 `code_20260520_0216_rerun`。

02:35 追加：R3M 即使 `tool multiplier=2.0`、`code multiplier=0.8`，Tool 仍四类全 0。结论更确定：不要用 code full-response span 作为 TRC 主损失。R3L（code-block topk384）Tool mean `0.7944`，说明扩大 code-block token budget 是更安全的 code 增强方向。

02:45 组合实验：当前最合理的下一步不是继续调 full-response，而是把两个正向信号合并：

- R3J 提供 `late3 memory`，Memory F1 mean `0.7673`，高于 R3D。
- R3L 提供 `code-block topk384`，Tool mean `0.7944`，说明它没有破坏 Tool。

因此启动：

- `trc_r3n_late3_codeblock384_globalfloor50_20260520`: late3 + code-block384 + global floor50。
- `trc_r3o_late3_codeblock384_taskfloor50_20260520`: late3 + code-block384 + task-aware floor50。

这两个是 Round3 后半夜主候选，用于争取同时维持 Tool/Memory，并给 Code 更安全的额外 hidden-state signal。

02:52 追加：R3L（uniform4 + code-block topk384 + global floor50）Tool mean `0.7944`，但 Memory mean F1 只有 `0.7494`，低于阈值。说明扩大 code-block token budget 本身是安全的 Tool 操作，但会挤压 memory；必须和 late3 memory budget 结合，不能只在 uniform4 上加 code topk。

02:59 追加：R3K（uniform4 + code full-response + global floor50）二次 bake 后通过 Tool/Memory：

- Tool mean `0.7944`
- Memory mean F1 `0.7715`，当前 Round3 最高

这修正了 R3I/R3M 的负结论：full-response span 不是绝对不能用，而是需要 global coefficient floor 稳住整体专家幅度；task-aware floor 或仅提高 tool loss multiplier 都不足以防止 Tool 崩。R3K 需要排 Code 评测，但要等当前 R3D/R3J CURE jobs 完成，避免 GPU 资源冲突。

R3J（late3 memory + task-aware floor50）也通过 Tool/Memory：

- Tool mean `0.7944`，与 R3D 基本持平。
- Memory mean F1 `0.7673`，略高于 R3D 的 `0.7636`。
- 说明 memory trajectory 不一定需要 uniform4；late3 update turns + final answer 可能更贴近最终 evidence integration，对 Memory 更有利且更省计算。

R3J 已送 Code/CURE（`code_20260520_0225`）。如果 R3J Code 不低于 R3D，late3 将成为 Round3 当前主候选。

## 03:20 Code Calibration Diagnosis

当前 Round3 code calibration 不是格式错误，而是目标覆盖不够强：

- 三套 Round3 calibration 的 code rows 完全一致，`32/32` 都是 ReasonFlux 成功轨迹。
- `32/32` 都有 fenced code block；平均 response 约 `353 words`，code-block 平均约 `111 words`。
- 直接来自 LiveBench/LiveCodeBench 的 formal anchor 很少，大约 `4/32`；其余主要是 CURE/CodeContests 或 paper96 风格。
- 这意味着 TRC 主要学“最终代码块形态”，但正式 Code 评测还依赖题意解析、算法规划、边界条件、IO protocol、隐藏测试鲁棒性。
- Code gate 在 R3D/R3J/R3K/R3O 中已经推到约 `1.16`，但这不等价于正式 Code 能力上升；如果 Code eval 不升，主要应怀疑 calibration/recovery bank 和 hidden span 目标，而不是继续盲目加 gate。

下一版 code calibration 应构造 `32-48` 条 eval-aligned recovery trajectories：

- 当前 merged model 失败、teacher 成功优先。
- LiveBench-style 与 LiveCodeBench-style 各占 `10-16` 条。
- CURE/CodeContests hard diverse 保留 `8` 条左右。
- IO/format/edge-case diagnostic 保留 `4-8` 条。
- teacher 不应只用 ReasonFlux；ReasonFlux、DeepSeek/R1、old code expert 的成功轨迹应交叉验证。
- TRC loss 主对齐 code-block；可给 algorithm/reasoning span 一个小权重，避免 full response 学到 verbose style。
