# 2026-05-21 PromptAttention-UtilityHarm 初步报告

## 目标

当前 OP-VEC/GRPO 线的主要困难是 calibration reward 不平滑、Code 信号弱、Tool 容易被破坏。PAUH 的目标是先验证一条更简单的结构化合并路线：不训练 gate，不依赖 reward，只用 prompt activation 判断每个 expert task vector 应该在哪些层更强、哪些层应该被压低。

核心假设：

- prompt 分布会激活任务相关的子空间；
- 如果某个 expert 的某层 delta 在 owner task prompt 上 exposure 高，在其他 task prompt 上 exposure 低，则该层适合保留或提高；
- 如果 exposure 主要来自非 owner task，则该层更可能是干扰源；
- attention projection 的输入 activation 可以作为层选择信号，但能力注入不一定只发生在 attention module。

## 方法

输入数据：

- calibration prompts: `/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl`
- mode manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`
- 每个任务最多取 32 条 prompt。

实现文件：

- `scripts/attention_pauh/core.py`
- `scripts/attention_pauh/build_prompt_attention_utility_harm_gates.py`
- `docs/config/20260521_prompt_attention_utility_harm.md`

计算步骤：

1. 只 forward prompt，不生成、不打 reward、不读 expert trajectory。
2. 在 base model attention `q/k/v/o` linear 输入处挂 hook。
3. 对 prompt tail token 统计 activation diagonal，默认 tail 长度 256。
4. 对每个 expert/layer 计算近似 exposure：
   `E ||X * delta^T||^2`。
5. 对每个 expert 计算 layer score：
   `raw_score = log(utility(owner task)+eps) - log(harm(other tasks)+eps)`，再在 expert 内做 z-score。
6. 把 score 映射为 gate：
   `weight_l = exp(beta * score_l)`，当前 `beta=0.7` 控制层间分配的尖锐程度。默认每个 expert 的平均 alpha 保持为 0.75，单层系数 clip 到 `[0.25, 1.25]` 后再归一化。

## 变体

| 变体 | gate 数量 | 含义 |
|---|---:|---|
| PAUH layer-all | 588 | 用 attention exposure 决定 layer score，但该层 attention + MLP 都按同一 expert 系数注入 |
| PAUH attn-only | 336 | 只注入 attention q/k/v/o delta，不注入 MLP delta |

## 已完成结果

### PAUH layer-all

模型：

`/tmp/shared-storage/ExpertGym/pauh/pauh_energy_layer_all_20260521/baked_policy`

gate summary:

- tool mean 0.75, min 0.25, max 1.25
- memory mean 0.75, min 0.25, max 1.25
- code mean 0.75, min 0.25, max 1.25

Tool quick:

| subset | score |
|---|---:|
| parallel | 0.905 |
| parallel_multiple | 0.860 |
| live_parallel | 0.750 |
| live_parallel_multiple | 0.667 |
| mean | 0.795 |

Memory quick:

| subset | avg F1 |
|---|---:|
| eval_50 | 0.735 |
| eval_100 | 0.737 |
| mean | 0.736 |

结论：Tool 没有崩，Memory 接近 TA0.75 级别，但未达到此前最强 memory gate 模型。

### PAUH attn-only

模型：

`/tmp/shared-storage/ExpertGym/pauh/pauh_energy_attn_only_20260521/baked_policy`

Tool quick:

| subset | score |
|---|---:|
| parallel | 0.915 |
| parallel_multiple | 0.860 |
| live_parallel | 0.688 |
| live_parallel_multiple | 0.708 |
| mean | 0.793 |

Memory quick:

| subset | avg F1 |
|---|---:|
| eval_50 | 0.665 |
| eval_100 | 0.614 |
| mean | 0.640 |

结论：attention-only 不能充分保留 memory 能力。当前更支持“attention activation 适合做层选择信号，但实际注入应覆盖该层 MLP/attention”的解释。

## Code 正式评测

PAUH layer-all Code 正式评测已完成：

- summary dir: `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_energy_layer_all_20260521/code_cure`
- log: `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_energy_layer_all_20260521/code_cure/logs/code_cure.log`

Code 正式评测已完成：

| dataset | acc | accumulate acc | BoN `(4,4)` acc | BoN `(4,4)` accumulate acc |
|---|---:|---:|---:|---:|
| LiveBench | 0.3887 | 0.4953 | 0.4375 | 0.5602 |
| LiveCodeBench | 0.3126 | 0.4562 | 0.3718 | 0.5442 |
| mean | 0.3506 | 0.4758 | 0.4047 | 0.5522 |

结果路径：

- feedback dir: `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/pauh-energy-layer-all-20260521/20260521_pauh_layer_all_code`
- LiveBench result: `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/pauh-energy-layer-all-20260521/20260521_pauh_layer_all_code/results/LiveBench.txt`
- LiveCodeBench result: `/tmp/shared-storage/OnPolicy/eval/cure_feedback/opvec-gated-grpo-full-eval-code/pauh-energy-layer-all-20260521/20260521_pauh_layer_all_code/results/LiveCodeBench.txt`

后续需要补齐：

- 与 TA0.75、best gate、TRC hidden-state 系列的横向比较；
- `PAUH` vs `layer-shuffle` vs `inverse-PAUH` 的证伪实验；
- top-layer / bottom-layer 单独注入的真实边际效应验证。

## 当前判断

PAUH 的价值不在于直接替代 RL gate，而在于提供一个更干净的结构化先验：

- 它不需要 reward，因此不会被 noisy calibration 直接带偏；
- 它能解释为什么不同 expert 应该在不同层更强；
- 它可以作为 init gate 或 pruning mask，后续再叠加少量 OPD/GRPO；
- 目前 `layer-all > attn-only` 是一个有意义的 insight：attention exposure 可以做 routing signal，但 MLP 仍是能力承载的重要部分。

## Subagent Review 要点

讨论后需要修正 PAUH 的定位：它当前不是严格的 utility/harm 因果估计器，而是 `activation-conditioned layer reweighting prior`。

关键风险：

- `E||X * delta^T||^2` 只能说明某层 task vector 在 prompt 分布上会产生多大扰动，不能说明扰动方向是否提升 reward。
- non-owner exposure 不一定是 harm，也可能是共享能力，例如 instruction following、格式控制、结构化输出。
- 当前只看 prompt tail，不能覆盖 decoding-time reasoning/code/tool-call 状态。
- diagonal covariance 近似忽略 feature correlation。
- 当前固定每个 expert 平均 alpha 为 0.75，只解决层间预算分配，不能解决 expert 全局 scale 应该多大的问题。

因此更严谨的论文表述应是：

> prompt activation exposure provides a training-free structural prior for where expert task vectors are likely to act; utility/harm must be validated by held-out task metrics or strengthened with signed first-order / contrastive trajectory signals.

最小证伪实验：

1. `TA0.75` vs `PAUH` vs `PAUH layer-shuffle` vs `inverse-PAUH`。
2. top-layer / bottom-layer 单独注入，测 owner gain 与 protected harm 是否和 PAUH score 正相关。
3. Code 任务加 same-prompt pass/fail contrast probe，验证 PAUH top layers 是否更接近 pass trajectory。

下一版最值得实现：

- absolute utility floor：低 utility 层即使 ratio 高也不能直接打到 max。
- layer smoothing：避免单层噪声尖峰。
- hybrid module scope：attention 用 PAUH gate，MLP 用 half-strength gate，而不是 `attn-only` 或全强度 `layer-all` 二选一。
- memory late floor / code mid floor：避免把 answer synthesis 或 algorithm planning 层压得过低。

## 2026-05-21 补充：alpha1 / inverse / shuffle 机制对照

补充产物：

| variant | gate | baked policy | eval |
|---|---|---|---|
| PAUH alpha1 layer-all | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_layer_all_20260521/pauh_gates.json` | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_layer_all_20260521/baked_policy` | `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_alpha1_layer_all_20260521/quick_tool_memory_rerun` |
| PAUH alpha1 inverse | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_inverse_20260521/pauh_gates.json` | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_inverse_20260521/baked_policy` | `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_alpha1_inverse_20260521/quick_tool_memory` |
| PAUH alpha1 shuffle | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_shuffle_20260521/pauh_gates.json` | `/tmp/shared-storage/ExpertGym/pauh/pauh_alpha1_shuffle_20260521/baked_policy` | `/tmp/shared-storage/ExpertGym/pauh/eval/pauh_alpha1_shuffle_20260521/quick_tool_memory` |

| variant | BFCL mean | live_parallel | live_parallel_multiple | parallel | parallel_multiple | Memory F1 mean |
|---|---:|---:|---:|---:|---:|---:|
| PAUH alpha0.75 layer-all | 0.7954 | 0.7500 | 0.6667 | 0.9050 | 0.8600 | 0.7362 |
| PAUH alpha1 layer-all | 0.7438 | 0.6250 | 0.6250 | 0.8800 | 0.8450 | 0.7362 |
| PAUH alpha1 inverse | 0.6698 | 0.6875 | 0.5417 | 0.7300 | 0.7200 | 0.7577 |
| PAUH alpha1 shuffle | 0.7854 | 0.7500 | 0.6667 | 0.8850 | 0.8400 | 0.7563 |
| PAUH attn-only | 0.7927 | 0.6875 | 0.7083 | 0.9150 | 0.8600 | 0.6397 |

新结论：

- `inverse` 明显伤 Tool，尤其 parallel / parallel_multiple，并引入大量 AST parse failure；说明错误层排序会破坏 tool-call 格式行为。
- `alpha1 layer-all` 也伤 Tool，说明 Tool 不适合盲目增加全 expert residual scale。
- `shuffle` 在当前 seed 下没有明显伤 Tool，但也没有超过 alpha0.75；说明 PAUH 层排序是 useful prior，不是充分因果解释。
- Memory 在 alpha1 inverse/shuffle 下仍不差，符合前面 MLP/raw residual 主导的诊断。

因此 PAUH 的论文定位应进一步收敛为：`activation-conditioned structural prior`，而不是完整 utility/harm estimator。
