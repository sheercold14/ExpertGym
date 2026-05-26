# 2026-05-25 Direct Memory Code-Component Projection Smoke

本文档记录不训练参数的 Memory prompt code-component projection 小试。

## 1. 方法

新增脚本：

- `scripts/hiddensteer/run_memory_code_component_projection.py`

方法定义：

```text
Delta_m = h_memory_expert - h_merged
Delta_c = h_code_expert - h_merged

M = PCA(Delta_m)
C = PCA(Delta_c)
B_bad = orthogonalize(C against M)
```

推理时不训练 corrector，直接在 selected layers 做闭式投影：

```text
h' = h - alpha * B_bad B_bad^T (h - merged_center)
```

其中 `merged_center`、`M/C/B_bad` 只由 train Memory prompts 建立，heldout prompts 不进入 geometry。

为了避免删除共享 reasoning 背景，只在 token 的 code-bad energy 超过 train Memory 分位数阈值时投影：

```text
energy = ||B_bad^T(h - merged_center)||^2 / ||h - merged_center||^2
project only if energy > tau
```

## 2. 实验设置

- model: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2`
- memory expert: `/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B`
- code expert: `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B`
- task: `memory_update`
- split: prompt-disjoint `20 / 8`
- layers: `16-27`
- basis rank: `4`
- response tail tokens: `64`
- Tool/Code retention: `4` examples each

Primary run:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_code_component_projection_20x8_l16-27_r4_p75p90_20260525`

Alpha sweep:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_code_component_projection_20x8_l16-27_r4_p90_alpha_sweep_20260525`

## 3. 结果

Baseline:

- train NLL: `0.8048`
- heldout NLL: `0.6210`

Primary sweep:

| alpha | threshold | heldout NLL | heldout delta | train delta | Tool delta | Code delta | projected/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | p75 | 0.6208 | -0.0001 | +0.0023 | -0.0045 | -0.0010 | 0.1290 |
| 0.25 | p90 | 0.6101 | -0.0109 | -0.0078 | -0.0001 | +0.0005 | 0.0557 |
| 0.50 | p75 | 0.6296 | +0.0087 | +0.0064 | -0.0068 | +0.0010 | 0.1032 |
| 0.50 | p90 | 0.6096 | -0.0114 | -0.0127 | -0.0026 | +0.0005 | 0.0469 |
| 1.00 | p75 | 0.6929 | +0.0719 | +0.0591 | -0.0030 | +0.0085 | 0.0894 |
| 1.00 | p90 | 0.6481 | +0.0271 | +0.0214 | -0.0032 | -0.0005 | 0.0425 |

P90 alpha sweep:

| alpha | threshold | heldout NLL | heldout delta | train delta | Tool delta | Code delta | projected/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | p90 | 0.6090 | -0.0120 | -0.0113 | -0.0028 | +0.0010 | 0.0509 |
| 0.50 | p90 | 0.6096 | -0.0114 | -0.0127 | -0.0026 | +0.0005 | 0.0469 |
| 0.65 | p90 | 0.6171 | -0.0039 | -0.0076 | -0.0028 | -0.0015 | 0.0447 |

## 4. 判断

这次结果和 trained corrector 相反：

- trained corrector: train NLL 大降，但 heldout 变差，Tool/Code retention 变差。
- direct projection: train 与 heldout 同时小幅改善，Tool 不受损，Code 变化接近 0。

当前最好点：

```text
layers = 16-27
rank = 4
threshold = p90
alpha = 0.35
heldout NLL: 0.6210 -> 0.6090
Tool retention delta: -0.0028
Code retention delta: +0.0010
```

这说明 “直接删除 Memory prompt 中过量的 code-on-memory 正交成分” 有初步正信号。它比 trained corrector 更适合作为方法候选，因为没有 adapter、没有 teacher fitting、没有 prompt memorization channel。

仍然不能直接 claim 解决 Memory：

- 当前只是 teacher NLL proxy，不是 HotpotQA answer score。
- heldout 只有 8 个 prompts。
- Code retention 只测了 4 个 rollout examples，`+0.0010` 虽小但需要更大子集确认。

下一步 gate：

- 固定 `layers=16-27, rank=4, p90, alpha=0.35/0.5`。
- 扩大 Memory heldout NLL 或运行小 HotpotQA answer subset。
- Tool/Code retention 至少扩大到 16 examples。
- 若仍然正向，再考虑 Memory-only trigger 下的 generation/eval。

## 5. Init1 HotpotQA F1 子集

新增脚本：

- `scripts/hiddensteer/run_memory_hotpotqa_projection_eval.py`

用途：

- 本地 HF 复现 MemAgent recurrent-boxed 流程。
- context 分块更新 memory，最后用 memory 生成 boxed answer。
- 同一批样本 paired 比较 baseline vs projection。
- 这是 deterministic HF 子集检查，不是正式 vLLM eval_50 复现。

Setting:

- model: `/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517`
- geometry: 用 init1 重新构建，不复用 rcrf geometry
- layers: `16-27`
- rank: `4`
- threshold: `p90`
- alpha: `0.35`
- generation: greedy, chat template, chunk size `5000`

### A. eval_50 first16

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_hotpotqa_projection_eval_init1_eval50_s16_l16-27_r4_a035_p90_20260525`

Result:

| setting | F1 | EM | sub-EM | wins/losses/ties |
|---|---:|---:|---:|---:|
| baseline | 0.7560 | 0.6250 | 0.8125 | - |
| projection | 0.7143 | 0.6250 | 0.7500 | 0 / 1 / 15 |

Update-only projection gave the same result:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_hotpotqa_projection_eval_init1_eval50_s16_updateonly_l16-27_r4_a035_p90_20260525`

Failure case:

```text
index = 1
gold = Chief of Protocol
baseline = Chief of Protocol of the United States
projection = United States ambassador
```

Interpretation:

- Projection can change answer-level candidate selection.
- Removing code-on-memory residual during memory update is not automatically safe; in this case it removed or weakened the clue needed to choose the more specific answer.

### B. official-init1-wrong16

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_hotpotqa_projection_eval_init1_eval50_official_wrong16_all_l16-27_r4_a035_p90_20260525`

This subset contains the first 16 examples that official init1 eval_50 marked wrong.

Result:

| setting | F1 | EM | sub-EM | wins/losses/ties |
|---|---:|---:|---:|---:|
| baseline | 0.2130 | 0.0000 | 0.1250 | - |
| projection | 0.2963 | 0.1250 | 0.1875 | 2 / 1 / 13 |

Changed examples:

```text
index 1:   Chief of Protocol of the United States -> United States ambassador  (loss)
index 134: Training Day -> Suicide Squad                                      (win)
index 137: Stapleton Cotton -> Lord Combermere                                (win)
```

Current answer-level judgment:

- Direct projection has real answer-level recovery signal on init1 weak cases.
- It is not monotonic: it can also damage candidate disambiguation.
- NLL improvement alone is insufficient; future promotion must use paired HotpotQA F1.
- The next meaningful gate is full eval_50 paired HF/vLLM comparison or at least official-wrong24 plus a matched correct subset.

### C. eval_50 full128

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_hotpotqa_projection_eval_init1_eval50_full128_all_l16-27_r4_a035_p90_20260525`

Result:

| setting | F1 | EM | sub-EM | wins/losses/ties |
|---|---:|---:|---:|---:|
| baseline | 0.7754 | 0.6094 | 0.8125 | - |
| projection | 0.7628 | 0.5938 | 0.7969 | 3 / 7 / 118 |

Recovered cases:

```text
index 123: Kohlberg Kravis Roberts -> KKR & Co. L.P.  (gold: KKR & Co)
index 134: Training Day -> Suicide Squad              (gold: Suicide Squad)
index 137: Stapleton Cotton -> Lord Combermere        (gold: Lord Combermere)
```

Damaged cases:

```text
index 1:   Chief of Protocol of the United States -> United States ambassador
index 24:  World's Best Goalkeeper -> IFFHS World's Best Goalkeeper
index 55:  Marion, South Australia -> Marion
index 82:  Conscription -> requiring only men to register for the draft
index 138: seasonal television specials -> children's television production
index 144: Nebo Zovyot -> Mechte Navstrechu
index 178: Matthew Alistair Grant -> Robin John Bailie
```

Full128 judgment:

- Full paired F1 does not improve; it drops by `0.0126`.
- Projection recovers a few genuinely wrong init1 cases, but unconditional enabling damages more correct or partially correct cases.
- The method is not ready as a default Memory inference-time correction.
- The useful signal is now narrower: code-on-memory projection is a recoverability operator, not a universal Memory boost. A future method would need a reliable trigger for recoverable states or a much safer projection criterion.

## 6. Recoverability-Gated Accept/Reject

新增脚本：

- `scripts/hiddensteer/analyze_memory_projection_gating.py`

目的：

- 不重跑模型，基于 full128 paired baseline/projection 输出做离线 accept/reject 分析。
- policy 决策只能使用预测文本和 projection 运行时统计；gold answer 只用于事后评估。

核心观察：

- 无条件 projection: wins/losses/ties = `3 / 7 / 118`，F1 下降。
- 损伤样本通常 projection 作用范围更广。
- 一个稳定恢复样本 `index=137` 的 projection 很局部：projected/token 约 `0.0816`。

结果：

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_projection_gating_analysis_init1_eval50_full128_20260525`

| policy | accepted | F1 | EM | sub-EM | wins/losses/ties |
|---|---:|---:|---:|---:|---:|
| always baseline | 0 | 0.7754 | 0.6094 | 0.8125 | 0 / 0 / 128 |
| always projection | 128 | 0.7628 | 0.5938 | 0.7969 | 3 / 7 / 118 |
| answer changed and projected/token < 0.085 | 2 | 0.7833 | 0.6172 | 0.8203 | 1 / 0 / 127 |
| answer changed and projected/token < 0.090 | 2 | 0.7833 | 0.6172 | 0.8203 | 1 / 0 / 127 |
| answer changed and projected/token < 0.095 | 3 | 0.7770 | 0.6172 | 0.8125 | 1 / 1 / 126 |
| answer changed and projected/token < 0.100 | 8 | 0.7652 | 0.5938 | 0.8047 | 2 / 5 / 121 |

Accepted changed examples for `< 0.09`:

```text
index 43:  Ethiopia's sovereignty -> Ethiopian sovereignty  (tie)
index 137: Stapleton Cotton -> Lord Combermere              (win)
```

Interpretation:

- There is a plausible side-effect reducer: do not trust every corrected answer; accept only localized projection changes.
- This converts the same full128 paired outputs from negative to positive: F1 `0.7754 -> 0.7833`.
- However, the threshold was selected after inspecting this run. It is a candidate gate, not a validated method.
- Next validation must freeze the rule, then test on a disjoint slice such as eval_100 or a different eval_50 offset/seed.

## 7. Disjoint eval_100 Validation

Path:

- paired run: `/tmp/shared-storage/ExpertGym/hiddensteer/memory_hotpotqa_projection_eval_init1_eval100_full128_all_l16-27_r4_a035_p90_20260525`
- gating run: `/tmp/shared-storage/ExpertGym/hiddensteer/memory_projection_gating_analysis_init1_eval100_full128_20260525`

Same init1 model, same geometry basis, same projection hyperparameters:

```text
layers 16-27, rank4, alpha 0.35, train-memory p90 trigger
```

### A. Unconditional projection

| split | setting | F1 | EM | sub-EM | wins/losses/ties |
|---|---|---:|---:|---:|---:|
| eval_50 full128 | baseline | 0.7754 | 0.6094 | 0.8125 | - |
| eval_50 full128 | projection | 0.7628 | 0.5938 | 0.7969 | 3 / 7 / 118 |
| eval_100 full128 | baseline | 0.7201 | 0.5703 | 0.7656 | - |
| eval_100 full128 | projection | 0.7520 | 0.6016 | 0.7891 | 8 / 4 / 116 |

Interpretation:

- Direct projection is not uniformly negative. On disjoint eval_100 it improves F1 by `+0.0320` and EM by `+0.0313`.
- Combined across eval_50 and eval_100 full128, the average F1 moves from roughly `0.7478` to `0.7574`.
- The remaining issue is side effect control: eval_100 has strong wins, but still 4 losses; eval_50 has more losses than wins.

### B. Frozen low-rate gate

The previously promising eval_50 rule was:

```text
accept projected answer iff answer changed and projected/token < 0.09
```

On eval_100 this does not validate:

| policy | F1 | EM | sub-EM | wins/losses/ties |
|---|---:|---:|---:|---:|
| baseline | 0.7201 | 0.5703 | 0.7656 | 0 / 0 / 128 |
| answer changed and projected/token < 0.09 | 0.7196 | 0.5781 | 0.7578 | 1 / 1 / 126 |

Conclusion:

- Projection rate alone is too weak as a recoverability trigger.
- It captures localness, but it cannot distinguish a useful correction from a wrong answer rewrite.

### C. Answer-shape side-effect guard

新增到脚本：

- `scripts/hiddensteer/analyze_memory_projection_gating.py`

Guard rule:

```text
default use projection,
but reject projected answer if it:
1. expands the baseline answer by adding tokens around it;
2. flips a pure numeric/year answer to a different number;
3. changes a short answer into a long unrelated phrase.
```

This rule uses only prediction strings; gold is used only for evaluation.

| split | policy | F1 | EM | sub-EM | wins/losses/ties |
|---|---|---:|---:|---:|---:|
| eval_50 full128 | projection | 0.7628 | 0.5938 | 0.7969 | 3 / 7 / 118 |
| eval_50 full128 | projection + answer-shape guard | 0.7717 | 0.6094 | 0.8047 | 3 / 5 / 120 |
| eval_100 full128 | projection | 0.7520 | 0.6016 | 0.7891 | 8 / 4 / 116 |
| eval_100 full128 | projection + answer-shape guard | 0.7606 | 0.6094 | 0.8125 | 7 / 0 / 121 |

Interpretation:

- The answer-shape guard reduces side effects better than projection-rate gating.
- On eval_100 it removes all 4 losses while keeping 7 of 8 wins.
- On eval_50 it improves over unconditional projection but still does not beat baseline; the remaining losses are entity substitutions or granularity changes that cannot be reliably judged by answer string shape alone.

Current method-level conclusion:

- The geometric projection itself has real recoverability signal.
- Side effects are not solved by a single scalar runtime statistic.
- A viable method should combine activation-level projection with an answer-level safety selector. The selector must reject risky rewrites, and likely needs a stronger no-gold confidence signal than string shape alone, such as residual-consistency or cheap candidate verification.
