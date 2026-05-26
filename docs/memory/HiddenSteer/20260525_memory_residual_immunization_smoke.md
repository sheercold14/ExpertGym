# 2026-05-25 Memory-Conditioned Residual Immunization Smoke

本文档记录 Memory-Conditioned Residual Immunization 的第一轮最小可行性验证。结论先行：

> HF 原型已跑通；在 prompt-disjoint 的 `memory_update` 小样本上，低秩 corrector 能显著降低 train NLL，但 20/8 ablation 显示 heldout NLL 稳定变差，Tool/Code retention 也小幅受损。当前 trained residual corrector 方向应停止，不进入 HotpotQA eval_50 或 vLLM 工程化。

## 1. 新增脚本

- `scripts/hiddensteer/train_memory_residual_immunization.py`

功能：

- 从 Memory expert rollout 中抽取 teacher trajectory。
- 顺序加载 merged / memory expert / code expert。
- 在 Memory prompt 上构建：

```text
memory_anchor = h_memory_expert - h_merged
code_on_memory = h_code_expert - h_merged
code_bad = code_on_memory orthogonalized against memory_anchor
```

- 在 merged model 上训练零输出初始化的低秩 residual corrector：

```text
h' = h + U_l V_l h
```

- loss：

```text
teacher response NLL
+ code_bad projection penalty
+ correction norm penalty
```

注意：zero-init 指 correction 初始输出为 0；实现上 down 随机小初始化、up 为 0，避免双零导致梯度消失。

## 2. 模型和数据

- merged model: `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2`
- memory expert: `/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B`
- code expert: `/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B`
- rollout: `/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/memory_expert_paper96_s2_seed20260514.jsonl`

## 3. Smoke 结果

### A. final-answer tiny

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_smoke_tiny_20260525`

Setting:

- turn: `final_answer`
- layer: `26`
- basis rank: `2`
- corrector rank: `2`
- train / heldout: `3 / 2`
- steps: `8`

Result:

| split | baseline NLL | corrected NLL | delta |
|---|---:|---:|---:|
| train | 0.000041 | 0.000010 | +0.000031 |
| heldout | 0.000004 | 0.000005 | -0.000002 |

判断：

- final-answer prompt 太容易，merged 已经几乎完全拟合 teacher response。
- 这个设置没有诊断价值，只能证明脚本端到端可跑。

### B. memory-update same-prompt tiny

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_update_tiny_20260525`

Setting:

- turn: `memory_update`
- layer: `26`
- basis rank: `2`
- corrector rank: `2`
- train / heldout: `3 / 2`
- steps: `8`

Result:

| split | baseline NLL | corrected NLL | delta |
|---|---:|---:|---:|
| train | 0.2490 | 0.1336 | +0.1154 |
| heldout | 0.1381 | 0.1314 | +0.0067 |

判断：

- update turn 比 final answer 有信号。
- 但 train / heldout 来自同一个 prompt 的不同 turn，因此证据偏弱。

### C. memory-update prompt-disjoint

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_update_promptsplit_v2_20260525`

Setting:

- turn: `memory_update`
- prompt-disjoint split
- layer: `26`
- basis rank: `2`
- corrector rank: `2`
- train / heldout: `4 / 3`
- steps: `12`
- `code_bad_weight=0.05`

Result:

| split | baseline NLL | corrected NLL | delta |
|---|---:|---:|---:|
| train | 0.7510 | 0.4055 | +0.3455 |
| heldout | 0.6354 | 0.6325 | +0.0029 |

Generation smoke on one heldout prompt:

- baseline output rambled into another question after 64 tokens.
- corrected output became shorter and changed answer content.
- This proves the corrector has behavioral effect, but not that the behavior is more correct.

### D. no-code-bad ablation

Path:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_update_promptsplit_nocodebad_20260525`

Same setting as C, but `code_bad_weight=0.0`.

Result:

| split | baseline NLL | corrected NLL | delta |
|---|---:|---:|---:|
| train | 0.7510 | 0.4063 | +0.3447 |
| heldout | 0.6354 | 0.6302 | +0.0052 |

判断：

- no-code-bad ablation is slightly better on this tiny heldout.
- Therefore the current positive signal should be attributed to Memory teacher NLL fitting, not to successful code residual immunization.
- The `code_bad` basis construction is implemented, but the current penalty is not yet proven useful.

### E. memory-update prompt-disjoint 20/8 ablation

Paths:

- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_update_20x8_w000_20260525`
- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_update_20x8_w005_20260525`
- `/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_update_20x8_w010_20260525`

Shared setting:

- turn: `memory_update`
- prompt-disjoint split
- layer: `26`
- basis rank: `2`
- corrector rank: `2`
- train / heldout: `20 / 8`
- steps: `60`
- `max_turns_per_prompt=1`
- Tool/Code retention: `4` expert-rollout examples each

Result:

| code_bad_weight | train NLL base -> corrected | heldout NLL base -> corrected | heldout delta, corrected - base | Tool retention NLL delta | Code retention NLL delta | train time |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.8048 -> 0.3582 | 0.6210 -> 0.7158 | +0.0948 | +0.0213 | +0.0259 | 3.43s |
| 0.05 | 0.8048 -> 0.3858 | 0.6210 -> 0.6763 | +0.0553 | +0.0185 | +0.0132 | 3.46s |
| 0.10 | 0.8048 -> 0.3588 | 0.6210 -> 0.7112 | +0.0902 | +0.0216 | +0.0269 | 3.45s |

Runtime notes:

- Single-layer rank-2 hook is computationally feasible in the HF prototype.
- On two heldout generation-smoke prompts, `w=0.05` wall-time ratios were about `0.97x` and `1.02x`.
- Ratios below `1.0x` in `w=0` / `w=0.1` are not meaningful speedups because one corrected output stopped at fewer tokens.

Judgment:

- All three weights show the same pattern: train improves sharply, heldout worsens.
- Stronger `code_bad` penalty does not rescue heldout generalization.
- Off-task Tool/Code NLL increases rather than decreases, so enabling this corrector outside Memory prompts would be harmful.
- Generation changes are real, but correctness is unproven and NLL evidence is negative.

## 4. Current Judgment

What is validated:

- Memory rollout extraction works.
- Sequential merged / memory / code model activation collection works.
- `code_bad` orthogonal basis construction works.
- Zero-output low-rank residual corrector trains without changing model weights.
- On `memory_update` spans, teacher NLL can be reduced on train.
- Inference hook changes generation and has measurable runtime output.
- A single selected layer with rank-2 correction is not the runtime bottleneck in HF.

What is not validated:

- No HotpotQA eval_50 score yet.
- No heldout Memory improvement at 20/8 scale.
- Tool/Code retention is not stable under the learned corrector.
- `code_bad` penalty is not shown to help; `w=0.05` only reduces the damage relative to `w=0/0.1`.
- Generation change may be too aggressive or semantically wrong.

Current decision:

- Stop the current trained residual-corrector variant.
- Do not claim Memory-Code conflict is solved.
- Do not run vLLM engineering.
- Do not run HotpotQA eval_50 for this variant; the gate failed before answer-level evaluation.

## 5. Next Gate

The next method should not be another sweep of the same learned corrector. If this line is reopened, it needs a stricter first-principles constraint:

- no prompt-specific fitting objective as the main signal;
- correction must be conditioned to Memory spans only, not globally enabled;
- the operation should be a precomputed projection or very small closed-form correction, not a trainable adapter with enough freedom to memorize teacher phrasing;
- success gate remains: heldout Memory NLL or answer quality improves, while Tool/Code retention NLL does not increase.
