# 20260520 Residual Coefficient Diagnostic

## 目的

当前 TRC loss 用的是：

```text
target = hidden(row expert, prompt + success response) - hidden(base, same text)
pred   = hidden(current gate, same text) - hidden(base, same text)
```

如果 row expert 是 Code，那么 target 就是 `1.0 * code_delta` 的 hidden residual。因此该 loss 只能鼓励 merged residual 靠近 full Code expert residual，不能判断 Code gate 是否已经过高。

本轮新增一个只读 diagnostic：

```text
scripts/trc/diagnose_residual_coefficients.py
```

它不训练、不写 checkpoint，只把某个 target residual 拟合到 expert basis 上：

```text
r_target ≈ alpha_tool v_tool + alpha_memory v_memory + alpha_code v_code
```

这样可以判断每条轨迹隐含的局部 `alpha*`。如果 `alpha*_code < current_code_gate`，该轨迹就能成为压低 Code gate 的证据。

## 当前轨迹选择机制

### TRC v1 / stable bank

Builder:

```text
scripts/trc/build_trc_calibration_v1.py
```

核心逻辑：

- 从 expert rollout JSONL 里找 `success=True` 或 `reward_train >= positive_threshold` 的 samples；
- 每任务选 32 条；
- unique prompt 优先，不够再补同 prompt 多 sample；
- materialized row 里统一标 `success=True`；
- 不检查 base 是否错误；
- 不要求当前 merged model 是否失败；
- 不保存 same-prompt base/current failure trajectory。

因此它保证的是：

```text
expert 对
```

但不保证：

```text
base 错 expert 对
```

### Round16 Code contrast

Builder:

```text
scripts/trc/build_trc_round16_nonleak_code_contrast_calibration.py
```

Code 逻辑：

- 同 prompt 找一个 positive sample：`success=True` 或 `reward_train >= 1.0`；
- 同 prompt 找一个 negative sample：`reward_train < 1.0`；
- positive 写入 `response`；
- negative 写入 `negative_response`；
- 训练时只有打开 `--contrastive-negative-loss-weight` 才使用 negative。

它保证的是：

```text
同 prompt 有 expert/pass 和 fail sample
```

但 negative 不一定是 base failure，也不一定是当前 gate failure。

## Smoke 结果

### Sanity: row-expert target

命令使用 `--target-source row-expert`，对 4 条 Code rows 拟合。

结果：

| target | alpha_tool | alpha_memory | alpha_code | relative error |
|---|---:|---:|---:|---:|
| Code expert residual | ~0.0 | ~0.0 | ~1.0000 | ~1e-6 |

这说明 diagnostic 数学路径是对的：如果 target 就是 Code expert residual，最优解会回到 full Code。

### Mixture witness target

命令使用：

```text
--target-source coefficients
--target-coefficients tool=0.75,memory=0.75,code=0.75
```

对 Round16 non-leak Code contrast bank 的前 8 条 Code rows 诊断。

输出目录：

```text
/tmp/shared-storage/OnPolicy/diagnostics/trc_residual_alpha_code8_c075_20260520
```

结果：

| rows | alpha_tool mean | alpha_memory mean | alpha_code mean | alpha_code median | clipped relative error |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.4586 | 0.5712 | 0.0446 | 0.0244 | 0.2335 |

关键观察：

- 在 `0.75/0.75/0.75` 这个 success-witness residual 下，Code rows 并不需要 high Code basis 来解释 hidden residual；
- 8/8 的 `alpha*_code < 0.75`；
- 有些 row 的 unconstrained `alpha_code` 甚至为负，被 clipped 到 0；
- row-expert sanity 又证明如果 target 是 full Code expert，alpha 会回到 1。

因此问题不是 least-squares 算错，而是 target 定义决定了梯度方向：

```text
target = full Code expert -> 推高 Code
target = successful merged/witness residual -> 可能压低 Code
```

## 对错是否应该参与轨迹选择

应该，而且必须参与。Residual 本身不是能力标签；它只有和 reward/correctness gap 绑定时才可能代表能力方向。

更合理的四类状态：

| base/current | expert/witness | 用法 |
|---|---|---|
| wrong | right | 核心 positive residual；最能说明缺什么能力 |
| right | right | preservation / retention；不应继续推 full expert |
| wrong | wrong | 暂时不作为 positive；除非有其他 teacher pass |
| right | wrong | anti-expert 或 ignore；防止学坏方向 |

当前 TRC stable bank 只用了第二列 `expert/witness right`，没有第一列 `base/current wrong`。这会把很多“base 本来会”的样本也当作需要 expert residual 的证据，从而产生过推。

## 更合理的训练想法

### 1. 轨迹 bank 改成 recoverability bank

每条 prompt 至少记录：

```text
base_reward
current_gate_reward
teacher_reward
teacher_success_response
current_or_base_fail_response
task / bucket / official verifier details
```

Code 优先选：

```text
base/current fail + teacher pass
```

如果 base/current 已经 pass，则进入 retention，不进入 positive residual push。

### 2. Positive target 不再等于 row expert

对 successful trajectory，用 target witness residual：

```text
r_success = hidden(success witness, text) - hidden(base, text)
```

witness 可以是：

- best-ever merged model；
- TA0.75；
- ReasonFlux / R1 / MemoryAgent 中真正 pass 的轨迹；
- 当前模型 BoN 中 pass 的 sample。

关键是：witness 不是 init，也不是最终答案；它只是提供成功行为 residual。

### 3. 训练 loss 需要能双向推动

建议下一版加入：

```text
L_fit = || sum_i alpha_i v_i - r_success ||^2
```

或者先离线解 `alpha*`，再训练：

```text
L_alpha = || alpha_gate - alpha* ||^2
```

这样当 `alpha*_code < current_code_gate` 时，梯度会自然压低 Code gate。

### 4. 同 prompt pass/fail contrast

继续保留：

```text
score(current, pass residual) > score(current, fail residual) + margin
```

但 negative 最好来自 base/current gate 的真实失败，而不是任意 expert sample failure。

## 下一步

1. 用同一个 diagnostic 跑 best-ever / TA0.75 gate checkpoint target，而不只用固定 0.75 mixture。
2. 对 Code eval-leak prompt 建 recoverability bank：base/current fail + witness pass。
3. 把 `alpha*` 写入 calibration row metadata。
4. 在 TRC trainer 中新增可选 `alpha-fit loss`，默认关闭，不影响旧实验。
5. 用 official eval 决定是否推广，不按 alpha 数值本身做结论。
