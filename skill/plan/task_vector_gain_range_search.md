# Task Vector 初始值范围搜索：Codex 实现文档

## 0. 文档目标

本文档用于实现一个可迁移的 **Verifier-Guided Expression Learning (VGEL)** 合并搜索器。目标不是人工 sweep 一个固定系数，而是在极小 calibration 数据下学习：

1. 全局 task-vector 幅值 `s` 对 verified reward 的一维响应；
2. 在高收益幅值区间附近，expert/layer 非均匀表达 `delta` 是否带来额外增益。

这使方法可以迁移到其他 merge 任务：只需要替换 expert checkpoints、merge builder 和 verified reward adapter，不需要知道某个任务上的最佳 alpha 先验。

### 0.1 Verified signal policy

对比 `AgentMerging_plan` 的评测 harness 和 `OnPolicyMerge` 的 source-reward/frontier 设计后，VGEL 采用混合信号：

- 最终报告使用 benchmark aggregate metrics，便于和历史 VGEC/TA/DARE/TIES 结果对齐。
- 搜索与曲面学习使用 per-sample verifier score，而不是只使用 `metrics.json` 的 aggregate score。
- per-sample score 来自同一套 verifier 标准化产物 `predictions.jsonl`：
  - Tool: 结构化 tool-call correctness score，覆盖函数名、参数名、参数值和调用结构。
  - Memory: final-answer verifier score，保留 F1/substring success；strict exact 可作为 success gate，但不直接替代连续 score。
  - Code: source/public tests pass rate；success 是 all-tests-pass，score 是 per-sample pass rate。

原因：

1. aggregate metrics 只有一个均值，无法估计校准噪声；
2. hard-clipped expert-gain 会把多个候选压成同分平台，丢失高分区域内部梯度；
3. OnPolicy/frontier 的关键优势不是固定 reward 权重，而是每条 prompt 有 verifier-confirmed reward、success、error type，可 bootstrap 出 LCB/UCB，并筛掉没有区分度的 prompt。

因此，方法中的 verified signal 定义为：

```text
z_{i,t}(alpha) = normalize_by_base_expert(score_{i,t}(alpha))

mu_t(alpha)    = mean_i z_{i,t}(alpha)
sigma_t(alpha) = bootstrap_std_i z_{i,t}(alpha)
lcb_t(alpha)   = mu_t(alpha) - k * sigma_t(alpha)
```

其中 `score_{i,t}` 是 verifier 给第 `i` 个 calibration 样本的连续 reward，`success_{i,t}` 仅作为可靠性约束或 frontier 筛选信号。

核心形式：

```text
theta(alpha) = theta_base + alpha * Delta
```

其中：

- `theta_base`：base model 参数。
- `Delta`：保结构的 task-vector 合并方向，可以来自 task arithmetic、TIES、DARE、平均 task vector 或已有最佳合并方向。
- `alpha`：全局 task-vector gain，也就是行为表达强度。

需要解决的问题：

```text
统一 alpha 的幅值会强烈影响 Agent 行为表达，但最佳区间未知；
非均匀 expert/layer 系数可能进一步提升能力，但必须证明其增益超过统一幅值本身。
```

因此 VGEL 将问题拆成两个 posterior：

```text
Stage 1: p(s, verified_reward)                    # 全局幅值响应
Stage 2: p(delta | s, residual_verified_gain)      # expert/layer residual 表达
```

最终报告不只看 raw score，还要看：

```text
residual_gain(alpha_structured)
  = verified_score(alpha_structured)
  - E_stage1[verified_score | mean(alpha_structured)]
```

如果 residual gain 稳定为正，才能说明 GMM/structured expression 找到了超出统一 alpha 的有效表达空间。

---

## 1. 研究假设与实现直觉

### 1.1 现象假设

在 RL-trained act model / agent model 中，task vector 的方向可能已经包含有效行为迁移方向，但其默认幅值未必能充分表达该行为方向。

因此：

```text
alpha 太小：task behavior 欠表达。
alpha 适中：task vector 诱导的行为偏移被充分表达，同时 base model 推理结构仍稳定。
alpha 太大：行为偏移过强，模型离开 functional trust region，轨迹性能下降。
```

这对应一个常见曲线：

```text
performance(alpha): under-expressed -> sweet spot -> over-expressed
```

### 1.2 实现上的核心选择

我们不直接学习高维参数，也不直接学习 layer-wise coefficient。第一版只搜索一个全局 gain：

```text
alpha in positive real numbers
```

为了避免线性尺度不稳定，所有搜索都在 log-space 进行：

```text
x = log(alpha)
alpha = exp(x)
```

后续优化器不直接优化 `alpha`，而优化 residual：

```text
alpha = alpha_center * exp(beta)
beta initialized as 0
```

---

## 2. 总体算法：VGEL

建议方法名：

```text
VGEL: Verifier-Guided Expression Learning
```

中文：

```text
验证器引导的 Task Vector 表达学习
```

算法分为 2 个阶段：

1. **Stage 1: zero-prior amplitude response modeling**  
   基础版 scalar Stage1 从 `s=0` 或极小半径出发做 adaptive bracket，不使用任务先验 alpha。每个 scalar candidate 用同一组极小 calibration verified reward 评测，拟合一维经验高斯混合响应：

   ```text
   p(s, y) = sum_k pi_k N([s, y] | mu_k, Sigma_k)
   y = aligned verified reward
   ```

   通过条件期望和不确定性选下一点：

   ```text
   acquisition(s) = E[y | s] + beta * Std[y | s]
   ```

   输出 observed best、recommended scale 和 high-reward region。

   主实验更推荐 per-expert Stage1。它不再强制所有 expert 使用同一个系数，而是先学习三维变量：

   ```text
   alpha[e, g] = s_e
   s = [s_tool, s_memory, s_code]
   ```

   也就是每个 expert 一个全局表达强度，layer group 暂时共享。响应模型不直接依赖人工解释某个点是否合理，而是先拟合 verified task surface：

   ```text
   r_tool(s), r_memory(s), r_code(s)
   s = [s_tool, s_memory, s_code]
   ```

   每个 task surface 用 empirical Gaussian mixture / RBF 条件期望。基础版可以只拟合均值：

   ```text
   w_i(s) = exp(-||s - s_i||^2 / (2h^2))
   E[r_t | s] = sum_i w_i(s) r_{i,t} / sum_i w_i(s)
   ```

   更新版应直接对 per-sample verifier reward 拟合任务级分布：

   ```text
   z_{j,t}(s_i) = normalized verifier score for sample j, task t, candidate i

   E[z_t | s]   = sum_i w_i(s) mean_j z_{j,t}(s_i) / sum_i w_i(s)
   Var[z_t | s] = epistemic_var_across_points + aleatoric_var_from_bootstrap
   LCB_t(s)     = E[z_t | s] - k * sqrt(Var[z_t | s])
   ```

   这里 `epistemic_var_across_points` 来自邻近 candidate 的 reward 差异，`aleatoric_var_from_bootstrap` 来自同一 candidate 的 calibration sample bootstrap。这样得到的是平滑分布曲面，而不只是几个 aggregate 点的插值。

   然后把预测的 verified task rewards 送入同一个 proxy objective，得到 acquisition：

   ```text
   y_proxy(s) = Proxy(E[r_tool | s], E[r_memory | s], E[r_code | s])
   acquisition(s) = y_proxy(s) + beta * Std[y_proxy | s]
   ```

   当前实现支持三类 proxy：

   ```text
   mean       = (tool + memory + code) / 3
   balanced   = mean - lambda * std(tool, memory, code)
   geometric  = (tool * memory * code)^(1/3)
   ```

   `geometric` 是更严格的 product-of-capabilities 目标，等价于在独立 task verifier 假设下最大化平均 verified log-success；它会自然惩罚任一能力坍塌，不需要人为指定某个 expert 更重要。

   但在 expert-normalized reward 上不应使用 hard clip 后的 geometric 作为唯一目标，因为 `clip(..., 0, 1)` 会把所有超过 expert baseline 的候选压成同分平台。推荐使用两层 acquisition：

   ```text
   feasibility(s) = min_t LCB_t(s)
   tie_break(s)   = mean_t softplus(E[z_t | s] - 1) + eta * mean_t E[z_t | s]
   A(s)           = feasibility(s) + gamma * tie_break(s)
   ```

   其中 `feasibility` 保证最弱能力不塌，`tie_break` 只在所有任务都接近或超过 expert retention 时区分平台内部的优劣。这样 code/tool 的过高得分不能补偿 memory 坍塌，但也不会把高分平台完全抹平。

   为了让不同 task verifier 的尺度一致，推荐在 proxy 前做 expert-normalized reward。令 `r_t` 是 merge 模型在任务 `t` 上的 raw verified reward，`r_t^base` 是 base 模型 reward，`r_t^expert` 是对应 expert 在同一 calibration 子集上的 reward：

   ```text
   retention_t = (r_t - r_t^base) / (r_t^expert - r_t^base)
   ```

   如果没有稳定 base baseline，也可以退化为：

   ```text
   ratio_t = r_t / r_t^expert
   ```

   训练 acquisition 内部可以对 `retention_t` 做 soft cap 或 winsorize，例如 `[0, 1.5]`，但报告里同时保留 raw retention 和 clipped retention。这样 proxy 衡量的是各 expert 能力保持率，而不是 raw metric 尺度；任一 expert 能力坍塌都会直接拉低总体目标，高分平台内部仍保留排序信息。

   高收益区域分布不直接等同于 reward 曲面。推荐先学习任务曲面，再定义 good-region posterior：

   ```text
   p_good(s) ∝ exp(A(s) / tau)
   ```

   然后用 weighted GMM / CEM 拟合 `p_good(s)`：

   ```text
   q_phi(s) = sum_k pi_k N(s | mu_k, Sigma_k)
   phi = argmin_phi KL(p_good || q_phi)
   ```

   输出的不只是 best alpha，而是：

   ```text
   good_region = {s : p_good(s) >= quantile_80}
   posterior_mean, posterior_cov, mixture_modes
   next_probe = argmax_s EIG(s) or UCB(A(s))
   ```

   这样可以区分两类现象：

   ```text
   共享 scalar 变大带来的总体表达增强
   vs.
   某个 expert 单独需要更大/更小表达强度
   ```

2. **Stage 2: structured residual expression modeling**  
   在 Stage 1 的高收益区间附近学习：

   ```text
   alpha[e, g] = s_e + delta_group[g] + delta_residual[e, g]
   ```

   对 structured candidate 的目标不是 raw score，而是相对 Stage 1 的增益：

   ```text
   residual_gain = verified_score(alpha[e,g]) - E_stage1[y | mean(alpha[e,g])]
   posterior_objective = verified_score + residual_gain - lambda * std(alpha)
   ```

   其中 `std(alpha)` 是复杂度惩罚，避免在极小 calibration 数据下产生无意义的高方差系数。

### 2.1 收敛与停止准则

VGEL 不用人工判断“扫到哪里算够”，而是输出可复现的 convergence diagnostic。

#### Stage 1 幅值响应收敛

一维响应模型维护当前最优观测：

```text
y_best = max_i y_i
```

以及 acquisition 上的最佳候选：

```text
s_next = argmax_s E[y | s] + beta * Std[y | s]
EI = max(0, acquisition(s_next) - y_best)
```

Stage 1 的全局扩展收敛条件：

```text
EI <= epsilon_stage1
and best point is not on expandable boundary
```

如果高收益区间仍然较宽：

```text
normalized_width(high_reward_region) > epsilon_region
```

则不再继续向外扩展，而是进入局部细化：

```text
decision = stage1_region_converged
next = local_refine(high_reward_region)
```

如果最优 scalar 点落在当前右边界，且还没有达到 `max_radius`，则判定为：

```text
decision = expand_amplitude
```

这保证算法可以自动发现类似 `0.4-0.8` 的大幅值区间，而不是预设 `0.75`。

#### Stage 2 residual 表达收敛

对每个 structured candidate，先用 Stage 1 模型给出同均值幅值下的期望：

```text
baseline = E_stage1[y | mean(alpha_structured)]
residual_gain = y_structured - baseline
```

再用 Stage 1 的局部不确定性构造置信区间：

```text
gain_lcb = residual_gain - z * Std_stage1[y | mean(alpha)]
gain_ucb = residual_gain + z * Std_stage1[y | mean(alpha)]
```

Stage 2 决策：

```text
if max(gain_lcb) > 0:
    accept_structured
elif num_structured < min_candidates:
    continue_structured_sampling
elif max(gain_ucb) <= epsilon_stage2:
    reject_structured_use_stage1
else:
    continue_structured_sampling
```

论文表述上，只有 `accept_structured` 才能声明非均匀 expert/layer 表达带来稳定增益；否则只能声明 Stage 1 的幅值建模有效，Stage 2 仍在学习或被拒绝。

最终输出：

```text
stage1_response.json: 一维幅值响应模型和高收益区间
stage2_residual_gain.json: structured candidate 相对统一幅值的增益
convergence.json: 当前是否收敛、继续扩展幅值还是继续采样 residual
summary.json: posterior mean/std、mixture components、best checkpoint
```

当前实现位置：

```text
/root/zero_prior_gmm_cem_search.py
/root/run_zero_prior_gmm_cem_eval.sh
```

---

## 3. 输入与输出

### 3.1 输入

必需输入：

```yaml
base_checkpoint: path/to/base
expert_checkpoints:
  - path/to/expert_task_1
  - path/to/expert_task_2
  - path/to/expert_task_3
merge_direction: task_arithmetic   # or ties / dare / custom_delta
calibration_data: path/to/calib.jsonl
proxy_objective: path/to/proxy_impl.py
```

可选输入：

```yaml
feature_type: policy_logits        # policy_logits / action_logits / hidden / value / custom
alpha_min: 0.05
alpha_max: 8.0
log_step: 0.693147                 # default log(2), i.e. multiply/divide by 2
num_initial_neighbors: 2           # alpha_center * exp(k * log_step), k=-2,-1,0,1,2
max_expand_steps: 3
balance_mode: mean_min             # mean / mean_std / mean_min
bootstrap: true
num_bootstrap: 100
score_higher_is_better: true
```

### 3.2 输出

输出 JSON：

```json
{
  "alpha_energy": 1.72,
  "alpha_grid_best": 2.00,
  "alpha_center": 1.88,
  "alpha_low": 1.00,
  "alpha_high": 3.50,
  "beta_low": -0.631,
  "beta_high": 0.621,
  "boundary_hit": false,
  "scores": [
    {"alpha": 0.50, "score": 0.61, "score_lcb": 0.55},
    {"alpha": 1.00, "score": 0.72, "score_lcb": 0.67},
    {"alpha": 2.00, "score": 0.81, "score_lcb": 0.77},
    {"alpha": 4.00, "score": 0.69, "score_lcb": 0.61}
  ],
  "diagnostics": {
    "energy_unit": 0.83,
    "energy_expert": 2.45,
    "energy_ratio_at_center": 1.03,
    "best_source": "parabolic_fit"
  }
}
```

输出 CSV：

```text
alpha,log_alpha,score,score_lcb,score_std,energy,energy_ratio,boundary_flag
```

---

## 4. 阶段一：构造保结构合并方向

### 4.1 基础 task vector

对每个 expert：

```text
tau_i = theta_i - theta_base
```

### 4.2 合并方向

第一版建议支持三种模式。

#### 模式 A：sum

```text
Delta = sum_i tau_i
```

#### 模式 B：mean

```text
Delta = mean_i tau_i
```

#### 模式 C：external

直接读取已有合并方向：

```text
Delta = theta_merged_direction - theta_base
```

例如已有 TIES/DARE/custom merge 的结果，则不要重新拆方向，直接把它当成固定方向。

### 4.3 重要原则

不要在该阶段做复杂剪枝、mask 或 shared/unique decomposition。当前方法的目的就是测试：

```text
在尽量保持方向结构不变的情况下，仅校准 alpha 是否能显著提升性能。
```

---

## 5. 阶段二：Functional Energy Matching

### 5.1 定义 feature function

定义模型在校准输入上的功能输出：

```text
f_theta(x)
```

推荐优先级：

1. RL Agent：`policy_logits` 或 `action_logits`。
2. 有 critic：可额外记录 `value_head`。
3. LLM Agent：可使用 action-token logits、tool-call logits。
4. 无法访问 logits：使用最后一层 hidden state。

### 5.2 unit merged energy

先构造 unit-gain 模型：

```text
theta_unit = theta_base + Delta
```

计算：

```text
E_unit = mean_{x in D_calib} || f_theta_unit(x) - f_theta_base(x) ||_2^2
```

### 5.3 expert target energy

对每个 expert 计算：

```text
E_expert_i = mean_{x in D_i} || f_theta_i(x) - f_theta_base(x) ||_2^2
```

总体 target：

```text
E_expert = mean_i E_expert_i
```

如果 calibration data 不区分任务，则直接在混合数据上计算所有 expert 的平均位移。

### 5.4 alpha_energy

在线性近似下：

```text
E(alpha) ≈ alpha^2 * E_unit
```

因此：

```text
alpha_energy = sqrt(E_expert / (E_unit + eps))
```

然后裁剪：

```text
alpha_energy = clip(alpha_energy, alpha_min, alpha_max)
```

### 5.5 fallback

如果出现以下情况：

```text
E_unit is NaN
E_unit is too small
feature extraction fails
no expert checkpoints available
```

则 fallback：

```text
alpha_energy = 1.0
```

并记录：

```json
{"energy_anchor_available": false}
```

---

## 6. 阶段三：Log-Scale Proxy Probe

### 6.1 构造候选点

以 `alpha_energy` 为中心，在 log-space 上采样：

```text
alpha_k = alpha_energy * exp(k * log_step)
```

其中：

```text
k in {-m, ..., -1, 0, 1, ..., m}
```

默认：

```text
m = 2
log_step = log(2)
```

即：

```text
{alpha_energy / 4, alpha_energy / 2, alpha_energy, 2 * alpha_energy, 4 * alpha_energy}
```

对每个候选点做裁剪和去重：

```text
alpha_k = clip(alpha_k, alpha_min, alpha_max)
unique sorted alpha values
```

### 6.2 代理目标接口

实现一个统一接口：

```python
def proxy_score(model, calibration_batch, task_id=None) -> float:
    """
    Returns higher-is-better score by default.
    If the raw proxy is a loss, convert it to score = -loss.
    """
```

如果每个任务有单独 score：

```python
def proxy_score_per_task(model, calibration_data_by_task) -> dict[int, float]:
    return {task_id: score_i}
```

### 6.3 多任务平衡 score

如果有任务级 score：

```text
s_i(alpha) = proxy score for task i
```

推荐先做归一化：

```text
z_i(alpha) = (s_i(alpha) - s_i(base)) / (s_i(expert_i) - s_i(base) + eps)
```

如果没有 base/expert 参考，直接使用 `s_i(alpha)`。

支持三种聚合：

#### mean

```text
S(alpha) = mean_i z_i(alpha)
```

#### mean_std

```text
S(alpha) = mean_i z_i(alpha) - lambda_std * std_i z_i(alpha)
```

#### mean_min

```text
S(alpha) = 0.5 * mean_i z_i(alpha) + 0.5 * min_i z_i(alpha)
```

第一版推荐：

```yaml
balance_mode: mean_min
```

因为它能避免某个任务被明显牺牲。

---

## 7. 阶段四：Adaptive Bracket Expansion

### 7.1 为什么需要 expansion

如果初始候选点里最优点在边界，例如：

```text
best alpha = max candidate alpha
```

说明 sweet spot 可能还在更大范围，需要继续扩展。

如果：

```text
best alpha = min candidate alpha
```

说明当前 gain 可能过大，需要向更小范围扩展。

### 7.2 expansion 规则

设当前候选集合为 sorted list：

```text
A = [a_1, a_2, ..., a_n]
```

如果最优是 `a_n`，则添加：

```text
a_new = min(a_n * expand_factor, alpha_max)
```

如果最优是 `a_1`，则添加：

```text
a_new = max(a_1 / expand_factor, alpha_min)
```

默认：

```yaml
expand_factor: 2.0
max_expand_steps: 3
```

### 7.3 停止条件

任一条件满足即停止：

```text
1. best alpha is not at boundary
2. reached alpha_min or alpha_max
3. max_expand_steps reached
4. new alpha duplicated with existing alpha
5. model evaluation fails or produces NaN
```

### 7.4 输出 bracket

如果最优点不是边界：

```text
alpha_low  = alpha before alpha_best
alpha_high = alpha after alpha_best
```

如果最优点仍在边界：

```text
boundary_hit = true
```

并设置保守 bracket：

```text
alpha_center = alpha_best
alpha_low  = max(alpha_best / expand_factor, alpha_min)
alpha_high = min(alpha_best * expand_factor, alpha_max)
```

注意：`boundary_hit=true` 表示搜索没有完全包住峰值，后续报告中必须标记。

---

## 8. 阶段五：Log-Parabolic Center Refinement

### 8.1 取三点

在 sorted candidates 中找到最优点 `alpha_best`，如果它左右都有邻居：

```text
alpha_left, alpha_best, alpha_right
```

令：

```text
x = log(alpha)
y = S(alpha)
```

拟合二次函数：

```text
q(x) = a x^2 + b x + c
```

### 8.2 解析解

如果三点等距，可用简单公式。更通用地，直接用最小二乘拟合三点：

```python
coef = np.polyfit(x_values, y_values, deg=2)
a, b, c = coef
```

如果：

```text
a < 0
```

则峰值：

```text
x_star = -b / (2a)
alpha_star = exp(x_star)
```

如果 `alpha_star` 在 bracket 内：

```text
alpha_center = alpha_star
best_source = "parabolic_fit"
```

否则：

```text
alpha_center = alpha_best
best_source = "grid_best"
```

如果：

```text
a >= 0
```

说明不是局部凹峰，使用：

```text
alpha_center = alpha_best
```

---

## 9. Bootstrap / LCB 处理噪声

如果 proxy 来自环境 rollout 或小样本 trajectory，score 方差会很大。建议支持 bootstrap lower confidence bound。

对每个 alpha 得到 per-sample 或 per-task score：

```text
scores_alpha = [score_1, score_2, ..., score_n]
```

做 bootstrap：

```text
S_mean = mean bootstrap aggregate score
S_std  = std bootstrap aggregate score
S_lcb  = S_mean - kappa * S_std
```

默认：

```yaml
bootstrap: true
num_bootstrap: 100
kappa: 1.0
```

搜索时使用：

```text
search_score = S_lcb
```

最终报告同时输出：

```text
score_mean, score_std, score_lcb
```

---

## 10. 后续优化器如何使用搜索结果

### 10.1 不要直接优化 alpha

直接优化：

```text
alpha
```

会让优化器受到尺度影响。推荐优化：

```text
beta = log(alpha / alpha_center)
```

即：

```text
alpha = alpha_center * exp(beta)
```

### 10.2 beta bounds

```text
beta_low  = log(alpha_low / alpha_center)
beta_high = log(alpha_high / alpha_center)
```

### 10.3 differentiable proxy

如果 proxy 可微：

```text
maximize S(theta_base + alpha_center * exp(beta) * Delta) - lambda_beta * beta^2
```

对应 loss：

```text
loss = -S + lambda_beta * beta^2
```

推荐：

```yaml
optimizer: Adam
lr: 0.03
steps: 20-50
lambda_beta: 0.01
```

每步后裁剪：

```python
beta.data.clamp_(beta_low, beta_high)
```

### 10.4 non-differentiable proxy / environment feedback

如果 proxy 不可微，不要用 Adam。使用局部 5 点或 7 点 residual search：

```text
beta candidates = linspace(beta_low, beta_high, 5 or 7)
```

选择 best LCB score。

---

## 11. Codex 实现结构

建议目录：

```text
task_vector_gain/
  __init__.py
  cli.py
  config.py
  checkpoint_io.py
  delta.py
  features.py
  energy.py
  proxy.py
  search.py
  bootstrap.py
  apply.py
  logging_utils.py
configs/
  gain_search.yaml
scripts/
  run_gain_search.sh
```

### 11.1 `checkpoint_io.py`

职责：

```text
- load model checkpoint
- load state_dict
- save merged checkpoint
- check parameter compatibility
```

关键函数：

```python
def load_state_dict(path: str) -> dict[str, torch.Tensor]: ...
def save_state_dict(state: dict[str, torch.Tensor], path: str) -> None: ...
def assert_compatible(base, expert) -> None: ...
```

### 11.2 `delta.py`

职责：

```text
- compute task vectors
- build merge direction Delta
- apply alpha to base model
```

关键函数：

```python
def compute_delta(base_sd, expert_sd):
    return {k: expert_sd[k] - base_sd[k] for k in base_sd.keys()}


def merge_deltas(deltas, mode="sum"):
    # mode: sum / mean / external
    ...


def apply_delta(base_sd, delta_sd, alpha: float):
    return {k: base_sd[k] + alpha * delta_sd[k] for k in base_sd.keys()}
```

注意：

```text
- 只处理 floating tensors。
- 非浮点参数直接复制 base。
- 如果 LoRA adapter，需要先 merge 到 full weights，或单独实现 adapter-space delta。
```

### 11.3 `features.py`

职责：

```text
- 在 calibration data 上抽取 functional output
```

接口：

```python
class FeatureExtractor:
    def __init__(self, feature_type: str): ...
    @torch.no_grad()
    def extract(self, model, batch) -> torch.Tensor: ...
```

支持：

```text
policy_logits
action_logits
hidden
value
custom
```

### 11.4 `energy.py`

职责：

```text
- 计算 E_unit
- 计算 E_expert
- 计算 alpha_energy
```

接口：

```python
def functional_energy(model_a, model_b, dataloader, feature_extractor) -> float:
    # mean ||f_a(x) - f_b(x)||^2
    ...


def estimate_alpha_energy(E_expert, E_unit, eps=1e-8, alpha_min=0.05, alpha_max=8.0):
    alpha = math.sqrt(E_expert / (E_unit + eps))
    return min(max(alpha, alpha_min), alpha_max)
```

### 11.5 `proxy.py`

职责：

```text
- 统一代理目标接口
- 将 loss 转换为 higher-is-better score
```

接口：

```python
class ProxyObjective:
    higher_is_better: bool = True

    def score(self, model, calib_data) -> dict:
        """
        Return:
        {
          "global": float,
          "per_task": {task_id: float},
          "per_sample": optional list[float]
        }
        """
```

### 11.6 `search.py`

职责：

```text
- 生成 alpha candidates
- evaluate candidates
- bracket expansion
- parabolic fit
- 输出 alpha_center/range
```

关键函数：

```python
def generate_log_candidates(center, log_step, m, alpha_min, alpha_max): ...
def evaluate_alpha(alpha, base_sd, delta_sd, model_builder, proxy): ...
def adaptive_expand(candidates, scores, direction, ...): ...
def fit_log_parabola(alphas, scores): ...
def search_gain_range(config) -> dict: ...
```

### 11.7 `apply.py`

职责：

```text
- 根据最终 alpha 保存合并模型
```

接口：

```python
def save_alpha_model(base_path, delta_path, alpha, output_path): ...
```

---

## 12. 搜索主流程伪代码

```python
def search_gain_range(cfg):
    # 1. Load base and experts
    base_sd = load_state_dict(cfg.base_checkpoint)
    expert_sds = [load_state_dict(p) for p in cfg.expert_checkpoints]

    # 2. Build Delta
    if cfg.merge_direction == "external":
        merged_sd = load_state_dict(cfg.external_merged_checkpoint)
        delta_sd = compute_delta(base_sd, merged_sd)
    else:
        deltas = [compute_delta(base_sd, e) for e in expert_sds]
        delta_sd = merge_deltas(deltas, mode=cfg.merge_direction)

    # 3. Estimate alpha_energy
    try:
        theta_unit_sd = apply_delta(base_sd, delta_sd, alpha=1.0)
        E_unit = compute_energy(theta_unit_sd, base_sd, cfg.calibration_data)
        E_expert = mean([
            compute_energy(expert_sd, base_sd, task_calib_data)
            for expert_sd, task_calib_data in zip(expert_sds, cfg.task_calibration_data)
        ])
        alpha_energy = sqrt(E_expert / (E_unit + cfg.eps))
        alpha_energy = clip(alpha_energy, cfg.alpha_min, cfg.alpha_max)
        energy_anchor_available = True
    except Exception:
        alpha_energy = 1.0
        energy_anchor_available = False

    # 4. Initial log candidates
    candidates = generate_log_candidates(
        center=alpha_energy,
        log_step=cfg.log_step,
        m=cfg.num_initial_neighbors,
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
    )

    # 5. Evaluate initial candidates
    results = {}
    for alpha in candidates:
        results[alpha] = evaluate_alpha(alpha, base_sd, delta_sd, cfg)

    # 6. Adaptive expansion
    boundary_hit = False
    for _ in range(cfg.max_expand_steps):
        best_alpha = select_best_alpha(results, use_lcb=cfg.bootstrap)
        sorted_alphas = sorted(results.keys())

        if best_alpha == sorted_alphas[-1]:
            new_alpha = min(best_alpha * cfg.expand_factor, cfg.alpha_max)
        elif best_alpha == sorted_alphas[0]:
            new_alpha = max(best_alpha / cfg.expand_factor, cfg.alpha_min)
        else:
            break

        if new_alpha in results or new_alpha == best_alpha:
            boundary_hit = True
            break

        results[new_alpha] = evaluate_alpha(new_alpha, base_sd, delta_sd, cfg)
    else:
        # loop exhausted
        best_alpha = select_best_alpha(results, use_lcb=cfg.bootstrap)
        sorted_alphas = sorted(results.keys())
        if best_alpha in [sorted_alphas[0], sorted_alphas[-1]]:
            boundary_hit = True

    # 7. Determine bracket
    sorted_alphas = sorted(results.keys())
    best_alpha = select_best_alpha(results, use_lcb=cfg.bootstrap)
    idx = sorted_alphas.index(best_alpha)

    if 0 < idx < len(sorted_alphas) - 1:
        alpha_low = sorted_alphas[idx - 1]
        alpha_high = sorted_alphas[idx + 1]
    else:
        alpha_low = max(best_alpha / cfg.expand_factor, cfg.alpha_min)
        alpha_high = min(best_alpha * cfg.expand_factor, cfg.alpha_max)
        boundary_hit = True

    # 8. Parabolic refinement
    alpha_center = best_alpha
    best_source = "grid_best"
    if 0 < idx < len(sorted_alphas) - 1:
        triplet = [sorted_alphas[idx - 1], sorted_alphas[idx], sorted_alphas[idx + 1]]
        triplet_scores = [results[a].search_score for a in triplet]
        alpha_fit = fit_log_parabola(triplet, triplet_scores)
        if alpha_fit is not None and alpha_low <= alpha_fit <= alpha_high:
            alpha_center = alpha_fit
            best_source = "parabolic_fit"

    # 9. Beta bounds
    beta_low = log(alpha_low / alpha_center)
    beta_high = log(alpha_high / alpha_center)

    # 10. Return summary
    return {
        "alpha_energy": alpha_energy,
        "alpha_grid_best": best_alpha,
        "alpha_center": alpha_center,
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
        "beta_low": beta_low,
        "beta_high": beta_high,
        "boundary_hit": boundary_hit,
        "energy_anchor_available": energy_anchor_available,
        "best_source": best_source,
        "scores": serialize_results(results),
    }
```

---

## 13. 配置文件示例

```yaml
seed: 42

device: cuda
precision: bf16

base_checkpoint: /path/to/base
expert_checkpoints:
  - /path/to/expert_1
  - /path/to/expert_2
  - /path/to/expert_3

merge_direction: mean
external_merged_checkpoint: null

calibration_data: /path/to/calibration.jsonl
task_calibration_data:
  0: /path/to/task_0_calib.jsonl
  1: /path/to/task_1_calib.jsonl
  2: /path/to/task_2_calib.jsonl

feature_type: policy_logits

alpha_min: 0.05
alpha_max: 8.0
log_step: 0.69314718056
num_initial_neighbors: 2
expand_factor: 2.0
max_expand_steps: 3

proxy:
  name: custom_xme_proxy
  module_path: /path/to/proxy_impl.py
  higher_is_better: true
  batch_size: 8

balance_mode: mean_min
lambda_std: 0.5

bootstrap: true
num_bootstrap: 100
kappa: 1.0

output_dir: /path/to/gain_search_outputs
save_best_model: true
```

---

## 14. CLI 设计

### 14.1 搜索 gain range

```bash
python -m task_vector_gain.cli search \
  --config configs/gain_search.yaml
```

### 14.2 只计算 functional energy

```bash
python -m task_vector_gain.cli energy \
  --config configs/gain_search.yaml
```

### 14.3 用指定 alpha 保存模型

```bash
python -m task_vector_gain.cli apply \
  --base /path/to/base \
  --delta /path/to/delta.pt \
  --alpha 1.88 \
  --output /path/to/merged_alpha_1.88
```

### 14.4 使用搜索结果保存 center 模型

```bash
python -m task_vector_gain.cli apply-from-result \
  --result /path/to/gain_search_result.json \
  --which alpha_center \
  --output /path/to/merged_center
```

---

## 15. 最小可运行版本要求

第一版只需要实现：

```text
1. load base/expert checkpoints
2. compute Delta by mean or sum
3. apply alpha
4. evaluate proxy score
5. log-grid search + adaptive expansion
6. parabolic fit
7. output JSON/CSV
```

可以暂时不实现：

```text
- layer-wise gain
- task-wise residual gain
- full environment rollout
- complex mask
- shared/unique decomposition
```

---

## 16. 推荐默认策略

默认配置建议：

```yaml
merge_direction: mean
alpha_min: 0.05
alpha_max: 8.0
log_step: log(2)
num_initial_neighbors: 2
expand_factor: 2.0
max_expand_steps: 3
balance_mode: mean_min
bootstrap: true
kappa: 1.0
```

默认搜索点大约是：

```text
alpha_energy / 4
alpha_energy / 2
alpha_energy
2 * alpha_energy
4 * alpha_energy
```

如果无法计算 `alpha_energy`，则退化为围绕 `alpha=1` 搜索：

```text
0.25, 0.5, 1.0, 2.0, 4.0
```

---

## 17. 异常与失败处理

### 17.1 alpha 越大 score 越高，没有下降

说明搜索区间没有覆盖过表达区域。

处理：

```text
- 继续扩展直到 alpha_max。
- 如果仍然 boundary_hit=true，则不要声称找到 sweet spot。
- 输出 best boundary alpha，并建议提高 alpha_max 或换更强 proxy。
```

### 17.2 alpha 越小 score 越高

说明当前 Delta 可能过强，或者方向本身不适合。

处理：

```text
- 向 alpha_min 扩展。
- 若 best 接近 0，说明该合并方向可能无效。
```

### 17.3 proxy score 很噪

处理：

```text
- 使用 paired calibration samples。
- 使用 bootstrap LCB。
- 增加每个 alpha 的重复评估。
- 减小 parabolic fit 权重，直接使用 grid best。
```

### 17.4 functional energy 与 proxy 不一致

这不是错误。`alpha_energy` 只是初始化 anchor，不是最终答案。

记录：

```text
energy_ratio_at_best
proxy_best_alpha
```

如果长期不一致，可以在报告中说明：

```text
functional energy provides a useful initialization but not a sufficient objective.
```

### 17.5 模型出现 NaN 或输出异常

处理：

```text
- 跳过该 alpha。
- 将该 alpha 标记为 invalid。
- 如果大 alpha 经常 invalid，自动降低 alpha_max。
```

---

## 18. 实验报告建议

每次运行至少画三张图。

### 18.1 Gain-response curve

```text
x-axis: alpha, log scale
y-axis: proxy score / environment score
```

目的：证明 gain sensitivity。

### 18.2 Functional energy curve

```text
x-axis: alpha, log scale
y-axis: E(alpha) / E_expert
```

目的：说明 alpha 控制 task-vector-induced functional energy。

### 18.3 Score vs energy ratio

```text
x-axis: E(alpha) / E_expert
y-axis: proxy score / environment score
```

目的：观察最佳性能是否落在中等 energy band。

---

## 19. 推荐消融实验

### 19.1 初始化方式对比

```text
alpha=1
alpha from fixed grid best
Stage1 response recommended scale
Stage1 observed best scalar
Stage2 structured residual posterior
oracle full sweep
```

核心结果：

```text
Stage1 应该显著优于 alpha=1，并接近 oracle full sweep；
Stage2 只有在 residual_gain 为正且 bootstrap 稳定时才声明超过 Stage1。
```

### 19.2 搜索预算对比

```text
3 probes
5 probes
7 probes
full sweep
```

目的：证明少量 probe 足够找到好的初始范围。

### 19.3 RL vs SFT task vector 对比

同一 base model 下分别构造：

```text
Delta_RL
Delta_SFT
```

比较：

```text
alpha_star
GainSens = J(alpha_star) - J(1)
curve sharpness
parameter energy
functional energy
```

这个实验用于验证：

```text
RL-induced task vector 是否更需要 gain calibration。
```

---

## 20. 给 Codex 的实现提示词

可以把下面这段直接交给 Codex：

```text
Implement a Python package named task_vector_gain for task-vector gain initialization range search.

Core equation:
  theta(alpha) = theta_base + alpha * Delta

Requirements:
1. Load base and expert checkpoints as PyTorch state_dicts.
2. Compute task vectors tau_i = expert_i - base.
3. Build Delta using mode: sum, mean, or external merged checkpoint.
4. Implement apply_delta(base_sd, delta_sd, alpha), handling only floating tensors and copying non-floating tensors from base.
5. Implement functional energy:
   E(model_a, model_b) = mean ||f_a(x) - f_b(x)||^2 on calibration data.
   Feature extractor can be a pluggable interface.
6. Estimate alpha_energy = sqrt(E_expert / (E_unit + eps)), clipped to [alpha_min, alpha_max].
7. Generate log-scale candidates around alpha_energy:
   alpha_k = alpha_energy * exp(k * log_step), k=-m,...,m.
8. Evaluate each alpha using a pluggable ProxyObjective interface. Higher score is better.
9. Aggregate per-task scores using mean, mean_std, or mean_min.
10. Implement bootstrap LCB: mean - kappa * std.
11. If best alpha is at boundary, expand by multiply/divide expand_factor until peak is bracketed or max_expand_steps is reached.
12. Fit a quadratic in log(alpha) around the best grid point when left and right neighbors exist. If concave and inside bracket, use its vertex as alpha_center; otherwise use grid best.
13. Output JSON and CSV containing alpha_energy, alpha_grid_best, alpha_center, alpha_low, alpha_high, beta_low, beta_high, boundary_hit, all candidate scores, and diagnostics.
14. Provide CLI commands: search, energy, apply, apply-from-result.
15. Add unit tests for candidate generation, delta application, alpha_energy estimation, bracket expansion, and parabolic fit.
```

---

## 21. 最终方法摘要

最终使用方式：

```text
1. 给定 base 和多个 expert。
2. 构造固定合并方向 Delta。
3. 用 functional energy matching 得到 alpha_energy。
4. 在 alpha_energy 附近做少量 log-scale proxy probe。
5. 自适应扩展，找到包含局部最优的 alpha 区间。
6. 用 log-parabolic fit 得到 alpha_center。
7. 后续优化器只优化 beta，其中 alpha = alpha_center * exp(beta)。
```

最终公式：

```text
theta* = theta_base + alpha_center * exp(beta*) * Delta
```

其中：

```text
beta* is optimized inside [log(alpha_low / alpha_center), log(alpha_high / alpha_center)]
```

这使得优化器不再从任意的 `alpha=1` 出发，而是从一个由 functional energy 和少量 proxy feedback 共同确定的初始范围出发。

---

## 22. 当前 VGEL/Frontier-LCB Pipeline

当前实验已从一维共享 `alpha` 推进到每个 expert 一个全局表达系数：

```text
s = [s_tool, s_memory, s_code]
theta(s) = theta_base
         + s_tool   * (theta_tool   - theta_base)
         + s_memory * (theta_memory - theta_base)
         + s_code   * (theta_code   - theta_base)
```

核心原则不是寻找人工先验区间，而是在极小 calibration set 上学习一个“能力保持可行域”：

```text
z_{i,t} = verifier_t(y_i)                         # sample-level verified signal
r_t(s) = normalize_by_expert_gain(mean_i z_{i,t}) # base/expert calibrated capability
LCB_t(s) = mean_i z_{i,t} - kappa * stderr_i z_{i,t}
A(s) = min_t LCB_t(s) + epsilon * mean_t LCB_t(s)
```

其中 `min_t LCB_t(s)` 是主目标，后面的均值项只用于极小 tie-break。这个目标对应 “所有 expert 能力都不能塌掉” 的 frontier 保持，而不是让某一个任务的高分补偿另一个任务失败。

### 22.1 代理信号

当前推荐使用 `per_sample` verified signal，而不是只用 aggregate metric：

```text
tool:   score
memory: 0.7 * F1-like score + 0.3 * correctness
code:   0.7 * pass-rate score + 0.3 * correctness
```

然后使用 expert-gain 归一化：

```text
g_t(s) = (r_t(s) - r_t(base)) / (r_t(expert_t) - r_t(base))
```

这样不同任务 reward 尺度可比，目标表达的是“保留了多少对应 expert 能力”。

### 22.2 分布建模与搜索

Stage 1 学习三维响应面：

```text
p_good(s) proportional exp(A(s) / tau)
```

实现上先使用 RBF/KDE 风格的局部响应模型和 CEM/GMM elite posterior：

```text
mu, Sigma <- weighted elite samples under exp(A/tau)
s_next   <- trust-region sample around observed good mass
```

为避免把模型不确定性误当成收益，acquisition 使用 conservative LCB + trust-region 惩罚：

```text
U(s) = E[A(s)] - beta * Std[A(s)] - lambda * d(s, S_observed) / h
```

也就是说，只允许在已有观测支撑附近探索；如果 RBF 在远离观测点的位置给出高分，不直接评测该外推点。早期曾用 UCB 做探索，但 `[0.6,1.2586,1.2]` 这类高不确定点在 verified 评测中明显退化，说明小样本下应把 response surface 解释为可行域下界估计，而不是乐观收益预测。

### 22.3 收敛判据

可以声明 Stage 1 收敛需要同时满足：

```text
1. best observed point 不在当前 trust-region 边界；
2. expected improvement < epsilon；
3. high-reward region width < width_epsilon；
4. worst-task LCB 的排序在最近一轮不再改变主瓶颈任务。
```

Stage 2 只有在结构化/分层参数带来正的 residual gain 且 LCB 大于 0 时才声明有效：

```text
Gain_residual = A(s_structured) - E_stage1[A(s_global)]
Accept if LCB(Gain_residual) > 0
```

### 22.4 当前实证结论

在 16-sample calibration 上，当前 best 为：

```text
s = [0.6, 1.6, 1.2]
```

它优于 `[0.8, 1.6, 1.2]` 的原因不是平均 reward 更高，而是 code 作为 bottleneck 的 LCB 提升，同时 tool LCB 没有下降。`[0.8, 1.6, 1.4]` 被拒绝，因为更大的 code 系数同时降低 code LCB 和 memory LCB，说明出现了跨任务干扰。

下一步完整评测使用 eval6 roadmap：

```text
model_name = vgel-frontier-lcb-060-160-120
model_path = /tmp/shared-storage/AgentMerging_plan/experiments/zero_prior_gmm_cem/vgel-frontier-lcb-local-refine-s16-20260510/candidates/zp-00-02-expert-0p600-1p600-1p200/model
```

eval6 完整评测已完成并追加到总表：

```text
Tool mean:     0.4567
Memory F1:     0.7041
Memory SubEM:  0.5801
Code pass mean:0.0889
CURE BoN mean: 0.1749
```

Stage2 residual posterior 的当前观察：expert-global `[0.65,1.55,1.2]` 在 16-sample Frontier-LCB 上为 `0.700649`，与 `[0.6,1.6,1.2]` 的 `0.700588` 基本不可区分；一个结构化样本 `zp-01-03-cem` 的 raw Frontier-LCB 达到 `0.737654`，但相对 Stage1 expert-global surface 的 residual gain 为 `-0.0018`，residual LCB 为 `-0.4683`，因此按方法论不接受为“结构化残差显著增益”。论文叙事上应优先强调 Stage1 的三维 capability frontier，Stage2 作为需要正 residual LCB 才接受的扩展，而不是强行宣称非均匀层系数已经带来收益。

## 23. Source-Reward 信号对齐修正

参考 `/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo` 后，当前 proxy 的主要问题是旧 `per_sample` 信号仍混合了诊断指标，而不是专家训练时实际优化的 reward。已将评测 harness 扩展为在 `predictions.jsonl` 中额外写入：

```text
source_reward
source_reward_norm
source_success
source_reward_name
source_reward_details
```

三类任务的 source reward 定义如下：

```text
Tool:   ToolRL strict format_reward + raw tool_call_correctness_reward, raw range [-3, 4],
        search 使用 source_reward_norm = (reward + 3) / 7。
Memory: MemAgent trajectory-level reward，先让模型逐 chunk 更新 memory，再用最终 memory 生成 final answer；
        reward 只支付给最终回答，即 final response 最后 300 字符中的最后一个 boxed exact match。
Code:   CURE code-side reward，即 ground-truth tests pass rate，GRPO 归一化前的任务 reward。
```

搜索器新增：

```text
--verified-signal source_reward
```

启动脚本 `/root/run_zero_prior_gmm_cem_eval.sh` 已默认使用该信号。这样 Stage1/Stage2 的响应面建模将基于“官方 RL 训练 reward 的归一化观测”，而不是旧的 `0.7 * F1/pass-rate + 0.3 * correct` 混合 proxy。旧 `score/accuracy` 仍保留用于报告和兼容已有结果。

这一步的理论意义是把 calibration feedback 从 benchmark-summary proxy 改为 per-sample source reward：

```text
r_source(y, x) = R_train_expert(y | x)
g_t = normalize_to_expert_gain_or_ratio(r_source)
A(s) = balanced/frontier aggregation over task-wise g_t
```

后续应重新跑三维 Stage1 surface；如果 `source_reward` 下找到的高分区仍不能迁移到完整 BFCL/Memory/CURE，则问题不再是 reward wrapper，而是 calibration 子集本身没有覆盖完整评测分布。
