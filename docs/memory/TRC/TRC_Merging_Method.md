# Trajectory Residual Calibrated Merging：一种面向多 RL Agent 的轻量可学习合并方法

## 1. 核心问题

给定同一个 base model 上训练出的多个 RL agent，例如 coding agent、memory agent、tool agent，每个 agent 都可以表示为：

\[
\theta_i = \theta_0 + \Delta_i
\]

其中 \(\theta_0\) 是 base model，\(\Delta_i\) 是第 \(i\) 个 agent 的 task vector。目标是构造一个 merged model：

\[
\theta_{merge} = \theta_0 + \sum_i \alpha_i \Delta_i
\]

但直接设定固定系数，例如所有 \(\alpha_i = 1\) 或所有 \(\alpha_i = 0.75\)，虽然简单，却无法解释不同 layer、module、token trajectory 上的能力保留与干扰。因此，本方法将 merge 过程参数化为一个轻量可学习问题：**不再手工设计 coding、memory、tool 各自的代理目标，而是让 merged model 在专家成功轨迹上对齐 expert 相对 base 的隐状态残差。**

核心思想是：

> RL agent 的 task vector 不应只被理解为参数空间中的静态差值，而应被理解为 expert 在成功执行任务轨迹时，相对于 base model 产生的行为残差。多 agent merging 的目标，就是学习一组轻量 merge gates，使 merged model 能够重现这些成功行为残差，同时避免在普通数据上产生过大的 base drift。

---

## 2. 方法命名

可以暂定为：

**Trajectory Residual Calibrated Merging，简称 TRC-Merging。**

也可以更简洁地写为：

**TR-Merging：Trajectory Residual Merging。**

方法定位：

- 不做 full finetuning；
- 不做额外 RL；
- 不需要为每个 agent 设计 task-specific reward；
- 不需要显式标注 coding span、memory span、tool span；
- 不做 unique/shared/conflict 参数划分；
- 所有 task vector 区域都通过可学习 gates 进行校准。

---

## 3. 参数化 merge 形式

对每个 agent 的 task vector \(\Delta_i\)，按 layer 或 module 分组。记第 \(g\) 个参数组为 \(\Delta_{i,g}\)，其中 \(g\) 可以是：

- layer-level group；
- attention / MLP module-level group；
- q_proj、k_proj、v_proj、o_proj、up_proj、down_proj、gate_proj 等更细粒度 group。

merged model 定义为：

\[
\theta(\phi) = \theta_0 + \sum_i \sum_g \alpha_{i,g} \Delta_{i,g}
\]

其中：

- \(\alpha_{i,g}\) 是可学习 merge gate；
- \(\phi = \{\alpha_{i,g}\}\) 是全部可学习参数；
- base model、expert models、task vectors 本身全部冻结；
- 训练时只更新这些少量 gate 参数。

最简单的版本可以使用 layer-wise gates：

\[
\theta(\phi) = \theta_0 + \sum_i \sum_l \alpha_{i,l} \Delta_{i,l}
\]

这比全局 \(\alpha_i\) 更有表达力，但仍然足够简单，不会让方法显得过度复杂。

---

## 4. Calibration 数据

对每个 expert agent，收集少量成功轨迹：

\[
D_i = \{\tau_i^1, \tau_i^2, ..., \tau_i^n\}
\]

其中每条轨迹可以包括：

- 用户输入；
- 中间推理；
- tool call；
- memory read / update；
- code generation；
- observation；
- final answer。

但方法本身不需要知道这些 token 属于哪一种能力。所有轨迹都只被当作普通 token sequence 处理。

同时准备一个 general instruction calibration set：

\[
D_{base}
\]

用于约束 merged model 不要在普通指令分布上过度偏离 base model。

---

## 5. Expert-base 隐状态残差

对第 \(i\) 个 expert，在其成功轨迹 \(\tau \in D_i\) 上做 teacher forcing。分别将同一条轨迹输入 base model 和 expert model，记录每一层、每个 token 位置的 hidden state：

\[
h_0^{l,t}, \quad h_i^{l,t}
\]

定义 expert 相对 base 的 trajectory residual：

\[
r_i^{l,t} = h_i^{l,t} - h_0^{l,t}
\]

这个 residual 表示：在该成功轨迹上，expert 为了完成任务，相对于 base 产生了怎样的隐空间行为偏移。

对 merged model，同样定义：

\[
r_{merge}^{l,t}(\phi) = h_{merge}^{l,t}(\phi) - h_0^{l,t}
\]

merge 的目标不是让 merged model 完全模仿 expert hidden state，而是让它模仿 expert 相对 base 的行为残差。

---

## 6. 自动选择关键轨迹位置

为了避免人工设计 task span，需要自动给不同 token/layer 位置赋权重。一个简单有效的权重可以由两部分组成：

### 6.1 Residual magnitude

\[
s_{res}^{l,t} = \|r_i^{l,t}\|_2
\]

如果 expert 相对 base 在某个位置产生较大 hidden shift，说明该位置可能承载了 agent-specific behavior。

### 6.2 Log-prob gain

在 teacher forcing 下，计算 expert 和 base 对真实轨迹 token \(y_t\) 的 log-prob：

\[
s_{lp}^{t} = \log p_i(y_t \mid y_{<t}) - \log p_0(y_t \mid y_{<t})
\]

如果 expert 比 base 更有信心生成该 token，说明该 token 更可能对应成功行为。

### 6.3 综合权重

可以定义：

\[
w_i^{l,t} = \text{Normalize}\left(\|r_i^{l,t}\|_2 + \eta \cdot \max(0, s_{lp}^{t})\right)
\]

实际实现中，为了简单，可以只保留 top-k 位置：

\[
w_i^{l,t} = 1 \quad \text{if } (l,t) \in \text{TopK}(s^{l,t})
\]

否则：

\[
w_i^{l,t} = 0
\]

这样，方法自动关注 expert 相对 base 变化最大、且对成功 token 更有信心的位置，而不需要显式判断这是 coding、tool 还是 memory 行为。

---

## 7. Residual alignment objective

对每个 agent 的成功轨迹，优化 merged model 的 residual，使其接近对应 expert 的 residual：

\[
\mathcal{L}_{res}
= 
\sum_i \sum_{\tau \in D_i} \sum_{l,t}
w_i^{l,t}
\left\|
r_{merge}^{l,t}(\phi) - r_i^{l,t}
\right\|_2^2
\]

这个目标表达的是：

> 在 expert 真正表现出成功行为的位置，merged model 应该产生与 expert 相似的 base-relative hidden shift。

它不是直接拟合 expert 输出，也不是为每个任务构造 reward，而是对齐 expert 相对 base 的成功行为残差。

---

## 8. Base drift regularization

如果只对齐 expert residual，merged model 可能在普通指令场景中过度激活 agent behavior。因此加入 base drift penalty。

在 general instruction set \(D_{base}\) 上，要求 merged model 接近 base model：

\[
\mathcal{L}_{base}
= 
\sum_{\tau \in D_{base}} \sum_{l,t}
\left\|
h_{merge}^{l,t}(\phi) - h_0^{l,t}
\right\|_2^2
\]

这个约束的作用是：

- 保持 general instruction following；
- 抑制 tool over-calling；
- 抑制 memory over-writing；
- 抑制 coding behavior 泄漏到普通对话；
- 防止多个 agent vector 同时注入造成整体分布漂移。

---

## 9. Gate regularization

为了避免 gates 学出极端值，可以加入一个简单的先验正则项：

\[
\mathcal{L}_{gate}
= 
\sum_{i,g} \left\|\alpha_{i,g} - \alpha_0\right\|_2^2
\]

其中 \(\alpha_0\) 可以设为：

- \(1.0\)：偏向完整保留每个 agent 的能力；
- \(0.75\)：偏向更稳健的多 agent 融合；
- 或者先用小规模 grid search 选择一个经验初值。

你当前的观察是：

- 所有 task vector 系数为 1 时，能力表达较充分；
- 所有 task vector 系数为 0.75 时，下游任务可能更稳。

因此可以把 \(\alpha_0 = 0.75\) 作为稳健初始化，也可以把 \(\alpha_0 = 1.0\) 作为能力保留初始化。更推荐的设置是：

\[
\alpha_{i,g}^{init} = 0.75
\]

然后通过 residual calibration 自动学习哪些 agent、哪些 layer、哪些 module 应该回到接近 1，哪些应该保持 shrink。

---

## 10. 总体优化目标

最终目标函数为：

\[
\mathcal{L}
= 
\mathcal{L}_{res}
+ \beta \mathcal{L}_{base}
+ \gamma \mathcal{L}_{gate}
\]

其中：

- \(\mathcal{L}_{res}\)：保留 expert 成功行为残差；
- \(\mathcal{L}_{base}\)：抑制普通指令分布上的漂移；
- \(\mathcal{L}_{gate}\)：防止 merge gates 过度偏离初始化。

训练时只更新 \(\alpha_{i,g}\)，不更新模型主体参数。

---

## 11. 训练流程

### Step 1：准备模型

给定：

\[
\theta_0, \theta_1, \theta_2, ..., \theta_N
\]

计算每个 expert 的 task vector：

\[
\Delta_i = \theta_i - \theta_0
\]

### Step 2：构建可学习 merged model

初始化：

\[
\alpha_{i,g} = \alpha_0
\]

构建：

\[
\theta(\phi) = \theta_0 + \sum_i \sum_g \alpha_{i,g} \Delta_{i,g}
\]

### Step 3：缓存 expert-base residual

对每个 expert 的成功轨迹做 teacher forcing，缓存：

\[
r_i^{l,t} = h_i^{l,t} - h_0^{l,t}
\]

同时计算位置权重：

\[
w_i^{l,t}
\]

这些可以离线完成，避免训练时反复跑所有 expert。

### Step 4：优化 merge gates

冻结 base model、expert models 和 task vectors，只更新 \(\alpha_{i,g}\)。

每个 batch 包含：

- 来自各 expert 的成功轨迹，用于 \(\mathcal{L}_{res}\)；
- general instruction data，用于 \(\mathcal{L}_{base}\)。

### Step 5：得到最终 merged model

训练结束后，将 gates 固化到参数中：

\[
\theta_{final} = \theta_0 + \sum_i \sum_g \alpha_{i,g}^{*} \Delta_{i,g}
\]

推理时不需要额外模块，也不需要动态 routing。

---

## 12. 极简版本

如果希望方法更简单，可以只学习每个 agent 每一层的 gate：

\[
\alpha_{i,l}
\]

那么参数量约为：

\[
N_{agents} \times N_{layers}
\]

例如 3 个 agents、32 层模型，只需要学习 96 个 scalar。这种设置非常适合作为主实验版本，因为它有三个优点：

1. 方法足够简单；
2. 不依赖任务类型；
3. 比全局 \(\alpha\) 更能处理不同层的干扰。

可以把更细粒度的 module-level gates 作为 ablation。

---

## 13. 方法优势

### 13.1 不需要手工代理目标

不需要分别定义：

- coding objective；
- memory objective；
- tool-use objective。

所有任务统一为：

> expert-base trajectory residual alignment。

### 13.2 不需要显式 behavior span

RAIN-Merging 中 thinking 和 instruction 的 behavior span 比较容易定义，但多 agent merging 中，coding、memory、tool 的边界并不干净。TRC-Merging 避免了这个问题，直接在 hidden space 中自动寻找 expert 相对 base 的关键变化位置。

### 13.3 不做 full training

只学习少量 gates，计算成本低，不改变原始 task vector。

### 13.4 能解释 1.0 与 0.75 的现象

- \(\alpha=1.0\)：能力表达充分，但 shared behavior drift 可能较大；
- \(\alpha=0.75\)：冲突更小，但部分能力可能被削弱；
- learned gates：让不同 agent、不同 layer、不同 module 自动选择更合适的表达强度。

---

## 14. 论文中的一句话表述

可以把方法概括为：

> We formulate multi-RL-agent merging as a trajectory residual calibration problem. Instead of designing task-specific proxy objectives for coding, memory, or tool-use agents, we learn lightweight merge gates over all task-vector components, such that the merged model reproduces each expert's base-relative hidden-state residuals on successful trajectories while remaining close to the base model on general instruction data.

中文表述：

> 我们将多 RL Agent 合并建模为轨迹残差校准问题。不同于为 coding、memory、tool-use 等 agent 分别设计任务代理目标，我们在所有 task vector 组件上学习轻量 merge gates，使合并模型在专家成功轨迹上复现 expert 相对 base 的隐状态残差，同时在通用指令数据上保持接近 base model。

---

## 15. 推荐实验设计

### Baselines

- Task Arithmetic；
- average merging；
- fixed coefficient merging，\(\alpha = 1.0\)；
- fixed coefficient merging，\(\alpha = 0.75\)；
- TIES / DARE 类方法；
- RAM-style RL agent merging；
- output distillation calibration。

### Main evaluation

分别评估：

- coding benchmark；
- tool-use benchmark；
- memory benchmark；
- mixed agent trajectory benchmark；
- general instruction benchmark。

### Ablation

- 去掉 \(\mathcal{L}_{res}\)；
- 去掉 \(\mathcal{L}_{base}\)；
- 去掉 \(\mathcal{L}_{gate}\)；
- residual magnitude only；
- log-prob gain only；
- layer-wise gates vs module-wise gates；
- \(\alpha_0 = 1.0\) vs \(\alpha_0 = 0.75\)。

---

## 16. 最终核心贡献

TRC-Merging 的核心贡献可以写成三点：

1. **Problem reformulation**：将多 RL agent merging 从 task-specific proxy objective 设计问题，转化为 task-agnostic trajectory residual calibration 问题。
2. **Lightweight learnable merging**：在所有 task vector 组件上学习少量 merge gates，不做 full finetuning，也不依赖 unique/shared/conflict 人工划分。
3. **Behavior-preserving calibration**：通过 expert-base hidden residual alignment 保留专家成功轨迹中的行为能力，并通过 base drift regularization 抑制普通指令分布上的能力泄漏和干扰。

---

## 17. 最简算法伪代码

```text
Input:
  Base model θ0
  Expert models {θi}
  Successful trajectories {Di}
  General instruction set Dbase

Compute:
  Δi = θi - θ0

Initialize:
  αi,g = α0
  θ(φ) = θ0 + Σi Σg αi,g Δi,g

Offline cache:
  For each expert i and trajectory τ in Di:
    Run base θ0 and expert θi with teacher forcing
    Compute residual ri(l,t) = hi(l,t) - h0(l,t)
    Compute weight wi(l,t) by residual norm and/or log-prob gain

Optimize:
  Freeze θ0, {θi}, {Δi}
  Update only αi,g by minimizing:
    L = Lres + β Lbase + γ Lgate

Output:
  θfinal = θ0 + Σi Σg α*i,g Δi,g
```

---

## 18. 一句话总结

**TRC-Merging 用专家成功轨迹中的 hidden residual 作为统一代理目标，学习所有 task vector 组件的 merge gates，从而在不手工定义 agent-specific objectives 的情况下，实现多 RL agent 能力的轻量、可泛化合并。**
