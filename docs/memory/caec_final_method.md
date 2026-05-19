# Compatibility-Aware Expert Composition (CAEC)

**版本**: final-method-v0.3  
**日期**: 2026-05-18  
**定位**: 面向 cross-scale / weakly compatible expert composition 的训练-free 模型合并与安全能力注入方法  
**核心结论**: compatible experts 用 Fisher-regression shrinkage QP 求全局可解释系数与区间；incompatible experts 不做全局合并，而是在 activation-normalized mode basis 上用 directional utility + protected Fisher harm 求稀疏安全注入系数。

---

## 0. 一句话方法

给定一个 base model 和多个专家模型，我们先判断每个 expert 是否与 base/已有 experts **全局兼容**。兼容 experts 使用 Fisher-regression QP 求可解释的全局系数：

$$
\theta_{\mathrm{C}} = \theta_0 + \sum_{e \in \mathcal C}\alpha_e \tau_e,
$$

其中 $\alpha_e$ 可以直接解释为 Task Arithmetic 系数，并可输出近最优安全区间。对于 R1 这类 delta scale 极端不匹配或 base/tokenizer/config 不完全同源的 incompatible expert，不求全局 $\alpha_{\mathrm{R1}}$，而是拆成 mode：

$$
\theta_{\mathrm{final}}
=
\theta_0
+
\sum_{e \in \mathcal C}\alpha_e \tau_e
+
\sum_{j \in \mathcal I}\gamma_j \widetilde b_j,
$$

其中 $\widetilde b_j$ 是 activation-normalized mode，$\gamma_j$ 由 target utility 和 protected harm 的 sparse QP 求出。

这不是“又一个 weighted averaging”。它解决的是：**当某个 expert 有用但不能安全全局合并时，如何只抽取安全、有用的局部能力。**

---

## 1. 最终方法名与 paper claim

### 1.1 推荐方法名

**CAEC: Compatibility-Aware Expert Composition**

子模块：

1. **FRS: Fisher-Regression Shrinkage**  
   用于 compatible experts 的全局系数求解。

2. **UHMS: Utility/Harm Mode Selection**  
   用于 incompatible experts 的 mode-level 稀疏安全注入。

整体可写作：

> CAEC decomposes expert composition into globally mergeable compatible experts and locally injectable incompatible experts. Compatible experts are merged by Fisher-regression shrinkage, while incompatible experts are injected through sparse utility/harm mode selection under a protected-task trust region.

### 1.2 最终 claim

推荐主 claim：

> We study cross-scale expert composition, where some experts are globally compatible with the base model while others are useful but unsafe to merge globally. We propose CAEC, a compatibility-aware framework that merges compatible experts via Fisher-regression shrinkage and extracts capabilities from incompatible experts through activation-normalized utility/harm mode selection.

中文版本：

> 我们研究 cross-scale expert composition：一部分 experts 可以作为常规 task vectors 全局合并，另一部分 experts 虽然有能力但由于尺度、基座或参数语义不兼容，不能直接全局合并。CAEC 用 Fisher-regression shrinkage 求 compatible experts 的可解释系数，用 utility/harm sparse mode injection 从 incompatible experts 中安全抽取能力。

### 1.3 不建议写的 claim

不要写：

- “我们首次提出 model merging。”不对。
- “我们首次学习 merging coefficient。”不对，AdaMerging 已经学习 task-wise/layer-wise coefficients。
- “我们首次用 Fisher 做 merging。”不对，Fisher Merging 和 EWC 都早已有相关思想。
- “我们的 v1 已经证明 continuous coefficient optimization。”不稳，因为当前 R1 modes 全部被 clip 到同一个 $z_{\max}$ 时，实验证明的主要是 selection，而不是连续系数细粒度差异。
- “统一 QP 一定自然恢复 0.75。”除非使用本文的 Fisher-regression QP 并完成 coefficient recovery 实验。

---

## 2. 问题设置

### 2.1 模型与 task vectors

设 base model 参数为：

$$
\theta_0 \in \mathbb R^d.
$$

有 $E$ 个专家模型：

$$
\theta_e, \quad e=1,\ldots,E.
$$

若 expert $e$ 与 base 同源、同 tokenizer/config、fine-tuning 起点一致或足够接近，则定义 classic task vector：

$$
\tau_e = \theta_e - \theta_0.
$$

对于 compatible experts，Task Arithmetic 假设基本成立：

$$
\theta_{\mathrm{merge}} = \theta_0 + \sum_e \alpha_e \tau_e.
$$

但是，对 incompatible expert，例如 reasoning distill model，如果它的真实起点不是 $\theta_0$，或者 tokenizer/config 有变化，或者 delta norm 极端大，那么

$$
\theta_e - \theta_0
$$

不再是严格意义上的 task vector。此时全局系数 $\alpha_e$ 的语义不可靠。

### 2.2 Compatible vs incompatible expert

将 experts 分为两类：

$$
\mathcal C = \{e: e \text{ is globally compatible}\},
$$

$$
\mathcal I = \{e: e \text{ is useful but globally unsafe}\}.
$$

判断标准包括：

1. **参数同源性**：是否来自同一 base、同一 tokenizer、同一 config。
2. **delta scale**：$\|\theta_e-\theta_0\|/\|\theta_c-\theta_0\|$ 是否处在同一量级。
3. **activation response scale**：expert delta 对 layer outputs 的 RMS perturbation 是否异常。
4. **Fisher risk**：$\tau_e^\top F_{\mathrm{protected}}\tau_e$ 是否远高于 compatible experts。
5. **small-alpha eval curve**：$\theta_0 + \alpha\tau_e$ 在 $\alpha \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}$ 是否快速 collapse。

在用户当前文档中，ToolRL、MemoryAgent、ReasonFlux-Coder 的 per-param delta norm 约为 $0.077$，而 DeepSeek-R1-Distill 约为 $114.15$，相对大约 $1492\times$。这个现象说明 R1 不应被当作普通 compatible task vector 直接全局合并，而应作为 incompatible expert 做小步、稀疏、安全注入。

### 2.3 目标

我们要构造一个 merged model：

$$
\theta
=
\theta_0
+
\Delta\theta,
$$

使它同时满足：

1. 保持 base / instruction-following / formatting 先验；
2. 保持 compatible experts 的 Tool、Memory、Code 等能力；
3. 从 incompatible reasoning expert 中抽取对 target skill 有帮助的局部能力；
4. 避免因 scale mismatch 或参数冲突导致 collapse。

---

## 3. 最终模型参数化

最终参数化分三层：

$$
\Delta\theta
=
\underbrace{B_C \alpha}_{\text{compatible global coefficients}}
+
\underbrace{B_R r}_{\text{optional compatible residual modes}}
+
\underbrace{\widetilde B_I \gamma}_{\text{incompatible sparse injection}}.
$$

其中：

- $B_C=[\tau_1,\ldots,\tau_m]$，$m=|\mathcal C|$；
- $\alpha \in \mathbb R^m$ 是 compatible expert 的全局系数；
- $B_R$ 是 compatible experts 的 residual mode basis，可选；
- $r$ 是 residual correction 系数，通常很小；
- $\widetilde B_I=[\widetilde b_1,\ldots,\widetilde b_J]$ 是 incompatible experts 的 activation-normalized mode basis；
- $\gamma \in \mathbb R^J$ 是稀疏注入系数。

最小可用版本可去掉 residual：

$$
\theta_{\mathrm{final}}
=
\theta_0
+
B_C\alpha
+
\widetilde B_I\gamma.
$$

这就是推荐的最终版本。

---

## 4. Fisher/KL 局部理论基础

### 4.1 KL 二阶近似

设 task $c$ 的 calibration distribution 为 $\mathcal D_c$，模型输出分布为 $p_\theta(y|x)$。对小扰动 $\Delta\theta$，有：

$$
\mathrm{KL}\left(p_{\theta}(\cdot|x)\,\|\,p_{\theta+\Delta\theta}(\cdot|x)\right)
\approx
\frac12 \Delta\theta^\top F_c(\theta)\Delta\theta,
$$

其中 empirical Fisher：

$$
F_c(\theta)
=
\mathbb E_{x\sim \mathcal D_c, y\sim p_\theta(\cdot|x)}
\left[
\nabla_\theta \log p_\theta(y|x)
\nabla_\theta \log p_\theta(y|x)^\top
\right].
$$

实际实现中使用 diagonal Fisher：

$$
F_c \approx \mathrm{diag}(f_c),
$$

其中：

$$
f_{c,p}
=
\mathbb E\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_p}\right)^2\right].
$$

### 4.2 Fisher 的角色

在本文中，Fisher 不是用来问：

> 哪个 expert 在这个 parameter 上更值得相信？

而是问：

> 如果我沿某个 direction 改参数，它对 protected tasks 的输出分布有多危险？

因此 protected harm 定义为：

$$
\mathrm{harm}(v)
=
v^\top F_{\mathrm{protected}}v.
$$

如果 $v$ 落在 protected tasks 的高 Fisher 方向，说明它会大幅改变 protected tasks 的输出，应该被压小或拒绝。

---

## 5. Compatible experts：Fisher-Regression Shrinkage

这一节用于求 Tool/Memory/Code 这类 compatible experts 的全局系数 $\alpha$。重点是：**$\alpha$ 仍然是 Task Arithmetic 系数，因此可解释。**

### 5.1 一维 shrinkage 推导

先看一个 expert $e$：

$$
\theta(\alpha)=\theta_0+\alpha\tau_e.
$$

希望两个目标同时成立：

1. 靠近 expert，在 owner task 上获得能力；
2. 不要过度离开 base/protected tasks。

用二次近似写 owner fitting cost：

$$
J_{\mathrm{owner}}(\alpha)
=
\frac12 A_e(1-\alpha)^2.
$$

其中

$$
A_e=\tau_e^\top F_e\tau_e.
$$

$A_e$ 表示沿 $\tau_e$ 到达 expert 对 owner task 的函数空间意义有多大。

protected harm：

$$
J_{\mathrm{prot}}(\alpha)
=
\frac12 B_e\alpha^2,
$$

其中

$$
B_e=\tau_e^\top F_{\mathrm{prot}}\tau_e.
$$

总目标：

$$
J(\alpha)
=
\frac12 A_e(1-\alpha)^2
+
\frac12 B_e\alpha^2.
$$

求导：

$$
\frac{\partial J}{\partial \alpha}
=
-A_e(1-\alpha)+B_e\alpha.
$$

令导数为 0：

$$
-A_e+A_e\alpha+B_e\alpha=0.
$$

得到：

$$
\boxed{
\alpha_e^*=
\frac{A_e}{A_e+B_e}
}
$$

这个公式非常重要。它解释了为什么某些 expert 的优秀系数可能是 $0.5$ 或 $0.75$。

如果：

$$
B_e=A_e,
$$

则：

$$
\alpha_e^*=0.5.
$$

如果：

$$
B_e=\frac13 A_e,
$$

则：

$$
\alpha_e^*=0.75.
$$

所以，$0.5$ 和 $0.75$ 不应该被讲成 magic numbers。它们应该来自 owner benefit curvature 和 protected harm curvature 的比值。

### 5.2 多 expert QP

设 compatible experts 为：

$$
\mathcal C=\{1,\ldots,m\},
$$

$$
B_C=[\tau_1,\ldots,\tau_m] \in \mathbb R^{d\times m}.
$$

merged update：

$$
\Delta\theta_C=B_C\alpha.
$$

对每个 task $c\in\mathcal C$，我们希望 $B_C\alpha$ 在 task $c$ 的 Fisher metric 下接近 $\tau_c$：

$$
D_c(\alpha)
=
\frac12
(B_C\alpha-\tau_c)^\top
F_c
(B_C\alpha-\tau_c).
$$

再加 base retention：

$$
R_0(\alpha)
=
\frac{\rho}{2}(B_C\alpha)^\top F_0(B_C\alpha).
$$

总目标：

$$
\boxed{
J_C(\alpha)
=
\frac12\sum_{c\in\mathcal C}w_c
(B_C\alpha-\tau_c)^\top F_c(B_C\alpha-\tau_c)
+
\frac{\rho}{2}(B_C\alpha)^\top F_0(B_C\alpha)
+
\frac{\lambda_\alpha}{2}\|\alpha\|_2^2
}
$$

约束：

$$
0\le \alpha_e\le \alpha_{\max}.
$$

推荐 $\alpha_{\max}\in[1.0,1.25]$，避免过度 extrapolation。

### 5.3 展开为标准 QP

展开 $J_C(\alpha)$：

$$
J_C(\alpha)
=
\frac12\alpha^\top M\alpha-r^\top\alpha+\mathrm{const},
$$

其中：

$$
\boxed{
M=
\sum_{c\in\mathcal C}w_c B_C^\top F_c B_C
+
\rho B_C^\top F_0 B_C
+
\lambda_\alpha I
}
$$

$$
\boxed{
r=
\sum_{c\in\mathcal C}w_c B_C^\top F_c\tau_c
}
$$

无约束闭式解：

$$
\boxed{
\alpha^*=M^{-1}r
}
$$

有 box constraint 时解：

$$
\boxed{
\alpha^*=
\arg\min_{0\le\alpha\le\alpha_{\max}}
\frac12\alpha^\top M\alpha-r^\top\alpha
}
$$

这是一个低维 convex QP。若只有 Tool、Memory、Code 三个 experts，则 $m=3$，求解非常稳定。

### 5.4 diagonal Fisher 下如何计算 $M$ 和 $r$

使用 diagonal Fisher $f_c\in\mathbb R^d$：

$$
(B_C^\top F_c B_C)_{ij}
=
\sum_{p=1}^{d} f_{c,p}\tau_{i,p}\tau_{j,p}.
$$

$$
(B_C^\top F_c\tau_c)_i
=
\sum_{p=1}^{d} f_{c,p}\tau_{i,p}\tau_{c,p}.
$$

因此不需要构造 $d\times d$ 的 Fisher matrix。只需逐参数 streaming 累加即可。

### 5.5 coefficient interval

只给 $\alpha^*$ 单点不够。我们要输出“优秀系数区间”。

令最优点附近可接受 objective tolerance 为 $\eta$：

$$
J_C(\alpha)\le J_C(\alpha^*)+\eta.
$$

因为 $J_C$ 是二次函数：

$$
J_C(\alpha)-J_C(\alpha^*)
=
\frac12(\alpha-\alpha^*)^\top M(\alpha-\alpha^*).
$$

所以近最优集合是椭球：

$$
\mathcal S_\eta
=
\left\{\alpha:
\frac12(\alpha-\alpha^*)^\top M(\alpha-\alpha^*)\le \eta
\right\}.
$$

如果不考虑 box constraint，第 $i$ 个系数的解析区间是：

$$
\boxed{
I_i(\eta)
=
\left[
\alpha_i^*-
\sqrt{2\eta(M^{-1})_{ii}},
\alpha_i^*+
\sqrt{2\eta(M^{-1})_{ii}}
\right]
}
$$

加 box 后：

$$
\boxed{
I_i(\eta)
=
\left[
\max\left(0,\alpha_i^*-
\sqrt{2\eta(M^{-1})_{ii}}\right),
\min\left(\alpha_{\max},\alpha_i^*+
\sqrt{2\eta(M^{-1})_{ii}}\right)
\right]
}
$$

$\eta$ 的选择有三种方式：

1. **bootstrap variance**：对 calibration samples bootstrap，求 $\alpha^*$ 的分布，用 5/95 percentile 作为区间。
2. **eval tolerance**：选择使 smoke eval 不下降超过阈值的最大 $\eta$。
3. **relative objective tolerance**：例如 $\eta=0.01\cdot J_C(0)$ 或 $0.01\cdot |J_C(\alpha^*)|$，再用 eval 校准。

推荐实际使用 bootstrap interval，因为它更容易解释。

### 5.6 这个 QP 如何解释旧的 grid result

如果 QP 输出：

$$
\alpha_{\mathrm{tool}}\in[0.45,0.60],
$$

$$
\alpha_{\mathrm{mem}}\in[0.68,0.82],
$$

$$
\alpha_{\mathrm{code}}\in[0.65,0.85],
$$

那么旧的 $(0.50,0.75,0.75)$ 就不再是 grid search artifact，而是 Fisher/KL 局部近似下的近最优 shrinkage 区间。

如果 QP 没输出类似区间，说明：

1. Fisher 估计点不对；
2. calibration 数据不代表最终 eval；
3. loss normalization 不对；
4. task vector 兼容性假设不成立；
5. 线性/二阶近似不够。

这时不要强行宣称 QP 恢复了 0.75，而应回到诊断。

---

## 6. Compatible residual correction，可选

全局 $\alpha$ 只能给每个 expert 一个整体系数，但不同 layer/module 可能需要细微差异。可以加入 residual modes：

$$
\Delta\theta_C=B_C\alpha+B_Rr.
$$

其中 $B_R$ 可以由 compatible experts 的 layer/module modes 构成：

$$
b_{e,p}=\tau_e[p].
$$

残差目标：

$$
\min_r
\frac12\sum_c w_c
(B_C\alpha+B_Rr-\tau_c)^\top F_c
(B_C\alpha+B_Rr-\tau_c)
+
\frac\rho2(B_C\alpha+B_Rr)^\top F_0(B_C\alpha+B_Rr)
+
\lambda_r\|r\|_2^2
+
\lambda_{r,1}\|r\|_1.
$$

约束：

$$
|r_j|\le r_{\max}.
$$

推荐默认先不启用 residual。原因：

1. $\alpha$ 已经能解释主要贡献；
2. residual mode 数量多，容易 calibration overfit；
3. 如果加入 residual，要证明它替代了 conflict gate，而不是另一个手工 trick。

如果启用 residual，它应该作为 **soft conflict gate**：符号冲突、高 protected Fisher、owner utility 低的 modes 会自动被压小。

---

## 7. Incompatible experts：Utility/Harm Mode Selection

这一节用于 R1 这类不能安全全局合并的 expert。

### 7.1 为什么不能求 $\alpha_{\mathrm{R1}}$

如果 R1 不是从同一个 $\theta_0$ fine-tune 而来，或者 tokenizer/config/base 有差异，则

$$
\theta_{\mathrm{R1}}-\theta_0
$$

包含：

$$
\underbrace{\theta_{\mathrm{R1}}-\theta_{\mathrm{R1-base}}}_{\text{reasoning distillation}}
+
\underbrace{\theta_{\mathrm{R1-base}}-\theta_0}_{\text{base mismatch}}
+
\underbrace{\text{tokenizer/config/alignment difference}}_{\text{semantic mismatch}}.
$$

此时全局 $\alpha_{\mathrm{R1}}$ 没有稳定语义。即使 $\alpha$ 很小，也可能触碰大量 protected high-Fisher directions。

因此 incompatible expert 只允许通过 mode-level small injection：

$$
\Delta\theta_I=\sum_j\gamma_j\widetilde b_j.
$$

### 7.2 Mode 定义

按 expert、layer、module 拆 mode：

$$
b_j=\delta_e[p],
$$

其中 $p$ 可以是：

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`

对 28 层 Qwen-like decoder，若每层 7 个 module，则每个 expert 有：

$$
28\times 7=196
$$

个 modes。

### 7.3 Activation normalization

原始 weight norm 在不同 layer/module 之间不可比，也会被 R1 的 extreme scale gap 主导。对每个 linear layer $W\in\mathbb R^{d_{out}\times d_{in}}$，mode delta 为 $\Delta W_j$，输入 activation 为 $h\in\mathbb R^{d_{in}}$。

该 mode 对 layer output 的均方扰动：

$$
\mathbb E\|\Delta W_j h\|^2
=
\mathbb E[h^\top\Delta W_j^\top\Delta W_jh]
=
\mathrm{tr}(\Delta W_j^\top\Delta W_j\Sigma_h),
$$

其中：

$$
\Sigma_h=\mathbb E[hh^\top].
$$

用 diagonal activation covariance 近似：

$$
\Sigma_h\approx\mathrm{diag}(\mathbb E[h_i^2]).
$$

则：

$$
s_j^2
=\sum_i \mathbb E[h_i^2]\|\Delta W_j[:,i]\|_2^2.
$$

定义 normalized mode：

$$
\boxed{
\widetilde b_j=\frac{b_j}{s_j+\epsilon}
}
$$

物理含义：$\gamma_j=1$ 表示注入一个单位 RMS output perturbation 的 mode。实际 $\gamma_j$ 会远小于 1。

### 7.4 Target utility

在当前 merge 点：

$$
\theta_*=\theta_0+B_C\alpha
$$

或包含 residual 的版本：

$$
\theta_*=\theta_0+B_C\alpha+B_Rr.
$$

定义 target skill loss：

$$
L_T(\theta)
$$

例如 Code expert imitation KL、R1 reasoning/code alignment KL，或 supervised code loss。

沿 mode $\widetilde b_j$ 做小扰动：

$$
\theta(\gamma_j)=\theta_*+\gamma_j\widetilde b_j.
$$

一阶展开：

$$
L_T(\theta_*+\gamma_j\widetilde b_j)
\approx
L_T(\theta_*)+
\gamma_j\left\langle
\nabla_\theta L_T(\theta_*),\widetilde b_j
\right\rangle.
$$

定义 utility：

$$
\boxed{
u_j
=-\left\langle
\nabla_\theta L_T(\theta_*),\widetilde b_j
\right\rangle
}
$$

解释：

- $u_j>0$：沿该 mode 走会降低 target loss，有帮助；
- $u_j<0$：沿该 mode 走会升高 target loss，有害；
- $|u_j|$ 大：该 mode 对 target skill 有强方向性影响。

这和只看 gradient magnitude 不同。gradient magnitude 只表示当前位置敏感不敏感；directional utility 表示“沿 expert mode 这个方向走是否正确”。

### 7.5 Protected harm

设 protected tasks 包括 Tool、Memory、base instruction、formatting 等。定义 protected Fisher：

$$
F_P
=
\sum_{c\in\mathcal P}\pi_cF_c.
$$

注入多个 modes：

$$
\Delta\theta_I=\widetilde B_I\gamma.
$$

protected KL 二阶近似：

$$
\mathrm{KL}_{P}
\approx
\frac12\gamma^\top G_P\gamma,
$$

其中：

$$
\boxed{
G_P=\widetilde B_I^\top F_P\widetilde B_I
}
$$

对角近似：

$$
\boxed{
G_{P,jj}=\widetilde b_j^\top F_P\widetilde b_j
}
$$

这个 $G_{jj}$ 才是严格 QP 里的二次 harm。不要把

$$
\max(\langle \nabla L_{\mathrm{protected}}, b_j\rangle,0)
$$

直接放到 denominator 当 harm。后者是一阶 protected-loss projection，可以作为线性 penalty 或约束，但不是二阶 curvature。

### 7.6 Sparse utility/harm QP

对 incompatible modes，求解：

$$
\boxed{
\gamma^*
=
\arg\min_\gamma
-u^\top\gamma
+
\frac12\gamma^\top(G_P+\lambda_2I)\gamma
+
\lambda_1\|\gamma\|_1
}
$$

可加 safety constraint：

$$
\boxed{
\gamma^\top G_P\gamma\le 2\epsilon_P
}
$$

以及 hard budget：

$$
\|\gamma\|_0\le K,
$$

$$
|\gamma_j|\le \gamma_{\max,j}.
$$

### 7.7 diagonal closed-form

如果使用 diagonal $G_P$，即 $G_P=\mathrm{diag}(h_j)$，则：

$$
\boxed{
\gamma_j^*
=
\frac{\mathrm{soft}(u_j,\lambda_1)}{h_j+
\lambda_2}
}
$$

其中：

$$
\mathrm{soft}(u,\lambda)
=
\mathrm{sign}(u)\max(|u|-\lambda,0).
$$

加 clipping：

$$
\boxed{
\gamma_j^*
=
\mathrm{clip}\left(
\frac{\mathrm{soft}(u_j,\lambda_1)}{h_j+\lambda_2},
-\gamma_{\max,j},
\gamma_{\max,j}
\right)
}
$$

若使用 per-mode Fisher safety cap：

$$
\frac12 h_j\gamma_j^2\le \epsilon_j,
$$

则：

$$
\boxed{
|\gamma_j|\le \sqrt{\frac{2\epsilon_j}{h_j+\delta}}
}
$$

最终可写：

$$
\boxed{
\gamma_j^*
=
\mathrm{clip}\left(
\frac{\mathrm{soft}(u_j,\lambda_1)}{h_j+\lambda_2},
-
\sqrt{\frac{2\epsilon_j}{h_j+\delta}},
\sqrt{\frac{2\epsilon_j}{h_j+\delta}}
\right)
}
$$

再从非零候选中按 predicted improvement 取 top-$K$：

$$
\Delta J_j
=
-u_j\gamma_j+
\frac12h_j\gamma_j^2+
\lambda_1|\gamma_j|.
$$

选择 $\Delta J_j$ 最负的 $K$ 个 modes。

### 7.8 什么时候需要 full/block $G_P$

diagonal $G_P$ 忽略 mode 间互相干扰。建议分三档：

1. **diagonal**：默认，成本最低；适合初始验证。
2. **block diagonal by layer**：同一层 modes 建 $G$ 的小矩阵，可捕捉 layer 内冲突。
3. **top-candidate full $G$**：先用 diagonal 选 top 128，再对候选构造 full $G$，求 constrained QP。

推荐最终 paper 至少做一个 diagonal vs block/full ablation，证明 diagonal 近似不是偶然。

---

## 8. 联合目标：最终理论形式

完整 CAEC 可写为：

$$
\boxed{
\begin{aligned}
\min_{\alpha,r,\gamma}\quad
&\frac12\sum_{c\in\mathcal C}w_c
(\Delta\theta-\tau_c)^\top F_c(\Delta\theta-\tau_c)
+\frac\rho2\Delta\theta^\top F_0\Delta\theta\\
&-u_I^\top\gamma
+\frac12\gamma^\top G_I\gamma
+\frac{\lambda_\alpha}{2}\|\alpha\|_2^2
+\frac{\lambda_r}{2}\|r\|_2^2
+\lambda_{r,1}\|r\|_1
+\lambda_{\gamma,1}\|\gamma\|_1
\end{aligned}
}
$$

其中：

$$
\Delta\theta=B_C\alpha+B_Rr+\widetilde B_I\gamma.
$$

约束：

$$
0\le\alpha\le\alpha_{\max},
$$

$$
|r_j|\le r_{\max},
$$

$$
\gamma^\top G_P\gamma\le2\epsilon_P,
$$

$$
\|\gamma\|_0\le K.
$$

但实际实现不建议一次性全量求解。推荐 sequential solve：

1. 先求 $\alpha$；
2. 再可选求 $r$；
3. 最后求 $\gamma$。

原因：

- $\alpha$ 低维、可解释、稳定；
- $r$ 高维，需强正则；
- $\gamma$ 来自 incompatible expert，必须在已经合并好的 $\theta_*$ 上重新计算 utility/harm。

---

## 9. 实际操作步骤

下面给出最终推荐 pipeline。

### Step 0：准备输入

输入：

```text
base_model: θ0
compatible_experts:
  - tool: θ_tool
  - memory: θ_mem
  - code: θ_code
incompatible_experts:
  - r1: θ_r1
calibration_data:
  - D_tool
  - D_memory
  - D_code
  - D_base / D_instruction / D_format
  - D_r1_target / D_code_reasoning
```

必须确认：

1. 所有 compatible experts 和 base 参数 shape 完全一致；
2. incompatible expert 参数 shape 可对齐；
3. tokenizer/config 差异被记录；
4. embedding/lm_head 是否参与 merge 要明确，默认建议先冻结或单独 audit。

### Step 1：compatibility audit

对每个 expert $e$ 计算：

```python
for expert in experts:
    delta = theta_expert - theta_base
    global_norm = ||delta||_2 / sqrt(num_params)
    per_module_norm[layer, module] = ||delta[layer,module]||_2 / sqrt(num_params_module)
    activation_rms[layer,module] = sqrt(E||DeltaW h||^2)
    fisher_risk[task] = delta^T F_task delta
```

输出表：

```text
expert | delta_norm | norm_ratio | activation_rms_ratio | fisher_risk_tool | fisher_risk_mem | fisher_risk_code | compatible?
```

判定规则建议：

```text
compatible if:
  tokenizer/config identical or semantically aligned
  norm_ratio <= 10x relative to compatible group
  fisher_risk not extreme
  small-alpha eval does not collapse

incompatible if:
  norm_ratio >> 10x, especially >100x
  or tokenizer/config/base mismatch
  or small-alpha eval collapses
```

对 R1 这种 $\sim1000\times$ scale gap，直接归入 incompatible。

### Step 2：估计 Fisher

在当前 base 或初始 merge 点估计 Fisher。推荐：

- compatible coefficient QP 的 $F_c$：可以先在 $\theta_0$ 上估计；如果最终效果不稳，再在 $\theta_0+B_C\alpha$ 上重估。
- incompatible injection 的 $F_P$：必须在 $\theta_* = \theta_0+B_C\alpha$ 上估计，因为注入发生在这个点。

Diagonal empirical Fisher 实现：

```python
model.zero_grad()
loss = cross_entropy_or_kl(model(batch), target)
loss.backward()
for name, param in model.named_parameters():
    fisher[name] += param.grad.detach() ** 2
fisher[name] /= num_batches
```

注意：

1. 对不同 task 的 loss scale 做 normalization；
2. Fisher 加 damping：

```python
fisher = fisher + fisher_damping
```

3. embedding/lm_head 的 Fisher 常常异常，先单独处理。

### Step 3：求 compatible global coefficients

构造：

```python
B = [tau_tool, tau_mem, tau_code]
```

Streaming 计算：

```python
M = zeros(m, m)
r = zeros(m)

for task c in compatible_tasks:
    F = fisher[c]
    for i in range(m):
        for j in range(m):
            M[i,j] += w[c] * sum(F[p] * tau[i][p] * tau[j][p])
        r[i] += w[c] * sum(F[p] * tau[i][p] * tau[c][p])

M += rho * B.T @ F_base @ B
M += lambda_alpha * I
```

求解：

```python
alpha_star = solve_box_qp(M, r, lower=0.0, upper=alpha_max)
```

若 $m=3$，可以用 scipy optimize：

```python
from scipy.optimize import minimize

def obj(alpha):
    return 0.5 * alpha @ M @ alpha - r @ alpha

def grad(alpha):
    return M @ alpha - r

res = minimize(
    obj,
    x0=np.ones(m) * 0.5,
    jac=grad,
    bounds=[(0.0, alpha_max)] * m,
    method="L-BFGS-B",
)
alpha_star = res.x
```

输出：

```json
{
  "alpha_tool": 0.50,
  "alpha_memory": 0.75,
  "alpha_code": 0.75,
  "objective": ...,
  "active_bounds": ...
}
```

注意：这里的 $0.50/0.75/0.75$ 是例子。最终必须以 QP 实际输出为准。

### Step 4：求 coefficient intervals

推荐 bootstrap：

```python
alphas = []
for b in range(B_bootstrap):
    D_b = bootstrap_resample(calibration_data)
    fisher_b = estimate_or_reweight_fisher(D_b)
    M_b, r_b = build_qp(fisher_b)
    alpha_b = solve_box_qp(M_b, r_b)
    alphas.append(alpha_b)

interval[i] = percentile(alphas[:, i], [5, 95])
```

同时用二次椭球解析区间：

```python
M_inv = np.linalg.inv(M + 1e-8 * np.eye(m))
for i in range(m):
    width = np.sqrt(2 * eta * M_inv[i,i])
    interval[i] = [max(0, alpha_star[i]-width), min(alpha_max, alpha_star[i]+width)]
```

最终报告两种区间：

```text
alpha_tool:   point=0.xx, bootstrap=[a,b], quadratic=[c,d]
alpha_memory: point=0.xx, bootstrap=[a,b], quadratic=[c,d]
alpha_code:   point=0.xx, bootstrap=[a,b], quadratic=[c,d]
```

如果 bootstrap 和 quadratic 差距很大，说明局部二次近似或 Fisher 估计不稳。

### Step 5：bake compatible merge

```python
for name in model_params:
    theta_merge[name] = theta_base[name]
    for e in compatible_experts:
        theta_merge[name] += alpha[e] * (theta_expert[e][name] - theta_base[name])
```

保存：

```text
/checkpoints/caec_stage1_alpha_merge/
```

跑 smoke eval：

```text
Tool small split
Memory small split
Code small split
Format sanity
Instruction-following sanity
```

若 Stage 1 明显差于 grid $(0.50,0.75,0.75)$，不要继续 R1 注入。先诊断 Fisher/QP。

### Step 6：可选 compatible residual correction

默认跳过。若需要替代 conflict gate，则：

1. 构造 compatible residual modes；
2. 计算 residual utility/harm；
3. 强正则、小步注入；
4. ablation 证明 residual 优于 hand-written conflict gate。

推荐先把 residual 作为附录实验，不作为主路径。

### Step 7：构造 incompatible mode basis

对 R1：

```python
modes = []
for layer in layers:
    for module in [q,k,v,o,gate,up,down]:
        b = theta_r1[layer,module] - theta_base[layer,module]
        modes.append((layer, module, b))
```

如果 R1 的 base/tokenizer/config 与 $\theta_0$ 不完全一致，必须在文档中写明：

```text
R1 delta is treated as a candidate direction library, not as a classical task vector.
```

即：它是 candidate mode bank，不是全局 task vector。

### Step 8：activation normalization

在 $\theta_*$ 或 $\theta_0$ 上跑 calibration forward，收集每个 linear layer input activation 的二阶矩：

```python
E_h2[name] = mean_over_tokens_and_samples(h ** 2)
```

对每个 mode：

```python
col_sq_norm = (delta_W ** 2).sum(dim=0)
s_sq = (E_h2[name] * col_sq_norm).sum()
s = sqrt(s_sq + eps)
b_tilde = delta_W / s
```

保存：

```json
mode_manifest.json
activation_norms.json
```

必须记录：

```text
mode_id
expert
layer
module
raw_norm
activation_norm_s
normalized_norm
```

### Step 9：在当前 merge 点计算 target utility

当前点：

$$
\theta_* = \theta_0+B_C\alpha.
$$

对 target data，例如 Code/R1 alignment：

```python
loss_target = KL(model_theta_star(batch), target_expert_logits)
loss_target.backward()
```

对每个 mode：

```python
u_j = -sum(grad[name] * b_tilde[name])
```

如果 mode 是一个 module 的整块矩阵，只对该矩阵做内积。

保存：

```json
utility_scores.json
```

### Step 10：计算 protected harm

protected tasks：

```text
Tool
Memory
Base instruction / format
possibly Code if R1 target is not exactly Code
```

在 $\theta_*$ 上估计 diagonal Fisher：

```python
F_prot = pi_tool * F_tool + pi_mem * F_mem + pi_base * F_base
```

对每个 mode：

```python
h_j = sum(F_prot[name] * b_tilde[name] ** 2)
```

保存：

```json
harm_scores.json
```

若实现 block/full $G$：

```python
G[i,j] = sum(F_prot[p] * b_tilde_i[p] * b_tilde_j[p])
```

只对 top candidates 构造即可。

### Step 11：求 sparse injection coefficients

Diagonal 版本：

```python
gamma = {}
for j in modes:
    raw = soft_threshold(u[j], lambda_1) / (h[j] + lambda_2)
    cap = sqrt(2 * eps_j / (h[j] + delta))
    gamma[j] = clip(raw, -cap, cap)
```

若加 top-$K$：

```python
predicted_gain[j] = -u[j] * gamma[j] + 0.5 * h[j] * gamma[j] ** 2 + lambda_1 * abs(gamma[j])
selected = argsort(predicted_gain)[:K]  # most negative
for j not in selected:
    gamma[j] = 0
```

Full/block QP 版本：

```python
minimize  -u.T @ gamma + 0.5 * gamma.T @ G @ gamma + lambda1 * ||gamma||_1
s.t.      gamma.T @ G @ gamma <= 2 * epsilon
          |gamma_j| <= gamma_max_j
```

可用 coordinate descent、proximal gradient、OSQP/CVXPy 处理候选 top-128 modes。

### Step 12：bake final model

```python
for name in model_params:
    theta_final[name] = theta_stage1[name]

for mode_id, gamma_j in selected_modes:
    name = mode.param_name
    theta_final[name] += gamma_j * b_tilde_j[name]
```

保存：

```text
/checkpoints/caec_final_alpha_plus_incompatible_modes/
```

同时保存 manifest：

```json
{
  "base_model": "...",
  "compatible_alpha": {...},
  "alpha_intervals": {...},
  "incompatible_modes": [
    {
      "mode_id": "r1.layer12.mlp.up_proj",
      "u": ...,
      "h": ...,
      "gamma": ...,
      "raw_norm": ...,
      "activation_norm": ...
    }
  ],
  "solver_config": {...}
}
```

### Step 13：eval 与安全验证

必须跑：

1. Tool full eval；
2. Memory full eval；
3. Code full eval；
4. formatting / refusal / chat template sanity；
5. generation smoke test；
6. perplexity 或 instruction-following sanity；
7. R1 target improvement eval。

如果 incompatible injection 提升 Code 但伤害 Tool/Memory，则调小：

- $\epsilon_P$；
- $K$；
- $\gamma_{\max}$；
- target/protected weights；
- 或改用 block/full $G$。

### Step 14：可选 sequential refinement

可做 1-2 轮 refinement：

```text
for round in [1, 2]:
    theta_current = bake(alpha, gamma)
    recompute target gradient at theta_current
    recompute protected Fisher at theta_current
    solve small local QP for delta_gamma
    update gamma with trust-region
```

不要做太多轮。多轮会开始拟合 calibration noise。

推荐 hard rule：

```text
max_rounds = 2
accept update only if smoke eval improves or protected metric unchanged
```

---

## 10. 推荐默认超参

| 超参 | 默认值 | 说明 |
|---|---:|---|
| $\alpha_{\max}$ | 1.25 | compatible expert 最大外推系数 |
| $\lambda_\alpha$ | $10^{-5}$ 到 $10^{-3}$ | QP 数值稳定 |
| $\rho$ | 0.1 到 1.0 | base retention 权重 |
| $w_c$ | task-balanced | 每个 compatible task 的 fitting 权重 |
| Fisher damping | $10^{-8}$ 到 $10^{-5}$ | 防止 zero Fisher |
| bootstrap B | 100 | coefficient interval |
| $K$ | 32/64/96/128 | incompatible selected modes 数 |
| $\lambda_1$ | by sparsity target | 控制 mode 稀疏 |
| $\lambda_2$ | 1.0 或 calibrated | trust regularization |
| $\epsilon_P$ | from protected KL budget | 全局 safety budget |
| $\gamma_{\max}$ | Fisher cap 或 sweep | 防止单 mode 过大 |
| refinement rounds | 0-2 | 避免 overfit |

实际 sweep 顺序：

1. 固定 $\alpha$，只 sweep $K$ 与 $\gamma_{\max}$；
2. 再 sweep $\epsilon_P$；
3. 最后调 task weights $w_c,\pi_c$。

不要一开始同时调所有超参。

---

## 11. 必做 ablation

### 11.1 证明 compatible coefficient solver 有用

| 实验 | 目的 |
|---|---|
| TA-1/3 | 常规平均系数 baseline |
| TA-0.75 | 旧强 baseline |
| grid best | 说明 grid 上界 |
| FRS-QP point | 证明 QP 能接近或超过 grid |
| FRS-QP interval samples | 证明区间内稳定 |
| wrong Fisher point | 证明 Fisher 估计点重要 |
| no base retention | 看是否破坏 format/base |

关键图：

```text
alpha_tool / alpha_mem / alpha_code 的 bootstrap distribution
```

如果 QP 的区间覆盖 $(0.50,0.75,0.75)$，这是非常强的证据。

### 11.2 证明 incompatible mode selection 有用

| 选择策略 | 目的 |
|---|---|
| random K modes | 排除随机注入也有效 |
| top raw norm K | 排除只是选大 delta |
| bottom raw norm K | 检查小 delta 是否安全但无效 |
| ReasonAny-style bottom gradient | 直接比较 low-gradient mask |
| top gradient magnitude | 比较敏感参数选择 |
| utility only | 检查 protected harm 是否必要 |
| harm only | 检查 target utility 是否必要 |
| utility/harm full | 主方法 |
| diagonal G vs block/full G | 检查 harm 近似质量 |

### 11.3 证明 continuous/safety coefficient 有用

当前若所有 $\gamma_j$ 都被 clip 到同一个上限，只能证明 selection 有用。必须补：

```text
γ_max ∈ {0.0005, 0.001, 0.003, 0.005, 0.01}
K ∈ {32, 64, 96, 128}
```

画 Pareto frontier：

```text
x-axis: protected score drop
 y-axis: target score gain
```

如果放松 cap 后 $\gamma$ 出现非均匀分布，并且 utility/harm full 的 frontier 优于 random/ReasonAny-style，就能证明 continuous coefficient 有价值。

### 11.4 scale-gap controlled experiment

人为缩放 incompatible delta：

$$
\delta_{I}^{(s)}=s\cdot\frac{\|\tau_{\mathrm{agent}}\|}{\|\delta_I\|}\delta_I,
$$

其中：

$$
s\in\{1,10,100,1000,10000\}.
$$

比较：

1. naive TA；
2. ReasonAny-style binary mask；
3. CAEC UHMS continuous injection。

预期：

- scale gap 小时，差异不大；
- scale gap 中等，naive TA 开始退化；
- scale gap 极大，binary mask 也可能过量注入；
- CAEC 通过 Fisher cap 和 continuous $\gamma$ 保持稳定。

这张图是证明“为什么需要 CAEC”的核心图。

### 11.5 base/tokenizer/config audit

尤其对 R1，必须报告：

1. tokenizer vocab size / special tokens；
2. config 差异；
3. embedding/lm_head 是否参与；
4. per-module delta norm；
5. R1 minus true R1-base vs R1 minus current base 的差异；
6. activation RMS perturbation 分布。

否则 reviewer 可以说 extreme scale gap 是 checkpoint mismatch artifact。

---

## 12. 结果报告模板

### 12.1 coefficient solver

```text
Compatible coefficient QP:
  alpha_tool   = 0.xx, bootstrap [0.xx, 0.xx], quadratic [0.xx, 0.xx]
  alpha_memory = 0.xx, bootstrap [0.xx, 0.xx], quadratic [0.xx, 0.xx]
  alpha_code   = 0.xx, bootstrap [0.xx, 0.xx], quadratic [0.xx, 0.xx]

Interpretation:
  Tool protected harm / owner benefit ≈ ... → shrinkage ≈ ...
  Memory protected harm / owner benefit ≈ ... → shrinkage ≈ ...
  Code protected harm / owner benefit ≈ ... → shrinkage ≈ ...
```

### 12.2 incompatible injection

```text
Selected R1 modes:
  total candidates: 196
  selected: 64
  mean gamma: ...
  max |gamma|: ...
  protected Fisher budget used: ... / epsilon
  target predicted improvement: ...

Top modes by utility/harm:
  layer.module | u | h | gamma | predicted_gain
```

### 12.3 final eval

```text
Method | Tool | Memory F1 | Code Acc | Code BoN | Format sanity
Base
TA-1/3
TA-0.75
FRS-QP only
FRS-QP + random R1 modes
FRS-QP + ReasonAny-style R1 modes
FRS-QP + UHMS R1 modes
```

重点：不要只报最终最好模型；要报每一步为什么有必要。

---

## 13. Related work 定位

### 13.1 Task Arithmetic

Task Arithmetic 定义 task vector 为同一 pretrained model fine-tune 后权重与原权重的差，并展示 task vectors 可以加减组合以改变模型行为。CAEC 的 compatible expert 分支继承了这个思想，但不假设所有 experts 都是 globally mergeable。

### 13.2 TIES

TIES 指出 merging 中存在 redundant parameter values 和 sign disagreement 两类 interference，并通过 trim、elect sign、merge 处理。CAEC 不使用 hard sign voting 作为主方法，而是通过 Fisher-regression QP 和 optional residual 让冲突在二次目标中软化。

### 13.3 DARE

DARE 通过随机 drop delta parameters 再 rescale，利用 fine-tuned delta 的冗余，缓解多模型 merging 干扰。CAEC 的 sparsity 不是随机，而是由 target directional utility 和 protected Fisher harm 决定。

### 13.4 AdaMerging

AdaMerging 自动学习 task-wise 或 layer-wise merging coefficients。CAEC 的 compatible coefficient solver 也学习系数，但目标函数不同：它显式拟合各 expert 的 Fisher metric behavior，同时用 protected/base Fisher 做 shrinkage，并输出 coefficient interval。

### 13.5 Fisher Merging 与 EWC

Fisher Merging 用 Fisher 作为 posterior precision 做 weighted averaging。EWC 用 Fisher 表示旧任务重要参数并减缓遗忘。CAEC 使用 Fisher 作为 protected-task trust-region metric，用来衡量一个 candidate update 是否危险，而不是作为“相信哪个模型”的 averaging 权重。

### 13.6 RegMean

RegMean 把 model merging 写成 linear-layer regression，并用 input covariance 得到 closed-form merge。CAEC 的 activation normalization 与其在思想上接近：都承认 raw weight distance 不如 function/output perturbation 更有意义。但 CAEC 的回归发生在 task-vector/mode coefficient space，而不是直接每层求 merged weight。

### 13.7 ReasonAny

ReasonAny 处理 “Reasoning + X” model merging，并发现 reasoning capability 多在 low-gradient sensitivity regions。CAEC 与其最相关，但回答的问题不同：ReasonAny 主要判断 reasoning 参数在哪里，CAEC 判断每个 incompatible mode 是否应该注入、注入多少，并显式加入 protected Fisher harm 和 continuous coefficients。CAEC 更适合 extreme scale gap 和 multi-expert setting。

---

## 14. 创新点 outline

### 14.1 主创新点

**Innovation 1: Cross-scale expert composition problem formulation**

现有 merging 多默认 experts 至少大体同源、同尺度、可全局加权。CAEC 明确研究 mixed compatibility setting：一部分 experts 可以全局合并，另一部分 experts 有用但不能全局合并。

**Innovation 2: Compatibility-aware decomposition**

CAEC 不把所有 experts 放进同一个 coefficient vector，而是分成：

$$
\text{compatible global merge}
+
\text{incompatible local injection}.
$$

这解决了 R1 这类 scale-mismatched expert 污染全局 QP 的问题。

**Innovation 3: Fisher-regression shrinkage with coefficient intervals**

对 compatible experts，CAEC 不是 grid search，而是从 Fisher/KL 二阶近似推导出低维 QP：

$$
\alpha^*=\arg\min
\frac12\sum_cw_c(B\alpha-\tau_c)^\top F_c(B\alpha-\tau_c)
+\frac\rho2(B\alpha)^\top F_0(B\alpha).
$$

它不仅给点估计，还能给 coefficient interval：

$$
\alpha_i\in
\left[\alpha_i^*\pm\sqrt{2\eta(M^{-1})_{ii}}\right].
$$

这让 $0.5/0.75$ 这类优秀系数变成可解释的 shrinkage ratio，而不是经验 magic number。

**Innovation 4: Directional utility rather than gradient magnitude**

CAEC 对每个 candidate mode 计算：

$$
u_j=-\langle\nabla L_T,\widetilde b_j\rangle.
$$

它判断的是“沿 expert mode 这个方向走是否降低 target loss”。这比只看 $|\nabla L|$ 或 raw norm 更精确。

**Innovation 5: Protected Fisher harm as a safety metric**

CAEC 对每个 candidate mode 计算：

$$
h_j=\widetilde b_j^\top F_P\widetilde b_j.
$$

它把 Fisher 用作 safety/trust-region metric，而不是 averaging weight。

**Innovation 6: Activation-normalized mode basis for incompatible expert injection**

CAEC 使用：

$$
\widetilde b_j=b_j/(s_j+\epsilon),
$$

其中 $s_j$ 是该 mode 的 expected output RMS perturbation。这使不同 layer/module/expert 的注入系数可比，避免 raw delta scale 主导选择。

**Innovation 7: Sparse continuous injection under explicit trust region**

CAEC 的 incompatible branch 解：

$$
\gamma^*=
\arg\min_
\gamma
-u^\top\gamma
+\frac12\gamma^\top G_P\gamma
+\lambda_1\|\gamma\|_1,
$$

并加入：

$$
\gamma^\top G_P\gamma\le2\epsilon_P.
$$

这比 binary mask 或全局 $\alpha$ 更适合 extreme scale gap。

### 14.2 不是创新但可作为组成部分

以下不要当主创新：

1. Task vector / Task Arithmetic；
2. Fisher information；
3. model merging coefficient learning；
4. sign conflict 处理；
5. sparsifying delta；
6. activation/input-statistics-aware merging。

CAEC 的价值在于把这些元素重新组织到 cross-scale expert composition 这个问题里，并给出可求解、可诊断、可解释的 framework。

### 14.3 推荐 contribution 写法

**Contribution 1 — Problem**  
We identify and formalize cross-scale expert composition, where useful experts can be globally incompatible due to extreme delta scale or weak parameter homology.

**Contribution 2 — Method**  
We propose CAEC, which merges compatible experts through Fisher-regression shrinkage and injects incompatible experts through activation-normalized utility/harm mode selection.

**Contribution 3 — Theory**  
We derive compatible coefficients as a convex Fisher-regression QP and derive sparse incompatible injection as a protected KL trust-region problem, yielding coefficient intervals and safety budgets.

**Contribution 4 — Evidence**  
We show that CAEC improves multi-agent capabilities while preserving protected tasks, and we validate each component through selection, scale-gap, Fisher-point, and coefficient-interval ablations.

---

## 15. 最小可实现版本

如果时间有限，先实现以下 minimal CAEC：

```text
1. Compatibility audit
2. Diagonal Fisher for Tool/Memory/Code/Base
3. 3-expert FRS-QP → alpha + intervals
4. Bake alpha merge
5. Activation-normalized R1 modes
6. Utility on Code/R1 target
7. Protected Fisher harm on Tool/Memory/Base
8. Diagonal sparse UHMS with top-K and Fisher cap
9. Bake final
10. Full eval + core ablations
```

不要一开始做：

- full all-mode joint QP；
- many iterative rounds；
- complicated residual branch；
- per-token dynamic routing；
- online RL refinement。

先把最小版本做扎实。

---

## 16. 伪代码

```python
def caec_merge(base, compatible_experts, incompatible_experts, data, cfg):
    # 0. audit
    audit_report = compatibility_audit(base, compatible_experts, incompatible_experts, data)

    # 1. task vectors for compatible experts
    tau = {
        e: subtract_params(compatible_experts[e], base)
        for e in compatible_experts
    }

    # 2. Fisher for compatible coefficient QP
    fisher = {
        task: estimate_diag_fisher(base, data[task], cfg.fisher)
        for task in cfg.compatible_tasks + cfg.base_tasks
    }

    # 3. solve compatible alpha
    M, r = build_fisher_regression_qp(tau, fisher, cfg.weights, cfg.rho, cfg.lambda_alpha)
    alpha = solve_box_qp(M, r, lower=0.0, upper=cfg.alpha_max)
    alpha_intervals = bootstrap_alpha_intervals(base, tau, data, cfg)

    # 4. bake stage1
    theta_stage1 = add_task_vectors(base, tau, alpha)

    # 5. build incompatible mode basis
    modes = build_modes(incompatible_experts, base, cfg.mode_granularity)
    act_stats = collect_activation_second_moments(theta_stage1, data[cfg.activation_data])
    modes_norm = activation_normalize_modes(modes, act_stats)

    # 6. compute utility
    utility = compute_directional_utility(
        model=theta_stage1,
        modes=modes_norm,
        target_data=data[cfg.target_task],
        target_teacher=cfg.target_teacher,
    )

    # 7. compute protected harm
    fisher_prot = estimate_protected_fisher(theta_stage1, data, cfg.protected_tasks)
    harm = compute_mode_fisher_harm(modes_norm, fisher_prot)

    # 8. solve sparse injection
    gamma = solve_sparse_utility_harm_qp(
        utility=utility,
        harm=harm,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        K=cfg.K,
        epsilon=cfg.epsilon_protected,
    )

    # 9. bake final
    theta_final = inject_modes(theta_stage1, modes_norm, gamma)

    # 10. save manifest
    save_manifest({
        "audit": audit_report,
        "alpha": alpha,
        "alpha_intervals": alpha_intervals,
        "selected_modes": summarize_modes(modes_norm, utility, harm, gamma),
        "config": cfg,
    })

    return theta_final
```

---

## 17. 最终判断

最终版本不要再把 v2 写成“all modes, all experts 用同一个一阶 harm denominator 的 iterative QP”。那样理论不稳，也解释不了为什么会得到优秀系数区间。

最终版本应写成：

1. compatible experts：用 Fisher-regression shrinkage QP 求 $\alpha$ 与 interval；
2. incompatible experts：用 activation-normalized mode basis + directional utility + protected Fisher harm 求 sparse $\gamma$；
3. 两者 sequentially solved；
4. 可选 residual branch 作为 soft conflict gate；
5. 用 scale-gap 和 selection ablation 证明这个处理不是 engineering trick。

如果实验能证明：

- QP 的 $\alpha$ 区间覆盖或接近旧 grid best；
- UHMS 优于 random/raw-norm/ReasonAny-style/utility-only/harm-only；
- scale gap 越大，CAEC 相对优势越明显；
- protected Fisher budget 能预测 collapse boundary；

那这个方法的创新和可信度都足够强。

---

## 18. References

1. Ilharco et al. **Editing Models with Task Arithmetic**. arXiv:2212.04089, 2022.
2. Yadav et al. **TIES-Merging: Resolving Interference When Merging Models**. arXiv:2306.01708, 2023.
3. Yu et al. **Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch**. arXiv:2311.03099, 2023.
4. Yang et al. **AdaMerging: Adaptive Model Merging for Multi-Task Learning**. arXiv:2310.02575, 2023/2024.
5. Matena and Raffel. **Merging Models with Fisher-Weighted Averaging**. arXiv:2111.09832, 2021.
6. Kirkpatrick et al. **Overcoming catastrophic forgetting in neural networks**. arXiv:1612.00796, 2016.
7. Jin et al. **Dataless Knowledge Fusion by Merging Weights of Language Models**. arXiv:2212.09849, 2022.
8. Yang et al. **ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging**. arXiv:2601.05560, 2026.
9. DeepSeek-AI. **DeepSeek-R1 repository and distill model notes**, 2025.
