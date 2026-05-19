# Compatibility-Aware Expert Composition (CAEC)

> 独立研究方向整理 | 2026-05-18
> 更新: 2026-05-19 Fisher-Regression Shrinkage QP 实验结果

---

## 0. Method Evolution

v1 (三阶段 pipeline):
```
Stage 1: Grid search → (0.50, 0.75, 0.75) → conflict gate   ← 不是我们的方法
Stage 2: TAME utility/harm profiling for R1 modes only       ← 我们的方法
Stage 3: Sparse QP injection                                  ← 我们的方法
```
问题: TA-0.75 来自 grid search，reviewer 会说"贡献只是在 0.75 上加了点 R1"。

v2 (统一 Iterative GD):
```
从 z=0 出发，梯度下降优化所有 expert 系数。
```
问题: 梯度振荡，raw utility 尺度不一致，收敛慢。3 轮实验（v3-v4b）均未收敛到预期系数。

**v3 (Fisher-Regression Shrinkage QP, 当前推荐):**
```
Step 1: 对每个 task 估计 diagonal Fisher
Step 2: Compatible experts 用 FRS-QP 求全局系数 (3x3 QP，有闭式解)
Step 3: Incompatible experts 用 UHMS 做 sparse mode injection
```
关键公式（一维 shrinkage）:
```
α_e* = A_e / (A_e + B_e)
其中 A_e = τ_e^T F_owner τ_e,  B_e = τ_e^T F_protected τ_e
```

### 0.1 FRS-QP 实验结果 (2026-05-19)

在 Qwen2.5-7B-Instruct base 上，diagonal empirical Fisher，rho=0.1, λ_alpha=1e-10。

#### 主要结果表 (推荐使用 n=50)

| | n=20 One-Expert | n=20 Multi-QP | n=50 One-Expert | n=50 Multi-QP | TA-grid |
|------|-----------------|---------------|-----------------|---------------|---------|
| Tool | 0.743 (B/A=0.35) | **0.575** | 0.787 (B/A=0.27) | **0.631** | 0.50 |
| Memory | 0.521 (B/A=0.92) | **0.341** | 0.462 (B/A=1.16) | **0.291** | 0.75 |
| Code | 0.007 (B/A=148) | **0.000** | 0.010 (B/A=100) | **0.000** | 0.75 |

One-Expert Shrinkage: α* = A_e/(A_e + B_e)。Multi-QP: box-constrained QP with cross-expert terms.

#### Bootstrap 置信区间 (B=200 resamples)

**n=20 samples:**

| Expert | Point | Median | 90% CI | std |
|--------|-------|--------|--------|-----|
| Tool   | 0.575 | 0.578  | [0.18, 0.92] | 0.248 |
| Memory | 0.341 | 0.339  | [0.04, 0.75] | 0.236 |
| Code   | 0.000 | 0.000  | [0.00, 0.001] | 0.000 |

**n=50 samples (推荐):**

| Expert | Point | Median | 90% CI | std |
|--------|-------|--------|--------|-----|
| Tool   | 0.631 | 0.643  | **[0.37, 0.86]** | 0.153 |
| Memory | 0.291 | 0.277  | **[0.09, 0.56]** | 0.144 |
| Code   | 0.000 | 0.000  | [0.00, 0.00] | 0.000 |

CIs 从 n=20 到 n=50 显著收窄 (~40%)。

**TA-grid coverage (n=50 multi-expert 90% CI):**
- tool=0.5: **YES** (in [0.37, 0.86])
- memory=0.75: **NO** (outside [0.09, 0.56]) — grid 的 memory=0.75 过高
- code=0.75: **NO** (far outside [0.00, 0.00])

#### rho Sensitivity (n=50, equal weights)

| rho   | tool  | memory | code  |
|-------|-------|--------|-------|
| 0.00  | 0.652 | 0.301  | 0.000 |
| 0.10  | 0.631 | 0.291  | 0.000 |
| 0.50  | 0.559 | 0.258  | 0.000 |
| 1.00  | 0.489 | 0.226  | 0.000 |

#### Task Weight Sensitivity (n=50, rho=0.1)

| Config          | tool  | memory | code  |
|-----------------|-------|--------|-------|
| equal (1:1:1)   | 0.631 | 0.291  | 0.000 |
| code 10x        | 0.606 | 0.282  | 0.036 |
| code 100x       | 0.438 | 0.213  | 0.322 |
| mem 3x (1:3:1)  | 0.379 | 0.552  | 0.000 |
| tool 3x (3:1:1) | 0.837 | 0.124  | 0.000 |

#### 关键发现

1. **Tool: B/A ≈ 0.27 → α* ≈ 0.79**。One-expert shrinkage 理论上解释了 TA-0.75。Multi-expert QP 给出 0.63（考虑 cross-expert interference）。
2. **Memory: B/A ≈ 1.16 → α* ≈ 0.46**。Memory expert 对其他 task 的 harm 略大于对自身的 benefit。Multi-expert QP 进一步压到 0.29。**TA-grid 的 memory=0.75 过高。**
3. **Code: B/A ≈ 100 → α* ≈ 0.01**。Code Fisher 异常小 (51 vs Tool 8810, 170× gap)。Multi-expert QP 给出 0.000。
4. **Fisher 估计随 sample 数变化**: Tool 从 12444(n=20)→8810(n=50)，Memory 从 7200→3970。更多样本给出更稳定但不同的估计。
5. **Bootstrap CI 收窄**: n=50 时 tool std=0.15 (vs n=20 时 0.25)，确认系数估计逐步稳定。

#### Code Fisher 异常诊断

Code calibration data (Code.json) 仅 25 条短样本 (avg ~950 tokens)，远少于 Tool (40条, 3740 chars) 和 Memory (827条, 22547 chars)。base model 对简单 code 任务已有高置信度 → Fisher 极低。**这是真实信号。**

Note: multi-expert QP 需要 lambda_alpha ≤ 1e-10，否则 Tikhonov 项主导。tokenize_sample 必须使用左截断（保留 response）。

#### 脚本和结果

完整 CAEC 理论和 pipeline 见: `docs/memory/caec_final_method.md`
脚本: `scripts/analysis/fisher_regression_qp.py`, `frs_qp_sensitivity.py`, `frs_qp_bootstrap.py`, `bake_frs_qp_model.py`
结果: `/tmp/shared-storage/ExpertGym/analysis/frs_qp_v3/` (n=20 point), `frs_qp_bootstrap_v2/` (n=20 bootstrap), `frs_qp_bootstrap_v3_50samples/` (n=50 bootstrap, 推荐)

---

## 1. 核心动机

将多个专家模型合并到一个基座模型时，面临一个根本挑战：**不同专家与基座的兼容性存在量级差异**。

在 Qwen2.5-7B-Instruct 上的实测数据：

| Expert | Per-param delta norm | 相对 base |
|--------|--------------------:|-----------|
| ToolRL | 0.0773 | 1.0x |
| MemoryAgent | 0.0769 | 1.0x |
| ReasonFlux-Coder | 0.0776 | 1.0x |
| DeepSeek-R1-Distill | 114.15 | **1492x** |

前三个 Agent 专家的 delta norm 量级一致（~0.077），彼此兼容。R1 的 delta norm 大 1492 倍，与 Agent 专家不兼容。

直接用 Task Arithmetic 合并 R1 会立即破坏 Agent 能力。但 R1 在 Code 上有巨大优势（BoN=0.484 vs ReasonFlux 的 0.422），不用太可惜。

**核心问题**：如何在保护已有能力的前提下，安全地从不兼容专家中提取有用知识？

---

## 2. 方法概览：三阶段 Pipeline

```
Stage 1: Compatible Expert Merge (Weighted TA + Conflict Gate)
  -> 生成强 baseline 模型 θ_merged

Stage 2: Utility/Harm Mode Profiling (TAME basis + gradient analysis)
  -> 识别 R1 中哪些 mode 有帮助（utility）、哪些有害（harm）

Stage 3: Sparse Mode Injection (Closed-form QP)
  -> 只注入高 utility / 低 harm 的 mode，得到最终模型
```

最终模型在全部三个任务上同时超越所有 baseline：

| 指标 | Best Baseline (DARE-TA) | 本方法 | Delta |
|------|------------------------:|-------:|------:|
| Tool mean | 0.7952 | **0.7954** | +0.0002 |
| Memory F1 | 0.6901 | **0.7720** | +0.0819 |
| Code Acc | 0.3365 | **0.3597** | +0.0232 |
| Code BoN(4,4) | 0.3900 | **0.4408** | +0.0508 |

对比 AdaMerging（learned baseline 中 Code 最强的）：

| 指标 | AdaMerging TW | 本方法 | Delta |
|------|-------------:|-------:|------:|
| Tool mean | 0.7835 | **0.7954** | +0.0119 |
| Memory F1 | 0.6678 | **0.7720** | +0.1042 |
| Code BoN(4,4) | 0.4242 | **0.4408** | +0.0166 |

---

## 3. Stage 1: Compatible Expert Merge

### 3.1 Weighted Task Arithmetic

对 3 个兼容 Agent 专家，找到最优系数组合：

$$\theta_{\text{wta}} = \theta_0 + \alpha_{\text{tool}} \cdot \tau_{\text{tool}} + \alpha_{\text{mem}} \cdot \tau_{\text{mem}} + \alpha_{\text{code}} \cdot \tau_{\text{code}}$$

**关键发现**：TA-0.75（等系数=0.75）优于 TA-1.0（等系数=1.0）。这在 model merging 文献中少见但有清晰解释——agent 任务要求保持 instruction following 格式，过强的 task vector 破坏 base model 的格式先验。

Weighted-TA sweep 的 Pareto 最优点：

$$(\alpha_{\text{tool}}, \alpha_{\text{mem}}, \alpha_{\text{code}}) = (0.50, 0.75, 0.75)$$

对比 TA-1/3 和 TA-0.75:

| Setting | Tool | Memory F1 | Code BoN |
|---------|-----:|----------:|---------:|
| TA-1/3 (0.33, 0.33, 0.33) | 0.7848 | 0.6465 | 0.4173 |
| TA-0.75 (0.75, 0.75, 0.75) | 0.7942 | 0.7361 | 0.3812 |
| Weighted (0.50, 0.75, 0.75) | ~0.785 | ~0.76 | ~0.42 |

### 3.2 Conflict-Gated Residual

在 Weighted-TA 基础上，进一步用 conflict gate 注入额外的 Tool 能力：

$$\theta_{\text{cg}} = \theta_{\text{wta}} + \alpha_{\text{extra}} \cdot M_{\text{agree}} \odot \tau_{\text{tool}}$$

其中 conflict gate mask:

$$M_{\text{agree}}[p] = \mathbb{1}[\text{sign}(\tau_{\text{tool}}[p]) = \text{sign}(\tau_{\text{code}}[p])]$$

即只在 Tool 和 Code 的 task vector 符号一致的参数位置注入额外 Tool residual。符号冲突的位置被归零，避免跨任务干扰。

**实验结果**（ITER_019 report）：

| Setting | alpha_extra | Tool (BFCL) | Code (CURE) | Memory (HotpotQA) |
|---------|:-----------:|:-----------:|:-----------:|:-----------------:|
| Base (0.50, 0.75, 0.75) | 0 | 368/440 | 18/50 | 201/256 |
| cg-tool-extra020 | 0.20 | **381/440** | **18/50** | **201/256** |
| cg-tool-extra030 | 0.30 | 381/440 | 15/50 (worse) | 201/256 |

alpha=0.20 成功提升 Tool 而不损害 Code/Memory。alpha=0.30 过强导致 Code 下降。

**cg-tool-extra020 完整配方**：
$$W = W_{\text{base}} + 0.50 \cdot \tau_{\text{tool}} + 0.75 \cdot \tau_{\text{mem}} + 0.75 \cdot \tau_{\text{code}} + 0.20 \cdot M_{\text{agree}} \odot \tau_{\text{tool}}$$

Checkpoint: `/tmp/shared-storage/AgentMerging_plan/experiments/conflict_gated_residual/ITER_019_cg_tool_extra020/model`

---

## 4. Stage 2: TAME Utility/Harm Mode Profiling

### 4.1 Mode 定义

将每个 expert 的 task vector 按 (expert, layer, module) 拆分为 **modes**：

$$b_j = \tau_{e_j}[p_j], \quad j = 1, \ldots, J$$

对 Qwen2.5-7B（28 layers x 7 modules/layer = 196 params/expert x 4 experts）= 784 total modes。
其中 R1 有 196 个 modes。

### 4.2 Activation Normalization

不同 mode 的原始尺度不可比（MLP 和 Attention 的 delta norm 差异巨大）。使用对角近似的 activation-aware normalization:

$$\tilde{b}_j = \frac{b_j}{s_j + \epsilon}$$

其中:

$$s_j = \sqrt{\sum_i \mathbb{E}[h_i^2] \cdot \|\Delta W_j[:, i]\|^2}$$

- $h_i$: 该 linear layer 在 calibration 数据上的第 $i$ 维输入 activation
- $\Delta W_j[:, i]$: task vector 矩阵的第 $i$ 列

**物理含义**: $s_j$ 度量了 mode $b_j$ 对 layer output 的 RMS 扰动。归一化后 $z_j = 1.0$ 表示"该 mode 的 response 扰动为 1 倍 RMS"。不同 layer/module/expert 的系数变得可比。

**实现**（`tame/basis.py:95-186`）:
```python
# Forward base model, hook each linear layer to collect E[h^2]
def make_hook(pname):
    def hook_fn(module, input, output):
        h = input[0].detach().float()  # [batch, seq, in_dim]
        h_sq = (h ** 2).mean(dim=(0, 1))  # [in_dim]
        ...

# Then for each basis:
col_sq_norms = (tv.float() ** 2).sum(dim=0)  # [in_dim]
s_sq = (d * col_sq_norms).sum().item()
s = s_sq ** 0.5
```

**计算代价**: ~100 calibration samples forward pass + O(d_in * d_out) per basis (纯 CPU)。

### 4.3 Utility Score

Utility 度量：mode $j$ 对 target skill（Code）的梯度帮助程度。

$$u_j = -\frac{\partial \mathcal{L}_{\text{skill}}}{\partial z_j}\bigg|_{z=0} = -\left\langle \nabla_{W_p} \mathcal{L}_{\text{skill}}, \tilde{b}_j \right\rangle_F$$

- $u_j > 0$: mode 减小 skill loss，**有帮助**
- $u_j < 0$: mode 增大 skill loss，**有害**

$\mathcal{L}_{\text{skill}}$ 是 merged model 对 target expert 的 KL divergence:

$$\mathcal{L}_{\text{skill}} = \sum_{c} w_c \cdot \text{KL}(\pi_{\theta(z)}(\cdot|x_c) \| \pi_{\theta_{\text{expert}(c)}}(\cdot|x_c))$$

**关键**：只需在 $z=0$（即 base merge）处计算一次梯度，不需要迭代训练。

### 4.4 Harm Score

Harm 度量：mode $j$ 对 protected tasks 的损害程度。

$$h_j = G_{jj} = \sum_{\text{elements}} F_{\text{element}} \cdot (\tilde{b}_{j,\text{element}})^2$$

- $F_{\text{element}}$: protected task 数据上 base model 的 empirical Fisher 对角
- $\tilde{b}_{j,\text{element}}$: 归一化 mode 的第 element 个参数

**物理含义**: $h_j$ 衡量 mode $j$ 在 Fisher metric 下对 protected tasks output 的二次扰动。高 $h_j$ 意味着修改该 mode 对 Tool/Memory 影响大。

**实现**（`tame/basis.py:263-269`）:
```python
def get_trust_region_weight(self, expert_idx, param_name):
    b_tilde = self.get_normalized_basis(expert_idx, param_name)
    fisher = self.fisher_diag.get(param_name, None)
    return (fisher * b_tilde.float() ** 2).sum().item()
```

### 4.5 Optimal z: Closed-Form Solution

给定 utility/harm scores，最优系数有闭式解（Sparse QP with L1 + quadratic regularization）:

$$z_j^* = \frac{\text{soft\_threshold}(u_j, \lambda_{\text{sparse}})}{h_j + \lambda_{\text{trust}}}$$

其中:

$$\text{soft\_threshold}(u, \lambda) = \text{sign}(u) \cdot \max(|u| - \lambda, 0)$$

**直觉**:
- 分子 = effective utility（扣除 sparsity 门槛后的剩余效用）
- 分母 = effective cost（harm + trust-region regularization）
- $|u_j| < \lambda_{\text{sparse}}$ 的 mode 被自动清零（稀疏选择）
- 高 harm 的 mode 即使 utility 大也只获得小系数

这就是 TAME 的核心：**utility-over-harm ratio determines mode selection**。

### 4.6 Budget Constraint

实际使用中加入额外约束保证安全：

- `max_modes = 64`: 最多选择 64 个 mode（196 个候选中的 ~33%）
- `max_abs_z = 0.001`: 每个系数绝对值不超过 0.001

QP solver（`global_budget_sparse_qp`）在闭式解基础上按 utility/harm ratio 排序，取 top-K mode:

```
Config (qwen7b_calibration_global.yaml):
  method:
    sparse_solver: global_budget_sparse_qp
    solver:
      lambda_sparse: 0.0
      lambda_trust: 1.0
      max_modes: 64
      max_abs_z: 0.001
```

---

## 5. Stage 3: Sparse Mode Injection

### 5.1 Bake 过程

将选出的 64 个 R1 modes 注入 base merge:

$$\theta_{\text{final}} = \theta_{\text{cg}} + \sum_{j \in \text{selected}} z_j^* \cdot \tilde{b}_j$$

其中 $\theta_{\text{cg}}$ 是 Stage 1 的 cg-tool-extra020。

### 5.2 Selected Modes

从 Z coefficients 文件（`z_coefficients.json`）可见:
- 全部 64 个 mode 来自 **r1code** expert
- 全部 z = **0.001**（达到 max_abs_z 上限）
- 覆盖 layer 0-27 的各种 module（mlp.gate_proj, mlp.up_proj, mlp.down_proj, self_attn.q/k/v/o_proj）

这说明当前 budget 很保守（z=0.001 非常小），**selection 本身是主要贡献**，而非系数大小。

### 5.3 最终模型

**模型名**: `tame-cg-r1calib-global-v2`

**构成公式**:
$$W = \underbrace{W_{\text{base}} + 0.50 \tau_{\text{tool}} + 0.75 \tau_{\text{mem}} + 0.75 \tau_{\text{code}} + 0.20 M_{\text{agree}} \odot \tau_{\text{tool}}}_{\text{Stage 1: cg-tool-extra020}} + \underbrace{\sum_{j=1}^{64} 0.001 \cdot \tilde{b}_{j}^{\text{R1}}}_{\text{Stage 3: R1 modes}}$$

---

## 6. 完整评测对比

### 6.1 vs 所有 Static Baselines

| Method | Tool | Memory F1 | Code Acc | Code BoN |
|--------|-----:|----------:|---------:|---------:|
| Qwen2.5-7B-Instruct (base) | 0.7500 | 0.5288 | 0.2800 | 0.3304 |
| TA-1/3 | 0.7848 | 0.6465 | 0.3409 | 0.4173 |
| TA-0.75 | 0.7942 | 0.7361 | 0.3441 | 0.3812 |
| TIES | 0.7642 | 0.6359 | 0.3355 | 0.3880 |
| DARE-TA | 0.7952 | 0.6901 | 0.3365 | 0.3900 |
| DARE-TIES | 0.7952 | 0.6891 | 0.3426 | 0.4007 |
| AdaMerging TW | 0.7835 | 0.6678 | 0.3406 | 0.4242 |
| AdaMerging LW | 0.7848 | 0.6674 | 0.3350 | 0.3949 |
| AdaMerging++ TW | 0.7629 | 0.6407 | 0.3309 | 0.3773 |
| Mixture GRPO | 0.7823 | 0.6643 | 0.3384 | 0.3782 |
| WUDI | 0.7823 | 0.6591 | 0.3304 | 0.4095 |
| **Ours** | **0.7954** | **0.7720** | **0.3597** | **0.4408** |

### 6.2 vs ExpertGym Online Learning

| Method | Tool | Memory F1 | Code BoN |
|--------|-----:|----------:|---------:|
| ExpertGym G-final (init=1, GRPO+OPD+Ret, 20 iters) | 0.7942 | 0.7548 | 0.4252 |
| **Ours (zero-shot, no training)** | **0.7954** | **0.7720** | **0.4408** |

注意：本方法 **不需要迭代训练**。utility/harm profiling + closed-form QP 是 one-shot 过程。

---

## 7. 代码指引

### 7.1 TAME 核心模块

| 文件 | 功能 |
|------|------|
| `AgentMerging/worktree/TAME/tame/basis.py` | TAMEBasis 类：task vector 计算、activation normalization、Fisher diagonal |
| `AgentMerging/worktree/TAME/tame/model.py` | TAMEModel 类：parametric merge（direct/tanh/gated 参数化）、梯度投影 |
| `AgentMerging/worktree/TAME/tame/losses.py` | 四项 loss：skill alignment KL、contract hinge、sparsity L1、trust-region quadratic |
| `AgentMerging/worktree/TAME/THEORY_REVIEW.md` | 完整理论推导和梯度可计算性证明 |

### 7.2 TAME Pipeline Scripts

| 文件 | 功能 |
|------|------|
| `TAME_harness/configs/tame/qwen7b_calibration_global.yaml` | 配置文件（solver 参数、数据 split、model 路径） |
| `TAME_harness/` (pipeline scripts) | 八阶段 pipeline: prepare → splits → basis_audit → basis_precompute → utility_harm → sparse_solve → bake_plan → smoke |

### 7.3 Conflict-Gated Merge

| 文件 | 功能 |
|------|------|
| `AgentMerging/skill/plan/v1-feedback/pro_report_.../ITER_019_020.md` | Conflict-gated 实验报告 |
| Checkpoint | `/tmp/shared-storage/AgentMerging_plan/experiments/conflict_gated_residual/ITER_019_cg_tool_extra020/model` |

### 7.4 Evaluation

| 文件 | 功能 |
|------|------|
| `ExpertGym/skill/command/run_full_eval_suite.sh` | 评测入口（Tool/Memory/Code） |
| `ExpertGym/docs/evaluation/best_ever_model.md` | 最佳模型记录 |
| `ExpertGym/docs/evaluation/20260518_baselines_eval6.md` | 全部 baseline 评测结果 |

### 7.5 关键产出物

| 产出 | 路径 |
|------|------|
| Baked model | `/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/baked_cg_plus_r1code_calib_global_v2/` |
| Z coefficients (64 modes) | `/tmp/shared-storage/TAME/experiments/tame-r1-core-20260504/calibration_global_r1_v2_solution/z_coefficients.json` |
| Mode manifest (588 modes) | `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json` |

---

## 8. 与 Related Work 的深度对比

### 8.1 vs Fisher Merging: Fisher 在两个方法中角色完全相反

**Fisher Merging (Matena & Raffel, 2022)** 用 Fisher 作为 **合并权重**：

$$\theta_{\text{merged}}[p] = \frac{\sum_i F_i[p] \cdot \theta_i[p]}{\sum_i F_i[p]}$$

- $F_i[p]$ 高 → 模型 $i$ 在 $p$ 上"更确定" → 合并时更相信模型 $i$ 的值
- 逐参数加权平均，无稀疏，无选择，所有参数都被修改

**我们** 用 Fisher 作为 **harm score**（安全约束）：

$$h_j = \sum_{\text{elements}} F_{\text{protected}}[p] \cdot \tilde{b}_j[p]^2$$

- $F_{\text{protected}}[p]$ 高 → protected task 对 $p$ 敏感 → 修改 $p$ 危险 → 少改或不改

| | Fisher Merging | 我们 |
|---|---|---|
| **Fisher 的角色** | 合并权重（pull toward expert） | 安全约束（push away from danger） |
| **方向** | F 高 → 更多采纳该 expert 的值 | F 高 → 更少修改该位置 |
| **操作** | 所有模型加权平均 | Base + 稀疏 delta 注入 |
| **稀疏性** | 无（每个参数都变） | 有（大部分 mode z=0） |
| **输出** | $\frac{\sum F_i \theta_i}{\sum F_i}$ | $\theta_{\text{base}} + \sum z_j \tilde{b}_j$ |
| **Expert 对称性** | 对称（所有 expert 平等） | 不对称（compatible 全合 + incompatible 选注） |
| **Fisher 来源** | 各 expert 自己的 Fisher | **Protected task** 的 Fisher |

**Fisher Merging 的 scale gap 问题**：当 R1 的 delta norm 是其他 expert 的 1492x 时，$F_{\text{R1}}$ 可能主导加权平均，结果退化为几乎只取 R1 的参数值 → collapse。我们的方法没有这个问题，因为 harm score 用的是 protected task 的 Fisher，跟 R1 自己的尺度无关。

**一句话区分**：Fisher Merging 是 importance-weighted averaging（谁说了算？），我们是 safety-constrained selective injection（改了会不会出事？）。

### 8.2 vs ReasonAny (Yang et al., 2026): 我们回答更深层的问题

**ReasonAny** ([arXiv:2601.05560](https://arxiv.org/abs/2601.05560)) 是与我们最相关的 concurrent work，同样解决 "Reasoning + X" 合并问题。

**ReasonAny 的核心发现**：Reasoning 能力存在于 **低梯度敏感**参数区域（反直觉）。

**ReasonAny 的方法 — Contrastive Gradient Identification (CGI)**:

1. 对 reasoning model 参数计算 gradient importance: $I(\theta_r, \mathcal{D}_r) = \mathbb{E}_{x}[|\nabla_\theta \mathcal{L}(x;\theta)|]$
2. Reasoning 参数 = **BottomK**(gradient magnitude, p=5%) — 选最低梯度的 5%
3. Domain 参数 = **TopK**(gradient magnitude, p=5%) — 选最高梯度的 5%
4. 取互斥集: $\mathcal{N}_r' = \mathcal{N}_r \setminus \mathcal{N}_t$, $\mathcal{N}_t' = \mathcal{N}_t \setminus \mathcal{N}_r$
5. Binary mask 合并: $\theta = \theta_{\text{base}} + \lambda_r(\tau_r \odot M_r) + \lambda_t(\tau_t \odot M_t)$, $\lambda=1.0$

**测试 domain**: Safety, Biomedicine, Finance（Qwen2.5-7B/14B, Llama-3.1-8B）

**逐维度对比**：

| 维度 | ReasonAny | 我们 |
|------|-----------|------|
| **问题** | Reasoning + 1 Domain（两模型） | 3 Compatible Agents + 1 Incompatible Reasoning（四模型） |
| **选择粒度** | Element-level（逐参数 binary mask） | Mode-level（layer x module 块，continuous 系数） |
| **选择标准** | 梯度 **幅度** BottomK/TopK | 梯度 **方向**（utility）+ Fisher（harm） |
| **系数** | Binary: 0 or $\lambda=1.0$ | Continuous: $z_j^* \in [-0.001, 0.001]$ |
| **Reasoning 注入量** | 5% 参数 × full magnitude | 64/196 modes × 极小系数 |
| **Domain 数量** | 1 domain + 1 reasoning | 3 compatible + 1 incompatible |
| **需要训练** | 否 | 否 |
| **Domain 测试** | Safety, Biomedicine, Finance | Agent (Tool, Memory, Code) |
| **Model family** | Qwen + Llama | Qwen only |

**关键区别 1: 幅度 vs 方向**

ReasonAny 用梯度 **幅度**（$|g_j|$）做选择：幅度低 → reasoning 参数。
我们用梯度 **方向**（$\langle g, \tilde{b}_j \rangle$）做 utility：方向对齐 → 有帮助。

幅度只告诉你"这个参数有没有在被用"，方向告诉你"动这个参数往哪走有用"。方向信息严格更多：两个参数梯度幅度相同，但一个的方向对齐 skill gradient，另一个正交——只有方向能区分它们。

**关键区别 2: Binary mask vs Continuous coefficient**

ReasonAny: 选或不选（0 or 1），$\lambda=1.0$。
我们: 连续系数 $z_j^*$，由 utility/harm ratio 决定。

对 R1 这种 1492x delta 的模型，$\lambda=1.0$ + 5% mask 意味着注入的绝对量仍然巨大。ReasonAny 能 work 是因为他们的 domain expert 都从 **同一个 base fine-tune**（Safety LoRA, Meditron），delta scale 差异远没有我们极端。在 1492x gap 的场景下，binary mask + $\lambda=1.0$ 极可能 collapse。

**关键区别 3: 两模型 vs 多模型**

ReasonAny 只处理 Reasoning + 1 Domain（两两合并）。我们处理 3 Compatible + 1 Incompatible，需要额外解决：
- Compatible experts 之间的干扰（Stage 1: weighted TA + conflict gate）
- 一个 R1 mode 可能对 Tool 有 utility 但对 Memory 有 harm（multi-task harm score）

**两者的 insight 互补**：

ReasonAny 的"reasoning 在低梯度区域"对我们有启发。如果 reasoning 参数天然 Fisher 低，那我们的 harm score 会给这些参数低分，使它们更容易被选中注入。两个方法的 insight 一致，但我们的 framework 更 principled（分解为独立的 utility 和 harm 两个轴）。

**一句话定位**: ReasonAny identifies *where* reasoning lives (low-gradient regions); we determine *how much* to inject safely into multi-expert systems (utility/harm ratio + continuous coefficients).

### 8.3 vs 其他方法的统一视角

```
Fisher Merging:  F 做权重，全参数平均，对称
ReasonAny:       |∇| 做选择（BottomK/TopK），binary mask，λ=1
TIES:            |τ| 做剪枝 + sign 投票，binary mask
DARE:            random mask + rescale，无信息选择
AdaMerging:      entropy minimization 学 task/layer 系数，无 harm 约束
我们:            ⟨∇,b⟩ 做 utility + F 做 harm，continuous z，closed-form QP
```

我们是唯一同时具备以下特性的方法：
1. **Directional utility**（梯度方向，非幅度）
2. **Explicit harm measurement**（Fisher-based，非 heuristic exclusion）
3. **Continuous coefficients**（非 binary mask）
4. **Asymmetric expert handling**（compatible 全合 + incompatible 选注）

### 8.4 为什么 Utility/Harm 优于全局系数搜索

传统方法（TA, TIES, AdaMerging 等）对 incompatible expert 只有一个选择：调整全局系数 $\alpha_{\text{R1}}$。但 R1 的 delta norm 是 Agent 专家的 1492 倍，任何非零全局系数都会造成大范围扰动。

Utility/Harm 的关键优势：

1. **Per-mode granularity**: 196 个 R1 mode 中只选 64 个，拒绝 132 个。被拒绝的 mode 要么 utility < 0（有害），要么 harm 太高（太危险）。
2. **Closed-form**: 不需要迭代搜索、不需要 GPU 训练。只需要一次 gradient + Fisher 计算。
3. **Safety guarantee**: max_abs_z bound + Fisher trust-region 双重保护，防止 catastrophic collapse。
4. **Interpretability**: 每个 mode 的 utility/harm score 直接可检查，知道"为什么选这个 layer"。

### 8.5 Activation Normalization 的必要性

不做 normalization 时，R1 delta norm 是 Agent 的 1492x，mode selection 会被 norm 主导（大 delta 的 mode 自动有大 utility）。Normalization 后:

$$\tilde{b}_j = b_j / s_j$$

使 utility score 反映的是"单位 activation 扰动下的 skill improvement"，而非"原始 delta 大小"。这是 R1 mode 能安全注入的前提。

### 8.6 与 ExpertGym (Online RL) 的互补关系

ExpertGym 通过在线 rollout + GRPO/OPD/Retention 迭代训练 gate 系数，20 轮后达到 G-final 水平。TAME utility/harm 是 zero-shot 方法，一步到位。

两者可组合：
1. TAME 提供 zero-shot 初始化（比均匀 1/3 好得多）
2. ExpertGym 在此基础上微调（可能只需 3-5 轮而非 20 轮）

---

## 9. 泛化方向与实验规划

方法的 general claim 不是"reasoning 注入"，而是 **cross-scale model composition**：当 expert deltas 的尺度/分布存在量级差异时，需要 mode-level utility/harm selection。Reasoning 注入是最好的 showcase，但需要泛化实验支撑。

### 9.1 Tier 1: 必做实验（直接对比 ReasonAny）

**实验 A: ReasonAny 作为 baseline（在我们的 Agent 任务上）**

在我们的 Agent 设置上跑 ReasonAny:
- Base: Qwen2.5-7B-Instruct
- Reasoning: DeepSeek-R1-Distill-Qwen-7B
- Domain: 先 TA 合 3 expert 为一个 domain model，再用 ReasonAny 注入 R1
- 或分别跟 R1 做 ReasonAny merge 再合并

预测：ReasonAny 的 binary mask + $\lambda=1.0$ 会因为 R1 delta 太大而 collapse（Tool 格式崩溃）。直接证明 continuous coefficient 的必要性。

**实验 B: 验证"低梯度→reasoning"假设**

```python
# 对每个 R1 mode，计算 gradient nuclear norm
for mode in r1_modes:
    grad_norm = nuclear_norm(gradient(mode, calibration_data))
    utility = our_utility_score(mode)
    harm = our_harm_score(mode)
    # 画散点图：grad_norm vs utility, grad_norm vs harm
```

如果确认低梯度 mode 也有高 utility，说明两个方法的 insight 一致但我们更 refined。

**实验 C: Selection criterion ablation（核心对比实验）**

在同一个任务上对比不同选择策略：

| Selection Method | 来源 | Description |
|-----------------|------|-------------|
| BottomK gradient (ReasonAny) | ReasonAny | p=5% lowest grad for reasoning |
| TopK gradient | Naive | p=5% highest grad for reasoning |
| Random selection | Control | Random 5% parameters |
| Utility only (no harm) | Our ablation | Select by gradient direction only |
| Harm only (no utility) | Our ablation | Avoid high Fisher only |
| **Utility/Harm ratio (full)** | **Ours** | Full method |

### 9.2 Tier 2: 泛化到 ReasonAny 的 Domain

在 ReasonAny 的 domain 上测试我们的方法，直接对比数字：

**Biomedicine**（最容易，有公开模型）：
- Base: Qwen2.5-7B-Instruct
- Reasoning: DeepSeek-R1-Distill-Qwen-7B
- Domain: Meditron3-Qwen2.5-7B
- Benchmark: MedQA, PubMedQA, GSM8K

**Safety**（如果能拿到对齐模型）：
- Benchmark: HarmBench, GSM8K

### 9.3 Tier 3: Scale Gap 控制实验

人为控制 delta scale gap，画 **"scale gap vs. method advantage"** 曲线：

```python
for scale_factor in [1, 10, 100, 1000, 10000]:
    scaled_r1_delta = r1_delta * (agent_norm / r1_norm) * scale_factor
    result_reasonany = run_reasonany(scaled_delta)   # binary mask + λ=1
    result_ours = run_ours(scaled_delta)              # utility/harm + continuous z
    result_naive_ta = run_ta(scaled_delta)            # naive TA
```

预测：
- Gap 小时：三者都 work，差异不大
- Gap 中等时：naive TA 开始退化，ReasonAny 和我们仍 work
- Gap 大时（~1000x+）：ReasonAny collapse（binary mask 注入量太大），我们仍 work
- 存在一个 **critical gap threshold**，过了这个阈值 continuous coefficient 的优势 emerge

这张曲线本身就是一个很强的 contribution — 回答了"什么时候需要用我们的方法"。

### 9.4 Tier 4: 多 Model Family

在 Llama-3.1-8B 上复现，证明不依赖 Qwen：
- Base: Llama-3.1-8B-Instruct
- Reasoning: DeepSeek-R1-Distill-Llama-8B
- Domain: 需要找 Llama 的 agent/domain fine-tune

### 9.5 放松 R1 Budget

当前设置极保守（max_modes=64, max_abs_z=0.001）。Code BoN 已到 0.4408，但 R1 单独的 Code BoN 是 0.484。建议实验：

| Setting | max_modes | max_abs_z | 预期 |
|---------|:---------:|:---------:|------|
| 当前 | 64 | 0.001 | Code BoN 0.4408 |
| 放松 1 | 96 | 0.003 | Code BoN ~0.46? |
| 放松 2 | 128 | 0.005 | 可能触及 collapse boundary |
| 激进 | 196 | 0.01 | 需要 harm check |

### 9.6 多源 Incompatible Expert

当前只有 R1 一个不兼容专家。如果有多个不兼容专家（如 R1 + Qwen2.5-Math-7B），TAME 的 mode selection 天然支持跨源 joint optimization:

$$z^* = \arg\min_z \ g^T z + \lambda_{\text{sparse}} \|z\|_1 + \frac{1}{2} z^T (G + \lambda_{\text{trust}} I) z$$

QP 在所有候选 mode 上联合求解。

### 9.7 理论 Bound: Collapse Boundary 预测

utility/harm ratio 可以给出 collapse boundary 的 analytical prediction:

- 当 $\sum_j z_j \cdot h_j > C_{\text{critical}}$ 时发生 collapse
- 可以用 Fisher norm 预测而非实际 eval

这给 "safe injection budget" 一个先验上界。

---

## 10. Paper Framing 建议

### 10.1 标题方向

不要叫"Reasoning Injection for Agent Merging"（太窄），建议：

> **Safe Expert Injection: Extracting Capabilities from Incompatible Models via Utility-Harm Mode Selection**

或：

> **Mode-Level Utility/Harm Selection for Cross-Scale Model Composition**

### 10.2 Story 结构

1. **Observation**: Expert deltas 的 scale 差异导致 standard merging 失效（不是新 observation，但没人正式研究）
2. **Analysis**: 为什么 scale gap 导致 collapse？（Fisher-based analysis + ReasonAny 的 low-gradient insight）
3. **Method**: Utility/harm per-mode selection + closed-form QP
4. **Primary evaluation**: R1 → Agent merging（最极端 case，最强结果）
5. **Generality**: Scale gap 控制实验 + Biomedicine domain + selection criterion ablation

### 10.3 vs ReasonAny 的差异化 narrative

> ReasonAny identifies *where* reasoning lives (low-gradient regions) and uses binary masks to separate parameter ownership. We ask a deeper question: *how much* of each reasoning mode should be injected, given explicit safety constraints? By decomposing the decision into orthogonal utility (gradient direction) and harm (Fisher magnitude) axes, we obtain continuous coefficients via closed-form QP that gracefully handle extreme scale gaps where binary masks fail.

---

## 11. 关键引用和相关工作

| 方法 | 与本方法关系 |
|------|-------------|
| Task Arithmetic (Ilharco et al., 2023) | 本方法的基础；Stage 1 就是 weighted TA |
| TIES (Yadav et al., 2024) | 本 baseline；per-parameter 剪枝 + 符号投票 |
| DARE (Yu et al., 2024) | 本 baseline；random dropout + rescale |
| AdaMerging (Yang et al., 2024) | 本 baseline；learned task/layer-wise 系数（entropy minimization） |
| **ReasonAny (Yang et al., 2026)** | **最相关 concurrent work**；梯度幅度做 binary selection，我们用梯度方向+Fisher 做 continuous selection |
| Fisher Merging (Matena & Raffel, 2022) | Fisher 做合并权重（pull toward expert），我们用 Fisher 做安全约束（push away from danger） |
| RegMean (Jin et al., 2023) | 用 input covariance 做 closed-form merge，与我们的 activation norm 思路类似 |
| WUDI (Wu et al., 2024) | 本 baseline；weight disentanglement |
| LED Merging | ReasonAny baseline；location-election-disjoint 参数选择 |

**本方法的独特贡献**:
1. 首次在 mode 粒度上做 utility/harm profiling（非 element 粒度或 task 粒度）
2. 明确区分 compatible vs incompatible experts，用不同策略处理
3. Closed-form sparse QP 输出 continuous coefficients（非 binary mask），处理极端 scale gap
4. 将 Fisher 从 importance weighting 重新定义为 safety constraint
5. 在 agent merging（function calling + long-context memory + code generation）场景验证有效

---

## 12. 统一 Iterative Utility/Harm QP（v2 方法）

### 12.1 动机

v1 方法的弱点：Stage 1 的 TA-0.75 系数来自 grid search，不是方法本身的产物。如果 reviewer 认为性能主要来自 grid search 找到的 0.75，我们的方法贡献就被削弱为"在 0.75 上加了点 R1"。

**解法**：将 utility/harm QP 同时应用于 **所有 expert 的所有 mode**（compatible + incompatible），从 z=0 出发迭代求解。Compatible experts 的最优系数（~0.75）会作为 QP 的自然输出 emerge，而不是人工设定的。

### 12.2 算法

```
输入: base model θ₀, expert task vectors {δ_{e,p}}, calibration data D
超参: T (iterations), λ_sparse, λ_trust, lr, max_abs_z_r1

初始化: z_{e,p} = 0  ∀(e, p)

For t = 1, ..., T:
    # 1. Apply current z
    ∀p: W_p ← W_{0,p} + Σ_e z_{e,p} · δ_{e,p}

    # 2. Gradient projection on calibration data
    For each task c, each sample x:
        loss = L(θ(z); x)    # supervised cross-entropy
        grad_p = ∂loss/∂W_p

        For each expert e, param p:
            proj = <grad_p, δ_{e,p}>

            # Utility: when task matches expert's target skill
            if c == owner_task(e):
                u_{e,p} += -proj

            # Harm: when task is protected and NOT expert's own
            if c ∈ protected_tasks and c ≠ owner_task(e):
                h_{e,p} += max(proj, 0)

    # 3. Average and solve QP
    ∀(e,p): u_{e,p} /= count,  h_{e,p} /= count

    ∀(e,p): z*_{e,p} = soft_threshold(u_{e,p}, λ_sparse) / (h_{e,p} + λ_trust)

    # 4. Clip R1 modes
    if e == "r1": z*_{e,p} = clip(z*, -max_abs_z_r1, max_abs_z_r1)

    # 5. Update with learning rate
    ∀(e,p): z_{e,p} ← z_{e,p} + lr · (z*_{e,p} - z_{e,p})

输出: z_{e,p} → bake into checkpoint
```

### 12.3 预期行为

| 迭代 | Compatible (Tool/Mem/Code) | R1 |
|:----:|:--------------------------:|:---:|
| t=0 | z = 0 (base model) | z = 0 |
| t=1 | z → ~0.3-0.5 | z ≈ 0 |
| t=2 | z → ~0.6-0.8 | 少数 mode 非零 |
| t=3 | z → ~0.75 (收敛) | mode selection 稳定 |
| t=4-5 | 微调收敛 | z ≈ 0.001-0.01 |

**关键验证点**：
- t=3 后 compatible experts 的 mean z 是否接近 (tool~0.50, memory~0.75, code~0.75)？
- 如果是，证明 TA-0.75 是 utility/harm QP 的自然输出
- 如果不是，分析 divergence 原因（linear approximation 不够？需要二阶信息？）

### 12.4 为什么 Conflict Gate 可能自然消解

当前 conflict gate 是手动 binary mask（Tool/Code sign agreement 处注入额外 Tool）。在统一 QP 中：
- Tool mode 在 sign conflict 区域对 Code task 的 gradient projection 为负 → harm 高
- QP 自然给这些 mode 更小的 z
- 这是 conflict gate 的 **soft, principled version**

### 12.5 实现

脚本: `scripts/analysis/unified_iterative_qp.py`

```bash
# 先跑 3-expert only（验证能否 recover TA-0.75）
python scripts/analysis/unified_iterative_qp.py \
    --experts tool,memory,code \
    --num-iters 5 \
    --samples-per-task 5 \
    --lambda-trust 0.01 \
    --output-dir /tmp/shared-storage/ExpertGym/analysis/unified_qp_3expert_v1

# 再跑 4-expert（加入 R1）
python scripts/analysis/unified_iterative_qp.py \
    --experts tool,memory,code,r1 \
    --num-iters 5 \
    --samples-per-task 5 \
    --lambda-trust 0.01 \
    --max-abs-z-r1 0.01 \
    --output-dir /tmp/shared-storage/ExpertGym/analysis/unified_qp_4expert_v1
```

### 12.6 成功标准

| 实验 | 成功条件 |
|------|---------|
| 3-expert QP | tool mean z ∈ [0.3, 0.8], memory ∈ [0.5, 1.0], code ∈ [0.5, 1.0] |
| 4-expert QP | 同上 + R1 modes sparse (< 50% nonzero), z_r1 small |
| Bake + eval | 3-expert bake ≥ TA-0.75 性能 |
| Bake + eval | 4-expert bake ≥ tame-cg-r1calib-global-v2 性能 |
