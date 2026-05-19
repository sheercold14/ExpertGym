# 2026-05-19 第一阶段 TRC 当前 loss 策略与代码步骤记忆

## 一句话定位

当前第一阶段主线不是 on-policy GRPO，也不是直接用 reward 更新 gate；它是 **TRC hidden-residual alignment**：冻结 base model 和 expert task vectors，只训练 layer-wise OP-VEC gate 系数，用 expert 成功轨迹在隐藏层 residual 方向上对齐，让合并模型先得到一个强初始化 checkpoint，再交给后续 GRPO/OPD reward refinement。

当前高价值候选是：

| candidate | train run | target epoch | checkpoint | 当前角色 |
|---|---|---:|---|---|
| `anchor_i4` | `/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_anchor_20260519` | 4 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i4_20260519` | 保守候选 |
| `anchor_i8` | `/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_anchor_20260519` | 8 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i8_20260519` | 保守加强候选 |
| `dir_i8` | `/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_20260519` | 8 | `/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_dir_i8_20260519` | 当前最强候选，Code 仍在正式评测 |

## 关键代码入口

| 阶段 | 文件 | 作用 |
|---|---|---|
| TRC 训练 | `scripts/trc/train_trc_layer_gates.py` | 读取 expert 轨迹，计算 hidden residual loss，只更新 gate |
| TRC 启动脚本 | `skill/command/run_20260519_trc_layer_init1_v3_directional.sh` | 统一设置 v3 directional 训练参数 |
| Bake checkpoint | `scripts/eval/opvec_bake_checkpoint.py` | 将 gate 系数 bake 成普通 HuggingFace checkpoint |
| 正式评测 | `skill/command/run_full_eval_suite.sh` | 调 Tool/BFCL、Memory/HotpotQA、Code/CURE 三套 harness |
| 实验前端 | `scripts/monitor/stage1_experiment_dashboard.py` | 只读展示正式评测、训练设置、gate 动态和 layer gate |
| 配置说明 | `docs/config/20260519_trc_stage1_harness.md` | 当前 TRC stage1 harness 设计与历史结论 |
| 评测表 | `docs/evaluation/20260519_stage1_candidates_eval.md` | 三个候选的正式评测记录 |

## 数据输入

当前 calibration：

```text
/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl
```

每行至少包含：

- `task`: `tool` / `memory` / `code`
- `expert`: 对应专家名
- `rendered_prompt` 或 `prompt`
- `response`: expert 成功轨迹
- `prompt_id`
- `trajectory_id`

当前设计是 96 条，三任务约 32/32/32。训练脚本会做 `validate_calibration_rows()`，检查 expert 是否在 config 中、prompt/response 是否非空。

## 模型与可学习参数

训练脚本加载：

```text
configs/gated_grpo_layer28.yaml
/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json
```

流程：

1. 加载 base causal LM。
2. 冻结 base model 所有参数。
3. 按 mode manifest 安装 gated linear。
4. 用 `make_torch_gate_manager(..., parameterization=layer-band-coefficient)` 创建可学习 gate。
5. 只优化 gate manager 参数。

当前 gate 显示为 28 层 × 3 expert：

```text
layer0.tool, layer0.memory, layer0.code
...
layer27.tool, layer27.memory, layer27.code
```

注意：

- `hidden_layers=8,16,24,28` 是用于计算 hidden-state loss 的观察层，不是可学习 gate 层数。
- bake 后 checkpoint 中仍会涉及更多 module delta entries，当前候选记录为 `num_delta_entries=588`。

## 当前 loss 总公式

对每个 calibration row：

```text
L_total
= L_residual
+ beta_base * L_base_drift
+ gamma_gate * L_gate_anchor
+ coefficient_floor_weight * L_coefficient_floor
```

当前 `dir_i8` 默认：

```text
residual_objective = directional
beta_base = 0.02
gamma_gate = 0.001
coefficient_floor = 0.9
coefficient_floor_weight = 0.05
directional_projection_floor = 0.8
directional_projection_weight = 0.1
residual_weight_power = 0.5
topk_tokens = 128
response_span_mode = auto
accumulation_steps = 96
optimizer = AdamW
lr = 0.03
grad_clip_norm = 1.0
```

`anchor` 版本更保守：

```text
lr = 0.02
beta_base = 0.05
gamma_gate = 0.005
coefficient_floor = 0.95
coefficient_floor_weight = 0.1
```

## 每行 loss 的真实计算步骤

代码：`compute_trc_row_loss()`

### 1. 拼接 prompt + expert response

```text
prompt_ids = tokenizer(prompt)
response_ids = tokenizer(response)
response_ids 截断到 max_response_tokens=512
总长度截断到 max_seq_length=1536
```

如果 prompt 或 response 为空，该行跳过。

### 2. 自动选 response span

代码：`response_token_span()` / `response_char_span()`

```text
tool   -> 优先只对齐 <tool_call>...</tool_call>
code   -> 优先只对齐最长 fenced code block
memory -> 对齐完整 response
fallback -> 找不到结构化 span 时回退完整 response
```

这个设计避免 tool/code 的无关解释文本主导 hidden residual。

### 3. 构造三个 hidden state

对同一条 prompt+response，分别 forward：

```text
h_base   = gate 全 0 时的 hidden states
h_expert = 只打开该 row 对应 expert，系数为 1 时的 hidden states
h_merge  = 当前可学习 gate 下的 hidden states
```

其中 `h_base` 和 `h_expert` 在 `torch.no_grad()` 下计算，只作为 target；`h_merge` 保留梯度，梯度只回到 gate。

### 4. 构造 residual target

对选中的 response token positions 和 hidden layers：

```text
r_target = h_expert - h_base
r_merge  = h_merge  - h_base
```

这就是 TRC 的核心：不是拟合 expert logits，也不是 NLL expert response，而是在隐藏层里让 merged model 包含 expert residual 方向。

### 5. token 权重与 top-k

代码：`hidden_residual_loss()`

每个 token 的权重：

```text
w_t = ||r_target_t|| ^ residual_weight_power
```

当前 `residual_weight_power=0.5`，即弱化大 residual token 的支配性。之后归一化到均值 1。

如果 token 数超过 `topk_tokens=128`，只保留 `w_t` 最大的 128 个 token。

### 6. 当前主 loss：directional residual

当前 v3 使用：

```text
L_dir = mean_t,l [ w_t * (1 - cos(r_merge, r_target)) ]
```

含义：

- 只要求 merged residual 包含目标 expert 能力方向。
- 不要求 `r_merge == r_target`。
- 因此不会把其他 expert 的额外正交 residual 当作错误。

这是从 v2 切到 v3 的关键原因：v2 的 `r_merge ≈ r_expert` 会在 tool 样本上惩罚 memory/code residual，在静态多能力合并中目标不合理。

### 7. projection floor

为了避免只方向接近但幅度太小，加入投影下界：

```text
projection_ratio = dot(r_merge, r_target) / ||r_target||^2
L_proj = mean_t,l [ w_t * relu(floor - projection_ratio)^2 ]
```

当前：

```text
floor = 0.8
projection_weight = 0.1
```

所以：

```text
L_residual = L_dir + 0.1 * L_proj
```

### 8. base drift loss

在 prompt 末尾最多 `prompt_drift_tokens=256` 个 token 上约束 merged hidden 不要偏离 base：

```text
L_base_drift = mean ||h_merge_prompt - h_base_prompt||^2
```

作用：减少 prompt 表示空间被 task vector 过度改写。

### 9. gate anchor loss

```text
L_gate_anchor = mean((gate - init_gate)^2)
```

当前 init 为 1.0。anchor 版本把 `gamma_gate` 加大，防止 gate 过度偏离初始 all-one 合并。

### 10. coefficient floor loss

```text
L_floor = mean(relu(coefficient_floor - gate)^2)
```

作用：防止某个 expert gate 被压得太低。当前不是硬约束，只是软惩罚。

## task-balanced loss

启动脚本打开了：

```text
--task-balanced-loss
```

实现：

```text
scale(task) = total_rows / (num_tasks * count(task))
```

当前三任务基本 32/32/32，所以每个 row 的 scale 约等于 1。保留这个开关是为了后续 calibration 不均衡时仍让 tool/memory/code 每个任务一票。

## optimizer 更新方式

当前训练循环：

1. 每个 epoch 遍历 96 条 calibration rows。
2. 每行计算 `L_total`。
3. `scaled_loss = L_total * task_scale / accumulation_steps`。
4. `scaled_loss.backward()`。
5. `accumulation_steps=96`，所以当前基本是一整个 epoch 累积一次梯度。
6. `clip_grad_norm_(gate_params, 1.0)`。
7. `optimizer.step()`。
8. 如果 gate manager 有 `project_()`，step 后执行投影。

因此当前 TRC 每个 epoch 只有一次 gate update；这和之前 on-policy 小 batch 更新不同。

## 产物结构

每个 TRC run 输出：

```text
trc_run_manifest.json     # 输入 config、mode、calibration、args、任务数量
trc_metrics.jsonl         # row event + epoch event
epoch_001.gates.json      # 每轮 gate values + epoch summary
...
epoch_008.gates.json
trc_gates.json            # final gate
trc_summary.json          # final summary
train.log                 # shell log
```

前端读取：

- `trc_metrics.jsonl` 的 epoch event 画训练动态。
- `epoch_XXX.gates.json` 读取候选 checkpoint 对应的 gate。
- `XXX` 来自候选的 target epoch：`anchor_i4=4`，`anchor_i8=8`，`dir_i8=8`。

## 当前 gate 动态摘要

| candidate | target epoch | tool gate | memory gate | code gate | 解读 |
|---|---:|---:|---:|---:|---|
| `anchor_i4` | 4 | 1.0606 | 0.9199 | 1.0801 | 保守，memory 保留较多 |
| `anchor_i8` | 8 | 1.1068 | 0.8391 | 1.1609 | 更推 tool/code，memory 下降但正式 F1 未掉 |
| `dir_i8` | 8 | 1.1799 | 0.7583 | 1.2416 | 最激进，目前 Tool/Memory 正式评测最好，Code 仍在跑 |

## Bake 步骤

TRC 训练只产出 gate JSON，不直接产出 HF checkpoint。需要 bake：

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
ROOT=/tmp/shared-storage/OnPolicy

$PY scripts/eval/opvec_bake_checkpoint.py \
  --config configs/gated_grpo_layer28.yaml \
  --mode-manifest $ROOT/modes/opvec4/mode_manifest.json \
  --gate-checkpoint /tmp/shared-storage/OnPolicy/runs/trc/<run>/epoch_XXX.gates.json \
  --output /tmp/shared-storage/OnPolicy/checkpoints/<checkpoint_name>
```

已 bake 的当前候选：

```text
/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i4_20260519
/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i8_20260519
/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_dir_i8_20260519
```

## 正式评测步骤

统一入口：

```bash
bash skill/command/run_full_eval_suite.sh /path/to/baked_model model_name
```

### Tool / BFCL

调用：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/scripts/run_bfcl_tool_harness.py
```

默认类别：

```text
parallel, parallel_multiple, live_parallel, live_parallel_multiple
```

前端显示：

- 四个子类 accuracy
- Tool mean
- Tool live mean

### Memory / HotpotQA

调用：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/scripts/run_hotpotqa_memory_harness.py
```

默认数据集：

```text
eval_50
eval_100
eval_qa_1_32768
eval_qa_1_65536
```

前端显示：

- 每个子集 EM / F1 / Sub EM
- mean EM
- mean F1

### Code / CURE

调用：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/Evaluation_all/scripts/run_cure_full_harness.sh
```

默认参数：

```text
CODE_MAX_TEST=8
CODE_MAX_GENERATION_TOKEN=10000
CODE_GPU_GROUPS="[[0,1]]" 或指定其他两卡
```

前端从 `logs/code_cure.log` 解析：

- `START_DATASET LiveBench`
- `START_DATASET LiveCodeBench`
- `code acc`
- `code accumulate acc`
- `BoN setting (4,4)`
- `process n/512`

当前 `dir_i8` Code 正在跑：

```text
tmux: eval_trc_stage1_dir_i8_code_20260519
log: /tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_dir_i8_20260519/stage1_20260519_dir_i8/logs/code_cure.log
```

## 当前正式评测读法

正式模型选择优先级：

1. 三轴都完成的 Tool / Memory / Code 结果。
2. 如果有 pending，只能做临时排序。
3. 不用 calibration loss 直接决定最终模型。

截至本 memory 记录时：

- `dir_i8` Tool mean / live mean 最高。
- `dir_i8` Memory EM / F1 也最高，没有出现 memory 坍缩。
- `dir_i8` Code 仍在正式评测，决定它能否作为 stage1 main checkpoint。
- `anchor_i8` 是当前保守备选：Tool/Code LiveBench 略优于 `anchor_i4`，Memory F1 基本持平。

## 前端步骤

启动：

```bash
tmux new-session -d -s stage1_dashboard_20260519 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && python scripts/monitor/stage1_experiment_dashboard.py --host 0.0.0.0 --port 8802'
```

访问：

```text
http://127.0.0.1:8802
```

当前前端只读，不写训练/评测文件。它显示：

- 正式评测 ranking
- 候选配置与路径
- target epoch gate mean
- 28 层 gate 表
- epoch loss / gate 动态
- Tool/Memory/Code 正式评测细节

## 和 on-policy GRPO/OPD 的边界

当前 TRC stage1：

- 不 rollout 当前 policy。
- 不计算 reward advantage。
- 不使用 GRPO surrogate。
- 不使用 OPD NLL 或 retention NLL。
- 不更新 base model。
- 只用 expert 成功轨迹的 hidden residual 训练 gate。

后续 reward refinement 才会进入 GRPO/OPD/retention 逻辑；不要把 TRC 的 `residual_loss` 和之前 on-policy 的 `ppo_loss_weight/opd_loss_weight/retention_loss` 混为一谈。

## 当前方法的合理性与风险

合理性：

- MSE 直接拟合 expert residual 会惩罚其他 expert 的额外能力；directional loss 更适合静态多专家合并。
- span-aware 让 tool/code 的结构化行为成为主要对齐区域。
- base drift 和 gate anchor 保留 base/instruct 行为，减少过冲。
- coefficient floor 防止某一专家被压到太低。

风险：

- directional loss 仍可能把 code/tool gate 推高、memory gate 压低；需要正式 Memory eval 验证，而不是只看 gate。
- Code 的 hidden residual 对齐不等价于 unit-test reward，最终必须看 CURE。
- 96 条 calibration 仍偏小，TRC 是 strong initializer，不应被包装成最终泛化保证。
- 当前 target epoch 的选择仍依赖正式评测闭环；如果 `dir_i8` Code 下降，需要退回 `anchor_i8` 或双候选进入下一阶段。
