# Optional Gate PCGrad

## 目的

`opvec_update_gates_from_rollouts.py` 新增了一个默认关闭的 gate-gradient PCGrad 分支，用来实验性缓解 `tool / memory / code` 多任务 gate 梯度冲突。

默认关闭时，训练仍走原有路径：

- row-level loss 仍按原逻辑 `backward()` 累积；
- reward、dynamic OPD scale、task-balanced row scale、task weight、retention、prior、clip、optimizer step 语义不变；
- 不调用 PCGrad helper；
- 不改写已有 `p.grad`；
- 不新增 row 日志字段。

## 新增参数

```bash
--pcgrad-gate-gradients
--pcgrad-eps 1e-12
--pcgrad-task tool --pcgrad-task memory --pcgrad-task code
```

`skill/command/run_qbank_c033333_gate_strategy.sh` 对应环境变量：

```bash
PCGRAD_GATE_GRADIENTS=1
PCGRAD_EPS=1e-12
PCGRAD_TASKS=tool,memory,code
```

限制：

- 第一版只支持 `--optimizer-step-scope epoch`。
- 如果开启 PCGrad 但 step scope 不是 `epoch`，脚本直接报错。
- `--pcgrad-task` 指定的是参与投影的 task allowlist；未列入的已观察 task 仍保留原始 task gradient，不被丢弃。

## 开启后的行为

PCGrad 只在 `_UpdateBatcher.flush(..., force=True)` 的 `optimizer.step()` 前介入：

1. 丢弃当前已累计的总梯度。
2. 按 task 重算 gate 参数梯度。
3. 每个 task 梯度包含该 task 的 frontier GRPO、OPD distill、OPD all-success、retention。
4. prior / regularizer 不参与 PCGrad 投影。
5. PCGrad 投影后，将 `sum(projected_task_grads) + regularizer_grad` 写回 gate 参数的 `p.grad`。
6. 后续仍走原有 `clip_grad_norm_ -> optimizer.step() -> projection`。

## 日志

只有开启 PCGrad 时，row 日志会额外记录：

- `pcgrad_enabled`
- `pcgrad_conflict_count`
- `pcgrad_task_grad_norms`
- `pcgrad_regularizer_grad_norm`
- `pcgrad_pre_cosines`
- `pcgrad_post_cosines`

## 非目标

本次只增加可选梯度后处理，不修改：

- code reward threshold；
- dynamic OPD 样本选择；
- OPD positive / negative 逻辑；
- task weight 语义；
- retention trust region；
- memory over-push 控制。
