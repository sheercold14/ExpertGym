# 实验配置与结果索引

这个目录用于记录每次 OnPolicy / Gated-GRPO 实验的可复现配置、结果路径和分析备注。原则是：所有正式启动过的实验，都要能从这里反查到“当时为什么这样跑、从哪个 checkpoint 接、loss 怎么配、结果在哪里”。

## 记录规范

每个实验批次单独建一个 Markdown：

```text
YYYYMMDD_short_name.md
```

每个批次至少记录：

| 字段 | 内容 |
|---|---|
| 实验目的 | 要验证的假设，例如 OPD-only 后期是否动力不足 |
| run_dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/...` |
| 初始化 | gate checkpoint、optimizer state 是否继承 |
| 参数化 | `global-coefficient` / `global-parameter` / `parameter` 等 |
| 数据 | seed manifest、prompt 数、samples-per-prompt、expert rollout |
| loss | GRPO/OPD/retention/PCGrad/normalization/任务均衡 |
| 优化器 | lr、SGD/AdamW、momentum、step scope、batch size |
| rollout | GPU、vLLM shards、token length、temperature/top-p |
| 结果路径 | loop manifest、per-iter summary、train.log、baked_policy、eval report |
| 观察结论 | proxy reward、gate 变化、all-fail/all-success、异常 |

## 当前活跃索引

- [20260515_opd_continue_abcd.md](20260515_opd_continue_abcd.md): A/B OPD-only continuation 与 C/D GRPO+OPD+retention PCGuard continuation。

## 结果文件优先级

排查问题时优先看：

1. `gated_grpo_bake_vllm_loop_manifest.json`: 每轮 bake / rollout / update 命令、耗时、输入输出。
2. `iter_*/gate_updates.summary.json`: loss 配置、frontier/retention/OPD 统计、optimizer 配置、gate 输出。
3. `iter_*/opd_distill_from_allfail.summary.json`: dynamic OPD 选择了哪些 all-fail 样本。
4. `iter_*/rollouts.summary.json`: rollout 样本数、任务分布、生成耗时。
5. `train.log`: tmux 内部完整运行日志。
6. `docs/report/*.md`: 实验解释、评测结果、失败分析。
