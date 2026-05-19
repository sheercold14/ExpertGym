# 2026-05-19 Stage1 Formal-Eval Dashboard

## 功能

`scripts/monitor/stage1_experiment_dashboard.py` 是只读轻量前端，用 Python 标准库 `ThreadingHTTPServer` 提供：

- 首页顶部 best candidate ranking：按已完成的 Tool mean / Memory F1 / Code Acc 求均值排序，并显式标出 pending 轴。
- 候选实验设置：展示每个 checkpoint 的训练族、epoch、init、lr、anchor/regularization、objective、calibration、checkpoint/run/eval 路径。
- 正式评测口径：Tool 展示 BFCL 子类与 live mean；Memory 展示 HotpotQA 子集、EM/F1；Code 展示 LiveBench/LiveCodeBench 的 Acc、TP、BoN 和 progress。
- 阅读引导：说明当前排序是否只是临时排序，哪些候选仍缺 Code，哪些候选需要优先补齐正式评测。

页面每 30 秒刷新 `/api/state`，服务端每次实时扫描本地日志/JSON，不写训练或 eval 主逻辑。

## 数据源

评测候选在脚本内 `EVALS` 字典维护：

- `anchor_i4`: `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_anchor_i4_20260519/stage1_20260519`
- `anchor_i8`: `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_anchor_i8_20260519/stage1_20260519_rerun`
- `dir_i8`: `/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_dir_i8_20260519/stage1_20260519_dir_i8`

解析规则：

- Tool: `logs/tool_bfcl.log` 末尾 JSON 的 `scores.*.accuracy`。
- Memory: `logs/memory_hotpotqa.log` 末尾 JSON 的 `datasets.*.avg_f1`、EM、Sub EM。
- Code: `logs/code_cure.log` 的 `START_DATASET` / `END_DATASET`、`code acc`、`code accumulate acc`、`BoN setting (4,4)` 和 `process n/total` 进度。

当前前端只展示本轮 TRC stage-1 候选；`M1/M2` 等训练诊断组已经隐藏，不参与当前模型选择。

## 启动命令

当前服务按要求启动在 tmux session `stage1_dashboard_20260519`：

```bash
tmux new-session -d -s stage1_dashboard_20260519 \
  'cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && python scripts/monitor/stage1_experiment_dashboard.py --host 0.0.0.0 --port 8802'
```

访问：

- 本机：`http://127.0.0.1:8802`
- 远端：`http://<host>:8802`

检查：

```bash
curl -s http://127.0.0.1:8802/api/state | python -m json.tool | head
tmux capture-pane -pt stage1_dashboard_20260519
```

停止：

```bash
tmux kill-session -t stage1_dashboard_20260519
```

## 维护方式

新增候选评测时，只需在 `EVALS` 添加：

```python
"new_eval_id": {
    "label": "display name",
    "path": "/path/to/full_suite/run_dir",
    "checkpoint": "checkpoint_name_for_code_logs",
    "checkpoint_path": "/path/to/baked_checkpoint",
    "train_run": "/path/to/training_run",
    "train_command": "skill/command/xxx.sh",
    "setting": {
        "family": "TRC directional",
        "epoch": "8",
        "init": "1.0",
        "lr": "0.03",
        "objective": "1 - cos(r_merge, r_expert)",
        "calibration": "trc96 expert trajectories",
    },
}
```

保持数据源只读；不要在 dashboard 中触碰训练/eval 主逻辑。若后续 Code 评测写出稳定的小型 summary JSON，可在 `parse_code_eval()` 中优先读取 summary，再回退到 CURE log 解析和 `process n/total` 进度。
