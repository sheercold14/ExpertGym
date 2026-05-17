# Eval Case Browser

这个目录存放 Tool/BFCL 与 Code/CURE 的 case-level 对比前端。前端只读构建后的 JSON，不直接修改原始评测结果。

## 构建数据

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/build_eval_case_browser.py \
  --registry docs/evaluation/eval_case_browser/models.json \
  --output-dir /tmp/shared-storage/OnPolicy/analysis/eval_case_browser
```

输出：

```text
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/app_data.json
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/cases.jsonl
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/pairwise_diffs.jsonl
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/bfcl_live_calibration_candidates.jsonl
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/model_metrics.json
```

`bfcl_live_calibration_candidates.jsonl` 只保存失败结构、tag、优先级和 synthetic data 要求，不保存官方 prompt、ground truth 或模型输出；它用于指导之后构造同分布但不泄漏的 BFCL-live-style calibration。

当前接入模型：

- `best-ever-tame-cg-r1calib-global-v2`
- `expertgym-B-codeaug-opd-i18`
- `ta-c075`

当前接入 benchmark：

- `BFCL`: `parallel` / `parallel_multiple` / `live_parallel` / `live_parallel_multiple`
- `CURE`: `LiveBench` / `LiveCodeBench`

Code/CURE 明细从 CURE `temp_data/outputs-*.json` 中离线抽取。原始 LiveCodeBench 明细每个模型约 `3.9G`，构建脚本使用 event-level parser，只保留 bool table、短 preview 和指标；不要把原始 `test_input` 或执行输出塞进前端数据。

## 启动页面

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/serve_eval_case_browser.py \
  --host 127.0.0.1 \
  --port 8791 \
  --site-dir docs/evaluation/eval_case_browser \
  --data-dir /tmp/shared-storage/OnPolicy/analysis/eval_case_browser
```

访问：

```text
http://127.0.0.1:8791
```

## 添加模型

只需要在 `models.json` 里增加模型条目，指向 BFCL `score` 和 `result` 根目录，再重新运行构建脚本。

每个模型需要：

```json
{
  "model_id": "stable-id",
  "display_name": "Readable name",
  "tool_score_root": ".../score/<model-name>",
  "tool_result_root": ".../result/<model-name>",
  "code_summary_path": ".../summary.json",
  "code_outputs": {
    "LiveBench": ".../outputs-...-LiveBench.json",
    "LiveCodeBench": ".../outputs-...-LiveCodeBench.json"
  },
  "tags": ["optional"]
}
```
