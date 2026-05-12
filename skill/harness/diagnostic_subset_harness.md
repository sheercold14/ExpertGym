# Diagnostic Subset Harness

本说明记录 AgentMerging/ExpertGym 当前使用过的轻量子集评测协议。该协议用于模型合并方法的快速诊断和方法迭代，不用于 untouched final claim。

## 入口脚本

主 harness：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/AgentMerging_plan/scripts/run_model_eval_suite.py
```

历史示例：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/tc-wudi-subset-20260506/commands.sh
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/tc-wudi-c-contrastive-subset-20260507/commands.sh
```

标准 eval config：

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/tc-wudi-subset-20260506/eval_tc_wudi.json
```

## 子集 Benchmark

| Name | Adapter | Data | Main Metric | Notes |
|---|---|---|---|---|
| `toolrl-rlla4k-test` | `toolrl` | `/tmp/shared-storage/dataset/ToolRL/dataset/rlla_4k/test.parquet` | `accuracy`, `avg_score` | 80 samples |
| `hotpotqa-eval-50` | `hotpotqa_vllm` | `/tmp/shared-storage/dataset/hotpotqa/eval_50.json` | SubEM/`accuracy`, `avg_f1` | 128 samples |
| `hotpotqa-eval-100` | `hotpotqa_vllm` | `/tmp/shared-storage/dataset/hotpotqa/eval_100.json` | SubEM/`accuracy`, `avg_f1` | 128 samples |
| `codecontests-train-diagnostic` | `codecontests` | `/tmp/shared-storage/dataset/CodeContests_train/train/CodeContests_train.json` | `accuracy`, `avg_test_pass_rate` | usually `--max-samples 50` |

All four benchmarks are configured as:

```json
"eval_role": "diagnostic_visible",
"allowed_for_overlap_analysis": true,
"allowed_for_final_claim": false,
"used_for_method_design": false
```

## Minimal Models Config

Create a JSON file:

```json
{
  "models": [
    {
      "name": "my-model",
      "role": "method_name",
      "path": "/path/to/checkpoint/model",
      "enabled": true,
      "notes": "diagnostic subset evaluation only"
    }
  ]
}
```

The checkpoint must contain a valid HF/Qwen-style model directory, including `config.json`, tokenizer files, and safetensors shards.

## Standard Commands

Tool + Memory:

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
HARNESS=/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/AgentMerging_plan/scripts/run_model_eval_suite.py
EVAL_ROOT=/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation
EVAL_CFG=${EVAL_ROOT}/tc-wudi-subset-20260506/eval_tc_wudi.json

${PY} ${HARNESS} \
  --models-config /path/to/models.json \
  --eval-config ${EVAL_CFG} \
  --run-id my-subset-run \
  --output-root ${EVAL_ROOT} \
  --feedback-root ${EVAL_ROOT} \
  --models my-model \
  --benchmarks toolrl-rlla4k-test,hotpotqa-eval-50,hotpotqa-eval-100 \
  --gpu 7 \
  --continue-on-error
```

CodeContests s50:

```bash
${PY} ${HARNESS} \
  --models-config /path/to/models.json \
  --eval-config ${EVAL_CFG} \
  --run-id my-subset-run \
  --output-root ${EVAL_ROOT} \
  --feedback-root ${EVAL_ROOT} \
  --models my-model \
  --benchmarks codecontests-train-diagnostic \
  --gpu 7 \
  --max-samples 50 \
  --continue-on-error
```

Running Tool/Memory and Code as two commands is preferred. It keeps partial results usable if one adapter fails.

## Output Layout

For `run_id=my-subset-run`, outputs are written under:

```text
${EVAL_ROOT}/my-subset-run/
```

Each model/benchmark writes:

```text
evaluations/<model>/<benchmark>/metrics.json
evaluations/<model>/<benchmark>/predictions.jsonl
evaluations/<model>/<benchmark>/failures.jsonl
evaluations/<model>/<benchmark>/metadata.json
evaluations/<model>/<benchmark>/report.md
```

Important fields in `metrics.json`:

- ToolRL: `accuracy`, `avg_score`, `total`, `correct`
- HotpotQA: `accuracy` means SubEM, plus `exact_match_rate`, `avg_f1`, `boxed_rate`
- CodeContests: `accuracy` means all public tests pass, plus `avg_test_pass_rate`, `code_extract_rate`

## Known Completed Four-Benchmark Runs

As of 2026-05-09, this subset harness has many historical runs. Four-benchmark-complete methods include WUDI, TA sweep points, TAME/CG, ExpertMerge checkpoints, VGEC variants, UER variants, and the new Auto-GBIS layer candidate.

Top complete runs by simple macro over ToolRL Acc, HotpotQA eval_50 SubEM, HotpotQA eval_100 SubEM, and CodeContests s50 Acc:

| Rank | Model | Run | Macro4 | Tool | Mem50 | Mem100 | Code |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `vgec-all-recoverable-traj` | `vgec-stage1-behavior-guided-full-20260507-all` | 0.6366 | 0.6250 | 0.7969 | 0.8047 | 0.3200 |
| 2 | `vgec-nosota-recoverable-traj` | `vgec-ablation-full-20260507-nosota` | 0.6330 | 0.6250 | 0.7812 | 0.7656 | 0.3600 |
| 3 | `vgec-nosota-bfcltool-traj` | `vgec-bfcltool-subset-20260508` | 0.6250 | 0.5400 | 0.8200 | 0.7800 | 0.3600 |
| 4 | `sota-tame-cg-r1calib-global-v2-20260504` | `vgec-stage0-behavior-matrix-20260507-sota` | 0.6149 | 0.6250 | 0.7734 | 0.7812 | 0.2800 |
| 5 | `ta-c100` | `vgec-stage0-behavior-matrix-20260507-ta-c100` | 0.6138 | 0.6250 | 0.7812 | 0.7891 | 0.2600 |
| 6 | `vgec-sota-only-recoverable-traj` | `vgec-ablation-full-20260507-sotaonly` | 0.6102 | 0.6250 | 0.7344 | 0.7812 | 0.3000 |
| 7 | `vgec-nosota-uer-traj` | `vgec-uer-subset-20260508-nosota` | 0.6100 | 0.5400 | 0.8200 | 0.7800 | 0.3000 |
| 8 | `expertmerge-0503-220915` | `vgec-stage0-behavior-matrix-20260507-em-0503-220915` | 0.6062 | 0.6250 | 0.7656 | 0.7344 | 0.3000 |
| 9 | `ta-c075` | `vgec-stage0-behavior-matrix-20260507-ta-c075` | 0.6010 | 0.6250 | 0.7812 | 0.7578 | 0.2400 |
| 10 | `vgec-bfclheldout0503-toolbeta-up-traj` | `vgec-tool-proj-up-subset-20260508` | 0.6000 | 0.5400 | 0.7600 | 0.8200 | 0.2800 |
| 19 | `auto-gbis-layer-hybrid` | `auto-gbis-layer-subset-20260509` | 0.5584 | 0.6375 | 0.6797 | 0.6562 | 0.2600 |

The Auto-GBIS layer candidate is not top by macro because memory lags the best VGEC/TA runs, but it is tied for the best ToolRL accuracy among complete runs and improves over several WUDI baselines on Tool and Code.

## Current Auto-GBIS Layer Result

Run:

```text
/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/auto-gbis-layer-subset-20260509
```

Metrics:

| Benchmark | Result |
|---|---:|
| ToolRL rlla4k-test Acc | 51/80 = 0.6375 |
| ToolRL avg_score | 0.8755 |
| HotpotQA eval_50 SubEM | 87/128 = 0.6797 |
| HotpotQA eval_50 F1 | 0.6833 |
| HotpotQA eval_100 SubEM | 84/128 = 0.6562 |
| HotpotQA eval_100 F1 | 0.6466 |
| CodeContests train s50 Acc | 13/50 = 0.2600 |
| CodeContests avg_test_pass_rate | 0.3260 |

## Inventory Command

To list all historical subset metrics:

```bash
find /mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation \
  -path '*/evaluations/*/*/metrics.json' -print
```

To aggregate all four-benchmark-complete methods, parse `metrics.json` grouped by:

```text
<run_id>/evaluations/<model_name>/<benchmark>/metrics.json
```

and require all four benchmark names:

```text
toolrl-rlla4k-test
hotpotqa-eval-50
hotpotqa-eval-100
codecontests-train-diagnostic
```

## Guardrails

- Treat these results as diagnostic-visible only.
- Do not mix these subset numbers with full BFCL/HotpotQA/CURE final tables.
- Always report `run_id`, model path, benchmark names, and `--max-samples` for CodeContests.
- CodeContests executes generated Python locally; run only in the intended evaluation environment.
- Some older runs are single-task ablations. Do not compare a Code-only run's single metric against a four-task macro without labeling it as partial.
