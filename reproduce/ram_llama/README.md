# RAM Llama 论文复现

本目录复现 arXiv:2601.13572 中 **Llama-3.2-3B-Instruct** 扩展实验。论文附录 B.4 的三类评测是：

| domain | evaluator | tasks |
|---|---|---|
| Math | LM-Evaluation-Harness | GSM8K, MATH500 |
| Search | ZeroSearch protocol | NQ, 2WikiMultiHopQA |
| Tool | BFCL AST | Live, Non-Live |

大模型、数据和结果默认放在 `/tmp/shared-storage/ExpertGym/LLaMA`，当前项目只保存配置和脚本。

## 资产

| role | Hugging Face repo |
|---|---|
| base | `meta-llama/Llama-3.2-3B-Instruct` |
| math expert | `sunblaze-ucb/Llama-3.2-3B-Instruct-GRPO-MATH-1EPOCH` |
| tool expert | `chengq9/ToolRL-Llama3.2-3B` |
| search expert | `Alibaba-NLP/ZeroSearch_google_V2_Llama_3.2_3B_Instruct` |

base 模型是 gated Llama，需要先在 Hugging Face 接受 license，然后配置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/shared-storage/ExpertGym/LLaMA/.hf_home
export HF_TOKEN=hf_xxx
```

下载模型、训练数据和评测数据：

```bash
python reproduce/ram_llama/scripts/download_assets.py --skip-gated
python reproduce/ram_llama/scripts/download_assets.py --models base --skip-datasets
python reproduce/ram_llama/scripts/download_assets.py --skip-models --skip-datasets --eval-datasets all
```

## 合并

RAM 使用 `r=0`；RAM+ 使用论文 Llama 组最优设置 `r=0.10`、`alpha=2.0`、`threshold=1e-5`。合并脚本会在输出目录写入 `ram_merge_config.json`，记录参数、专家路径和 unique-region scaling 统计。

```bash
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  bash reproduce/ram_llama/scripts/paper_aligned/run_merge_paper_aligned.sh both
```

输出：

```text
/tmp/shared-storage/ExpertGym/LLaMA/models/merged/ram_math_tool_search
/tmp/shared-storage/ExpertGym/LLaMA/models/merged/ram_plus_math_tool_search
```

## 论文对齐评测

先准备官方源码引用，不复制到本项目：

```bash
bash reproduce/ram_llama/scripts/paper_aligned/prepare_paper_aligned.sh
```

完整评测六个模型：`base,math,tool,search,ram,ram_plus`。

```bash
GPU=0 PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  bash reproduce/ram_llama/scripts/paper_aligned/run_paper_aligned_all.sh llama-paper-aligned
```

Math 默认任务是 `gsm8k,minerva_math500`，并且默认不套 chat template。
汇总表里的 `math_avg` 是论文 Figure 6 对应的 Math 域分数：
`avg(gsm8k_flex, math500_exact)`。`math500_verify` 只作为辅助诊断指标。

常用环境变量：

```bash
MODEL_SET=base,ram_plus        # 只评测部分模型
MAX_SAMPLES=20                 # Search smoke test
RUN_MATH=0 RUN_TOOL=1 RUN_SEARCH=0
SEARCH_BACKEND=wiki            # wiki/google/serper/exa/local_context
SEARCH_URL=http://localhost:6002/retrieve
SER_API_KEY=xxx                # SEARCH_BACKEND=google 时需要
SERPER_API_KEY=xxx             # SEARCH_BACKEND=serper 时需要
SERPER_API_KEY_FILE=/path/key.json
EXA_API_KEY=xxx                # SEARCH_BACKEND=exa 时需要
SEARCH_CACHE_PATH=/path/cache.jsonl
ZEROSEARCH_INVALID_FEEDBACK=1  # 无效动作后继续让 agent 重试，贴近 ZeroSearch 官方交互
```

结果写到：

```text
/tmp/shared-storage/ExpertGym/LLaMA/results/<run_id>/
  run_manifest.txt
  <model>/math_lmeval/
  <model>/bfcl_project/
  <model>/summary.json
  paper_aligned_summary.md
  paper_aligned_summary.json
```

## 单项评测

Math：

```bash
GPU=0 PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  bash reproduce/ram_llama/scripts/paper_aligned/run_math_lmeval.sh \
  /path/to/model my_model my_run
```

Tool：

```bash
GPU=0 PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  bash reproduce/ram_llama/scripts/paper_aligned/run_bfcl_official.sh \
  /path/to/model my_model my_run
```

Search：

```bash
GPU=0 PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python SEARCH_BACKEND=wiki \
  bash reproduce/ram_llama/scripts/paper_aligned/run_search_zerosearch.sh \
  /path/to/model my_model my_run
```

说明：ZeroSearch 官方仓库只提供交互式 inference 入口和训练入口；这里的批量 Search 评测复用了它的 prompt、stop sequence、`<search>/<information>/<answer>` 协议和默认采样设置。要更贴近 `ZeroSearch_google` expert，使用 `SEARCH_BACKEND=google` 或 `SEARCH_BACKEND=serper`；要离线无 API，则启动 wiki retriever 或用 `SEARCH_BACKEND=local_context` 做近似 smoke test。`SEARCH_CACHE_PATH` 默认写到当前 run 目录，同样的 backend/query/topk 会直接复用缓存。

## 快速本地评测

下面是轻量 evaluator，适合快速比较任意模型，但不是论文主结果口径：

```bash
bash reproduce/ram_llama/scripts/run_eval_suite.sh /path/to/model smoke 5
```

已有一次完整近似评测结果：

```text
/tmp/shared-storage/ExpertGym/LLaMA/results/llama-three-domain-full-vllm/aggregate_three_domain_summary.md
```

## 环境缺口

如果 `run_math_lmeval.sh` 报缺少 `lm_eval`：

```bash
pip install -r reproduce/ram_llama/requirements-paper-aligned.txt
```

如果 `run_bfcl_official.sh` 找不到 BFCL，设置：

```bash
export BFCL_REPO=/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard
```
