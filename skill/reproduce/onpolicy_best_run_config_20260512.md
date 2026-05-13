# OnPolicy Gated-GRPO 今晚最佳可跑配置

日期：2026-05-12  
仓库：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym`  
目标：先跑一个**正确性优先**的 1/3 task-vector 初始点训练，监控 reward 和 task-vector 系数变化，避免今天定位到的 vLLM old-logprob 错梯度问题。

## 结论先行

今晚主线配置：

```text
rollout: vLLM 只生成文本，不保存 token old_logprobs
reward: 官方 Tool / MemAgent / Code reward
update: HF 侧统一补 old_logprobs，并计算 current logprobs
loss: token-level GRPO
batch: update_batch_size=4，optimizer 层面 batch
rollout 加速: ROLLOUT_SHARDS=auto，多 GPU 分片 rollout 后自动 merge
init: global gate = 1/3, 不是 0.75
```

核心参数：

```bash
LOSS_GRANULARITY=token
STORE_TOKEN_LOGPROBS=0
UPDATE_BATCH_SIZE=4
TASK_NORMALIZE_ADVANTAGES=1
LENGTH_NORMALIZE_POLICY_LOGPROB=1
ROLLOUT_SHARDS=auto
FRONTIER_ORDER=task-interleaved
```

不要用：

```bash
STORE_TOKEN_LOGPROBS=1
```

原因：固定同一份 rollout 的对照显示，`token + vLLM old_logprob` 会把 tool/memory gate 从 `0.3333` 推到约 `0.316`，而 `token + HF-fill old_logprob` 会让 tool 上涨到约 `0.340`，memory/code 基本保持。也就是说 vLLM old 与 HF current 的 logprob 不同源，PPO ratio 会错。

## 推荐启动命令

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

export RUN_NAME=qbank_c033333_global_token_hffill_sharded_100x4_i2_$(date +%Y%m%d_%H%M%S)
export RUN_DIR=/tmp/shared-storage/OnPolicy/runs/gated_grpo/$RUN_NAME

tmux new-session -d -s opvec_$RUN_NAME "\
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && \
RUN_NAME=$RUN_NAME \
RUN_DIR=$RUN_DIR \
GPU_LIST=0,1,2,3,4,5,6,7 \
ROLLOUT_GPUS=0,1,2,3,4,5,6,7 \
ROLLOUT_SHARDS=auto \
ROLLOUT_SHARD_STAGGER_SECONDS=8 \
STRATEGY=global \
INIT_VALUE=0.3333333333333333 \
NUM_ITERS=2 \
NUM_PROMPTS=100 \
SAMPLES_PER_PROMPT=4 \
MAX_NEW_TOKENS=1024 \
MAX_PROMPT_TOKENS=8192 \
MAX_MODEL_LEN=12288 \
MAX_LOGPROB_TOKENS=8192 \
ROLLOUT_BATCH_SIZE=32 \
TENSOR_PARALLEL_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.82 \
POST_BAKE_SLEEP_SECONDS=5 \
LOSS_GRANULARITY=token \
FRONTIER_ORDER=task-interleaved \
STORE_TOKEN_LOGPROBS=0 \
UPDATE_BATCH_SIZE=4 \
BATCH_LOSS_REDUCTION=mean \
TASK_NORMALIZE_ADVANTAGES=1 \
LENGTH_NORMALIZE_POLICY_LOGPROB=1 \
LR=0.003 \
PRIOR_LOSS_WEIGHT=0.02 \
MAX_COEFF_DELTA=0.05 \
PYTHONDONTWRITEBYTECODE=1 \
bash skill/command/run_qbank_c033333_gate_strategy.sh 2>&1 | tee $RUN_DIR/run.log"
```

进入训练窗口：

```bash
tmux attach -t opvec_$RUN_NAME
```

只看日志：

```bash
tail -f $RUN_DIR/run.log
```

GPU 监控：

```bash
watch -n 2 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

## 前端监控

如果 8771 端口空闲：

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/monitor/opvec_run_monitor.py \
  --host 0.0.0.0 \
  --port 8771 \
  --run-dir "$RUN_NAME=$RUN_DIR"
```

浏览器访问：

```text
http://127.0.0.1:8771
```

如果已有监控进程，可改端口：

```bash
--port 8772
```

## 运行阶段怎么看

每轮结构：

```text
iter_i/
  baked_policy/          # 当前 gate 烘焙成 HF/vLLM checkpoint
  rollouts.jsonl         # vLLM 生成 + reward 后的样本
  rollouts.summary.json  # rollout timing / reward summary
  gate_updates.jsonl     # 每个 frontier row 的 update 日志
  gate_updates.summary.json
  gate_updates.gates.json
```

关键字段：

```text
rollouts.summary.json
  token_logprob_samples 应为 0

gate_updates.summary.json
  filled_missing_old_logprobs 应大于 0
  frontier_task_counts 看三类任务是否都有训练信号

gate_updates.gates.json
  common + *_residual = 对应 expert 的实际 task-vector 系数
```

快速看 gate：

```bash
python - "$RUN_NAME" <<'PY'
import json, pathlib, sys
run = pathlib.Path("/tmp/shared-storage/OnPolicy/runs/gated_grpo") / sys.argv[1]
init = 1/3
for it in sorted(run.glob("iter_*")):
    p = it / "gate_updates.gates.json"
    if not p.exists():
        continue
    raw = json.loads(p.read_text()).get("gates", {})
    c = float(raw["common"])
    eff = {k: c + float(raw[f"{k}_residual"]) for k in ("tool", "memory", "code")}
    delta = {k: round(v - init, 6) for k, v in eff.items()}
    print(it.name, "eff=", {k: round(v, 6) for k, v in eff.items()}, "delta=", delta)
PY
```

## 为什么这是今晚最佳配置

### 1. 避免已确认的 vLLM old-logprob 问题

错误路径：

```text
old_logprobs: vLLM 生成时保存
current_logprobs: HF update 侧重新计算
```

风险：

```text
tokenizer / chat template / token 对齐 / logprob 语义任一处不一致，ratio = exp(current-old) 就会错。
```

实验证据：

```text
固定 24 rows，同一 frontier:

token + vLLM old:
  tool   0.316037
  memory 0.315810
  code   0.332518

token + HF-fill old:
  tool   0.340001
  memory 0.333257
  code   0.333910
```

因此今晚不要为了省一次 HF old-logprob forward 使用 vLLM token old。

### 2. 保留 token-level loss

推荐：

```text
LOSS_GRANULARITY=token
```

原因：

- Memory 是长轨迹：`memory_update turns + final answer`。
- sequence-level 会把整段 logprob 压成一个总分，credit assignment 太粗。
- 固定对照里 sequence + HF-fill 会压低 memory，尤其 row-by-row 更明显。
- token-level + HF-fill old 在同一份 rollout 上方向更稳。

### 3. 使用 task-level advantage normalization

当前命令开启：

```bash
TASK_NORMALIZE_ADVANTAGES=1
```

训练逻辑：

```text
每个 prompt 内:
  advantage = frontier_weight * (reward - mean(reward)) / std(reward)

每个 task 内:
  统计 mean(abs(advantage))

跨 task:
  令 tool / memory / code 的平均 advantage 尺度接近
```

它归一化的是 policy gradient 的 advantage 尺度，不是 raw reward 本身。它能缓解 Tool reward 范围大、Code/Memory reward 范围小的问题，但不能修复 old/current logprob 不同源。

### 4. 多卡 rollout 分片

命令中：

```bash
ROLLOUT_SHARDS=auto
ROLLOUT_GPUS=0,1,2,3,4,5,6,7
```

含义：

```text
100 prompts 会按 GPU 数量切成多个 prompt range。
每个 vLLM 进程只用 1 张卡。
每个 shard 写 rollouts.shard_XX.jsonl。
loop 自动 merge 成 iter_i/rollouts.jsonl。
update 阶段只读取 merge 后的 rollouts.jsonl。
```

`ROLLOUT_SHARD_STAGGER_SECONDS=8` 是为了避免 8 个 vLLM 进程同时从共享存储加载 8 份 baked checkpoint。

## 预计时间

以 `100 prompts * 4 samples * max_new_tokens=1024` 粗估：

```text
bake:    ~1 min
rollout: sharded 后约 4-8 min，受 memory/code 长输出影响
update:  HF-fill old + current/backward 约 18-30 min
total:   单轮约 25-40 min
```

如果 `MAX_NEW_TOKENS=2048`：

```text
单轮可能到 40-70 min
```

今晚先用 1024 跑通趋势；如果 memory 明显因为截断失败，再单独加长 memory 或跑 2048 对照。

## 结果判断标准

优先看：

```text
1. iter_001 后 gate 是否没有整体塌到 1/3 以下
2. tool / memory / code 是否至少有一个方向持续上涨
3. reward 是否不是纯随机波动
4. frontier_task_counts 是否三类任务都有有效行
5. filled_missing_old_logprobs 是否 > 0
```

危险信号：

```text
token_logprob_samples > 0
filled_missing_old_logprobs = 0
tool/memory/code 三者连续两轮全部下降
frontier_task_counts 某一类长期为 0
reward 全对比例过高导致 frontier 过少
```

如果出现危险信号，先停实验，不要继续多轮硬训。

## 与昨晚 native 的关系

昨晚上涨的版本更接近：

```text
sequence + HF-fill old + row update
```

今晚不完全复刻它，因为今天固定对照显示 sequence 对 Memory 有压制倾向。今晚推荐：

```text
token + HF-fill old + batch update
```

这是在保持 old/current 同源的前提下，给 Memory 更细 token-level credit assignment 的版本。
