# OP-VEC Gated-GRPO VeRL 路线说明

更新时间：2026-05-12

## 目标

把 OP-VEC gated-GRPO 接到 VeRL 原生主循环：

```text
VeRL actor: Qwen base + OP-VEC GatedLinear + trainable gate manager
VeRL rollout: vLLM
VeRL reward: OP-VEC RewardRouter / official verifier
VeRL update: GRPO/PPO actor update, 只更新 gate 参数
```

核心权重仍然是：

```text
W_eff = W_base
      + c_tool   * Delta_tool
      + c_memory * Delta_memory
      + c_code   * Delta_code
```

rollout 侧不能直接消费 `GatedLinear.base_linear / delta_* / gate_manager` 的内部 state_dict。当前 OP-VEC 适配在 actor-to-vLLM 同步点导出普通 HF key 的 `W_eff`，让 vLLM 看到常规模型权重。

## 负责范围内代码

```text
verl/experimental/opvec/
  path_utils.py       # 把 VeRL checkout 接到 OP-VEC repo
  external_lib.py     # VeRL external_lib hook，构建 HF actor 时安装 OP-VEC gates
  gate_actor.py       # 安装 gate actor；从 FSDP/普通模型导出 W_eff state_dict
  reward_fn.py        # VeRL custom_reward_function -> OP-VEC RewardRouter
  prepare_data.py     # OP-VEC prompt manifest -> VeRL parquet

examples/opvec_gated_grpo/
  run_smoke10_native_bridge.sh # 旧 bridge 对照
  run_verl_grpo_smoke10.sh     # VeRL main_ppo 原生入口
```

同时有 OP-VEC 权重同步适配：

```text
verl/workers/engine/fsdp/transformer_impl.py
verl/workers/engine/automodel/transformer_impl.py
verl/workers/engine/veomni/transformer_impl.py
```

当模型里存在 `opvec_gate_manager` 时，`get_per_tensor_param()` 同步 `export_effective_hf_state_dict(model)`；否则保持 VeRL 原逻辑。

## BFCL_verl 环境

环境路径：

```bash
/tmp/shared-storage/OnPolicy/envs/BFCL_verl
```

该环境由原 BFCL clone 得到，原 BFCL 环境未修改。补装 VeRL 所需最小依赖：

```bash
/tmp/shared-storage/OnPolicy/envs/BFCL_verl/bin/python -m pip install \
  omegaconf hydra-core tensordict torchdata codetiming wandb liger-kernel peft
```

当前关键版本：

```text
python==3.10.20
torch==2.10.0+cu126
vllm==0.19.0
ray==2.55.1
transformers==4.57.6
datasets==4.8.5
omegaconf==2.3.0
hydra-core==1.3.2
tensordict==0.12.2
torchdata==0.11.0
codetiming==1.4.0
wandb==0.26.1
liger-kernel==0.8.0
peft==0.19.1
flash-attn: not installed
```

注意：这套环境目前用于定位问题，不应视为最终可迁移训练环境。当前 VeRL checkout 的
`setup.py` 写死的 vLLM 约束是：

```text
ray[default]>=2.41.0
tensordict>=0.8.0,<=0.10.0,!=0.9.0
vllm>=0.8.5,<=0.12.0
```

因此 `BFCL_verl` 里的 `vllm==0.19.0`、`tensordict==0.12.2` 已经超出这个 checkout 的
声明支持范围。此前的核心 blocker 是版本栈不兼容，而不是 OP-VEC reward/gate/数据适配错误。

## 当前原因

主要问题不是 OP-VEC 的 gate、reward 或数据本身，而是 VeRL 原生异步 rollout 对版本栈很敏感。

已定位到的原因链：

```text
P0: BFCL_verl 使用 vllm==0.19.0 / ray==2.55.1 / torch==2.10.0，超出当前 VeRL checkout 的声明支持范围。
    现象：裸 vLLM 可启动，但 Ray actor 内 vLLM EngineCore 初始化失败。

P1: /root/.cache/huggingface/datasets 在当前容器里有 filelock 释放异常。
    处理：HF cache 默认转到 $ROOT/cache/huggingface。

P2: VeRL async agent loop 默认 num_workers=8，小 smoke 只有 LIMIT * n = 4 条请求，DataProto 无法等分。
    处理：run_verl_grpo_smoke10.sh 暴露 AGENT_LOOP_NUM_WORKERS，默认等于 GPU 数。

P3: 新环境没有 flash-attn，VeRL old_logprob 的 padding 工具默认依赖 flash_attn.bert_padding。
    处理：verl/utils/attention_utils.py 增加纯 torch fallback；有 flash-attn 时仍走原路径。
```

当前已验证：换到 VeRL 支持范围内的 `vllm==0.12.0` 后，Ray actor 内 vLLM server 能启动，2 条数据 tiny smoke 已经完整跑过 rollout、reward、old logprob、gate update 和 vLLM 权重同步。

## 可迁移环境准则

迁移到其他服务器时，复现“VeRL 支持范围内”的环境，不复制 `BFCL_verl` 的失败版本栈：

```text
Python: 3.10
Torch: 跟随 vLLM 0.12.0 wheel 自动解析；当前解析为 torch==2.9.0
vLLM: 0.12.0
Ray: 2.49.2
tensordict: 0.10.0
numpy: <2.0.0
flash-attn: 可选；未安装也可跑，依赖本地 torch fallback，命令保留 sdpa override
HF cache: 默认放到 $ROOT/cache/huggingface，不使用 /root/.cache/huggingface/datasets
```

已跑通环境：

```bash
/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310
```

当前关键版本：

```text
python==3.10.20
torch==2.9.0
vllm==0.12.0
ray==2.49.2
transformers==4.57.6
datasets==4.8.5
numpy==1.26.4
tensordict==0.10.0
torchdata==0.11.0
verl==0.8.0.dev0
flash-attn: not installed
```

推荐迁移安装命令：

```bash
conda create -p /tmp/shared-storage/OnPolicy/envs/verl_opvec_py310 python=3.10 -y

/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python -m pip install -U pip setuptools wheel

/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python -m pip install \
  "numpy<2.0.0" \
  "tensordict==0.10.0" \
  "ray[default]==2.49.2" \
  "vllm==0.12.0"

/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python -m pip install -e \
  /mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/third_party/verl
```

`run_verl_grpo_smoke10.sh` 已默认设置：

```bash
HF_HOME=${HF_HOME:-$ROOT/cache/huggingface}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
HF_MODULES_CACHE=${HF_MODULES_CACHE:-$HF_HOME/modules}
```

迁移服务器时如果换 cache 目录，先用最小 `FileLock` 检查该目录支持正常加锁/释放。训练时还要保证：

```text
LIMIT * SAMPLES_PER_PROMPT 能被 AGENT_LOOP_NUM_WORKERS 整除。
单卡小 smoke 推荐 AGENT_LOOP_NUM_WORKERS=1。
多卡正式训练可设 AGENT_LOOP_NUM_WORKERS=$NGPUS_PER_NODE 或其约数。
```

迁移后的 health check 必须按顺序跑：

```bash
# A. 版本检查
/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python - <<'PY'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for name in ["torch", "vllm", "ray", "transformers", "tensordict", "flash-attn"]:
    try:
        print(name, md.version(name))
    except Exception:
        print(name, "NOT_INSTALLED")
PY

# B. VeRL 配置解析；不占 GPU
env PY=/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python \
  GPU_LIST=0 LIMIT=2 MAX_GATED_MODULES=1 MAX_RESPONSE_LENGTH=64 \
  RUN_NAME=verl_opvec_cfgcheck \
  bash third_party/verl/examples/opvec_gated_grpo/run_verl_grpo_smoke10.sh --cfg job

# C. 裸 vLLM dummy 启动；确认 vLLM 自身可用
PYTHONPATH=/mnt/cache/wuruixiao/users/lsc/AgentMerging/worktree/OnPolicyMerge_gated_grpo/third_party/verl:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct \
  --load-format dummy \
  --worker-extension-cls verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension \
  --max-model-len 2304 \
  --host 127.0.0.1 \
  --port 18080

# D. VeRL tiny smoke；这是最终判据
env PY=/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python \
  GPU_LIST=0 \
  LIMIT=2 \
  MAX_GATED_MODULES=1 \
  MAX_RESPONSE_LENGTH=64 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_MODEL_LEN=2304 \
  ROLLOUT_GPU_MEM_UTIL=0.30 \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 \
  ROLLOUT_MAX_NUM_SEQS=8 \
  AGENT_LOOP_NUM_WORKERS=1 \
  SAMPLES_PER_PROMPT=2 \
  RUN_NAME=verl_opvec_tiny_migration_check \
  RAY_TMPDIR=/tmp/rvop \
  WANDB_MODE=disabled \
  VLLM_LOGGING_LEVEL=INFO \
  bash third_party/verl/examples/opvec_gated_grpo/run_verl_grpo_smoke10.sh \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
```

判据：

```text
A 失败：环境包没装对。
B 失败：Hydra/OP-VEC 参数或 PYTHONPATH 问题。
C 失败：vLLM 自身不可用，先别看 VeRL。
D 失败且 C 成功：优先看 VeRL Ray actor + vLLM server 集成、agent loop worker 切分、flash-attn/fallback 三类问题。
```

## 启动命令

配置级检查：

```bash
env PY=/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python \
  GPU_LIST=0,1 LIMIT=2 MAX_GATED_MODULES=1 MAX_RESPONSE_LENGTH=64 \
  RUN_NAME=verl_opvec_tiny_cfgcheck \
  bash third_party/verl/examples/opvec_gated_grpo/run_verl_grpo_smoke10.sh --cfg job
```

tiny smoke 使用短 Ray 临时目录，避免 Unix socket 路径超过 107 字节：

```bash
env PY=/tmp/shared-storage/OnPolicy/envs/verl_opvec_py310/bin/python \
  GPU_LIST=7 \
  LIMIT=2 \
  MAX_GATED_MODULES=1 \
  MAX_RESPONSE_LENGTH=64 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_MODEL_LEN=2304 \
  ROLLOUT_GPU_MEM_UTIL=0.30 \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 \
  ROLLOUT_MAX_NUM_SEQS=8 \
  AGENT_LOOP_NUM_WORKERS=1 \
  SAMPLES_PER_PROMPT=2 \
  RUN_NAME=verl_opvec_tiny_migration_check \
  RAY_TMPDIR=/tmp/rvop \
  WANDB_MODE=disabled \
  VLLM_LOGGING_LEVEL=INFO \
  bash third_party/verl/examples/opvec_gated_grpo/run_verl_grpo_smoke10.sh \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
```

`attn_implementation=sdpa` 是为了不强依赖 `flash-attn`。当前环境没有安装 `flash-attn`，VeRL 的 padding 工具已经有 torch fallback，可以先保证迁移可跑。

## 已完成验证

已确认 easyrl/vLLM 0.8 兼容补丁没有残留：

```text
verl/workers/rollout/vllm_rollout/vllm_async_server.py: 无本地 diff
verl/trainer/constants_ppo.py: 无本地 diff
run_verl_grpo_smoke10.sh: 不再默认 export VLLM_USE_V1
```

已跑通：

```text
verl_opvec_py310 Python/import 检查
main_ppo --cfg job Hydra 覆盖解析
OP-VEC 数据转换到 parquet
HF actor + OP-VEC external_lib 加载到 FSDP 前置阶段
Ray actor 内 vLLM 0.12 server 初始化
AgentLoop rollout
OP-VEC RewardRouter 计分
old logprob 计算
gate update
actor-to-vLLM W_eff 权重同步
```

裸 vLLM 验证也通过，包括带 VeRL worker extension 的独立启动：

```text
python -m vllm.entrypoints.openai.api_server
  --model /mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct
  --load-format dummy
  --worker-extension-cls verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension
  --max-model-len 2304
```

该独立 vLLM 在 GPU7 上可以完成 EngineCore 初始化、dummy model load 和 HTTP server ready。

## 速度结论

2 条、`n=2`、`max_response_length=64` 的 tiny smoke 已完整跑通：

```text
run_dir: /tmp/shared-storage/OnPolicy/runs/verl_opvec/verl_opvec_tiny_vllm012_cache3_0512
rollout: /tmp/shared-storage/OnPolicy/runs/verl_opvec/verl_opvec_tiny_vllm012_cache3_0512/rollouts/1.jsonl
metrics: /tmp/shared-storage/OnPolicy/runs/verl_opvec/verl_opvec_tiny_vllm012_cache3_0512/metrics.jsonl
step time: 9.43s，不含 actor/FSDP/vLLM 冷启动
generation time: 6.90s
old_log_prob: 0.50s
update_actor: 1.01s
update_weights: 1.02s
throughput: 256.45 tokens/s
```

## 主要风险

1. Full OP-VEC deltas 会作为 frozen buffers 进入 actor，显存主要由 base + deltas + vLLM rollout 占用。
2. `STRATEGY=parameter/global-parameter` 的可学习参数不多，但 `W_eff` 导出要遍历所有 gated weights，同步会比普通 LoRA 慢。
3. 当前是 single-turn VeRL rollout；真正工具交互式 multi-turn 需要再接 VeRL agent-loop 工具接口。
4. 默认不保存整模 checkpoint；需要保存 gate 时应单独导出 `opvec_gate_manager`。
