# 2026-05-24 HiddenSteer Tool HF Projection Smoke

本文档记录 HiddenSteer 第一轮 Tool-only HF hook 验证。结论先行：

> 当前低秩 task-vector module-output projection 框架工程上可跑，推理开销在小强度下可接受，但没有改变 Tool 输出；暂时不能进入 Memory/Code，也不值得改 vLLM。

## 1. 新增脚本

- `scripts/hiddensteer/build_tool_lowrank_basis.py`
  - 从已有 Tool activation-update projection diagnostics 中选模块。
  - 对 Tool/Memory/Code task-vector delta 做低秩 SVD。
  - 输出推理 hook 可加载的 `basis_manifest.json` 和 `lowrank_factors.pt`。

- `scripts/hiddensteer/run_hf_bfcl_tool_hiddensteer.py`
  - 使用 HF `generate` 跑 BFCL Tool prompt。
  - 可选加载 HiddenSteer basis 并注册 forward hook。
  - 输出 BFCL result JSON、partial BFCL score、wall-clock、tokens/s、hook stats。

编译验证：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/hiddensteer/build_tool_lowrank_basis.py \
  scripts/hiddensteer/run_hf_bfcl_tool_hiddensteer.py
```

## 2. Basis 产物

### rank4 / 2 modules

- Path: `/tmp/shared-storage/ExpertGym/hiddensteer/tool_lowrank_basis_rank4_m2_20260524`
- Selected modules:
  - `model.layers.24.self_attn.k_proj.weight`
  - `model.layers.26.self_attn.q_proj.weight`
- Factor size: `272K`

### rank8 / 8 modules

- Path: `/tmp/shared-storage/ExpertGym/hiddensteer/tool_lowrank_basis_rank8_m8_20260524`
- Factor size: `2.1M`
- Selected modules:
  - `model.layers.24.self_attn.k_proj.weight`
  - `model.layers.26.self_attn.q_proj.weight`
  - `model.layers.25.self_attn.v_proj.weight`
  - `model.layers.26.self_attn.o_proj.weight`
  - `model.layers.24.self_attn.q_proj.weight`
  - `model.layers.4.self_attn.k_proj.weight`
  - `model.layers.24.self_attn.o_proj.weight`
  - `model.layers.3.self_attn.v_proj.weight`

这些模块来自 `tool_memory_signature_s2_20260523` 中 Tool task 上 Memory/Code 非 owner residual 对 Tool owner update 的负投影证据。

## 3. Smoke 结果

模型：

- `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2`

### 4-sample smoke: `parallel` 2 + `live_parallel_multiple` 2

| setting | parallel | live_parallel_multiple | parallel wall | live-multiple wall | hook notes |
| --- | ---: | ---: | ---: | ---: | --- |
| HF baseline | 2/2 | 1/2 | 1.751s | 1.316s | no hook |
| rank4 m2, strength 1 | 2/2 | 1/2 | 1.835s | 1.471s | conflict token frac `0.0067` |
| rank8 m8, strength 1 | 2/2 | 1/2 | 1.864s | 1.637s | conflict token frac `0.1339` |
| rank8 m8, strength 10 | 2/2 | 1/2 | 1.958s | 1.541s | outputs still identical |

Observation:

- rank4/m2 is too conservative; almost no token receives correction.
- rank8/m8 has reasonable trigger coverage and still acceptable latency.
- However outputs were byte-identical to baseline on all 4 samples.

### Full `live_parallel_multiple` 24

| setting | score | wall time | tokens/s | errors |
| --- | ---: | ---: | ---: | --- |
| HF baseline | 16/24 = 0.667 | 14.741s | 72.73 | `cannot_find_match`: 7, `wrong_count`: 1 |
| rank8 m8, strength 10 | 16/24 = 0.667 | 18.911s | 56.60 | `cannot_find_match`: 7, `wrong_count`: 1 |

Wall-clock ratio: `1.283x`.

Outputs were byte-identical across all 24 samples.

### Extreme strength diagnostic

Setting:

- rank8 m8
- `projection_strength=100`
- `max_alpha=5`
- `live_parallel_multiple` first 2 samples

Result:

- Score unchanged: 1/2.
- Output unchanged.
- wall time: 2.524s for 2 samples.
- `correction_norm_mean`: `0.116996`
- `output_norm_mean`: `54.0335`
- `correction_to_output_norm`: `0.00217`

Interpretation:

- Even extreme strength only changes selected module outputs by about `0.22%` of the module output norm.
- This explains why greedy decoding is unchanged.
- Continuing to increase strength would stop being a principled projection method before it becomes useful.

## 4. Current Judgment

Tool validation is not complete in the positive sense. What is validated:

- HF hook infrastructure works.
- BFCL partial scoring works.
- Low-rank basis is tiny and fast to load.
- rank8/m8 overhead on full live-multiple is about `1.28x`, near the acceptable threshold.

What is not validated:

- No Tool sub-ability was improved.
- No output changed, even when correction strength was made very large.
- Current module-output correction is too small relative to normal module activations.

Therefore:

- Do not expand this version to Memory/Code.
- Do not patch vLLM for this version.
- Do not claim HiddenSteer works yet.

## 5. Next Technical Direction

The failure suggests that task-vector module-output deltas are too small compared with the full merged model activations. The next version should move the intervention target:

1. Prefer residual-stream activation basis over individual linear module output basis.
2. Normalize intervention by activation scale, not raw task-vector delta scale.
3. Keep the same HF feasibility harness and require output-level change before running larger BFCL.
4. Only if Tool shows real score movement under `<=1.3x` overhead should Memory/Code be attempted.

Concrete next test:

```text
Build residual-stream Tool anchor/interference basis from successful Tool trajectories.
Hook layer residual output after selected transformer blocks.
Use correction scale as a small fraction of residual-stream norm.
Run live_parallel_multiple 24 and parallel subset.
```

