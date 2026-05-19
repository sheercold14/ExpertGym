# Task Vector Norm Diagnostics

生成时间：`2026-05-17T06:24:31.687756+00:00`

## 输入

- mode manifest: `/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json`
- diagnostics: `/tmp/shared-storage/OnPolicy/modes/opvec4/diagnostics.json`
- mergeable params: `196`

## 总范数

| expert | total L2 | 相对 code |
|---|---:|---:|
| tool | 1.254945 | 2.0232 |
| memory | 5.362092 | 8.6449 |
| code | 0.620264 | 1.0000 |

解读：当前 code 不是缺少 module 覆盖，而是 delta 幅度显著小。code total L2 只有 tool 的 `0.4943`，只有 memory 的 `0.1157`。

## 覆盖与能量分布

| expert | params | MLP params | attention params | MLP energy frac | attention energy frac |
|---|---:|---:|---:|---:|---:|
| tool | 196 | 84 | 112 | 0.8694 | 0.1306 |
| memory | 196 | 84 | 112 | 0.8713 | 0.1287 |
| code | 196 | 84 | 112 | 0.8546 | 0.1454 |

## Code 有效稀疏性

| code energy coverage | modules needed | total modules |
|---:|---:|---:|
| 0.50 | 39 | 196 |
| 0.80 | 74 | 196 |
| 0.90 | 99 | 196 |
| 0.95 | 121 | 196 |

Code 相对其他 expert 的 module 级范数：

| condition | count | total |
|---|---:|---:|
| code_lt_0.25x_tool | 0 | 196 |
| code_lt_0.5x_tool | 79 | 196 |
| code_lt_0.75x_tool | 192 | 196 |
| code_lt_1x_tool | 196 | 196 |
| code_lt_0.25x_memory | 196 | 196 |
| code_lt_0.5x_memory | 196 | 196 |
| code_lt_0.75x_memory | 196 | 196 |
| code_lt_1x_memory | 196 | 196 |

## Code Top Modules

| rank | param | layer | module | code L2 | code/tool | code/memory |
|---:|---|---:|---|---:|---:|---:|
| 1 | `model.layers.12.mlp.up_proj.weight` | 12 | up_proj | 0.076931 | 0.6297 | 0.1376 |
| 2 | `model.layers.9.mlp.down_proj.weight` | 9 | down_proj | 0.076865 | 0.6326 | 0.1388 |
| 3 | `model.layers.12.mlp.down_proj.weight` | 12 | down_proj | 0.076827 | 0.6246 | 0.1375 |
| 4 | `model.layers.9.mlp.up_proj.weight` | 9 | up_proj | 0.075331 | 0.6222 | 0.1360 |
| 5 | `model.layers.11.mlp.down_proj.weight` | 11 | down_proj | 0.075147 | 0.6165 | 0.1338 |
| 6 | `model.layers.13.mlp.down_proj.weight` | 13 | down_proj | 0.074640 | 0.6101 | 0.1338 |
| 7 | `model.layers.11.mlp.up_proj.weight` | 11 | up_proj | 0.074542 | 0.6176 | 0.1323 |
| 8 | `model.layers.14.mlp.down_proj.weight` | 14 | down_proj | 0.074189 | 0.6029 | 0.1337 |
| 9 | `model.layers.10.mlp.down_proj.weight` | 10 | down_proj | 0.074085 | 0.5968 | 0.1320 |
| 10 | `model.layers.13.mlp.up_proj.weight` | 13 | up_proj | 0.074080 | 0.6055 | 0.1323 |
| 11 | `model.layers.14.mlp.up_proj.weight` | 14 | up_proj | 0.073470 | 0.5983 | 0.1304 |
| 12 | `model.layers.10.mlp.up_proj.weight` | 10 | up_proj | 0.073282 | 0.5967 | 0.1303 |
| 13 | `model.layers.17.mlp.up_proj.weight` | 17 | up_proj | 0.073022 | 0.5848 | 0.1328 |
| 14 | `model.layers.15.mlp.up_proj.weight` | 15 | up_proj | 0.072387 | 0.5825 | 0.1297 |
| 15 | `model.layers.15.mlp.down_proj.weight` | 15 | down_proj | 0.072126 | 0.5787 | 0.1312 |
| 16 | `model.layers.16.mlp.up_proj.weight` | 16 | up_proj | 0.071828 | 0.5782 | 0.1290 |
| 17 | `model.layers.16.mlp.down_proj.weight` | 16 | down_proj | 0.071495 | 0.5751 | 0.1300 |
| 18 | `model.layers.8.mlp.down_proj.weight` | 8 | down_proj | 0.071420 | 0.5828 | 0.1274 |
| 19 | `model.layers.12.mlp.gate_proj.weight` | 12 | gate_proj | 0.070949 | 0.5611 | 0.1273 |
| 20 | `model.layers.17.mlp.down_proj.weight` | 17 | down_proj | 0.070831 | 0.5628 | 0.1289 |

## Norm-aware Init 候选

### all_ones

all expert coefficients are initialized to 1.0; preserves full expert task-vector strength before learning

| expert | coefficient | effective L2 |
|---|---:|---:|
| tool | 1.000000 | 1.254945 |
| memory | 1.000000 | 5.362092 |
| code | 1.000000 | 0.620264 |

### all1_sqrt_weak_compensation

strongest expert stays at 1.0; weaker experts are compensated by sqrt(max ||Delta|| / ||Delta_e||). This is a conservative non-sweep stress test for weak code/tool deltas.

| expert | coefficient | effective L2 |
|---|---:|---:|
| tool | 2.067068 | 2.594057 |
| memory | 1.000000 | 5.362092 |
| code | 2.940214 | 1.823708 |

### all1_linear_weak_compensation

strongest expert stays at 1.0; weaker experts are linearly compensated to equal effective L2. This is aggressive and should be used only as a short stability diagnostic.

| expert | coefficient | effective L2 |
|---|---:|---:|
| tool | 4.272770 | 5.362092 |
| memory | 1.000000 | 5.362092 |
| code | 8.644856 | 5.362092 |

### sum1_equal_effective_l2

coefficients are proportional to 1 / ||Delta_e|| and sum to 1

| expert | coefficient | effective L2 |
|---|---:|---:|
| tool | 0.307004 | 0.385273 |
| memory | 0.071851 | 0.385273 |
| code | 0.621144 | 0.385273 |

### baseline_mean_effective_l2

coefficients equalize effective L2 to the mean perturbation produced by equal coefficients summing to 1

| expert | coefficient | effective L2 |
|---|---:|---:|
| tool | 0.640781 | 0.804144 |
| memory | 0.149968 | 0.804144 |
| code | 1.296456 | 0.804144 |

## 结论

- `1/3` 系数并不等价于三类能力同等注入；memory delta 的真实模型位移远大于 code。
- 如果要让 code 能力在 merged model 中可见，必须采用 norm-aware init、code-specific modes 或更强 code/reasoning delta。
- sweep 只应作为 oracle/diagnostic；论文方法应使用确定性 norm-aware init，然后用 on-policy gate learning 微调。
