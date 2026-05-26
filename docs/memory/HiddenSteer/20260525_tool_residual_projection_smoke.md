# 2026-05-25 HiddenSteer Tool Residual Projection Smoke

本文档记录第二轮 Tool-only HiddenSteer 验证：把干预目标从 task-vector module output 移到 transformer block residual stream。结论先行：

> residual-stream projection 的 HF 原型可跑，rank-4 basis 很小，满集开销约 `1.13x`；但 Tool `live_parallel_multiple` 没有提升。保守 failure-only projection 保持 16/24，不修复原失败；full failure projection 降到 13/24。因此当前不进入 Memory/Code，也不改 vLLM。

## 1. 新增/修改脚本

- `scripts/hiddensteer/build_tool_residual_basis.py`
  - 从 BFCL Tool baseline result/score 中抽取成功和失败轨迹。
  - teacher-force prompt + model response，hook selected transformer blocks 的 residual output。
  - 保存每层 success basis、failure basis、failure-orthogonal-to-success basis。

- `scripts/hiddensteer/run_hf_bfcl_tool_hiddensteer.py`
  - 新增 `--residual-basis-manifest`。
  - 新增 residual intervention mode：
    - `anchor_boost`
    - `remove_failure`
    - `remove_failure_orthogonal`
    - `boost_success_remove_failure`
  - 记录 BFCL score、wall-clock、tokens/s、hook stats。

编译验证已通过：

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python -m py_compile \
  scripts/hiddensteer/build_tool_residual_basis.py \
  scripts/hiddensteer/run_hf_bfcl_tool_hiddensteer.py
```

## 2. Basis 产物

模型：

- `/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2`

Baseline 数据：

- `/tmp/shared-storage/ExpertGym/hiddensteer/hf_baseline_lpm24_20260524`
- HF baseline `live_parallel_multiple`: `16/24 = 0.6667`

Residual contrast basis：

- Path: `/tmp/shared-storage/ExpertGym/hiddensteer/tool_residual_contrast_basis_rank4_l24-27_lpm24_20260525`
- Basis file size: `456K`
- Layers: `24,25,26,27`
- Rank: `4`
- Success rows: `16`
- Failure rows: `8`
- Tokens per layer: success `713`, failure `328`

## 3. Results

### Small 2-sample smoke

| mode | strength / cap | score | output change | note |
| --- | ---: | ---: | --- | --- |
| baseline | none | 1/2 | n/a | first two LPM samples |
| `anchor_boost` | 0.05 / 0.05 | 1/2 | no | correction/hidden `0.030` |
| `anchor_boost` | 0.20 / 0.20 | 1/2 | no | correction/hidden `0.132` |
| `anchor_boost` | 1.00 / 1.00 | 0/2 | yes, broken | repeated punctuation |
| `remove_failure_orthogonal` | 0.20 / 0.20 | 1/2 | no | correction/hidden `0.023` |
| `remove_failure` | 0.20 / 0.20 | 1/2 | no | correction/hidden `0.095` |
| `remove_failure` | 1.00 / 1.00 | 0/2 | yes, broken | over-correction |

Logits diagnostic for one failed sample showed the hook is numerically active:

- next-token max logit diff: `6.1875`
- mean absolute logit diff: `0.7888`
- top-10 token ranking changed
- argmax did not change

Interpretation: the hook reaches logits, but low/moderate projection does not cross the greedy decoding boundary; strong projection destroys format.

### Full `live_parallel_multiple` 24

| setting | score | wall time | output tokens/s | changed outputs | gain/loss |
| --- | ---: | ---: | ---: | ---: | --- |
| HF baseline | 16/24 = 0.6667 | 14.741s | 72.73 | n/a | n/a |
| `remove_failure`, 0.20 / 0.20 | 13/24 = 0.5417 | 16.578s | 61.26 | 13/24 | gained 0, lost 3 |
| `remove_failure_orthogonal`, 0.20 / 0.20 | 16/24 = 0.6667 | 16.651s | 64.07 | 2/24 | gained 0, lost 0 |

Overhead:

- conservative `remove_failure_orthogonal`: `16.651 / 14.741 = 1.13x`
- full `remove_failure`: `16.578 / 14.741 = 1.12x`

The overhead is acceptable for an HF prototype, but there is no accuracy gain to justify vLLM engineering.

## 4. Error Movement

`remove_failure` 0.20 changed 13 outputs and hurt 3 originally correct samples:

- `live_parallel_multiple_11-10-0`: dropped optional `include_subdirectories=True`.
- `live_parallel_multiple_18-16-0`: merged two event calls into one `Music|Theater` call.
- `live_parallel_multiple_8-7-0`: changed `directory_name="nodejs-welcome"` to `"."` in later calls.

It fixed no originally failed sample.

`remove_failure_orthogonal` changed only two originally failed outputs:

- Korean appliance command changed, still wrong.
- quote style changed in a math function string, still wrong.

Thus the projection is behaviorally active but not aligned with correct Tool sub-decisions.

## 5. Current Decision

Do not expand this version to Memory/Code.

Do not patch vLLM yet.

Reason:

- Effectiveness gate failed: no Tool score improvement, no error-type reduction.
- The only setting that preserves score adds about `1.13x` HF overhead but gives zero gain.
- Stronger settings expose a narrow or absent usable operating region: moderate strength is inert or neutral; high strength breaks Tool formatting.

The useful research signal is negative but clear:

> Tool errors in this subset are not simply a removable low-rank failure residual in late-layer residual stream. Many failures are exact argument canonicalization, optional parameter preservation, call count separation, or schema-value matching errors. These are trajectory-level decoding decisions, not a stable post-hoc activation subspace that can be projected out after merging.

Next method attempt should not be another scalar alpha sweep on this projection. If continuing HiddenSteer, the next hypothesis must use span-specific or decision-specific basis, e.g. function-name span vs argument-value span vs multi-call delimiter/count span, with trigger limited to those spans.
