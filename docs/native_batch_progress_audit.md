# Native Batch Gated-GRPO 进度审计

日期：2026-05-12  
仓库：`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym`

## 目标拆解

最终目标不是只让测试通过，而是形成一个可审计、低显存、native-first 的 OP-VEC gated GRPO 管线：

```text
固定 gate_t
  -> batch rollout
  -> batch reward
  -> old/new token logprob + mask
  -> mini-batch gate update
  -> gate_{t+1}
```

## Checklist

| # | 要求 | 当前状态 | 证据 | 缺口 |
|---|---|---|---|---|
| 1 | 每轮固定 `gate_t`，rollout/reward/old_logprob 绑定 behavior policy | 部分完成 | `docs/native_batch_training_contract.md`；rollout row 写 `gate_id/gate_values/gate_checkpoint` | 需要真实 run manifest 审计确认无混 gate |
| 2 | 统一 rollout schema：`prompt_id/task/group_id/sample_id/gate_id/seed/token/logprob/mask/reward details` | 基本完成 | `opvec/data/schema.py`；HF/vLLM collector 写 `group_id/gate_id/seed`；HF/vLLM 均可写 token logprob | 需要真实 vLLM rollout 文件抽样审计 |
| 3 | batch rollout：Tool/Code vLLM batch，Memory batched active-state loop | 部分完成 | `scripts/train/opvec_collect_vllm_rollouts.py` 已 batch generate；Memory 按 chunk active-state 生成 | 还不是常驻 server；未做 gate-only sync |
| 4 | batch reward：`RewardRouter.batch_score()` 统一入口 | 完成第一版 | `opvec/rewards/router.py`；vLLM collector 已调用 batch_score | 内部仍串行 adapter，后续可并行化 expensive reward |
| 5 | batch update：模型/gate manager 常驻，按 mini-batch 聚合后 step | 完成第一版 | `_UpdateBatcher`；`--update-batch-size`；单测覆盖延迟 step | 还不是全 token batch forward；logprob 仍逐 sample/turn forward |
| 6 | VeRL loss 语义：token-level PPO/GRPO + mask + scalar advantage broadcast | 部分完成 | `opvec/train/gated_grpo.py` token loss；updater `--loss-granularity token`；vLLM 可直存 old token logprobs | 默认仍 sequence；需要 vLLM token smoke 后再考虑切默认 |
| 7 | Memory credit assignment：final reward 分配到 update turns + final turn token | 部分完成 | HF token rollout 聚合 trajectory；token mode 可对全 trajectory token 反传 | 需要监控 Memory token 数是否压制 Tool/Code |
| 8 | vLLM gate-only sync | 未开始 | 当前 bake+vLLM loop 每轮 bake checkpoint | 等 batch native 语义稳定后再做 OP-VEC-aware vLLM server |
| 9 | 监控与审计：reward histogram/frontier/advantage/clip/KL/grad/gate | 部分完成 | update rows 有 `clip_frac/approx_kl/grad_norm/gates`；epoch summary 有均值 | reward histogram、advantage std 的统一 dashboard 仍需补 |
| 10 | 验证：synthetic 单测 + 10 条数据跑 2-3 轮 | 部分完成 | 58 个轻量测试；`run_smoke_sequence_vs_token.sh` dry-run 通过；vLLM token dry-run 通过 | 尚未实际跑 vLLM 10-prompt token smoke |

## 当前可运行验证入口

```bash
DRY_RUN=1 skill/command/run_smoke_sequence_vs_token.sh
```

真实 10-prompt 对照：

```bash
GPU_LIST=0,1,2,3 \
NUM_PROMPTS=10 \
SAMPLES_PER_PROMPT=4 \
UPDATE_BATCH_SIZE=4 \
skill/command/run_smoke_sequence_vs_token.sh
```

## 结论

当前已经完成 native batch update、batch reward API、token-level schema/loss/update 的第一版闭环；但目标还没有完成。下一步必须实际跑同一批 rollout 的 sequence/token 对照，检查：

- `updates` 和 `optimizer_steps` 是否非零。
- `clip_frac` / `approx_kl` 是否合理。
- `final_gates` 是否产生可解释变化。
- Memory token 轨迹是否导致梯度尺度明显大于 Tool/Code。

## 2026-05-12 Partial Smoke 更新

已运行 partial sequence/token 对照：

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/sequence_vs_token_smoke10_native_20260512_153139
```

结果记录在：

```text
docs/native_sequence_token_smoke_report.md
```

结论：token-level update 能跑通，并且在 1 个 Code frontier row 上与 sequence loss 的最终 gate 变化几乎一致。但由于 HF rollout 太慢，中途停止在 2 条 rollout；该结果尚未覆盖 Memory，也不是完整 10-prompt / 2-3 轮验证。

## 2026-05-12 vLLM Token Logprob 更新

已完成非常驻 vLLM 路径的低风险加速项：

- `opvec_collect_vllm_rollouts.py --store-token-logprobs` 直接保存 vLLM sampled token ids 和 old token logprobs。
- Memory trajectory 每个 update/final turn 保存 token payload，sample 级聚合所有 turn 的 token payload。
- update token mode 优先用 rollout 中的 `response_token_ids` 重算 current logprob，避免文本重分词漂移。
- bake+vLLM loop 透传 `--store-token-logprobs`。
- official/qbank vLLM launcher 在 `LOSS_GRANULARITY=token` 时默认开启 token logprob 存储，并让 `MAX_LOGPROB_TOKENS` 默认跟随 `MAX_MODEL_LEN`。

验证：

```text
bash -n skill/command/run_official_gated_grpo_global_vllm_one_iter.sh skill/command/run_qbank_c033333_gate_strategy.sh
python -m py_compile opvec/modeling/logprob.py scripts/train/opvec_collect_vllm_rollouts.py scripts/train/opvec_update_gates_from_rollouts.py scripts/train/opvec_gated_grpo_bake_vllm_loop.py
DRY_RUN=1 LOSS_GRANULARITY=token UPDATE_BATCH_SIZE=4 skill/command/run_official_gated_grpo_global_vllm_one_iter.sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests
```

结果：58 tests OK，dry-run collect 命令已包含 `--store-token-logprobs`，update 命令使用 `--loss-granularity token --max-logprob-tokens 12288`。
