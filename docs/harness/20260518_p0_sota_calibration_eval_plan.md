# 2026-05-18 P0 SOTA-Oriented Calibration And Evaluation Plan

## Priority Shift

当前主线不再是“补齐所有论文表格”，而是先把核心方法推到强评测结果：

> task vectors provide structured priors; executable feedback learns their composition.

P0 目标是让 ExpertGym 在 Tool / Memory / Code 的正式能力上接近或超过现有强参考，而不是只证明训练 proxy 能上涨。

## Target Metrics

| task | primary metric | secondary metric | role in P0 |
|---|---|---|---|
| Tool | BFCL non-live + ToolRL all80 overall correct | BFCL live mean | Tool live 波动大，不作为唯一调参目标 |
| Memory | HotpotQA Eval6 mean F1 | EM / long-context subsets | 必须把 memory 能力真正推出来 |
| Code | CURE mean Acc + BoN(4,4) | any-pass / generation vs selection failure | 当前最大短板，calibration 必须直接对齐 CURE 能力轴 |

Tool 的新内部评测口径：

```text
ToolRL rlla_4k/test all80 overall correct = correct prompts / 80
```

不按子类平均，不只看 live；它用于检查 ToolRL 源分布是否被破坏。BFCL live 仍报告，但不因为 live_parallel 的小样本波动而过度改 gate。

## Local Artifacts

| artifact | path |
|---|---|
| ToolRL local repo | `/tmp/shared-storage/OnPolicy/external_repos/ToolRL` |
| ToolRL train parquet | `/tmp/shared-storage/OnPolicy/external_repos/ToolRL/dataset/rlla_4k/train.parquet` |
| ToolRL test parquet | `/tmp/shared-storage/OnPolicy/external_repos/ToolRL/dataset/rlla_4k/test.parquet` |
| ToolRL all80 eval manifest | `/tmp/shared-storage/OnPolicy/data/evaluation/toolrl_rlla4k_test_20260518/toolrl_rlla4k_test_all80.prompts.jsonl` |
| ToolRL tool-call-only manifest | `/tmp/shared-storage/OnPolicy/data/evaluation/toolrl_rlla4k_test_20260518/toolrl_rlla4k_test.prompts.jsonl` |
| ToolRL eval launcher | `skill/command/run_20260518_toolrl_rlla4k_eval.sh` |
| Current eval-targeted96 | `/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl` |
| Case browser data | `/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/` |

ToolRL all80 manifest audit:

- source rows: `80`
- tool-call rows: `71`
- response-only rows: `9`
- source reward: current `ToolRewardAdapter`, aligned to ToolRL `format_reward + tool_call_correctness_reward`

## How To Run ToolRL Test

For any baked HF checkpoint:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

MODEL_PATH=/path/to/baked_policy \
RUN_ID=my-model-toolrl-all80 \
GPU_LIST=0 \
bash skill/command/run_20260518_toolrl_rlla4k_eval.sh
```

Output:

```text
/tmp/shared-storage/OnPolicy/eval/toolrl_rlla4k_20260518/<RUN_ID>/summary.json
```

Read:

- `task_stats.tool.success_rate`: all80 overall correct ratio.
- `task_stats.tool.tool_exact_rate`: exact tool-call rate on rows where exact tool-call details exist.
- `task_stats.tool.mean_reward`: ToolRL normalized training reward, secondary only.

## P0 Calibration Redesign

### Tool

Tool is not the bottleneck to solve by chasing BFCL live alone.

Use three-way Tool evidence:

| source | purpose | training use |
|---|---|---|
| ToolRL train/test style | preserve source expert behavior | retention + frontier |
| BFCL-style synthetic | cover schema/default/canonical/parallel patterns | GRPO + OPD if expert succeeds |
| BFCL official Eval6 | final report only | no training leakage |

Training target:

- Keep ToolRL all80 no worse than strong static baselines.
- Keep BFCL mean near `0.78-0.80`.
- Treat BFCL live as diagnostic; do not overfit to 24 live_parallel_multiple cases.

### Memory

Memory must be trained as trajectory behavior, not only final-answer text.

Calibration bank should contain:

| state | construction |
|---|---|
| recoverable | current merge all-fail; RL-MemoryAgent same prompt final answer verified positive; OPD over update turns + final turn |
| frontier | current K rollouts have reward variance under HotpotQA final verifier |
| stable | current all-success; NLL preservation on accepted trajectory |
| monitor | disjoint HotpotQA question id and article id |

Immediate data rule:

- Use HotpotQA train/dev disjoint from Eval6.
- Store full trajectory if available.
- If full official trajectory is missing, do not pretend final answer OPD is equivalent; mark it as weak memory OPD.

### Code

Code must be split into generation and selection. Current training often moves proxy reward but not CURE.

| pool | symptom | target signal |
|---|---|---|
| generation | no correct code sample | verified passing code OPD; increase sample acc / any-pass |
| selection | any-pass exists but BoN chooses wrong | generated-test / counterexample-test OPD; increase BoN |
| partial edge | passes examples but fails hidden-like tests | edge-case tests + verified solution |

Data rule:

- Do not train on official LiveBench / LiveCodeBench prompts or hidden tests.
- Use CURE case study only as blueprint tags.
- Generate or select fresh CodeContests-train / synthetic tasks with source tests.
- Code reward must use executable tests, not text similarity.

## Next Experimental Queue

### P0-A: Calibration Bank V2

Build a new bank, not overwriting `paper96` or `eval_targeted96_cure_aligned_20260517`:

```text
/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/
  train128.prompts.jsonl
  monitor64.prompts.jsonl
  guard64.prompts.jsonl
  expert_rollouts/
  state_audit/
  summary.json
```

Recommended train128:

| task | rows | composition |
|---|---:|---|
| Tool | 32 | 16 ToolRL/RLLA source + 16 BFCL-style synthetic |
| Memory | 48 | 32 HotpotQA trajectory/recoverable + 16 stable/retention |
| Code | 48 | 24 generation + 16 selection + 8 partial-edge |

Reason: Memory and Code are currently under-trained relative to the desired SOTA direction; Tool needs preservation more than extra live overfitting.

### P0-B: Strong-Prior Main Runs

Run no more than two main variants first:

| run | init | gate space | loss | goal |
|---|---|---|---|---|
| `sota-v2-gc` | scale-calibrated / TA-0.75-like | global 3 | GRPO + OPD + retention | strongest overall |
| `sota-v2-gp` | same | common+residual 4 | GRPO + OPD + retention | more flexible non-regression |

Do not start module-wise/588 until global 3/4 shows real monitor gain.

### P0-C: Monitor And Stop Rule

Every 2-3 iterations:

- rollout `monitor64`;
- compute ToolRL all80 for candidate checkpoints when Tool looks unstable;
- track Memory F1 proxy and Code generation/selection proxy separately;
- stop if train reward rises but monitor does not.

Promotion to Eval6:

| metric | minimum |
|---|---:|
| Tool BFCL mean | no worse than TA-1/3 by `0.01` |
| ToolRL all80 | no worse than TA/static reference |
| Memory monitor | clear positive trend |
| Code monitor | generation or selection improves; not just reward_train |
| worst-task collapse | none |

## Immediate Implementation Gaps

| gap | status | action |
|---|---|---|
| ToolRL all80 eval | done | manifest + launcher added |
| Calibration bank V2 | done | `/tmp/shared-storage/OnPolicy/data/calibration/sota_calib_v2_20260518/{train128,monitor64,guard64}.prompts.jsonl` |
| Memory trajectory bank V2 | done-first-pass | HotpotQA train trajectory-style rows; still need expert coverage audit before main training |
| Code generation/selection bank V2 | done-first-pass | CodeContests-train CURE-style probes + paper96 code anchors; still need monitor reward audit |
| monitor64/guard64 split | done | disjoint partition by `prompt_hash`, stratified by role |
| per-task monitor dashboard | partial | current monitor reads train run; add monitor rollout summaries later |

Build / run config:

- `docs/config/20260518_p0_sota_calib_v2.md`
- `skill/command/build_20260518_sota_calib_v2.sh`
- `skill/command/run_20260518_sota_v2_expert_rollouts.sh`
- `skill/command/run_20260518_p0_sota_v2.sh`

## Key Scientific Decision

For SOTA-oriented runs, `1/3` is no longer the only main initialization. The paper can still say task vectors are structured priors, but the strongest method should start from a scale-calibrated prior if that is what the evidence supports. The executable-feedback claim is then:

> starting from a strong but static task-vector prior, ExpertGym uses verified rollouts to choose non-degenerate coefficients and recover Memory/Code without sacrificing Tool.
