# OP-VEC Dynamic Loss Scale Monitoring Report

Date: 2026-05-15

## Objective

监视并迭代四个 OP-VEC gate 训练实验。若出现清晰失败迹象，停止对应实验，定位原因后重启；所有操作和结果必须可追溯记录。

## Code State

Current objective logic:

- GRPO/frontier: 保持官方 reward + row-centered group advantage，`PPO_LOSS_WEIGHT=6.0` 作为基准项。
- OPD: 对每轮 policy all-fail 且 expert 可恢复的 prompt 构造 OPD row。
- Retention: 对 all-success row 使用 NLL preservation。
- LR: 控制 gate 总步幅。
- OPD/retention relative scale: 通过 no-grad 预估 loss 幅值后动态缩放，使辅助 loss 落到显式 GRPO-relative target。

Implemented controls:

- `--opd-dynamic-scale`
- `--opd-task-balanced-loss-scale`
- `--retention-dynamic-scale`
- `--retention-task-balanced-loss-scale`
- `--retention-scale-target`

## Smoke Run

Run tag: `20260515_dynopdscale_smoke_i1`

Purpose: 用 8 prompts、1 iteration 验证动态 OPD scale 是否能把 gate 推出 `1/3` 附近，并确认 no-OPD 对照不应显著移动。

| Run | Strategy | OPD | Retention | Grad max | Gate delta max | Final mean gates |
|---|---:|---:|---:|---:|---:|---|
| `dynscale_A_gc_20260515_dynopdscale_smoke_i1` | global-coefficient | dynamic, task-balanced | fixed NLL | 1.6195 | 0.0431 | code 0.3712 / memory 0.3509 / tool 0.3764 |
| `dynscale_B_gp_20260515_dynopdscale_smoke_i1` | global-parameter | dynamic, task-balanced | fixed NLL | 1.8808 | 0.0348 | code 0.3323 / memory 0.3648 / tool 0.3369 |
| `dynscale_C_gp_fixedopd_20260515_dynopdscale_smoke_i1` | global-parameter | fixed OPD reference | fixed NLL | 0.6919 | 0.0215 | code 0.3534 / memory 0.3154 / tool 0.3391 |
| `dynscale_D_gc_noopd_20260515_dynopdscale_smoke_i1` | global-coefficient | disabled | fixed NLL | 0.0087 | 0.0005 | code 0.3332 / memory 0.3332 / tool 0.3338 |

OPD scale evidence:

- A/code: raw mean OPD loss `218.62`, scale `0.1372`, target loss `30.0`.
- A/tool: raw mean OPD loss `17.22`, scale `1.7418`, target loss `30.0`.
- B/tool: raw mean OPD loss `18.79`, scale `1.5962`, target loss `30.0`.

Conclusion:

- Dynamic OPD scale is functioning: it normalizes task-specific OPD loss to the intended target range and creates gate deltas in the previous successful experiment scale.
- No-OPD control barely moves, confirming the current immediate driver is OPD, not retention or weak GRPO alone.
- Global-parameter can move differently by module and task, but one 8-prompt smoke is too small to judge reward direction.

## Failure Checks

No hard failure observed in smoke:

- All four runs produced `gate_updates.gates.json`.
- GPU processes exited cleanly.
- No OOM or traceback observed in `run.log`.
- Dry-run after retention dynamic scale patch confirmed updater receives all new CLI arguments.

Potential risks to monitor in full runs:

- Tool collapse if OPD rows for tool are sparse in later iterations.
- Memory/code dominance if long sequences produce larger raw gradients despite scaling.
- Over-push if `grad_norm_max` stays above previous successful range for several iterations.
- Weak GRPO if frontier rows remain too few or all-success/all-fail saturation dominates.

## Full Matrix Plan

Script: `skill/command/run_dynamic_opd_scale_20260515.sh`

Default full-run settings:

- `NUM_PROMPTS=96`
- `SAMPLES_PER_PROMPT=4`
- `NUM_ITERS=10` unless overridden
- `INIT_VALUE=0.3333333333333333`
- `OPTIMIZER=sgd`
- `SGD_MOMENTUM=0.5`
- `OPTIMIZER_STEP_SCOPE=epoch`
- `LOSS_GRANULARITY=sequence`
- `LENGTH_NORMALIZE_POLICY_LOGPROB=1`
- `OPD_LENGTH_NORMALIZE_LOGPROB=0`
- `RETENTION_LENGTH_NORMALIZE_LOGPROB=1`
- `OPD_DYNAMIC_SCALE=1`
- `OPD_TASK_BALANCED_LOSS_SCALE=1`
- `RETENTION_DYNAMIC_SCALE=1`
- `RETENTION_TASK_BALANCED_LOSS_SCALE=1`
- `RETENTION_SCALE_TARGET=0.5`

Four runs:

- A: `global-coefficient`, dynamic OPD, dynamic retention, GPUs `0,1`, LR `0.06`.
- B: `global-parameter`, dynamic OPD, dynamic retention, GPUs `2,3`, LR `0.035`.
- C: `global-parameter`, fixed OPD reference, dynamic retention, GPUs `4,5`, LR `0.04`.
- D: `global-coefficient`, no-OPD control, dynamic retention, GPUs `6,7`, LR `0.06`.

Stop / restart criteria:

- Stop a run if overall reward drops for 3 consecutive iterations and tool/memory/code one task collapses sharply.
- Stop a run if gate delta repeatedly hits `MAX_COEFF_DELTA=0.45`.
- Stop a run if `grad_norm_max` explodes far above the smoke range without reward improvement.
- If dynamic OPD rows become mostly single-task, restart with per-task OPD quota or lower per-task target for that task.

## Full Matrix Launch

Launch time: 2026-05-15 03:46 Asia/Shanghai

Command:

```bash
RUN_TAG=20260515_dynloss_i15_v1 \
NUM_ITERS=15 \
NUM_PROMPTS=96 \
SAMPLES_PER_PROMPT=4 \
MONITOR_PORT=8783 \
DRY_RUN=0 \
bash skill/command/run_dynamic_opd_scale_20260515.sh
```

Monitor:

- `http://127.0.0.1:8783`

Run directories:

- A: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_A_gc_20260515_dynloss_i15_v1`
- B: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_B_gp_20260515_dynloss_i15_v1`
- C: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_C_gp_fixedopd_20260515_dynloss_i15_v1`
- D: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_D_gc_noopd_20260515_dynloss_i15_v1`

Initial status:

- All four training tmux sessions created.
- Monitor tmux session created.
- Launch logs confirm `RETENTION_DYNAMIC_SCALE=1`, `RETENTION_TASK_BALANCED_LOSS_SCALE=1`, `RETENTION_SCALE_TARGET=0.5`.
- All four runs entered `iter_001` bake stage.

04:00 status:

- All four runs finished `iter_001` sharded vLLM rollout and entered HF update.
- Rollout time was about 390-410s per run with two rollout shards per run.
- Dynamic OPD all-fail recoverable rows:
  - A: 17 rows, `code=1`, `memory=9`, `tool=7`.
  - B: 22 rows, `code=4`, `memory=11`, `tool=7`.
  - C: 18 rows, `code=2`, `memory=6`, `tool=10`.
  - D: no OPD control.
- No `gate_updates.summary.json` had been written yet at 04:00; updater processes were alive and consuming GPU/CPU.
- Current bottleneck is HF update-side policy logprob plus OPD/retention NLL scale estimation, not vLLM rollout.

04:25 status:

- `iter_001` completed for A, B, and D. C was still in update, likely due fixed OPD pairwise extra forwards.
- A `global-coefficient + dynamic OPD`:
  - `grad_norm_max=25.82`, `gate_delta_max=0.0598`.
  - Gates: `code=0.3334`, `memory=0.3931`, `tool=0.3285`.
  - OPD rows: `code=1`, `memory=9`, `tool=7`.
  - Interpretation: strong movement exists, but first step is memory-dominant; tool slightly decreases.
- B `global-parameter + dynamic OPD`:
  - `grad_norm_max=24.54`, `gate_delta_max=0.0356`.
  - Mean gates: `code=0.3341`, `memory=0.3682`, `tool=0.3319`.
  - OPD rows: `code=4`, `memory=11`, `tool=7`.
  - Interpretation: same direction as A, with smaller per-parameter step.
- D `global-coefficient + no OPD`:
  - `grad_norm_max=0.47`, `gate_delta_max=0.0266`.
  - Gates: `code=0.3340`, `memory=0.3067`, `tool=0.3234`.
  - Interpretation: no-OPD control does not push task vectors upward; retention/GRPO alone tends to suppress memory/tool here.
- Dynamic scale behavior:
  - OPD task scales correctly compensate raw NLL magnitude: raw memory/code NLL about 120-143, raw tool NLL about 15-16, so tool receives larger component scale.
  - Retention target is held at `PPO_LOSS_WEIGHT * 0.5 = 3.0` per active task; retention rows are sparse in `iter_001`.
- Decision: keep A/B/D running into `iter_002`; keep C alive for now as a slow control, but treat it as secondary if it blocks resources.

C follow-up:

- C finished `iter_001` shortly after 04:25.
- `grad_norm_max=3.69`, `gate_delta_max=0.0405`.
- Mean gates: `code=0.3328`, `memory=0.3730`, `tool=0.3291`.
- C confirms the old fixed OPD+pairwise reference also pushes mainly memory, but is much slower than dynamic OPD and has smaller gradient norm.

Rollout reward after first update:

- A `iter_002` sample mean reward:
  - Tool: `0.4048 -> 0.9528`, all-success rows `0 -> 19`.
  - Memory: `0.2578 -> 0.4062`.
  - Code: `0.4424 -> 0.3818`.
- B `iter_002` sample mean reward:
  - Tool: `0.4422 -> 0.7464`, all-success rows `1 -> 11`.
  - Memory: `0.3125 -> 0.3047`.
  - Code: `0.3213 -> 0.3676`.
- Initial read: dynamic OPD did not destroy tool; tool reward improved sharply despite small tool coefficient decrease. Code/memory tradeoff remains the main risk.

04:54 status:

- `iter_002` completed for A, B, and D. C was still in `iter_002` update.
- A:
  - Gates after update: `code=0.3325`, `memory=0.4830`, `tool=0.3271`.
  - `grad_norm_max=14.50`, `gate_delta_max=0.0899`.
  - OPD rows dropped to 6: `memory=5`, `tool=1`.
  - Reward trend: tool sharply improved, memory improved, code decreased.
  - Risk: global-coefficient may over-allocate to memory because remaining all-fail OPD rows are memory-heavy.
- B:
  - Mean gates after update: `code=0.3346`, `memory=0.4206`, `tool=0.3318`.
  - `grad_norm_max=24.53`, `gate_delta_max=0.0536`.
  - OPD rows: `code=4`, `memory=7`, `tool=4`.
  - Reward trend: tool improved strongly, code improved, memory roughly flat.
  - Current best-looking run among A/B/D because it preserves code better than A.
- D:
  - Gates after update: `code=0.3305`, `memory=0.3033`, `tool=0.3204`.
  - `grad_norm_max=0.18`, `gate_delta_max=0.0035`.
  - Reward trend: code/memory/tool all decreased from `iter_001`.
  - Interpretation: no-OPD control is useful as evidence but is not a good optimization setting.

Operational note:

- Update remains the bottleneck. A/B/D `iter_002` updates took about 20-30 minutes after rollout.
- No immediate stop yet: A/B still have useful reward gains; D is kept temporarily as an ablation.

05:06 status:

- A/B/D finished `iter_003` rollout; A/B/D `iter_003` update was not complete yet.
- A `iter_003` rollout reward:
  - Code `0.3818 -> 0.4912`.
  - Memory `0.4062 -> 0.6328`.
  - Tool `0.9528 -> 0.9708`.
  - OPD rows for next update dropped to 4: `code=1`, `memory=3`, `tool=0`.
  - Interpretation: A recovered the `iter_002` code drop and is now strongly improving memory/tool.
- B `iter_003` rollout reward:
  - Code `0.3676 -> 0.4424`.
  - Memory `0.3047 -> 0.3828`.
  - Tool `0.7464 -> 0.9579`.
  - OPD rows for next update: 11, `code=2`, `memory=8`, `tool=1`.
  - Interpretation: B remains more balanced than A, with less aggressive memory coefficient movement.
- D `iter_003` rollout reward:
  - Code `0.3623 -> 0.4502`.
  - Memory `0.2969 -> 0.4375`.
  - Tool `0.3564 -> 0.4223`.
  - Interpretation: D is noisy and can recover through GRPO/retention, but it still lacks the clear task-vector-upward signal seen in A/B.
- Current decision:
  - Continue A/B.
  - Keep C as slow fixed-OPD reference.
  - Keep D until at least `iter_005` unless it blocks resources or clearly underperforms.

05:38 intervention:

- D was stopped early after `iter_004` rollout because tool reward collapsed:
  - Tool sample mean `0.4223 -> 0.0600`.
  - Tool all-fail rows `7 -> 27`.
  - No-OPD control had already shown weak/unstable coefficient learning, so continuing it was not useful for finding the best model.
- Replacement E launched on GPUs `6,7`:
  - Run: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_E_gp_lr025_m02_20260515_dynloss_i15_v1`
  - Strategy: `global-parameter`.
  - Dynamic OPD and dynamic retention enabled.
  - LR lowered to `0.025`.
  - SGD momentum set to `0.2`.
  - Purpose: compare against B with a slower, lower-momentum coefficient trajectory to reduce memory over-push and preserve code/tool.
- Monitor restarted on `http://127.0.0.1:8783` with A/B/C/D/E visible.

06:06 intervention:

- A was stopped after `iter_005` rollout because tool collapsed:
  - Tool sample mean `0.8166 -> 0.0268`.
  - Tool all-fail rows `1 -> 31`.
  - A had pushed memory coefficient to `0.6895` after `iter_004`; this is likely beyond the stable range for the global-coefficient parameterization.
- A useful checkpoints to preserve:
  - `iter_003`: strong tool/memory and non-collapsed code.
  - `iter_004`: memory high but tool already weakened; use cautiously.
- Replacement F launched on GPUs `0,1`:
  - Run: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_F_gc_lr025_m02_20260515_dynloss_i15_v1`
  - Strategy: `global-coefficient`.
  - Dynamic OPD and dynamic retention enabled.
  - LR lowered from A's `0.06` to `0.025`.
  - SGD momentum lowered from `0.5` to `0.2`.
  - Purpose: test whether the 3-coefficient global setting can work when memory is not over-pushed.
- Monitor restarted on `http://127.0.0.1:8783` with A/B/C/D/E/F visible.

06:07 B checkpoint:

- B `iter_005` rollout remained stable:
  - Code sample mean: `0.4004 -> 0.4018`.
  - Memory sample mean: `0.6250 -> 0.7734`.
  - Tool sample mean: `0.9639 -> 0.9564`.
- Compared with A, B can raise memory while keeping tool intact. This makes B the current primary candidate for final evaluation.

06:26 status:

- B `iter_005` update completed:
  - Mean gates: `code=0.3377`, `memory=0.6143`, `tool=0.3337`.
  - `grad_norm_max=6.01`, `gate_delta_max=0.0691`.
  - OPD rows: `code=3`, `memory=2`, `tool=1`.
  - Interpretation: B is still correcting multiple tasks, not only memory, and remains the main run to continue.
- C `iter_005` rollout:
  - Code `0.4213 -> 0.3623`.
  - Memory `0.7344 -> 0.7969`.
  - Tool `0.9721 -> 0.9185`.
  - Interpretation: fixed OPD reference is usable but starts trading off code/tool as memory rises.
- E low-LR global-parameter:
  - First update gates: `code=0.3332`, `memory=0.3582`, `tool=0.3319`.
  - `iter_002` rollout: tool improved to `0.6806`, memory flat, code dropped to `0.3486`.
  - Interpretation: lower LR slows memory as intended, but may under-push early memory and still needs more iterations.

06:33 intervention:

- B was stopped after `iter_006` rollout because tool collapsed:
  - Tool sample mean `0.9564 -> 0.1370`.
  - Tool all-fail rows `0 -> 19`.
  - Memory sample mean rose to `0.8281`, so the failure pattern matches memory over-push after a strong `iter_005` checkpoint.
- B useful checkpoint:
  - `iter_005` gate checkpoint is the current best global-parameter dynamic-OPD candidate: memory high, tool intact, code not collapsed.
- Replacement G launched on GPUs `2,3`:
  - Run: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_G_gp_target3_lr03_m02_20260515_dynloss_i15_v1`
  - Strategy: `global-parameter`.
  - LR `0.03`, SGD momentum `0.2`.
  - OPD target ratios lowered from `5/3/1/0.33` to `3/2/0.67/0.2`.
  - Purpose: test whether a lower OPD target avoids the memory-overpush collapse while retaining faster progress than E.
- Monitor restarted on `http://127.0.0.1:8783` with A/B/C/D/E/F/G visible.

06:50 intervention:

- C was stopped after `iter_006` rollout because tool collapsed:
  - Tool sample mean `0.9185 -> 0.0268`.
  - Tool all-fail rows `0 -> 31`.
  - The fixed OPD reference also reaches the same memory-overpush failure mode, though later than A.
- C useful checkpoints:
  - `iter_004`: tool and memory strong, code still acceptable.
  - `iter_005`: memory higher, tool still usable, code weaker.
- Replacement H launched on GPUs `4,5`:
  - Run: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_H_gp_target3_lr025_m02_20260515_dynloss_i15_v1`
  - Strategy: `global-parameter`.
  - LR `0.025`, SGD momentum `0.2`.
  - OPD target ratios `3/2/0.67/0.2`.
  - Purpose: combine E's lower LR with G's lower OPD target; this is the most conservative anti-overpush setting.
- Monitor restarted on `http://127.0.0.1:8783` with A/B/C/D/E/F/G/H visible.

07:27 conservative-run status:

- E `global-parameter, LR=0.025, momentum=0.2, original OPD target`:
  - `iter_004` rollout: code `0.4287`, memory `0.4219`, tool `0.9649`.
  - No tool collapse by iteration 4.
  - Current read: safest ongoing run; slower memory growth but preserves tool/code better.
- F `global-coefficient, LR=0.025, momentum=0.2`:
  - `iter_003` rollout: code `0.4062`, memory `0.4375`, tool `0.9067`.
  - Current read: low-LR global-coefficient avoids A's early overpush so far.
- G `global-parameter, lowered OPD target, LR=0.03, momentum=0.2`:
  - First update: `grad_norm_max=9.32`, memory gate `0.3632`, tool gate `0.3307`.
  - `iter_002` rollout: code `0.3730`, memory `0.3672`, tool `0.7463`.
  - Current read: lower OPD target substantially reduces first-step gradient and may be stable, but memory progress is slower.
- H `global-parameter, lowered OPD target, LR=0.025, momentum=0.2`:
  - First update: `grad_norm_max=14.55`, memory gate `0.3582`, tool gate `0.3313`.
  - Current read: most conservative anti-overpush run; needs more rollout evidence.

08:00 stability checkpoint:

- E reached `iter_005` rollout without tool collapse:
  - Code sample mean `0.4375`.
  - Memory sample mean `0.4922`.
  - Tool sample mean `0.9668`.
  - Gates after `iter_004`: `code=0.3339`, `memory=0.4502`, `tool=0.3316`.
  - Interpretation: E is now the best live run because it improves tool strongly, raises memory gradually, and preserves code better than B/A/C near their collapse point.
- F reached `iter_004` rollout:
  - Code sample mean `0.3955`.
  - Memory sample mean `0.4531`.
  - Tool sample mean `0.9610`.
  - Interpretation: low-LR global-coefficient also looks stable so far, but code is lower than E.
- H reached `iter_002` update:
  - Gates: `code=0.3344`, `memory=0.3879`, `tool=0.3279`.
  - Tool rollout at `iter_002` is `0.6695`, below E/F at comparable later stages.
- Current priority:
  - Continue E and F.
  - Continue G/H as conservative target ablations.
  - Preserve B `iter_005`, A `iter_003`, C `iter_004/005` as stopped-run candidate checkpoints.

08:36 stability checkpoint:

- E `global-parameter, LR=0.025, momentum=0.2, original OPD targets 5/3/1/0.33`:
  - `iter_006` rollout finished before update.
  - Reward: code `0.4375 -> 0.3490`, memory `0.4922 -> 0.6328`, tool `0.9668 -> 0.9705`.
  - OPD rows for `iter_006` dropped to 4: `code=3`, `memory=1`, `tool=0`.
  - Gates after `iter_005`: `code=0.3341`, `memory=0.4814`, `tool=0.3314`.
  - Interpretation: E is still stable; OPD has naturally weakened because all-fail recoverable rows are now sparse. After this point the run should be judged mostly by GRPO frontier + retention, not by OPD push.
- F `global-coefficient, LR=0.025, momentum=0.2, original OPD targets`:
  - `iter_005` rollout finished; update is running.
  - Reward: code `0.3955 -> 0.4277`, memory `0.4531 -> 0.5078`, tool `0.9610 -> 0.9719`.
  - OPD rows for `iter_005`: `code=2`, `memory=4`, `tool=0`.
  - Gates after `iter_004`: `code=0.3335`, `memory=0.4505`, `tool=0.3318`.
  - Interpretation: F is the cleanest 3-coefficient run so far; it improves all three proxy rewards through `iter_005` and has not shown the high-LR overpush failure.
- G `global-parameter, LR=0.03, momentum=0.2, lowered OPD targets 3/2/0.67/0.2`:
  - `iter_004` rollout finished; update is running.
  - Reward: code `0.3770 -> 0.4268`, memory `0.3750 -> 0.4141`, tool `0.9427 -> 0.9490`.
  - Gates after `iter_003`: `code=0.3347`, `memory=0.4362`, `tool=0.3292`.
  - Interpretation: G is stable but slower on memory than E/F; useful as lower-OPD-target ablation.
- H `global-parameter, LR=0.025, momentum=0.2, lowered OPD targets`:
  - `iter_003` rollout/update finished and `iter_004` rollout is running.
  - Reward through `iter_003`: code `0.3770`, memory `0.4219`, tool `0.9060`.
  - Gates after `iter_003`: `code=0.3346`, `memory=0.4188`, `tool=0.3277`.
  - Interpretation: H is the most conservative run. It has not collapsed, but it may under-push memory relative to E/F.

Current conclusion:

- The current logic is aligned with the intended principle: LR controls the absolute step size; dynamic OPD/retention scaling controls the relative loss range; task-balanced scaling prevents a single task's raw NLL length from dominating the auxiliary loss.
- The empirical stable region is lower than the earlier high-LR runs. Memory gate around `0.60+` was associated with later tool collapse in A/B/C, while E/F remain stable below roughly `0.50` so far.
- Immediate decision: do not stop E/F/G/H yet. Preserve E/F `iter_005` and B `iter_005` as important checkpoint candidates if later iterations collapse.

08:56 G rollout checkpoint:

- G `iter_004` update completed:
  - Mean gates: `code=0.3352`, `memory=0.4735`, `tool=0.3286`.
  - `grad_norm_max=14.37`, `gate_delta_max=0.0383`.
  - OPD rows: `code=1`, `memory=7`, `tool=0`; retention rows `12`.
- G `iter_005` rollout:
  - Code `0.4268 -> 0.3428`.
  - Memory `0.4141 -> 0.5703`.
  - Tool `0.9490 -> 0.9677`.
  - Interpretation: G avoids tool collapse at memory gate around `0.47`, but code drops sharply. This is a useful lower-OPD-target ablation, not yet the best candidate.
- E/F/H updates are still running but active:
  - F `iter_005` update elapsed about 26 minutes; E `iter_006` about 22 minutes; H `iter_004` about 17 minutes at the time of check.
  - CPU and GPU utilization were nonzero, so no intervention was made.

09:08 F/H checkpoint:

- F `iter_005` update completed:
  - Gates: `code=0.3337`, `memory=0.4817`, `tool=0.3315`.
  - `grad_norm_max=13.91`, `gate_delta_max=0.0312`.
  - OPD rows: `code=2`, `memory=4`, `tool=0`; retention rows `18`.
- F `iter_006` rollout:
  - Code `0.4277 -> 0.3916`.
  - Memory `0.5078 -> 0.6016`.
  - Tool `0.9719 -> 0.9569`.
  - Interpretation: F still avoids tool collapse and is now the best low-LR global-coefficient candidate. Main risk is code degradation while memory keeps rising.
- H `iter_004` update completed:
  - Gates: `code=0.3346`, `memory=0.4499`, `tool=0.3276`.
  - `grad_norm_max=12.90`, `gate_delta_max=0.0319`.
  - OPD rows: `code=1`, `memory=7`, `tool=1`; retention rows `12`.
  - Interpretation: H remains conservative. It has not collapsed, but memory/code gains are likely weaker than F/E.

09:14 E rollout checkpoint:

- E `iter_007` rollout after `iter_006` gate update:
  - Gates entering rollout: `code=0.3348`, `memory=0.5125`, `tool=0.3305`.
  - Code `0.3490 -> 0.3633`.
  - Memory `0.6328 -> 0.7656`.
  - Tool `0.9705 -> 0.9608`.
  - Interpretation: E is now a strong candidate. It raises memory substantially while keeping tool stable. Code remains the main weakness, but there is no collapse. Preserve E `iter_006` and E `iter_007` as candidate checkpoints pending the next update/rollout.

09:28 G rollout checkpoint:

- G `iter_005` update:
  - Gates: `code=0.3369`, `memory=0.5109`, `tool=0.3277`.
  - `grad_norm_max=3.86`, `gate_delta_max=0.0383`.
  - OPD rows: `code=2`, `memory=2`, `tool=0`; retention rows `17`.
- G `iter_006` rollout:
  - Code `0.3428 -> 0.3711`.
  - Memory `0.5703 -> 0.6875`.
  - Tool `0.9677 -> 0.9631`.
  - Interpretation: G remains stable after entering the `memory_gate ~= 0.51` region. It is a viable lower-OPD-target candidate, with slightly weaker memory than E but comparable tool stability.

09:40 E/F checkpoint:

- E `iter_008` rollout:
  - Code `0.3633 -> 0.3359`.
  - Memory `0.7656 -> 0.7656`.
  - Tool `0.9608 -> 0.9571`.
  - Dynamic OPD for the next update selected `code=6`, `memory=2`, `tool=1`, so the next update has a plausible code-recovery signal.
  - Interpretation: E remains high-overall and tool-stable, but code degradation makes `iter_007` a safer candidate than `iter_008` unless the next rollout recovers code.
- F `iter_006` update:
  - Gates: `code=0.3363`, `memory=0.5128`, `tool=0.3310`.
  - `grad_norm_max=3.94`, `gate_delta_max=0.0311`.
- F `iter_007` rollout:
  - Code `0.3916 -> 0.4189`.
  - Memory `0.6016 -> 0.6641`.
  - Tool `0.9569 -> 0.9527`.
  - Interpretation: F is currently the best balanced global-coefficient candidate. It raises memory, keeps tool, and recovers code better than E at comparable stage.

09:53 G/H checkpoint:

- G `iter_006` update:
  - Gates: `code=0.3376`, `memory=0.5483`, `tool=0.3279`.
  - `grad_norm_max=5.58`, `gate_delta_max=0.0385`.
  - OPD rows: `code=2`, `memory=1`, `tool=0`; retention rows `18`.
  - Interpretation: G is approaching the upper stable memory range. The next rollout is important for detecting tool/code tradeoff.
- H `iter_006` rollout:
  - Code `0.3545 -> 0.3965`.
  - Memory `0.4297 -> 0.6094`.
  - Tool `0.9599 -> 0.9681`.
  - Interpretation: H improved after the conservative `iter_005` update and should not be stopped. It is now a stable low-LR/lower-target candidate.

10:05 E/G checkpoint:

- E `iter_008` update:
  - Gates: `code=0.3384`, `memory=0.5745`, `tool=0.3324`.
  - `grad_norm_max=5.92`, `gate_delta_max=0.0318`.
  - OPD rows: `code=6`, `memory=2`, `tool=1`; retention rows `18`.
  - Interpretation: E is close to the high-memory region but still below the previous collapse zone. The next rollout should decide whether to keep pushing or freeze E at `iter_007/008`.
- G `iter_007` rollout:
  - Code `0.3711 -> 0.3617`.
  - Memory `0.6875 -> 0.7812`.
  - Tool `0.9631 -> 0.9641`.
  - Interpretation: G is stable and memory/tool-strong, but code is weaker than F. F remains the most balanced candidate so far.

10:16 E/F checkpoint:

- E `iter_009` rollout, generated by `iter_008/gate_updates.gates.json`:
  - Code `0.3359 -> 0.4066`.
  - Memory `0.7656 -> 0.8125`.
  - Tool `0.9571 -> 0.9555`.
  - Interpretation: E recovered code while further improving memory and preserving tool. This is the strongest live global-parameter candidate so far. For external evaluation, the checkpoint corresponding to this behavior is E `iter_008/gate_updates.gates.json`.
- F `iter_007` update:
  - Gates: `code=0.3372`, `memory=0.5440`, `tool=0.3296`.
  - `grad_norm_max=7.65`, `gate_delta_max=0.0312`.
  - OPD rows: `code=2`, `memory=2`, `tool=1`; retention rows `18`.
  - Interpretation: F remains balanced, but the next rollout is needed to see whether it matches E's memory gain without code loss.

10:32 F/G/H checkpoint:

- F `iter_008` rollout, generated by `iter_007/gate_updates.gates.json`:
  - Code `0.4189 -> 0.4346`.
  - Memory `0.6641 -> 0.7891`.
  - Tool `0.9527 -> 0.9711`.
  - Interpretation: F is currently the best balanced candidate. It improves all three proxy rewards at once. For external evaluation, the corresponding checkpoint is F `iter_007/gate_updates.gates.json`.
- G `iter_007` update:
  - Gates: `code=0.3400`, `memory=0.5856`, `tool=0.3280`.
  - `grad_norm_max=5.85`, `gate_delta_max=0.0385`.
  - Interpretation: G is close to the previous high-risk memory region. Its next rollout should be monitored for tool/code collapse.
- H `iter_006` update:
  - Gates: `code=0.3350`, `memory=0.5122`, `tool=0.3271`.
  - `grad_norm_max=10.03`, `gate_delta_max=0.0319`.
  - Interpretation: H remains lower-risk and now has meaningful memory movement.

10:42 E/H checkpoint:

- E `iter_009` update:
  - Gates: `code=0.3410`, `memory=0.6054`, `tool=0.3346`.
  - `grad_norm_max=6.55`, `gate_delta_max=0.0320`.
  - OPD rows: `code=4`, `memory=1`, `tool=2`; retention rows `19`.
  - Interpretation: E has entered the previous high-risk memory region. Continue only to observe `iter_010` rollout; if tool collapses, stop E and keep `iter_008` as the best E checkpoint.
- H `iter_007` rollout:
  - Code `0.3965 -> 0.4154`.
  - Memory `0.6094 -> 0.7109`.
  - Tool `0.9681 -> 0.9589`.
  - Interpretation: H is now a real lower-target/low-LR candidate, not just a weak conservative control.

10:50 E stop:

- E `iter_010` rollout, generated by E `iter_009/gate_updates.gates.json`, triggered the stop rule:
  - Code `0.4066 -> 0.4057`.
  - Memory `0.8125 -> 0.8047`.
  - Tool `0.9555 -> 0.2755`.
  - Tool all-fail rows increased to `12/32`.
- Interpretation:
  - E `iter_009` update pushed memory gate to `0.6054`, which crossed the empirically risky region.
  - The failure is specifically tool degradation after over-pushing memory, not a global reward collapse.
  - E `iter_008/gate_updates.gates.json` remains the best E checkpoint because it generated the strong `iter_009` rollout.
- Action:
  - Stopped tmux session `opvec_dynscale_E_gp_lr025_m02_20260515_dynloss_i15_v1`.
  - GPUs `6,7` were released after stopping E.

10:27 E Eval6 launch:

- Added reusable Eval6 addon runner:
  - `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_dynscale_20260515_addons.py`
- Launched E-best full Eval6 on GPU `6`:
  - Model name: `dynscale-e-gp-lr025-m02-best-iter8gate`
  - Model path: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_E_gp_lr025_m02_20260515_dynloss_i15_v1/iter_009/baked_policy`
  - Tmux: `eval_dynscale_E_best_20260515`
  - Log: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/logs/dynscale_E_best_20260515_eval.log`

10:55 F/eval checkpoint:

- F `iter_008` update:
  - Gates: `code=0.3381`, `memory=0.5752`, `tool=0.3297`.
  - `grad_norm_max=8.50`, `gate_delta_max=0.0312`.
  - OPD rows: `code=2`, `memory=1`; retention rows `17`.
  - Interpretation: F is approaching E's high-memory region. Continue to `iter_009` rollout and stop if tool drops sharply.
- E Eval6:
  - BFCL phase completed.
  - Memory phase is running on GPU `6`.

11:00 F Eval6 launch:

- F `iter_009` rollout:
  - Code `0.4346 -> 0.4008`.
  - Memory `0.7891 -> 0.7969`.
  - Tool `0.9711 -> 0.9429`.
  - Interpretation: F remains stable but `iter_008` is still the best proxy point.
- Launched F-best full Eval6 on GPU `7`:
  - Model name: `dynscale-f-gc-lr025-m02-best-iter7gate`
  - Model path: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_F_gc_lr025_m02_20260515_dynloss_i15_v1/iter_008/baked_policy`
  - Tmux: `eval_dynscale_F_best_20260515`
  - Log: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/logs/dynscale_F_best_20260515_eval.log`

11:04 G checkpoint:

- G `iter_008` update:
  - Gates: `code=0.3410`, `memory=0.6211`, `tool=0.3384`.
  - `grad_norm_max=5.27`, `gate_delta_max=0.0366`.
  - OPD rows: `code=3`, `memory=1`, `tool=3`; retention rows `18`.
  - Interpretation: G has entered the high-risk memory region. Continue only to observe the next rollout; if tool drops sharply, stop G and keep `iter_008/baked_policy` as the best G candidate.

11:08 Eval/monitor checkpoint:

- Training status:
  - F: `g8/r9`; currently updating `iter_009`.
  - G: `g8/r8`; currently rolling out after high-risk `memory_gate=0.6211`.
  - H: `g7/r7`; currently rolling out / updating normally.
- Eval6 status:
  - E-best: BFCL completed; memory inference completed; CURE running on GPU `6`.
  - F-best: BFCL completed; memory inference running on GPU `7`.
- Current candidate ranking by proxy:
  - F `iter_008/baked_policy`: best balanced proxy, code `0.4346`, memory `0.7891`, tool `0.9711`.
  - E `iter_009/baked_policy`: strong memory/code/tool before overpush, but E later collapsed after memory gate crossed `0.60`.
  - H latest stable: conservative lower-target candidate.
  - G `iter_008/baked_policy`: useful but G has entered high-risk gate region.

11:12 G stop:

- G `iter_009` rollout, generated after G `iter_008` update, triggered the stop rule:
  - Code `0.4316 -> 0.4033`.
  - Memory `0.7969 -> 0.8359`.
  - Tool `0.8835 -> 0.0569`.
  - Tool all-fail rows increased to `27/32`.
- Interpretation:
  - G `iter_008` update pushed memory gate to `0.6211`.
  - This reproduces E's overpush failure mode: memory continues improving while tool collapses.
  - G `iter_008/baked_policy` remains the best G candidate; G should not be continued past this point.
- Action:
  - Stopped tmux session `opvec_dynscale_G_gp_target3_lr03_m02_20260515_dynloss_i15_v1`.

10:57 F stop and replacement I launch:

- F `iter_009` update pushed global-coefficient memory gate to `0.6065`:
  - Gates after update: `code=0.3384`, `memory=0.6065`, `tool=0.3309`.
  - OPD rows: `code=3`, `memory=2`, `tool=2`; retention rows `17`.
- F `iter_010` rollout was stopped on partial evidence before the updater could consume it:
  - Partial completed rows at stop check: `tool=26`, `memory=26`, `code=24` prompts.
  - Partial reward: `code=0.3555`, `memory=0.7981`, `tool=0.3321`.
  - Partial all-fail: `tool=10/26`, `memory=4/26`, `code=8/24`.
- Interpretation:
  - This is the same failure mode as E/G: once memory gate crosses roughly `0.60`, tool collapses in the next rollout.
  - F best checkpoint remains `iter_008/baked_policy`, generated by `iter_007/gate_updates.gates.json`: `code=0.4346`, `memory=0.7891`, `tool=0.9711`.
- Action:
  - Stopped tmux session `opvec_dynscale_F_gc_lr025_m02_20260515_dynloss_i15_v1`.
  - Released GPUs `0,1`.
  - Launched replacement I on GPUs `0,1`.

Replacement I:

- Run: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_I_gc_target3_lr018_m02_20260515_dynloss_i15_v1`
- Tmux: `opvec_dynscale_I_gc_target3_lr018_m02_20260515_dynloss_i15_v1`
- Strategy: `global-coefficient`.
- Init: `1/3`.
- LR: `0.018`.
- Momentum: `0.2`.
- OPD target ratios: `3/2/0.67/0.2`, lower than F's original `5/3/1/0.33`.
- Retention: dynamic task-balanced NLL, target `0.5`.
- Purpose:
  - Test whether the 3-coefficient global setting can preserve F's useful early trajectory while slowing the memory push enough to avoid the `memory ~= 0.60` tool-collapse boundary.

Monitor update:

- Frontend restarted on `http://127.0.0.1:8783`.
- Visible runs: stopped A/B/C/D/E/F/G plus live H/I.

## ABCD Gate Movement Analysis

Question: whether A/B/C/D gates were actually pushed.

Answer: A/B/C were pushed clearly, but mostly along the memory coefficient; D/no-OPD was the only setting without useful task-vector-upward movement.

Observed gate trajectories:

| Run | Setting | Last completed gate | Movement from 1/3 | Outcome |
|---|---|---:|---|---|
| A | global-coefficient, dynamic OPD, high LR/target | code `0.3389`, memory `0.6895`, tool `0.3581` at iter 4 | memory strongly up | tool collapsed at next rollout |
| B | global-parameter, dynamic OPD, high LR/target | code `0.3377`, memory `0.6143`, tool `0.3337` at iter 5 | memory strongly up | tool collapsed at next rollout |
| C | global-parameter, fixed OPD reference | code `0.3324`, memory `0.6542`, tool `0.3228` at iter 5 | memory strongly up, tool down | tool collapsed at next rollout |
| D | global-coefficient, no OPD | code `0.3237`, memory `0.3167`, tool `0.2763` at iter 3 | no useful upward push | unstable/noisy, stopped |

Why it can look like "not pushed":

- If only watching code/tool gates, they stay near `1/3`; the real movement is concentrated in memory.
- For global-parameter, the frontend may average 588 coefficients; mean movement is still visible, but module-level variation can visually dilute changes.
- D is a true negative control: without OPD, GRPO+retention produced small gradients and did not push the task vector upward.

Why A/B/C mostly push memory:

- Dynamic OPD selects current policy all-fail prompts that the expert can solve. After the first rollout, the remaining recoverable all-fail rows are heavily memory-skewed.
- Tool is already near-saturated on many calibration rows after early updates, so tool contributes mostly all-success retention rather than OPD repair pressure.
- Code has fewer clean all-fail/expert-success rows in this 96-prompt set, so its OPD pressure is intermittent.
- GRPO frontier rows only provide signal when samples under the same prompt have reward variance. All-success/all-fail prompts give no GRPO advantage; they enter only retention or OPD.
- Retention NLL preserves already successful behavior, but it does not explicitly push the corresponding task-vector coefficient upward.
- Therefore the net objective in A/B/C is: memory OPD pushes hard, tool/code mostly preserve, and once memory crosses roughly `0.60`, the merged model leaves the tool-stable region.

Current implication:

- The issue is not "gate cannot move"; it moves too easily along the dominant memory repair direction.
- The next useful experiments should slow or constrain memory movement while preserving the early good region. Replacement I does this by keeping global-coefficient but lowering OPD target and LR.

## Comparison To 2026-05-14 Best GP-OPD

Reference report:

`/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opvec_best_gp_opd_reproduction_20260514.md`

Short answer:

- If judged only by on-policy proxy reward, the 2026-05-15 dynamic-loss runs produced several points close to the 2026-05-14 best.
- If judged by the stricter standard of "same kind of success" as 2026-05-14, then not yet. The current runs have not reproduced the same useful gate geometry, and formal Eval6 results are still pending.

Reference best from 2026-05-14:

| Item | Value |
|---|---:|
| run | `qbank_c033333_gp_epoch_opdcompact_sgd_m08_n48_i10_20260514_0154` |
| best rollout | iter9, using iter8 gate |
| proxy overall | `0.7189` |
| proxy reward | tool `0.9809`, memory `0.7188`, code `0.4570` |
| gate effective mean | tool `0.2903`, memory `0.5439`, code `0.7136` |
| formal Eval6 | Tool kept, Memory improved, Code slightly improved |

Closest 2026-05-15 proxy candidates:

| Run | Best rollout | Proxy overall | Proxy reward | Gate entering rollout | Later behavior |
|---|---:|---:|---|---|---|
| F | iter8 | `0.7316` | tool `0.9711`, memory `0.7891`, code `0.4346` | code `0.3372`, memory `0.5440`, tool `0.3296` | next updates pushed memory to `0.6065`; iter10 partial showed tool collapse |
| E | iter9 | `0.7249` | tool `0.9555`, memory `0.8125`, code `0.4066` | code `0.3384`, memory `0.5745`, tool `0.3324` | next update pushed memory to `0.6054`; iter10 tool collapsed |
| H | iter9 | `0.7250` | tool `0.9602`, memory `0.7812`, code `0.4336` | code `0.3367`, memory `0.5745`, tool `0.3268` | currently live; must watch next update/rollout |
| G | iter8 | `0.7040` | tool `0.8835`, memory `0.7969`, code `0.4316` | code `0.3400`, memory `0.5856`, tool `0.3280` | next update pushed memory to `0.6211`; iter9 tool collapsed |

Key distinction:

- 2026-05-14 best did not just reach a high proxy reward. It found a useful **code + memory** gate geometry: code was pushed strongly (`~0.71`), memory moderately (`~0.54`), tool lower but still behaviorally retained.
- 2026-05-15 dynamic-loss runs mostly found a **memory-only** direction: memory rises, code stays around `0.34`, tool stays around `0.33` until memory crosses a hidden stability boundary.
- Therefore F/E/H can look numerically competitive on proxy reward, but they are not yet equivalent to the 2026-05-14 success.

Why 2026-05-15 did not reproduce the same success mode:

1. Different OPD data construction.
   - 2026-05-14 used a fixed compact OPD replay: 21 rows, nominally 7 per task, from high-information historical samples.
   - In practice, the old compact file produced strong non-zero OPD mainly for memory/code, with tool mostly maintained by on-policy GRPO.
   - 2026-05-15 uses dynamic same-prompt OPD from current all-fail rows. After tool is quickly repaired and many prompts become all-success, the remaining recoverable all-fail rows become memory-heavy; code signal is sparse.

2. Different gate geometry pressure.
   - 2026-05-14 pushed code and memory apart in the 588-parameter global-parameter space.
   - 2026-05-15 mostly pushes memory while code remains near the `1/3` initialization.
   - This matters because memory-only movement eventually damages tool behavior even if tool gate itself does not decrease much.

3. Dynamic OPD scale fixes gradient magnitude but not direction.
   - The dynamic scaler successfully prevents the "no gradient" failure and gives stable `grad_norm_max` around `5-10`.
   - But it scales whichever OPD rows are available. If available OPD rows are memory-skewed, it faithfully amplifies memory pressure.
   - This is why A/B/C/E/F/G/H all show the same pattern: memory improves first, then tool collapses once memory gate reaches about `0.60`.

4. Retention is not a true trust-region constraint.
   - Current retention is all-success NLL preservation. It gives some non-zero gradient on successful rows.
   - It does not explicitly constrain the merged model to stay within a tool-stable region as memory gate moves.
   - Empirically, retention did not prevent E/F/G collapse after memory crossed `~0.60`.

5. Proxy reward is partly saturated.
   - Tool proxy becomes high early on many paper96 prompts.
   - A run can therefore match the 2026-05-14 overall proxy by combining saturated tool + high memory, even if code is weaker and the gate geometry is less robust.
   - Formal Eval6 is required before treating F/E/H as true successes.

Current interpretation:

- H/F/E are promising proxy checkpoints, not confirmed replacements for the 2026-05-14 best.
- The strongest current live candidate is H iter9 because it matches the proxy level without collapse so far: `overall=0.7250`, tool `0.9602`, memory `0.7812`, code `0.4336`.
- The next decisive check is whether H's next update pushes memory above the `0.60` boundary and reproduces E/F/G collapse. If it does, the failure is systematic: current dynamic all-fail OPD needs either task-direction balancing, a memory trust-region/cap, or a code-positive OPD source closer to the 2026-05-14 compact replay.

11:19 H9 Eval6 launch:

- H `iter_009` rollout became the strongest live proxy candidate:
  - `overall=0.7250`, tool `0.9602`, memory `0.7812`, code `0.4336`.
  - Gate entering this rollout was H `iter_008/gate_updates.gates.json`: mean code `0.3367`, memory `0.5745`, tool `0.3268`.
- This is still not equivalent to the 2026-05-14 best because the gate geometry is memory-heavy and code remains near `1/3`.
- Since GPU `3` was free, H iter9 baked policy was sent to full Eval6:
  - Model name: `dynscale-h-gp-target3-lr025-m02-best-iter8gate`
  - Model path: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_H_gp_target3_lr025_m02_20260515_dynloss_i15_v1/iter_009/baked_policy`
  - Tmux: `eval_dynscale_H9_best_20260515`
  - Log: `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/logs/dynscale_H9_best_20260515_eval.log`
- Runner update:
  - Added non-destructive `H9` entry to `/mnt/cache/wuruixiao/users/lsc/AgentMerging/skill/plan/v1—feedback/evaluation/eval-batch/eval6-20260502-125748/run_dynscale_20260515_addons.py`.
  - Existing `H` entry was left unchanged.

11:39 I iter1 update:

- Replacement I completed `iter_001` update.
- Setting: `global-coefficient`, LR `0.018`, momentum `0.2`, lowered OPD targets `3/2/0.67/0.2`.
- Initial rollout reward:
  - overall `0.3800`, tool `0.4151`, memory `0.3281`, code `0.3969`.
  - OPD selected rows: `tool=7`, `memory=6`, `code=1`.
- Update result:
  - Gates: code `0.333361`, memory `0.351276`, tool `0.331898`.
  - `grad_norm_max=10.3303`, `gate_delta_max=0.01794`.
  - Frontier counts: code `25`, memory `23`, tool `25`; retention rows `3`.
- Interpretation:
  - I behaves as intended: it still moves, but the memory slope is much slower than F/A.
  - Compare F first update memory `0.3931` with I first update memory `0.3513`.
  - This is a useful anti-overpush ablation; the next check is whether iter2 rollout improves tool/memory without rapidly entering the `memory ~= 0.60` collapse region.

11:35 H stop:

- H `iter_009` update completed after the strong H `iter_009` rollout.
- Gate after update:
  - code `0.337877`, memory `0.605451`, tool `0.329314`.
  - `grad_norm_max=4.9874`, `gate_delta_max=0.0320`.
  - OPD rows: `code=1`, `memory=1`, `tool=1`; retention rows `16`.
- Interpretation:
  - H crossed the empirical `memory ~= 0.60` tool-collapse boundary.
  - E, F, and G all collapsed on the rollout immediately after crossing this region.
  - Since H9 has already been preserved and sent to Eval6, continuing H would mostly spend compute to reproduce the known collapse mode.
- Action:
  - Stopped tmux session `opvec_dynscale_H_gp_target3_lr025_m02_20260515_dynloss_i15_v1`.
  - Released GPUs `4,5`.
  - Frontend restarted on `http://127.0.0.1:8783` with H marked as stopped and I live.
- Preserved H candidate:
  - `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_H_gp_target3_lr025_m02_20260515_dynloss_i15_v1/iter_009/baked_policy`

11:37 Replacement J launch:

- Failure reason addressed:
  - E/F/G/H all show the same boundary: memory coefficient around `0.60+` predicts or triggers tool collapse.
- Replacement J tests an explicit coefficient trust-region:
  - Run: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_J_gp_target3_lr018_cap024_m02_20260515_dynloss_i15_v1`
  - Tmux: `opvec_dynscale_J_gp_target3_lr018_cap024_m02_20260515_dynloss_i15_v1`
  - GPUs: `4,5`
  - Strategy: `global-parameter`
  - LR: `0.018`
  - Momentum: `0.2`
  - OPD targets: `3/2/0.67/0.2`
  - `MAX_COEFF_DELTA=0.24`, so the nominal upper bound from `1/3` is about `0.573`.
- Purpose:
  - Determine whether a bounded global-parameter run can preserve the H9-style high proxy region without crossing the observed tool-collapse boundary.
- Monitor update:
  - Frontend restarted on `http://127.0.0.1:8783` with stopped A-H and live I/J.

11:43 I iter2 rollout:

- I `iter_002` rollout used I `iter_001` gate:
  - code `0.333361`, memory `0.351276`, tool `0.331898`.
- Reward improved across all three proxy tasks:
  - overall `0.3800 -> 0.4630`.
  - tool `0.4151 -> 0.5569`.
  - memory `0.3281 -> 0.4219`.
  - code `0.3969 -> 0.4102`.
- All-fail rows:
  - tool `5 -> 4`.
  - memory `9 -> 8`.
  - code `5 -> 3`.
- Dynamic OPD rows for the next update:
  - `tool=8`, `memory=6`, `code=2`, total `16`.
- Interpretation:
  - Lower LR/lower OPD target does not kill learning; it produces a slower but healthier first improvement.
  - Unlike F/H, I is still far from the memory-collapse boundary after one update.
  - Continue I; next important check is whether iter2 update keeps memory slope controlled.

## 中文结论：是否复现了 2026-05-14 Best

对照 `/mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/docs/report/opvec_best_gp_opd_reproduction_20260514.md`，昨晚到今天上午的 dynamic-loss 系列**还没有复现出同一种成功**。

严格说，有几个 proxy reward 接近甚至略高的点，但它们不是 2026-05-14 best 那种可解释、可迁移的成功。2026-05-14 best 的关键不是单轮 proxy overall 高，而是 `global-parameter` 的 588 个 gate 在任务方向上形成了合理几何：`code ~= 0.71`、`memory ~= 0.54`、`tool ~= 0.29`，正式 Eval6 里 Tool 保持、Memory 明显增强、Code 略有收益。昨晚的 E/F/G/H 候选主要是 memory 方向被推高，code 基本停在 `1/3` 附近，tool 也基本停在 `1/3` 附近；proxy 上看起来高，是因为 tool 很早饱和，再叠加 memory 提升，但没有学到 2026-05-14 的 code+memory 分离结构。

| 实验 | 最好 proxy 点 | proxy reward | gate 形态 | 后续行为 | 判断 |
|---|---:|---|---|---|---|
| 2026-05-14 best | iter9 | tool `0.9809`, memory `0.7188`, code `0.4570` | code `0.7136`, memory `0.5439`, tool `0.2903` | Eval6 通过，三能力相对稳 | 真成功 |
| F | iter8 | tool `0.9711`, memory `0.7891`, code `0.4346` | code `0.3372`, memory `0.5440`, tool `0.3296` | memory 后续到 `0.6065`，tool 崩 | proxy 成功，不是同类成功 |
| E | iter9 | tool `0.9555`, memory `0.8125`, code `0.4066` | code `0.3384`, memory `0.5745`, tool `0.3324` | memory 后续到 `0.6054`，tool 崩 | proxy 成功，不是同类成功 |
| H | iter9 | tool `0.9602`, memory `0.7812`, code `0.4336` | code `0.3367`, memory `0.5745`, tool `0.3268` | update 后 memory 到 `0.6055`，按 E/F/G 经验高风险 | 待 Eval6，但形态不对 |
| G | iter8 | tool `0.8835`, memory `0.7969`, code `0.4316` | code `0.3400`, memory `0.5856`, tool `0.3280` | memory 后续到 `0.6211`，tool 崩 | 失败复现 |

核心问题不是 gate 推不动，而是**推的方向错了**。dynamic OPD 当前按“policy 全错、expert 能做对”抽样；第一两轮 tool 被修复后，剩余 all-fail recoverable 样本明显偏 memory，code 的可恢复样本稀疏。因此动态 loss scale 虽然把梯度幅值救回来了，却只是在放大当前可见的 memory repair 信号。结果是 memory gate 稳定上升，code gate 没被充分推开，最终一旦 memory 均值进入 `0.60` 左右区域，就触发 tool 行为崩溃。这个边界在 E/F/G/H 上反复出现，说明是系统性 merge geometry 问题，不是单个 run 的偶然噪声。

另一个差异是 OPD 数据源。2026-05-14 best 用的是固定 compact OPD replay，虽然名义上 21 条、三任务各 7 条，但实际非零梯度更像 memory/code 强 OPD + tool 由 GRPO 维持；这恰好推动出了 code+memory 的有效方向。昨晚的新 dynamic OPD 更“在线”，但它依赖当前 policy 的 all-fail 分布，而这个分布会快速变成 memory-heavy。也就是说，新方法更规范，但没有保证任务方向的可辨识性；旧方法不够干净，却意外给了 code/memory 更强、更合适的方向信号。

retention 目前也没有解决这个问题。它是 all-success NLL preservation，能提供保守约束，但不是真正的 task-vector trust region；它不会显式告诉优化器“memory 再涨会破坏 tool”，也不会主动把 code gate 推到 0.6-0.8 区间。因此它能延缓崩溃，不能阻止 memory-only 过推。

结论：昨晚没有得到 2026-05-14 best 那种可作为论文主结果的成功，只得到了若干“proxy 高、但 gate 几何不对”的候选。下一步最合理的方向不是继续单纯调大 OPD/LR，而是让训练信号重新具备任务区分能力：保留 dynamic OPD，但对 code/memory/tool 的可恢复样本做方向约束或配额；对 memory 设置显式 trust region 或 cap；补充更接近 2026-05-14 compact replay 的 code-positive OPD；同时用 Eval6 而不是 paper96 proxy 决定最终 checkpoint。

## 11:47 Live Status

Training:

| Run | State | Completed | Latest reward | Latest gate | Interpretation |
|---|---|---|---|---|---|
| I | active, `iter_002` update running | `g1/r2` | iter2 overall `0.4630`, tool `0.5569`, memory `0.4219`, code `0.4102` | iter1 code `0.3334`, memory `0.3513`, tool `0.3319` | Lower LR/lower OPD target is learning more slowly and has not entered the memory-collapse zone. |
| J | active, `iter_001` update running | `g0/r1` | iter1 overall `0.4135`, tool `0.4367`, memory `0.3984`, code `0.4053` | init `1/3`, capped max delta `0.24` | First dynamic OPD rows are tool-heavy: tool `10`, memory `3`, code `2`; wait for update direction before judging. |

Eval6 results:

E/F/G/H9 have all completed Tool/Memory/Code Eval6 as of 2026-05-15 13:02.

| Model | Tool mean | Tool detail | Memory mean F1 | Memory detail | Code |
|---|---:|---|---:|---|---|
| E `dynscale-e-gp-lr025-m02-best-iter8gate` | `0.7810` | live_parallel `0.6875`, live_parallel_multiple `0.6667`, parallel `0.9050`, parallel_multiple `0.8650` | `0.7539` | eval_50 `0.7192`, eval_100 `0.7357`, qa_32768 `0.7808`, qa_65536 `0.7798` | LiveBench `0.3926`, LiveCodeBench `0.3087`, mean `0.3506` |
| F `dynscale-f-gc-lr025-m02-best-iter7gate` | `0.7823` | live_parallel `0.6875`, live_parallel_multiple `0.6667`, parallel `0.9050`, parallel_multiple `0.8700` | `0.7459` | eval_50 `0.7169`, eval_100 `0.7468`, qa_32768 `0.7531`, qa_65536 `0.7670` | LiveBench `0.4023`, LiveCodeBench `0.3009`, mean `0.3516` |
| G `dynscale-g-gp-target3-lr03-m02-best-iter7gate` | `0.7810` | live_parallel `0.6875`, live_parallel_multiple `0.6667`, parallel `0.9050`, parallel_multiple `0.8650` | `0.7589` | eval_50 `0.7417`, eval_100 `0.7432`, qa_32768 `0.7736`, qa_65536 `0.7770` | LiveBench `0.3594`, LiveCodeBench `0.3151`, mean `0.3372` |
| H9 `dynscale-h-gp-target3-lr025-m02-best-iter8gate` | `0.7810` | live_parallel `0.6875`, live_parallel_multiple `0.6667`, parallel `0.9050`, parallel_multiple `0.8650` | `0.7635` | eval_50 `0.7416`, eval_100 `0.7759`, qa_32768 `0.7659`, qa_65536 `0.7706` | LiveBench `0.3887`, LiveCodeBench `0.3058`, mean `0.3472` |

Comparison with 2026-05-14 best partial Eval6:

- Tool: current candidates are essentially tied with 2026-05-14 best (`~0.781-0.782` vs `0.7835`), so Tool is not the immediate blocker at the selected checkpoints.
- Memory: H9 is closest to 2026-05-14 best (`0.7635` vs `0.7649`), G is slightly lower, E/F lower.
- Code remains decisive. E/F are slightly above the 2026-05-14 best CURE mean (`0.3487`), H9 is slightly below, and G is clearly below.
- H9 is the closest Memory candidate but does not exceed the 2026-05-14 best, and its Code mean is slightly lower.

## 11:52 Principle Correction

User correction:

> 不可以对着期望结果调参，核心还是控制训练信号。

Operational decision:

- Stop treating LR lowering, OPD target lowering, and `MAX_COEFF_DELTA` cap as the main path to success.
- E/F/G/H already showed the failure mode clearly enough: memory-only pressure pushes the policy into a tool-collapse region.
- I/J were therefore reclassified as diagnostics, not valid mainline experiments.
- Stopped I and J tmux sessions to avoid spending compute on result-shaped hyperparameter tuning:
  - `opvec_dynscale_I_gc_target3_lr018_m02_20260515_dynloss_i15_v1`
  - `opvec_dynscale_J_gp_target3_lr018_cap024_m02_20260515_dynloss_i15_v1`
- Restarted monitor with I/J marked as stopped/diagnostic on `http://127.0.0.1:8783`.

What this means for the next valid experiment:

- Do not choose coefficients, caps, or LR schedules because they keep memory near an observed good region.
- Keep optimization hyperparameters simple and fixed.
- Change the **training signal** instead:
  - OPD source must be task-balanced after filtering, not merely before filtering.
  - If one task has few all-fail recoverable prompts, do not let memory fill the whole OPD budget by default.
  - Add explicit code-positive and tool-retention signal from the same calibration pool instead of relying on memory-heavy all-fail rows.
  - Separate "repair signal" from "preservation signal": all-fail expert OPD repairs missing ability; all-success NLL preserves already-good behavior; frontier GRPO handles within-prompt preference.
  - Measure per-task gradient contribution before stepping, so a run can be rejected because the signal is memory-dominated before observing collapse.

Current state after correction:

- Training runs A-H: stopped, with candidate checkpoints preserved.
- Diagnostic runs I/J: stopped by principle correction before becoming mainline.
- Eval6 for E/F/G/H9 continues; these evaluations are still useful for diagnosing whether high proxy reward transfers to formal Tool/Memory/Code metrics.

## 12:04 E Eval6 Complete

Model: `dynscale-e-gp-lr025-m02-best-iter8gate`

Path: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_E_gp_lr025_m02_20260515_dynloss_i15_v1/iter_009/baked_policy`

| Task | Metric | Result |
|---|---|---:|
| Tool/BFCL | mean | `0.7810` |
| Memory/HotpotQA | mean F1 | `0.7539` |
| Code/CURE | LiveBench code_acc | `0.3926` |
| Code/CURE | LiveCodeBench code_acc | `0.3087` |
| Code/CURE | mean code_acc | `0.3506` |

Comparison:

- Code is slightly above 2026-05-14 best CURE mean (`0.3506` vs `0.3487`).
- Tool is slightly below 2026-05-14 best (`0.7810` vs `0.7835`).
- Memory is below 2026-05-14 best (`0.7539` vs `0.7649`).
- E is therefore a credible evaluated checkpoint, but it is not a clean replacement for 2026-05-14 best: formal metrics are roughly tied/slightly mixed, while the gate geometry remains memory-heavy and its next update collapses Tool.

## 12:07 F Eval6 Complete

Model: `dynscale-f-gc-lr025-m02-best-iter7gate`

Path: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_F_gc_lr025_m02_20260515_dynloss_i15_v1/iter_008/baked_policy`

| Task | Metric | Result |
|---|---|---:|
| Tool/BFCL | mean | `0.7823` |
| Memory/HotpotQA | mean F1 | `0.7459` |
| Code/CURE | LiveBench code_acc | `0.4023` |
| Code/CURE | LiveCodeBench code_acc | `0.3009` |
| Code/CURE | mean code_acc | `0.3516` |

Comparison:

- F has the best Code mean so far among completed E/F (`0.3516`), slightly above both E and 2026-05-14 best.
- Tool is close to 2026-05-14 best but still slightly lower (`0.7823` vs `0.7835`).
- Memory is clearly lower than 2026-05-14 best (`0.7459` vs `0.7649`).
- Like E, F is useful evidence that the proxy-high checkpoints transfer partly to formal Code/Tool, but it does not solve the training-signal issue: the checkpoint is selected before a later memory over-push collapse, not because the objective learned a robust task-separated gate geometry.

## Code OPD Threshold Audit

User concern:

```text
CodeRewardAdapter success = score >= 0.95
DYNAMIC_OPD_POSITIVE_THRESHOLD = 1.0
OPD_POSITIVE_REWARD_THRESHOLD = 1.0
```

Code review:

- `opvec/rewards/simple.py`: public-example fallback can set code reward to `0.95`, while `success = score >= 0.95`.
- `scripts/data/build_opd_distill_from_expert_rollouts.py`: dynamic OPD builder selects expert positives by `reward_train >= positive_threshold`.
- `scripts/train/opvec_update_gates_from_rollouts.py`: update side also splits OPD positives/negatives by `reward_train` threshold.

Audit on current paper96 expert rollouts:

| Check | Result |
|---|---:|
| code expert prompts | `32` |
| code expert samples | `64` |
| `reward_train=1.0` samples | `24` |
| `success=True` samples | `24` |
| `success=True && reward_train<1.0` | `0` |
| `reward_train=0.95` samples | `0` |
| expert-positive prompts at threshold `1.0` | `17` |
| expert-positive prompts at threshold `0.95` | `17` |

Counterfactual on E/F/G/H code all-fail rows:

- `threshold=1.0` and `threshold=0.95` recover exactly the same number of code prompts in every checked iteration.
- Policy code samples across current runs also have no `reward_train=0.95`; success samples are all `reward_train=1.0`.
- OPD code positive samples are all `reward_train=1.0`.

Conclusion:

- The threshold mismatch is a real future-data risk, especially for code prompts without source tests where public-example fallback gives `0.95`.
- It is **not** the main reason code was weak in this paper96 dynamic-OPD run.
- The immediate code bottleneck is same-prompt expert coverage: only 17/32 code prompts have expert positives at `1.0`, and many policy all-fail code prompts do not match those recoverable prompts.
- Do not blindly lower global OPD threshold as a "fix" for this run. A cleaner future fix is task-specific positive semantics: allow code OPD positive by `success == true` or task-specific threshold, and record both `positive_by_success` and `positive_by_reward_threshold` in the OPD summary.

## 12:30 G Eval6 Complete

Model: `dynscale-g-gp-target3-lr03-m02-best-iter7gate`

Path: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_G_gp_target3_lr03_m02_20260515_dynloss_i15_v1/iter_008/baked_policy`

| Task | Metric | Result |
|---|---|---:|
| Tool/BFCL | mean | `0.7810` |
| Memory/HotpotQA | mean F1 | `0.7589` |
| Code/CURE | LiveBench code_acc | `0.3594` |
| Code/CURE | LiveCodeBench code_acc | `0.3151` |
| Code/CURE | mean code_acc | `0.3372` |

Comparison:

- G has better Memory than E/F, but worse Code than 2026-05-14 best (`0.3372` vs `0.3487`).
- This supports the core diagnosis: a high-memory proxy checkpoint can still lose code formal performance if the learned gate geometry is memory-heavy rather than code+memory separated.

## 13:02 H9 Eval6 Complete

Model: `dynscale-h-gp-target3-lr025-m02-best-iter8gate`

Path: `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_H_gp_target3_lr025_m02_20260515_dynloss_i15_v1/iter_009/baked_policy`

| Task | Metric | Result |
|---|---|---:|
| Tool/BFCL | mean | `0.7810` |
| Memory/HotpotQA | mean F1 | `0.7635` |
| Code/CURE | LiveBench code_acc | `0.3887` |
| Code/CURE | LiveCodeBench code_acc | `0.3058` |
| Code/CURE | mean code_acc | `0.3472` |

Comparison:

- H9 is the closest 2026-05-15 candidate to the 2026-05-14 best on Memory (`0.7635` vs `0.7649`).
- H9 Code is slightly below the 2026-05-14 best (`0.3472` vs `0.3487`) and below E/F.
- This reinforces the conclusion that 2026-05-15 dynamic-loss settings can find usable proxy checkpoints, but they did not reproduce the 2026-05-14 best gate geometry.

## 12:39 K Direct-Parameter Diagnostic Launch

User request: try full task-vector coefficient fine-tuning without common+residual.

Strategy name:

```text
STRATEGY=parameter
```

Why:

- `parameter` trains one direct coefficient per mergeable parameter/expert: `196 × 3 = 588` effective coefficients.
- `global-parameter` is not the requested setting. It trains global expert strengths plus parameter-specific residuals.
- `global-coefficient` trains only 3 direct expert coefficients.

Launched run:

| Item | Value |
|---|---|
| run | `dynscale_K_parameter_direct_signal_20260515_i8_v1` |
| run_dir | `/tmp/shared-storage/OnPolicy/runs/gated_grpo/dynscale_K_parameter_direct_signal_20260515_i8_v1` |
| tmux | `opvec_dynscale_K_parameter_direct_signal_20260515_i8_v1` |
| GPUs | `0,1` |
| parameterization | `parameter` |
| effective learnable coefficients | `588` |
| prompts / samples | `96 × 4` |
| iterations | `8` |
| init | all effective coefficients `1/3` |
| optimizer | SGD, momentum `0.2` |
| LR / prior / max delta | `0.01 / 0.02 / 0.1` |
| signal | dynamic OPD, dynamic NLL retention, GRPO frontier |

Interpretation rule:

- This is a parameterization diagnostic, not a result-shaped hyperparameter search.
- If `parameter` still moves mainly along memory-heavy directions, the main bottleneck is confirmed to be training signal / recoverable sample distribution rather than common+residual coupling.
- If it separates code/memory/tool by module while keeping formal metrics, then direct 588 coefficients are a viable paper setting.

13:28 stop decision:

- Rollout-1 completed in `388.6s`.
- Rollout-1 reward mean: code `0.4711`, memory `0.3984`, tool `0.3998`; overall `0.4231`.
- Rollout-1 all-fail rows: code `7`, memory `8`, tool `6`.
- Dynamic OPD selected `17` all-fail/expert-positive rows: code `4`, memory `6`, tool `7`.
- Iter-1 update completed and wrote `gate_updates.summary.json`.
- Iter-1 gate stats:
  - tool: mean `0.333296`, std `0.000136`, min `0.332766`, max `0.333753`.
  - memory: mean `0.333747`, std `0.000564`, min `0.330369`, max `0.335478`.
  - code: mean `0.333347`, std `0.000037`, min `0.333294`, max `0.333823`.
- Iter-1 optimizer stats: `grad_norm_max=1.18779`, `gate_delta_max=0.002964`, `clip_frac_mean=0.0`.
- Rollout-2 completed, but reward declined: code `0.3545`, memory `0.4062`, tool `0.4009`; overall `0.3872`.
- Rollout-2 dynamic OPD again selected `17` rows: code `3`, memory `7`, tool `7`.
- Iter-2 update started and was actively computing, but was stopped before summary because the diagnostic already showed high cost plus extremely weak first-step coefficient movement.
- Stop command: `tmux kill-session -t opvec_dynscale_K_parameter_direct_signal_20260515_i8_v1`.
- GPU 0/1 were released after stop; the frontend monitor remained alive.

Interpretation:

- Direct 588-parameter tuning does create nonzero local variation, but the first-step scale is too small to justify continuing this run under the current signal.
- The only visibly larger movement is memory, while code/tool remain essentially pinned to `1/3`; this matches the broader 2026-05-15 diagnosis that the training signal is memory-heavy rather than that common+residual coupling alone is the bottleneck.
- Because rollout-2 reward fell, especially code, this run was not a promising candidate for formal Eval6. The useful result is diagnostic: under the current OPD/GRPO/retention construction, `parameter` is more expensive and does not immediately produce useful local specialization.
