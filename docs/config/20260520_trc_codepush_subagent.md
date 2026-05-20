# 20260520 TRC Code-Push Subagent Recommendations

## Context Read

Evidence checked:

- R8/R10/R11/R12 config docs and R4/R5/R7/R8/R10 eval docs.
- TRC run summaries under `/tmp/shared-storage/OnPolicy/runs/trc`.
- R12D RF-only tag-quota bank:
  `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl`.

Main constraints:

- Do not change reward/main TRC objective.
- Keep Tool/Memory protection first: quick gate remains Tool mean `>=0.79`, Memory F1 `>=0.76`.
- Avoid repeating known weak moves: code loss multiplier push, Tool multiplier `1.5`, response topK128 as a default, response384 as a default.

## Readout

- R8D RF-only + code-block384 had the best Round8 LiveBench start (`0.3730`) but weak LiveCodeBench (`0.2676`).
- R8A-e08 RF-only + response256 had the best Round8 mean Acc (`0.3218`) and better LiveCodeBench (`0.2842`), so early-stop is real.
- R10D tag-quota + response256 + Memory `1.8` repaired R10B Memory (`0.7572 -> 0.7679`) while preserving Tool (`0.7944`), and started Code better than R10A (`0.3672` vs `0.3477` LiveBench).
- R10C tag-quota + code-block384 failed Tool/Memory (`Tool=0.7788`, Memory `0.7570`), but R8D code-block on RF-only passed. This makes RF-only purity the likely condition for trying code-block again.
- R11G/H training finished cleanly. R11G response256 has final gate means code/memory/tool about `1.2405/0.9951/1.2276`; R11H code-block384 has `1.2401/0.9962/1.2063`. The code-block run has lower tool gate and should be treated as higher Tool-risk.
- R12D removes DeepSeek fallback entirely and keeps stable late3 Tool/Memory. It has 32 RF-only Code rows, but only 29 unique Code prompts, with 3 duplicate prompts for graph/DP/greedy quota fill.

## Recommended Batch

Run order if GPUs are scarce:

1. `R12E` response256 + Memory `1.8`: safest main RF-only tag-quota test.
2. `R12F` code-block384 + Memory `1.8`: highest chance to reproduce R8D LiveBench behavior.
3. `R12G` response320 + Memory `1.8`: optional topK resolution test; start only if R11F is not clearly bad or if spare pair exists.

All three are 2-GPU TRC training jobs and should finish well under 5h each. Prior runs took about 19-20 minutes for 12 epochs plus bake time; quick gate/formal Code eval are separate scheduling decisions.

## R12E: RF-only Tag-Quota Response256 Mem18

Purpose: isolate whether R10D's best response256 behavior improves when Code source is RF-only while keeping stable Tool/Memory rows.

Expected:

- Tool/Memory should look closest to R11G/R10D and likely pass quick gate.
- Code may improve LiveCodeBench more than code-block variants because response span carries reasoning/repair tokens.

Risk:

- R12D has 3 duplicate Code prompts; if unique prompt coverage matters more than RF purity, Code may not beat R11G/R10D.

GPU need: 2 GPUs, normal 7B bake/train memory. Use one free pair.

```bash
GPU_PAIR=0,1
tmux new -d -s train_r12e_rfonly_tag_response256_mem18_e12_20260520 "cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && env \
EXP_ID=trc_r12e_rfonly_tag_response256_mem18_e12_20260520 \
GPU_LIST=$GPU_PAIR \
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/trc/trc_r12e_rfonly_tag_response256_mem18_e12_20260520 \
CONFIG=configs/gated_grpo_layer28_wide.yaml \
MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
EPOCHS=12 LR=0.02 BETA_BASE=0.05 GAMMA_GATE=0.005 \
COEFFICIENT_FLOOR=0.95 COEFFICIENT_FLOOR_WEIGHT=0.1 \
TASK_EXPERT_COEFFICIENT_FLOOR=1.0 TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0 \
TASK_HIDDEN_LAYERS=code=4,8,12,16,20,24,28 \
TASK_TOPK_TOKENS=code=256 \
TASK_DIRECTIONAL_PROJECTION_FLOOR=code=0.95 \
TASK_DIRECTIONAL_PROJECTION_WEIGHT=code=0.25 \
TASK_RESPONSE_SPAN_MODE='tool=tool-call code=response memory=response' \
TASK_LOSS_MULTIPLIER='code=1.0 memory=1.8 tool=1.2' \
TRAJECTORY_TURN_LOSS_TASKS=memory GRADIENT_CHECKPOINTING=0 \
bash skill/command/run_20260519_trc_round_train_one.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/trc/trc_r12e_rfonly_tag_response256_mem18_e12_20260520.launch.log"
```

Early-stop bakes after training:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
ROOT=/tmp/shared-storage/OnPolicy
EXP_ID=trc_r12e_rfonly_tag_response256_mem18_e12_20260520
for E in 08 10 12; do
  $PY scripts/eval/opvec_bake_checkpoint.py \
    --config configs/gated_grpo_layer28_wide.yaml \
    --mode-manifest $ROOT/modes/opvec4/mode_manifest.json \
    --gate-checkpoint $ROOT/runs/trc/$EXP_ID/epoch_0${E}.gates.json \
    --output $ROOT/checkpoints/${EXP_ID}_e${E}-selected
done
```

## R12F: RF-only Tag-Quota Codeblock384 Mem18

Purpose: direct R12 counterpart of R8D/R11H. This is the most direct test of whether R8D's LiveBench gain was RF-only plus code-block span.

Expected:

- Best chance among the batch to lift LiveBench/BoN if R8D's signal was real.
- Memory should be protected by stable late3 rows and Memory `1.8`.

Risk:

- Tool risk is real: R10C code-block failed Tool, and R11H's final tool gate is lower than R11G. Do Tool quick gate before any Code eval.
- May repeat R8D's pattern: stronger LiveBench but weak LiveCodeBench.

GPU need: 2 GPUs.

```bash
GPU_PAIR=2,3
tmux new -d -s train_r12f_rfonly_tag_codeblock384_mem18_e12_20260520 "cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && env \
EXP_ID=trc_r12f_rfonly_tag_codeblock384_mem18_e12_20260520 \
GPU_LIST=$GPU_PAIR \
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/trc/trc_r12f_rfonly_tag_codeblock384_mem18_e12_20260520 \
CONFIG=configs/gated_grpo_layer28_wide.yaml \
MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
EPOCHS=12 LR=0.02 BETA_BASE=0.05 GAMMA_GATE=0.005 \
COEFFICIENT_FLOOR=0.95 COEFFICIENT_FLOOR_WEIGHT=0.1 \
TASK_EXPERT_COEFFICIENT_FLOOR=1.0 TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0 \
TASK_HIDDEN_LAYERS=code=4,8,12,16,20,24,28 \
TASK_TOPK_TOKENS=code=384 \
TASK_DIRECTIONAL_PROJECTION_FLOOR=code=0.95 \
TASK_DIRECTIONAL_PROJECTION_WEIGHT=code=0.25 \
TASK_RESPONSE_SPAN_MODE='tool=tool-call code=code-block memory=response' \
TASK_LOSS_MULTIPLIER='code=1.0 memory=1.8 tool=1.2' \
TRAJECTORY_TURN_LOSS_TASKS=memory GRADIENT_CHECKPOINTING=0 \
bash skill/command/run_20260519_trc_round_train_one.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/trc/trc_r12f_rfonly_tag_codeblock384_mem18_e12_20260520.launch.log"
```

Early-stop bakes:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
ROOT=/tmp/shared-storage/OnPolicy
EXP_ID=trc_r12f_rfonly_tag_codeblock384_mem18_e12_20260520
for E in 08 10 12; do
  $PY scripts/eval/opvec_bake_checkpoint.py \
    --config configs/gated_grpo_layer28_wide.yaml \
    --mode-manifest $ROOT/modes/opvec4/mode_manifest.json \
    --gate-checkpoint $ROOT/runs/trc/$EXP_ID/epoch_0${E}.gates.json \
    --output $ROOT/checkpoints/${EXP_ID}_e${E}-selected
done
```

## R12G: RF-only Tag-Quota Response320 Mem18

Purpose: topK resolution test between response256 and too-wide response384/code-block. This is lower priority than R12E/F because R11F is already testing response320 on the R10 tag bank.

Expected:

- If R11F response320 is neutral or promising, R12G checks whether RF-only purity plus slightly broader response span adds Code signal without moving to full response384.
- Should be safer than topK128 for Tool because R9 topK128 failed Tool and R10 topK128 only recovered after tag-quota.

Risk:

- If R11F misses quick gate or Code, skip this run; it is likely redundant.
- Broader response can add explanation noise; R5D response384 was weak on Code.

GPU need: 2 GPUs. Run only after R12E/F unless spare pair is idle.

```bash
GPU_PAIR=4,5
tmux new -d -s train_r12g_rfonly_tag_response320_mem18_e12_20260520 "cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym && env \
EXP_ID=trc_r12g_rfonly_tag_response320_mem18_e12_20260520 \
GPU_LIST=$GPU_PAIR \
CALIB=/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl \
RUN_DIR=/tmp/shared-storage/OnPolicy/runs/trc/trc_r12g_rfonly_tag_response320_mem18_e12_20260520 \
CONFIG=configs/gated_grpo_layer28_wide.yaml \
MODE=/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \
EPOCHS=12 LR=0.02 BETA_BASE=0.05 GAMMA_GATE=0.005 \
COEFFICIENT_FLOOR=0.95 COEFFICIENT_FLOOR_WEIGHT=0.1 \
TASK_EXPERT_COEFFICIENT_FLOOR=1.0 TASK_EXPERT_COEFFICIENT_FLOOR_WEIGHT=50.0 \
TASK_HIDDEN_LAYERS=code=4,8,12,16,20,24,28 \
TASK_TOPK_TOKENS=code=320 \
TASK_DIRECTIONAL_PROJECTION_FLOOR=code=0.95 \
TASK_DIRECTIONAL_PROJECTION_WEIGHT=code=0.25 \
TASK_RESPONSE_SPAN_MODE='tool=tool-call code=response memory=response' \
TASK_LOSS_MULTIPLIER='code=1.0 memory=1.8 tool=1.2' \
TRAJECTORY_TURN_LOSS_TASKS=memory GRADIENT_CHECKPOINTING=0 \
bash skill/command/run_20260519_trc_round_train_one.sh 2>&1 | tee /tmp/shared-storage/OnPolicy/runs/trc/trc_r12g_rfonly_tag_response320_mem18_e12_20260520.launch.log"
```

Early-stop bakes:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym
PY=/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python
ROOT=/tmp/shared-storage/OnPolicy
EXP_ID=trc_r12g_rfonly_tag_response320_mem18_e12_20260520
for E in 08 10 12; do
  $PY scripts/eval/opvec_bake_checkpoint.py \
    --config configs/gated_grpo_layer28_wide.yaml \
    --mode-manifest $ROOT/modes/opvec4/mode_manifest.json \
    --gate-checkpoint $ROOT/runs/trc/$EXP_ID/epoch_0${E}.gates.json \
    --output $ROOT/checkpoints/${EXP_ID}_e${E}-selected
done
```

## Eval Triage

After each training run:

1. First quick gate only on `e12` and `e08` for R12E/R12F. Add `e10` only if one of them is borderline.
2. Promote to Code only if Tool mean `>=0.79` and Memory F1 `>=0.76`.
3. If R12F e12 fails Tool but e08 passes, prioritize R12F-e08 for Code because R8A-e08 already showed early stopping can improve mean Code.
4. Do not launch Code formal eval for R12G until R11F has at least Tool/Memory pass or an explicit Code reason to keep topK320 alive.

Useful quick eval wrapper, with GPU choices controlled by the main agent:

```bash
RUN_TOOL=1 RUN_MEMORY=1 RUN_CODE=0 \
TOOL_GPU=0 TOOL_PORT=8001 MEMORY_GPU_IDS=1 MEMORY_TP=1 \
RUN_ID=r12e_e08_tm_20260520 \
bash skill/command/run_full_eval_suite.sh \
  /tmp/shared-storage/OnPolicy/checkpoints/trc_r12e_rfonly_tag_response256_mem18_e12_20260520_e08-selected \
  trc_r12e_rfonly_tag_response256_mem18_e12_20260520_e08-selected
```
