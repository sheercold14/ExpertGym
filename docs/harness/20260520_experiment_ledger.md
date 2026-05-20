# 2026-05-20 ExpertGym 自动实验 Ledger

## 目标

明早前至少完成 `35` 个有记录的尝试，其中不要求全部是完整评测；每个 attempt 必须能回答一个具体假设，并给出 `promote / reject / retain` 决策。

## 三层定义

- `probe`: 低成本训练/数据/诊断尝试，通常 4-10 epoch 或只做数据统计；失败只留日志，不保留 baked checkpoint。
- `train`: 通过 probe 的正式训练，必须有 config 文档、run.env、metrics、selected gates。
- `eval`: 只对候选 checkpoint 做 Tool/Memory 快评；两者达标后才测 Code。

## 当前保留阈值

- Tool BFCL mean `>= 0.79`
- Memory mean F1 `>= 0.76`
- Code 只对 Tool/Memory 过线候选跑正式评测。

## Attempt 表

| ID | layer | run_id / artifact | axis | status | hypothesis | decision |
|---|---|---|---|---|---|---|
| A001 | probe | `build_20260520_trc_memorytraj_calibrations.sh` | memory/data | done | memory final-answer proxy 太短，需要 turn-level trajectory | promote |
| A002 | probe | `mtr_full_toolaug_code_rf` | memory/data | done | full trajectory 覆盖完整 MemAgent span | queued train |
| A003 | probe | `mtr_uniform4_toolaug_code_rf` | memory/data | done | 4 update + final 平衡覆盖率与速度 | promoted |
| A004 | probe | `mtr_late3_toolaug_code_rf` | memory/data | done | late updates 可能是最高信号 span | promoted |
| A005 | train | `trc_r3a_mtr_u4_main_20260520` | mixed | failed | uniform4 trajectory 修复 memory，同时保持 tool | OOM at 4096 seq; deleted empty run |
| A006 | train | `trc_r3b_mtr_late3_20260520` | memory | failed | late3 可能比 uniform4 更稳 | OOM/launch failed at 4096 seq; deleted empty run |
| A007 | train | `trc_r3c_mtr_u4_memlayers_20260520` | memory/code | failed | memory 更宽层/topk 是否提升 F1 | OOM/launch failed at 4096 seq; deleted empty run |
| A008 | eval | `trc_r2d_tm_20260520` | tool/memory | done | R2D Tool 强但 memory proxy 是否过线 | reject Code: memory F1 0.7403 |
| A009 | eval | `trc_r2e_tm_20260520` | tool/memory | done | R2E 是否比 R2D 更均衡 | reject Code: memory F1 0.7530 |
| A010 | probe | `debug_r3_foreground` | infra | done | 诊断 R3 启动失败原因 | 单卡可见 GPU0 到 79GB；需要 balanced device_map + 更短 seq/topk |
| A011 | train | `trc_r3a2_mtr_u4_safe_20260520` | infra/memory | failed | balanced+seq2048 是否足够 | merged forward 仍 OOM；需要 gradient checkpointing |
| A012 | train | `trc_r3b2_mtr_late3_safe_20260520` | infra/memory | killed | late3 safe | 同批次取消，避免重复 OOM |
| A013 | train | `trc_r3a3_mtr_u4_ckpt_1536_20260520` | mixed | killed | gradient checkpointing + uniform4 trajectory | memory trajectory span works, but memory residual too tiny; memory gate 1.00 -> 0.96 by epoch2 |
| A014 | train | `trc_r3b3_mtr_late3_ckpt_1536_20260520` | memory | killed | gradient checkpointing + late3 trajectory | same issue; memory loss too weak despite 4-turn trajectory |
| A015 | probe | `trc_p3u4_8x4_seq1024_20260520` | memory/infra | rejected | 8x3 rows, seq1024 是否足以给 memory 方向 | memory gate 1.00 -> 0.92；deleted baked checkpoint |
| A016 | probe | `trc_p3u4_8x4_toolprotect_20260520` | tool/memory | rejected | 小样本 tool multiplier 提升是否能保护 Tool | tool 更稳但 memory 仍 1.00 -> 0.92；deleted baked checkpoint |
| A017 | probe | `trc_p3u4_8x4_memfloor105_20260520` | memory | rejected | memory projection floor=1.05 是否阻止 memory gate 下降 | residual 增至 0.019 但 memory gate 仍 1.00 -> 0.92；deleted baked checkpoint |
| A018 | probe | `trc_p3u4_8x4_memfloor110_20260520` | memory | rejected | memory projection floor=1.10 是否给 memory 足够幅度动力 | residual 增至 0.035 但 memory gate 仍 1.00 -> 0.92；deleted baked checkpoint |
| A019 | probe | `trc_p3u4_8x4_memscale50_20260520` | memory | rejected | 只放大 memory loss 是否足以推回 memory gate | memory scale=50 仍到 0.9368；deleted baked checkpoint |
| A020 | probe | `trc_p3u4_8x4_codeoff_memtool_20260520` | conflict | rejected | code loss 关闭后 memory 是否仍下降 | code off 后 memory 仍到 0.9292，tool 上升；deleted baked checkpoint |
| A021 | probe | `trc_p3u4_8x4_coefffloor1w5_20260520` | coefficient-retention | rejected | coefficient floor=1.0 是否阻止专家系数被压低 | memory epoch2 仍 0.9616；deleted baked checkpoint |
| A022 | probe | `trc_p3u4_8x4_anchor1_20260520` | coefficient-retention | rejected | gamma anchor=1.0 是否稳定 init1 能力附近搜索 | memory epoch2 仍 0.9601；deleted baked checkpoint |
| A023 | probe | `trc_p3u4_8x4_coefffloor1w20_20260520` | coefficient-retention | partial | 更强 floor weight=20 是否能稳定所有专家系数 | memory epoch4 0.9732，当前最好但仍低于 1 |
| A024 | probe | `trc_p3u4_8x4_anchor5_20260520` | coefficient-retention | rejected | 更强 anchor=5 是否稳定 init1 附近搜索 | memory epoch4 0.9331；deleted baked checkpoint |
| A025 | probe | `trc_p3u4_8x4_taskfloor1w5_20260520` | task-aware-retention | rejected | 去掉全系数平均稀释后，task floor w5 是否保住 memory | memory epoch4 0.9375；deleted baked checkpoint |
| A026 | probe | `trc_p3u4_8x4_taskfloor1w20_20260520` | task-aware-retention | partial | task floor w20 是否足以保住 memory | memory epoch4 0.9818；证明 task-aware floor 有效，probe baked deleted |
| A027 | train | `trc_r3c_globalfloor20_u4_20260520` | coefficient-retention | done | global floor w20 全量 96 条是否达到 Tool/Memory 快评阈值 | selected e8: T=1.1519/M=0.9798/C=1.1601；低优先级 |
| A028 | train | `trc_r3d_globalfloor50_u4_20260520` | coefficient-retention | eval | global floor w50 是否比 w20 更稳地保住 memory | selected e8: T=1.1520/M=0.9996/C=1.1599；baked, Tool+Memory eval running |
| A029 | train | `trc_r3e_taskfloor20_u4_20260520` | task-aware-retention | done | task-aware floor w20 全量化是否比 global floor 更合理 | selected e8: T=1.1519/M=0.9884/C=1.1600；低优先级 |
| A030 | train | `trc_r3f_taskfloor50_u4_20260520` | task-aware-retention | eval | task-aware floor w50 是否兼顾 memory 保护与任务归因 | selected e8: T=1.1520/M=1.0019/C=1.1599；baked, Tool+Memory eval running |
| A031 | train | `trc_r3g_full_taskfloor50_20260520` | memory/trajectory | failed | full memory trajectory + task floor50 是否优于 uniform4 | launch exited before metrics; replaced by logged small probe |
| A032 | train | `trc_r3h_full_globalfloor50_20260520` | memory/trajectory | failed | full memory trajectory + global floor50 是否更稳 | launch exited before metrics; replaced by logged small probe |
| A033 | train | `trc_r3i_u4_codefull_taskfloor50_20260520` | code/span | eval | code full response span 是否比 code-block span 更贴近评测能力 | selected e8: T=1.1567/M=1.0018/C=1.1599；Tool+Memory eval running |
| A034 | probe | `trc_r3g2_full_probe_taskfloor50_20260520` | memory/trajectory | rejected | full trajectory 小样本能否避免 OOM | OOM even with 8 rows/task at seq1536; no baked checkpoint |
| A035 | train | `trc_r3j_late3_taskfloor50_20260520` | memory/trajectory | eval | late3 trajectory + task floor50 是否接近 uniform4 | selected e8: T=1.1523/M=1.0017/C=1.1599；Tool+Memory eval running |
| A036 | eval | `eval_r3d_tm_fix_20260520` | tool/memory | done | R3D 是否达进入 Code 的门槛 | Tool mean 0.7944；Memory mean F1 0.7636；promote to Code |
| A037 | eval | `eval_r3f_tm_fix_20260520` | tool/memory | running | R3F task-aware floor 是否比 R3D 更稳 | Tool mean 0.7775 below threshold; Memory still running |
| A038 | eval | `eval_r3d_code_20260520` | code | running | R3D 是否在 Code 上保持/提升 | pending |
| A039 | train | `trc_r3k_u4_codefull_globalfloor50_20260520` | code/span | eval | code full response span + global floor50 是否比 task-aware 更保 Tool | selected e8: T=1.1567/M=0.9997/C=1.1599；Tool+Memory eval running |

## 下一批候选 Attempt

| ID | layer | run_id | axis | trigger | hypothesis |
|---|---|---|---|---|---|
| A040 | eval | `eval_r3i_tm_20260520` | eval | R3I baked | If Tool/Memory pass, send Code; tests code-full span |
| A041 | eval | `eval_r3j_tm_20260520` | eval | R3J baked | Tool mean 0.7944; Memory mean F1 0.7673; promoted to Code |
| A042 | train | `r3_next_toolprotect` | tool | if R3I/R3K Tool drops | strengthen tool-call span / retention only |
| A043 | train | `r3_next_memory_budgeted_full` | memory | if memory still marginal | use sampled full turns, not all turns, to avoid OOM |
| A044 | train | `trc_r3l_u4_codeblock384_globalfloor50_20260520` | code/span | eval | Keep code-block span but increase code topk to 384 | selected e8: T=1.1523/M=0.9995/C=1.1599；Tool+Memory eval running |
| A045 | train | `trc_r3m_u4_codefull_toolprotect_globalfloor50_20260520` | code/tool | rejected | Full code response with tool multiplier=2.0, code multiplier=0.8 | Tool mean 0.0；stopped eval and deleted baked checkpoint |
| A046 | eval | `eval_r3d_code_rerun_20260520` | code | done | Rerun R3D Code on isolated GPUs 4/7 after resource-collision OOM | LiveBench Acc 0.3867 / BoN 0.4766; LiveCodeBench Acc 0.2710 / BoN 0.3542; mean Acc 0.3289 |
| A047 | eval | `eval_r3k_tm_20260520` | eval | R3K baked | first eval failed because checkpoint shard 4 was missing; checkpoint deleted and rebaked |
| A048 | eval | `eval_r3j_code_20260520` | code | done | R3J late3 candidate passed Tool/Memory | LiveBench Acc 0.3496 / BoN 0.4375; LiveCodeBench Acc 0.2779 / BoN 0.3562; mean Acc 0.3137 |
| A049 | eval | `eval_r3l_tm_20260520` | eval | R3L baked | Tool mean 0.7944; Memory mean F1 0.7494; rejected and deleted baked checkpoint |
| A050 | eval | `eval_r3k_tm_rebake_20260520` | eval | R3K rebaked | Tool mean 0.7944; Memory mean F1 0.7715; queue Code after current CURE jobs |
| A051 | train | `trc_r3n_late3_codeblock384_globalfloor50_20260520` | memory/code | rejected | Combine R3J late3 memory with R3L code-block topk384 under global floor50 | Tool mean 0.0；stopped eval and deleted baked checkpoint |
| A052 | train | `trc_r3o_late3_codeblock384_taskfloor50_20260520` | memory/code | rejected | Same combined span budget under task-aware floor50 | Tool mean 0.7944; Memory mean F1 0.7585 below 0.76; baked checkpoint deleted |
| A053 | train | `trc_r4a_late3_taskfloor50_e12_20260520` | code/epoch | running | Extend the R3O late3 + code-block384 + task-floor50 setting from 8 to 12 epochs | check whether code/tool gains continue without memory collapse |
| A054 | train | `trc_r4b_late3_codeboost_taskfloor50_e12_20260520` | code/loss-scale | rejected | Same as R4A but code loss multiplier 1.4 | Tool mean 0.0; code loss boost breaks tool-call behavior despite similar gate means; baked checkpoint deleted |
| A055 | train | `trc_r4c_u4_codefull_globalfloor50_e12_20260520` | code/span | done | Extend R3K code-full + global-floor50 from 8 to 12 epochs | selected e12: code 1.2398 / memory 0.9965 / tool 1.2280; Tool/Memory eval as A061 |
| A056 | eval | `eval_r3k_code_20260520` | code | done | R3K has current best Tool/Memory; run CURE on GPU1/7 after R3D completed | LiveBench Acc 0.3652 / BoN 0.4531; LiveCodeBench Acc 0.2794 / BoN 0.3542; mean Acc 0.3223 |
| A057 | eval | `eval_r4a_tm_20260520` | tool/memory | infra-failed | R4A reached code 1.2395 / memory 1.0017 / tool 1.2225 | BFCL model_config corrupted by concurrent registration; rerun as A059 |
| A058 | eval | `eval_r4b_tm_20260520` | tool/memory | infra-failed | R4B reached code 1.2398 / memory 0.9967 / tool 1.2174 | BFCL model_config corrupted by concurrent registration; rerun as A060 |
| A059 | eval | `eval_r4a_tm_0400_20260520` | tool/memory | done | BFCL config repaired and model names pre-registered | Tool mean 0.7944; Memory mean F1 0.7638; promoted to Code |
| A060 | eval | `eval_r4b_tm_0400_20260520` | tool/memory | rejected | BFCL config repaired and model names pre-registered | Tool mean 0.0; stopped Memory eval and deleted baked checkpoint |
| A061 | eval | `eval_r4c_tm_0400_20260520` | tool/memory | rejected | R4C reached code 1.2398 / memory 0.9965 / tool 1.2280 | Tool mean 0.7892 below threshold; Memory 0.7612; baked checkpoint deleted |
| A062 | train | `trc_r4d_late3_codeproj_taskfloor50_e12_20260520` | code/projection | done | Same as R4A but code directional projection floor=0.95, weight=0.25 | selected e12: code 1.2393 / memory 0.9956 / tool 1.2131; Tool/Memory eval as A065 |
| A063 | train | `trc_r4e_u4_codeblock384_taskfloor50_e12_20260520` | code/memory | rejected | uniform4 + code-block384 + task-aware floor50 | Tool mean 0.7788; stopped Memory eval and deleted baked checkpoint |
| A064 | eval | `eval_r4a_code_20260520` | code | done | R4A is the first e12 stronger-gate candidate that passes Tool/Memory | LiveBench Acc 0.3613 / BoN 0.4609; LiveCodeBench Acc 0.2872 / BoN 0.3738; mean Acc 0.3243 |
| A065 | eval | `eval_r4d_tm_0420_20260520` | tool/memory | done | R4D code-projection candidate | Tool mean 0.8048; Memory mean F1 0.7669; promoted to Code |
| A066 | eval | `eval_r4e_tm_0420_20260520` | tool/memory | rejected | R4E uniform4+task-aware+codeblock384 candidate | Tool mean 0.7788; stopped Memory eval and deleted baked checkpoint |
| A067 | train | `trc_r4f_late3_memboost_taskfloor50_e12_20260520` | memory/loss-scale | rejected | R4A + memory loss multiplier 2.0, code scale unchanged | selected e12: code 1.2393 / memory 1.0058 / tool 1.2206; Tool collapsed in A070; baked checkpoint deleted |
| A068 | bake/eval | `trc_r4g_r4a_epoch10_earlystop_20260520` | early-stop | rejected | Bake R4A epoch10 gates: code 1.1997 / memory 0.9994 / tool 1.1886 | Tool mean 0.7892 below threshold in A071; baked checkpoint deleted |
| A069 | eval | `eval_r4d_code_20260520` | code | done | R4D has the best Tool/Memory combined score in Round4 | LiveBench Acc 0.3496 / BoN 0.4531; LiveCodeBench Acc 0.2657 / BoN 0.3718; mean Acc 0.3076 |
| A070 | eval | `eval_r4f_tm_0445_20260520` | tool/memory | rejected | R4F memory-boost candidate | Tool mean 0.0; stopped Memory eval and deleted baked checkpoint |
| A071 | eval | `eval_r4g_tm_0430b_20260520` | tool/memory | rejected | R4A epoch10 early-stop candidate | Tool mean 0.7892; stopped Memory eval and deleted baked checkpoint |
| A072 | data | `mtr_late3_toolv2_codev2` | calibration | done | Build v2 recovery-oriented 96-row TRC bank while keeping 32 rows/task | Tool 32 unique, Memory 32 unique late3 trajectories, Code 32 unique from ReasonFlux/DeepSeek/fallback |
| A073 | train | `trc_r5a_v2_late3_codeproj_taskfloor50_e12_20260520` | calibration/code | done | R4D objective unchanged, replace calibration with v2 SOTA/recovery bank | selected e12: code 1.2393 / memory 0.9952 / tool 1.2039; Tool+Memory eval as A077 |
| A074 | train | `trc_r5b_v2_late3_codeproj105_taskfloor50_e12_20260520` | code/projection | done | Same v2 calibration as R5A, but code directional projection floor 1.05 | selected e12: code 1.2394 / memory 0.9950 / tool 1.1971; queued in A077 |
| A075 | train | `trc_r5c_v2_late3_coderesponse_taskfloor50_e12_20260520` | code/span | done | Same v2 calibration as R5A, but code span=response and code topk=256 | selected e12: code 1.2397 / memory 0.9936 / tool 1.2174; queued in A077 |
| A076 | train | `trc_r5d_v2_late3_coderesponse384_taskfloor50_e12_20260520` | code/span | done | Same as R5C but code response topk=384 | selected e12: code 1.2396 / memory 0.9935 / tool 1.2140; queued in A077 |
| A077 | eval-queue | `eval_r5_tm_queue_20260520` | tool/memory | done | Sequential Tool+Memory quick eval for R5A-D after each baked checkpoint appears | R5A 0.8035/0.7638; R5B 0.7931/0.7677; R5C 0.7944/0.7690; R5D 0.7931/0.7634 |
| A078 | eval | `eval_r5a_code_20260520` | code | done | R5A passed Tool/Memory; run CURE on GPU0/2 | LiveBench Acc 0.3672 / BoN 0.4844; LiveCodeBench Acc 0.2715 / BoN 0.3777; mean Acc 0.3194 / BoN 0.4310 |
| A079 | eval | `eval_r5b_code_20260520` | code | done | R5B passed Tool/Memory; run CURE on GPU3/5 | LiveBench Acc 0.3672 / BoN 0.4609; LiveCodeBench Acc 0.2725 / BoN 0.3581; mean Acc 0.3198 / BoN 0.4095 |
| A080 | eval | `eval_r5c_code_20260520` | code | done | R5C passed Tool/Memory; run CURE on GPU1/7 | LiveBench Acc 0.3711 / BoN 0.4453; LiveCodeBench Acc 0.2803 / BoN 0.3659; mean Acc 0.3257 / BoN 0.4056 |
| A085 | eval | `eval_r5d_code_20260520` | code | done | R5D passed Tool/Memory; run CURE on GPU1/7 after R5C generation entered unit-test phase | LiveBench Acc 0.3477 / BoN 0.4219; LiveCodeBench Acc 0.2608 / BoN 0.3620; mean Acc 0.3042 |
| A081 | data | `cure_success32` | diagnostic/calibration | done | Build eval-leak diagnostic TRC bank with 32 R3D LiveBench hidden-test-passing Code trajectories | not paper-main; tests whether CURE-aligned code trajectories are learnable |
| A082 | data | `cure_success16lb16lcb` | diagnostic/calibration | done | Build eval-leak diagnostic TRC bank with 16 LiveBench + 16 LiveCodeBench hidden-test-passing Code trajectories | not paper-main; balanced CURE diagnostic |
| A083 | train | `trc_r6a_curediag_lb_codeblock_e8_20260520` | diagnostic/code | rejected | CURE diagnostic: LiveBench-passing code rows, code-block span | selected e8: code 1.1499 / memory 0.9983 / tool 1.1506; Tool mean 0.7800 below threshold |
| A084 | train | `trc_r6b_curediag_bal_response_e8_20260520` | diagnostic/code | done | CURE diagnostic: balanced LiveBench/LiveCodeBench passing code rows, response span | selected e8: code 1.1569 / memory 0.9968 / tool 1.1566; Tool/Memory eval as A087 |
| A086 | eval-queue | `eval_r6_tm_queue_20260520` | diagnostic/tool-memory | stopped | Sequential Tool+Memory quick eval for R6A/R6B after bake | stopped after R6A Tool mean 0.7800; restarted R6B-only as A087 to save time |
| A087 | eval | `eval_r6b_tm_20260520` | diagnostic/tool-memory | done | Tool+Memory quick eval for R6B diagnostic candidate | Tool mean 0.7944; Memory F1 0.7604; promoted to Code |
| A088 | eval | `eval_r6b_code_20260520` | diagnostic/code | infra-failed | R6B passed Tool/Memory; run CURE as diagnostic upper-bound | OOM from overlapping R7A on GPUs 0,2; session stopped and must be rerun on free GPUs |
| A089 | train | `trc_r7a_v2_response_e16_20260520` | code/epoch | done | R5C setting extended to 16 epochs | selected e16: code 1.3192 / memory 0.9893 / tool 1.2691; Tool/Memory eval as A091 |
| A090 | train | `trc_r7b_v2_response_proj105_e12_20260520` | code/projection | done | R5C setting with code projection floor 1.05 | selected e12: code 1.2397 / memory 0.9933 / tool 1.2150; Tool/Memory eval as A091 |
| A091 | eval-queue | `eval_r7_tm_queue_20260520` | tool/memory | done | Sequential Tool+Memory quick eval for R7B then R7A | R7B Tool 0.8048 / Memory 0.7821 promoted to Code; R7A Tool 0.7944 / Memory 0.7553 rejected |
| A092 | eval | `eval_r6b_code_rerun_20260520` | diagnostic/code | running | Clean rerun of R6B diagnostic Code after the first attempt OOMed | running on GPUs 3,5 |
| A093 | data | `round8_codep0_rf_only_late3` | calibration/code | done | Build non-leak 96-row TRC bank with CodeP0-v3 ReasonFlux successful trajectories only | Code 32 rows, 29 unique prompts + 3 duplicate successful trajectories; Tool/Memory same as R5 |
| A094 | data | `round8_codep0_rf_then_ds_late3` | calibration/code | done | Build non-leak 96-row TRC bank with ReasonFlux first and DeepSeek fallback | Code 32 unique prompts; tests coverage vs expert-vector purity |
| A095 | train | `trc_r8a_codep0_rf_response_e12_20260520` | calibration/code | done | R5C objective with CodeP0-v3 ReasonFlux-only calibration | selected e12: code 1.2406 / memory 0.9928 / tool 1.2303; Tool/Memory eval as A099 |
| A096 | train | `trc_r8b_codep0_rfds_response_e12_20260520` | calibration/code | done | R5C objective with CodeP0-v3 RF+DeepSeek coverage calibration | selected e12: code 1.2405 / memory 0.9933 / tool 1.2296; Tool/Memory eval as A099 |
| A097 | data | `round8_codep0_rf_rolequota_late3` | calibration/code | done | Build role-balanced CodeP0-v3 calibration with ReasonFlux-only successful trajectories | Code roles frontier=12 / partial_edge=10 / generation=8 / stable=2; 28 unique prompts |
| A098 | eval | `eval_r7b_code_20260520` | code | done | R7B passed Tool/Memory with strongest R7 quick metrics | LiveBench Acc 0.3379 / BoN 0.4141; LiveCodeBench Acc 0.2642 / BoN 0.3483; mean Acc 0.3010, rejected for Code |
| A099 | eval-queue | `eval_r8_tm_queue_20260520` | tool/memory | done | Sequential Tool+Memory quick eval for R8A/R8B as soon as baked checkpoints appear | R8A Tool 0.8035 / Memory 0.7563 rejected at e12; R8B Tool 0.7944 / Memory 0.7687 promoted to Code |
| A100 | train | `trc_r8c_codep0_rf_relmse_e10_20260520` | objective/code | rejected | CodeP0-v3 RF-only calibration with relative-MSE residual objective | stopped at epoch2: memory gate 0.9618 and loss scale too large; no bake |
| A101 | train/eval | `trc_r8e_codep0_rolequota_response_e12_20260520` | calibration/code | eval | CodeP0-v3 role-quota calibration with directional response objective | selected e12: code 1.2405 / memory 0.9929 / tool 1.2283; Tool+Memory eval running on GPU4 |
| A102 | train | `trc_r8d_codep0_rf_codeblock_e12_20260520` | span/code | running | CodeP0-v3 RF-only calibration with Code code-block span/topK384 | reached e10: code 1.2002 / memory 0.9953 / tool 1.1814; still training on GPUs 0,1 |
| A103 | train | `trc_r9a_codep0_rf_response128_e12_20260520` | span/code | eval | CodeP0-v3 RF-only calibration with focused Code response topK128 | selected e12: code 1.2410 / memory 0.9935 / tool 1.2307; Tool/Memory eval pending |
| A104 | eval | `eval_r8e_tm_20260520` | tool/memory | rejected | Quick-gate R8E role-quota candidate before expensive Code eval | Tool 0.7944 / Memory 0.7580; below Memory gate, do not run Code |
| A105 | train | `trc_r9b_rolequota_response128_e12_20260520` | span/code | eval | R8E role-quota calibration with focused Code response topK128 | selected e12: code 1.2409 / memory 0.9934 / tool 1.2286; Tool/Memory eval pending |
| A106 | eval | `eval_r8d_tm_20260520` | tool/memory | done | Quick-gate R8D code-block-span candidate | Tool 0.7944 / Memory 0.7668; promoted to Code |
| A107 | bake/eval | `eval_r8a_early_memory_20260520` | early-stop/memory | done | Bake R8A e08/e10 and run Memory-only before Tool/Code | R8A-e08 Memory 0.7716; duplicate e10 queue stopped before wasting more GPU |
| A108 | data | `round10_codep0_tag_quota_default_late3` | calibration/code | done | Build non-leak CodeP0-v3 tag-quota 96-row TRC bank | Code 32 rows / 32 unique prompts; primary tags string 11, math 7, graph 5, DP 4, greedy 4, format 1; ready for R10A/R10B |
| A109 | eval | `eval_r8b_code_20260520` | code | done | R8B passed Tool/Memory; run CURE Code to test RF+DeepSeek CodeP0 transfer | LiveBench Acc 0.3477 / BoN 0.4141; LiveCodeBench Acc 0.2642 / BoN 0.3542; mean Acc 0.3059 / BoN 0.3841 |
| A110 | train | `trc_r10a_codep0_tag_response128_e12_20260520` | calibration/code | running | Tag-quota CodeP0-v3 calibration with focused response topK128 | started on GPUs 2,3; tests whether algorithm-tag coverage improves Code transfer without changing loss |
| A111 | eval | `eval_r9a_tm_20260520` | tool/memory | rejected | Quick-gate R9A focused response topK128 candidate | Tool mean 0.7788 below gate; stopped before Memory/Code |
| A112 | eval | `eval_r8a_e10_memory_20260520` | early-stop/memory | done | Parallelize R8A e10 Memory-only while e08 runs on GPU1 | Memory mean F1 0.7602 (0.7720/0.7342/0.7780/0.7566), below e08 mean 0.7716; keep e08 as R8A representative and do not expand e10 |
| A113 | eval | `eval_r8a_e08_tool_20260520` | early-stop/tool | done | R8A-e08 passed Memory; run Tool before possible Code | Tool 0.7931; passes quick gate with Memory 0.7716 |
| A114 | eval | `eval_r8d_code_20260520` | code | done | R8D passed Tool/Memory; run CURE Code | LiveBench Acc 0.3730 / BoN 0.4766; LiveCodeBench Acc 0.2676 / BoN 0.3601; mean Acc 0.3203 / BoN 0.4183 |
| A115 | train | `trc_r10b_codep0_tag_response256_e12_20260520` | calibration/code | done | Tag-quota CodeP0-v3 calibration with broader response topK256 | selected e12: code 1.2404 / memory 0.9930 / tool 1.2136; quick gate as A121 |
| A116 | eval | `eval_r8a_e08_code_20260520` | code | done | R8A early-stop e08 passed Tool/Memory; run CURE Code | LiveBench Acc 0.3594 / BoN 0.4297; LiveCodeBench Acc 0.2842 / BoN 0.3640; mean Acc 0.3218 / BoN 0.3968 |
| A117 | eval | `eval_r9b_tm_20260520` | tool/memory | rejected | Quick-gate R9B role-quota + topK128 candidate | Tool mean 0.7788 below gate; stopped before Memory/Code. Confirms topK128 hurts Tool even with role-quota rows |
| A118 | eval | `eval_r10a_tm_20260520` | tool/memory | done | Quick-gate R10A tag-quota + response topK128 selected checkpoint | Tool 0.7944; Memory 0.7625 (0.7861/0.7847/0.7357/0.7436); promoted to Code despite weak long-context splits |
| A119 | train | `trc_r10c_codep0_tag_codeblock384_e12_20260520` | calibration/span | done | Tag-quota CodeP0-v3 calibration with Code code-block span and topK384 | selected e12: code 1.2400 / memory 0.9944 / tool 1.2016; quick gate as A124 |
| A120 | train | `trc_r10d_tag_response256_mem18_e12_20260520` | memory/loss-scale | done | R10B tag-quota response256 with Memory loss multiplier 1.8 instead of 1.6 | selected e12: code 1.2404 / memory 0.9947 / tool 1.2109; quick gate as A125 |
| A121 | eval | `eval_r10b_tm_20260520` | tool/memory | rejected | Quick-gate R10B tag-quota + response topK256 selected checkpoint | Tool 0.7944; Memory 0.7572 (0.7637/0.7354/0.7716/0.7581), below quick gate; do not run Code unless threshold is relaxed |
| A122 | train | `trc_r10e_tag_response256_tool15_e12_20260520` | tool/loss-scale | done | R10B tag-quota response256 with Tool loss multiplier 1.5 instead of 1.2 | selected e12: code 1.2404 / memory 0.9912 / tool 1.2239; quick gate as A127 |
| A123 | eval | `eval_r10a_code_20260520` | code | done | R10A passed Tool/Memory quick gate; run CURE Code | LiveBench Acc 0.3477 / BoN 0.4453; LiveCodeBench Acc 0.2715 / BoN 0.3581; mean Acc 0.3096 / BoN 0.4017 |
| A124 | eval | `eval_r10c_tm_20260520` | tool/memory | rejected | Quick-gate R10C tag-quota + code-block384 selected checkpoint | Tool 0.7788 (live_parallel drops to 0.75) and Memory 0.7570; reject before Code |
| A125 | eval | `eval_r10d_tm_20260520` | tool/memory | done | Quick-gate R10D tag-quota response256 + Memory multiplier 1.8 selected checkpoint | Tool 0.7944; Memory 0.7679 (0.7622/0.7477/0.7823/0.7794); promoted to Code, launch `code_20260520_1022` observed |
| A126 | bake/eval | `trc_r11a_r10b_tag_response256_e08_20260520` | early-stop | rejected | Bake R10B epoch 8 gate as low-cost early-stop candidate | Tool 0.7931 but Memory 0.7335 (0.7622/0.7337/0.7387/0.6995); reject by Memory |
| A127 | eval | `eval_r10e_tm_20260520` | tool/memory | rejected | Quick-gate R10E tag-quota response256 + Tool multiplier 1.5 selected checkpoint | Tool 0.7788 below gate (live_parallel 0.75); Memory 0.7737 (0.7814/0.7901/0.7642/0.7590); do not run Code |
| A128 | train | `trc_r11f_tag_response320_e12_20260520` | code/span | done | Tag-quota CodeP0-v3 calibration with Code response topK320 | selected e12: code 1.2403 / memory 0.9929 / tool 1.2123; quick gate as A137 |
| A129 | bake | `trc_r8d_codep0_rf_codeblock_e08_20260520-selected` | early-stop | baked | Bake R8D epoch 8 as lower-cost CodeP0 code-block early-stop candidate | checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8d_codep0_rf_codeblock_e08_20260520-selected`; source gate: `trc_r8d_codep0_rf_codeblock_e12_20260520/epoch_008.gates.json`; gate means C/M/T=1.1601/0.9974/1.1499; this worker did not launch quick gate because 2026-05-20 10:18 resource check had no safe free GPU pair; duplicate eval `eval_r11b_tm_20260520` on `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11b_r8d_codeblock_e08_20260520-selected` has Tool=0.7944 and Memory running; no Code recommendation until Memory passes |
| A130 | bake | `trc_r8d_codep0_rf_codeblock_e10_20260520-selected` | early-stop | baked | Bake R8D epoch 10 as early-stop between e08 and strong e12 LiveBench | checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8d_codep0_rf_codeblock_e10_20260520-selected`; source gate: `trc_r8d_codep0_rf_codeblock_e12_20260520/epoch_010.gates.json`; gate means C/M/T=1.2002/0.9953/1.1814; quick gate skipped because no safe free GPU pair; no Code recommendation yet |
| A131 | bake | `trc_r8b_codep0_rfds_response_e08_20260520-selected` | early-stop | baked | Bake R8B epoch 8 to compare against R8B e12 and R8A-e08 early stop | checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8b_codep0_rfds_response_e08_20260520-selected`; source gate: `trc_r8b_codep0_rfds_response_e12_20260520/epoch_008.gates.json`; gate means C/M/T=1.1601/0.9978/1.1573; quick gate skipped because no safe free GPU pair; no Code recommendation yet |
| A132 | eval | `eval_r11b_tm_20260520` | early-stop/code | done | Quick-gate R8D epoch 8 alias checkpoint as early-stop variant of the strongest LiveBench branch | Tool 0.7944; Memory 0.7715 (0.7755/0.7285/0.8139/0.7683); promote to Code |
| A133 | eval | `eval_r10d_code_20260520` | code | done | R10D passed Tool/Memory quick gate; run CURE Code | LiveBench Acc 0.3672 / BoN 0.4453; LiveCodeBench Acc 0.2652 / BoN 0.3601; mean Acc 0.3162 / BoN 0.4027 |
| A134 | cleanup | `trc_r11c_r8d_codeblock_e10_20260520` | early-stop/code | done | Removed duplicate R8D epoch 10 alias checkpoint | kept canonical worker-baked checkpoint `/tmp/shared-storage/OnPolicy/checkpoints/trc_r8d_codep0_rf_codeblock_e10_20260520-selected`; deleted duplicate 15GB alias |
| A135 | train | `trc_r11g_hybrid_response256_mem18_e12_20260520` | calibration/code | done | Hybrid calibration: 24 R10 tag-quota Code rows + 8 RF-only supplement, stable Tool/Memory, response topK256, Memory multiplier 1.8 | selected e12: code 1.2405 / memory 0.9951 / tool 1.2276; quick gate as A140 |
| A136 | eval | `eval_r11c_tm_20260520` | early-stop/code | rejected | Quick-gate canonical R8D epoch 10 checkpoint | Tool 0.7944 but Memory 0.7567 (0.7709/0.7444/0.7683/0.7433); do not run Code |
| A137 | eval | `eval_r11f_tm_20260520` | tool/memory | done | Quick-gate R11F tag-quota response topK320 selected checkpoint | Tool 0.8048; Memory 0.7726 (0.7780/0.7524/0.8030/0.7571); promoted to Code as A141 |
| A138 | train | `trc_r11h_hybrid_codeblock384_mem18_e12_20260520` | calibration/span | done | Hybrid calibration with Code code-block topK384 and Memory multiplier 1.8 | baked at 2026-05-20 10:56 CST; selected e12 C/M/T=1.2401/0.9962/1.2063; checkpoint `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11h_hybrid_codeblock384_mem18_e12_20260520-selected`; Tool/Memory quick gate not launched by this monitor |
| A139 | eval | `eval_r11b_code_20260520` | code | done | R11B/R8D-e08 passed Tool/Memory quick gate; run CURE Code | LiveBench Acc 0.3750 / BoN 0.5000; LiveCodeBench Acc 0.2794 / BoN 0.3620; mean Acc 0.3272 / BoN 0.4310 |
| A140 | eval | `eval_r11g_tm_20260520` | tool/memory | done | Quick-gate R11G hybrid response256 + Memory multiplier 1.8 selected checkpoint | Tool 0.7944; Memory mean F1 0.7600 (0.7616/0.7646/0.7559/0.7576); promoted to Code as A145 |
| A141 | eval | `eval_r11f_code_20260520` | code | done | R11F passed Tool/Memory quick gate; run CURE Code | LiveBench Acc 0.3555 / BoN 0.4688; LiveCodeBench Acc 0.2657 / BoN 0.3444; mean Acc 0.3106 / BoN 0.4066 |
| A142 | train | `trc_r12d_rfonly_tagquota_codeblock384_mem16_e12_20260520` | calibration/code | done | R12D RF-only tag-quota calibration, Code `code-block` topK384, Memory multiplier 1.6, Tool 1.2 | selected/baked at epoch 12; gate means C/M/T `1.2403/0.9945/1.2110`; quick gate as A148 |
| A143 | eval | `eval_r11h_tm_20260520` | tool/memory | done | Quick-gate R11H hybrid Code `code-block` topK384 + Memory multiplier 1.8 selected checkpoint | Tool 0.8048; Memory mean F1 0.7619 (0.7746/0.7575/0.7725/0.7429); promoted to Code as A147 |
| A144 | train | `trc_r12b_tag_codeblock384_mem18_e12_20260520` | calibration/span | running | R10 tag-quota calibration, Code `code-block` topK384, Memory multiplier 1.8, Tool 1.2 | latest epoch 8 gates C/M/T `1.1600/0.9991/1.1434`; not baked, no quick gate yet |
| A145 | eval | `eval_r11g_code_20260520` | code | done | R11G passed quick gate with Tool 0.7944 and Memory mean F1 0.7600 | LiveBench Acc 0.3516 / BoN 0.4297; LiveCodeBench Acc 0.2784 / BoN 0.3542; mean Acc 0.3150 / BoN 0.3919 |
| A146 | train | `trc_r12e_rfonly_tagquota_response256_mem18_e12_20260520` | calibration/span | running | R12D RF-only tag-quota calibration, Code `response` topK256, Memory multiplier 1.8, Tool 1.2 | launched 2026-05-20 11:15 CST on GPUs 0,7; direct contrast to R12D's RF-only `code-block384` |
| A147 | eval | `eval_r11h_code_20260520` | code | done | R11H passed quick gate with Tool 0.8048 and Memory mean F1 0.7619 | LiveBench Acc 0.3477 / BoN 0.4375; LiveCodeBench Acc 0.2652 / BoN 0.3640; mean Acc 0.3064 / BoN 0.4007 |
| A148 | eval | `eval_r12d_tm_20260520` | tool/memory | rejected | Quick-gate R12D RF-only tag-quota code-block384 selected checkpoint | Tool 0.7788 (0.7500/0.6250/0.8850/0.8550); Memory 0.7590 (0.7779/0.7329/0.7465/0.7787); reject before Code |
| A149 | data | `trc_round13_evalleak_code16` | hiddenstate/code-diagnostic | done | Build formal-code eval-leak diagnostic banks with ability-span marked Code trajectories | `rfmem_only`: Tool32/Memory32/Code10, no DeepSeek/R1; `all_with_r1`: Tool32/Memory32/Code23, includes DeepSeek/R1 mapped to reasoning |
| A150 | train | `trc_r13a_evalleak_rfmem_response512_e8_20260520` | hiddenstate/code-diagnostic | rejected/infra | No-R1 task-vector diagnostic with `MAX_SEQ_LENGTH=3072`, response 1024 | OOM during Memory trajectory-turn merged forward; no checkpoint baked; superseded by compact600 rerun A153 |
| A151 | train | `trc_r13b_evalleak_all_r1_response512_e8_20260520` | hiddenstate/code-diagnostic | rejected/infra | R1 control with `MAX_SEQ_LENGTH=3072`, response 1024 | OOM during Memory trajectory-turn merged forward; no checkpoint baked; superseded by compact600 rerun A154 |
| A152 | eval | `eval_r12b_tm_20260520` | tool/memory | done/pass | Quick-gate R12B mixed tag-quota code-block384 selected checkpoint | Tool `0.7944`, Memory `0.7675`; pass quick gate and launch Code as A155 |
| A153 | train | `trc_r13a_evalleak_rfmem_compact600_e8_20260520` | hiddenstate/code-diagnostic | done/baked | No-R1 task-vector diagnostic rerun: Code ability span uses 600-char reasoning context, seq1536/resp512 | auto-selected e4: C/M/T `1.0389/1.0177/1.0685`; e8 reaches C/M/T `1.0821/1.0332/1.1346` and is force-baked as A159 |
| A154 | train | `trc_r13b_evalleak_all_r1_compact600_e8_20260520` | hiddenstate/code-diagnostic | rejected/infra | R1 control rerun: all Code16 positives + scaled correct-R1 task vector, compact ability span, seq1536/resp512 | two-card OOM during memory trajectory-turn forward; no checkpoint baked |
| A155 | eval | `eval_r12b_code_20260520` | code | running | R12B passed quick gate; run CURE Code | LiveBench Acc 0.3574 / BoN 0.4688; LiveCodeBench running |
| A156 | eval | `eval_r13a_tm_20260520` | hiddenstate/tool-memory | running | R13A eval-leak RF/Mem-only diagnostic passed train/bake; quick-gate before Code | launched 2026-05-20 12:09 CST on GPU2 |
| A157 | eval | `eval_r12e_tm_20260520` | tool/memory | running | Quick-gate R12E RF-only tag-quota response256 + Memory x1.8 selected checkpoint | launched 2026-05-20 12:16 CST on GPU3 |
| A158 | train | `trc_r13b_evalleak_all_r1_finalmem_e8_20260520` | hiddenstate/code-diagnostic | done/baked | Low-memory R1 control: same all+R1 Code16 bank, but memory uses final response instead of trajectory-turn loss | auto-selected e4; e8 reaches C/M/R/T `1.1256/1.0271/1.1502/1.1370` and is force-baked as A160 |
| A159 | eval | `eval_r13a_e08_forced_tm_20260520` | hiddenstate/tool-memory | running | Forced epoch-8 R13A checkpoint to test whether loss-plateau selection under-pushed Code/Tool gates | checkpoint `/tmp/shared-storage/OnPolicy/checkpoints/trc_r13a_evalleak_rfmem_compact600_e08_forced_20260520-selected`; Tool/Memory quick gate launched on GPUs 1/2 |
| A160 | eval | `eval_r13b_e08_forced_tm_20260520` | hiddenstate/tool-memory | pending | Forced epoch-8 R13B low-memory R1 checkpoint | checkpoint `/tmp/shared-storage/OnPolicy/checkpoints/trc_r13b_evalleak_all_r1_finalmem_e08_forced_20260520-selected`; wait for BFCL slot after A159 |
| A161 | data | `trc_round14_code_contrast_v1` | hiddenstate/code-diagnostic | done | Build Code contrastive bank by adding same-prompt failed CURE responses from R11B eval outputs to R13A positive rows | 74 rows: Tool32/Memory32/Code10; Code 10/10 have `negative_response`; output `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_code_contrast_v1/trc_expert_trajectories.jsonl` |
| A162 | train | `trc_r14a_evalleak_contrast_rfmem_e8_20260520` | hiddenstate/code-diagnostic | running | R13A setting plus Code negative contrastive hinge loss; diagnostic eval-leak only | `CONTRASTIVE_NEGATIVE_LOSS_WEIGHT=0.5`, margin `0.05`, task `code`; launched on GPUs 3,7 |
| A163 | data | `trc_round14_mixed_train_contrast_v1` | calibration/code | done | Build mixed Code BoN-to-Acc bank: stable Tool32/Memory32 + Code train24 + formal contrast8 | output `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_mixed_train_contrast_v1/trc96_expert_trajectories.jsonl`; Code 32 rows, negative_response rows 8 |
| A164 | train | `trc_r14b_mixed_train24_contrast8_e8_20260520` | hiddenstate/code | done/baked | R5A-like training with Code contrast hard anchors; short 8-epoch diagnostic | selected e8: C/M/T `1.1598/0.9974/1.1506`; quick Tool/Memory launched as A168 |
| A165 | train | `trc_r15a_r5a_repro_e12_20260520` | reproducibility | running | Strict R5A reproduction to test whether high-BoN anchor is stable | epoch6 observed: C/M/T `1.1199/0.9990/1.1140`, matching original R5A trajectory |
| A166 | data/train | `trc_r15b_r5a_train24_contrast8_e12_20260520` | code/contrast | done/baked | R5A Tool/Memory with Code train24 + formal contrast8 and contrast weight 1.5 | selected e12: C/M/T `1.2383/0.9983/1.2152`; contrast active rate near end `0.0625`, which suggests current contrast signal becomes weak after gates rise |
| A167 | eval | `eval_r5a_r11b_code_repeat_20260520` | code/stability | running | Repeat CURE Code for R5A and R11B because both have high BoN but modest Acc | R5A process `376/512`, R11B process `323/512` observed at 2026-05-20 14:02 CST |
| A168 | eval | `eval_r14b_tool_memory_20260520` | quick-gate | done/reject | Run Tool on GPU0 and Memory on GPU1 for R14B before deciding Code eval | Tool passed: 0.7944 (`0.8125/0.6250/0.8800/0.8600`); Memory mean F1 `0.7578`, strict reject |
| A169 | eval | `eval_r14b_code_20260520` | code | running | R14B Tool passed and GPU0 became free, so launch CURE Code while Memory continues | run id `code_20260520_1408_r14b`; if Memory later fails, keep result as diagnostic |
| A170 | eval | `toolrl_r14b_r15a_r15b_all80_20260520` | tool/source | done | Add ToolRL rlla_4k/test all80 sanity test for R14B/R15A/R15B | R14B/R15A/R15B all success `0.6375`; mean reward `0.8363/0.8363/0.8352`; source ToolRL behavior is preserved |
| A171 | eval | `eval_r15b_tool_memory_20260520` | quick-gate | running | Quick BFCL Tool + HotpotQA Memory for R15B before any expensive Code eval | BFCL Tool mean `0.7788` (`0.7500/0.6250/0.8850/0.8550`), below strict gate; Memory still running |
| A172 | data | `trc_round16_nonleak_code_contrast_v1` | calibration/code | done | Build paper-main-safe CodeP0 pass/fail contrast bank without formal eval anchors | Tool32/Memory32 from R5A stable bank; Code22 same-prompt pass/fail + Code10 positive fill, all Code rows from `CodeContests_train`; output `/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl` |
| A173 | train | `trc_r16a_nonleak_contrast22_w15_e12_20260520` | code/contrast | done/baked | R5A/R15B settings with non-leak CodeP0 contrast22, contrast weight 1.5 | selected e12: C/M/T `1.2399/0.9945/1.1887`; Code task contrast stayed active around `0.50`; quick Tool/Memory launched as A175 |
| A174 | train | `trc_r16b_nonleak_contrast22_w30_e12_20260520` | code/contrast | done/baked | Same as R16A but contrast weight 3.0 to test signal strength | selected e12: C/M/T `1.2403/0.9945/1.1809`; wait for BFCL slot after R16A quick eval |
| A175 | eval | `eval_r16a_tm_20260520` | quick-gate | done/reject | Run Tool/Memory quick eval for R16A before any expensive Code eval | Tool mean `0.7944` passed; Memory mean F1 `0.7506` below `0.76`, so Code eval skipped |
| A176 | train | `trc_r16c_rfonly_contrast16_w15_e12_20260520` | code/contrast | running | RF-only Code rows: 16 pass/fail contrast + 16 positive fill; tests whether DeepSeek fallback hidden trajectories hurt ReasonFlux code vector | launched on GPUs 4,5; all Code rows from CodeP0-v3 train and source `ReasonFlux` |
| A177 | train | `trc_r16d_nonleak_contrast22_response256_w15_e12_20260520` | code/span | running | Same data/weight as R16A but Code response span topK256 instead of code-block384 | launched on GPUs 6,7; tests whether reasoning span helps Code Acc |
| A178 | code | `prompt_residual_loss` | trc/loss | done | Add optional prompt-tail expert residual alignment to TRC trainer | default off; enables R17B while preserving all existing R16 paths |
| A179 | train | `trc_r17a_no_prompt_drift_contrast22_w15_e12_20260520` | prompt/span | done/baked | R16A setting with `BETA_BASE=0`, no prompt residual | selected e12: C/M/T `1.2430/0.7571/1.2321`; quick eval launched because selection is by metrics, not gate |
| A180 | train | `trc_r17b_prompt_residual_contrast22_w15_e12_20260520` | prompt/span | done/baked | R17A plus prompt-tail expert residual loss | selected e12: C/M/T `1.2429/0.7573/1.2319`; prompt residual lowers loss but shows same Memory-telemetry risk as R17A |
| A181 | eval | `eval_r16b_code_20260520` | code | running | R16B passed quick gate, so launch CURE Code | Tool mean `0.7944`, Memory mean F1 `0.7724`; running on GPUs 6,7 |
| A182 | eval | `eval_r16c_tm_20260520` | quick-gate | done/pass | R16C completed and was manually selected/baked from epoch 12 | Tool mean `0.7931`; Memory mean F1 `0.7632` (`0.7633/0.7608/0.7936/0.7353`); promote to Code |
| A183 | eval | `eval_r17a_tm_20260520` | quick-gate | done/pass | R17A no-prompt-drift checkpoint quick eval | Tool mean `0.7969`; Memory mean F1 `0.7645` (`0.7494/0.7549/0.7963/0.7574`); promoted to Code candidate |
| A184 | eval | `eval_r16d_tm_20260520` | quick-gate | done/pass | R16D response-span Code control after manual bake | Tool mean `0.7944` (`0.8125/0.6250/0.8850/0.8550`); Memory mean F1 `0.7604` (`0.7555/0.7263/0.7978/0.7619`); promoted to Code candidate |
| A185 | eval | `eval_r16c_code_20260520` | code | running | R16C passed quick gate; run CURE Code to test RF-only Code trajectories | LiveBench Acc `0.3477` / BoN `0.4219`; LiveCodeBench running |
| A186 | eval | `eval_r17b_tm_20260520` | quick-gate | done/pass | R17B prompt-residual checkpoint quick eval | Tool mean `0.7969` (`0.8125/0.6250/0.8950/0.8550`); Memory mean F1 `0.7643` (`0.7477/0.7770/0.7933/0.7391`); promoted to Code candidate |

## Live Monitor Snapshot 2026-05-20 11:45 CST

- tmux sessions present: `eval_r11b_code_20260520`, `eval_r11f_code_20260520`,
  `eval_r11g_code_20260520`, `eval_r11h_code_20260520`; `eval_r10d_code_20260520`
  and `eval_r12d_tm_20260520` have exited.
- Completed since prior snapshot: R10D Code final metrics are recorded in
  Round10 and A133; R11H Tool/Memory quick gate passed and Code launched.
- Pending Code progress: R11B LiveBench `0.3750/0.5000`, LiveCodeBench
  `321/512`; R11F LiveBench `0.3555/0.4688`, LiveCodeBench `230/512`; R11G
  LiveBench `0.3516/0.4297`, LiveCodeBench ground-truth tests `24/512`; R11H LiveBench
  `0.3477/0.4375`, LiveCodeBench generation about `2153/4088` prompts.
- R12D quick gate complete and exited: Tool mean `0.7788`
  (`0.7500/0.6250/0.8850/0.8550`) and Memory mean F1 `0.7590`
  (`0.7779/0.7329/0.7465/0.7787`); reject before Code.

## 记录规则

- 所有训练 run 的 `run.env` 是 config truth；文档只做索引和解释。
- 无效 baked checkpoint 删除；run metrics、selected gates、eval logs 保留。
- 每完成一个 attempt，必须写 `status / signal / metrics / decision / next` 五项。

## Open TODO

| priority | item | why | next action |
|---|---|---|---|
| P0.5 | Tool non-live robustness | Current Tool quick gate can pass by mean while BFCL non-live parallel / parallel_multiple still leave a gap to stronger historical runs; this is not a gate-only issue. | Build a Tool-specific audit that separates `parallel` / `parallel_multiple` from live, counts `cannot_find_match` / `wrong_count` / `decoder_failed`, then add non-leak BFCL-style non-live calibration anchors with real model-success trajectories where possible. |
| P0.5 | Tool output-format preservation | Tool failures often come from parser-level mismatch rather than semantic tool absence; hidden-span training must protect the exact call-format behavior. | Compare ToolRL all80, BFCL non-live, and BFCL live-style cases; consider tool-call span contrast / format retention before changing task weights. |
