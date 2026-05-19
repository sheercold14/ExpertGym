# 2026-05-18 Local WUDI / ExpertMerging Model Dirs

Purpose: record local WUDI / ExpertMerging model storage paths without relaunching
these baselines. These paths are indices for later audit/comparison only; they
are not part of the current baseline rerun batch.

## WUDI Main

Primary WUDI checkpoint found locally:

```text
/tmp/shared-storage/AgentMerging_plan/experiments/wudi_merging/wudi_qwen7b_3expert_diag/model
```

Related eval artifacts:

```text
/tmp/shared-storage/AgentMerging_plan/evaluation_workdirs/eval6-20260502-125748-wudi-qwen7b-3expert
/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/eval6-memory-hotpotqa/wudi-qwen7b-3expert
```

## TC-WUDI / VGEC Variants

Root:

```text
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/
```

Representative model directories:

```text
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-iter001/tc-diagwudi-traj-alpha050-codeprotect/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-iter002/tc-diagwudi-traj-alpha025-codeprotect/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-iter003/tc-diagwudi-traj-alpha050-codeprotect-code115/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-ae-stage1-20260506/tc-wudi-a-normmatch-traj/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-ae-stage1-20260506/tc-wudi-b-autolambda-traj/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-ae-stage1-20260506/tc-wudi-d-common-residual-traj/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/tc-wudi-loop-ae-stage1-20260506/tc-wudi-e-rowcol-traj/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/vgec-sign-protect-finemid-20260508/vgec-tool-no-lower-codesignsoft-midedge-traj/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/vgec-tool-layerwise-complement-20260508/vgec-bfclheldout0503-toolbeta-up-down-no-lower-traj/model
/tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi/vgec-tool-proj-pair-20260508/vgec-bfclheldout0503-toolbeta-up-down-traj/model
```

Full local listing command:

```bash
find /tmp/shared-storage/AgentMerging_plan/experiments/tc_wudi \
  -maxdepth 5 -type d -name model | sort
```

## ExpertMerging

Current shared-storage root:

```text
/tmp/shared-storage/expert_merging/
```

Representative model directories:

```text
/tmp/shared-storage/expert_merging/0425-232232/model
/tmp/shared-storage/expert_merging/0428-151150/model
/tmp/shared-storage/expert_merging/0428-203833/model
/tmp/shared-storage/expert_merging/0503-152926/model
/tmp/shared-storage/expert_merging/0503-193353/model
/tmp/shared-storage/expert_merging/0505-104853/model
/tmp/shared-storage/expert_merging/0505-171400/model
/tmp/shared-storage/expert_merging/0506-185713/model
```

Related eval artifacts:

```text
/tmp/shared-storage/AgentMerging_plan/evaluation_workdirs/eval6-20260502-125748-expert-merging-0425-232232-retest
/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/memory/eval6-memory-hotpotqa/expert-merging-0425-232232
```

Full local listing command:

```bash
find /tmp/shared-storage/expert_merging -maxdepth 2 -type d -name model | sort
```

## Older RAIN / ExpertMerging Style Storage

```text
/tmp/shared-storage/RAIN/ExpertMerging/
```

## Current Policy

- Do not rerun WUDI / ExpertMerging in the current baseline batch.
- Use the paths above only for later audit, comparison, or re-evaluation under a
  separate, explicitly documented protocol.
- Prioritize reproducible baselines in `scripts/baselines/`: TA, TIES, DARE,
  AdaMerging, Mixture/full-GRPO, then the pending Fisher gap.
