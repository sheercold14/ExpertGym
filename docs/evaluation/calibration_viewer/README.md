# Calibration Viewer

Default embedded dataset:

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl
```

Related files:

```text
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/summary.json
/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl
```

Open the viewer:

```text
docs/evaluation/calibration_viewer/index.html
```

The page embeds a compact copy of the 96-prompt manifest, so it can be opened
directly from the filesystem. Use `Load JSONL` to inspect another calibration
manifest from disk.

The current embedded data also includes real code rollout samples from:

```text
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_001/rollouts.jsonl
/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_006/rollouts.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260517.jsonl
/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260517.jsonl
```

For code prompts, open the `Rollouts` tab to inspect model output text,
`reward`, `reward_train`, `success`, and CURE source-test details.

The page also has a `LiveCodeBench` tab. That tab is not calibration training
data; it is a compact audit view built from real CURE eval artifacts:

```text
/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/app_data.json
/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/temp_data/outputs-eval-.tmp.shared-storage.OnPolicy.runs.gated_grpo.expB_gp_code_opd_aug_20260516.iter_018.baked_policy-LiveCodeBench.json
/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.runs.gated_grpo.expB_gp_code_opd_aug_20260516.iter_018.baked_policy-LiveCodeBench.txt
```

It shows the CURE flow, global LiveCodeBench metrics, and two traceable examples
of code candidates, generated tests, bool tables, BoN selection, and hidden-test
scoring.

Regenerate the embedded data:

```bash
/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python \
  scripts/analysis/build_calibration_viewer_data.py \
  --input /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/eval_targeted96.prompts.jsonl \
  --summary /tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/summary.json \
  --name eval_targeted96_cure_aligned_20260517 \
  --rollout policy_init1_iter001=/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_001/rollouts.jsonl \
  --rollout policy_init1_iter006=/tmp/shared-storage/OnPolicy/runs/gated_grpo/eg72_main_gc_init1_evaltarget_fast_20260518/iter_006/rollouts.jsonl \
  --rollout expert_deepseek_s8=/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_deepseek_r1_distill_qwen7b_evaltarget_s8_seed20260517.jsonl \
  --rollout expert_reasonflux_s8=/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_cure_aligned_20260517/expert_rollouts/code_expert_reasonflux_coder7b_evaltarget_s8_seed20260517.jsonl \
  --rollout-task code \
  --max-samples-per-source 4 \
  --livecode-app-data /tmp/shared-storage/OnPolicy/analysis/eval_case_browser/app_data.json \
  --livecode-model-id expertgym-B-codeaug-opd-i18 \
  --livecode-result /mnt/cache/wuruixiao/users/lsc/CURE/evaluation/results/results-eval-.tmp.shared-storage.OnPolicy.runs.gated_grpo.expB_gp_code_opd_aug_20260516.iter_018.baked_policy-LiveCodeBench.txt \
  --livecode-max-examples 2 \
  --output docs/evaluation/calibration_viewer/data/eval_targeted96_cure_aligned_20260517.js
```
