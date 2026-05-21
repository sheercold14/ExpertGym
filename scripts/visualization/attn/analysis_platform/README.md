# RCRF Mechanism Discovery Workbench

Local Streamlit workbench for RCRF mechanism analysis. It reads RCRF signed utility, gate, probe, and eval artifacts, normalizes them into long-form JSONL schemas, and shows module-level attention/MLP mechanism views.

Run from this directory:

```bash
cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym/scripts/visualization/attn
pip install -r analysis_platform/requirements.txt
streamlit run analysis_platform/app.py
```

Or run from the repository root:

```bash
streamlit run scripts/visualization/attn/analysis_platform/app.py
```

Default real inputs:

- `/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521`
- `/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521`
- `/tmp/shared-storage/ExpertGym/rcrf/eval/rcrf_v1_20260521`
- `/tmp/shared-storage/ExpertGym/rcrf/eval/toolrl_rlla4k_20260521`
- `/tmp/shared-storage/OnPolicy/eval/cure_feedback/rcrf-v1-rcrf-code/rcrf-v1-rcrf/rcrf_v1_rcrf_code_quick_20260521`

If these paths are missing, the app starts with deterministic 96-prompt example data instead of crashing.

Long-form schemas:

- `residual_records.jsonl`: `run_id, expert, task, prompt_id, span_type, layer, module, module_family, signed_effect, expression, positive_fraction`
- `interference_records.jsonl`: `expert_a, expert_b, task, layer, module, module_family, cosine, conflict_score, cross_harm`
- `gate_records.jsonl`: `candidate, expert, layer, module, module_family, alpha, owner_signal, synergy_signal, harm_signal, noise_score`
- `eval_records.jsonl`: `candidate, task, subset, metric, score, n_examples`

Data collection:

```bash
python scripts/analysis/rcrf_collect_mechanism_data.py \
  --output-dir scripts/visualization/attn/analysis_platform/data
```

Paper figure export:

```bash
python scripts/analysis/rcrf_export_figures.py \
  --data-dir scripts/visualization/attn/analysis_platform/data \
  --output-dir /tmp/rcrf_figures
```

PNG/PDF export requires Plotly Kaleido:

```bash
pip install kaleido
```
