#!/usr/bin/env python3
"""Build a standalone RCF-BC/RCRF diagnostic dashboard.

The dashboard is intentionally static and dependency-free.  It reads the
operating-point comparison artifacts produced by
``compare_rcrf_operating_points.py`` and writes a self-contained HTML report
with layer/module/expert heatmaps, role aggregates, and a row explorer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_COMPARISON_DIR = ROOT / "analysis" / "rcrf_operating_point_compare_20260522"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_diagnostic_dashboard_20260522"
MODULE_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
EXPERT_ORDER = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = comparison_dir / "operating_point_comparison_summary.json"
    rows_path = comparison_dir / "operating_point_rows.jsonl"
    summary = load_json(summary_path)
    rows = load_jsonl(rows_path)
    candidates = sorted(summary.get("candidate_overview", {}))
    if not candidates:
        raise ValueError(f"No candidate_overview found in {summary_path}")

    dashboard = build_dashboard_data(summary, rows, candidates)
    write_json(output_dir / "dashboard_data.json", dashboard)
    write_csv(output_dir / "role_candidate_summary.csv", dashboard["role_candidate_summary"])
    write_csv(output_dir / "layer_module_expert_summary.csv", dashboard["layer_module_expert_summary"])
    (output_dir / "index.html").write_text(render_html(dashboard), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(dashboard, output_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index": str(output_dir / "index.html"),
                "rows": len(rows),
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_dashboard_data(summary: dict[str, Any], rows: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    reference = str(summary.get("reference") or "")
    compares = [str(item) for item in summary.get("compares") or []]
    overview_rows = candidate_overview_rows(summary, candidates)
    role_candidate_summary = aggregate_rows(rows, candidates, ["role"])
    layer_module_expert_summary = aggregate_rows(rows, candidates, ["layer", "module", "expert"])
    layer_band_module_expert_summary = aggregate_rows(rows, candidates, ["layer_band", "module_family", "expert"])
    heatmap = build_heatmap(rows, candidates)
    compact_rows = [compact_row(row, candidates, reference) for row in rows]
    role_counts = counter_rows(rows, "role")
    expert_counts = counter_rows(rows, "expert")
    module_counts = counter_rows(rows, "module")
    reference_gap_summary = summary.get("reference_gap_summary", {})
    return {
        "format": "rcrf_diagnostic_dashboard_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference": reference,
        "compares": compares,
        "candidates": candidates,
        "source_summary": {
            "atlas_rows": summary.get("atlas_rows"),
            "base_gates": summary.get("base_gates"),
            "candidate_paths": summary.get("candidate_paths", {}),
        },
        "overview_rows": overview_rows,
        "role_counts": role_counts,
        "expert_counts": expert_counts,
        "module_counts": module_counts,
        "reference_gap_summary": reference_gap_summary,
        "role_candidate_summary": role_candidate_summary,
        "layer_module_expert_summary": layer_module_expert_summary,
        "layer_band_module_expert_summary": layer_band_module_expert_summary,
        "heatmap": heatmap,
        "rows": compact_rows,
        "method_notes": [
            "RCF-BC keeps continuous Code pass/fail evidence and constrains it with Tool/Memory behavior utility/harm.",
            "The atlas is used for audit and ablation, not as a hard replacement for the continuous capability field.",
            "Global task-scalar shrinkage is a negative control: it can release Memory but destroys Code.",
        ],
    }


def candidate_overview_rows(summary: dict[str, Any], candidates: list[str]) -> list[dict[str, Any]]:
    overview = summary.get("candidate_overview", {})
    output = []
    for candidate in candidates:
        item = overview.get(candidate, {})
        row = {
            "candidate": candidate,
            "changed_count": safe_int(item.get("changed_count")),
            "positive_count": safe_int(item.get("positive_count")),
            "negative_count": safe_int(item.get("negative_count")),
            "mean_delta": safe_float(item.get("mean_delta")),
            "mean_abs_delta": safe_float(item.get("mean_abs_delta")),
        }
        for expert in EXPERT_ORDER:
            expert_stats = (item.get("delta_by_expert") or {}).get(expert, {})
            row[f"{expert}_changed"] = safe_int(expert_stats.get("changed"))
            row[f"{expert}_mean_abs_delta"] = safe_float(expert_stats.get("mean_abs"))
        output.append(row)
    return output


def aggregate_rows(rows: list[dict[str, Any]], candidates: list[str], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item = {key: value for key, value in zip(group_keys, group)}
        item["row_count"] = len(group_rows)
        for candidate in candidates:
            values = [safe_float(row.get(f"{candidate}_delta")) for row in group_rows]
            changed = [value for value in values if abs(value) > 1e-12]
            item[f"{candidate}_changed"] = len(changed)
            item[f"{candidate}_positive"] = sum(1 for value in changed if value > 0.0)
            item[f"{candidate}_negative"] = sum(1 for value in changed if value < 0.0)
            item[f"{candidate}_mean_delta"] = mean(values) if values else 0.0
            item[f"{candidate}_mean_abs_delta"] = mean(abs(value) for value in values) if values else 0.0
        output.append(item)
    return output


def build_heatmap(rows: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        layer = safe_int(row.get("layer"))
        module = str(row.get("module") or "")
        expert = str(row.get("expert") or "")
        if layer < 0 or module not in MODULE_ORDER or expert not in EXPERT_ORDER:
            continue
        for candidate in candidates:
            grouped[(candidate, expert, layer, module)].append(safe_float(row.get(f"{candidate}_delta")))

    cells: list[dict[str, Any]] = []
    max_abs = 0.0
    for (candidate, expert, layer, module), values in sorted(grouped.items()):
        value = mean(values) if values else 0.0
        max_abs = max(max_abs, abs(value))
        cells.append(
            {
                "candidate": candidate,
                "expert": expert,
                "layer": layer,
                "module": module,
                "mean_delta": value,
                "mean_abs_delta": mean(abs(v) for v in values) if values else 0.0,
                "changed": sum(1 for value in values if abs(value) > 1e-12),
                "count": len(values),
            }
        )
    return {"module_order": list(MODULE_ORDER), "expert_order": list(EXPERT_ORDER), "max_abs_delta": max_abs, "cells": cells}


def compact_row(row: dict[str, Any], candidates: list[str], reference: str) -> dict[str, Any]:
    output = {
        "key": row.get("key", ""),
        "param_name": row.get("param_name", ""),
        "expert": row.get("expert", ""),
        "layer": safe_int(row.get("layer")),
        "layer_band": row.get("layer_band", ""),
        "module": row.get("module", ""),
        "module_family": row.get("module_family", ""),
        "role": row.get("role", ""),
        "code_positive_sources": row.get("code_positive_sources", ""),
        "code_negative_sources": row.get("code_negative_sources", ""),
        "code_positive_strength": safe_float(row.get("code_positive_strength")),
        "code_negative_strength": safe_float(row.get("code_negative_strength")),
        "protected_support_tasks": row.get("protected_support_tasks", ""),
        "protected_harm_tasks": row.get("protected_harm_tasks", ""),
        "protected_max_harm_norm": safe_float(row.get("protected_max_harm_norm")),
        "base_coefficient": safe_float(row.get("base_coefficient")),
    }
    for candidate in candidates:
        output[f"{candidate}_delta"] = safe_float(row.get(f"{candidate}_delta"))
        output[f"{candidate}_coefficient"] = safe_float(row.get(f"{candidate}_coefficient"))
        output[f"{candidate}_sign"] = row.get(f"{candidate}_sign", "")
        if reference and candidate != reference:
            output[f"{candidate}_gap_from_{reference}"] = safe_float(row.get(f"{reference}_delta")) - safe_float(row.get(f"{candidate}_delta"))
    return output


def counter_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(key) or "") for row in rows)
    return [{"name": name, "count": count} for name, count in counter.most_common()]


def render_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RCF-BC Diagnostic Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #65717f;
      --line: #d8dce2;
      --blue: #2364aa;
      --red: #b54747;
      --green: #26734d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 22px 28px 18px;
      background: #27313f;
      color: white;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #d8dee9; max-width: 1100px; }}
    main {{ padding: 20px 28px 40px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfbfa; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; color: var(--muted); background: #f1f3f5; position: sticky; top: 0; }}
    .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .scroll {{ overflow: auto; max-height: 520px; border: 1px solid var(--line); border-radius: 6px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }}
    select, input {{
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 8px;
      background: white;
      color: var(--ink);
    }}
    .heatmap-wrap {{ overflow-x: auto; }}
    .heatmap {{ border-collapse: separate; border-spacing: 2px; width: auto; }}
    .heatmap th, .heatmap td {{ border: 0; min-width: 54px; height: 24px; padding: 3px 5px; text-align: center; }}
    .heatmap th {{ position: static; background: transparent; color: var(--muted); }}
    .heatmap td {{ border-radius: 4px; font-size: 11px; font-variant-numeric: tabular-nums; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      background: #eef1f4;
      color: #394554;
      font-size: 12px;
      margin: 1px 2px 1px 0;
      white-space: nowrap;
    }}
    .path {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; word-break: break-all; }}
    .pos {{ color: var(--green); }}
    .neg {{ color: var(--red); }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>RCF-BC Diagnostic Dashboard</h1>
    <p>Residual Capability Field with Behavior Constraints: inspect capability deltas, behavior constraints, and lost operating-point decisions at the residual row level.</p>
  </header>
  <main>
    <section>
      <h2>Summary</h2>
      <div id="cards" class="cards"></div>
      <p id="source" class="path"></p>
    </section>
    <section>
      <h2>Candidate Overview</h2>
      <div class="scroll" id="overview"></div>
    </section>
    <section>
      <h2>Layer / Module Delta Heatmap</h2>
      <div class="controls">
        <label>candidate <select id="candidateSelect"></select></label>
        <label>expert <select id="expertSelect"></select></label>
      </div>
      <div class="heatmap-wrap" id="heatmap"></div>
    </section>
    <section>
      <h2>Role Aggregates</h2>
      <div class="scroll" id="roles"></div>
    </section>
    <section>
      <h2>Residual Row Explorer</h2>
      <div class="controls">
        <label>candidate <select id="rowCandidateSelect"></select></label>
        <label>role <select id="roleSelect"></select></label>
        <label>expert <select id="rowExpertSelect"></select></label>
        <label>search <input id="searchInput" placeholder="module, role, source, key"></label>
        <label>min |delta| <input id="minDeltaInput" value="0" size="5"></label>
      </div>
      <div class="scroll" id="rows"></div>
    </section>
  </main>
  <script id="dashboard-data" type="application/json">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    const fmt = (x, d=4) => Number(x || 0).toFixed(d);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    function init() {{
      renderCards();
      renderOverview();
      fillSelect('candidateSelect', DATA.candidates, DATA.candidates.includes('v18_rcf_bc') ? 'v18_rcf_bc' : DATA.candidates[0]);
      fillSelect('rowCandidateSelect', DATA.candidates, DATA.candidates.includes('v18_rcf_bc') ? 'v18_rcf_bc' : DATA.candidates[0]);
      fillSelect('expertSelect', DATA.heatmap.expert_order, 'code');
      fillSelect('rowExpertSelect', ['all', ...DATA.heatmap.expert_order], 'all');
      fillSelect('roleSelect', ['all', ...DATA.role_counts.map(x => x.name)], 'all');
      for (const id of ['candidateSelect', 'expertSelect']) document.getElementById(id).addEventListener('change', renderHeatmap);
      for (const id of ['rowCandidateSelect', 'rowExpertSelect', 'roleSelect', 'searchInput', 'minDeltaInput']) document.getElementById(id).addEventListener('input', renderRows);
      renderHeatmap();
      renderRoles();
      renderRows();
    }}
    function fillSelect(id, values, selected) {{
      const el = document.getElementById(id);
      el.innerHTML = values.map(v => `<option value="${{esc(v)}}" ${{v === selected ? 'selected' : ''}}>${{esc(v)}}</option>`).join('');
    }}
    function renderCards() {{
      const ref = DATA.reference;
      const refRow = DATA.overview_rows.find(r => r.candidate === ref) || {{}};
      const v18 = DATA.overview_rows.find(r => r.candidate === 'v18_rcf_bc') || refRow;
      const cards = [
        ['reference', ref],
        ['candidates', DATA.candidates.length],
        ['residual rows', DATA.rows.length],
        ['v18 changed', v18.changed_count],
        ['v18 +/-', `${{v18.positive_count}} / ${{v18.negative_count}}`],
        ['v18 mean |delta|', fmt(v18.mean_abs_delta, 6)],
      ];
      document.getElementById('cards').innerHTML = cards.map(([k,v]) => `<div class="card"><div class="label">${{esc(k)}}</div><div class="value">${{esc(v)}}</div></div>`).join('');
      document.getElementById('source').textContent = `atlas=${{DATA.source_summary.atlas_rows}} | base=${{DATA.source_summary.base_gates}}`;
    }}
    function renderOverview() {{
      const cols = ['candidate','changed_count','positive_count','negative_count','mean_abs_delta','tool_changed','memory_changed','code_changed','tool_mean_abs_delta','memory_mean_abs_delta','code_mean_abs_delta'];
      document.getElementById('overview').innerHTML = table(DATA.overview_rows, cols);
    }}
    function renderHeatmap() {{
      const cand = document.getElementById('candidateSelect').value;
      const expert = document.getElementById('expertSelect').value;
      const mods = DATA.heatmap.module_order;
      const cells = new Map();
      for (const c of DATA.heatmap.cells) if (c.candidate === cand && c.expert === expert) cells.set(`${{c.layer}}|${{c.module}}`, c);
      let html = '<table class="heatmap"><thead><tr><th>layer</th>' + mods.map(m => `<th>${{esc(m)}}</th>`).join('') + '</tr></thead><tbody>';
      for (let layer = 0; layer < 28; layer++) {{
        html += `<tr><th>${{layer}}</th>`;
        for (const mod of mods) {{
          const c = cells.get(`${{layer}}|${{mod}}`) || {{mean_delta:0, changed:0, count:0}};
          html += `<td style="${{cellStyle(c.mean_delta)}} " title="${{esc(cand)}} ${{esc(expert)}} L${{layer}} ${{esc(mod)}} mean_delta=${{fmt(c.mean_delta,6)}} changed=${{c.changed}}/${{c.count}}">${{fmt(c.mean_delta,3)}}</td>`;
        }}
        html += '</tr>';
      }}
      html += '</tbody></table>';
      document.getElementById('heatmap').innerHTML = html;
    }}
    function cellStyle(value) {{
      const maxAbs = Math.max(DATA.heatmap.max_abs_delta || 0.001, 0.001);
      const t = Math.min(Math.abs(value) / maxAbs, 1);
      if (Math.abs(value) < 1e-12) return 'background:#f0f2f4;color:#65717f';
      if (value > 0) return `background:rgba(38,115,77,${{0.18 + 0.72*t}});color:#10261b`;
      return `background:rgba(181,71,71,${{0.18 + 0.72*t}});color:#2d1010`;
    }}
    function renderRoles() {{
      const cand = DATA.candidates.includes('v18_rcf_bc') ? 'v18_rcf_bc' : DATA.reference;
      const rows = DATA.role_candidate_summary.map(r => ({{
        role: r.role,
        row_count: r.row_count,
        changed: r[`${{cand}}_changed`],
        positive: r[`${{cand}}_positive`],
        negative: r[`${{cand}}_negative`],
        mean_delta: r[`${{cand}}_mean_delta`],
        mean_abs_delta: r[`${{cand}}_mean_abs_delta`],
      }})).sort((a,b) => b.changed - a.changed || b.row_count - a.row_count);
      document.getElementById('roles').innerHTML = table(rows, ['role','row_count','changed','positive','negative','mean_delta','mean_abs_delta']);
    }}
    function renderRows() {{
      const cand = document.getElementById('rowCandidateSelect').value;
      const expert = document.getElementById('rowExpertSelect').value;
      const role = document.getElementById('roleSelect').value;
      const query = document.getElementById('searchInput').value.toLowerCase();
      const minDelta = Number(document.getElementById('minDeltaInput').value || 0);
      let rows = DATA.rows.filter(r => {{
        if (expert !== 'all' && r.expert !== expert) return false;
        if (role !== 'all' && r.role !== role) return false;
        const d = Math.abs(Number(r[`${{cand}}_delta`] || 0));
        if (d < minDelta) return false;
        if (query) {{
          const hay = [r.key, r.role, r.module, r.module_family, r.code_positive_sources, r.code_negative_sources, r.protected_support_tasks, r.protected_harm_tasks].join(' ').toLowerCase();
          if (!hay.includes(query)) return false;
        }}
        return true;
      }});
      rows = rows.sort((a,b) => Math.abs(Number(b[`${{cand}}_delta`] || 0)) - Math.abs(Number(a[`${{cand}}_delta`] || 0))).slice(0, 220);
      const shown = rows.map(r => ({{
        layer: r.layer,
        module: r.module,
        expert: r.expert,
        role: r.role,
        delta: r[`${{cand}}_delta`],
        coeff: r[`${{cand}}_coefficient`],
        code_pos: r.code_positive_sources,
        code_neg: r.code_negative_sources,
        support: r.protected_support_tasks,
        harm: r.protected_harm_tasks,
        key: r.key,
      }}));
      document.getElementById('rows').innerHTML = table(shown, ['layer','module','expert','role','delta','coeff','code_pos','code_neg','support','harm','key']);
    }}
    function table(rows, cols) {{
      let html = '<table><thead><tr>' + cols.map(c => `<th>${{esc(c)}}</th>`).join('') + '</tr></thead><tbody>';
      for (const row of rows) {{
        html += '<tr>' + cols.map(c => {{
          let v = row[c];
          let cls = typeof v === 'number' ? 'num' : '';
          if (c.includes('delta') && Number(v) > 0) cls += ' pos';
          if (c.includes('delta') && Number(v) < 0) cls += ' neg';
          if (typeof v === 'number') v = Math.abs(v) < 100 ? fmt(v, 6) : String(v);
          return `<td class="${{cls}}">${{esc(v)}}</td>`;
        }}).join('') + '</tr>';
      }}
      return html + '</tbody></table>';
    }}
    init();
  </script>
</body>
</html>
"""


def render_readme(data: dict[str, Any], output_dir: Path) -> str:
    candidates = ", ".join(data["candidates"])
    return f"""# RCF-BC Diagnostic Dashboard

Open:

```text
{output_dir / "index.html"}
```

Data:

```text
{output_dir / "dashboard_data.json"}
```

Candidates:

```text
{candidates}
```

Use this dashboard to inspect layer/module/expert deltas, atlas roles, and row-level residual evidence for RCF-BC.
"""


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
